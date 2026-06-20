# Local AI Personal Assistant — Build Specification

> A buildable spec for an incrementally-developed, local-first AI personal assistant.
> Hand this to Claude Code one phase at a time. Each phase is a self-contained,
> reviewable unit with explicit acceptance criteria.

---

## 1. Overview

Build a personal AI assistant that runs entirely on local hardware, reasons over
tools, searches the live web, and remembers the user across sessions. It is built
in four incremental phases, each one a reviewable pull request that adds one
capability without rewriting what came before.

**Design principles**

- **Local-first.** All inference, embeddings, and storage run on the user's machine. No cloud LLM calls. (The web-search tool is the only outbound network dependency, and it is swappable for a self-hosted option.)
- **Model-agnostic.** The model is reached through an OpenAI-compatible API and selected by a single config value. Swapping models is a one-line change.
- **Incremental.** Each phase adds exactly one capability. The core loop from Phase 1 is never replaced — later phases only wrap it or feed data into it.
- **Reviewable.** Every phase has a concrete checkpoint that proves it works before the next phase begins.

**End state:** a CLI assistant that streams responses, calls tools, searches and cites the web, and recalls durable facts about the user after a full restart.

---

## 2. Architecture

The model is **stateless**. It appears to "remember" a conversation only because the full message history is resent on every call. This single fact drives the architecture: everything is about what goes *into* the `messages` list before a call, and how the program reacts to what comes *back*.

```
┌──────────────────────────────────────────────────────────────┐
│  CLI / entry loop                                             │
│   reads user input, prints streamed output                    │
└───────────────┬──────────────────────────────────────────────┘
                │
        ┌───────▼─────────┐     recall relevant memories (Phase 4)
        │  agent_turn()   │◄──────────────── memory store (Chroma)
        │  the agent loop │─────────────────► write durable facts (Phase 4)
        └───────┬─────────┘
                │ tools=[...]            ┌──────────────────────┐
        ┌───────▼─────────┐  tool_calls │  tool registry       │
        │  chat()         │◄────────────┤  get_time            │
        │  the ONLY call  │  results    │  web_search (Phase 3) │
        │  to the model   │─────────────►  fetch_url  (Phase 3) │
        └───────┬─────────┘             └──────────────────────┘
                │
        ┌───────▼─────────┐
        │  Ollama server  │  http://localhost:11434/v1
        │  (gemma3 / etc.)│  + nomic-embed-text for embeddings
        └─────────────────┘
```

**Component responsibilities**

- `chat()` — the single function that talks to the model. Nothing else calls the model API directly. This isolation is what makes the model swappable.
- `agent_turn()` — the agent loop: send messages + tools, detect tool calls, execute them, feed results back, repeat until the model returns a final answer.
- **Tool registry** — a name→schema list (sent to the model) and a name→function map (executed by the loop). Adding a capability = adding one entry to each.
- **Memory store** — a vector database queried before each turn (recall) and written after each turn (durable facts).

---

## 3. Tech stack

| Concern | Choice | Notes |
|---|---|---|
| Model runtime | **Ollama** | Background daemon, OpenAI-compatible API at `http://localhost:11434/v1`, official SDKs, good for code/concurrency. |
| Default model | **gemma3** | Swappable. See Gemma caveat (§9). For reliable tool use, `qwen2.5` or `llama3.1` are strong fallbacks. |
| Embedding model | **nomic-embed-text** | Served by Ollama via `/v1/embeddings`. Keep separate from the chat model. |
| Language | **Python 3.11+** | Best ecosystem for vector DBs / embeddings. |
| Model client | **openai** SDK | Point `base_url` at Ollama. Keeps code portable. |
| Vector store | **chromadb** | In-process, persists to disk, no server. Swappable for LanceDB/Qdrant. |
| Web search | **Tavily** (default) | LLM-optimized results. Alternatives: Brave Search API, or self-hosted SearXNG for full privacy. |
| Page fetch | **httpx** + **beautifulsoup4** | For pulling full article text when snippets are insufficient. |

**Dependencies**

```
pip install openai chromadb tavily-python httpx beautifulsoup4
```

```
ollama pull gemma3
ollama pull nomic-embed-text
```

---

## 4. Project structure

```
assistant/
├── config.py          # model name, base_url, keys, tunables — one source of truth
├── llm.py             # chat(): the only function that calls the model
├── agent.py           # agent_turn(): the tool loop
├── tools/
│   ├── __init__.py    # TOOLS schema list + TOOL_FUNCTIONS registry
│   ├── time_tool.py   # Phase 2 trivial tool
│   ├── search.py      # Phase 3 web_search + fetch_url
│   └── memory_tool.py # Phase 4 optional model-driven save
├── memory/
│   ├── store.py       # embed(), save_memory(), recall()
│   └── memory_db/     # Chroma persistence (gitignored)
├── main.py            # CLI entry loop, history management
└── tests/
    └── test_phases.py # acceptance checks per phase
```

Build order maps to files: Phase 1 → `config/llm/main`; Phase 2 → `agent/tools/`; Phase 3 → `tools/search.py`; Phase 4 → `memory/`.

---

## 5. Configuration

All tunables live in `config.py`. Nothing downstream hardcodes a model name or URL.

```python
# config.py
OLLAMA_BASE_URL = "http://localhost:11434/v1"
CHAT_MODEL      = "gemma3"            # swap to qwen2.5 / llama3.1 if tool use is flaky
EMBED_MODEL     = "nomic-embed-text"

MEMORY_DB_PATH  = "./memory/memory_db"
RECALL_K        = 3                   # how many memories to inject per turn
RECALL_MAX_DIST = 1.0                 # drop matches weaker than this (tune empirically)

TAVILY_API_KEY  = os.environ["TAVILY_API_KEY"]
MAX_FETCH_CHARS = 8000               # truncate fetched pages to protect context window

SYSTEM_PROMPT = (
    "You are a concise, helpful personal assistant running locally. "
    "Today's date is {date}. "                # inject real date at runtime
    "When a question needs current data, computation, or facts you are unsure of, "
    "you MUST call the appropriate tool rather than guessing. "
    "When you use web_search, cite sources inline like [1], [2] matching the results."
)
```

---

## 6. Phase 1 — Core chat loop

**Goal:** a streaming, multi-turn CLI chat with the model call isolated behind one function.

**Requirements**

1. `chat(messages, stream=False)` in `llm.py` is the only place that calls the model API. It uses the `openai` SDK pointed at `OLLAMA_BASE_URL`.
2. `main.py` maintains a `messages` list seeded with the system prompt (date injected at runtime).
3. Each turn appends the user message, calls the model, prints the reply, and **appends the assistant reply back to `messages`** (this is what creates conversational memory).
4. Responses stream token-by-token to the terminal.
5. History trimming/summarization stub exists for when the conversation grows (can be a no-op initially, but the seam must be there).
6. **Startup health check** — before the CLI loop starts, confirm the Ollama daemon is reachable and that `CHAT_MODEL` (and, once Phase 4 lands, `EMBED_MODEL`) is pulled. On failure, exit with a clear, actionable message (e.g. "Ollama not reachable at {url} — is `ollama serve` running?" or "Model 'gemma3' not found — run `ollama pull gemma3`") rather than letting the first model call throw a raw connection error.

**Interface**

```python
def chat(messages: list[dict], stream: bool = False, tools: list | None = None):
    """Single entry point to the model. Returns the message object (non-stream)
    or yields content deltas (stream)."""
```

**Acceptance criteria**

- A multi-turn conversation correctly recalls something said two turns earlier.
- Output streams rather than appearing all at once.
- Changing `CHAT_MODEL` in config (to any pulled model) works with no other edits.
- With Ollama stopped, the program exits immediately with a clear message; with a model not pulled, it names the exact `ollama pull` command to run.

---

## 7. Phase 2 — Tool-calling scaffold

**Goal:** the model can call a function, receive the result, and answer from it. Prove the registry pattern scales with two tools.

**Requirements**

1. A tool is defined as two synced parts: a JSON-schema entry in `TOOLS` (sent to the model) and a callable in `TOOL_FUNCTIONS` (executed by the loop).
2. `agent_turn()` implements the loop: send `messages` + `TOOLS`; if the response has `tool_calls`, append the assistant message, execute each call, append a `tool` message carrying the matching `tool_call_id`, then loop again; if no tool calls, return the final answer.
3. Tool execution is wrapped in try/except; errors are returned to the model as text (`"ERROR: ..."`) rather than crashing.
4. Ship two trivial tools (`get_current_time`, `calculate`) to prove adding a tool requires no change to the loop.
5. Build this **non-streaming first** — streaming + tool calls is fiddly and comes later.

**The message contract** (must be exact or the API errors on orphaned tool calls):

1. `user` → request
2. `assistant` → no content, populated `tool_calls`
3. `tool` → result, matching `tool_call_id`
4. `assistant` → final answer (produced on the next loop iteration)

When appending the step-2 assistant message back into history, serialize it (`msg.model_dump()`) rather than appending the raw SDK object.

**Interface**

```python
def agent_turn(messages: list[dict]) -> str:
    """Run one full turn, resolving any tool calls. Returns final assistant text."""
```

**Acceptance criteria**

- "What time is it?" triggers the time tool and the answer reflects the result.
- Adding the second tool requires editing only `TOOLS` + `TOOL_FUNCTIONS`, not `agent_turn`.
- A tool that raises is reported gracefully and the model recovers.

---

## 8. Phase 3 — Web search

**Goal:** the assistant looks up current information and cites it. Web search is just the first *real* tool.

**Requirements**

1. `web_search(query, max_results=5)` calls the search provider and returns **text** (not raw JSON), formatted as numbered, citeable blocks: `[n] title\nURL\nsnippet`.
2. Register it in `TOOLS` + `TOOL_FUNCTIONS`. `agent_turn` is unchanged.
3. The search provider is isolated inside the function so Tavily → Brave → SearXNG is a swap of one function body.
4. Optional `fetch_url(url)` tool returns cleaned page text, truncated to `MAX_FETCH_CHARS`, for when snippets are insufficient or the user pastes a link.
5. System prompt instructs the model to cite sources inline `[1]`, `[2]` matching the numbered results, and to say so when results don't answer the question.
6. The current date is injected into the system prompt so the model can reason about what counts as "recent."

**Acceptance criteria**

- A question about current events triggers a search and the answer cites sources.
- A question the model already knows does **not** trigger an unnecessary search.
- `fetch_url` returns readable text for a known article URL, capped at the char limit.

---

## 9. Phase 4 — Long-term memory

**Goal:** the assistant recalls durable facts about the user across sessions via semantic retrieval.

**Requirements**

1. `embed(text)` returns an embedding from `EMBED_MODEL` via Ollama's `/v1/embeddings`.
2. Memory store is a persistent Chroma collection at `MEMORY_DB_PATH`.
3. **Write path** — `save_memory(text, kind="fact")` embeds and stores text with metadata (`kind`, timestamp) and a unique id. Before writing, check for a near-duplicate and skip/update instead of adding.
4. **Recall path** — `recall(query, k=RECALL_K, max_distance=RECALL_MAX_DIST)` embeds the query, returns the nearest memories, and **drops matches weaker than the distance threshold** so irrelevant memories aren't injected.
5. **Injection** — before each turn, `recall(user_input)` runs; any hits are injected as a transient system message ("Relevant things you remember about the user: ...") for that turn only. The rest of the loop is unchanged.
6. **Write policy** — extract only durable facts (preferences, names, constraints), not chitchat. Default to a heuristic/code-driven extraction pass (more reliable than tool calls with Gemma); optionally also expose `save_memory` as a model-callable tool.

**Acceptance criteria**

- Tell it a durable fact, **fully restart the program**, ask a related question, and it recalls and respects the fact.
- Asking something unrelated injects no memories (threshold works).
- Saying the same fact twice does not create two memory entries.

**Known issues to handle**

- **Staleness/contradiction** — "I live in Boston" then "I moved to Austin": prefer recent memories via timestamp; include a periodic reconciliation pass.
- **Recall pollution** — keep `k` small (3–5) and the distance filter on; precision over recall.
- **Context budget** — injected memories + tool results + history compete for the (smaller, local) context window; summarize old turns when history grows.
- **Embedding consistency** — all vectors in a collection must come from the same embedding model (dimensions must match). Changing `EMBED_MODEL` requires rebuilding the store; don't mix.

---

## 10. Cross-cutting concerns

**Model swappability.** Every model reference goes through `config.py` and `chat()`. Test with `gemma3` and at least one of `qwen2.5` / `llama3.1` to confirm nothing else is coupled to the model.

**The Gemma caveat.** Gemma has no native tool-calling tokens — tool use is simulated via the chat template, and reliability varies. Symptoms: ignoring tools, or emitting a tool call as plain text instead of populating `tool_calls`. Mitigations, in order: (a) explicit system-prompt instruction to use tools; (b) swap the agent/controller model to `qwen2.5`/`llama3.1`; (c) consider FunctionGemma as a small routing model. Because of this, prefer **code-driven** memory writes over model-driven tool calls.

**Streaming + tools.** Build the tool loop non-streaming. Add streaming only for the final assistant answer once the mechanism is solid (streamed tool-call deltas must be accumulated across chunks before execution).

**Error handling.** Tool exceptions return to the model as text, never crash the loop. Network failures in search degrade to "I couldn't reach the web" rather than a stack trace.

**Logging.** Log each model call, tool call, and memory write/recall at debug level — essential for diagnosing the "why didn't it call the tool / recall the memory" questions that dominate this build.

**Privacy.** Everything is local except the search provider. For a fully offline posture, switch web search to self-hosted SearXNG.

---

## 11. Testing & acceptance summary

| Phase | One-line proof it works |
|---|---|
| 1 | Multi-turn chat recalls earlier context; output streams; model swaps via config only. |
| 2 | "What time is it?" calls the tool; second tool added without touching the loop. |
| 3 | Current-events question searches + cites; known question does not search. |
| 4 | Fact survives a full program restart and is recalled on a related question. |

Each phase must pass its proof before the next begins.

---

## 12. Stretch goals (Phase 5+)

- **MCP servers** — expose tools via Model Context Protocol so the same tools work across clients; Ollama and LM Studio both support MCP.
- **More tools** — filesystem/RAG over the user's own documents, calendar, email, shell.
- **A UI** — replace the CLI with a web frontend (e.g. Open WebUI) or a small custom app.
- **Voice** — local STT (whisper.cpp) + TTS for a spoken assistant.
- **Memory reflection** — a scheduled background pass that consolidates, deduplicates, and reconciles contradictory memories.
- **Multi-model routing** — a small fast model (FunctionGemma) routes/handles simple calls; a larger model handles reasoning.

---

## Appendix — Reference patterns

These are illustrative interfaces, not prescriptive implementations. Claude Code should implement them idiomatically.

```python
# llm.py — the single model entry point
from openai import OpenAI
import config

client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")  # key ignored locally

def chat(messages, stream=False, tools=None):
    return client.chat.completions.create(
        model=config.CHAT_MODEL, messages=messages, tools=tools, stream=stream
    )
```

```python
# health.py — startup check; call once before the CLI loop
import sys, httpx
import config

def preflight(required_models: list[str]):
    base = config.OLLAMA_BASE_URL.rsplit("/v1", 1)[0]   # -> http://localhost:11434
    try:
        tags = httpx.get(f"{base}/api/tags", timeout=5).json()
    except Exception:
        sys.exit(f"Ollama not reachable at {base} — is `ollama serve` running?")
    installed = {m["name"].split(":")[0] for m in tags.get("models", [])}
    for model in required_models:
        if model.split(":")[0] not in installed:
            sys.exit(f"Model '{model}' not found — run `ollama pull {model}`")

# usage in main.py, before the loop:
#   preflight([config.CHAT_MODEL])                       # Phase 1–3
#   preflight([config.CHAT_MODEL, config.EMBED_MODEL])   # once Phase 4 lands
```

```python
# agent.py — the tool loop (non-streaming)
import json
from llm import chat
from tools import TOOLS, TOOL_FUNCTIONS

def agent_turn(messages):
    while True:
        msg = chat(messages, tools=TOOLS).choices[0].message
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content
        messages.append(msg.model_dump())  # serialize, don't append the raw SDK object
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            try:
                result = TOOL_FUNCTIONS[call.function.name](**args)
            except Exception as e:
                result = f"ERROR: {e}"
            messages.append({
                "role": "tool", "tool_call_id": call.id, "content": str(result),
            })
```

```python
# memory/store.py — embed, write, recall
import time, uuid, chromadb
from openai import OpenAI
import config

client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
col = chromadb.PersistentClient(path=config.MEMORY_DB_PATH) \
        .get_or_create_collection("memories")

def embed(text):
    return client.embeddings.create(model=config.EMBED_MODEL, input=text).data[0].embedding

def save_memory(text, kind="fact"):
    # TODO: near-duplicate check before adding
    col.add(ids=[str(uuid.uuid4())], embeddings=[embed(text)],
            documents=[text], metadatas=[{"kind": kind, "ts": time.time()}])

def recall(query, k=config.RECALL_K, max_distance=config.RECALL_MAX_DIST):
    res = col.query(query_embeddings=[embed(query)], n_results=k)
    return [d for d, dist in zip(res["documents"][0], res["distances"][0])
            if dist <= max_distance]
```

---

*Build one phase, prove its checkpoint, review, then proceed. The Phase 1 loop is the spine; nothing later replaces it.*
