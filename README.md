# Kara — Local AI Coding Agent

**Kara** is a local-first, model-agnostic CLI **coding agent** built incrementally on
[`local-assistant-spec.md`](./local-assistant-spec.md). She reads/writes files and
runs commands in a sandboxed workspace, and (upcoming) searches the web and remembers
project context. All inference, embeddings, and storage run on your machine via
[Ollama](https://ollama.com) with `qwen3-coder:30b`; the only outbound dependency is
the web-search tool (Phase 3), and it's swappable.

The name is set by `AGENT_NAME` in [`assistant/config.py`](./assistant/config.py)
(or the `AGENT_NAME` env var) and flows into the system prompt and CLI.

## Status

| Phase | Capability | State |
|---|---|---|
| 1 | Streaming multi-turn chat, model call isolated behind `chat()` | ✅ done |
| 2 | Tool loop + coding tools (`read_file`, `write_file`, `list_dir`, `run_command`, `get_current_time`, `calculate`) | ✅ done |
| 3 | Web search + cite (`web_search`, `fetch_url`) — Tavily or self-hosted SearXNG | ✅ done |
| 4 | Long-term memory — recalls facts across restarts (Chroma + embeddings) | ✅ done |
| 5 | Voice — local speech-to-text (Whisper) + text-to-speech (`say`) | ✅ done |

All four core phases are complete, plus a voice mode. Kara streams chat, calls tools,
edits/runs code in a sandbox, searches and cites the web, remembers durable facts
across restarts, and can listen and speak.

The agent edits and runs code inside a **workspace sandbox** (`WORKSPACE_ROOT`,
defaults to the launch directory). File tools cannot escape that root; shell commands
run there with a timeout.

**Shell approval gate.** `run_command` is gated by `COMMAND_APPROVAL`:

| Mode | Behavior |
|---|---|
| `prompt` (default) | Asks `y` / `n` / `a` (always this session) before each command. |
| `auto` | Runs without asking — only for workspaces/tasks you trust. |
| `deny` | Never runs shell commands. |

A denied command is reported back to the model (it adapts) rather than executed. In
`prompt` mode with no interactive approver available (e.g. a piped/non-interactive
run), the gate **fails safe to deny**.

```bash
COMMAND_APPROVAL=auto .venv/bin/python assistant/main.py   # no prompts (trusted task)
```

> **Tool-call reliability note.** qwen3-coder often emits its tool calls as text
> (its `<function=…>` DSL) rather than populating the API's structured `tool_calls`
> field through Ollama. `tool_parse.py` recovers those, and the agent loop synthesizes
> a valid `tool_calls` message — so tool use is reliable despite the quirk.

> **New here?** See [`GETTING_STARTED.md`](./GETTING_STARTED.md) for a 2-minute
> walkthrough of starting the Ollama server and running the CLI.

## Install

```bash
./install.sh                 # macOS; idempotent — safe to re-run
```

The installer ensures Ollama + Python 3.11+ are present, creates `.venv` and installs
deps, pulls the chat and embedding models, and links a `kara` command onto your `PATH`.
Use `./install.sh --no-models` to skip the (large) model downloads.

Prereq: [Homebrew](https://brew.sh). The installer uses it to fetch Ollama/Python if missing.

## Run

`kara` works from **any** directory — the directory you launch it in becomes its
workspace (the only place its file tools and shell can touch):

```bash
cd /path/to/your/project
kara
```

The launcher auto-starts the Ollama daemon if it isn't already running.

Type a request (e.g. "add a test for parse_config and run it"). Kara resolves tool
calls then prints her answer. `exit`, `quit`, or Ctrl-D to leave.

Useful env overrides:

```bash
WORKSPACE_ROOT=/some/project kara   # target a different workspace
COMMAND_APPROVAL=auto kara          # don't prompt before shell commands (trusted task)
ASSISTANT_DEBUG=1 kara              # log every model + tool call to stderr
AGENT_NAME=Ada kara                 # rename the agent
```

## Web search (Phase 3)

Kara can search the web and cite sources. The provider is set by `SEARCH_PROVIDER`:

**Tavily (default)** — cloud, LLM-optimized results. Needs a free key:

```bash
export TAVILY_API_KEY=tvly-...        # get one at https://tavily.com
kara
```

**SearXNG (self-hosted)** — private, no key, runs in Docker:

```bash
bin/searxng start                     # starts the container (needs Docker)
SEARCH_PROVIDER=searxng kara
bin/searxng stop                      # when done
```

`web_search` returns numbered, citeable blocks; Kara cites `[1]`, `[2]` inline and
won't search for things it already knows. `fetch_url` pulls cleaned page text
(capped at `MAX_FETCH_CHARS`) when a snippet isn't enough. Network failures degrade
to a readable message rather than crashing.

## Long-term memory (Phase 4)

Kara remembers durable facts about you (name, preferences, where you live/work)
across restarts, via semantic recall over a local [Chroma](https://www.trychroma.com)
store embedded with `nomic-embed-text`.

- **Write** — after each turn, durable facts are stored (near-duplicates skipped), via
  two **deterministic** code passes (no dependence on the model's flaky extraction):
  - A **regex pass** for obvious self-facts ("My name is…", "I prefer…", "I live in…").
  - A **remember-cue pass**: when you signal intent ("remember…", "remind me…", "don't
    forget…", anywhere in the sentence), it strips the cue, normalizes pronouns, and
    stores exactly what you said — so multi-fact requests ("remember my girlfriend's name
    is X, born Y, from Z") keep every detail, and reminders are framed as "Remind Wontaek
    to …". This never silently drops a detail the way model-based extraction did.

  Each save is confirmed on screen (`· remembered: …`). The model's own `save_memory`
  tool is off by default (its tool-calling is inconsistent); `MEMORY_TOOL=1` to enable.
- **Two scopes** — memory is **global** (cross-project, this install's store) or **local**
  (this launch directory, in `<dir>/.kara_memory`). Recall searches both.
  - "remember X **forever**" / "everywhere" → global; "remember X **for this project**" /
    "locally" → local; a plain "remember X" → Kara **asks** "forever or just for this project?"
    and saves to your answer (`· remembered (everywhere): …` / `· remembered (this project): …`).
- **Honest recall & retraction** — Kara distinguishes:
  - *Telling* her ("remember X") → saves and concisely confirms ("Got it — saved that").
  - *Asking* her ("do you remember X?") → answers only from memory; if she doesn't have
    it she says so and offers to remember it, saving only after you confirm ("yes").
  - *Retracting* — a bare "that was a joke" / "never mind" drops the **last** thing saved
    immediately; a targeted "forget X" finds the closest memory and **confirms before
    deleting** (so an ordinary "delete that function" coding request never touches memory).
    Deletes are soft → a recently-deleted stockpile (`· moved to recently deleted: …`),
    auto-emptied after a year.
  - *Permanent delete* ("permanently delete X") → erased for good, no undo.
  - *Restore* — when you later ask about something only in the stockpile, she offers to
    restore it; say "yes" and it's back (`· restored: …`).

  She never claims to remember something she's only just been told.
- **Recall** — before each turn, the most relevant memories (cosine distance under
  `RECALL_MAX_DIST`) are folded into your message so Kara answers from them. Unrelated
  questions recall nothing.
- **Storage** — `assistant/memory/memory_db/` (gitignored). All vectors must come from
  one embedding model; changing `EMBED_MODEL` means rebuilding the store (delete that dir).
- Thresholds were calibrated empirically — re-run `scripts/calibrate_memory.py` if you
  change the embedding model.

## Voice mode (Phase 5)

Talk to Kara and hear her replies — all local. Speech-to-text is Whisper
(`faster-whisper`); text-to-speech is the macOS `say` command.

```bash
kara --voice          # or: VOICE=1 kara
```

**Hands-free (default):** say **"hey Kara"** to start (or "hey Cara", or any homophone
Whisper produces — she doesn't fuss over the pronunciation). After that the conversation
stays open — **just keep talking, no wake word needed** for the back-and-forth. If you go
quiet for ~`VOICE_FOLLOWUP_TIMEOUT` seconds (default 10), she pauses and waits for "hey
Kara" again. **Tap any key while she's talking to interrupt.** Say "hey Kara, goodbye" or
press Ctrl-C to quit.

Set `VOICE_HANDS_FREE=0` for classic **push-to-talk** (Enter to start/stop, `t` to
type a turn).

The first run prompts for **microphone permission** (System Settings → Privacy &
Security → Microphone) and downloads the Whisper model once. Tuned for **speakers**
(mic is closed while Kara speaks, so she won't hear herself).

VAD tunables if it's too eager/sluggish: `VAD_AGGRESSIVENESS` (0–3, higher = stricter),
`VAD_SILENCE_MS` (pause that ends your turn, default 800), `VAD_MIN_SPEECH_MS`.

**Spoken-response protocol.** The full reply is always printed to the screen, but
spoken output is **hard-capped at ~30 seconds** (`VOICE_SUMMARY_THRESHOLD_S`): anything
longer is condensed to a ~30s summary spoken directly (no preamble). For more detail on
a point, just ask about it — that's a fresh (also ≤30s) answer. **Code is never spoken
aloud** — she points you to the printed version. Tune with `VOICE_SUMMARY_THRESHOLD_S`
and `VOICE_WPM` (speaking-rate estimate).

**Voice quality.** TTS is selected by `TTS_ENGINE`:

- `auto` (default) — uses **Piper** (local neural TTS) if its model is present, else
  macOS `say`. A natural voice (`en_US-amy-medium`) is downloaded to `voices/` during setup.
- `piper` / `say` — force one engine.

When using `say`, Kara auto-picks the most natural installed macOS voice (Premium >
Enhanced > default); download a Premium voice via System Settings → Accessibility →
Spoken Content for a quality jump, or just rely on Piper.

Other Piper voices: drop a `.onnx` + `.onnx.json` from
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) into `voices/` and
set `PIPER_MODEL=voices/<name>.onnx`.

Tunables: `WHISPER_MODEL` (default `base.en`), `TTS_ENGINE`, `PIPER_MODEL`,
`PIPER_LENGTH_SCALE` (>1 slower), `SAY_VOICE`, `SAY_RATE`. Name pronunciations for
spoken output live in `config.PRONUNCIATIONS`.

## Configuration

Everything tunable lives in [`assistant/config.py`](./assistant/config.py) — the single
source of truth. To swap the model, change `CHAT_MODEL` (or set the `CHAT_MODEL` env var)
to any pulled Ollama model; no other edit is needed.

```bash
CHAT_MODEL=qwen2.5 .venv/bin/python assistant/main.py   # one-line model swap
```

## Tests

```bash
.venv/bin/python -m pytest assistant/tests/ -v
```

Unit tests (preflight messaging, history trimming) always run. Live tests that need a
running model are skipped automatically when Ollama isn't reachable or the model isn't pulled.

## Architecture

The model is **stateless** — it appears to remember a conversation only because the full
message history is resent on every call. Everything is about what goes *into* the
`messages` list before a call and how we react to what comes back.

- `llm.py::chat()` — the **only** function that talks to the model. Isolating it here is
  what makes the model swappable.
- `health.py::preflight()` — startup check: fails fast with an actionable message if Ollama
  is down or the model isn't pulled.
- `history.py::trim()` — the seam where history trimming/summarization lives as the
  conversation grows past the context window.
- `main.py` — the CLI loop: read input → stream reply → append reply back to history.
