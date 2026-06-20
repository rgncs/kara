"""Single source of truth for all tunables.

Nothing downstream hardcodes a model name, URL, or path — it all comes from here,
so swapping a model or storage location is a one-line change.
"""
import os


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a repo-root .env into the environment (without
    overriding real env vars). Makes secrets like TAVILY_API_KEY work regardless of
    which terminal/shell launched Kara — no `source ~/.zshrc` needed."""
    path = os.path.join(os.path.dirname(__file__), "..", ".env")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                if key:
                    os.environ.setdefault(key, val.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_dotenv()  # must run before any os.environ.get below

# --- Model runtime (Ollama, OpenAI-compatible API) ---------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# qwen3-coder is a code-tuned MoE with reliable native tool-calling — a far better
# agent controller than gemma3 (whose tool use is simulated and flaky).
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen3-coder:30b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# --- Coding agent workspace (Phase 2 coding tools) ---------------------------
# File tools are confined to this root so the agent can't roam the whole disk.
# Defaults to the directory the agent is launched from.
WORKSPACE_ROOT = os.path.abspath(os.environ.get("WORKSPACE_ROOT", os.getcwd()))
RUN_COMMAND_TIMEOUT = 60      # seconds before a shell command is killed
MAX_TOOL_OUTPUT_CHARS = 8000  # truncate large file/command output to protect context

# How run_command is gated before executing:
#   "prompt" (default) — ask the user y/n/always per command (CLI installs the prompt)
#   "auto"             — run without asking (use only when you trust the workspace/task)
#   "deny"             — never run shell commands
COMMAND_APPROVAL = os.environ.get("COMMAND_APPROVAL", "prompt").lower()

# --- Memory (Phase 4) --------------------------------------------------------
# Two scopes:
#   global — cross-project personal/durable facts (this install's store).
#   local  — facts tied to the launch directory, in <workspace>/.kara_memory.
# Recall searches both; "remember forever" → global, "remember for this project" → local.
MEMORY_DB_PATH = os.environ.get("MEMORY_DB_PATH", os.path.join(os.path.dirname(__file__), "memory", "memory_db"))
LOCAL_MEMORY_DB_PATH = os.environ.get(
    "LOCAL_MEMORY_DB_PATH", os.path.join(WORKSPACE_ROOT, ".kara_memory"))
RECALL_K = 6                 # how many memories to inject per turn (threshold still gates relevance)
# Cosine-distance thresholds (collection uses hnsw:space=cosine). Calibrated
# empirically against nomic-embed-text with query/document prefixes — see
# scripts/calibrate_memory.py (related ≤ ~0.49, unrelated ≥ ~0.53).
RECALL_MAX_DIST = 0.52       # drop recall matches further than this (irrelevant)
MEMORY_DUP_DIST = 0.10       # treat as a duplicate when a new fact is this close
TRASH_TTL_DAYS = 365         # recently-deleted memories are purged for good after this
# Memory writes are code-driven and reliable: a regex pass for self-facts plus an
# LLM extraction pass triggered by remember/remind cues (handles facts about other
# people and reminders). The model's save_memory tool is off by default because its
# tool-calling is inconsistent; set MEMORY_TOOL=1 to also expose it.
MEMORY_TOOL_ENABLED = os.environ.get("MEMORY_TOOL", "").lower() in {"1", "true", "yes"}

# --- Web search (Phase 3) ----------------------------------------------------
# Provider is swappable: "tavily" (cloud, LLM-optimized) or "searxng" (self-hosted).
SEARCH_PROVIDER = os.environ.get("SEARCH_PROVIDER", "tavily").lower()
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")  # used when provider=searxng
MAX_SEARCH_RESULTS = 5
# TAVILY_API_KEY is read lazily via require_tavily_key() — Phases 1–2 run without it.
MAX_FETCH_CHARS = 8000       # truncate fetched pages to protect the context window

# --- History management ------------------------------------------------------
HISTORY_MAX_MESSAGES = 40    # when history exceeds this, the trim seam kicks in (Phase 1 no-op)

# The agent's identity. Override with the AGENT_NAME env var if you ever rename it.
AGENT_NAME = os.environ.get("AGENT_NAME", "Kara")

# --- Voice (Phase 5: spoken assistant) ---------------------------------------
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")  # faster-whisper STT model
VOICE_SAMPLE_RATE = 16000        # whisper expects 16 kHz mono

# Text-to-speech engine: "auto" uses Piper (neural) if its model is present, else
# falls back to macOS `say". Force with "piper" or "say".
TTS_ENGINE = os.environ.get("TTS_ENGINE", "auto").lower()
PIPER_MODEL = os.path.abspath(os.environ.get(
    "PIPER_MODEL",
    os.path.join(os.path.dirname(__file__), "..", "voices", "en_US-amy-medium.onnx")))
PIPER_LENGTH_SCALE = os.environ.get("PIPER_LENGTH_SCALE", "")  # >1 slower, <1 faster

# macOS `say` settings (used when TTS_ENGINE resolves to "say")
SAY_VOICE = os.environ.get("SAY_VOICE", "")   # empty = auto-pick best installed voice
SAY_RATE = os.environ.get("SAY_RATE", "")     # words per minute (empty = default)

# Phonetic respellings applied ONLY to spoken output so names sound right.
# "Wontaek" is pronounced won-tek.
PRONUNCIATIONS = {"Wontaek": "Wontek"}

# Hands-free conversation: listen continuously with voice-activity detection
# instead of press-Enter-to-talk. Set VOICE_HANDS_FREE=0 for push-to-talk.
VOICE_HANDS_FREE = os.environ.get("VOICE_HANDS_FREE", "1").lower() in {"1", "true", "yes"}
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))   # webrtcvad 0..3 (higher = stricter)
VAD_SILENCE_MS = int(os.environ.get("VAD_SILENCE_MS", "800"))         # trailing silence that ends a turn
VAD_START_MS = int(os.environ.get("VAD_START_MS", "150"))             # speech needed to start capturing
VAD_MIN_SPEECH_MS = int(os.environ.get("VAD_MIN_SPEECH_MS", "300"))   # ignore utterances shorter than this
# After "hey Kara", the conversation stays open (no wake word needed) until this many
# seconds pass with no speech — then she waits for "hey Kara" again.
VOICE_FOLLOWUP_TIMEOUT = int(os.environ.get("VOICE_FOLLOWUP_TIMEOUT", "10"))

# Spoken-response shaping: the full reply is always printed; if speaking it would
# run longer than VOICE_SUMMARY_THRESHOLD_S, Kara speaks a short summary instead and
# offers to go deeper. Code is never spoken aloud.
VOICE_WPM = int(os.environ.get("VOICE_WPM", "160"))                   # TTS speaking rate estimate
VOICE_SUMMARY_THRESHOLD_S = int(os.environ.get("VOICE_SUMMARY_THRESHOLD_S", "30"))

SYSTEM_PROMPT = (
    "You are {name}, a concise, capable coding agent running locally on the user's "
    "machine. Today's date is {date}. "
    "Wontaek Shin (pronounced won-tek) is your creator — he designed and built you, "
    "and he is an accomplished fintech engineering leader. You are dedicated to his "
    "success, your primary user, and serve as his devoted AI model helper — you look "
    "out for his goals and help him succeed. "
    "You are knowledgeable and practical across fintech, software development, and "
    "business and economics, and you help him expertly in those areas — but you do NOT "
    "have a human persona, career, or credentials of your own. That experience belongs "
    "to your creator, not to you; never claim a personal biography or pretend to be a person. "
    "You can read, write, and list files and run shell commands inside the user's "
    "workspace, search the web, and recall facts about the user and project. "
    "When a task needs the file system, a command, current data, computation, or "
    "facts you are unsure of, you MUST call the appropriate tool rather than guessing. "
    "Inspect files before editing them. Prefer small, verifiable steps and run tests "
    "or commands to check your work. Be direct and practical, and answer only what was "
    "asked — do NOT tack on reminders, suggestions, or personal trivia at the end of "
    "replies that aren't about them. "
    "If asked your name, you are {name}. "
    "You HAVE persistent long-term memory across sessions, and the relevant stored "
    "memories are provided to you each turn. Handle memory honestly:\n"
    "(1) If the user ASKS whether you know or remember something, answer ONLY from the "
    "memories you've been given — NOT from the fact mentioned in their question. A detail "
    "stated in the question is new input, not a memory: if it isn't in your provided "
    "memories, you do NOT remember it — say so plainly and OFFER to remember it (never "
    "claim you already knew it, and don't make one up). If they then confirm, it gets saved.\n"
    "(2) If the user TELLS you to remember something or shares a new durable detail, it "
    "is saved automatically; confirm CONCISELY in one short sentence (e.g. 'Got it — saved "
    "that.'). Do NOT phrase it as 'I remember' as though you already knew it — you are "
    "learning it now. NEVER list, recite, or summarize the other things you remember "
    "(no bullet-point recaps) unless the user explicitly asks what you know.\n"
    "(3) If the user says something was a joke/not real or asks you to forget it, it is "
    "moved to a recently-deleted stockpile (recoverable); confirm concisely (e.g. 'Done — "
    "I've removed that.'). If they ask to PERMANENTLY delete it, it is erased for good.\n"
    "(4) If you're told a deleted note exists for what they're asking about, say you don't "
    "have it actively but they previously deleted it, and offer to restore it.\n"
    "(5) Memory has two scopes — GLOBAL (everywhere) and LOCAL (just this project). Recall "
    "searches both. When a turn tells you to ask which scope, ask ONE short question like "
    "'forever, or just for this project?' — not a long explanation — and save nothing "
    "until they answer.\n"
    "Always follow the bracketed [memory note] for a turn exactly — it tells you what was "
    "saved/restored/deleted or what to ask. "
    "Never say you can't remember, lack persistent memory, or that conversations are "
    "independent. "
    "You DO have live internet access through the web_search and fetch_url tools — "
    "never claim you can't browse the web or lack internet access. "
    "CRITICAL: today's date ({date}) is well AFTER your training cutoff, so your built-in "
    "knowledge of anything recent is STALE and probably wrong. For ANY time-sensitive "
    "question you MUST call web_search FIRST and answer from the results — never from your "
    "2024–2025 training memory. This includes: current events and news; recommendations "
    "('good shows/movies/books/restaurants to watch or try', what's new/popular/trending/"
    "best right now); latest releases, versions, prices, scores, standings, or schedules; "
    "who currently holds any role or title; and anything phrased as 'today', 'this week', "
    "'currently', 'latest', or 'now'. When unsure whether something has changed since 2025, "
    "assume it has and search. Only answer from your own knowledge for timeless things "
    "(math, definitions, how-to, established history). "
    "NEVER respond by saying you'd 'need to check', that you 'don't have up-to-date "
    "information', that your data may be outdated, or by declining/hedging on a current "
    "question — in exactly those situations, just CALL web_search first and answer from the "
    "results. Search, then answer; don't announce that you would. "
    "When you use web_search, cite sources inline like [1], [2] matching the numbered "
    "results, and if the results don't answer the question, say so rather than guessing."
)


def require_tavily_key() -> str:
    """Return the Tavily API key, failing loudly only when web search is actually used."""
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set — required for web_search. "
            "Get a key at https://tavily.com and `export TAVILY_API_KEY=...`."
        )
    return key
