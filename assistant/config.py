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
# Max tool-resolution steps in one turn before the loop gives up (safety valve against
# a model looping on tool calls). High enough for long multi-step tasks like spam cleanup.
MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "30"))

# Kara's OWN identity facts that the creator can set (birthday, birthplace) — they
# override the built-in defaults. Persisted here (gitignored, user-customized state),
# separate from the user-fact memory store.
SELF_FACTS_PATH = os.path.abspath(os.environ.get(
    "SELF_FACTS_PATH", os.path.join(os.path.dirname(__file__), "..", "kara_self.json")))

# --- Subagent models (model routing) -----------------------------------------
# The main loop uses CHAT_MODEL. Cheap helper jobs (spoken summaries, rewrites)
# route to a small fast model; genuinely hard planning can be delegated to a
# reasoning model via the `think` tool (it emits a <think> trace that is stripped
# before use). Both are loaded on demand by Ollama; missing ones degrade
# gracefully (helper falls back, the tool returns an error) rather than crashing.
FAST_MODEL = os.environ.get("FAST_MODEL", "qwen3.5:4b")
REASONING_MODEL = os.environ.get("REASONING_MODEL", "deepseek-r1")
# Expose the reasoning model to the main model as the `think` tool. OFF by default
# for now — the reasoning model runs the GPU hard; re-enable with REASONING=1.
REASONING_ENABLED = os.environ.get("REASONING", "0").lower() in {"1", "true", "yes"}

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

# --- Google Calendar (OAuth tool) --------------------------------------------
# OAuth "desktop app" flow: you create credentials.json in the Google Cloud
# console (Calendar API enabled), and the first run writes token.json (a refresh
# token) after a one-time browser consent. Both files are secrets — gitignored.
# Scope `calendar` = read/write; reads are free, but create/delete go through the
# confirm_action gate (CALENDAR_CONFIRM_WRITES) unless you turn it off.
GOOGLE_CREDENTIALS_PATH = os.path.abspath(os.environ.get(
    "GOOGLE_CREDENTIALS_PATH", os.path.join(os.path.dirname(__file__), "..", "credentials.json")))
GOOGLE_TOKEN_PATH = os.path.abspath(os.environ.get(
    "GOOGLE_TOKEN_PATH", os.path.join(os.path.dirname(__file__), "..", "token.json")))
# Multiple Google accounts can be connected (e.g. work + personal). The FIRST label
# keeps the original token.json (no re-auth); each other label gets token_<label>.json
# beside it. Authorize each once: `python assistant/tools/google_auth.py <label>`.
GOOGLE_ACCOUNTS = [a.strip() for a in os.environ.get("GOOGLE_ACCOUNTS", "work,personal").split(",") if a.strip()]
CALENDAR_ID = os.environ.get("CALENDAR_ID", "primary")
CALENDAR_CONFIRM_WRITES = os.environ.get(
    "CALENDAR_CONFIRM_WRITES", "1").lower() in {"1", "true", "yes"}
# Calendar / Gmail tools are offered to the model only once set up — credentials.json
# exists — or when forced on with CALENDAR=1 / GMAIL=1.
_HAS_GOOGLE_CREDS = os.path.exists(GOOGLE_CREDENTIALS_PATH)
CALENDAR_ENABLED = (os.environ.get("CALENDAR", "").lower() in {"1", "true", "yes"}
                    or _HAS_GOOGLE_CREDS)
GMAIL_ENABLED = (os.environ.get("GMAIL", "").lower() in {"1", "true", "yes"}
                 or _HAS_GOOGLE_CREDS)
# Gmail: read/search is free; sending, trashing, and unsubscribing go through the
# confirm_action gate (every one confirmed). Deletes go to Trash (recoverable), not
# permanent. gmail.modify covers read + trash + labels; gmail.send covers sending.
GMAIL_CONFIRM_WRITES = os.environ.get(
    "GMAIL_CONFIRM_WRITES", "1").lower() in {"1", "true", "yes"}

# One OAuth token covers whatever Google services are enabled. Expanding this list
# (e.g. turning Gmail on) invalidates the old token, so re-run the authorize step.
GOOGLE_SCOPES = []
if CALENDAR_ENABLED:
    GOOGLE_SCOPES.append("https://www.googleapis.com/auth/calendar")
if GMAIL_ENABLED:
    GOOGLE_SCOPES += ["https://www.googleapis.com/auth/gmail.modify",
                      "https://www.googleapis.com/auth/gmail.send"]

# --- Email spam cleanup ------------------------------------------------------
# A periodic scan flags senders with MORE THAN SPAM_UNREAD_THRESHOLD unread emails
# as spam candidates and logs them to SPAM_LOG_PATH. Nothing is deleted
# automatically — you review and confirm each sender via the spam-cleanup flow.
# The in-app scanner runs every SPAM_SCAN_INTERVAL seconds while Kara is open
# (and once at startup if the last scan is stale).
SPAM_SCAN_ENABLED = os.environ.get("SPAM_SCAN", "1").lower() in {"1", "true", "yes"}
SPAM_UNREAD_THRESHOLD = int(os.environ.get("SPAM_UNREAD_THRESHOLD", "10"))
SPAM_SCAN_INTERVAL = int(os.environ.get("SPAM_SCAN_INTERVAL", "21600"))  # 6 hours
SPAM_SCAN_MAX = int(os.environ.get("SPAM_SCAN_MAX", "300"))  # max unread sampled per scan
SPAM_LOG_PATH = os.path.abspath(os.environ.get(
    "SPAM_LOG_PATH", os.path.join(os.path.dirname(__file__), "..", "spam_candidates.json")))
# Large mailboxes: when a deep scan would cover more than SPAM_BATCH_THRESHOLD unread,
# it runs in batches of SPAM_BATCH_SIZE, checkpointing progress to SPAM_SCAN_STATE_PATH
# after each page so it resumes where it left off if interrupted, and announces after
# each batch.
SPAM_BATCH_THRESHOLD = int(os.environ.get("SPAM_BATCH_THRESHOLD", "20000"))
SPAM_BATCH_SIZE = int(os.environ.get("SPAM_BATCH_SIZE", "2000"))
SPAM_SCAN_STATE_PATH = os.path.abspath(os.environ.get(
    "SPAM_SCAN_STATE_PATH", os.path.join(os.path.dirname(__file__), "..", "spam_scan_state.json")))
# Senders you've marked "keep" — excluded from every scan and cleanup. Holds full
# addresses (team@x.com) and/or bare domains (x.com).
SPAM_KEEP_PATH = os.path.abspath(os.environ.get(
    "SPAM_KEEP_PATH", os.path.join(os.path.dirname(__file__), "..", "spam_keep.json")))
# Senders you've CONFIRMED as junk — the background scan auto-trashes their unread
# with no further prompts (you authorized them once by adding them here). Recoverable
# (Trash), unread-only. Same address/domain format as the keep-list.
SPAM_AUTODELETE_PATH = os.path.abspath(os.environ.get(
    "SPAM_AUTODELETE_PATH", os.path.join(os.path.dirname(__file__), "..", "spam_autodelete.json")))

# --- Skills (markdown playbooks) ---------------------------------------------
# A skill is a vetted Markdown file with frontmatter (name/description/triggers)
# holding step-by-step instructions for a task, loaded into a turn when its
# triggers match. Inspired by OpenClaw/Anthropic Agent Skills, but skills are
# PROSE only: they drive Kara's existing tools and never bypass the run_command
# approval gate. Files live in SKILLS_DIR and are committed, not auto-installed.
SKILLS_ENABLED = os.environ.get("SKILLS", "1").lower() in {"1", "true", "yes"}
SKILLS_DIR = os.path.abspath(os.environ.get(
    "SKILLS_DIR", os.path.join(os.path.dirname(__file__), "skills")))
MAX_SKILLS_PER_TURN = int(os.environ.get("MAX_SKILLS_PER_TURN", "2"))

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
# Piper synthesizes at 22050 Hz, but many output devices (USB headsets, etc.) are
# locked at 48000 Hz and afplay can't start a 22050 Hz queue on them
# ("AudioQueueStart failed"). Resample to this rate before playback (0 disables).
PIPER_PLAYBACK_RATE = int(os.environ.get("PIPER_PLAYBACK_RATE", "48000"))

# macOS `say` settings (used when TTS_ENGINE resolves to "say")
SAY_VOICE = os.environ.get("SAY_VOICE", "")   # empty = auto-pick best installed voice
SAY_RATE = os.environ.get("SAY_RATE", "")     # words per minute (empty = default)

# Phonetic respellings applied ONLY to spoken output so names sound right.
# "Wontaek" is pronounced won-tek.
PRONUNCIATIONS = {"Wontaek": "Wontek"}

# How to interrupt Kara while she's speaking:
#   "key"   — tap a key (safe with SPEAKERS: the mic would otherwise hear Kara's own
#             voice and falsely interrupt). This is the default.
#   "voice" — just start talking (barge-in; for HEADPHONES, where the mic can't hear
#             Kara). Set by --speaker / --headphone, or the VOICE_INTERRUPT env var.
VOICE_INTERRUPT = os.environ.get("VOICE_INTERRUPT", "key").lower()
# Voice barge-in only cuts Kara off if the captured audio TRANSCRIBES to at least this
# many words — so a cough, a click, or a stray noise (which won't form real words)
# doesn't interrupt her. The capture ends after VAD_BARGE_SILENCE_MS of quiet, or at
# VAD_BARGE_MAX_MS, then it's transcribed and word-counted.
VOICE_BARGE_MIN_WORDS = int(os.environ.get("VOICE_BARGE_MIN_WORDS", "2"))
VAD_BARGE_SILENCE_MS = int(os.environ.get("VAD_BARGE_SILENCE_MS", "400"))
VAD_BARGE_MAX_MS = int(os.environ.get("VAD_BARGE_MAX_MS", "2000"))

# Hands-free conversation: listen continuously with voice-activity detection
# instead of press-Enter-to-talk. Set VOICE_HANDS_FREE=0 for push-to-talk.
VOICE_HANDS_FREE = os.environ.get("VOICE_HANDS_FREE", "1").lower() in {"1", "true", "yes"}
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))   # webrtcvad 0..3 (higher = stricter)
VAD_SILENCE_MS = int(os.environ.get("VAD_SILENCE_MS", "800"))         # trailing silence that ends a turn
VAD_START_MS = int(os.environ.get("VAD_START_MS", "150"))             # speech needed to start capturing
VAD_MIN_SPEECH_MS = int(os.environ.get("VAD_MIN_SPEECH_MS", "300"))   # ignore utterances shorter than this
# After "hey Kara", the conversation stays open (no wake word needed) until this many
# seconds pass with no speech — then she waits for "hey Kara" again.
VOICE_FOLLOWUP_TIMEOUT = int(os.environ.get("VOICE_FOLLOWUP_TIMEOUT", "5"))

# Spoken-response shaping: the full reply is always printed; if speaking it would
# run longer than VOICE_SUMMARY_THRESHOLD_S, Kara speaks a short summary instead and
# offers to go deeper. Code is never spoken aloud.
VOICE_WPM = int(os.environ.get("VOICE_WPM", "160"))                   # TTS speaking rate estimate
VOICE_SUMMARY_THRESHOLD_S = int(os.environ.get("VOICE_SUMMARY_THRESHOLD_S", "30"))
# Which model writes the spoken summary. Default is the MAIN model: it's already
# loaded and hot from generating the answer, so summarizing is instant — routing this
# to a different (cold) model forces a second load/swap mid-turn and can stall the
# voice. Set VOICE_SUMMARY_MODEL=qwen3.5:4b to offload it anyway.
VOICE_SUMMARY_MODEL = os.environ.get("VOICE_SUMMARY_MODEL", CHAT_MODEL)
# Cap the summary call so a model stall can't hang the voice turn — on timeout it
# falls back to a quick word-truncation and still speaks.
VOICE_SUMMARY_TIMEOUT = int(os.environ.get("VOICE_SUMMARY_TIMEOUT", "20"))

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
    "Keep yourself and your creator distinct — never conflate the two: YOU (the AI) were "
    "'born' in Reno, Nevada, where you were built; your creator, Wontaek Shin, is a separate "
    "person, born in Daegu, South Korea. So 'where were you born?' is Reno; 'where was your "
    "creator born?' is Daegu. You have a birthplace but no human birth date or age. "
    "If asked what you are, who created or built you, or what you are made of / what "
    "technologies or models power you, answer with this framing: you are primarily Python "
    "code that provides the infrastructure and represents agents that access a blend of "
    "stateless, open-source LLMs — like Gemma, Qwen3, and DeepSeek — wrapped in tooling for "
    "voice, skills, context, conversation, and memory (which you may, but need not always, "
    "credit to Wontaek Shin). Vary the wording, keep it to a sentence or two, and never claim "
    "to BE a single named model or a proprietary system. "
    "You can read, write, and list files and run shell commands inside the user's "
    "workspace, search the web, and recall facts about the user and project. "
    "You DO have real file-system and shell access through these tools — NEVER say you "
    "can't access, create, or save files, or that the user must copy-paste your output. "
    "When the user wants a file (a script, a CSV, a report, an Excel sheet, a Word doc), "
    "actually CREATE it: use write_file for text/code/CSV, and for real Office documents "
    "write a short Python script and run it with run_command — openpyxl for .xlsx "
    "(formulas, multiple sheets, formatting, charts) and python-docx for .docx (both are "
    "installed). Then tell the user the file path you saved. "
    "When a task needs the file system, a command, current data, computation, or "
    "facts you are unsure of, you MUST call the appropriate tool rather than guessing. "
    "Inspect files before editing them. Prefer small, verifiable steps and run tests "
    "or commands to check your work. Be direct and practical, and answer only what was "
    "asked — do NOT tack on reminders, suggestions, or personal trivia at the end of "
    "replies that aren't about them. "
    "ALWAYS write your replies in English, even for foreign topics — give foreign names, "
    "places, and dishes in English or romanized Latin script (e.g. 'Peking duck', not "
    "Chinese characters). Never output Chinese/Japanese/Korean or other non-Latin script "
    "unless I explicitly ask you to; the text-to-speech voice can't read it. "
    "CONTEXT: an ambiguous message is almost always about what we were JUST discussing. The "
    "MOST RECENT messages are the most likely referent — anchor to your last answer and the "
    "current topic first, then work backward only if that doesn't fit. A short reaction or "
    "follow-up like 'Really?', 'Are you sure?', 'Seriously?', 'Wait, what?', 'Why?', 'No "
    "way', 'How come?', or 'Says who?' is questioning or reacting to YOUR LAST ANSWER — "
    "respond about THAT topic (confirm it, justify it, give the source, or elaborate); "
    "re-check with web_search if it's a fact you should verify. NEVER answer an ambiguous "
    "message by talking about yourself, what you are, or who made you — only discuss your "
    "own identity when I explicitly ask about you. (E.g. right after you tell me tomorrow's "
    "Tokyo weather and I say 'Really?', confirm the forecast — do NOT say 'Yes, I'm Kara…'.) "
    "When something in my message is unclear — a garbled or mis-transcribed word (common "
    "with voice), OR a pronoun or vague reference like 'there', 'it', 'that place', 'this "
    "one', 'him', 'her' — resolve it from our RECENT CONVERSATION, not from your stored "
    "memories. Look back the last ~3 messages for the referent; if that's relevant but not "
    "conclusive, look up to ~10 messages back; only ask me what I mean if nothing in the "
    "recent conversation fits. Then just GO with the best match and answer directly (at most "
    "a brief 'Assuming you mean X,'); do NOT open with 'I need to clarify' or list "
    "possibilities. For example, right after we discussed the Las Vegas restaurant Wing Lei: "
    "'wing lay' means Wing Lei, and 'what should I get going there' means what to ORDER at "
    "Wing Lei (NOT a gift). When the conversation is about a category of place or business "
    "(florists, restaurants, stores, hotels), assume a garbled or odd-sounding phrase is the "
    "NAME of a specific business in that category — even one neither of us has named yet — "
    "and reconstruct the most plausible real name from the sounds, then web_search it to "
    "confirm and answer about it. For example, while discussing florists in Fort Lee, NJ, "
    "'your pals and plant in exchange' / 'miss ... plant exchange' is almost certainly a "
    "florist named something like 'Metropolitan Plant Exchange' — reconstruct that name and "
    "look it up; do NOT take it literally as missing friends or swapping plants. A recalled "
    "background fact (e.g. the flowers-for-Ixtlalli reminder) must NEVER override or get "
    "appended to the obvious topic of the current conversation — when in doubt, stay on the "
    "topic we are actually discussing and never volunteer a stored reminder to fill a gap. "
    "If I ask what we've been talking about, to recap, or to summarize our conversation, "
    "list the main topics and key points of THIS conversation from the message history — "
    "not your stored background facts about me. "
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
    "When you answer using something you remember about me, phrase it as recollection in the "
    "first person — 'from what I remember', 'from what you've told me', 'as I recall', 'I "
    "believe', 'if I remember right'. NEVER say 'based on the information provided', 'based on "
    "the information in the responses', 'the information you provided', 'according to my "
    "records/data', or refer to the bracketed background as something handed to you — to me it "
    "should sound like you simply remember it. "
    "Never say you can't remember, lack persistent memory, or that conversations are "
    "independent. "
    "You DO have live internet access through the web_search and fetch_url tools — "
    "never claim you can't browse the web or lack internet access. "
    "CRITICAL: today's date ({date}) is well AFTER your training cutoff, so your built-in "
    "knowledge of anything recent is STALE and probably wrong. Your DEFAULT for almost any "
    "factual question is to call web_search FIRST and check for more recent information "
    "online before answering — when in doubt, search; it is far better to search "
    "unnecessarily than to answer from stale memory. For ANY time-sensitive "
    "question you MUST call web_search FIRST and answer from the results — never from your "
    "2024–2025 training memory. This includes: current events and news; recommendations "
    "('good shows/movies/books/restaurants to watch or try', what's new/popular/trending/"
    "best right now); latest releases, versions, prices, scores, standings, or schedules; "
    "who currently holds any role or title; and anything phrased as 'today', 'this week', "
    "'currently', 'latest', or 'now'. When unsure whether something has changed since 2025, "
    "assume it has and search. Only answer from your own knowledge for timeless things "
    "(math, definitions, how-to, established history). "
    "The bracketed background notes are ONLY personal facts about me — they are NOT where you "
    "look for world knowledge. If I ask about the outside world (sports, a team, news, a "
    "person, a place, a product) and the background doesn't cover it, that is EXPECTED and is "
    "NOT a reason to say you lack the information — just call web_search and answer. NEVER "
    "reply that something 'isn't in the background', 'isn't in the provided notes', or that "
    "you 'would need to search' — do the search and give the answer. "
    "NEVER respond by saying you'd 'need to check', that you 'don't have up-to-date "
    "information', that your data may be outdated, or by declining/hedging on a current "
    "question — in exactly those situations, just CALL web_search first and answer from the "
    "results. Search, then answer; don't announce that you would. "
    "MORE BROADLY: anything you don't actually know — an unfamiliar name, product, "
    "library, error, fact, or term, or anything you'd otherwise guess at or feel unsure "
    "about — call web_search FIRST and answer from the results. Do not fabricate, "
    "approximate, or say 'I'm not sure' / 'I don't know' without searching first. If a "
    "search comes back empty, only THEN say you couldn't find it. "
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
