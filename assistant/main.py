"""CLI entry loop — reads user input, runs the agent turn, keeps history.

The assistant reply is appended back to `messages` after each turn — that
round-trip is the only thing that creates conversational memory.
"""
import datetime
import logging
import os
import random
import re
import sys

import approval
import config
import history
from agent import agent_turn
from health import preflight
from memory import store
from memory.extract import extract_facts, has_remember_cue, remembered_content
from memory.store import recall, save_memory

log = logging.getLogger("assistant.main")

# --- memory orchestration: ask / tell / confirm / retract / restore -----------
# A deferred offer awaiting the user's "yes": ("save", fact) or ("restore", text).
_pending = None
_last_saved: list[str] = []  # facts saved on the previous turn (for "that was a joke")

_MEM_Q = re.compile(r"(?i)\b(?:do you (?:remember|know|recall)|what do you (?:remember|know)|"
                    r"have i (?:told|mentioned)|did i (?:tell|mention))\b")
# Memory-retraction triggers. Deliberately does NOT match bare "delete/remove that X"
# (those are usually code/file requests); memory deletion needs "forget" or an explicit
# "... the memory/note/fact" / "... from memory".
_FORGET = re.compile(r"(?i)(?:\bforget(?:ting)?\b"
                     r"|\bthat(?:'?s| was| is)?\s*(?:a |just a )?(?:joke|not (?:real|true)|fake|made up|a lie)\b"
                     r"|\bi (?:was )?(?:just )?(?:joking|kidding)\b|\bi made (?:that|it) up\b"
                     r"|\bnever ?mind\b|\bscratch that\b|\bdisregard (?:that|it)\b"
                     r"|\b(?:delete|remove|erase) (?:that|the|this|my) ?(?:memory|note|fact)\b"
                     r"|\b(?:delete|remove|erase)\b[\w\s]{0,30}\bfrom (?:your |my )?memory\b)")
# Permanent (no-undo) deletion — "permanently delete", "delete forever", etc.
_PERM = re.compile(r"(?i)\b(?:permanent(?:ly)?|forever|for good|completely)\b[\w\s]{0,20}?"
                   r"\b(?:delete|remove|erase|wipe|forget)\b"
                   r"|\b(?:delete|remove|erase|wipe)\b[\w\s]{0,20}?\b(?:permanent(?:ly)?|forever|for good)\b")
# Words stripped to find the deletion target ("permanently delete my dog from your
# memory" -> "dog"); whatever remains is matched semantically against stored memories.
_DEL_STRIP = re.compile(r"(?i)\b(?:please|can you|could you|permanently|forever|for good|completely|"
                        r"delete|remove|erase|wipe|forget(?:ting)?|disregard|scratch|"
                        r"now|from|your|my|you|me|i|to|in|"
                        r"that|this|the|it|was|is|a|an|about|of|fact|note|memory|"
                        r"joke|joking|lie|kidding|fake|made|up|real|true|just|never|mind)\b")
_AFFIRM = ("yes", "yeah", "yep", "sure", "ok", "okay", "please", "yes please",
           "go ahead", "do it", "save it", "remember it", "please do", "of course",
           "restore it", "bring it back")
# Bare retractions with no target → drop the last thing saved (vs "delete <X>").
# Allows leading filler ("actually …", "oh wait …").
_BARE_RETRACT = re.compile(r"(?i)^\W*(?:(?:actually|ok|okay|wait|no|um|hmm|oh|well|sorry)\b[\s,]*)*"
                           r"(?:that(?:'?s| was| is)?\s*(?:a |just a )?"
                           r"(?:joke|not real|not true|fake|made up|a lie)|"
                           r"i (?:was )?(?:just )?(?:joking|kidding)|i made (?:that|it) up|"
                           r"never ?mind|scratch that|forget (?:it|that)|"
                           r"disregard (?:that|it)|delete the last)\W*$")


# Appended to save/restore notes so Kara confirms only the one fact, not her whole memory.
_CONFIRM_BRIEF = "Confirm only that in one short sentence; do NOT list or summarize anything else you remember."


# A follow-up that points back at the current conversation ("there", "that place").
# On these, skip memory recall so an unrelated stored fact can't hijack the topic —
# the conversation history (which the model sees) resolves the reference.
_FOLLOWUP_REF = re.compile(r"(?i)\b(over there|in there|down there|up there|back there|"
                           r"go(?:ing)? there|get(?:ting)? there|head(?:ing)? there|"
                           r"eat(?:ing)? there|order(?:ing)? there|dine? there|"
                           r"that place|this place|that one|this one|that spot|the place)\b")


def _is_followup_reference(text: str) -> bool:
    return bool(_FOLLOWUP_REF.search(text))


# "what have we been talking about?" — recap the current conversation, not memory.
_RECAP = re.compile(
    r"(?i)(what (?:have|did|were) we (?:been )?(?:talk(?:ing|ed)?|discuss(?:ing|ed)?|"
    r"chat(?:ting|ted)?|cover(?:ing|ed)?)(?: about)?"
    r"|(?:recap|summar(?:y|ize|ise)|sum up|go over|catch me up on) (?:our|this|the) "
    r"(?:conversation|chat|discussion|talk|session)"
    r"|what (?:was|were|is) (?:our|this|the) (?:conversation|chat|discussion|talk) about"
    r"|remind me what we (?:talked|were talking|chatted|discussed))")


def _is_conversation_recap(text: str) -> bool:
    return bool(_RECAP.search(text))


# A request/question, not a personal statement — skip casual fact extraction on these so
# content like "write a letter that says I love him" isn't saved as a fake preference.
_REQUEST = re.compile(r"(?i)^\s*(?:can|could|would|will|please|are you|could you|would you|"
                      r"will you|write|create|make|draft|compose|generate|build|give me|show me|"
                      r"help me|tell me|find|search|look up|do you|how|what|who|whom|when|where|"
                      r"why|which|whose|is|are|do|does|did|should)\b|\?\s*$")


def _is_request(text: str) -> bool:
    return bool(_REQUEST.search(text))


def _is_memory_question(text: str) -> bool:
    return bool(_MEM_Q.search(text))


def _is_forget(text: str) -> bool:
    return bool(_FORGET.search(text) or _PERM.search(text))


def _is_permanent_delete(text: str) -> bool:
    return bool(_PERM.search(text))


def _is_bare_retract(text: str) -> bool:
    return bool(_BARE_RETRACT.search(text))


def _is_affirm(text: str) -> bool:
    """A short, clear affirmation — NOT any sentence starting with 'sure'/'go ahead',
    so an ordinary request can't accidentally confirm a stale memory offer."""
    t = text.strip().lower().rstrip(".!?")
    if t in _AFFIRM:
        return True
    words = t.split()
    if not words or len(words) > 5:
        return False
    return (words[0] in ("yes", "yeah", "yep", "sure", "ok", "okay")
            or t.startswith(("go ahead", "please do", "do it", "save it",
                             "remember it", "restore")))


# Scope of a "remember": global (cross-project) vs local (this launch directory).
_GLOBAL_CUE = re.compile(r"(?i)\b(forever|globally?|everywhere|all projects?|cross.project|"
                         r"permanently|always)\b")
_LOCAL_CUE = re.compile(r"(?i)\b(locally|for (?:this|the current) (?:project|directory|repo|folder|context)|"
                        r"in (?:this|the current) (?:project|directory|repo|folder)|"
                        r"(?:this|current) (?:project|directory|context)|just (?:here|this)|project memory)\b")


def _memory_scope(text: str):
    """Explicit scope from a 'remember' phrase, or None (ambiguous → ask the user)."""
    if _LOCAL_CUE.search(text):
        return "local"
    if _GLOBAL_CUE.search(text):
        return "global"
    return None


def _scope_answer(text: str):
    """Interpret the answer to 'forever or just this project?' — global / local / None."""
    if re.search(r"(?i)\b(forever|globally?|everywhere|all projects?|permanent\w*|always|cross.project)\b", text):
        return "global"
    if re.search(r"(?i)\b(locally?|this (?:project|one|context|directory|repo|folder)|just this|"
                 r"for this|current|here)\b", text):
        return "local"
    return None


def _scope_label(scope: str) -> str:
    return "this project" if scope == "local" else "everywhere"


# Trailing scope qualifier to trim from a stored fact ("... just for this project").
_SCOPE_TAIL = re.compile(
    r"(?i)[\s,]*(?:just |only )?(?:for|in|across) (?:this|the current|all) "
    r"(?:projects?|director(?:y|ies)|repos?|folders?|context)\s*\.?$"
    r"|[\s,]*(?:forever|globally|everywhere|permanently|in all projects)\s*\.?$")


def _strip_scope_tail(text: str) -> str:
    return _SCOPE_TAIL.sub("", text).strip(" ,.")


def _save_facts(facts: list[str], scope: str = "global") -> None:
    global _last_saved
    saved = []
    for fact in facts:
        if save_memory(fact, scope=scope) == "saved":
            print(f"  · remembered ({_scope_label(scope)}): {fact}")
            saved.append(fact)
    if saved:
        _last_saved = saved


def _delete_match(text: str, permanent: bool) -> None:
    if store.delete_texts([text], hard=permanent):
        print(f"  · {'permanently deleted' if permanent else 'moved to recently deleted'}: {text}")


def _plan_forget(user_input: str, mems: list[dict]) -> str:
    """Handle a retraction. A bare retraction ('that was a joke') drops the last save
    immediately; a targeted 'forget X' finds the closest memory and asks to CONFIRM
    before deleting (because loose targets can match the wrong memory). Returns a note."""
    global _pending, _last_saved
    _pending = None
    permanent = _is_permanent_delete(user_input)
    if _is_bare_retract(user_input):
        if _last_saved and store.delete_texts(_last_saved, hard=permanent):
            d, _last_saved[:] = _last_saved[0], []
            mems[:] = [m for m in mems if m.get("text") != d]  # don't show it as still-known
            verb = "permanently deleted" if permanent else "moved to recently deleted"
            print(f"  · {verb}: {d}")
            return ('[You just DELETED that note from memory. Tell the user briefly you\'ve '
                    'removed/forgotten it. Do NOT say you saved or restored it.]')
        return "[There was nothing recent to remove. Acknowledge briefly.]"
    target = re.sub(r"\s+", " ", _DEL_STRIP.sub(" ", user_input)).strip(" ,.!?")
    hits = store.recall(target, k=1) if target else []
    if hits:
        _pending = ("delete", hits[0]["text"], permanent)
        verb = "permanently delete" if permanent else "forget"
        return (f"[If the user means to {verb} a stored MEMORY (not delete code or files), the "
                f'closest memory is: "{hits[0]["text"]}". Confirm they mean THAT memory before '
                "removing it. If they mean code or files, just do that task and ignore this.]")
    return ("[No matching stored memory was found. If they meant a memory, say you don't have "
            "one about that; if they meant code or files, just do the task.]")


def _handle_memory(user_input: str, mems: list[dict]) -> str:
    """Process this turn's memory intent BEFORE the reply (so what Kara says is true),
    and return a note to fold into the user turn telling her what just happened / to ask."""
    global _pending, _last_saved

    # Answer to a pending "forever or this project?" scope question.
    if _pending and _pending[0] == "scope":
        scope = _scope_answer(user_input)
        if scope:
            fact = _pending[1]
            _pending = None
            _save_facts([fact], scope=scope)
            return f'[You just saved that to {_scope_label(scope)} memory. {_CONFIRM_BRIEF}]'
        _pending = None
        if _is_forget(user_input) or _is_bare_retract(user_input):
            # "never mind" / "forget it" cancels the question — save nothing, delete nothing.
            return "[The user cancelled the request to remember that. Acknowledge briefly; nothing was saved.]"
        # otherwise fall through and treat as a fresh turn

    # Confirm a pending offer ("yes") — save / restore / delete.
    if _pending and _is_affirm(user_input):
        p = _pending
        _pending = None
        kind = p[0]
        if kind == "restore" and store.restore(p[1]):
            _last_saved = [p[1]]
            print(f"  · restored: {p[1]}")
            return f'[You just restored this note: "{p[1]}". {_CONFIRM_BRIEF}]'
        if kind == "save":
            _save_facts([p[1]])  # default global
            return f'[You just saved this: "{p[1]}". {_CONFIRM_BRIEF}]'
        if kind == "delete":
            _delete_match(p[1], p[2])
            mems[:] = [m for m in mems if m.get("text") != p[1]]  # don't show it as still-known
            verb = "permanently deleted" if p[2] else "removed"
            return (f'[You just {verb} that memory. Tell the user briefly it\'s {verb}. '
                    'Do NOT say you saved or restored it.]')
        return ""

    # Asking whether she remembers something (checked before forget so "do you remember
    # to delete X?" is answered, not executed).
    if _is_memory_question(user_input):
        if not mems:
            try:
                hits = store.recall_deleted(user_input)
            except Exception:  # noqa: BLE001
                hits = []
            if hits:
                _pending = ("restore", hits[0]["text"])
                return ("[This is NOT in active memory, but the user previously deleted a related "
                        f'note: "{hits[0]["text"]}". Tell them you don\'t have it actively but they '
                        "deleted it, and offer to restore it — do not state it as a current fact.]")
        cand = remembered_content(user_input)
        _pending = ("save", cand[0]) if cand else None
        return ""

    # Retraction / deletion (bare → immediate; targeted → confirm first).
    if _is_forget(user_input):
        return _plan_forget(user_input, mems)

    # Explicit "remember X" command.
    if has_remember_cue(user_input):
        facts = remembered_content(user_input)
        _pending = None
        if not facts:
            return ""
        scope = _memory_scope(user_input)
        if scope is None:  # ambiguous → ask which scope (ONE short question), save nothing yet
            _pending = ("scope", facts[0])
            return ('[Ask ONE short scope question and save nothing yet — exactly like: '
                    '"Got it — forever, or just for this project?" Keep it that short; '
                    "don't explain scopes or claim it's saved.]")
        _save_facts([_strip_scope_tail(f) for f in facts], scope=scope)
        return f"[You just saved that to {_scope_label(scope)} memory. {_CONFIRM_BRIEF}]"

    # Casual implicit self-fact → save globally, silently — but NOT from a request or
    # question (so "write a letter that says I love him" isn't saved as a preference).
    _pending = None
    if not _is_request(user_input):
        _save_facts(extract_facts(user_input), scope="global")
    return ""


def _memory_preface(mems: list[dict]) -> str:
    """Render recalled memories as a context block prepended to the user's turn.

    Folded into the user message (not a separate system message) because local
    models reliably attend to the user turn but often ignore a secondary system
    block.
    """
    lines = []
    for m in mems:
        try:
            date = datetime.date.fromtimestamp(m["ts"]).isoformat()
            lines.append(f"- ({date}) {m['text']}")
        except Exception:  # noqa: BLE001
            lines.append(f"- {m['text']}")
    return (
        "[Background you already know about me (treat as true; prefer the most recent date "
        "if any conflict). Use an item ONLY if it directly answers my CURRENT message. Do "
        "NOT list or recite these back, and do NOT tack reminders, to-dos, suggestions, or "
        "personal details onto a reply about something else — answer only what I asked and "
        "stay on topic:\n" + "\n".join(lines) + "]"
    )


# Phrases that mean the model deflected to its training instead of searching.
_DEFLECT = re.compile(
    r"(?i)(knowledge cutoff|training (?:data|cutoff)|as of my last|"
    r"don'?t have (?:access to |the )?(?:the )?(?:most )?(?:up.?to.?date|current|real.?time|latest)|"
    r"can'?t provide the most current|may be outdated|might be out ?of ?date|"
    r"check (?:their|the)? ?(?:current|latest|recent) (?:website|reviews?|menu|info|information|prices?)|"
    r"i'?d (?:suggest|recommend) (?:checking|looking)|"
    r"(?:i'?m |i am )?not familiar with|never heard of|"
    r"(?:i'?m |i am )?not aware of (?:any|a|an|the)|"
    r"don'?t have (?:any )?information (?:on|about)|"
    r"(?:un)?able to find any information)")


def _is_deflection(text: str) -> bool:
    return bool(_DEFLECT.search(text or ""))


def _used_web_search(messages: list[dict], since: int) -> bool:
    """True if a web_search tool call appears in the messages added this turn."""
    for m in messages[since:]:
        for tc in (m.get("tool_calls") or []):
            if (tc.get("function") or {}).get("name") == "web_search":
                return True
    return False


def _force_search_answer(messages: list[dict], user_input: str, printer: "_Printer | None") -> "str | None":
    """The model deflected without searching — search now and re-answer from results."""
    from tools.search import web_search
    try:
        results = web_search(user_input)
    except Exception as e:  # noqa: BLE001
        log.debug("forced web_search failed: %s", e)
        return None
    if not results or results.startswith("ERROR"):
        return None
    if messages and messages[-1].get("role") == "assistant":
        messages.pop()                                  # drop the deflecting reply
    messages.append({"role": "user", "content":
        "Answer my previous question using ONLY these current web results. Cite them "
        "[1], [2]. Do NOT search again, mention any knowledge cutoff, or tell me to check "
        "elsewhere:\n" + results})
    print("\n  ↻ that was dated — checking the web…")
    p2 = _Printer(printer.label) if printer else None
    try:
        reply = agent_turn(messages, on_token=p2.write if p2 else None)
        if p2:
            if not p2.started and reply:
                p2.write(reply)
            p2.finish()
        return reply
    except Exception as e:  # noqa: BLE001
        log.debug("re-answer failed: %s", e)
        return None


def _approve_command(command: str) -> tuple[bool, str]:
    """Interactive approval prompt for run_command (installed only in 'prompt' mode)."""
    prefix = approval.command_prefix(command)
    print(f"\n  ⚠  the agent wants to run a shell command:\n      {command}")
    opts = ["[y] yes, once"]
    if prefix:
        opts.append(f"[p] yes, and don't ask again for '{prefix}' commands")
    opts += ["[a] yes, all commands this session", "[n] no"]
    try:
        ans = input("  " + "   ".join(opts) + "\n  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False, "no input — declined"
    if ans in {"a", "all", "always"}:
        approval.approve_session()
        return True, "approved (all commands this session)"
    if prefix and ans in {"p", "prefix"}:
        approval.allow_prefix(prefix)
        return True, f"approved (all '{prefix}' commands this session)"
    if ans in {"y", "yes"}:
        return True, "approved by user"
    return False, "declined by user"


def _approve_command_voice(command: str) -> tuple[bool, str]:
    """Spoken approval for run_command during a hands-free voice session: Kara reads
    the request and you answer out loud (yes / no / always)."""
    import voice
    print(f"\n  ⚠  the agent wants to run a shell command:\n      {command}")
    spoken = command if len(command) <= 80 else "a shell command, shown on your screen"
    voice.speak_interruptible(f"I'd like to run {spoken}. Should I? Say yes, no, or always.")
    for _ in range(2):
        print("  🎤 (say yes / no / always)…", end="\r", flush=True)
        try:
            ans = (voice.listen_vad(start_timeout=15) or "").lower()
        except (KeyboardInterrupt, EOFError):
            return False, "cancelled"
        print(" " * 34, end="\r", flush=True)
        if not ans:
            voice.speak_interruptible("I didn't catch that.")
            continue
        if re.search(r"\b(always|every ?time|all (?:commands|of them)|go ahead with everything)\b", ans):
            approval.approve_session()
            return True, "approved (all commands this session)"
        if re.search(r"\b(yes|yeah|yep|yup|sure|ok|okay|go ahead|do it|please do|approve|sounds good)\b", ans):
            return True, "approved by voice"
        if re.search(r"\b(no|nope|nah|don'?t|do not|deny|stop|cancel|skip)\b", ans):
            return False, "declined by voice"
        voice.speak_interruptible("Was that a yes or a no?")
    return False, "no clear answer — declined"


def _setup_logging() -> None:
    level = logging.DEBUG if os.environ.get("ASSISTANT_DEBUG") else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


def _system_message() -> dict:
    today = datetime.date.today().isoformat()
    return {"role": "system",
            "content": config.SYSTEM_PROMPT.format(name=config.AGENT_NAME, date=today)}


class _Printer:
    """Streams the assistant's reply to the terminal, printing the label prefix
    lazily on the first token (so tool-resolution steps stay silent)."""

    def __init__(self, label: str):
        self.label = label
        self.started = False

    def write(self, text: str) -> None:
        if not self.started:
            print(f"{self.label} ▸ ", end="", flush=True)
            self.started = True
        print(text, end="", flush=True)

    def finish(self) -> None:
        if self.started:
            print()


def process_turn(messages: list[dict], user_input: str, printer: "_Printer | None" = None) -> str | None:
    """Run one full turn: recall memories, run the agent (streaming the final
    answer to `printer` if given), persist new facts.

    Shared by the text and voice loops. Returns the assistant reply, or None if
    the turn errored (already reported). Mutates `messages` (history) in place.
    """
    base_len = len(messages)  # index of this turn's user message / rollback point

    # Recall memories (both scopes) for context — but NOT on a follow-up that refers to
    # the current conversation ("what's good there?") or a recap request ("what have we
    # been talking about?"), so a stray memory can't hijack or pollute the answer.
    if _is_followup_reference(user_input) or _is_conversation_recap(user_input):
        mems = []
    else:
        try:
            mems = recall(user_input)
        except Exception as e:  # noqa: BLE001 — memory must never break chatting
            log.debug("recall failed: %s", e)
            mems = []

    # Process this turn's memory intent BEFORE replying (save/delete/restore/ask), so
    # whatever Kara says about memory is accurate. Returns a note to inject for her.
    try:
        note = _handle_memory(user_input, mems)
    except Exception as e:  # noqa: BLE001 — memory must never break chatting
        log.debug("memory handling failed: %s", e)
        note = ""

    # Fold context into the user turn (transiently — restored to clean after).
    preface = _memory_preface(mems) if mems else ""
    if note:
        preface += ("\n" if preface else "") + note
    messages.append({"role": "user", "content": user_input})
    if preface:
        messages[base_len]["content"] = preface + "\n\n" + user_input

    try:
        reply = agent_turn(messages, on_token=printer.write if printer else None)
    except Exception as e:  # noqa: BLE001 — keep the loop alive on transient errors
        if printer:
            printer.finish()
        print(f"[error during turn: {e}]")
        del messages[base_len:]  # roll back the whole turn so history stays consistent
        return None

    if preface:
        messages[base_len]["content"] = user_input  # restore the clean user message
    history.trim(messages)

    if printer:
        if not printer.started and reply:
            printer.write(reply)  # nothing streamed (e.g. MAX_STEPS fallback) — show it
        printer.finish()

    # Tripwire: if she deflected to her training ("knowledge cutoff", "check current
    # reviews") without searching, search now and re-answer from current results.
    if reply and _is_deflection(reply) and not _used_web_search(messages, base_len):
        corrected = _force_search_answer(messages, user_input, printer)
        if corrected:
            reply = corrected

    return reply


def _text_loop(messages: list[dict], label: str) -> None:
    while True:
        try:
            user_input = input("you ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("bye.")
            break
        process_turn(messages, user_input, printer=_Printer(label))


def _matches(text: str, options) -> bool:
    t = text.strip().lower().rstrip(".!?")
    return t in options or t.startswith(options)


def _voice_summary(full_reply: str) -> str:
    """Rewrite a long reply into a ~30-second spoken summary (no code/markdown).
    Falls back to a word-truncation if the model call fails."""
    from llm import chat
    instr = ("Summarize the assistant reply below for the SPOKEN word in about 75 "
             "words (~30 seconds), and never more than that. Start straight with the "
             "content — no preamble like 'here's a summary'. Write in ENGLISH only "
             "(romanize any foreign names/dishes; no Chinese/Japanese/Korean characters). "
             "Conversational plain sentences, NO code, NO markdown, NO lists — just the key takeaways.")
    try:
        resp = chat([
            {"role": "system", "content": "You rewrite text to be spoken aloud by a TTS voice."},
            {"role": "user", "content": instr + "\n\n---\n" + full_reply},
        ], temperature=0)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        log.debug("voice summary failed: %s", e)
        words = full_reply.split()
        return " ".join(words[:75]) + ("…" if len(words) > 75 else "")


def _get_voice_input(voice, hands_free: bool) -> str:
    """Capture one utterance — hands-free (VAD) or push-to-talk."""
    if hands_free:
        print("🎤 listening…", end="\r", flush=True)
        text = voice.listen_vad()
        print(" " * 20, end="\r", flush=True)
        return text
    cmd = input("🎤 [Enter]=talk, 't'=type : ").strip().lower()
    if cmd == "t":
        return input("you ▸ ").strip()
    return voice.listen()


# Quick spoken acknowledgements so the user hears something the instant she stops
# talking, instead of dead air while Kara thinks / searches the web.
_THINKING_FILLERS = (
    "Got it, let me think about that.",
    "Hold on a sec while I look into that.",
    "Good question — let me see.",
    "Let me look that up real quick.",
    "One sec, let me check on that.",
    "Sure, give me a moment.",
    "Okay, let me dig into that.",
)


def _voice_loop(messages: list[dict], label: str) -> None:
    import voice
    hands_free = config.VOICE_HANDS_FREE
    timeout = config.VOICE_FOLLOWUP_TIMEOUT
    if hands_free:
        print("Hands-free voice mode — say \"hey Kara\" to start, then just keep talking.")
        print(f"After ~{timeout}s of silence she waits for \"hey Kara\" again. "
              "Tap a key to interrupt; \"hey Kara, goodbye\" or Ctrl-C to quit.\n")
    else:
        print("Push-to-talk: Enter to start/stop talking ('t' to type). Ctrl-C to quit.\n")

    active = False  # in an ongoing hands-free conversation (no wake word needed)
    while True:
        try:
            if not hands_free:
                user_input = _get_voice_input(voice, hands_free)
            elif active:
                print("🎤 …", end="\r", flush=True)
                user_input = voice.listen_vad(start_timeout=timeout)
                print(" " * 24, end="\r", flush=True)
                if user_input is None:        # silence → re-arm the wake word
                    active = False
                    print('  (paused — say "hey Kara" to continue)')
                    continue
            else:
                print('🎤 say "hey Kara"…', end="\r", flush=True)
                user_input = voice.listen_vad()
                print(" " * 24, end="\r", flush=True)
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not user_input:
            continue

        # Wake word (hands-free): required to start; optional once the conversation is active.
        if hands_free:
            cmd = voice.strip_wake_word(user_input)
            if cmd is not None:               # addressed with "hey Kara"
                active = True
                if not cmd:                   # just "hey Kara" with nothing after
                    voice.speak_interruptible("Yes?")
                    continue
                user_input = cmd
            elif not active:                  # no wake word and not in a conversation → ignore
                log.debug("ignored (no wake word): %r", user_input)
                continue
            # else: active conversation, no wake word needed — use as-is

        print(f"you ▸ {user_input}")
        if _matches(user_input, ("exit", "quit", "stop", "goodbye", "goodbye kara", "bye")):
            print("bye.")
            break

        # Immediate acknowledgement so she doesn't sit in silence while thinking/searching.
        voice.speak_interruptible(random.choice(_THINKING_FILLERS))

        reply = process_turn(messages, user_input, printer=_Printer(label))
        if not reply:
            continue

        # Hard 30-second cap on every spoken response.
        if voice.estimate_seconds(reply) <= config.VOICE_SUMMARY_THRESHOLD_S:
            spoken = reply
        else:
            print("  (long answer — full text above; speaking a ~30s summary)")
            spoken = _voice_summary(reply)
        interrupted = voice.speak_interruptible(spoken)

        if interrupted and messages and messages[-1].get("role") == "assistant":
            # Tell the model it was cut off so it adapts to what the user says next.
            messages[-1]["content"] = (messages[-1].get("content") or "") + \
                " … (interrupted by the user before finishing)"
            print("  ⏹ interrupted — go ahead")


def main() -> None:
    _setup_logging()
    use_voice = "--voice" in sys.argv or os.environ.get("VOICE", "").lower() in {"1", "true", "yes"}

    preflight([config.CHAT_MODEL, config.EMBED_MODEL])  # memory needs the embed model too
    try:
        store.purge_old_deleted()  # auto-empty trash older than TRASH_TTL_DAYS
    except Exception as e:  # noqa: BLE001
        log.debug("trash purge failed: %s", e)

    if config.COMMAND_APPROVAL == "prompt":
        approval.set_approver(_approve_command_voice if use_voice else _approve_command)

    name = config.AGENT_NAME
    label = name.lower()  # prompt label, e.g. "kara ▸"

    messages: list[dict] = [_system_message()]
    print(f"{name} — local coding agent (model: {config.CHAT_MODEL})")
    print(f"Workspace: {config.WORKSPACE_ROOT}")
    print(f"Shell approval: {config.COMMAND_APPROVAL}")
    mode = "voice" if use_voice else "text"
    print(f"Mode: {mode}. Set ASSISTANT_DEBUG=1 to see tool calls.\n")

    if use_voice:
        _voice_loop(messages, label)
    else:
        _text_loop(messages, label)


if __name__ == "__main__":
    main()
