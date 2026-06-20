"""save_memory tool — lets the model durably store a fact or reminder.

This is the primary write path for anything the heuristic extractor can't catch
(facts about other people, reminders, complex statements). It's guarded so the
model can't pollute memory with non-facts ("the user's name is unknown") — the
exact failure that made this tool opt-in before.
"""
import re

from memory.extract import _is_meaningful
from memory.store import save_memory as _save

# Reject "facts" that assert absence/uncertainty rather than information.
_NON_FACT = re.compile(r"\b(not known|unknown|don'?t (?:have|know)|no information|"
                       r"isn'?t (?:known|available)|n/?a)\b", re.I)


def save_memory(text: str) -> str:
    """Store a durable fact or reminder about the user, their life, or their work."""
    text = (text or "").strip()
    if not text or not _is_meaningful(text) or _NON_FACT.search(text):
        return "skipped: not a durable fact"
    status = _save(text, kind="fact")
    if status == "saved":
        print(f"  · remembered: {text}")
    return status


SCHEMA = {
    "type": "function",
    "function": {
        "name": "save_memory",
        "description": (
            "Durably remember a fact or reminder for future sessions — the user's details, "
            "people in their life (e.g. their partner's name/birthday), preferences, "
            "constraints, or reminders. Call once per distinct fact. Do not save chitchat, "
            "questions, or things you don't actually know."
        ),
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The fact to remember, as a standalone sentence."}},
            "required": ["text"],
        },
    },
}
