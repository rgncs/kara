"""Conversation history management.

The model is stateless — it "remembers" only because we resend the full message
list every call. As a conversation grows it will eventually exceed the (smaller,
local) context window, so this module is the seam where trimming/summarization
lives. Phase 1 ships a safe no-op-ish trim; later phases can replace the body
with summarization without touching the rest of the program.
"""
import logging

import config

log = logging.getLogger("assistant.history")


def trim(messages: list[dict]) -> list[dict]:
    """Keep history within budget, preserving the system prompt at index 0.

    Phase 1: a simple sliding window over the most recent turns. The seam exists
    so a later phase can summarize dropped turns instead of discarding them.
    Mutates and returns `messages`.
    """
    if len(messages) <= config.HISTORY_MAX_MESSAGES:
        return messages

    system = messages[0:1] if messages and messages[0].get("role") == "system" else []
    body = messages[len(system):]
    keep = config.HISTORY_MAX_MESSAGES - len(system)
    if len(body) <= keep:
        messages[:] = system + body
        return messages

    window = body[-keep:]
    # Critical: never start the kept window on a `tool` message or an
    # `assistant` that carries tool_calls — that would orphan a tool result (or
    # leave a tool_calls with no following result), and the API rejects the next
    # call. Advance to the first clean turn boundary (a `user` message). If the
    # window contains no user message (pathological), don't trim rather than
    # produce an invalid history.
    start = next((i for i, m in enumerate(window) if m.get("role") == "user"), None)
    if start is None:
        log.debug("history.trim: no clean boundary in window — leaving history intact")
        return messages
    log.debug("history.trim dropping %d oldest messages (TODO: summarize)",
              len(body) - len(window[start:]))
    messages[:] = system + window[start:]
    return messages
