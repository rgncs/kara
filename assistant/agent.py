"""agent_turn(): the tool loop.

Send messages + TOOLS; if the model wants tools, append the assistant message,
execute each call, append a `tool` result carrying the matching tool_call_id,
then loop again; if there are no tool calls, return the final text.

Tool calls are taken from the API's structured `tool_calls` field when present,
and otherwise recovered from message text (see tool_parse) — qwen3-coder and
similar models often emit their call DSL as plain content through Ollama. Either
way the loop appends a valid assistant tool_calls message so the API contract
(assistant.tool_calls -> tool result) is never violated.

The tool-resolution loop is non-streaming. The FINAL answer streams to the
terminal when an `on_token` callback is supplied (spec §10). Streaming is made
safe against qwen3-coder's habit of emitting tool calls as plain text by a
peek-buffer: content is withheld until we can tell it's prose, not a leaked
`<function=…>` / `<tool_call>` DSL — so a text tool-call is never shown, only
executed.
"""
import json
import logging
import uuid
from types import SimpleNamespace

import config
from llm import chat
from tool_parse import extract_text_tool_calls
from tools import TOOLS, TOOL_FUNCTIONS

log = logging.getLogger("assistant.agent")

# Safety valve: a confused model can loop on tool calls forever. Raised to accommodate
# legitimately long multi-step turns (e.g. spam cleanup across many senders, multi-account
# sweeps). Override with MAX_AGENT_STEPS.
MAX_STEPS = config.MAX_AGENT_STEPS


def _normalize_calls(msg) -> list[dict]:
    """Return a uniform [{"id", "name", "args", "error"}] from a model message.

    `args` is a parsed dict; `error` is set (and args empty) when arguments
    couldn't be parsed, so the loop can feed the error back to the model.
    """
    calls: list[dict] = []

    if msg.tool_calls:
        for tc in msg.tool_calls:
            error = None
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                args, error = {}, f"invalid JSON arguments: {e}"
            calls.append({"id": tc.id, "name": tc.function.name, "args": args, "error": error})
        return calls

    # Fallback: recover tool calls the model emitted as text. Use unique ids
    # (not call_0, call_1 reset per step) so multi-step text-emitted calls never
    # collide on tool_call_id across the conversation.
    for parsed in extract_text_tool_calls(msg.content or ""):
        calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "name": parsed["name"],
                      "args": parsed["args"], "error": None})
    if calls:
        log.debug("recovered %d tool call(s) from message text", len(calls))
    return calls


def _assistant_tool_message(calls: list[dict]) -> dict:
    """Build an assistant message that declares `calls` as structured tool_calls,
    so the following `tool` results have a valid parent regardless of source."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": c["id"],
                "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
            }
            for c in calls
        ],
    }


def _execute(name: str, args: dict) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"ERROR: unknown tool '{name}'"
    try:
        log.debug("tool call %s(%s)", name, args)
        return str(fn(**args))
    except Exception as e:  # noqa: BLE001 — report, never crash the loop
        log.debug("tool %s raised: %s", name, e)
        return f"ERROR: {e}"


_TOOL_MARKERS = ("<function=", "<tool_call>")


def _stream_collect(messages, on_token):
    """Stream one model call. Display final-answer prose live via on_token, but
    withhold content that turns out to be a leaked tool-call DSL. Returns a
    message-like object with .content and .tool_calls (or None)."""
    parts: list[str] = []
    native: dict[int, dict] = {}
    buf = ""
    live = False        # we've started showing tokens this call
    suppress = False    # content is a text tool-call — never show it

    for chunk in chat(messages, tools=TOOLS, stream=True):
        delta = chunk.choices[0].delta
        for tcd in (getattr(delta, "tool_calls", None) or []):
            slot = native.setdefault(tcd.index, {"id": None, "name": "", "args": ""})
            if tcd.id:
                slot["id"] = tcd.id
            fn = getattr(tcd, "function", None)
            if fn and fn.name:
                slot["name"] += fn.name
            if fn and fn.arguments:
                slot["args"] += fn.arguments

        piece = getattr(delta, "content", None)
        if not piece:
            continue
        parts.append(piece)
        if suppress:
            continue
        if live:
            on_token(piece)
            continue
        buf += piece
        head = buf.lstrip()
        if not head:
            continue                                   # only whitespace so far
        if head.startswith(_TOOL_MARKERS):
            suppress = True                            # leaked tool call — hide it
        elif head[0] == "<" and len(head) < 11:
            continue                                   # ambiguous "<" prefix — wait
        elif len(head) >= 24 or "\n" in head:
            on_token(buf)                              # confident it's prose
            live = True

    if not suppress and not live and buf:
        on_token(buf)                                  # short final answer — flush
        live = True

    tool_calls = None
    if native:
        tool_calls = [
            SimpleNamespace(id=s["id"] or f"call_{uuid.uuid4().hex[:8]}",
                            function=SimpleNamespace(name=s["name"], arguments=s["args"]))
            for _, s in sorted(native.items())
        ]
    return SimpleNamespace(content="".join(parts), tool_calls=tool_calls), live


def agent_turn(messages: list[dict], on_token=None) -> str:
    """Run one full turn, resolving any tool calls. Returns final assistant text.

    When on_token is given, the final answer is streamed to it token-by-token;
    otherwise the call is non-streaming (used by tests).
    """
    for _ in range(MAX_STEPS):
        if on_token is None:
            msg = chat(messages, tools=TOOLS).choices[0].message
            displayed = False
        else:
            msg, displayed = _stream_collect(messages, on_token)
        calls = _normalize_calls(msg)

        if not calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or ""

        # If a "thinking out loud" preamble was streamed before this tool step,
        # break the line so it doesn't run into what comes after.
        if displayed and on_token:
            on_token("\n")
        messages.append(_assistant_tool_message(calls))
        for c in calls:
            result = c["error"] and f"ERROR: {c['error']}" or _execute(c["name"], c["args"])
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})

    log.warning("agent_turn hit MAX_STEPS=%d without a final answer", MAX_STEPS)
    fallback = "[stopped: too many tool steps without a final answer]"
    messages.append({"role": "assistant", "content": fallback})
    return fallback
