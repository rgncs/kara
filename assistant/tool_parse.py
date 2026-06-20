"""Fallback parser for models that emit tool calls as TEXT instead of populating
the API's structured `tool_calls` field.

qwen3-coder (and other Qwen/Hermes-style models) frequently leak their tool-call
DSL into message content when served through Ollama's OpenAI-compatible endpoint.
This module recovers those calls so the agent loop works regardless. Native
`tool_calls` are always preferred; this is only consulted when that field is empty.

Two formats are recognized:
  A. Qwen XML:   <function=NAME><parameter=KEY>VALUE</parameter>...</function>
  B. Hermes JSON: <tool_call>{"name": "NAME", "arguments": {...}}</tool_call>
"""
import json
import logging
import re

log = logging.getLogger("assistant.tool_parse")

_FUNC_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)
_HERMES_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _clean(value: str) -> str:
    # Models pad parameter values with framing newlines; strip those but keep
    # interior content (including indentation) intact.
    return value.strip("\n")


def extract_text_tool_calls(content: str) -> list[dict]:
    """Return [{"name": str, "args": dict}, ...] parsed from message text.

    Empty list when no recognizable tool-call markup is present (so normal prose
    is never misread as a tool call).
    """
    if not content:
        return []

    calls: list[dict] = []

    for name, body in _FUNC_RE.findall(content):
        args = {key: _clean(val) for key, val in _PARAM_RE.findall(body)}
        calls.append({"name": name.strip(), "args": args})

    for blob in _HERMES_RE.findall(content):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        if name:
            args = obj.get("arguments") or obj.get("parameters") or {}
            calls.append({"name": name, "args": args if isinstance(args, dict) else {}})

    # Surface a malformed leak (an opening marker that didn't parse) instead of
    # silently treating the model's tool intent as a normal text reply.
    if not calls and ("<function=" in content or "<tool_call>" in content):
        log.warning("tool-call markup present but unparsable — call may be lost: %.120r", content)

    return calls
