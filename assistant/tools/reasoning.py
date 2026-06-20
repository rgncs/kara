"""The `think` tool: delegate a hard problem to a dedicated reasoning model.

The main controller (qwen3-coder) is great at tool-calling and code, but for
genuinely hard step-by-step reasoning — tricky logic, algorithm design, planning
a multi-file change — it can delegate to a reasoning model (config.REASONING_MODEL,
e.g. deepseek-r1). That model is loaded on demand by Ollama and emits a
<think>…</think> trace; we strip it and return only the distilled plan/answer, so
the controller gets a clean result it can act on.

Failures degrade gracefully to an "ERROR: …" string (e.g. the reasoning model
isn't pulled), never an exception.
"""
import re

import config
from llm import chat

# DeepSeek-R1 and similar emit their chain-of-thought wrapped in <think>…</think>;
# strip it so only the conclusion reaches the controller (and the user).
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    text = _THINK.sub("", text or "")
    # An unterminated trace (truncated output) — drop everything up to a lone </think>.
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def think(problem: str) -> str:
    """Delegate a hard reasoning/planning problem to the reasoning model."""
    try:
        resp = chat(
            [
                {"role": "system", "content":
                    "You are a careful reasoning model. Think step by step, then give a "
                    "clear, concise plan or answer. End with the conclusion only."},
                {"role": "user", "content": problem},
            ],
            model=config.REASONING_MODEL,
        )
        return _strip_think(resp.choices[0].message.content) or "(reasoning model returned nothing)"
    except Exception as e:  # noqa: BLE001 — surface as a tool error, never crash the loop
        return f"ERROR: reasoning model '{config.REASONING_MODEL}' unavailable: {e}"


SCHEMA = {
    "type": "function",
    "function": {
        "name": "think",
        "description": "Delegate a HARD problem to a dedicated reasoning model for careful "
                       "step-by-step analysis: complex multi-step logic, tricky debugging, "
                       "algorithm or data-structure design, math, or planning a multi-file "
                       "change. Do NOT use it for simple questions, lookups, or things you "
                       "can answer directly — it is slower. Returns a distilled plan/answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description":
                            "The full problem statement, including all context the reasoner "
                            "needs (it does not see our conversation)."},
            },
            "required": ["problem"],
        },
    },
}
