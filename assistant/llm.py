"""chat(): the ONLY function that talks to the model.

Isolating the model call behind one function is what makes the model swappable —
nothing else in the codebase touches the model API directly.
"""
import logging

from openai import OpenAI

import config

log = logging.getLogger("assistant.llm")

# api_key is required by the SDK but ignored by Ollama.
client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")


def chat(messages: list[dict], stream: bool = False, tools: list | None = None,
         temperature: float | None = None, model: str | None = None):
    """Single entry point to the model.

    Non-streaming: returns the SDK response object (caller reads
    `.choices[0].message`).
    Streaming: returns the streaming iterator of chunks (caller reads
    `chunk.choices[0].delta`).
    `temperature` overrides sampling — pass 0 for deterministic tasks like extraction.
    `model` overrides the model — defaults to CHAT_MODEL; pass FAST_MODEL or
    REASONING_MODEL to route a call to a subagent model.
    """
    model = model or config.CHAT_MODEL
    log.debug("chat() model=%s stream=%s tools=%d messages=%d temp=%s",
              model, stream, len(tools or []), len(messages), temperature)
    kwargs = {} if temperature is None else {"temperature": temperature}
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=stream,
        **kwargs,
    )
