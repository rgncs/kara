"""Startup health check — run once before the CLI loop.

Fails fast with an actionable message instead of letting the first model call
throw a raw connection error.
"""
import sys

import httpx

import config


def preflight(required_models: list[str]) -> None:
    """Confirm the Ollama daemon is reachable and every required model is pulled.

    Exits the process with a clear, actionable message on failure.
    """
    base = config.OLLAMA_BASE_URL.rsplit("/v1", 1)[0]  # -> http://localhost:11434
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
        tags = resp.json()
    except Exception:
        sys.exit(f"Ollama not reachable at {base} — is `ollama serve` running?")

    # Ollama reports models as "gemma3:latest"; match on the bare name. Tolerate
    # schema skew across Ollama versions ("name" vs "model", missing keys).
    installed = {
        (m.get("name") or m.get("model") or "").split(":")[0]
        for m in tags.get("models", [])
    }
    installed.discard("")
    for model in required_models:
        if model.split(":")[0] not in installed:
            sys.exit(f"Model '{model}' not found — run `ollama pull {model}`")
