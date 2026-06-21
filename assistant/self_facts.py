"""Kara's own identity facts, settable by its creator (birthday, birthplace, …).

These OVERRIDE the built-in canned identity answers — so when the user teaches Kara a
fact about itself ("you were born on June 20th, 2026"), the age/birthplace handlers
report that instead of the default. Persisted to config.SELF_FACTS_PATH (gitignored,
user-customized state) — kept separate from the user-fact memory store, which is for
facts about the USER, not about Kara.
"""
import json
import logging
import os

import config

log = logging.getLogger("assistant.self_facts")


def load() -> dict:
    try:
        with open(config.SELF_FACTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}


def get(key: str, default=None):
    return load().get(key, default)


def set_fact(key: str, value: str) -> dict:
    """Set one self-fact, atomically. Returns the updated dict."""
    facts = load()
    facts[key] = value
    tmp = config.SELF_FACTS_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(facts, f)
        os.replace(tmp, config.SELF_FACTS_PATH)
    except OSError as e:  # noqa: BLE001
        log.debug("could not write self-facts: %s", e)
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return facts
