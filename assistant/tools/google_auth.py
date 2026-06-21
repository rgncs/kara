"""Shared Google OAuth for the Calendar and Gmail tools.

One desktop-OAuth flow, one token.json, covering whatever Google scopes are
enabled (config.GOOGLE_SCOPES). Run this module directly to (re-)authorize:

    python assistant/tools/google_auth.py

Adding a scope (e.g. turning Gmail on) invalidates the old token, so the
credentials are rebuilt automatically when the saved token doesn't cover every
requested scope. The google-* libraries are imported lazily so the rest of Kara
runs without them.
"""
import logging
import os
import sys

# Allow running directly for one-time OAuth setup: put assistant/ on the path so
# `config` resolves the same way it does under main.py.
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

log = logging.getLogger("assistant.google")

_services: dict = {}  # cache built API clients by (api, version)


def _save(creds) -> None:
    with open(config.GOOGLE_TOKEN_PATH, "w", encoding="utf-8") as f:
        f.write(creds.to_json())


def _granted_scopes() -> set:
    """Scopes actually granted by the saved token (read from the file, NOT from a
    Credentials object — which mirrors the *requested* scopes and would lie)."""
    import json
    try:
        with open(config.GOOGLE_TOKEN_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f).get("scopes", []) or [])
    except (OSError, ValueError):
        return set()


def _credentials():
    """Return authorized OAuth credentials covering config.GOOGLE_SCOPES."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    scopes = config.GOOGLE_SCOPES
    creds = None
    # Only reuse the saved token if it actually GRANTS every scope we now need
    # (e.g. don't try to refresh a calendar-only token up to Gmail scopes — Google
    # rejects that as invalid_scope; we must re-consent instead).
    if os.path.exists(config.GOOGLE_TOKEN_PATH) and set(scopes).issubset(_granted_scopes()):
        creds = Credentials.from_authorized_user_file(config.GOOGLE_TOKEN_PATH, scopes)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save(creds)
        return creds
    # (Re-)consent needed: run the desktop flow for the full scope set.
    if not os.path.exists(config.GOOGLE_CREDENTIALS_PATH):
        raise FileNotFoundError(
            f"missing {config.GOOGLE_CREDENTIALS_PATH} — download an OAuth "
            "'desktop app' credentials.json from the Google Cloud console first")
    log.debug("re-consenting for scopes: %s", scopes)
    flow = InstalledAppFlow.from_client_secrets_file(config.GOOGLE_CREDENTIALS_PATH, scopes)
    creds = flow.run_local_server(port=0)
    _save(creds)
    return creds


def service(api: str, version: str):
    """Return a cached, authenticated Google API client (e.g. service('gmail', 'v1'))."""
    key = (api, version)
    if key not in _services:
        from googleapiclient.discovery import build
        _services[key] = build(api, version, credentials=_credentials(), cache_discovery=False)
    return _services[key]


if __name__ == "__main__":
    print(f"Authorizing Google access for scopes:\n  " + "\n  ".join(config.GOOGLE_SCOPES))
    print("A browser window will open…")
    _credentials()
    print(f"Done — token saved to {config.GOOGLE_TOKEN_PATH}")
