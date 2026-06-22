"""Shared Google OAuth for the Calendar and Gmail tools — multi-account.

Each connected Google account (config.GOOGLE_ACCOUNTS, e.g. work + personal) has its
own token file: the FIRST label keeps the original token.json (no re-auth), the rest
use token_<label>.json beside it. Authorize one at a time, signing in with that
account in the browser:

    python assistant/tools/google_auth.py <account>     # e.g. personal

Run with no argument to (re-)authorize the primary account. Adding a scope re-consents
automatically (the saved token's granted scopes are checked). The google-* libraries
are imported lazily so the rest of Kara runs without them.
"""
import logging
import os
import re
import sys

# Allow running directly for one-time OAuth setup.
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

log = logging.getLogger("assistant.google")

_services: dict = {}  # (api, version, account) -> client


def primary_account() -> str:
    return config.GOOGLE_ACCOUNTS[0] if config.GOOGLE_ACCOUNTS else "default"


def _token_path(account: str) -> str:
    """Token file for an account: the first label keeps token.json (back-compat)."""
    if account == primary_account():
        return config.GOOGLE_TOKEN_PATH
    return os.path.join(os.path.dirname(config.GOOGLE_TOKEN_PATH), f"token_{account}.json")


def available_accounts() -> list:
    """Accounts with a saved token — discovered from disk (token.json for the primary,
    token_<label>.json for the rest), so a newly-connected label appears even if it
    isn't in GOOGLE_ACCOUNTS. Primary first."""
    accts = []
    if os.path.exists(config.GOOGLE_TOKEN_PATH):
        accts.append(primary_account())
    try:
        for fn in sorted(os.listdir(os.path.dirname(config.GOOGLE_TOKEN_PATH))):
            m = re.match(r"token_(.+)\.json$", fn)
            if m and m.group(1) != primary_account():
                accts.append(m.group(1))
    except OSError:
        pass
    return accts


def authorize(account: str):
    """Run the OAuth flow for `account` (opens a browser if it isn't already connected)."""
    return _credentials(account)


def account_email(account: str) -> "str | None":
    """The email address of a connected account (via Gmail getProfile), or None."""
    try:
        return service("gmail", "v1", account).users().getProfile(
            userId="me").execute().get("emailAddress")
    except Exception as e:  # noqa: BLE001
        log.debug("account_email(%s) failed: %s", account, e)
        return None


def accounts_for(account: "str | None") -> list:
    """Resolve a request's account arg: None/'all' -> every authorized account; a
    specific label -> just that one (if authorized, else empty)."""
    if account and account.lower() != "all":
        return [account] if os.path.exists(_token_path(account)) else []
    return available_accounts()


def _granted_scopes(path: str) -> set:
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f).get("scopes", []) or [])
    except (OSError, ValueError):
        return set()


def _save(creds, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(creds.to_json())


def _credentials(account: str):
    """Authorized OAuth credentials for `account`, covering config.GOOGLE_SCOPES."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    scopes = config.GOOGLE_SCOPES
    path = _token_path(account)
    creds = None
    # Reuse the saved token only if it grants every scope we need (else re-consent —
    # Google rejects refreshing a token up to broader scopes as invalid_scope).
    if os.path.exists(path) and set(scopes).issubset(_granted_scopes(path)):
        creds = Credentials.from_authorized_user_file(path, scopes)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save(creds, path)
        return creds
    if not os.path.exists(config.GOOGLE_CREDENTIALS_PATH):
        raise FileNotFoundError(
            f"missing {config.GOOGLE_CREDENTIALS_PATH} — download an OAuth "
            "'desktop app' credentials.json from the Google Cloud console first")
    log.debug("consenting account %s for scopes: %s", account, scopes)
    flow = InstalledAppFlow.from_client_secrets_file(config.GOOGLE_CREDENTIALS_PATH, scopes)
    creds = flow.run_local_server(port=0)
    _save(creds, path)
    return creds


def service(api: str, version: str, account: "str | None" = None):
    """Authenticated Google API client for an account (defaults to the primary)."""
    account = account or primary_account()
    key = (api, version, account)
    if key not in _services:
        from googleapiclient.discovery import build
        _services[key] = build(api, version, credentials=_credentials(account), cache_discovery=False)
    return _services[key]


if __name__ == "__main__":
    acct = sys.argv[1] if len(sys.argv) > 1 else primary_account()
    print(f"Authorizing Google account '{acct}' for scopes:\n  " + "\n  ".join(config.GOOGLE_SCOPES))
    print("Sign in with the CORRECT Google account when the browser opens…")
    _credentials(acct)
    print(f"Done — token saved to {_token_path(acct)}")
