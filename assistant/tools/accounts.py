"""Tools for the user to manage Kara's connected Google accounts from inside a chat:
list what's connected, and connect a new account (runs the OAuth browser flow).
"""
import logging
import os

import config

log = logging.getLogger("assistant.accounts")


def list_google_accounts() -> str:
    """List the Google (Gmail + Calendar) accounts Kara is connected to."""
    from . import google_auth
    connected = google_auth.available_accounts()
    lines = []
    for acct in connected:
        email = google_auth.account_email(acct) or "(address unavailable)"
        tag = " — primary" if acct == google_auth.primary_account() else ""
        lines.append(f"- {acct}: {email}{tag}")
    out = "Connected Google accounts:\n" + ("\n".join(lines) if lines else "  (none yet)")
    pending = [a for a in config.GOOGLE_ACCOUNTS if a not in connected]
    if pending:
        out += ("\nConfigured but not yet connected: " + ", ".join(pending)
                + " — ask me to connect one and I'll open the browser.")
    return out


def connect_google_account(account: str) -> str:
    """Connect a Google account by running the OAuth flow (opens a browser). Call this
    only once the user is ready, since it opens a browser they must complete."""
    from . import google_auth
    account = (account or "").strip().lower()
    if not account:
        return "Tell me a short label for the account (e.g. 'personal' or 'work')."
    if not os.path.exists(config.GOOGLE_CREDENTIALS_PATH):
        return (
            "First-time Google setup is needed. This credentials.json is created ONCE and "
            "then connects ANY account — personal OR business — so you don't repeat it per "
            "account:\n"
            "1. In the Google Cloud console, create a project.\n"
            "2. Enable the Gmail API and the Google Calendar API.\n"
            "3. OAuth consent screen → 'External' (use 'Internal' only if you want to "
            "restrict it to your own Workspace org) → publish to production.\n"
            "4. Create an OAuth client ID of type 'Desktop app' and download the JSON.\n"
            f"5. Save it as credentials.json in the Kara folder ({config.GOOGLE_CREDENTIALS_PATH}).\n"
            "Then ask me to connect the account. Note the account-type difference at sign-in:\n"
            "- Personal Gmail: connects directly (click through the one-time 'unverified app' "
            "screen).\n"
            "- Business/Workspace Gmail: your IT admin may block third-party apps — if it "
            "won't connect, ask them to allow it, or have a Workspace admin create the OAuth "
            "app as 'Internal'.")
    if account in google_auth.available_accounts():
        email = google_auth.account_email(account) or ""
        return f"'{account}'{f' ({email})' if email else ''} is already connected."
    try:
        google_auth.authorize(account)  # opens the browser; blocks until the user finishes
    except Exception as e:  # noqa: BLE001
        return f"Couldn't connect '{account}': {e}"
    email = google_auth.account_email(account) or ""
    return (f"Connected '{account}'{f' ({email})' if email else ''} — it's now included in "
            "your email and calendar.")


LIST_ACCOUNTS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_google_accounts",
        "description": "List which Google (Gmail + Calendar) accounts Kara is connected to, "
                       "with their email addresses. Use when the user asks what accounts "
                       "you're connected to.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

CONNECT_ACCOUNT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "connect_google_account",
        "description": "Connect a new Google account by opening the OAuth browser flow. The "
                       "user signs in with that account and approves. Call this ONLY after "
                       "confirming the user is ready (it opens a browser they must complete). "
                       "If first-time setup is missing it returns the steps to relay.",
        "parameters": {
            "type": "object",
            "properties": {"account": {"type": "string", "description": "A short label for the account, e.g. 'personal' or 'work'."}},
            "required": ["account"],
        },
    },
}
