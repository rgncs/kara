"""Gmail tools: list_messages, read_message, send_message, trash_message, unsubscribe.

Auth is the shared Google OAuth (see google_auth.py). Reading/searching is free;
sending, trashing, and unsubscribing are outward-facing and each go through
approval.confirm_action(always_ask=True) — every one is confirmed, with no
"yes to all" bypass. Deletes go to Trash (recoverable), never permanent.

The google-* libraries are imported lazily; every function returns a readable
"ERROR: ..." string instead of raising, matching the other tools.
"""
import base64
import logging
from email.message import EmailMessage
from email.utils import parseaddr

import approval
import config

log = logging.getLogger("assistant.gmail")


def _service(account: "str | None" = None):
    from . import google_auth
    return google_auth.service("gmail", "v1", account)


def _confirm(prompt: str) -> bool:
    """Gate an outward Gmail action — always asks (no auto / no 'yes to all')."""
    return not config.GMAIL_CONFIRM_WRITES or approval.confirm_action(prompt, always_ask=True)


def _header(payload: dict, name: str) -> str:
    for h in payload.get("headers", []):
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _plain_text(payload: dict) -> str:
    """Extract a readable text body from a message payload (prefers text/plain
    anywhere in the tree, then falls back to tag-stripped text/html)."""
    def find(part, want):
        body = part.get("body", {})
        data = body.get("data")
        if part.get("mimeType", "") == want and data:
            return base64.urlsafe_b64decode(data).decode("utf-8", "replace")
        for sub in part.get("parts", []) or []:
            got = find(sub, want)
            if got:
                return got
        return ""
    text = find(payload, "text/plain")
    if text:
        return text.strip()
    html = find(payload, "text/html")
    if html:
        import re
        return re.sub(r"<[^>]+>", " ", html).strip()
    return ""


def list_messages(query: str = None, max_results: int = 10, unread_only: bool = False,
                  account: str = None) -> str:
    """List/search inbox messages across connected accounts (or one, if `account` is
    given). `query` uses Gmail search syntax. Ids are tagged with their account."""
    from . import google_auth
    accts = google_auth.accounts_for(account)
    if not accts:
        return ("No Google account is connected for that — authorize one with "
                "`python assistant/tools/google_auth.py <account>`.")
    q = (query or "").strip()
    if unread_only:
        q = (q + " is:unread").strip()
    multi = len(accts) > 1
    lines = []
    for acct in accts:
        try:
            svc = _service(acct)
            res = svc.users().messages().list(
                userId="me", q=q or None, maxResults=max(1, min(int(max_results), 25))).execute()
            for m in res.get("messages", []):
                full = svc.users().messages().get(
                    userId="me", id=m["id"], format="metadata",
                    metadataHeaders=["From", "Subject", "Date"]).execute()
                p = full.get("payload", {})
                frm = parseaddr(_header(p, "From"))[1] or _header(p, "From")
                subj = _header(p, "Subject") or "(no subject)"
                unread = "•" if "UNREAD" in full.get("labelIds", []) else " "
                snippet = (full.get("snippet", "") or "")[:80]
                tag = f"[{acct}] " if multi else ""
                lines.append(f"{unread} {tag}{frm} — {subj}  [{snippet}…]  [id: {acct}:{m['id']}]")
        except Exception as e:  # noqa: BLE001 — one account failing shouldn't sink the rest
            log.debug("gmail list failed for %s: %s", acct, e)
    return "\n".join(lines) if lines else "No messages matched."


def _split_id(message_id: str):
    """Accept an 'account:id' tag (from list_messages) -> (account, id); plain id -> (None, id)."""
    if ":" in message_id:
        acct, _, mid = message_id.partition(":")
        return acct, mid
    return None, message_id


def read_message(message_id: str, account: str = None) -> str:
    """Return the full text of a message (headers + body). message_id may be 'account:id'."""
    acct, message_id = _split_id(message_id)
    try:
        svc = _service(account or acct)
        msg = svc.users().messages().get(userId="me", id=message_id, format="full").execute()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    p = msg.get("payload", {})
    head = (f"From: {_header(p, 'From')}\nTo: {_header(p, 'To')}\n"
            f"Date: {_header(p, 'Date')}\nSubject: {_header(p, 'Subject')}\n\n")
    body = _plain_text(p) or msg.get("snippet", "")
    out = head + body
    return out[:config.MAX_TOOL_OUTPUT_CHARS]


def send_message(to: str, subject: str, body: str, cc: str = None, account: str = None) -> str:
    """Send an email FROM a connected account (default: primary). Confirms before sending."""
    from . import google_auth
    acct = account or google_auth.primary_account()
    try:
        svc = _service(acct)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    if not _confirm(f"Send email from your {acct} account to {to} — subject '{subject}'?"):
        return "DENIED: email not sent (you declined)."
    msg = EmailMessage()
    msg["To"] = to
    if cc:
        msg["Cc"] = cc
    msg["Subject"] = subject
    msg.set_content(body)
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    try:
        sent = svc.users().messages().send(userId="me", body={"raw": raw}).execute()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    return f"Sent from {acct} to {to} — message id {sent.get('id')}"


def trash_message(message_id: str, account: str = None) -> str:
    """Move a message to Trash (recoverable). Confirms first. message_id may be 'account:id'."""
    acct, message_id = _split_id(message_id)
    try:
        svc = _service(account or acct)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    # Identify it for the confirmation so you know what you're trashing.
    try:
        meta = svc.users().messages().get(
            userId="me", id=message_id, format="metadata",
            metadataHeaders=["From", "Subject"]).execute()
        p = meta.get("payload", {})
        desc = f"'{_header(p, 'Subject') or '(no subject)'}' from {_header(p, 'From')}"
    except Exception:  # noqa: BLE001
        desc = f"id {message_id}"
    if not _confirm(f"Move email {desc} to Trash?"):
        return "DENIED: email not trashed (you declined)."
    try:
        svc.users().messages().trash(userId="me", id=message_id).execute()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    return f"Moved to Trash: {desc}"


def unsubscribe(message_id: str, account: str = None) -> str:
    """Unsubscribe from the sender of a message via its List-Unsubscribe header.

    Sends the unsubscribe email (mailto:) or calls the one-click/HTTP link.
    Confirms before acting. message_id may be 'account:id'.
    """
    acct, message_id = _split_id(message_id)
    try:
        svc = _service(account or acct)
        msg = svc.users().messages().get(
            userId="me", id=message_id, format="metadata",
            metadataHeaders=["List-Unsubscribe", "List-Unsubscribe-Post", "From", "Subject"]).execute()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    p = msg.get("payload", {})
    raw = _header(p, "List-Unsubscribe")
    sender = _header(p, "From")
    if not raw:
        return (f"No unsubscribe link found in the message from {sender}. You may need to "
                "open it and unsubscribe manually.")
    # The header holds one or more <…> targets: mailto: and/or http(s):.
    import re
    targets = re.findall(r"<([^>]+)>", raw) or [raw]
    mailtos = [t for t in targets if t.lower().startswith("mailto:")]
    https = [t for t in targets if t.lower().startswith("http")]
    one_click = "one-click" in _header(p, "List-Unsubscribe-Post").lower()

    if not _confirm(f"Unsubscribe from {sender}?"):
        return "DENIED: did not unsubscribe (you declined)."

    # Prefer the HTTP one-click (RFC 8058), then a plain HTTPS GET, then mailto.
    if https:
        url = https[0]
        try:
            import httpx
            if one_click:
                r = httpx.post(url, data={"List-Unsubscribe": "One-Click"},
                               timeout=15, follow_redirects=True)
            else:
                r = httpx.get(url, timeout=15, follow_redirects=True)
            return (f"Sent unsubscribe request to {sender} (HTTP {r.status_code}). It may take "
                    "a little while to take effect.")
        except Exception as e:  # noqa: BLE001
            if not mailtos:
                return f"ERROR: unsubscribe link failed: {e}"
    if mailtos:
        addr = mailtos[0][len("mailto:"):].split("?")[0]
        subj = "unsubscribe"
        m = re.search(r"[?&]subject=([^&]+)", mailtos[0])
        if m:
            from urllib.parse import unquote
            subj = unquote(m.group(1))
        out = EmailMessage()
        out["To"] = addr
        out["Subject"] = subj
        out.set_content("Please unsubscribe me from this list.")
        raw_b = base64.urlsafe_b64encode(out.as_bytes()).decode()
        try:
            svc.users().messages().send(userId="me", body={"raw": raw_b}).execute()
        except Exception as e:  # noqa: BLE001
            return f"ERROR: unsubscribe email failed: {e}"
        return f"Sent an unsubscribe email to {addr} on behalf of unsubscribing from {sender}."
    return f"Couldn't act on the unsubscribe target for {sender}: {raw}"


def _count_unread(svc, query: str, cap: int = 5000) -> int:
    """Exact count of messages matching `query` (paginated)."""
    total, req = 0, svc.users().messages().list(userId="me", q=query, maxResults=500)
    while req is not None and total < cap:
        resp = req.execute()
        total += len(resp.get("messages", []))
        req = svc.users().messages().list_next(req, resp)
    return total


def find_spam_candidates(threshold: int = None, max_scan: int = None, exclude=None) -> list:
    """Scan unread mail, group by sender, and return senders with MORE THAN
    `threshold` unread messages — likely spam/newsletters. Read-only; deletes
    nothing. `max_scan` caps how many unread are sampled to DISCOVER senders;
    pass 0 (or negative) to scan the entire unread history. The reported `count`
    is the sender's TRUE unread total (a targeted query), not just the sample.
    Each item: {sender, count, unsubscribe, ids}.
    """
    from collections import defaultdict
    if threshold is None:
        threshold = config.SPAM_UNREAD_THRESHOLD
    if max_scan is None:
        max_scan = config.SPAM_SCAN_MAX
    unlimited = max_scan <= 0
    keep = {s.lower() for s in (exclude or [])}   # addresses/domains to skip
    svc = _service()  # raises on missing creds — callers handle
    ids, req = [], svc.users().messages().list(userId="me", q="is:unread", maxResults=500)
    while req is not None and (unlimited or len(ids) < max_scan):
        resp = req.execute()
        ids += resp.get("messages", [])
        req = svc.users().messages().list_next(req, resp)
    if not unlimited:
        ids = ids[:max_scan]
    by_sender = defaultdict(list)
    has_unsub = {}
    for m in ids:
        meta = svc.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["From", "List-Unsubscribe"]).execute()
        p = meta.get("payload", {})
        sender = parseaddr(_header(p, "From"))[1] or _header(p, "From") or "(unknown)"
        if keep and _excluded(sender, keep):
            continue                                       # kept/excluded — never flag
        by_sender[sender].append(m["id"])
        has_unsub[sender] = has_unsub.get(sender, False) or bool(_header(p, "List-Unsubscribe"))
    candidates = []
    for sender, mids in by_sender.items():
        if len(mids) <= threshold:
            continue                                       # not a candidate in the sample
        # True total for the sender (the sample may undercount), so the number shown
        # and acted on is real.
        true_count = _count_unread(svc, f"from:{sender} is:unread") if not unlimited else len(mids)
        if true_count > threshold:
            candidates.append({"sender": sender, "count": true_count,
                               "unsubscribe": has_unsub.get(sender, False), "ids": mids})
    candidates.sort(key=lambda c: -c["count"])
    return candidates


_PICK_HINT = ('\nSay e.g. "always delete from 1, 3" (auto-delete those from now on), '
              '"delete 2" (trash once), "unsubscribe 4", or "keep 5".')


def find_spam_candidates_text(threshold: int = None, full: bool = False) -> str:
    """Human/model-readable version of find_spam_candidates (for the tool registry).
    `full=True` scans the ENTIRE unread history (deeper, slower) instead of a sample.
    Keep-list and auto-delete senders are excluded; results are numbered."""
    try:
        import spam
        cands = find_spam_candidates(threshold, max_scan=0 if full else None,
                                     exclude=spam.load_keep() | spam.load_autodelete())
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    scope = "entire unread history" if full else "a sample of recent unread"
    if not cands:
        t = threshold if threshold is not None else config.SPAM_UNREAD_THRESHOLD
        return f"No senders with more than {t} unread emails (scanned {scope})."
    return (f"Spam candidates (scanned {scope}):\n" + _format_candidates(cands) + _PICK_HINT)


def _unread_ids_from(svc, sender: str, max_delete: int = 500) -> list:
    """Ids of UNREAD messages from `sender` (unopened only — never read mail)."""
    ids, req = [], svc.users().messages().list(
        userId="me", q=f"from:{sender} is:unread", maxResults=500)
    while req is not None and len(ids) < max_delete:
        resp = req.execute()
        ids += resp.get("messages", [])
        req = svc.users().messages().list_next(req, resp)
    return [m["id"] for m in ids[:max_delete]]


def _trash_ids(svc, ids: list) -> None:
    """Move ids to Trash, chunked (batchModify takes up to 1000 ids per call)."""
    for i in range(0, len(ids), 1000):
        chunk = ids[i:i + 1000]
        if chunk:
            svc.users().messages().batchModify(
                userId="me", body={"ids": chunk, "addLabelIds": ["TRASH"]}).execute()


def _excluded(sender: str, entries: set) -> bool:
    """Match by exact address, exact domain, or parent domain ('regenics.com' also
    covers 'send.regenics.com')."""
    s = sender.lower()
    if s in entries:
        return True
    dom = s.split("@")[-1]
    return any(dom == e or dom.endswith("." + e) for e in entries)


def _format_candidates(cands: list) -> str:
    """Number the candidates so the user can pick by number."""
    return "\n".join(
        f"{i}. {c['sender']} — {c['count']} unread"
        + ("  (can unsubscribe)" if c["unsubscribe"] else "")
        for i, c in enumerate(cands, 1))


def trash_from_sender(sender: str, max_delete: int = 500) -> str:
    """Move all UNREAD messages from `sender` to Trash, after ONE confirmation naming
    the sender and count. Only touches UNREAD/unopened mail; recoverable (Trash)."""
    try:
        svc = _service()
        ids = _unread_ids_from(svc, sender, max_delete)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    if not ids:
        return f"No matching messages from {sender}."
    if not _confirm(f"Move {len(ids)} email(s) from {sender} to Trash?"):
        return "DENIED: nothing trashed (you declined)."
    try:
        _trash_ids(svc, ids)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    return f"Moved {len(ids)} email(s) from {sender} to Trash."


def auto_trash_blocked(senders) -> "tuple[int, dict]":
    """Trash unread from each pre-confirmed auto-delete sender WITHOUT prompting (the
    user authorized these by adding them to the list). Returns (total, per-sender)."""
    svc = _service()  # raises on missing creds — caller handles
    total, per = 0, {}
    for s in senders:
        try:
            ids = _unread_ids_from(svc, s)
            if ids:
                _trash_ids(svc, ids)
                per[s] = len(ids)
                total += len(ids)
        except Exception as e:  # noqa: BLE001
            log.debug("auto-trash %s failed: %s", s, e)
    return total, per


def auto_delete_sender(sender: str) -> str:
    """Add a sender to the auto-delete list and clear their current unread now."""
    import spam
    spam.add_autodelete(sender)
    try:
        svc = _service()
        ids = _unread_ids_from(svc, sender)
        _trash_ids(svc, ids)
    except Exception as e:  # noqa: BLE001
        return (f"Added {sender} to the auto-delete list (couldn't clear existing now: {e}). "
                "Future unread will be auto-trashed.")
    return (f"Added {sender} to the auto-delete list and trashed {len(ids)} unread now. "
            "Future unread from them is auto-trashed by the background scan, no prompts.")


def keep_sender(sender: str) -> str:
    """Mark a sender (address or domain) as 'keep' so spam cleanup never flags it."""
    import spam
    spam.add_keep(sender)
    return f"Added {sender} to the keep-list — it won't be flagged as spam again."


KEEP_SENDER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "keep_sender",
        "description": "Mark a sender (email address or domain) as KEEP so it's excluded "
                       "from all future spam scans and cleanups. Use when the user says to "
                       "keep, whitelist, or never delete a sender.",
        "parameters": {
            "type": "object",
            "properties": {"sender": {"type": "string", "description": "Email address (team@x.com) or domain (x.com)."}},
            "required": ["sender"],
        },
    },
}

def scan_candidates_batched(threshold: int = None, exclude=None) -> list:
    """Full unread scan in resumable batches: checkpoint after each Gmail page so an
    interruption resumes where it left off, and announce after each batch. Exact counts
    (every message is counted). Used for large mailboxes by deep_spam_cleanup."""
    from collections import defaultdict
    import spam
    if threshold is None:
        threshold = config.SPAM_UNREAD_THRESHOLD
    keep = {s.lower() for s in (exclude or [])}
    svc = _service()

    st = spam.load_scan_state()
    by_sender = defaultdict(int, st.get("by_sender", {}))
    has_unsub = dict(st.get("unsub", {}))
    scanned = int(st.get("scanned", 0))
    page_token = st.get("page_token")
    total_est = st.get("total_est") or svc.users().messages().list(
        userId="me", q="is:unread", maxResults=1).execute().get("resultSizeEstimate", 0)
    if scanned:
        spam.announce(f"Resuming the scan at {scanned:,} of ~{total_est:,} emails…")

    since = 0
    first = True
    # Guard a narrow resume window: if a checkpoint has page_token=None but work was
    # already done (scanned>0), the scan had finished — don't re-scan page 1 and double
    # count; fall through to build candidates from the accumulated totals.
    while page_token or (first and not scanned):
        params = {"userId": "me", "q": "is:unread", "maxResults": 500}
        if page_token:
            params["pageToken"] = page_token
        first = False
        resp = svc.users().messages().list(**params).execute()
        for m in resp.get("messages", []):
            meta = svc.users().messages().get(
                userId="me", id=m["id"], format="metadata",
                metadataHeaders=["From", "List-Unsubscribe"]).execute()
            p = meta.get("payload", {})
            sender = parseaddr(_header(p, "From"))[1] or _header(p, "From") or "(unknown)"
            scanned += 1
            since += 1
            if _excluded(sender, keep):
                continue
            by_sender[sender] += 1
            if _header(p, "List-Unsubscribe"):
                has_unsub[sender] = True
        page_token = resp.get("nextPageToken")
        spam.save_scan_state({"by_sender": dict(by_sender), "unsub": has_unsub,
                              "scanned": scanned, "page_token": page_token, "total_est": total_est})
        if since >= config.SPAM_BATCH_SIZE or not page_token:
            spam.announce(f"Scanned {scanned:,} of ~{total_est:,} emails…")
            since = 0

    cands = [{"sender": s, "count": c, "unsubscribe": has_unsub.get(s, False), "ids": []}
             for s, c in by_sender.items() if c > threshold]
    cands.sort(key=lambda c: -c["count"])
    spam.clear_scan_state()
    return cands


def deep_spam_cleanup() -> str:
    """DEEP cleanup: across the ENTIRE history, auto-trash every unread from the
    confirmed auto-delete list, then full-scan for NEW candidates (numbered, with
    keep-list / @regenics.com excluded). Returns a report for the user to pick from."""
    import spam
    try:
        svc = _service()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    block = spam.load_autodelete()
    total, per = 0, {}
    if block:
        spam.announce("Clearing unread from your auto-delete senders…")
    for s in block:
        try:
            ids = _unread_ids_from(svc, s, max_delete=100000)  # whole history
            if ids:
                _trash_ids(svc, ids)
                per[s] = len(ids)
                total += len(ids)
        except Exception as e:  # noqa: BLE001
            log.debug("deep auto-trash %s failed: %s", s, e)
    if total:
        spam.announce(f"Auto-deleted {total:,} unread from your confirmed list.")

    # New-candidate scan: resumable batches if a scan is in progress or the mailbox is
    # large (>SPAM_BATCH_THRESHOLD unread); otherwise a single fast pass.
    exclude = spam.load_keep() | block
    try:
        resuming = bool(spam.load_scan_state())
        est = svc.users().messages().list(
            userId="me", q="is:unread", maxResults=1).execute().get("resultSizeEstimate", 0)
        if resuming or est > config.SPAM_BATCH_THRESHOLD:
            spam.announce(f"Scanning ~{est:,} unread in batches (resumes if interrupted)…")
            cands = scan_candidates_batched(exclude=exclude)
        else:
            cands = find_spam_candidates(max_scan=0, exclude=exclude)
    except Exception as e:  # noqa: BLE001
        return (f"Auto-deleted {total} unread from your safe list, but the new-candidate scan "
                f"failed (it will resume next time): {e}")
    spam.record_candidates(cands)
    head = f"Deep cleanup done. Auto-deleted {total:,} unread"
    head += (f" from your auto-delete list ({len(block)} sender(s) on it)."
             if block else " (no senders on your auto-delete list yet).")
    out = [head]
    out += [f"  - {s}: {n}" for s, n in sorted(per.items(), key=lambda x: -x[1])]
    if not cands:
        out.append("No new spam candidates to review.")
    else:
        out.append(f"\n{len(cands)} new candidate(s) to review:")
        out.append(_format_candidates(cands))
        out.append(_PICK_HINT)
    return "\n".join(out)


DEEP_CLEANUP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "deep_spam_cleanup",
        "description": "Run a DEEP email spam cleanup: scan the ENTIRE history, auto-trash "
                       "all unread from the confirmed auto-delete list, then return the NEW "
                       "spam candidates (numbered) to review. Use when the user asks for a "
                       "'deep' email/spam cleanup. After showing the result, act on the "
                       "numbers the user picks.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

AUTO_DELETE_SENDER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "auto_delete_sender",
        "description": "Mark a sender as CONFIRMED junk: trash their current unread now AND "
                       "auto-trash future unread from them on every scan with NO further "
                       "prompts. Only call this when the user clearly says to always/auto "
                       "delete from a sender (it's their standing authorization). For a "
                       "one-time delete, use trash_from_sender instead.",
        "parameters": {
            "type": "object",
            "properties": {"sender": {"type": "string", "description": "Email address (x@y.com) or domain (y.com)."}},
            "required": ["sender"],
        },
    },
}

FIND_SPAM_SCHEMA = {
    "type": "function",
    "function": {
        "name": "find_spam_candidates",
        "description": "Scan unread email and list senders with many unread messages "
                       "(likely spam/newsletters). Read-only — deletes nothing. Use this to "
                       "review before cleaning up. By default it scans a recent sample; set "
                       "full=true when the user asks to 'go deeper', scan everything, or the "
                       "whole inbox/history (slower, but catches every sender).",
        "parameters": {
            "type": "object",
            "properties": {
                "threshold": {"type": "integer", "description": "Min unread from one sender to flag (default 10)."},
                "full": {"type": "boolean", "description": "Scan the entire unread history instead of a sample (deeper, slower)."},
            },
            "required": [],
        },
    },
}

TRASH_FROM_SENDER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "trash_from_sender",
        "description": "Move all UNREAD emails from a given sender to Trash (recoverable) "
                       "after one confirmation. Only unopened mail is ever deleted — read "
                       "messages are left alone. Use for spam cleanup once the user OKs a sender.",
        "parameters": {
            "type": "object",
            "properties": {
                "sender": {"type": "string", "description": "The sender's email address."},
            },
            "required": ["sender"],
        },
    },
}

LIST_MESSAGES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_messages",
        "description": "List or search Gmail messages across the user's connected accounts. "
                       "`query` uses Gmail search syntax (e.g. 'from:kevin newer_than:7d', "
                       "'is:unread', 'subject:invoice'). Ids come back as 'account:id'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Gmail search query (optional)."},
                "max_results": {"type": "integer", "description": "Max messages per account (default 10)."},
                "unread_only": {"type": "boolean", "description": "Only unread messages."},
                "account": {"type": "string", "description": "Which account (e.g. 'work', 'personal'). Omit to search ALL connected accounts."},
            },
            "required": [],
        },
    },
}

READ_MESSAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_message",
        "description": "Read the full text of a Gmail message by id (from list_messages; pass "
                       "the full 'account:id' tag).",
        "parameters": {
            "type": "object",
            "properties": {"message_id": {"type": "string", "description": "The message id ('account:id' from list_messages)."}},
            "required": ["message_id"],
        },
    },
}

SEND_MESSAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": "Send an email from the user's Gmail. Confirms with the user before "
                       "sending. Use for new emails and replies.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Subject line."},
                "body": {"type": "string", "description": "Plain-text email body."},
                "cc": {"type": "string", "description": "CC address(es), optional."},
                "account": {"type": "string", "description": "Which account to send FROM (e.g. 'work', 'personal'). Default: primary."},
            },
            "required": ["to", "subject", "body"],
        },
    },
}

TRASH_MESSAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "trash_message",
        "description": "Move a Gmail message to Trash (recoverable). Confirms first. Get the "
                       "id from list_messages.",
        "parameters": {
            "type": "object",
            "properties": {"message_id": {"type": "string", "description": "The message id ('account:id' from list_messages)."}},
            "required": ["message_id"],
        },
    },
}

UNSUBSCRIBE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "unsubscribe",
        "description": "Unsubscribe from the sender of a Gmail message (uses its "
                       "List-Unsubscribe link). Confirms first. Get the id from list_messages.",
        "parameters": {
            "type": "object",
            "properties": {"message_id": {"type": "string", "description": "The message id to unsubscribe from."}},
            "required": ["message_id"],
        },
    },
}
