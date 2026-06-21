"""Spam-candidate log + the periodic scan that fills it.

The scan is read-only: it flags senders with many unread emails and records them
to config.SPAM_LOG_PATH. NOTHING is deleted here — the user reviews and confirms
each sender through the spam-cleanup flow. The in-app scanner (started from
main()) runs every config.SPAM_SCAN_INTERVAL seconds (default 6 hours) while Kara
is open.
"""
import json
import logging
import os
import sys
import threading
import time

# Allow running directly for an on-the-spot scan: `python assistant/spam.py`.
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

log = logging.getLogger("assistant.spam")


def record_candidates(candidates: list) -> None:
    """Persist the latest scan result (timestamped) to the log file."""
    try:
        with open(config.SPAM_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "candidates": candidates}, f)
    except OSError as e:
        log.debug("could not write spam log: %s", e)


def _load() -> dict:
    try:
        with open(config.SPAM_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load_keep() -> set:
    """Return the set of kept (never-spam) senders/domains, lowercased."""
    try:
        with open(config.SPAM_KEEP_PATH, "r", encoding="utf-8") as f:
            return {s.strip().lower() for s in json.load(f) if s.strip()}
    except (OSError, ValueError):
        return set()


def add_keep(senders) -> set:
    """Add one or more senders/domains to the keep-list. Returns the new set."""
    if isinstance(senders, str):
        senders = [senders]
    keep = load_keep()
    keep.update(s.strip().lower() for s in senders if s and s.strip())
    try:
        with open(config.SPAM_KEEP_PATH, "w", encoding="utf-8") as f:
            json.dump(sorted(keep), f)
    except OSError as e:
        log.debug("could not write keep list: %s", e)
    return keep


def matches(sender: str, entries: set) -> bool:
    """True if `sender` matches a list entry — by exact address, exact domain, or a
    parent domain (so a 'regenics.com' entry also covers 'send.regenics.com')."""
    if not entries:
        return False
    s = (sender or "").lower()
    if s in entries:
        return True
    dom = s.split("@")[-1]
    return any(dom == e or dom.endswith("." + e) for e in entries)


def is_kept(sender: str, keep: set = None) -> bool:
    """True if `sender` (or its domain) is on the keep-list."""
    return matches(sender, keep if keep is not None else load_keep())


def load_autodelete() -> set:
    """Senders/domains the user confirmed as junk — auto-trashed each scan."""
    try:
        with open(config.SPAM_AUTODELETE_PATH, "r", encoding="utf-8") as f:
            return {s.strip().lower() for s in json.load(f) if s.strip()}
    except (OSError, ValueError):
        return set()


def add_autodelete(senders) -> set:
    """Add sender(s)/domain(s) to the auto-delete list. Returns the new set."""
    if isinstance(senders, str):
        senders = [senders]
    block = load_autodelete()
    block.update(s.strip().lower() for s in senders if s and s.strip())
    try:
        with open(config.SPAM_AUTODELETE_PATH, "w", encoding="utf-8") as f:
            json.dump(sorted(block), f)
    except OSError as e:
        log.debug("could not write auto-delete list: %s", e)
    return block


def is_autodelete(sender: str, block: set = None) -> bool:
    return matches(sender, block if block is not None else load_autodelete())


def load_candidates() -> list:
    """Most recent scan's candidates, with kept and auto-delete senders filtered out
    (so a stale log never resurfaces a sender you've since chosen to keep or auto-delete)."""
    keep, block = load_keep(), load_autodelete()
    return [c for c in (_load().get("candidates", []) or [])
            if not is_kept(c.get("sender", ""), keep)
            and not is_autodelete(c.get("sender", ""), block)]


def last_scan_age() -> "float | None":
    """Seconds since the last recorded scan, or None if there's never been one."""
    ts = _load().get("ts")
    return (time.time() - ts) if ts else None


# --- progress announcements (set by main() so updates print and/or speak) -----
_announcer = None


def set_announcer(fn) -> None:
    """Register how batch-progress updates reach the user (print and/or speak)."""
    global _announcer
    _announcer = fn


def announce(msg: str) -> None:
    log.info(msg)
    try:
        (_announcer or (lambda m: print(f"  📬 {m}")))(msg)
    except Exception as e:  # noqa: BLE001 — a progress update must never break the scan
        log.debug("announce failed: %s", e)


# --- resumable scan checkpoint (so a huge scan survives interruption) ----------
def save_scan_state(state: dict) -> None:
    try:
        with open(config.SPAM_SCAN_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except OSError as e:
        log.debug("could not write scan state: %s", e)


def load_scan_state() -> dict:
    try:
        with open(config.SPAM_SCAN_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def clear_scan_state() -> None:
    try:
        os.unlink(config.SPAM_SCAN_STATE_PATH)
    except OSError:
        pass


def run_scan(max_scan: int = None) -> list:
    """Scan now (live Gmail): auto-trash unread from confirmed auto-delete senders,
    then record the remaining candidates for review. Skips kept + auto-delete senders.
    Pass max_scan=0 to scan the ENTIRE unread history (slower). Returns the candidates."""
    from tools import gmail_tool
    block = load_autodelete()
    auto_n = 0
    if block:
        try:
            auto_n, _ = gmail_tool.auto_trash_blocked(block)
        except Exception as e:  # noqa: BLE001
            log.debug("auto-trash failed: %s", e)
    cands = gmail_tool.find_spam_candidates(max_scan=max_scan, exclude=load_keep() | block)
    record_candidates(cands)
    log.info("spam scan: auto-trashed %d, %d sender(s) to review", auto_n, len(cands))
    return cands


def start_background_scanner() -> None:
    """Start the hourly scanner in a daemon thread (read-only; logs only)."""
    if not (config.GMAIL_ENABLED and config.SPAM_SCAN_ENABLED):
        return

    def _loop():
        # Skip the immediate scan if a recent one already exists (avoids hammering
        # Gmail when Kara is relaunched often).
        age = last_scan_age()
        if age is not None and age < config.SPAM_SCAN_INTERVAL:
            time.sleep(config.SPAM_SCAN_INTERVAL - age)
        while True:
            try:
                run_scan()
            except Exception as e:  # noqa: BLE001 — never let the scanner crash Kara
                log.debug("background spam scan failed: %s", e)
            time.sleep(config.SPAM_SCAN_INTERVAL)

    threading.Thread(target=_loop, name="spam-scanner", daemon=True).start()
    log.debug("spam scanner started (every %ds)", config.SPAM_SCAN_INTERVAL)


if __name__ == "__main__":
    # On-the-spot scan: report senders with many unread and record them.
    # Pass --full to scan the ENTIRE unread history (no sample cap; slower).
    full = "--full" in sys.argv
    where = "ALL unread mail" if full else f"the first {config.SPAM_SCAN_MAX} unread"
    print(f"Scanning {where} for senders with more than "
          f"{config.SPAM_UNREAD_THRESHOLD} unread…")
    cands = run_scan(max_scan=0 if full else None)
    if not cands:
        print("No spam candidates found.")
    else:
        print(f"\n{len(cands)} sender(s) over the threshold:")
        for c in cands:
            unsub = "  (can unsubscribe)" if c.get("unsubscribe") else ""
            print(f"  - {c['sender']}: {c['count']} unread{unsub}")
        print('\nRun "spam cleanup" in Kara to review and delete/unsubscribe.')
