"""Google Calendar tools: list_events, create_event, delete_event.

Auth is the shared Google OAuth (see google_auth.py); authorize once with
`python assistant/tools/google_auth.py`. The google-* libraries are imported
lazily so the rest of Kara runs even when they aren't installed — every function
returns a readable "ERROR: ..." string instead of raising. Reads are free;
create/delete go through approval.confirm_action() (the same human-in-the-loop
gate as the shell) unless CALENDAR_CONFIRM_WRITES is off.
"""
import logging

import approval
import config

log = logging.getLogger("assistant.calendar")

def _service(account: "str | None" = None):
    """Return an authenticated Calendar API client for an account (default: primary)."""
    from . import google_auth
    return google_auth.service("calendar", "v3", account)


def _calendar_tz(service) -> "str | None":
    try:
        return service.calendars().get(calendarId=config.CALENDAR_ID).execute().get("timeZone")
    except Exception:  # noqa: BLE001
        return None


# --- time normalization ------------------------------------------------------
# qwen is unreliable at producing RFC3339 timestamps with the right timezone, and
# the Calendar API 400s on anything else ("next week", "2026-06-23"). So the TOOL
# resolves times: relative phrases and date-only/naive inputs all become RFC3339.
import datetime  # noqa: E402

_RELATIVE = {"now", "today", "tonight", "tomorrow", "this week", "next week",
             "this weekend", "next weekend", "this month", "next month"}


def _local_now() -> "datetime.datetime":
    return datetime.datetime.now().astimezone()


def _sod(dt):   # start of day
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _eod(dt):   # end of day
    return dt.replace(hour=23, minute=59, second=59, microsecond=0)


def _month_end(first):
    nm = (first.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
    return _eod(nm - datetime.timedelta(days=1))


def _relative_window(phrase: str, now):
    """Map a relative phrase to (start, end) datetimes, or None if unrecognized."""
    p = phrase.strip().lower()
    sod, wd = _sod(now), now.weekday()  # Monday=0 .. Sunday=6
    if p == "now":
        return now, None
    if p in ("today", "tonight"):
        return now, _eod(now)
    if p == "tomorrow":
        t = sod + datetime.timedelta(days=1)
        return t, _eod(t)
    if p == "this week":
        return now, _eod(sod + datetime.timedelta(days=6 - wd))
    if p == "next week":
        start = sod + datetime.timedelta(days=7 - wd)
        return start, _eod(start + datetime.timedelta(days=6))
    if p in ("this weekend", "next weekend"):
        sat = sod + datetime.timedelta(days=(5 - wd) % 7)
        if p == "next weekend" and (5 - wd) % 7 == 0:
            sat += datetime.timedelta(days=7)
        return sat, _eod(sat + datetime.timedelta(days=1))
    if p == "this month":
        return now, _month_end(sod.replace(day=1))
    if p == "next month":
        first = (sod.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        return first, _month_end(first)
    return None


def _normalize_dt(value: str, end_of_day: bool, now):
    """Coerce a concrete date/datetime string to an RFC3339 string, or None if unparseable."""
    value = value.strip()
    if "T" not in value:                                   # date-only → start/end of that day
        try:
            d = datetime.date.fromisoformat(value)
        except ValueError:
            return None
        base = datetime.datetime.combine(d, datetime.time(), tzinfo=now.tzinfo)
        return (_eod(base) if end_of_day else base).isoformat()
    if value.endswith("Z") or "+" in value[11:] or "-" in value[11:]:
        return value                                       # already has an offset
    try:                                                   # naive datetime → attach local tz
        return datetime.datetime.fromisoformat(value).astimezone().isoformat()
    except ValueError:
        return None


def _resolve_one(value: str, end_of_day: bool, now):
    """Normalize a single bound (relative phrase, date, naive, or ISO) to RFC3339."""
    if value.strip().lower() in _RELATIVE:
        start, end = _relative_window(value, now)
        chosen = end if (end_of_day and end) else start
        return chosen.isoformat()
    return _normalize_dt(value, end_of_day, now)


def _resolve_window(time_min: "str | None", time_max: "str | None", now=None):
    """Return (time_min, time_max) as RFC3339 strings, accepting relative phrases,
    date-only, naive, or full timestamps. Empty time_min defaults to now."""
    now = now or _local_now()
    if time_min and time_min.strip().lower() in _RELATIVE:
        start, end = _relative_window(time_min, now)
        # An explicit time_max is normalized too (it may be a phrase/date), not passed raw.
        tmax = _resolve_one(time_max, True, now) if time_max else (end.isoformat() if end else None)
        return start.isoformat(), tmax
    tmin = _normalize_dt(time_min, False, now) if time_min else None
    tmax = _normalize_dt(time_max, True, now) if time_max else None
    return (tmin or now.isoformat()), tmax


def _time_field(value: str, tz: "str | None") -> dict:
    """Build a Calendar API start/end field from an ISO string.

    'YYYY-MM-DD' -> all-day; 'YYYY-MM-DDTHH:MM[:SS]' -> timed (tz attached when the
    string carries no offset).
    """
    value = value.strip()
    if "T" not in value:
        return {"date": value}
    field = {"dateTime": value}
    has_offset = value.endswith("Z") or ("+" in value[11:]) or ("-" in value[11:])
    if tz and not has_offset:
        field["timeZone"] = tz
    return field


def list_events(time_min: str = None, time_max: str = None,
                max_results: int = 10, query: str = None, account: str = None) -> str:
    """List upcoming events across connected accounts (or one, if `account` is given).
    Events from every account are merged and sorted; each is tagged with its account."""
    from . import google_auth
    accts = google_auth.accounts_for(account)
    if not accts:
        return ("No Google account is connected for that — authorize one with "
                "`python assistant/tools/google_auth.py <account>`.")
    time_min, time_max = _resolve_window(time_min, time_max)
    rows = []  # (when, account, event)
    for acct in accts:
        params = {"calendarId": config.CALENDAR_ID, "timeMin": time_min,
                  "maxResults": max(1, min(int(max_results), 50)),
                  "singleEvents": True, "orderBy": "startTime"}
        if time_max:
            params["timeMax"] = time_max
        if query:
            params["q"] = query
        try:
            items = _service(acct).events().list(**params).execute().get("items", [])
        except Exception as e:  # noqa: BLE001 — one account failing shouldn't sink the rest
            log.debug("calendar list failed for %s: %s", acct, e)
            continue
        for ev in items:
            start = ev.get("start", {})
            rows.append((start.get("dateTime") or start.get("date") or "", acct, ev))
    if not rows:
        return "No upcoming events found for that query."
    rows.sort(key=lambda r: r[0])
    multi = len(accts) > 1
    lines = []
    for when, acct, ev in rows:
        tag = f"[{acct}] " if multi else ""
        loc = f" @ {ev['location']}" if ev.get("location") else ""
        lines.append(f"- {when or '?'}  {tag}{ev.get('summary', '(no title)')}{loc}  [id: {ev.get('id')}]")
    return "\n".join(lines)


def create_event(summary: str, start: str, end: str, description: str = None,
                 location: str = None, account: str = None) -> str:
    """Create an event on an account's calendar (default: primary). start/end are ISO."""
    from . import google_auth
    acct = account or google_auth.primary_account()
    try:
        service = _service(acct)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    if config.CALENDAR_CONFIRM_WRITES and not approval.confirm_action(
            f"Create calendar event '{summary}' from {start} to {end} on your {acct} calendar?",
            always_ask=True):
        return "DENIED: event not created (you declined)."
    tz = _calendar_tz(service)
    body = {"summary": summary,
            "start": _time_field(start, tz), "end": _time_field(end, tz)}
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    try:
        ev = service.events().insert(calendarId=config.CALENDAR_ID, body=body).execute()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    return f"Created '{summary}' on {acct} — {ev.get('htmlLink', '(no link)')}  [id: {ev.get('id')}]"


def delete_event(event_id: str, account: str = None) -> str:
    """Delete an event by id from an account's calendar (default: primary)."""
    from . import google_auth
    acct = account or google_auth.primary_account()
    try:
        service = _service(acct)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    if config.CALENDAR_CONFIRM_WRITES and not approval.confirm_action(
            f"Delete calendar event {event_id} from your {acct} calendar?", always_ask=True):
        return "DENIED: event not deleted (you declined)."
    try:
        service.events().delete(calendarId=config.CALENDAR_ID, eventId=event_id).execute()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    return f"Deleted event {event_id} from {acct}."


LIST_EVENTS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_events",
        "description": "List upcoming Google Calendar events, optionally within a time "
                       "window or matching a text query. time_min/time_max accept a "
                       "relative phrase ('today', 'tomorrow', 'this week', 'next week', "
                       "'this weekend', 'next month'), a date ('2026-06-23'), or a full "
                       "ISO datetime — the tool resolves them, so prefer the simple phrase.",
        "parameters": {
            "type": "object",
            "properties": {
                "time_min": {"type": "string", "description": "Start of window: relative phrase, date, or ISO datetime (default: now)."},
                "time_max": {"type": "string", "description": "End of window (optional); same formats. A relative phrase in time_min sets this automatically."},
                "max_results": {"type": "integer", "description": "Max events to return (default 10)."},
                "query": {"type": "string", "description": "Free-text filter (optional)."},
                "account": {"type": "string", "description": "Which connected account's calendar (e.g. 'work', 'personal'). Omit to combine ALL accounts."},
            },
            "required": [],
        },
    },
}

CREATE_EVENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_event",
        "description": "Create a Google Calendar event. Confirms with the user before "
                       "writing. start/end are ISO 8601 ('2026-06-21T15:00:00') or a date "
                       "('2026-06-21') for all-day.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Event title."},
                "start": {"type": "string", "description": "ISO start datetime or date."},
                "end": {"type": "string", "description": "ISO end datetime or date."},
                "description": {"type": "string", "description": "Event details (optional)."},
                "location": {"type": "string", "description": "Event location (optional)."},
                "account": {"type": "string", "description": "Which account's calendar (e.g. 'work', 'personal'). Default: primary."},
            },
            "required": ["summary", "start", "end"],
        },
    },
}

DELETE_EVENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delete_event",
        "description": "Delete a Google Calendar event by id (from list_events). Confirms first.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "description": "The event's id."},
                "account": {"type": "string", "description": "The account the event is on (from list_events). Default: primary."},
            },
            "required": ["event_id"],
        },
    },
}


