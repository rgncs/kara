"""Google Calendar tools: list_events, create_event, delete_event.

Auth is the OAuth "desktop app" flow. You create credentials.json in the Google
Cloud console (Calendar API enabled); the first authenticated call (or running
this module directly) opens a browser once and writes token.json with a refresh
token. Later runs reuse and silently refresh it.

The google-* libraries are imported lazily so the rest of Kara runs even when
they aren't installed — every function returns a readable "ERROR: ..." string
instead of raising, matching the other tools. Reads are free; create/delete go
through approval.confirm_action() (the same human-in-the-loop gate as the shell)
unless CALENDAR_CONFIRM_WRITES is off.
"""
import logging
import os
import sys

# Allow running this file directly for one-time OAuth setup
# (`python assistant/tools/calendar_tool.py`): put assistant/ on the path so the
# sibling `approval`/`config` modules resolve the same way they do under main.py.
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import approval
import config

log = logging.getLogger("assistant.calendar")

_SERVICE = None  # cached authenticated Calendar API client


def _service():
    """Return an authenticated Calendar API client, running the OAuth flow if needed."""
    global _SERVICE
    if _SERVICE is not None:
        return _SERVICE
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    scopes = config.GOOGLE_CALENDAR_SCOPES
    creds = None
    if os.path.exists(config.GOOGLE_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(config.GOOGLE_TOKEN_PATH, scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(config.GOOGLE_CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"missing {config.GOOGLE_CREDENTIALS_PATH} — download an OAuth "
                    "'desktop app' credentials.json from Google Cloud console first")
            flow = InstalledAppFlow.from_client_secrets_file(config.GOOGLE_CREDENTIALS_PATH, scopes)
            creds = flow.run_local_server(port=0)
        with open(config.GOOGLE_TOKEN_PATH, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    _SERVICE = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return _SERVICE


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


def _resolve_window(time_min: "str | None", time_max: "str | None", now=None):
    """Return (time_min, time_max) as RFC3339 strings, accepting relative phrases,
    date-only, naive, or full timestamps. Empty time_min defaults to now."""
    now = now or _local_now()
    if time_min and time_min.strip().lower() in _RELATIVE:
        start, end = _relative_window(time_min, now)
        return start.isoformat(), (time_max or (end.isoformat() if end else None))
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
                max_results: int = 10, query: str = None) -> str:
    """List upcoming events (optionally within a window or matching a text query)."""
    try:
        service = _service()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    time_min, time_max = _resolve_window(time_min, time_max)
    params = {"calendarId": config.CALENDAR_ID, "timeMin": time_min,
              "maxResults": max(1, min(int(max_results), 50)),
              "singleEvents": True, "orderBy": "startTime"}
    if time_max:
        params["timeMax"] = time_max
    if query:
        params["q"] = query
    try:
        items = service.events().list(**params).execute().get("items", [])
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    if not items:
        return "No upcoming events found for that query."
    lines = []
    for ev in items:
        start = ev.get("start", {})
        when = start.get("dateTime") or start.get("date") or "?"
        summary = ev.get("summary", "(no title)")
        loc = f" @ {ev['location']}" if ev.get("location") else ""
        lines.append(f"- {when}  {summary}{loc}  [id: {ev.get('id')}]")
    return "\n".join(lines)


def create_event(summary: str, start: str, end: str,
                 description: str = None, location: str = None) -> str:
    """Create an event. start/end are ISO strings ('2026-06-21T15:00:00' or a date)."""
    try:
        service = _service()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    if config.CALENDAR_CONFIRM_WRITES and not approval.confirm_action(
            f"Create calendar event '{summary}' from {start} to {end}?"):
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
    return f"Created '{summary}' — {ev.get('htmlLink', '(no link)')}  [id: {ev.get('id')}]"


def delete_event(event_id: str) -> str:
    """Delete an event by its id (get the id from list_events first)."""
    try:
        service = _service()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    if config.CALENDAR_CONFIRM_WRITES and not approval.confirm_action(
            f"Delete calendar event with id {event_id}?"):
        return "DENIED: event not deleted (you declined)."
    try:
        service.events().delete(calendarId=config.CALENDAR_ID, eventId=event_id).execute()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    return f"Deleted event {event_id}."


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
            "properties": {"event_id": {"type": "string", "description": "The event's id."}},
            "required": ["event_id"],
        },
    },
}


if __name__ == "__main__":
    # One-time setup: trigger the OAuth consent flow and cache token.json.
    print("Authorizing Google Calendar access (a browser window will open)…")
    _service()
    print(f"Done — token saved to {config.GOOGLE_TOKEN_PATH}")
