---
name: calendar
description: Check, create, or delete Google Calendar events.
triggers: calendar, my schedule, what('s| is| do i have).*(today|tomorrow|this week|on), am i free, do i have (anything|plans|meetings), schedule (a|an|me)?, (add|put|create|book|set up).*(meeting|event|appointment|reminder).*(calendar|schedule)?, cancel (my|the).*(meeting|event|appointment), free on
---
I may have MULTIPLE Google accounts connected (e.g. work, personal). When I ask about
my schedule generally, COMBINE all of them — call `list_events` with no `account` so it
merges every calendar (each event is tagged with its account). Only pass `account` when
I name one ("my work calendar"). For create/delete, use the account I indicate, or the
primary by default; the confirmation will name the account.

When I ask about my schedule or to add/change events, use the calendar tools:

1. To answer "what's on my calendar / am I free / do I have plans" — call
   `list_events`. Narrow it with `time_min`/`time_max` (ISO 8601) for a day or
   week, and summarize the results in plain language (don't dump raw ids unless I
   ask). If there's nothing, say the time looks free.
2. To add something — call `create_event` with a clear `summary`, an ISO `start`
   and `end` (e.g. `2026-06-21T15:00:00`; use a date like `2026-06-21` for
   all-day). Infer reasonable times from how I phrase it ("lunch tomorrow" → ~1h
   around noon tomorrow) but keep the title faithful to what I said. The tool will
   ask me to confirm before it writes — that's expected.
3. To cancel something — `list_events` first to find the matching event's id, then
   `delete_event` with that id. Confirm which event if more than one could match.
4. Resolve relative dates ("today", "next Friday") against the current date, and
   when a time is ambiguous, briefly state the assumption rather than guessing
   silently. Never invent events that aren't really on the calendar.
