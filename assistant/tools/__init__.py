"""Tool registry — two synced halves.

TOOLS          : JSON-schema list sent to the model (what it's allowed to call).
TOOL_FUNCTIONS : name -> callable map executed by agent_turn().

Adding a capability = add one entry to each. agent_turn() never changes.
"""
import config

from . import calc_tool, coding, memory_tool, search, time_tool

TOOLS = [
    time_tool.SCHEMA,
    calc_tool.SCHEMA,
    coding.READ_FILE_SCHEMA,
    coding.WRITE_FILE_SCHEMA,
    coding.LIST_DIR_SCHEMA,
    coding.RUN_COMMAND_SCHEMA,
    search.WEB_SEARCH_SCHEMA,
    search.FETCH_URL_SCHEMA,
]

TOOL_FUNCTIONS = {
    "get_current_time": time_tool.get_current_time,
    "calculate": calc_tool.calculate,
    "read_file": coding.read_file,
    "write_file": coding.write_file,
    "list_dir": coding.list_dir,
    "run_command": coding.run_command,
    "web_search": search.web_search,
    "fetch_url": search.fetch_url,
}

# Memory writes are code-driven by default; only expose save_memory to the model
# when explicitly enabled (it tends to store noise — see config.MEMORY_TOOL_ENABLED).
if config.MEMORY_TOOL_ENABLED:
    TOOLS.append(memory_tool.SCHEMA)
    TOOL_FUNCTIONS["save_memory"] = memory_tool.save_memory

# The reasoning subagent (`think`) lets the controller delegate hard problems to
# config.REASONING_MODEL. On by default; disable with REASONING=0.
if config.REASONING_ENABLED:
    from . import reasoning
    TOOLS.append(reasoning.SCHEMA)
    TOOL_FUNCTIONS["think"] = reasoning.think

# Google Calendar tools appear only when set up (credentials.json present) or forced
# on with CALENDAR=1, so they don't clutter the tool list otherwise.
if config.CALENDAR_ENABLED:
    from . import calendar_tool
    TOOLS += [calendar_tool.LIST_EVENTS_SCHEMA, calendar_tool.CREATE_EVENT_SCHEMA,
              calendar_tool.DELETE_EVENT_SCHEMA]
    TOOL_FUNCTIONS.update({
        "list_events": calendar_tool.list_events,
        "create_event": calendar_tool.create_event,
        "delete_event": calendar_tool.delete_event,
    })

# Guard against the registry halves drifting out of sync.
_schema_names = {t["function"]["name"] for t in TOOLS}
assert _schema_names == set(TOOL_FUNCTIONS), (
    f"TOOLS / TOOL_FUNCTIONS mismatch: {_schema_names ^ set(TOOL_FUNCTIONS)}"
)
