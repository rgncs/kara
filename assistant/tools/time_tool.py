"""get_current_time — the spec's trivial tool that proves the loop works."""
import datetime


def get_current_time() -> str:
    """Return the current local date and time."""
    now = datetime.datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current local date and time.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
