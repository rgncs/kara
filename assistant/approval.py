"""Human-in-the-loop approval for shell commands the agent wants to run.

run_command consults is_approved() before executing. The actual interactive
prompt is injected by the CLI via set_approver() so this module stays free of
terminal I/O and is easy to test. When no approver is registered (e.g. a
non-interactive run) the default is to DENY — fail safe, never silently execute.

Approval scope, narrowest to widest:
  - once          : just this command
  - prefix        : every command sharing a leading program name (e.g. all `git …`)
  - session/all   : every command for the rest of the session
"""
import logging
import shlex

import config

log = logging.getLogger("assistant.approval")

# Shell metacharacters that make a command "compound" — prefix-approval is NOT
# offered for these, so an approved `git` prefix can't smuggle in `&& rm -rf`.
_SHELL_OPS = ("&&", "||", ";", "|", "`", "$(", ">", "<", "\n")

_session_auto = False               # user approved everything for this session
_session_allowed: set[str] = set()  # specific exact commands approved this session
_session_prefixes: set[str] = set() # approved leading program names (e.g. "git")
_approver = None                    # callable(command) -> (bool, reason), set by the CLI


def command_prefix(command: str):
    """Return the leading program name if it's safe to approve by prefix, else None.

    None for empty input, unparseable quoting, or compound commands containing
    shell operators — those can only be approved one-shot.
    """
    cmd = command.strip()
    if not cmd or any(op in cmd for op in _SHELL_OPS):
        return None
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return None
    return tokens[0] if tokens else None


def set_approver(fn) -> None:
    """Register the interactive approver (CLI provides one; tests may stub it)."""
    global _approver
    _approver = fn


def approve_session() -> None:
    """Approve all subsequent commands for the rest of this session."""
    global _session_auto
    _session_auto = True


def allow_command(command: str) -> None:
    """Approve this exact command for the rest of this session."""
    _session_allowed.add(command.strip())


def allow_prefix(prefix: str) -> None:
    """Approve every command sharing this leading program name for the session."""
    _session_prefixes.add(prefix)


def reset() -> None:
    """Clear all session approval state (used by tests)."""
    global _session_auto, _approver
    _session_auto = False
    _session_allowed.clear()
    _session_prefixes.clear()
    _approver = None


def is_approved(command: str) -> tuple[bool, str]:
    """Decide whether `command` may run. Returns (allowed, human-readable reason)."""
    command = command.strip()
    mode = config.COMMAND_APPROVAL

    if mode == "auto":
        return True, "auto-approve mode"
    if mode == "deny":
        return False, "shell disabled (COMMAND_APPROVAL=deny)"

    # mode == "prompt" (or anything unrecognized → fail safe to prompting)
    if _session_auto:
        return True, "approved for session"
    if command in _session_allowed:
        return True, "previously approved this session"
    prefix = command_prefix(command)
    if prefix and prefix in _session_prefixes:
        return True, f"approved all '{prefix}' commands this session"
    if _approver is None:
        return False, "no approver available (non-interactive); set COMMAND_APPROVAL=auto to allow"

    allowed, reason = _approver(command)
    log.debug("approver(%r) -> %s (%s)", command, allowed, reason)
    return allowed, reason
