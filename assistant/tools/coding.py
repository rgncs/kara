"""Coding tools: read_file, write_file, list_dir, run_command.

The FILE tools (read_file, write_file, list_dir) are confined to
config.WORKSPACE_ROOT via _resolve(), which resolves symlinks so they cannot be
used to escape the root.

run_command is NOT sandboxed: it runs a real shell (`cat /etc/passwd`, `curl … | sh`,
etc. all work) in the workspace directory. Its only guard is the human approval
gate (see approval.py / config.COMMAND_APPROVAL). Treat the approval gate — not the
file-path confinement — as the security boundary whenever the shell is enabled.

Output is truncated to protect the context window. Tool errors are returned as
strings; agent_turn surfaces raised exceptions as "ERROR: ..." to the model.
"""
import os
import signal
import subprocess
import sys

import approval
import config

# Put Kara's own venv first on PATH so `python`/`pip` in run_command resolve to the
# interpreter that has Kara's dependencies (openpyxl, python-docx, pandas, …).
_ENV = os.environ.copy()
_ENV["PATH"] = os.path.dirname(sys.executable) + os.pathsep + _ENV.get("PATH", "")


def _resolve(path: str) -> str:
    """Resolve a (possibly relative) path against the workspace and confine it there.

    Uses realpath so a symlink inside the workspace cannot point outside it and
    slip past the prefix check.
    """
    root = os.path.realpath(config.WORKSPACE_ROOT)
    full = os.path.realpath(os.path.join(root, path))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError(f"path escapes workspace root: {path}")
    return full


def _truncate(text: str) -> str:
    limit = config.MAX_TOOL_OUTPUT_CHARS
    if len(text) > limit:
        return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"
    return text


def read_file(path: str) -> str:
    """Return the text contents of a file in the workspace."""
    full = _resolve(path)
    with open(full, "r", encoding="utf-8", errors="replace") as f:
        return _truncate(f.read())


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file in the workspace with the given content."""
    full = _resolve(path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"wrote {len(content)} chars to {path}"


def list_dir(path: str = ".") -> str:
    """List entries in a workspace directory (directories marked with a trailing /)."""
    full = _resolve(path)
    if not os.path.isdir(full):
        return f"ERROR: not a directory: {path}"
    entries = []
    for name in sorted(os.listdir(full)):
        marker = "/" if os.path.isdir(os.path.join(full, name)) else ""
        entries.append(name + marker)
    return _truncate("\n".join(entries) or "(empty)")


def run_command(command: str) -> str:
    """Run a shell command in the workspace and return combined stdout/stderr + exit code.

    Gated by the approval policy — a denied command is reported back to the model
    rather than executed, so it can adapt or explain.
    """
    allowed, reason = approval.is_approved(command)
    if not allowed:
        return f"DENIED: command not run ({reason}). Command was: {command}"
    # start_new_session=True puts the shell in its own process group so a timeout
    # can kill any grandchildren it spawned, not just the shell itself.
    proc = subprocess.Popen(
        command, shell=True, cwd=config.WORKSPACE_ROOT, text=True, env=_ENV,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=config.RUN_COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        return f"ERROR: command timed out after {config.RUN_COMMAND_TIMEOUT}s"
    return _truncate(f"[exit {proc.returncode}]\n{(out or '') + (err or '')}".rstrip())


READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read and return the text contents of a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path relative to the workspace root."}},
            "required": ["path"],
        },
    },
}

WRITE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Create or overwrite a file in the workspace. Creates parent directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root."},
                "content": {"type": "string", "description": "Full file contents to write."},
            },
            "required": ["path", "content"],
        },
    },
}

LIST_DIR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_dir",
        "description": "List the entries in a workspace directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path relative to the workspace root (default '.')."}},
            "required": [],
        },
    },
}

RUN_COMMAND_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Run a shell command in the workspace (e.g. run tests, git, build). Returns stdout/stderr and exit code.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to execute."}},
            "required": ["command"],
        },
    },
}
