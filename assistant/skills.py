"""Skills — Markdown playbooks injected into a turn when relevant.

A skill is a vetted Markdown file with simple frontmatter:

    ---
    name: arxiv-summarizer
    description: Find and summarize recent papers from arXiv on a topic.
    triggers: arxiv, \\bpaper(s)?\\b, preprint
    ---
    <step-by-step instructions, in Markdown, for doing the task>

This mirrors OpenClaw / Anthropic "Agent Skills": skills are PROSE, not code, so
they are model-agnostic and portable. The trade-off is that a skill only works
through tools Kara already has (read_file, write_file, list_dir, run_command,
web_search, fetch_url) — it can't conjure new capabilities, and it never bypasses
the run_command approval gate. Skills are committed, vetted files; they are NOT
auto-installed from any registry (that would be an untrusted-instruction / RCE
vector — see the discussion that motivated this module).

A skill is matched to a turn when any of its `triggers` (case-insensitive regex,
comma-separated) matches the user's message. Matched skills are folded into the
turn as a playbook block, the same way recalled memories are.
"""
import logging
import os
import re

import config

log = logging.getLogger("assistant.skills")

_CACHE: "list[dict] | None" = None


def _parse(path: str) -> "dict | None":
    """Parse one skill file into {name, description, triggers, body, path}.

    Returns None for a file without valid `---` frontmatter and a `name`.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        log.debug("could not read skill %s: %s", path, e)
        return None

    m = re.match(r"\s*---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL)
    if not m:
        log.debug("skill %s has no frontmatter — skipped", path)
        return None
    front, body = m.group(1), m.group(2).strip()

    meta: dict[str, str] = {}
    for line in front.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip().lower()] = val.strip()

    name = meta.get("name") or os.path.splitext(os.path.basename(path))[0]
    if not name or not body:
        return None

    triggers = []
    for raw in (meta.get("triggers") or "").split(","):
        pat = raw.strip()
        if not pat:
            continue
        try:
            triggers.append(re.compile(pat, re.IGNORECASE))
        except re.error:                       # treat a bad pattern as a literal phrase
            triggers.append(re.compile(re.escape(pat), re.IGNORECASE))

    return {"name": name, "description": meta.get("description", ""),
            "triggers": triggers, "body": body, "path": path}


def load_skills(directory: "str | None" = None, force: bool = False) -> list[dict]:
    """Load and cache all skills from SKILLS_DIR (or `directory`)."""
    global _CACHE
    if directory is None and _CACHE is not None and not force:
        return _CACHE
    root = directory or config.SKILLS_DIR
    skills: list[dict] = []
    if os.path.isdir(root):
        for fn in sorted(os.listdir(root)):
            if fn.endswith(".md") and not fn.startswith("_"):
                skill = _parse(os.path.join(root, fn))
                if skill:
                    skills.append(skill)
    log.debug("loaded %d skill(s) from %s", len(skills), root)
    if directory is None:
        _CACHE = skills
    return skills


def match_skills(user_input: str, skills: "list[dict] | None" = None,
                 limit: int = None) -> list[dict]:
    """Return skills whose triggers match `user_input`, in file order, up to `limit`."""
    if not config.SKILLS_ENABLED or not (user_input or "").strip():
        return []
    if limit is None:
        limit = config.MAX_SKILLS_PER_TURN
    pool = skills if skills is not None else load_skills()
    matched = [s for s in pool if any(t.search(user_input) for t in s["triggers"])]
    return matched[:limit]


def skills_preface(matched: list[dict]) -> str:
    """Render matched skills as a playbook block to fold into the user turn."""
    if not matched:
        return ""
    blocks = [f"### Skill: {s['name']}\n{s['body']}" for s in matched]
    return (
        "[Relevant skill playbook(s) for this request. Follow the steps that apply, "
        "using your normal tools; the run_command approval gate still applies to any "
        "shell command. If a skill doesn't actually fit what I asked, ignore it.\n\n"
        + "\n\n".join(blocks) + "]"
    )
