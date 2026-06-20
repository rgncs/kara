---
name: code-review
description: Structured review of a code change, file, or diff.
triggers: review (this|my|the) (code|change|diff|pr|file|function), code review, critique (this|my) code, look over (this|my) code, any (bugs|issues) in
---
When I ask you to review code:

1. Get the code first. If I named a file, `read_file` it. If I asked about
   uncommitted changes, run `git diff` (or `git diff --staged`) with
   `run_command`. If I pasted the code, use that. Never review from memory.
2. Read it carefully, then report findings grouped by severity:
   - **Bugs / correctness** — logic errors, off-by-one, unhandled cases, races.
   - **Security** — injection, unsafe shell/eval, secrets in code, path escapes.
   - **Design / maintainability** — naming, duplication, missing error handling.
   - **Nits** — style, typos (keep these brief).
3. For each finding give `file:line`, what's wrong, and a concrete fix — ideally a
   small code snippet. Be specific; avoid vague advice.
4. If something is genuinely good, say so briefly. If you find nothing material,
   say the change looks sound rather than padding the list.
5. Do NOT modify files unless I ask you to apply the fixes.
