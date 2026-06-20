---
name: git-commit
description: Stage changes and write a clear, conventional commit message.
triggers: \bcommit\b, git commit, stage (and|&) commit, write a commit message
---
When I ask you to commit changes:

1. First run `git status` and `git diff` (and `git diff --staged`) with
   `run_command` so you commit based on the ACTUAL changes, not assumptions.
2. Scan the diff for anything that must not be committed — API keys, tokens,
   passwords, `.env` contents. If you see a secret, STOP and warn me instead of
   committing.
3. Stage what belongs together (`git add <paths>`), keeping unrelated changes out.
4. Write a concise message: a ~50-char summary line in the imperative
   ("Add X", "Fix Y"), a blank line, then 1–4 bullets on what changed and why.
   Don't describe every line; capture the intent.
5. Show me the message and the files you're about to commit before running
   `git commit`. Every git command goes through the approval gate — only push if I
   explicitly ask.
