# Getting Started with Kara

A 2-minute guide to launching the Ollama server and running the Kara CLI.

> First time on this machine? Run `./install.sh` once first (installs Ollama +
> Python deps, pulls the models, and adds the `kara` command).

---

## 1. Start the Ollama server

Kara needs the Ollama daemon running (it serves the model at
`http://localhost:11434`). You have two options:

**Option A — let Kara start it for you (easiest).**
The `kara` command auto-starts the daemon if it isn't already running. You can skip
straight to step 2.

**Option B — start it yourself.**

```bash
ollama serve            # runs in the foreground (Ctrl-C to stop)
# ...or in the background:
ollama serve &
```

**Or run it permanently** (auto-starts at login, survives reboots):

```bash
brew services start ollama
```

Verify it's up:

```bash
curl -s http://localhost:11434/api/tags && echo "  <- Ollama is running"
```

---

## 2. Run the Kara CLI

`kara` works from **any** directory. The folder you launch it in becomes its
**workspace** — the only place its file tools and shell commands can touch.

```bash
cd ~/path/to/your/project     # the project you want Kara to work on
kara
```

You'll see:

```
Kara — local coding agent (model: qwen3-coder:30b)
Workspace: /Users/you/path/to/your/project
Shell approval: prompt
Ctrl-D or 'exit' to quit. Set ASSISTANT_DEBUG=1 to see tool calls.

you ▸
```

Type a request and press Enter, e.g.:

```
you ▸ list the python files here, then add a docstring to the top of main.py
```

When Kara wants to run a shell command, you'll be asked to approve it:

```
  ⚠  the agent wants to run a shell command:
      pytest -q
  [y] yes, once   [p] yes, and don't ask again for 'pytest' commands   [a] yes, all commands this session   [n] no
  >
```

To quit: type `exit`, `quit`, or press **Ctrl-D**.

---

## 3. Handy options

```bash
COMMAND_APPROVAL=auto kara     # don't prompt before shell commands (trusted task)
ASSISTANT_DEBUG=1 kara         # print every model + tool call (debugging)
WORKSPACE_ROOT=/some/dir kara  # use a different workspace than the current folder
kara --voice                   # talk to Kara (push-to-talk) and hear her replies
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Ollama not reachable …` | Start the server: `ollama serve &` (or `brew services start ollama`). |
| `Model 'qwen3-coder:30b' not found` | `ollama pull qwen3-coder:30b` |
| `kara: command not found` | Re-run `./install.sh`, or open a new terminal so `PATH` refreshes. |
| Commands never run | You're in `prompt` mode and answering `n` — answer `y`/`p`/`a`, or launch with `COMMAND_APPROVAL=auto`. |
