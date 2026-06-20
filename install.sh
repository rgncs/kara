#!/bin/bash
# Kara installer — sets up the local coding agent on macOS.
#
# Idempotent: safe to re-run. It will
#   1. ensure Homebrew, Ollama, and a Python 3.11+ are present
#   2. create the .venv and install Python deps
#   3. pull the chat + embedding models (skip with --no-models)
#   4. link the `kara` command onto your PATH
#
# Usage:
#   ./install.sh                # full install
#   ./install.sh --no-models    # skip the (large) model downloads

set -euo pipefail

# --- locate the repo (this script's directory) ------------------------------
KARA_HOME="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
VENV="$KARA_HOME/.venv"
PULL_MODELS=1
[ "${1:-}" = "--no-models" ] && PULL_MODELS=0

say()  { printf "\033[1;36m==>\033[0m %s\n" "$1"; }
warn() { printf "\033[1;33m!!\033[0m %s\n" "$1" >&2; }
die()  { printf "\033[1;31mxx\033[0m %s\n" "$1" >&2; exit 1; }

[ "$(uname)" = "Darwin" ] || die "This installer targets macOS. On Linux, install Ollama + Python 3.11 manually, then run bin/kara."

# --- 1. Homebrew ------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  die "Homebrew not found. Install it from https://brew.sh then re-run ./install.sh"
fi

# --- 2. Ollama --------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1 && [ ! -x /opt/homebrew/opt/ollama/bin/ollama ]; then
  say "Installing Ollama..."
  brew install ollama
else
  say "Ollama already installed."
fi
OLLAMA_BIN="$(command -v ollama || echo /opt/homebrew/opt/ollama/bin/ollama)"

# --- 3. Python 3.11+ --------------------------------------------------------
find_python() {
  for c in python3.13 python3.12 python3.11; do
    command -v "$c" >/dev/null 2>&1 && { echo "$c"; return 0; }
  done
  if command -v python3 >/dev/null 2>&1 && \
     python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,11) else 1)'; then
    echo python3; return 0
  fi
  return 1
}
if ! PYBIN="$(find_python)"; then
  say "Installing Python 3.12..."
  brew install python@3.12
  PYBIN="$(find_python)" || die "Python 3.11+ still not found after install."
fi
say "Using Python: $($PYBIN --version) ($PYBIN)"

# --- 4. venv + deps ---------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  say "Creating virtualenv at .venv ..."
  "$PYBIN" -m venv "$VENV"
fi
say "Installing Python dependencies..."
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -r "$KARA_HOME/requirements.txt"

# Piper neural TTS voice (for `kara --voice`) — download once if missing.
PIPER_VOICE="$KARA_HOME/voices/en_US-amy-medium.onnx"
if [ ! -f "$PIPER_VOICE" ]; then
  say "Downloading Piper neural voice (~63 MB)..."
  mkdir -p "$KARA_HOME/voices"
  PV="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
  curl -fsSL "$PV" -o "$PIPER_VOICE" && curl -fsSL "$PV.json" -o "$PIPER_VOICE.json" \
    || warn "Piper voice download failed — voice mode will fall back to macOS \`say\`."
else
  say "Piper voice already present."
fi

# --- 5. models --------------------------------------------------------------
# Read the model names straight from config so this stays in sync.
read -r CHAT_MODEL EMBED_MODEL < <(
  PYTHONPATH="$KARA_HOME/assistant" "$VENV/bin/python" -c \
    "import config; print(config.CHAT_MODEL, config.EMBED_MODEL)"
)
if [ "$PULL_MODELS" = "1" ]; then
  if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
    say "Starting Ollama daemon..."
    "$OLLAMA_BIN" serve >/tmp/ollama.log 2>&1 &
    for _ in $(seq 1 30); do
      curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1 && break
      sleep 0.5
    done
  fi
  for m in "$CHAT_MODEL" "$EMBED_MODEL"; do
    if "$OLLAMA_BIN" list 2>/dev/null | awk '{print $1}' | grep -q "^${m%%:*}"; then
      say "Model '$m' already present."
    else
      say "Pulling model '$m' (this can take a while)..."
      "$OLLAMA_BIN" pull "$m"
    fi
  done
else
  warn "Skipping model downloads (--no-models). Pull later: ollama pull $CHAT_MODEL && ollama pull $EMBED_MODEL"
fi

# --- 6. link the `kara` command ---------------------------------------------
chmod +x "$KARA_HOME/bin/kara"
link_into() {  # $1 = target dir on PATH
  ln -sf "$KARA_HOME/bin/kara" "$1/kara" && say "Linked 'kara' -> $1/kara"
}
if [ -d /opt/homebrew/bin ] && [ -w /opt/homebrew/bin ]; then
  link_into /opt/homebrew/bin
elif [ -w /usr/local/bin ]; then
  link_into /usr/local/bin
else
  mkdir -p "$HOME/.local/bin"
  link_into "$HOME/.local/bin"
  case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) warn "Add this to your ~/.zshrc:  export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
  esac
fi

echo
say "Done. Start Kara from any project directory:"
echo "    cd /path/to/your/project && kara"
