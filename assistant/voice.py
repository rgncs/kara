"""Voice I/O — local speech-to-text (faster-whisper) and text-to-speech (macOS `say`).

Everything stays on-device, matching Kara's local-first design. STT uses
push-to-talk capture via sounddevice; TTS shells out to macOS `say`.

speak() and transcribe() never raise — voice is a convenience layer and must not
take down the agent. The first mic use will prompt for macOS microphone permission.
"""
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections import deque

import config

log = logging.getLogger("assistant.voice")

_model = None  # lazily-loaded WhisperModel (loading is slow; do it once)
_best = False  # cached best-voice lookup (False = not computed yet)


# --- text-to-speech ----------------------------------------------------------

def _best_voice():
    """Pick the most natural installed English `say` voice: a Premium (neural)
    voice if available, else Enhanced, else None (system default). Cached.

    Lets Kara automatically upgrade the moment the user downloads a Premium voice
    in System Settings — no config needed. config.SAY_VOICE still overrides.
    """
    global _best
    if _best is not False:
        return _best
    _best = None
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    except Exception as e:  # noqa: BLE001
        log.debug("voice list failed: %s", e)
        return _best
    premium, enhanced = [], []
    for line in out.splitlines():
        m = re.match(r"^(.*?)\s{2,}(en[_-]\w+)\s", line)
        if not m:
            continue
        name = m.group(1).strip()
        if "(Premium)" in name:
            premium.append(name)
        elif "(Enhanced)" in name:
            enhanced.append(name)
    pool = premium or enhanced
    _best = pool[0] if pool else None
    if _best:
        log.debug("auto-selected voice: %s", _best)
    return _best

def _clean_for_speech(text: str) -> str:
    """Strip markdown and CODE so spoken output is natural — Kara never recites
    code aloud; she points to the printed version instead."""
    text = re.sub(r"```.*?```", " . I've put the code on your screen. . ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)        # inline code -> bare word
    # Trailing "[1] https://…" reference list: don't read URLs aloud — just note
    # that references are on screen.
    text, n_refs = re.subn(r"(?m)^\s*\[\d+\]\s+https?://\S+.*$", "", text)
    text = re.sub(r"\[(\d+)\]", r"", text)          # inline citation markers
    text = re.sub(r"[*_#>|]+", " ", text)           # markdown punctuation
    text = re.sub(r"https?://\S+", "a link", text)  # any stray URLs are unspeakable
    if n_refs:
        text = text.rstrip() + " . References provided."
    # Drop CJK / other non-Latin scripts the English voice can't read (espeak would
    # otherwise say "Chinese letter" for each character).
    text = re.sub(r"[　-〿぀-ヿ㐀-䶿一-鿿"
                  r"豈-﫿가-힯＀-￯]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def estimate_seconds(text: str) -> float:
    """Estimate how long `text` takes to speak (after code/markdown is stripped)."""
    words = len(_apply_pronunciations(_clean_for_speech(text)).split())
    return words / max(1, config.VOICE_WPM) * 60


# --- wake word --------------------------------------------------------------

# Generous set of ways Whisper renders "Kara"/"Cara" — we never fuss over the
# mispronunciation, we just go with it.
_WAKE_NAME = (r"kara|cara|carra|karra|kahra|khara|kaara|karah|carah|caro|cora|"
              r"kira|kyra|keira|kaira|qara|khara|ckara")
_WAKE_RE = re.compile(
    rf"(?i)^\W*(?:(?:hey|hi|hello|okay|ok|yo|hay)\b[\s,!.]*)?(?:{_WAKE_NAME})\b[\s,.!:?\"'-]*"
)


def strip_wake_word(text: str):
    """If the utterance is addressed to Kara (starts with 'hey kara' / 'cara' / a
    homophone, optionally without 'hey'), return the command after it (may be '').
    Otherwise return None — meaning it wasn't addressed to her, so ignore it."""
    if not text:
        return None
    m = _WAKE_RE.match(text)
    if not m:
        return None
    return text[m.end():].strip()


def _apply_pronunciations(text: str) -> str:
    """Phonetically respell configured words so the TTS says them correctly."""
    for word, phonetic in config.PRONUNCIATIONS.items():
        text = re.sub(rf"\b{re.escape(word)}\b", phonetic, text, flags=re.IGNORECASE)
    return text


def _resolve_engine() -> str:
    engine = config.TTS_ENGINE
    if engine in ("say", "piper"):
        return engine
    return "piper" if os.path.exists(config.PIPER_MODEL) else "say"  # auto


def speak(text: str) -> None:
    """Speak `text` aloud (best-effort). Uses Piper (neural) or macOS `say`."""
    spoken = _apply_pronunciations(_clean_for_speech(text))
    if not spoken:
        return
    if _resolve_engine() == "piper":
        _speak_piper(spoken)
    else:
        _speak_say(spoken)


def _speak_say(text: str) -> None:
    cmd = ["say"]
    chosen = config.SAY_VOICE or _best_voice()  # explicit override, else best installed
    if chosen:
        cmd += ["-v", chosen]
    if config.SAY_RATE:
        cmd += ["-r", str(config.SAY_RATE)]
    cmd.append(text)
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:  # noqa: BLE001
        log.debug("say failed: %s", e)


def _piper_synth(text: str):
    """Synthesize `text` to a temp WAV with Piper. Returns the path, or None on failure."""
    wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav = f.name
        args = [sys.executable, "-m", "piper", "-m", config.PIPER_MODEL, "-f", wav]
        if config.PIPER_LENGTH_SCALE:
            args += ["--length-scale", str(config.PIPER_LENGTH_SCALE)]
        subprocess.run(args, input=text, text=True, capture_output=True, check=True)
        return wav
    except Exception as e:  # noqa: BLE001
        log.debug("piper synth failed: %s", e)
        if wav:
            try:
                os.unlink(wav)
            except OSError:
                pass
        return None


def _speak_piper(text: str) -> None:
    wav = _piper_synth(text)
    if not wav:
        _speak_say(text)
        return
    try:
        subprocess.run(["afplay", wav], check=False)
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass


# --- interruptible playback (tap a key to cut Kara off) ----------------------

def _wait_or_key(proc) -> bool:
    """Wait for playback `proc` to finish; if a key is pressed first (interactive
    terminal only), kill it and return True (interrupted)."""
    if not sys.stdin.isatty():
        proc.wait()
        return False
    import select
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while proc.poll() is None:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                sys.stdin.read(1)               # consume the keypress
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return True
        return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        try:
            termios.tcflush(fd, termios.TCIFLUSH)   # drop stray keys
        except termios.error:
            pass


def speak_interruptible(text: str) -> bool:
    """Speak `text`, but let a keypress cut it off. Returns True if interrupted."""
    spoken = _apply_pronunciations(_clean_for_speech(text))
    if not spoken:
        return False
    if _resolve_engine() == "piper":
        wav = _piper_synth(spoken)
        if wav:
            try:
                return _wait_or_key(subprocess.Popen(["afplay", wav]))
            finally:
                try:
                    os.unlink(wav)
                except OSError:
                    pass
    cmd = ["say"]
    chosen = config.SAY_VOICE or _best_voice()
    if chosen:
        cmd += ["-v", chosen]
    if config.SAY_RATE:
        cmd += ["-r", str(config.SAY_RATE)]
    cmd.append(spoken)
    try:
        return _wait_or_key(subprocess.Popen(cmd))
    except Exception as e:  # noqa: BLE001
        log.debug("speak failed: %s", e)
        return False


# --- speech-to-text ----------------------------------------------------------

def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        log.debug("loading whisper model %s", config.WHISPER_MODEL)
        _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def transcribe(audio) -> str:
    """Transcribe a 16 kHz mono float32 numpy array (or an audio file path) to text."""
    try:
        segments, _ = _get_model().transcribe(audio, language="en", vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception as e:  # noqa: BLE001
        log.debug("transcribe failed: %s", e)
        return ""


def record_until_enter():
    """Record from the mic until the user presses Enter. Returns a float32 array."""
    import numpy as np
    import sounddevice as sd

    frames = []

    def _cb(indata, _frames, _time, _status):
        frames.append(indata.copy())

    print("  ● recording… press Enter to stop", end="", flush=True)
    try:
        with sd.InputStream(samplerate=config.VOICE_SAMPLE_RATE, channels=1,
                            dtype="float32", callback=_cb):
            input()  # blocks here while the stream records, until Enter
    except Exception as e:  # noqa: BLE001 — most likely missing mic permission
        print(f"\n  [mic error: {e} — grant Terminal microphone access in "
              "System Settings → Privacy & Security → Microphone]")
        return np.zeros(0, dtype="float32")
    if not frames:
        return np.zeros(0, dtype="float32")
    return np.concatenate(frames, axis=0).flatten()


def listen() -> str:
    """Push-to-talk: record until Enter, then transcribe. Returns recognized text."""
    audio = record_until_enter()
    if getattr(audio, "size", 0) == 0:
        return ""
    print("\r  ◌ transcribing…            ", end="", flush=True)
    text = transcribe(audio)
    print("\r" + " " * 32 + "\r", end="", flush=True)
    return text


# --- hands-free listening (voice-activity detection) ------------------------

_vad = False  # cached webrtcvad instance (False = not computed, None = unavailable)


def _get_vad():
    global _vad
    if _vad is False:
        try:
            import webrtcvad
            _vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        except Exception as e:  # noqa: BLE001
            log.debug("webrtcvad unavailable (%s); using energy VAD", e)
            _vad = None
    return _vad


def _is_speech(frame_i16, sample_rate) -> bool:
    """Voiced-frame test: webrtcvad if available, else an energy threshold."""
    vad = _get_vad()
    if vad is not None:
        try:
            return vad.is_speech(frame_i16.tobytes(), sample_rate)
        except Exception:  # noqa: BLE001 — bad frame size etc.
            pass
    import numpy as np
    rms = float(np.sqrt(np.mean((frame_i16.astype("float32")) ** 2)))
    return rms > 500.0  # ~ -36 dBFS; coarse fallback


def listen_vad(start_timeout: "float | None" = None) -> "str | None":
    """Hands-free: open the mic, wait for speech, capture until the user stops talking
    (trailing silence), then transcribe. Returns recognized text ('' if a blip was too
    short to use). If `start_timeout` seconds pass with no speech beginning, returns
    None. Raises KeyboardInterrupt through to the caller on Ctrl-C."""
    import numpy as np
    import sounddevice as sd

    sr = config.VOICE_SAMPLE_RATE
    frame_ms = 30
    n = int(sr * frame_ms / 1000)                       # samples per VAD frame
    need_start = max(1, config.VAD_START_MS // frame_ms)
    need_silence = max(1, config.VAD_SILENCE_MS // frame_ms)
    max_wait = int(start_timeout * 1000 / frame_ms) if start_timeout else None
    preroll = deque(maxlen=need_start + 4)

    voiced, triggered, speech_run, silence_run, waited = [], False, 0, 0, 0
    try:
        stream = sd.InputStream(samplerate=sr, channels=1, dtype="int16", blocksize=n)
    except Exception as e:  # noqa: BLE001 — usually missing mic permission
        print(f"\n  [mic error: {e} — grant Terminal microphone access in "
              "System Settings → Privacy & Security → Microphone]")
        return ""

    with stream:
        while True:
            data, _ = stream.read(n)
            frame = data[:, 0]
            speech = _is_speech(frame, sr)
            if not triggered:
                if max_wait is not None and waited >= max_wait:
                    return None                         # no speech within the timeout
                waited += 1
                preroll.append(frame)
                speech_run = speech_run + 1 if speech else 0
                if speech_run >= need_start:
                    triggered = True
                    voiced.extend(preroll)              # keep the onset
                    preroll.clear()
            else:
                voiced.append(frame)
                silence_run = 0 if speech else silence_run + 1
                if silence_run >= need_silence:
                    break

    audio = np.concatenate(voiced).astype("float32") / 32768.0
    if len(audio) / sr * 1000 < config.VAD_MIN_SPEECH_MS:
        return ""                                       # too short — noise blip
    return transcribe(audio)
