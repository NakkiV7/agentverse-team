"""Text to speech, fully on the laptop, with a choice of engines (config TTS_ENGINE).

Engines (buddy/config.py, TTS_ENGINE):
  piper   Piper (VITS, onnxruntime on the CPU), en_GB voice from ~/.cache/piper: about 50 ms a sentence alone and
          under 200 ms with the demo running. The default: it never waits for the GPU that Gemma and MuJoCo share.
  kokoro  Kokoro-82M through the kokoro package (torch), on Apple's GPU (TTS_DEVICE=mps, about 150 ms a sentence
          alone) or the CPU (TTS_DEVICE=cpu, 350 to 650 ms alone, 1.5 to 4 s inside the running demo). The nicest
          voice, kept as an option.
  say     Apple's built-in `say` command (Daniel, Eddy ...), written to a WAV file and played: 300 to 700 ms, no
          model to load. The fallback when nothing else is installed.

Pipeline: text -> sentences (a long first sentence is split at its commas so the first audio comes sooner) -> the
engine synthesises piece 1 -> it starts playing while piece 2 is synthesised. So Penny starts talking after one
piece's worth of synthesis, not after the whole reply.

Two threads: the synthesiser turns queued text into audio chunks, the player plays chunks in order through one
sounddevice OutputStream per reply (reopened once, after resetting PortAudio, if the output device changed under it:
the old `sd.play` path died with "Internal PortAudio error" then). `speaking` is True from the first chunk until the
last one has played; the microphone listener ignores audio during that time (plus a short tail) so Penny does not
hear itself.
"""
from __future__ import annotations

import queue
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .. import config as C

_SENT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE = re.compile(r"(?<=[,;:])\s+")
LONG_FIRST = 12                 # words: a longer first sentence is split at its commas (slow engines only)
MIN_PIECE = 4                   # words: never make a piece shorter than this


def split_text(text: str, split_first: bool = False) -> list[str]:
    """Sentences; with split_first, a long first sentence becomes clauses so the first audio is ready sooner."""
    sentences = [s for s in _SENT.split(text.strip()) if s]
    if not sentences or not split_first or len(sentences[0].split()) <= LONG_FIRST:
        return sentences
    pieces, cur = [], ""
    for clause in _CLAUSE.split(sentences[0]):
        cur = f"{cur} {clause}".strip()
        if len(cur.split()) >= MIN_PIECE:
            pieces.append(cur)
            cur = ""
    if cur:                                                   # a short tail joins the previous piece
        if pieces:
            pieces[-1] = f"{pieces[-1]} {cur}"
        else:
            pieces.append(cur)
    return pieces + sentences[1:]


def fade(audio: np.ndarray, sr: int, ms: float = 5.0) -> np.ndarray:
    """A short raised-cosine fade in and out, so chunk edges never click."""
    n = min(int(sr * ms / 1000), len(audio) // 2)
    if n <= 0:
        return audio
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n, dtype=np.float32))
    audio = np.array(audio, dtype=np.float32, copy=True)
    audio[:n] *= ramp
    audio[-n:] *= ramp[::-1]
    return audio


# ---------------------------------------------------------------------------------------------------------- engines
class KokoroEngine:
    name = "kokoro"
    slow = True                       # worth splitting a long first sentence

    def __init__(self, device: str = C.TTS_DEVICE):
        from kokoro import KPipeline                 # imported here: loading takes a few seconds
        import torch
        self.pipe = KPipeline(lang_code=C.TTS_LANG, repo_id="hexgrad/Kokoro-82M")
        self.device = "cpu"
        if device == "mps" and torch.backends.mps.is_available() and self.pipe.model is not None:
            self.pipe.model.to("mps")
            self.device = "mps"
        self.sr = 24000
        self.label = f"Kokoro-82M {C.TTS_VOICE} on {self.device}"

    def synth(self, text: str) -> np.ndarray:
        chunks = [a.detach().cpu().numpy() if hasattr(a, "numpy") else np.asarray(a)
                  for _, _, a in self.pipe(text, voice=C.TTS_VOICE, speed=C.TTS_SPEED, split_pattern=None)]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)

    def warm(self):
        for _ in range(3 if self.device == "mps" else 1):   # the first MPS call compiles kernels (about 3.7 s)
            self.synth("Hello there.")


class PiperEngine:
    name = "piper"
    slow = False

    def __init__(self, voice: str = C.PIPER_VOICE, directory: Path = C.PIPER_DIR):
        from piper import PiperVoice, SynthesisConfig
        directory = Path(directory).expanduser()
        model = directory / f"{voice}.onnx"
        if not model.exists():                                  # first run: fetch the voice (about 60 MB)
            from piper.download_voices import download_voice
            directory.mkdir(parents=True, exist_ok=True)
            download_voice(voice, directory)
        self.voice = PiperVoice.load(model)
        self.cfg = SynthesisConfig(length_scale=C.PIPER_LENGTH_SCALE, volume=C.PIPER_VOLUME)
        self.sr = self.voice.config.sample_rate
        self.label = f"Piper {voice} on cpu"

    def synth(self, text: str) -> np.ndarray:
        chunks = [ch.audio_float_array for ch in self.voice.synthesize(text, self.cfg)]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)

    def warm(self):
        self.synth("Hello there.")


class SayEngine:
    name = "say"
    slow = False

    def __init__(self, voice: str = C.SAY_VOICE, rate: int = C.SAY_RATE):
        self.voice, self.rate = voice, rate
        self.sr = 22050
        self.label = f"macOS say {voice} at {rate} wpm"

    def synth(self, text: str) -> np.ndarray:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            subprocess.run(["say", "-v", self.voice, "-r", str(self.rate), "-o", path,
                            f"--data-format=LEI16@{self.sr}", text], check=True, capture_output=True, timeout=30)
            audio, sr = sf.read(path, dtype="float32")
        finally:
            Path(path).unlink(missing_ok=True)
        if audio.ndim > 1:
            audio = audio[:, 0]
        self.sr = sr
        return audio

    def warm(self):
        self.synth("Hello there.")


ENGINES = {"piper": PiperEngine, "kokoro": KokoroEngine, "say": SayEngine}


def make_engine(name: str | None = None):
    """The configured engine; if it cannot load (package or model missing) the next one in line is tried."""
    name = (name or C.TTS_ENGINE).lower()
    order = [name] + [n for n in ("piper", "kokoro", "say") if n != name]
    errors = []
    for n in order:
        try:
            return ENGINES[n](), errors
        except Exception as e:  # noqa: BLE001  (a missing package or model file: try the next engine)
            errors.append(f"{n}: {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError("no speech engine could be loaded: " + "; ".join(errors))


# ---------------------------------------------------------------------------------------------------------- speaker
class Speaker:
    def __init__(self, on_event: Callable[[str, str], None] = lambda kind, text: None, play: bool = True,
                 engine: str | None = None):
        t0 = time.time()
        self.engine, fallbacks = make_engine(engine)
        for f in fallbacks:
            on_event("say", f"(speech engine unavailable, {f})")
        self.engine.warm()                                    # the first call loads the voice / compiles kernels
        self.load_s = time.time() - t0
        self.sr = self.engine.sr
        self.label = self.engine.label
        self.on_event, self.play_audio = on_event, play
        self.text_q: "queue.Queue[tuple[str, threading.Event]]" = queue.Queue()
        self.audio_q: "queue.Queue" = queue.Queue()
        self.speaking = False
        self._synthesising = False                            # a reply is being turned into audio right now
        self.last_audio_end = 0.0
        self.first_audio_ms: list[float] = []                 # time to first audio of every reply, for the trace
        self._stream = None
        self._stream_sr = 0
        threading.Thread(target=self._synth_loop, daemon=True, name="tts-synth").start()
        threading.Thread(target=self._play_loop, daemon=True, name="tts-play").start()

    # the old name, used by scripts
    def synth(self, text: str) -> np.ndarray:
        return self.engine.synth(text)

    def say(self, text: str) -> threading.Event:
        """Queue text; returns an Event set when it has finished playing."""
        done = threading.Event()
        self.text_q.put((text, done))
        return done

    def busy(self) -> bool:
        return self.speaking or self._synthesising or not self.text_q.empty() or not self.audio_q.empty()

    # ------------------------------------------------------------------ synthesis thread
    def _synth_loop(self):
        while True:
            text, done = self.text_q.get()
            self._synthesising = True
            pieces = split_text(text, split_first=self.engine.slow)
            t0 = time.time()
            for i, piece in enumerate(pieces):
                try:
                    audio = self.engine.synth(piece)
                except Exception as e:  # noqa: BLE001
                    self.on_event("say", f"(speech synthesis failed: {e})")
                    audio = np.zeros(0, np.float32)
                if i == 0:
                    ms = 1000 * (time.time() - t0)
                    self.first_audio_ms.append(ms)
                    self.on_event("say", f"(voice ready in {ms:.0f} ms) {text}")
                self.audio_q.put((fade(audio, self.sr), done if i == len(pieces) - 1 else None))
            self._synthesising = False
            if not pieces:
                done.set()

    # ------------------------------------------------------------------ playback thread
    def _open_stream(self):
        import sounddevice as sd
        if self._stream is not None and self._stream_sr == self.sr:
            return self._stream
        self._close_stream()
        self._stream = sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32")
        self._stream.start()
        self._stream_sr = self.sr
        return self._stream

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
        self._stream = None

    def _play(self, audio: np.ndarray):
        """Write the chunk to the output stream; on a PortAudio error (the output device changed), reset PortAudio and
        reopen once. If that fails too, wait the chunk's length so the conversation keeps its timing."""
        import sounddevice as sd
        for attempt in (1, 2):
            try:
                stream = self._open_stream()
                stream.write(audio.reshape(-1, 1))
                return True
            except Exception as e:  # noqa: BLE001
                self._close_stream()
                if attempt == 1:
                    self.on_event("say", f"(audio output failed: {e}; reopening the output device)")
                    try:
                        sd._terminate()                       # forget the old device list
                        sd._initialize()
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    self.on_event("say", f"(audio output unavailable: {e})")
        time.sleep(len(audio) / self.sr)
        return False

    def _play_loop(self):
        while True:
            audio, done = self.audio_q.get()
            self.speaking = True
            if self.play_audio and len(audio):
                self._play(audio)
            else:
                time.sleep(len(audio) / self.sr)
            if self.audio_q.empty():
                if self._stream is not None and self.play_audio:
                    try:                                      # let the buffered tail play out
                        time.sleep(float(self._stream.latency or 0.0))
                    except Exception:  # noqa: BLE001
                        pass
                    if self.text_q.empty() and not self._synthesising:
                        self._close_stream()                  # a device change takes effect on the next reply
                self.last_audio_end = time.time()
                self.speaking = False
            else:
                self.last_audio_end = time.time()
            if done is not None:
                done.set()
