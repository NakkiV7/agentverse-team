"""Speech to text, fully on the laptop: Silero VAD finds where speech starts and ends, whisper.cpp transcribes it,
and while the player is still speaking a small model writes down what has been heard so far.

microphone (16 kHz) -> Silero VAD in 32 ms frames -> one utterance (speech + 0.6 s of silence)
   while it is active: every STT_PARTIAL_S the audio so far -> whisper.cpp base.en -> `partial` (for the page)
   when it ends:       the whole utterance -> whisper.cpp small.en -> text -> the agent

whisper.cpp stays resident in the process through pywhispercpp (Metal), so a transcription costs the model's run,
not a whisper-cli start plus model load (small.en: about 250 ms for a 4 s utterance instead of 650 ms and more).
If pywhispercpp is missing the old whisper-cli path is used for the final text and there are no partials.

While Penny is speaking (and for 0.4 s after), microphone audio is dropped so it never hears itself.

Listener.hearing() is the page's view: {"active": bool, "partial": str, "level": float}.
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

SR = 16000
FRAME = 512                       # Silero VAD frame at 16 kHz (32 ms)
_NOISE = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")      # [BLANK_AUDIO], (typing), *music*


def clean(text: str) -> str:
    text = _NOISE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------------------------------------- transcribers
class WhisperCpp:
    """whisper.cpp resident in the process (pywhispercpp). One instance per model; the first costs about 7 s
    (Metal shader compilation), the next ones a fraction of a second."""

    def __init__(self, model: Path = C.WHISPER_MODEL, threads: int = 4):
        from pywhispercpp.model import Model
        t0 = time.time()
        self.model = Model(str(model), n_threads=threads, print_realtime=False, print_progress=False,
                           redirect_whispercpp_logs_to=None)
        self.load_s = time.time() - t0
        self.name = Path(model).stem.replace("ggml-", "")

    def transcribe(self, audio16k: np.ndarray) -> str:
        segs = self.model.transcribe(np.ascontiguousarray(audio16k, dtype=np.float32), language="en",
                                     no_context=True, suppress_blank=True, suppress_nst=True)
        return clean(" ".join(s.text for s in segs))


class WhisperCli:
    """The old path: whisper-cli started per utterance (model loaded every time)."""

    def __init__(self, model: Path = C.WHISPER_MODEL):
        self.model, self.load_s = model, 0.0
        self.name = Path(model).stem.replace("ggml-", "") + " (whisper-cli)"

    def transcribe(self, audio16k: np.ndarray) -> str:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio16k, SR, subtype="PCM_16")
            out = subprocess.run([C.WHISPER_CLI, "-m", str(self.model), "-f", f.name, "-nt", "-np", "-l", "en",
                                  "-t", "4"], capture_output=True, text=True, timeout=60)
        Path(f.name).unlink(missing_ok=True)
        return clean(" ".join(line.strip() for line in out.stdout.splitlines() if line.strip()))


def load_transcriber(model: Path = C.WHISPER_MODEL):
    try:
        return WhisperCpp(model)
    except Exception:  # noqa: BLE001  (pywhispercpp not installed)
        return WhisperCli(model)


_default: Optional[object] = None


def transcribe(audio16k: np.ndarray) -> tuple[str, float]:
    """One-off helper (scripts): whisper on a mono float32 16 kHz signal. Returns (text, seconds taken)."""
    global _default
    if _default is None:
        _default = load_transcriber()
    t0 = time.time()
    return _default.transcribe(audio16k), time.time() - t0


# ------------------------------------------------------------------------------------------------- the listener
class Listener:
    def __init__(self, on_text: Callable[[str, float], None], is_speaking: Callable[[], float],
                 on_event: Callable[[str, str], None] = lambda k, t: None, final=None, partial=None,
                 on_partial: Callable[[str, float], None] = lambda t, dt: None):
        from silero_vad import VADIterator, load_silero_vad
        self.model = load_silero_vad()
        self.vad = VADIterator(self.model, sampling_rate=SR, threshold=0.5, min_silence_duration_ms=600,
                               speech_pad_ms=200)
        self.on_text, self.is_speaking, self.on_event, self.on_partial = on_text, is_speaking, on_event, on_partial
        self.final = final if final is not None else load_transcriber(C.WHISPER_MODEL)
        if partial is None and isinstance(self.final, WhisperCpp) and Path(C.STT_PARTIAL_MODEL).exists():
            partial = WhisperCpp(C.STT_PARTIAL_MODEL)
        self.partial_model = partial                           # None: no live text (whisper-cli fallback)
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.enabled = True
        self.level = 0.0
        self.active = False                                    # speech started, final text not yet delivered
        self.partial = ""                                      # the latest partial transcription
        self.transcribing = False                              # the final transcription is running
        self._speech: list[np.ndarray] = []
        self._lock = threading.Lock()                          # _speech, partial, active
        self._wlock = threading.Lock()                         # one whisper run at a time (shared Metal queue)
        self._wake = threading.Event()
        self.stream = None

    @property
    def label(self) -> str:
        p = f", live text {self.partial_model.name} every {C.STT_PARTIAL_S:.1f} s" if self.partial_model else ""
        return f"Silero VAD + whisper.cpp {self.final.name}{p}"

    def hearing(self) -> dict:
        """The page's view of the ears: active while an utterance is being spoken or transcribed, the partial
        text so far (greyed in the input box, replaced by the final text when it reaches the transcript)."""
        return {"active": self.active, "partial": self.partial, "level": round(self.level, 4)}

    def start(self):
        import sounddevice as sd
        self.stream = sd.InputStream(samplerate=SR, channels=1, blocksize=FRAME, dtype="float32",
                                     callback=lambda data, n, t, s: self.q.put(data[:, 0].copy()))
        self.stream.start()
        self.start_threads()

    def start_threads(self):
        threading.Thread(target=self._loop, daemon=True, name="stt-vad").start()
        if self.partial_model is not None:
            threading.Thread(target=self._partial_loop, daemon=True, name="stt-partial").start()

    def feed(self, frame: np.ndarray):
        """Audio from somewhere other than the microphone (tests, voice_check): 512-sample float32 frames."""
        self.q.put(frame)

    # ------------------------------------------------------------------ VAD + final transcription
    def _reset(self):
        with self._lock:
            self._speech, self.active, self.partial = [], False, ""
        self.vad.reset_states()

    def _loop(self):
        import torch
        while True:
            frame = self.q.get()
            self.level = float(np.sqrt(np.mean(frame ** 2)))
            if not self.enabled or self.is_speaking():
                if self.active:
                    self._reset()
                else:
                    self.vad.reset_states()
                continue
            ev = self.vad(torch.from_numpy(frame), return_seconds=False)
            if ev and "start" in ev:
                with self._lock:
                    self.active, self.partial, self._speech = True, "", []
                self.on_event("hear", "(speech started)")
                self._wake.set()
            if self.active:
                with self._lock:
                    self._speech.append(frame)
            if ev and "end" in ev and self.active:
                with self._lock:
                    audio = np.concatenate(self._speech) if self._speech else np.zeros(0, np.float32)
                    self._speech = []
                if len(audio) > 0.3 * SR:
                    self.transcribing = True
                    t0 = time.time()
                    with self._wlock:
                        text = self.final.transcribe(audio)
                    dt = time.time() - t0
                    self.transcribing = False
                    with self._lock:
                        self.active, self.partial = False, ""
                    if text:
                        self.on_text(text, dt)
                else:
                    with self._lock:
                        self.active, self.partial = False, ""

    # ------------------------------------------------------------------ partial transcription (own thread)
    def _partial_loop(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            while True:
                time.sleep(C.STT_PARTIAL_S)
                with self._lock:
                    if not self.active or self.transcribing or not self._speech:
                        break
                    audio = np.concatenate(self._speech)
                if len(audio) < C.STT_MIN_PARTIAL_S * SR:
                    continue
                t0 = time.time()
                with self._wlock:
                    if not self.active or self.transcribing:  # the utterance ended while we waited
                        break
                    text = self.partial_model.transcribe(audio)
                with self._lock:
                    if self.active and not self.transcribing and text:
                        self.partial = text
                        changed = True
                    else:
                        changed = False
                if changed:
                    self.on_partial(text, time.time() - t0)
