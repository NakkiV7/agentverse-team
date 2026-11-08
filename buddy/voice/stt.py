"""Speech to text, fully on the laptop: Silero VAD finds where speech starts and ends, whisper.cpp transcribes it.

microphone (16 kHz) -> Silero VAD in 32 ms frames -> one utterance (speech + 0.6 s of silence) -> whisper.cpp
(small.en) -> text -> the agent.

While Penny is speaking (and for 0.4 s after), microphone audio is dropped so it never hears itself.
"""
from __future__ import annotations

import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .. import config as C

SR = 16000
FRAME = 512                       # Silero VAD frame at 16 kHz (32 ms)


def transcribe(audio16k: np.ndarray) -> tuple[str, float]:
    """whisper.cpp on a mono float32 16 kHz signal. Returns (text, seconds taken)."""
    import soundfile as sf
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio16k, SR, subtype="PCM_16")
        out = subprocess.run([C.WHISPER_CLI, "-m", str(C.WHISPER_MODEL), "-f", f.name, "-nt", "-np", "-l", "en",
                              "-t", "4"], capture_output=True, text=True, timeout=60)
    Path(f.name).unlink(missing_ok=True)
    text = " ".join(line.strip() for line in out.stdout.splitlines() if line.strip())
    text = text.replace("[BLANK_AUDIO]", "").strip()
    return text, time.time() - t0


class Listener:
    def __init__(self, on_text: Callable[[str, float], None], is_speaking: Callable[[], float],
                 on_event: Callable[[str, str], None] = lambda k, t: None):
        from silero_vad import VADIterator, load_silero_vad
        self.model = load_silero_vad()
        self.vad = VADIterator(self.model, sampling_rate=SR, threshold=0.5, min_silence_duration_ms=600,
                               speech_pad_ms=200)
        self.on_text, self.is_speaking, self.on_event = on_text, is_speaking, on_event
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.enabled = True
        self.level = 0.0

    def start(self):
        import sounddevice as sd
        self.stream = sd.InputStream(samplerate=SR, channels=1, blocksize=FRAME, dtype="float32",
                                     callback=lambda data, n, t, s: self.q.put(data[:, 0].copy()))
        self.stream.start()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        import torch
        speech, active = [], False
        while True:
            frame = self.q.get()
            self.level = float(np.sqrt(np.mean(frame ** 2)))
            if not self.enabled or self.is_speaking():
                speech, active = [], False
                self.vad.reset_states()
                continue
            ev = self.vad(torch.from_numpy(frame), return_seconds=False)
            if ev and "start" in ev:
                active = True
                self.on_event("hear", "(speech started)")
            if active:
                speech.append(frame)
            if ev and "end" in ev and active:
                active = False
                audio = np.concatenate(speech)
                speech = []
                if len(audio) > 0.3 * SR:
                    text, dt = transcribe(audio)
                    if text:
                        self.on_text(text, dt)
