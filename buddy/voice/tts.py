"""Text to speech with Kokoro-82M (Apache 2.0), fully on the laptop.

Pipeline: text -> split into sentences -> Kokoro synthesises sentence 1 -> it starts playing while sentence 2 is
being synthesised. So Penny starts talking after one sentence's worth of synthesis, not after the whole reply.

Two threads: the synthesiser turns queued text into audio chunks, the player plays chunks in order. `speaking`
is True from the first chunk until the last one has played; the microphone listener ignores audio during that
time (plus a short tail) so Penny does not hear itself.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from typing import Callable, Optional

import numpy as np

from .. import config as C

SR = 24000
_SENT = re.compile(r"(?<=[.!?])\s+")


class Speaker:
    def __init__(self, on_event: Callable[[str, str], None] = lambda kind, text: None, play: bool = True):
        from kokoro import KPipeline                     # imported here: loading takes a few seconds
        t0 = time.time()
        self.pipe = KPipeline(lang_code=C.TTS_LANG, repo_id="hexgrad/Kokoro-82M")
        self.load_s = time.time() - t0
        self.on_event, self.play_audio = on_event, play
        self.text_q: "queue.Queue[tuple[str, threading.Event]]" = queue.Queue()
        self.audio_q: "queue.Queue" = queue.Queue()
        self.speaking = False
        self.last_audio_end = 0.0
        self.synth("Hello.")                              # warm up (first call compiles / loads the voice)
        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def synth(self, text: str) -> np.ndarray:
        chunks = [a.numpy() if hasattr(a, "numpy") else np.asarray(a)
                  for _, _, a in self.pipe(text, voice=C.TTS_VOICE, speed=C.TTS_SPEED, split_pattern=None)]
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def say(self, text: str) -> threading.Event:
        """Queue text; returns an Event set when it has finished playing."""
        done = threading.Event()
        self.text_q.put((text, done))
        return done

    def busy(self) -> bool:
        return self.speaking or not self.text_q.empty() or not self.audio_q.empty()

    def _synth_loop(self):
        while True:
            text, done = self.text_q.get()
            sentences = [s for s in _SENT.split(text.strip()) if s]
            for i, s in enumerate(sentences):
                t0 = time.time()
                audio = self.synth(s)
                if i == 0:
                    self.on_event("say", f"(voice ready in {1000 * (time.time() - t0):.0f} ms) {text}")
                self.audio_q.put((audio, done if i == len(sentences) - 1 else None))
            if not sentences:
                done.set()

    def _play_loop(self):
        import sounddevice as sd
        while True:
            audio, done = self.audio_q.get()
            self.speaking = True
            if self.play_audio and len(audio):
                try:
                    sd.play(audio, SR)
                    sd.wait()
                except Exception as e:                    # no audio device: keep going silently
                    self.on_event("say", f"(audio output unavailable: {e})")
                    time.sleep(len(audio) / SR)
            else:
                time.sleep(len(audio) / SR)
            self.last_audio_end = time.time()
            if self.audio_q.empty():
                self.speaking = False
            if done is not None:
                done.set()
