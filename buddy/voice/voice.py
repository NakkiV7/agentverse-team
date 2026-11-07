"""Owner: Voice and stage.

Hour 11 milestone: voice in and out.
  in:  Silero VAD on the laptop microphone -> whisper.cpp (small.en) -> text -> LLM.intent -> bus.intents
       Fallbacks the PDF asks for: a push-to-talk key and typed input (the ui already provides typed input).
  out: bus.say -> Kokoro-82M, sentence by sentence, slightly slower rate, a British voice; "urgent"
       lines interrupt whatever is playing.
Candidates: `pip install kokoro soundfile sounddevice silero-vad` and whisper.cpp via `brew install whisper-cpp`.
Until then, Speaker.run() prints what would be said so the rest of the system can be tested.
"""
from __future__ import annotations

from buddy.bus import Bus


class Speaker:
    def __init__(self, bus: Bus):
        self.bus = bus

    def run(self):
        while not self.bus.stop.is_set():
            s = self.bus.say.get(timeout=0.1)
            if s is not None:
                print(f"[say/{s.priority}] {s.text}", flush=True)


class Listener:
    """TODO Voice: VAD + whisper. Publishes Intents on bus.intents (via LLM.intent)."""

    def __init__(self, bus: Bus, llm=None):
        self.bus = bus
        self.llm = llm

    def run(self):
        self.bus.stop.wait()
