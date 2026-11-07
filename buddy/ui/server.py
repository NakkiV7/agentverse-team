"""Owner: Voice and stage. Local web page for the demo and for the judges' stream.

Plan: a tiny HTTP server (stdlib http.server or aiohttp) serving
  GET  /            the page (camera MJPEG + transcript + state + "why this move")
  GET  /stream      MJPEG of Percept.overlay (fallback: SensorFrame.rgb)
  POST /say         typed input -> LLM.intent -> bus.intents   (typed fallback for noisy rooms)
  POST /scene       buttons that call World.move_sheet / set_hand / add_human_mark for live demos
  WS   /events      transcript lines, state changes, game events
Until then this is a stub that keeps the thread alive.
"""
from __future__ import annotations

from buddy.bus import Bus


class UI:
    def __init__(self, bus: Bus, world=None, llm=None, port: int = 8000):
        self.bus, self.world, self.llm, self.port = bus, world, llm, port

    def run(self):
        self.bus.stop.wait()
