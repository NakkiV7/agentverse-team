"""Buddy, the whole system in one process.

Threads
  main        the MuJoCo simulation in real time: arm controller at 50 Hz, physics at 500 Hz, overhead camera at
              10 Hz, scene view at 10 Hz (OpenGL must stay on the main thread on macOS)
  perception  reads the latest camera frame at 5 Hz, finds the sheet, draws the overlay shown on the demo page
  brain       conversation state machine + game setup behaviour (buddy/agent/brain.py)
  voice       Kokoro speaker (synthesis + playback) and the microphone listener (Silero VAD + Whisper)
  web         the demo page at http://localhost:8860
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque

import cv2
import numpy as np

from . import config as C
from .agent.brain import Brain
from .motion.controller import ArmController
from .motion.ik import PenIK
from .perception.sheet import CameraModel, draw_overlay, find_sheet
from .sim.world import World

DEFAULT_SHEET = C.SHEET_POSE0


class App:
    def __init__(self, voice: bool = True, mic: bool = True):
        C.RUNTIME.mkdir(exist_ok=True)
        self.t0 = time.time()
        self._lock = threading.Lock()
        self._trace: deque = deque(maxlen=400)
        self._transcript: deque = deque(maxlen=60)
        self.trace("system", "building the simulated world (SO-101 + pen, A4 sheet, overhead RGB-D camera)")
        self.world = World()
        self.ik = PenIK(self.world.model, self.world.pen.tip_body, self.world.pen.axis_body, World.ARM)
        self.ik_planner = PenIK(self.world.model, self.world.pen.tip_body, self.world.pen.axis_body, World.ARM)
        self.ctl = ArmController(self.ik, self.world.arm_q(), log=self.trace)
        self.cam = CameraModel(*self.world.camera_model())
        self._frame, self._frame_t = None, 0.0
        self.overhead_jpg = b""
        self.scene_jpg = b""
        self.plan_pose, self.plan_lines, self.coverage = None, None, None
        self.sheet_seen = None
        self.scene_actions: "queue.Queue[str]" = queue.Queue()
        self.running = True
        self.sim_time = 0.0
        self.speaker = None
        if voice:
            self.trace("system", "loading the voice (Kokoro-82M)")
            from .voice.tts import Speaker
            self.speaker = Speaker(on_event=self.trace)
            self.trace("system", f"voice ready ({self.speaker.load_s:.1f} s)")
        self.brain = Brain(self)
        self.listener = None
        if mic:
            try:
                from .voice.stt import Listener
                self.listener = Listener(on_text=lambda t, dt: self.brain.hear(t, "mic", dt),
                                         is_speaking=self._speaking_recently, on_event=self._mic_event)
                self.listener.start()
                self.trace("system", "microphone on (Silero VAD + Whisper small.en); say 'Buddy' to start")
            except Exception as e:
                self.listener = None
                self.trace("system", f"microphone unavailable ({e}); type in the box instead")
        threading.Thread(target=self._warm_llm, daemon=True).start()
        threading.Thread(target=self._perception_loop, daemon=True, name="perception").start()

    # ------------------------------------------------------------------ shared state
    def trace(self, stage: str, msg: str):
        t = time.time() - self.t0
        with self._lock:
            self._trace.append({"t": round(t, 1), "stage": stage, "msg": msg})
        print(f"[{t:6.1f}s] {stage.upper():10} {msg}", flush=True)

    def _mic_event(self, stage, msg):
        if msg != "(speech started)":
            self.trace(stage, msg)

    def transcript_add(self, who: str, text: str):
        with self._lock:
            self._transcript.append({"t": round(time.time() - self.t0, 1), "who": who, "text": text})

    def say(self, text: str):
        if self.speaker is not None:
            return self.speaker.say(text)
        self.trace("say", text)
        return None

    def _speaking_recently(self) -> bool:
        return self.speaker is not None and (self.speaker.busy() or time.time() - self.speaker.last_audio_end < 0.4)

    def latest_frame(self):
        with self._lock:
            return self._frame, self._frame_t

    def set_plan(self, pose, lines):
        self.plan_pose, self.plan_lines, self.coverage = pose, lines, None

    def set_coverage(self, cov):
        self.coverage = cov

    def _warm_llm(self):
        from .agent.llm import warm_up
        try:
            dt = warm_up()
            self.trace("system", f"language model ready (Gemma 3 4B via Ollama, loaded in {dt:.1f} s)")
        except Exception as e:
            self.trace("system", f"language model unavailable ({e}); phrase rules only")

    def state(self) -> dict:
        s = self.sheet_seen
        with self._lock:
            trace, transcript = list(self._trace)[-120:], list(self._transcript)
        return {
            "state": self.brain.state, "trace": trace, "transcript": transcript,
            "speaking": bool(self.speaker and self.speaker.busy()),
            "mic": {"available": self.listener is not None,
                    "enabled": bool(self.listener and self.listener.enabled),
                    "level": round(self.listener.level, 4) if self.listener else 0.0},
            "sheet": None if s is None else {"x_cm": round(100 * s.x, 1), "y_cm": round(100 * s.y, 1),
                                             "rot_deg": round(float(np.degrees(s.yaw)), 1),
                                             "conf": round(s.confidence, 2)},
            "coverage": self.coverage, "touch_N": round(self.world.touch(), 2),
            "arm": "home" if self.ctl.at_home and not self.ctl.busy else "working",
            "sim_time": round(self.sim_time, 1),
        }

    # ------------------------------------------------------------------ perception for the demo page
    def _perception_loop(self):
        last = 0.0
        while self.running:
            rgb, t = self.latest_frame()
            if rgb is not None and t != last:
                last = t
                if self.ctl.at_home and not self.ctl.busy:
                    sheet = find_sheet(rgb, self.cam)
                    self.sheet_seen = sheet
                    note = "SEE: arm parked, camera has a clear view" if sheet else "SEE: no sheet in view"
                else:
                    sheet = self.sheet_seen
                    note = "arm over the sheet: showing the last sheet estimate"
                overlay = draw_overlay(rgb, self.cam, sheet, self.plan_lines if self.plan_pose else None,
                                       self.coverage, note)
                small = cv2.resize(overlay, (960, 720), interpolation=cv2.INTER_AREA)
                self.overhead_jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            time.sleep(0.2)

    # ------------------------------------------------------------------ scene changes (buttons on the page)
    def _scene_action(self, action: str):
        x, y, yaw = self.world.sheet_pose()
        if action == "new_sheet":
            self.world.clear_ink()
            self.world.move_sheet(*DEFAULT_SHEET)
            self.plan_pose, self.plan_lines, self.coverage = None, None, None
        elif action == "slide":
            self.world.move_sheet(x + 0.025, y - 0.02, yaw)
        elif action == "rotate":
            self.world.move_sheet(x, y, yaw + 0.2)
        elif action == "far":
            self.world.move_sheet(0.33, y, yaw)                # still in view, but the far grid corners are out of reach
        elif action == "remove":
            self.world.move_sheet(1.5, 0.0, yaw)
        elif action == "back":
            self.world.move_sheet(*DEFAULT_SHEET)
        else:
            return
        self.trace("system", f"scene change: {action.replace('_', ' ')}")

    # ------------------------------------------------------------------ main loop (main thread)
    def run(self, until=None, realtime: bool = True):
        spt = int(round(1.0 / C.CONTROL_HZ / C.TIMESTEP))
        dt = 1.0 / C.CONTROL_HZ
        next_cam = next_scene = 0.0
        start = time.perf_counter()
        while self.running:
            while not self.scene_actions.empty():
                self._scene_action(self.scene_actions.get())
            self.world.command(self.ctl.tick(self.world.touch()))
            self.world.step(spt)
            self.sim_time += dt
            if self.sim_time >= next_cam:
                rgb, _depth = self.world.render_overhead()
                with self._lock:
                    self._frame, self._frame_t = rgb, time.time()
                next_cam += 1.0 / C.CAMERA_HZ
            if self.sim_time >= next_scene:
                scene = cv2.cvtColor(self.world.render_scene(), cv2.COLOR_RGB2BGR)
                self.scene_jpg = cv2.imencode(".jpg", scene, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                next_scene += 0.1
            if until is not None and until(self):
                break
            ahead = (start + self.sim_time) - time.perf_counter()
            if not realtime:
                continue
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -1.0:                           # fell behind (slow machine): do not try to catch up
                start = time.perf_counter() - self.sim_time

    def close(self):
        self.running = False
        self.world.close()
