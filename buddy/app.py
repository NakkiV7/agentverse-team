"""Buddy, the whole system in one process.

Threads
  main        MuJoCo in real time: arm controller 50 Hz, physics 500 Hz, overhead camera 10 Hz, and the SAFETY REFLEX
              (depth + segmentation at camera rate: a hand over the paper freezes the arm). OpenGL stays on this
              thread (macOS).
  perception  every camera frame: find the sheet and track it (also while the arm draws), read the board with the
              mark classifier when the view is clear, draw the overlays for the demo page
  brain       conversation + game (buddy/agent/brain.py), with workers for setup and moves
  voice       Kokoro speaker; microphone listener (Silero VAD + Whisper)
  web         the demo page at http://localhost:8860
"""
from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import config as C
from .agent.brain import Brain
from .game import marks as M
from .motion.controller import ArmController, Job
from .motion.drawing import sheet_to_world
from .motion.ik import PenIK
from .perception import safety
from .perception.board import MarkReader, draw_board
from .perception.sheet import CameraModel, SheetPose, SheetTracker, Track, draw_overlay, find_sheet
from .sim.world import World

DEFAULT_SHEET = C.SHEET_POSE0


@dataclass
class Look:
    rgb: np.ndarray
    sheet: Optional[SheetPose]
    track: Optional[Track]
    t: float


@dataclass
class BoardSnapshot:
    cells: list
    t: float
    usable: bool              # arm parked, no hand over the paper, sheet visible and steady


class App:
    def __init__(self, voice: bool = True, mic: bool = True):
        C.RUNTIME.mkdir(exist_ok=True)
        self.t0 = time.time()
        self._lock = threading.Lock()
        self._trace: deque = deque(maxlen=500)
        self._transcript: deque = deque(maxlen=80)
        self.trace("system", "building the simulated world (SO-101 + pen, A4 sheet, overhead RGB-D camera)")
        self.world = World()
        self.ik = PenIK(self.world.model, self.world.pen.tip_body, self.world.pen.axis_body, World.ARM)
        self.ik_planner = PenIK(self.world.model, self.world.pen.tip_body, self.world.pen.axis_body, World.ARM)
        self.ctl = ArmController(self.ik, self.world.arm_q(), log=self._ctl_log)
        self.cam = CameraModel(*self.world.camera_model())
        self.cam_d = safety.depth_camera(self.cam)
        self.tracker = SheetTracker()
        self.reader = MarkReader()
        self.trace("system", "mark classifier loaded (CNN trained on simulator images, assets/models/mark_cnn.pt)")
        self._frame, self._frame_t, self._last_sheet = None, -1.0, None
        self.board_snap: Optional[BoardSnapshot] = None
        self.overhead_jpg = self.scene_jpg = self.sheet_jpg = b""
        self.overlay_lines, self.overlay_cov = None, None
        self.scene_actions: "queue.Queue" = queue.Queue()
        self.running = True
        self.sim_time = 0.0
        self.intrusion = safety.Intrusion(False, 0, 0.0)
        self._clear_since = 0.0
        self._said_adapt = None
        self.speaker = None
        if voice:
            self.trace("system", "loading the voice (Kokoro-82M)")
            from .voice.tts import Speaker
            self.speaker = Speaker(on_event=self.trace)
            self.trace("system", f"voice ready ({self.speaker.load_s:.1f} s)")
        self.brain = Brain(self)
        self.listener = None
        if mic:
            from .voice.stt import Listener
            self.listener = Listener(on_text=lambda t, dt: self.brain.hear(t, "mic", dt),
                                     is_speaking=self._speaking_recently, on_event=self._mic_event)
            self.listener.start()
            self.trace("system", "microphone on (Silero VAD + Whisper small.en); say 'Buddy' to start")
        threading.Thread(target=self._warm_llm, daemon=True).start()
        threading.Thread(target=self._perception_loop, daemon=True, name="perception").start()

    # ------------------------------------------------------------------ shared state
    def trace(self, stage: str, msg: str):
        t = time.time() - self.t0
        with self._lock:
            self._trace.append({"t": round(t, 1), "stage": stage, "msg": msg})
        print(f"[{t:6.1f}s] {stage.upper():10} {msg}", flush=True)

    def _ctl_log(self, stage, msg):
        self.trace(stage, msg)
        job = self.ctl.current
        if stage == "adapt" and "moved" in msg and job is not None and self._said_adapt is not job:
            self._said_adapt = job
            self.brain.say("Oh, the paper moved. Let me find our board again.")

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

    def _warm_llm(self):
        from .agent.llm import warm_up
        dt = warm_up(self.brain.conv)
        self.trace("system", f"language model ready (Gemma 3 4B via Ollama, warmed up in {dt:.1f} s)")
        try:
            from .agent.teach import TEACH_MODEL
            from .agent.teach import warm_up as warm_teach
            dt = warm_teach()
            self.trace("system", f"teaching model ready ({TEACH_MODEL} via Ollama, warmed up in {dt:.1f} s)")
        except Exception as e:  # noqa: BLE001
            self.trace("system", f"teaching model not available ({e}); teaching a new game will not work")

    # ------------------------------------------------------------------ services for the brain
    def draw(self, name: str, strokes_mm, on_stroke=None, home: bool = True) -> Job:
        return self.ctl.submit(Job(name, strokes_mm=[np.asarray(s, float) for s in strokes_mm],
                                   sheet=lambda: self.tracker.track, home=home, on_stroke=on_stroke))

    def view_clear(self) -> bool:
        return (self.ctl.at_home and not self.ctl.busy and not self.intrusion.present
                and not self.world.busy_animating())

    def look(self, timeout: float = 10.0) -> Optional[Look]:
        """A camera frame taken after this call with the arm parked and nothing over the paper."""
        t_req = self.sim_time
        t_end = time.time() + timeout
        while time.time() < t_end:
            with self._lock:
                rgb, t, sheet = self._frame, self._frame_t, self._last_sheet
            if rgb is not None and t > t_req + 0.15 and self.view_clear():
                tr = self.tracker.track
                if sheet is None or sheet.cut_off or (tr is not None and tr.stable):
                    return Look(rgb, sheet, tr, t)
            time.sleep(0.05)
        with self._lock:
            return Look(self._frame, self._last_sheet, self.tracker.track, self._frame_t) if self._frame is not None else None

    def board_snapshot(self, fresh: bool = False, timeout: float = 0.0) -> Optional[BoardSnapshot]:
        if not fresh:
            return self.board_snap
        t_req = self.sim_time
        t_end = time.time() + timeout
        while time.time() < t_end:
            s = self.board_snap
            if s is not None and s.usable and s.t > t_req + 0.15:
                return s
            time.sleep(0.05)
        return None

    def grid_reachable(self, pose) -> tuple[bool, str]:
        h = C.GRID_SIZE_MM / 2
        corners = sheet_to_world(pose, [(x, y) for x in (-h, h) for y in (-h, h)])
        seed = np.array([0.0, 0.0, 0.5, 1.2, 0.0])
        worst = 0.0
        for x, y in corners:
            for z in (C.SHEET_THICK, C.SHEET_THICK + C.HOVER_MM / 1000):
                r = self.ik_planner.solve([x, y, z], seed)
                worst = max(worst, r.tilt_deg)
                if not r.ok:
                    return False, (f"grid corner at ({100 * x:.1f}, {100 * y:.1f}) cm, "
                                   f"{100 * math.hypot(x - 0.039, y):.1f} cm from my shoulder, is out of reach")
        return True, f"all 4 grid corners reachable with the pen upright (worst lean {worst:.1f} deg)"

    def set_overlay_lines(self, lines, cov):
        self.overlay_lines, self.overlay_cov = lines, cov

    # ------------------------------------------------------------------ demo page state
    def state(self) -> dict:
        tr = self.tracker.track
        b, snap = self.brain, self.board_snap
        with self._lock:
            trace, transcript = list(self._trace)[-150:], list(self._transcript)
        return {
            "state": b.state, "trace": trace, "transcript": transcript,
            "speaking": bool(self.speaker and self.speaker.busy()),
            "mic": {"available": self.listener is not None,
                    "enabled": bool(self.listener and self.listener.enabled),
                    "level": round(self.listener.level, 4) if self.listener else 0.0},
            "sheet": None if tr is None else {"x_cm": round(100 * tr.pose[0], 1), "y_cm": round(100 * tr.pose[1], 1) + 0.0,
                                              "rot_deg": round(math.degrees(tr.pose[2]), 1),
                                              "conf": round(tr.confidence, 2), "visible": tr.visible,
                                              "stable": tr.stable},
            "board": b.board, "player": b.player_sym, "robot": b.robot_sym,
            "game": {"name": b.game_name, "rules": b.rules_text(), "taught": b.spec is not None,
                     "legal": b.legal_cells() if b.state in ("PLAYER_TURN", "CONFIRM_MARK") else []},
            "cells": None if snap is None else [{"mark": c.mark, "conf": round(c.confidence, 3)} for c in snap.cells],
            "safety": {"frozen": self.ctl.paused, "pixels": self.intrusion.pixels,
                       "height_cm": round(self.intrusion.max_height_cm, 1)},
            "strength": round(b.difficulty.strength, 2), "games": b.games,
            "touch_N": round(self.world.touch(), 2),
            "arm": "frozen" if self.ctl.paused else ("home" if self.ctl.at_home and not self.ctl.busy else "working"),
            "sim_time": round(self.sim_time, 1),
        }

    # ------------------------------------------------------------------ perception (own thread)
    def _perception_loop(self):
        last = -1.0
        while self.running:
            with self._lock:
                rgb, t = self._frame, self._frame_t
            if rgb is None or t == last:
                time.sleep(0.01)
                continue
            last = t
            sheet = find_sheet(rgb, self.cam)
            with self._lock:
                self._last_sheet = sheet
            tr = self.tracker.update(sheet, t)
            clear = self.view_clear()
            cells, rect = None, None
            if tr is not None and tr.visible:
                cells, rect = self.reader.read(rgb, self.cam, tr.pose)
                self.board_snap = BoardSnapshot(cells, t, clear and tr.stable)
            note = ("SAFETY: hand over the paper, arm frozen" if self.ctl.paused else
                    "SEE: clear view" if clear else "arm or hand over the paper: tracking the sheet")
            show = SheetPose(*tr.pose, tr.confidence, sheet.size_mm if sheet else (0, 0),
                             sheet.corners_px if sheet else np.zeros((4, 2))) if tr is not None else sheet
            overlay = draw_overlay(rgb, self.cam, show, self.overlay_lines, self.overlay_cov, note)
            self.overhead_jpg = cv2.imencode(".jpg", cv2.resize(overlay, (960, 720), interpolation=cv2.INTER_AREA),
                                             [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            if rect is not None:
                readable = clear and tr.stable
                img = draw_board(rect, cells if readable else None,
                                 "" if readable else "arm or hand over the paper: reading paused")
                self.sheet_jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    # ------------------------------------------------------------------ safety reflex (main thread, camera rate)
    def _reflex(self):
        tr = self.tracker.track
        if tr is None:
            return
        depth, robot = self.world.render_depth()
        self.intrusion = safety.check(depth, robot, self.cam_d, tr.pose)
        if self.intrusion.present:
            self._clear_since = self.sim_time
            if not self.ctl.paused:
                self.ctl.paused = True
                if self.ctl.busy:
                    self.trace("safety", f"something is over the paper ({self.intrusion.pixels} depth pixels, up to "
                                         f"{self.intrusion.max_height_cm:.0f} cm high): arm frozen")
                    self.brain.say("I'll wait until your hand is out of the way.")
        elif self.ctl.paused and self.sim_time - self._clear_since >= safety.CLEAR_S:
            self.ctl.paused = False
            if self.ctl.busy:
                self.trace("safety", f"the drawing area has been clear for {safety.CLEAR_S:.0f} s: carrying on")

    # ------------------------------------------------------------------ scene changes (buttons on the page)
    def _scene_action(self, action, arg=None):
        w = self.world
        x, y, yaw = w.sheet_pose()
        if action == "new_sheet":
            w.clear_ink()
            w.move_sheet(*DEFAULT_SHEET)
            self.overlay_lines, self.overlay_cov = None, None
        elif action == "slide":
            w.slide_sheet(0.02, -0.025, 0.0)
        elif action == "rotate":
            w.slide_sheet(0.0, 0.0, 0.18)
        elif action == "far":
            w.slide_sheet(0.33 - x, 0.0, 0.0)
        elif action == "remove":
            w.move_sheet(1.5, 0.0, yaw)
        elif action == "back":
            w.slide_sheet(DEFAULT_SHEET[0] - x, DEFAULT_SHEET[1] - y, DEFAULT_SHEET[2] - yaw)
        elif action == "reach":
            w.reach_in(4.0)
        elif action == "player_mark" and arg is not None:
            sym = self.brain.player_sym or "X"
            w.hand_strokes(M.hand_mark(sym, int(arg), np.random.default_rng()))
            action = f"the player draws a {'cross' if sym == 'X' else 'nought'}"
        elif action == "strokes" and arg:
            w.hand_strokes(arg)
            action = "the player draws on the paper"
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
                self._scene_action(*self.scene_actions.get())
            self.world.command(self.ctl.tick(self.world.touch()))
            self.world.step(spt)
            self.sim_time += dt
            if self.sim_time >= next_cam:
                rgb = self.world.render_overhead()
                with self._lock:
                    self._frame, self._frame_t = rgb, self.sim_time
                self._reflex()
                next_cam += 1.0 / C.CAMERA_HZ
            if self.sim_time >= next_scene:
                scene = cv2.cvtColor(self.world.render_scene(), cv2.COLOR_RGB2BGR)
                self.scene_jpg = cv2.imencode(".jpg", scene, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                next_scene += 0.1
            if until is not None and until(self):
                break
            if not realtime:
                time.sleep(0.0005)                  # let the other threads breathe
                continue
            ahead = (start + self.sim_time) - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -1.0:
                start = time.perf_counter() - self.sim_time

    def close(self):
        self.running = False
        self.world.close()
