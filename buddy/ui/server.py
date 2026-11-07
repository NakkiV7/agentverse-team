"""Owner: Voice and stage. Local web page for the demo and for the judges' stream.

Delivers:
  Hour 1: stream page skeleton, MJPEG of Percept.overlay, transcript, state, typed input -> bus.intents.
  Hour 3: scene buttons calling World.move_sheet, World.set_hand, World.add_human_mark.
  Hour 7: "why this move" panel, game events log, FPS and latency readouts (perception ms, Gemma s).
"""
from __future__ import annotations

import io
import json
import math
import os
import queue
import re
import socketserver
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import numpy as np

from buddy.bus import Bus
from buddy.contracts import Intent, Say

# Try importing image encoders with graceful fallbacks
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

INDEX_HTML_PATH = Path(__file__).resolve().parent / "index.html"


def _encode_jpeg(img_bgr_or_rgb: np.ndarray, is_rgb: bool = False) -> bytes:
    """Encode a numpy image to JPEG bytes cleanly."""
    if _HAS_CV2:
        img_bgr = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2BGR) if is_rgb else img_bgr_or_rgb
        ret, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            return buf.tobytes()
    elif _HAS_PIL:
        if is_rgb:
            im = Image.fromarray(img_bgr_or_rgb)
        else:
            im = Image.fromarray(img_bgr_or_rgb[:, :, ::-1])
        bio = io.BytesIO()
        im.save(bio, format="JPEG", quality=80)
        return bio.getvalue()

    # Minimal 1x1 black JPEG fallback if no imaging library is installed
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"


def _create_placeholder_frame(status_text: str = "Awaiting Camera Feed...") -> bytes:
    """Generate a clean dark placeholder frame."""
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    arr[:] = (26, 26, 32)
    if _HAS_CV2:
        cv2.putText(arr, "BUDDY CAM STREAM", (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 160, 255), 2)
        cv2.putText(arr, status_text, (190, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        _, buf = cv2.imencode(".jpg", arr)
        return buf.tobytes()
    return _encode_jpeg(arr, is_rgb=False)


class UI:
    """Owner: Voice and stage. Serves the stream page, API state, scene buttons, and typed fallback."""

    def __init__(self, bus: Bus, world=None, llm=None, port: int = 8000):
        self.bus = bus
        self.world = world
        self.llm = llm
        self.port = port
        self.server: Optional[ThreadingHTTPServer] = None

        # Thread-safe storage for transcript and events
        self._lock = threading.Lock()
        self.transcript_history: list[str] = []
        self.game_events_history: list[dict[str, Any]] = []

        # Performance and state telemetry
        self.fps: float = 0.0
        self.last_perception_ms: float = 0.0
        self.last_gemma_s: float = 0.0
        self.current_agent_state: str = "IDLE"
        self.latest_why_this_move: str = "Awaiting game move..."
        self.ptt_active: bool = False

        # Frame counter for FPS tracking
        self._frame_count: int = 0
        self._last_fps_time: float = time.monotonic()
        self._last_percept_t: float = -1.0

        # Hook into bus.transcript so we never miss items drained by other threads
        self._install_bus_hooks()

    def _install_bus_hooks(self) -> None:
        """Mirror bus.transcript and bus.game_events puts into UI history."""
        orig_transcript_put = self.bus.transcript.put

        def hooked_transcript_put(item: str) -> None:
            with self._lock:
                self.transcript_history.append(item)
                if len(self.transcript_history) > 300:
                    self.transcript_history.pop(0)
            orig_transcript_put(item)

        self.bus.transcript.put = hooked_transcript_put

        orig_events_put = self.bus.game_events.put

        def hooked_events_put(ev: Any) -> None:
            with self._lock:
                ev_dict = {
                    "kind": getattr(ev, "kind", str(ev)),
                    "cell": getattr(ev, "cell", None),
                    "winner": getattr(ev, "winner", None),
                    "text": getattr(ev, "text", ""),
                    "time": time.strftime("%H:%M:%S"),
                }
                self.game_events_history.append(ev_dict)
                if len(self.game_events_history) > 100:
                    self.game_events_history.pop(0)
                if ev_dict["kind"] == "robot_move":
                    cell = ev_dict["cell"]
                    self.latest_why_this_move = (
                        f"Robot placed X at cell {cell} according to minimax score optimization."
                    )
            orig_events_put(ev)

        self.bus.game_events.put = hooked_events_put

    def _update_telemetry(self) -> None:
        """Compute live FPS and latencies from the latest Percept."""
        p = self.bus.percept.get()
        if p is not None and p.t != self._last_percept_t:
            self._frame_count += 1
            self._last_percept_t = p.t
            now = time.monotonic()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self.fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now

            dt_ms = (now - p.t) * 1000.0
            if 0 <= dt_ms <= 1000.0:
                self.last_perception_ms = dt_ms

    def get_latest_jpeg(self) -> bytes:
        """Fetch latest overlay frame from Percept or fallback to sensor frame."""
        p = self.bus.percept.get()
        if p is not None and p.overlay is not None:
            return _encode_jpeg(p.overlay, is_rgb=False)

        sf = self.bus.sensors.get()
        if sf is not None and sf.rgb is not None:
            return _encode_jpeg(sf.rgb, is_rgb=True)

        return _create_placeholder_frame("Starting MuJoCo overhead camera...")

    def run(self) -> None:
        """Start the HTTP server and serve until bus.stop is set."""
        ui_self = self

        class RequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                path = self.path.split("?")[0]

                if path in ("/", "/index.html"):
                    if INDEX_HTML_PATH.exists():
                        body = INDEX_HTML_PATH.read_bytes()
                    else:
                        body = b"<h1>Buddy Stage UI</h1>"
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                elif path == "/stream":
                    self.send_response(HTTPStatus.OK)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame",
                    )
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.end_headers()

                    try:
                        while not ui_self.bus.stop.is_set():
                            ui_self._update_telemetry()
                            jpg_bytes = ui_self.get_latest_jpeg()
                            header = (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                b"Content-Length: "
                                + str(len(jpg_bytes)).encode("ascii")
                                + b"\r\n\r\n"
                            )
                            self.wfile.write(header + jpg_bytes + b"\r\n")
                            self.wfile.flush()
                            time.sleep(0.066)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

                elif path == "/frame.jpg":
                    jpg_bytes = ui_self.get_latest_jpeg()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg_bytes)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(jpg_bytes)

                elif path in ("/api/state", "/state.json"):
                    ui_self._update_telemetry()
                    p = ui_self.bus.percept.get()
                    sf = ui_self.bus.sensors.get()
                    hand_flag = bool(ui_self.bus.hand_flag.get())

                    sheet_dict = None
                    if p and p.sheet:
                        sheet_dict = {
                            "x": p.sheet.x,
                            "y": p.sheet.y,
                            "yaw": p.sheet.yaw,
                            "confidence": p.sheet.confidence,
                        }

                    cells_list = []
                    if p and p.cells:
                        cells_list = [c.mark for c in p.cells]

                    with ui_self._lock:
                        data = {
                            "fps": ui_self.fps,
                            "perception_ms": ui_self.last_perception_ms,
                            "gemma_s": ui_self.last_gemma_s,
                            "agent_state": ui_self.current_agent_state,
                            "why": ui_self.latest_why_this_move,
                            "hand_in_area": hand_flag,
                            "sheet": sheet_dict,
                            "cells": cells_list,
                            "touch_N": sf.touch_force if sf else 0.0,
                            "transcript": list(ui_self.transcript_history),
                            "game_events": list(ui_self.game_events_history),
                        }

                    payload = json.dumps(data).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(payload)

                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def do_POST(self) -> None:
                path = self.path.split("?")[0]
                length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(length) if length > 0 else b"{}"

                try:
                    payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                except json.JSONDecodeError:
                    payload = {}

                if path == "/say":
                    text = str(payload.get("text", "")).strip()
                    if text:
                        intent_obj = None
                        if ui_self.llm is not None:
                            try:
                                t0 = time.perf_counter()
                                it, timing = ui_self.llm.intent(text)
                                dt = time.perf_counter() - t0
                                ui_self.last_gemma_s = timing.get("total_s", dt)
                                intent_obj = it
                            except Exception as e:
                                ui_self.bus.transcript.put(f"[ui] LLM intent error: {e}")

                        if intent_obj is None:
                            t_lower = text.lower()
                            if "stop" in t_lower or "enough" in t_lower:
                                intent_obj = Intent(kind="stop", text=text)
                            elif "play" in t_lower or "start" in t_lower or "yes" in t_lower:
                                intent_obj = Intent(kind="play", text=text)
                            elif "repeat" in t_lower or "again" in t_lower:
                                intent_obj = Intent(kind="repeat", text=text)
                            else:
                                m = re.search(r"\b([0-8])\b", text)
                                if m:
                                    intent_obj = Intent(kind="move", cell=int(m.group(1)), text=text)
                                else:
                                    intent_obj = Intent(kind="chat", text=text)

                        ui_self.bus.intents.put(intent_obj)

                    resp = json.dumps({"ok": True}).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)

                elif path == "/scene":
                    action = str(payload.get("action", ""))
                    w = ui_self.world

                    if w is not None:
                        if action == "sheet_left":
                            w.move_sheet(dy_m=-0.02)
                            ui_self.bus.transcript.put("[scene] Sheet moved left (-2 cm)")
                        elif action == "sheet_right":
                            w.move_sheet(dy_m=0.02)
                            ui_self.bus.transcript.put("[scene] Sheet moved right (+2 cm)")
                        elif action == "sheet_forward":
                            w.move_sheet(dx_m=0.02)
                            ui_self.bus.transcript.put("[scene] Sheet moved forward (+2 cm)")
                        elif action == "sheet_back":
                            w.move_sheet(dx_m=-0.02)
                            ui_self.bus.transcript.put("[scene] Sheet moved back (-2 cm)")
                        elif action == "sheet_cw":
                            w.move_sheet(dyaw_rad=-0.1)
                            ui_self.bus.transcript.put("[scene] Sheet rotated clockwise (-5.7 deg)")
                        elif action == "sheet_ccw":
                            w.move_sheet(dyaw_rad=0.1)
                            ui_self.bus.transcript.put("[scene] Sheet rotated counter-clockwise (+5.7 deg)")
                        elif action == "sheet_reset":
                            w.move_sheet(dx_m=0, dy_m=0, dyaw_rad=0)
                            if w.ink is not None:
                                w.ink.clear()
                            ui_self.bus.transcript.put("[scene] Sheet reset and ink cleared")
                        elif action == "hand_over":
                            w.set_hand(True)
                            ui_self.bus.transcript.put("[scene] Hand placed over sheet (triggering reflex)")
                        elif action == "hand_away":
                            w.set_hand(False)
                            ui_self.bus.transcript.put("[scene] Hand removed from sheet area")
                        elif action.startswith("mark_"):
                            try:
                                cell_idx = int(action.split("_")[1])
                                if 0 <= cell_idx <= 8:
                                    w.add_human_mark(cell_idx, "O")
                                    ui_self.bus.transcript.put(f"[scene] Added human mark 'O' at cell {cell_idx}")
                            except (ValueError, IndexError):
                                pass

                    resp = json.dumps({"ok": True, "action": action}).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)

                elif path == "/ptt":
                    active = bool(payload.get("active", False))
                    ui_self.ptt_active = active
                    resp = json.dumps({"ok": True, "ptt": active}).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)

                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        socketserver.TCPServer.allow_reuse_address = True
        try:
            self.server = ThreadingHTTPServer(("0.0.0.0", self.port), RequestHandler)
            print(f"[ui] Server running at http://localhost:{self.port}", flush=True)
        except OSError:
            alt_port = self.port + 80
            self.server = ThreadingHTTPServer(("0.0.0.0", alt_port), RequestHandler)
            self.port = alt_port
            print(f"[ui] Port conflict: server running at http://localhost:{self.port}", flush=True)

        server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        server_thread.start()

        self.bus.stop.wait()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
