"""The demo page server: http://localhost:8860

GET  /             the page (buddy/ui/index.html)
GET  /overhead.jpg overhead camera with the perception overlay
GET  /scene.jpg    3/4 view of the arm
GET  /sheet.jpg    the sheet straightened top-down, with the mark classifier's reading of each cell
GET  /state.json   state machine, pipeline trace, conversation, sensors
POST /say          {"text": ...}  typed input, same path as the microphone
POST /scene        {"action": new_sheet | slide | rotate | far | remove | back | reach | player_mark, "arg": cell}
POST /draw         {"strokes": [[[x_mm, y_mm], ...], ...]}  the player's hand draws these on the sheet
POST /mic          {"enabled": true/false}
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).resolve().parent / "index.html"


def serve(app, port: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/overhead.jpg":
                self._send(app.overhead_jpg, "image/jpeg")
            elif path == "/scene.jpg":
                self._send(app.scene_jpg, "image/jpeg")
            elif path == "/sheet.jpg":
                self._send(app.sheet_jpg, "image/jpeg")
            elif path == "/state.json":
                self._send(json.dumps(app.state()).encode(), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                data = {}
            path = self.path.split("?")[0]
            if path == "/say" and str(data.get("text", "")).strip():
                app.brain.hear(str(data["text"]).strip()[:300], "typed")
            elif path == "/scene":
                app.scene_actions.put((str(data.get("action", "")), data.get("arg")))
            elif path == "/draw":
                strokes = [[(float(p[0]), float(p[1])) for p in st] for st in data.get("strokes", []) if len(st) >= 2]
                if strokes:
                    app.scene_actions.put(("strokes", strokes))
            elif path == "/mic" and app.listener is not None:
                app.listener.enabled = bool(data.get("enabled", True))
            self._send(b"{}", "application/json")

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="web").start()
    return httpd
