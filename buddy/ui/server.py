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


# Modern, polished dark-themed dashboard HTML
HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Buddy · Live Stage & Stream</title>
<style>
  :root {
    --bg: #0e1117;
    --card: #161b22;
    --card-border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --accent-hover: #79b8ff;
    --green: #3fb950;
    --red: #f85149;
    --yellow: #d29922;
    --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
    padding: 16px;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--card-border);
    margin-bottom: 16px;
    flex-wrap: wrap;
    gap: 12px;
  }
  .title-group { display: flex; align-items: center; gap: 12px; }
  h1 { font-size: 20px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
  .badge {
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 12px;
    background: #21262d;
    border: 1px solid var(--card-border);
    color: var(--text-muted);
  }
  .badge.state { background: rgba(88, 166, 255, 0.15); color: var(--accent); border-color: rgba(88, 166, 255, 0.3); }
  .badge.urgent { background: rgba(248, 81, 73, 0.15); color: var(--red); border-color: rgba(248, 81, 73, 0.3); }
  .badge.live { background: rgba(63, 185, 80, 0.15); color: var(--green); border-color: rgba(63, 185, 80, 0.3); }

  .telemetry-bar {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }
  .stat-chip {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .stat-chip b { color: #fff; font-family: monospace; }

  main {
    display: grid;
    grid-template-columns: minmax(420px, 1.4fr) minmax(360px, 1fr);
    gap: 16px;
  }
  @media (max-width: 960px) {
    main { grid-template-columns: 1fr; }
  }

  .panel {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    position: relative;
  }
  .panel-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  /* Video stream */
  .stream-container {
    width: 100%;
    aspect-ratio: 4/3;
    background: #000;
    border-radius: 6px;
    overflow: hidden;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .stream-container img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .warning-banner {
    display: none;
    position: absolute;
    top: 10px;
    left: 10px;
    right: 10px;
    background: rgba(248, 81, 73, 0.9);
    color: #fff;
    padding: 8px 12px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 13px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    animation: pulse 1s infinite alternate;
  }
  @keyframes pulse { from { opacity: 0.8; } to { opacity: 1; } }

  /* Why this move */
  .why-box {
    background: rgba(88, 166, 255, 0.08);
    border: 1px solid rgba(88, 166, 255, 0.25);
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 13px;
    color: #e6edf3;
  }
  .why-box b { color: var(--accent); }

  /* Chat / Transcript */
  .transcript-box {
    height: 220px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-right: 4px;
    font-size: 13px;
  }
  .chat-msg {
    max-width: 88%;
    padding: 6px 12px;
    border-radius: 12px;
    word-wrap: break-word;
  }
  .chat-msg.buddy {
    background: #1f2937;
    border: 1px solid #374151;
    align-self: flex-start;
    color: #93c5fd;
  }
  .chat-msg.player {
    background: #064e3b;
    border: 1px solid #047857;
    align-self: flex-end;
    color: #a7f3d0;
  }
  .chat-msg.system {
    background: #27272a;
    border: 1px solid #3f3f46;
    align-self: center;
    font-size: 11.5px;
    color: var(--text-muted);
  }

  /* Typed input */
  .input-form {
    display: flex;
    gap: 8px;
    margin-top: 4px;
  }
  input[type=text] {
    flex: 1;
    background: #0d1117;
    border: 1px solid var(--card-border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
    outline: none;
  }
  input[type=text]:focus { border-color: var(--accent); }
  button {
    background: #21262d;
    color: var(--text);
    border: 1px solid var(--card-border);
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.15s ease;
  }
  button:hover { background: #30363d; border-color: var(--text-muted); }
  button.primary {
    background: #238636;
    border-color: rgba(240, 246, 252, 0.1);
    color: #fff;
  }
  button.primary:hover { background: #2ea043; }
  button.warn { background: rgba(248, 81, 73, 0.2); border-color: var(--red); color: var(--red); }
  button.warn:hover { background: rgba(248, 81, 73, 0.35); }

  /* Scene controls */
  .button-group {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .grid-3x3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    max-width: 180px;
  }
  .grid-3x3 button {
    height: 38px;
    font-size: 14px;
    font-weight: 700;
  }

  /* Game events */
  .events-box {
    height: 140px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 11.5px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    background: #0d1117;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 6px 10px;
  }
  .event-row { color: var(--text-muted); }
  .event-row b { color: var(--purple); }
</style>
</head>
<body>

<header>
  <div class="title-group">
    <h1>🤖 Buddy <span style="font-weight:400; font-size:15px; color:var(--text-muted);">· Stage & Stream</span></h1>
    <span class="badge state" id="badge-state">STATE: IDLE</span>
    <span class="badge live" id="badge-reflex">REFLEX: ARMED</span>
  </div>

  <div class="telemetry-bar">
    <div class="stat-chip">Perception: <b id="stat-fps">0</b> FPS (<b id="stat-perc-ms">0</b> ms)</div>
    <div class="stat-chip">Gemma: <b id="stat-gemma-s">0.0</b> s</div>
    <div class="stat-chip">Touch: <b id="stat-touch">0.00</b> N</div>
    <div class="stat-chip">Sheet: <b id="stat-sheet">--</b></div>
  </div>
</header>

<main>
  <!-- LEFT COLUMN: CAMERA OVERLAY & WHY THIS MOVE -->
  <div style="display:flex; flex-direction:column; gap:16px;">
    <div class="panel">
      <div class="panel-title">
        <span>Live Overhead Camera · OpenCV Perception Overlay</span>
        <span class="badge" id="stream-status">STREAMING</span>
      </div>
      <div class="stream-container">
        <img id="cam-stream" src="/stream" alt="Overhead camera overlay stream">
        <div class="warning-banner" id="hand-warning">⚠️ HAND IN AREA — REFLEX LAYER FROZEN</div>
      </div>
    </div>

    <!-- WHY THIS MOVE PANEL (Hour 7) -->
    <div class="panel">
      <div class="panel-title">💡 Why this move (Minimax Reasoning)</div>
      <div class="why-box" id="why-box">
        Waiting for initial robot move or evaluation...
      </div>
    </div>

    <!-- GAME EVENTS LOG (Hour 7) -->
    <div class="panel">
      <div class="panel-title">🎮 Game Events</div>
      <div class="events-box" id="events-log">
        <div class="event-row">[system] Game subsystem initialized.</div>
      </div>
    </div>
  </div>

  <!-- RIGHT COLUMN: TRANSCRIPT, TYPED FALLBACK, SCENE BUTTONS -->
  <div style="display:flex; flex-direction:column; gap:16px;">
    <!-- TRANSCRIPT & TYPED INPUT (Hour 1) -->
    <div class="panel">
      <div class="panel-title">💬 Conversation & Voice Transcript</div>
      <div class="transcript-box" id="transcript-box">
        <div class="chat-msg system">Buddy system ready. Speak "Buddy" or type below.</div>
      </div>
      <form class="input-form" id="typed-form">
        <input type="text" id="typed-input" placeholder="Type what to say (e.g. 'let's play', 'cell 4', 'stop')..." autocomplete="off">
        <button type="submit" class="primary">Send</button>
      </form>
      <div style="display:flex; gap:6px; align-items:center; margin-top:4px;">
        <button type="button" id="btn-ptt" style="flex:1;">🎙️ Push-to-Talk (Hold Space)</button>
        <span style="font-size:11px; color:var(--text-muted);">Noisy room fallback: active</span>
      </div>
    </div>

    <!-- SCENE BUTTONS (Hour 3) -->
    <div class="panel">
      <div class="panel-title">🎬 Scene Controls (Live Demo Actions)</div>

      <div style="display:flex; flex-direction:column; gap:10px;">
        <!-- Sheet Translations & Rotations -->
        <div>
          <div style="font-size:12px; color:var(--text-muted); margin-bottom:4px;">Move / Rotate Sheet:</div>
          <div class="button-group">
            <button onclick="postScene('sheet_left')">← Left</button>
            <button onclick="postScene('sheet_right')">→ Right</button>
            <button onclick="postScene('sheet_forward')">↑ Forward</button>
            <button onclick="postScene('sheet_back')">↓ Back</button>
            <button onclick="postScene('sheet_ccw')">↺ Rot CCW</button>
            <button onclick="postScene('sheet_cw')">↻ Rot CW</button>
            <button onclick="postScene('sheet_reset')">Reset Sheet</button>
          </div>
        </div>

        <!-- Hand Reflex Freeze Controls -->
        <div>
          <div style="font-size:12px; color:var(--text-muted); margin-bottom:4px;">Safety & Reflex Trigger:</div>
          <div class="button-group">
            <button class="warn" onclick="postScene('hand_over')">✋ Place Hand Over Sheet</button>
            <button onclick="postScene('hand_away')">👋 Remove Hand</button>
          </div>
        </div>

        <!-- Add Human Mark O on Cell (0..8) -->
        <div>
          <div style="font-size:12px; color:var(--text-muted); margin-bottom:4px;">Add Player Mark 'O' (Simulate drawing):</div>
          <div style="display:flex; gap:16px; align-items:center;">
            <div class="grid-3x3">
              <button onclick="postScene('mark_0')">0</button>
              <button onclick="postScene('mark_1')">1</button>
              <button onclick="postScene('mark_2')">2</button>
              <button onclick="postScene('mark_3')">3</button>
              <button onclick="postScene('mark_4')">4</button>
              <button onclick="postScene('mark_5')">5</button>
              <button onclick="postScene('mark_6')">6</button>
              <button onclick="postScene('mark_7')">7</button>
              <button onclick="postScene('mark_8')">8</button>
            </div>
            <div style="font-size:11px; color:var(--text-muted); line-height:1.4;">
              Click a cell to draw an <b>O</b> into the paper texture. Perception will detect it on the next frame!
            </div>
          </div>
        </div>

      </div>
    </div>

  </div>
</main>

<script>
async function postScene(action) {
  try {
    await fetch('/scene', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: action})
    });
  } catch(e) {
    console.error('Scene command failed:', e);
  }
}

// Typed input submission
document.getElementById('typed-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('typed-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  try {
    await fetch('/say', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: text})
    });
  } catch(e) {
    console.error('Say failed:', e);
  }
});

// Push to talk button
const pttBtn = document.getElementById('btn-ptt');
let pttActive = false;
function setPTT(active) {
  if (pttActive === active) return;
  pttActive = active;
  pttBtn.style.background = active ? '#dc2626' : '';
  pttBtn.textContent = active ? '🔴 Recording...' : '🎙️ Push-to-Talk (Hold Space)';
  fetch('/ptt', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({active: active})
  }).catch(()=>{});
}
pttBtn.addEventListener('mousedown', () => setPTT(true));
window.addEventListener('mouseup', () => setPTT(false));
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && document.activeElement.tagName !== 'INPUT') {
    e.preventDefault();
    setPTT(true);
  }
});
window.addEventListener('keyup', (e) => {
  if (e.code === 'Space' && document.activeElement.tagName !== 'INPUT') {
    setPTT(false);
  }
});

// Periodic polling for telemetry & state sync
let lastMsgCount = 0;
let lastEventsCount = 0;

async function syncState() {
  try {
    const res = await fetch('/api/state', {cache: 'no-store'});
    if (!res.ok) return;
    const d = await res.json();

    // Telemetry updates
    document.getElementById('stat-fps').textContent = d.fps !== undefined ? d.fps.toFixed(1) : '0';
    document.getElementById('stat-perc-ms').textContent = d.perception_ms !== undefined ? d.perception_ms.toFixed(0) : '0';
    document.getElementById('stat-gemma-s').textContent = d.gemma_s !== undefined ? d.gemma_s.toFixed(2) : '0.0';
    document.getElementById('stat-touch').textContent = d.touch_N !== undefined ? d.touch_N.toFixed(2) : '0.00';
    document.getElementById('stat-sheet').textContent = d.sheet ? `${(d.sheet.x*100).toFixed(0)}, ${(d.sheet.y*100).toFixed(0)}cm` : 'searching';

    // State badge
    const badgeState = document.getElementById('badge-state');
    badgeState.textContent = 'STATE: ' + (d.agent_state || 'IDLE').toUpperCase();

    // Hand warning banner & reflex badge
    const banner = document.getElementById('hand-warning');
    const badgeReflex = document.getElementById('badge-reflex');
    if (d.hand_in_area) {
      banner.style.display = 'block';
      badgeReflex.className = 'badge urgent';
      badgeReflex.textContent = 'REFLEX: FROZEN';
    } else {
      banner.style.display = 'none';
      badgeReflex.className = 'badge live';
      badgeReflex.textContent = 'REFLEX: ARMED';
    }

    // Why this move
    if (d.why) {
      document.getElementById('why-box').innerHTML = `<b>Minimax Decision:</b> ${d.why}`;
    }

    // Transcript update
    if (d.transcript && d.transcript.length !== lastMsgCount) {
      lastMsgCount = d.transcript.length;
      const tBox = document.getElementById('transcript-box');
      tBox.innerHTML = '';
      d.transcript.forEach(line => {
        const div = document.createElement('div');
        if (line.startsWith('Buddy:')) {
          div.className = 'chat-msg buddy';
          div.textContent = line;
        } else if (line.startsWith('Player:')) {
          div.className = 'chat-msg player';
          div.textContent = line;
        } else {
          div.className = 'chat-msg system';
          div.textContent = line;
        }
        tBox.appendChild(div);
      });
      tBox.scrollTop = tBox.scrollHeight;
    }

    // Game events update
    if (d.game_events && d.game_events.length !== lastEventsCount) {
      lastEventsCount = d.game_events.length;
      const eBox = document.getElementById('events-log');
      eBox.innerHTML = '';
      d.game_events.forEach(ev => {
        const div = document.createElement('div');
        div.className = 'event-row';
        div.innerHTML = `[${ev.time || 'now'}] <b>${ev.kind}</b>: ${ev.text || ''}`;
        eBox.appendChild(div);
      });
      eBox.scrollTop = eBox.scrollHeight;
    }

  } catch(e) {
    // ignore fetch errors during startup
  }
}

setInterval(syncState, 400);
syncState();

// Handle stream disconnect/reconnect
const streamImg = document.getElementById('cam-stream');
streamImg.onerror = () => {
  document.getElementById('stream-status').textContent = 'RECONNECTING';
  document.getElementById('stream-status').className = 'badge urgent';
  setTimeout(() => {
    streamImg.src = '/stream?t=' + Date.now();
    document.getElementById('stream-status').textContent = 'STREAMING';
    document.getElementById('stream-status').className = 'badge';
  }, 1000);
};
</script>
</body>
</html>
"""


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
            # Percept timestamp delta
            now = time.monotonic()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self.fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now

            # If Percept is fresh, estimate perception latency
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
                # Silence normal request logging to avoid cluttering transcript
                pass

            def do_GET(self) -> None:
                path = self.path.split("?")[0]

                if path in ("/", "/index.html"):
                    body = HTML_PAGE.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                elif path == "/stream":
                    # MJPEG streaming endpoint
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
                            time.sleep(0.066)  # ~15 FPS cap
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
                    # Typed input box -> LLM.intent -> bus.intents (Noisy-room fallback)
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

                        # Fallback parsing if LLM is unavailable
                        if intent_obj is None:
                            t_lower = text.lower()
                            if "stop" in t_lower or "enough" in t_lower:
                                intent_obj = Intent(kind="stop", text=text)
                            elif "play" in t_lower or "start" in t_lower or "yes" in t_lower:
                                intent_obj = Intent(kind="play", text=text)
                            elif "repeat" in t_lower or "again" in t_lower:
                                intent_obj = Intent(kind="repeat", text=text)
                            else:
                                # Check for cell numbers
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
                    # Scene buttons that call World methods live
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

        # Allow quick port reuse
        socketserver.TCPServer.allow_reuse_address = True
        try:
            self.server = ThreadingHTTPServer(("0.0.0.0", self.port), RequestHandler)
            print(f"[ui] Server running at http://localhost:{self.port}", flush=True)
        except OSError:
            # Fallback to alternate port if 8000 is occupied
            alt_port = self.port + 80
            self.server = ThreadingHTTPServer(("0.0.0.0", alt_port), RequestHandler)
            self.port = alt_port
            print(f"[ui] Port conflict: server running at http://localhost:{self.port}", flush=True)

        server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        server_thread.start()

        # Keep thread alive until bus.stop
        self.bus.stop.wait()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
