"""Latency and memory figures for the technical summary, measured on this Mac with the real pieces.

    .venv/bin/python tools/measure_m5.py            # about 3 minutes; run it while the demo is up for honest numbers

Prints typical (median) and worst case for: Whisper per sentence (small.en, resident), sheet finding per frame,
board reading for nine squares, hand enters -> arm frozen, drawing the grid, drawing one mark; then the resident
memory of every Penny-related process (this script, any running demo, Ollama and its model runners). Writes
runtime/measure_m5.json.
"""
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def stats(xs, unit="ms", scale=1000.0):
    xs = [x * scale for x in xs]
    return {"typical": round(statistics.median(xs), 1), "worst": round(max(xs), 1), "n": len(xs), "unit": unit}


def memory():
    out = subprocess.run(["ps", "-axo", "rss=,command="], capture_output=True, text=True).stdout.splitlines()
    rows = []
    for line in out:
        rss, _, cmd = line.strip().partition(" ")
        if any(k in cmd for k in ("python -m buddy", "measure_m5", "ollama", "Ollama")) and "grep" not in cmd:
            rows.append((int(rss) / 1024, cmd[:90]))
    return rows


def main():
    with tempfile.TemporaryDirectory(prefix="penny-measure-") as tmp:
        os.environ["BUDDY_GAMES_FILE"] = str(Path(tmp) / "games.json")
        os.environ["BUDDY_PROFILE_FILE"] = str(Path(tmp) / "profile.json")
        from buddy import config as C
        from buddy.app import App
        from buddy.game import marks as M
        from buddy.motion.drawing import grid_lines_mm
        from buddy.perception.sheet import find_sheet

        result = {}
        # -- Whisper, as the demo runs it: small.en resident, on a sentence Penny's own voice speaks
        try:
            from buddy.voice import stt, tts
            speaker = tts.Speaker(play=False)
            listener_model = stt.WhisperCpp(C.WHISPER_MODEL)
            clips = [speaker.synth(s) for s in ("Penny, let's play noughts and crosses.",
                                                "Put my cross in the top right corner, please.",
                                                "What letters have I asked for so far?")]
            sr_in = getattr(speaker.engine, "sr", 22050)
            times = []
            for audio in clips * 2:
                a16 = np.interp(np.linspace(0, len(audio), int(len(audio) * 16000 / sr_in), endpoint=False),
                                np.arange(len(audio)), audio).astype(np.float32)
                t0 = time.perf_counter()
                text = listener_model.transcribe(a16) if listener_model else stt.transcribe(a16)[0]
                times.append(time.perf_counter() - t0)
            result["whisper_per_sentence"] = stats(times)
            result["whisper_example"] = text
        except Exception as e:                                         # no audio stack: report, carry on
            result["whisper_per_sentence"] = {"error": str(e)}

        app = App(voice=False, mic=False)
        out = {}

        def work():
            try:
                t_end = time.time() + 60
                while app.tracker.track is None and time.time() < t_end:
                    time.sleep(0.1)
                frames = []
                for _ in range(20):
                    look = app.look(timeout=10)
                    frames.append(look)
                    time.sleep(0.05)
                t = []
                for look in frames:
                    t0 = time.perf_counter(); find_sheet(look.rgb, app.cam); t.append(time.perf_counter() - t0)
                out["sheet_finding_per_frame"] = stats(t)
                job = app.draw("the grid", grid_lines_mm()); t0 = time.time(); job.done.wait()
                out["grid_drawing"] = stats([time.time() - t0], "s", 1.0)
                marks = []
                for cell in (4, 0, 8):
                    job = app.draw("a mark", M.robot_mark("O", cell)); t0 = time.time(); job.done.wait()
                    marks.append(time.time() - t0)
                out["one_mark"] = stats(marks, "s", 1.0)
                t = []
                for _ in range(10):
                    look = app.look(timeout=10)
                    t0 = time.perf_counter(); app.reader.read(look.rgb, app.cam, look.track.pose); t.append(time.perf_counter() - t0)
                out["board_reading_9_squares"] = stats(t)
                freeze = []
                for _ in range(3):
                    job = app.draw("a mark", M.robot_mark("X", 2))
                    time.sleep(1.0)
                    t0 = time.time(); app.scene_actions.put(("reach", None))
                    while not app.ctl.paused and time.time() - t0 < 10:
                        time.sleep(0.005)
                    freeze.append(time.time() - t0)
                    job.done.wait(timeout=60)
                    time.sleep(1.0)
                out["hand_enters_to_arm_frozen"] = stats(freeze)
                out["memory_mb"] = [{"mb": round(mb), "process": cmd} for mb, cmd in memory()]
            except Exception as e:
                out["error"] = repr(e)
            finally:
                app.running = False

        threading.Thread(target=work, daemon=True).start()
        try:
            app.run(until=lambda a: not a.running, realtime=True)
        finally:
            app.close()
        result.update(out)
        result["load_average"] = os.getloadavg()
        result["total_memory_mb"] = sum(r["mb"] for r in result.get("memory_mb", []))
        Path(C.RUNTIME / "measure_m5.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
