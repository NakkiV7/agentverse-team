"""Scripted dry run of the whole setup conversation, no voice, no microphone, simulation as fast as it goes.

    .venv/bin/python tools/dry_run.py            # prints every pipeline step; exits 0 if Buddy reaches READY

Script: small talk (goes to Gemma) -> "let's play a game" -> "go on then" -> Buddy finds the sheet, draws and checks
the grid -> "crosses please". With --far the sheet starts out of reach, so Buddy has to ask for it to be moved.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buddy.app import App                                              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--far", action="store_true", help="start with the sheet out of reach")
args = ap.parse_args()

app = App(voice=False, mic=False)
if args.far:
    app.scene_actions.put("far")


def wait_state(*states, timeout=120):
    t0 = time.time()
    while app.brain.state not in states:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"stuck in {app.brain.state}, expected {states}")
        time.sleep(0.1)


def player():
    time.sleep(1.0)
    app.brain.hear("Buddy, how are you today?")
    wait_state("CHAT")
    time.sleep(4.0)                                   # Gemma answers the small talk
    app.brain.hear("let's play a game")
    wait_state("CHOOSE_GAME")
    app.brain.hear("go on then")
    if args.far:
        wait_state("WAIT_SHEET")
        time.sleep(1.0)
        app.trace("system", "(the player slides the paper back)")
        app.scene_actions.put("back")
    wait_state("ASK_SIDE", timeout=240)
    app.brain.hear("crosses please")
    wait_state("READY")
    time.sleep(0.5)
    app.running = False


t0 = time.time()
threading.Thread(target=player, daemon=True).start()
app.run(until=lambda a: not a.running, realtime=False)
print(f"\nreached {app.brain.state} in {time.time() - t0:.1f} s wall, {app.sim_time:.1f} s simulated")
sys.exit(0 if app.brain.state == "READY" else 1)
