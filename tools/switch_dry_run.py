"""Changing your mind mid-game, scripted: start noughts and crosses, ask for dots and boxes while the grid is being
drawn (Penny finishes the stroke, then switches and asks for a fresh sheet), play two dots turns, ask for hangman
(switch again), then stop. Exits 0 when every switch happened.

    .venv/bin/python tools/switch_dry_run.py
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("BUDDY_GAMES_FILE", os.path.join(tempfile.mkdtemp(), "games.json"))
os.environ.setdefault("BUDDY_PROFILE_FILE", os.path.join(tempfile.mkdtemp(), "profile.json"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buddy.app import App                                              # noqa: E402

app = App(voice=False, mic=False)
b = app.brain
got = {"switched_to_dots": False, "switched_to_hangman": False, "stopped": False}


def wait(cond, what, timeout=300):
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"waiting for {what}; state {b.state}")
        time.sleep(0.1)


def say(text):
    app.trace("system", f"(player says) {text}")
    said = b.last_said
    b.hear(text)
    wait(lambda: b.last_said != said, "Penny's reply", 60)


def player():
    time.sleep(2.0)
    say("Penny, let's play noughts and crosses.")
    wait(lambda: b.state == "ASK_LEVEL", "the level question")
    say("medium")
    wait(lambda: b.state == "SETUP" and b.busy, "the grid drawing to start", 60)
    time.sleep(4.0)
    say("No wait, let's not play noughts and crosses, let's play dots and boxes instead.")
    wait(lambda: b.state == "DOTS_SIZE", "the switch to dots and boxes", 200)
    got["switched_to_dots"] = True
    say("Two by two.")
    wait(lambda: b.state == "DOTS_SETUP", "dots setup")
    wait(lambda: "fresh" in b.last_said.lower() or "blank" in b.last_said.lower() or b.state == "DOTS_PLAYER_TURN", "the fresh-sheet request", 60)
    if b.state != "DOTS_PLAYER_TURN":
        app.trace("system", "(the player puts a fresh sheet down)")
        app.scene_actions.put(("new_sheet", None))
    wait(lambda: b.state == "DOTS_PLAYER_TURN", "the dots to be drawn", 300)
    app.trace("system", "(the player draws the top side of the top left box)")
    app.scene_actions.put(("player_edge", ((2, 2), 0)))
    wait(lambda: b.state == "DOTS_ROBOT_TURN", "Penny's turn", 90)
    time.sleep(1.0)
    say("Actually, can we play hangman instead?")
    wait(lambda: b.state in ("HANGMAN_SETUP", "HANGMAN_TURN"), "the switch to hangman", 240)
    got["switched_to_hangman"] = True
    wait(lambda: "fresh" in b.last_said.lower() or "blank" in b.last_said.lower() or b.state == "HANGMAN_TURN", "hangman's sheet request", 90)
    if b.state != "HANGMAN_TURN":
        app.trace("system", "(the player puts a fresh sheet down)")
        app.scene_actions.put(("new_sheet", None))
        time.sleep(1.5)
        say("Yes, there you go.")                 # hangman waits for a yes after a fresh sheet
    wait(lambda: b.state == "HANGMAN_TURN" and not b.busy, "the dashes", 300)
    say("E")
    wait(lambda: not b.busy, "the letter", 120)
    say("That's enough for now, thank you. Stop.")
    wait(lambda: b.state == "IDLE", "the stop", 120)
    got["stopped"] = True
    time.sleep(1.0)
    app.running = False


def runner():
    try:
        player()
    except Exception as e:
        app.trace("system", f"dry run stopped: {e}")
        app.running = False


t0 = time.time()
threading.Thread(target=runner, daemon=True).start()
app.run(until=lambda a: not a.running, realtime=False)
print(f"\nswitches: {got} | final state {b.state} | {time.time() - t0:.0f} s wall")
sys.exit(0 if all(got.values()) else 1)
