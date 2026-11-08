"""The menu, difficulty and memory, scripted: Buddy lists its games, the player picks Footsteps by name and asks
for impossible mode; the game is played on the paper; the result is recorded in a temporary games file.

    .venv/bin/python tools/menu_dry_run.py      # exits 0 at GAME_OVER with the record written
"""
import os
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("BUDDY_GAMES_FILE", os.path.join(tempfile.mkdtemp(), "games.json"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("BUDDY_PROFILE_FILE", os.path.join(tempfile.mkdtemp(), "profile.json"))

from buddy.app import App                                              # noqa: E402
from buddy.game.rules import cell_name, result                         # noqa: E402

app = App(voice=False, mic=False)
b = app.brain
rng = random.Random(11)
ok = {"done": False}


def wait(cond, what, timeout=240):
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"waiting for {what}; state {b.state}")
        time.sleep(0.1)


def say(text):
    app.trace("system", f"(player says) {text}")
    said = b.last_said
    b.hear(text)
    wait(lambda: b.last_said != said, "Buddy's reply", 60)


def player():
    time.sleep(2.0)
    say("Buddy, what games do you know?")
    wait(lambda: b.state == "CHOOSE_GAME", "the list")
    say("Play your best, don't hold back.")
    say("Let's play footsteps.")
    wait(lambda: b.state == "ASK_SIDE", "the board")
    say("Crosses please.")
    wait(lambda: b.state in ("PLAYER_TURN", "ROBOT_TURN"), "the game to start")
    while b.state != "GAME_OVER":
        if b.state == "PLAYER_TURN" and not b.busy:
            legal = b.legal_cells()
            if not legal:
                time.sleep(0.2); continue
            cell = rng.choice(legal)
            app.trace("system", f"(the player draws a cross in the {cell_name(b.spec, cell)})")
            app.scene_actions.put(("player_mark", cell))
            wait(lambda: b.state != "PLAYER_TURN", "Buddy to see my move", 60)
        elif b.state == "CONFIRM_MARK":
            say("yes"); wait(lambda: b.state != "CONFIRM_MARK", "the confirmation", 30)
        elif b.state == "ROBOT_TURN":
            wait(lambda: b.state != "ROBOT_TURN", "Buddy's move", 120)
        else:
            time.sleep(0.1)
    wait(lambda: not b.busy, "the end", 120)
    time.sleep(1.0)
    ok["done"] = True
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
e = b.entry
print(f"\ngame: {b.game_name} | level {b.difficulty.level} ({100 * b.difficulty.strength:.0f}%) | "
      f"record: played {e.plays}, player {e.player_wins}, Buddy {e.robot_wins}, drawn {e.draws} -> {b.library.path}")
print(f"result: {result(b.spec, b.gstate) if b.gstate else None} | {time.time() - t0:.0f} s wall")
sys.exit(0 if ok["done"] and e.plays == 1 else 1)
