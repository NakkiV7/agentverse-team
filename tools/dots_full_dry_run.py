"""Dots and boxes through the conversation, with the things that go wrong at a real table. Scripted, no audio.

The player asks for the game, is asked the size and says "three across and two down"; draws a line by hand; names
one by voice ("the left side of the ..."); draws one too short (Penny finishes it); scribbles in a box (Penny asks for
a fresh sheet, redraws the dots, every line and every box so far, and play continues); then plays to the end.

    .venv/bin/python tools/dots_full_dry_run.py            # exits 0 at GAME_OVER with the record written
    .venv/bin/python tools/dots_full_dry_run.py --plain    # no mishaps, 2x2, the quick version
"""
import argparse
import os
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("BUDDY_GAMES_FILE", os.path.join(tempfile.mkdtemp(), "games.json"))
os.environ.setdefault("BUDDY_PROFILE_FILE", os.path.join(tempfile.mkdtemp(), "profile.json"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buddy.app import App                                              # noqa: E402
from buddy.game import dotsboxes as D                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--plain", action="store_true")
args = ap.parse_args()

app = App(voice=False, mic=False)
b = app.brain
rng = random.Random(3)
ok = {"done": False, "finished_short": False, "restored": False, "spoken": False}


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


def my_turn():
    return b.state == "DOTS_PLAYER_TURN" and not b.busy and b.dots and b.dots.state and b.dots.state.to_move == "P"


def hand(action, arg):
    app.scene_actions.put((action, arg))



def _wait_level(timeout=20):
    """Penny asks how hard to play before the first game; give it a moment to ask."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if b.state == "ASK_LEVEL":
            return True
        if b.state not in ("CHOOSE_GAME", "OFFER_TAUGHT", "CHAT", "IDLE"):
            return False
        time.sleep(0.1)
    return False


def player():
    time.sleep(2.0)
    if args.plain:
        say("Penny, let's play dots and boxes, two by two.")
        if _wait_level():
            say("authentic please")
    else:
        say("Penny, let's play dots and boxes.")
        if _wait_level():
            say("authentic please")
        wait(lambda: b.state == "DOTS_SIZE", "the size question")
        say("Three across and two down please.")
    wait(lambda: b.state == "DOTS_PLAYER_TURN", "the dots to be drawn", 400)
    N = b.dots.n
    move = 0
    while b.state != "GAME_OVER":
        if my_turn():
            st = b.dots.state
            e = D.greedy_player(st, rng)
            move += 1
            before = st.edges
            if not args.plain and move == 2:
                name = D.edge_name(N, e)
                say(f"Put my line on {name}.")
                ok["spoken"] = True
            elif not args.plain and move == 3:
                app.trace("system", f"(the player draws only part of {D.edge_name(N, e)})")
                hand("player_short_edge", (N, e))
            elif not args.plain and move == 4:
                k = rng.choice([i for i, o in enumerate(st.owners) if not o])
                app.trace("system", f"(the player scribbles in the {D.box_name(N, k)})")
                hand("player_scribble", (N, k))
                wait(lambda: b.state == "DOTS_SETUP", "Penny to ask for a fresh sheet", 60)
                wait(lambda: "fresh sheet" in b.last_said.lower() or "blank sheet" in b.last_said.lower(), "the request", 60)
                time.sleep(1.0)
                app.trace("system", "(the player puts a fresh sheet down)")
                hand("new_sheet", None)
                wait(lambda: b.state in ("DOTS_PLAYER_TURN", "DOTS_ROBOT_TURN") and b.dots.state.edges == before, "the board to be redrawn", 400)
                ok["restored"] = True
                continue
            else:
                app.trace("system", f"(the player draws {D.edge_name(N, e)})")
                hand("player_edge", (N, e))
            wait(lambda: b.state != "DOTS_PLAYER_TURN" or (b.dots.state and b.dots.state.edges != before), "Penny to see my line", 120)
            if not args.plain and move == 3 and b.dots.state.edges != before:
                ok["finished_short"] = True
            time.sleep(1.5)
        else:
            time.sleep(0.2)
    wait(lambda: not b.busy, "the end", 60)
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
st = b.dots.state if b.dots else None
print(f"\ngame: {b.game_name} {D.size_words(b.dots.n) if b.dots else ''} | boxes: player {st.score('P') if st else '-'}, "
      f"Penny {st.score('B') if st else '-'} | {time.time() - t0:.0f} s wall | record: {b.entry.plays} played | "
      f"spoken line {ok['spoken']}, short line finished {ok['finished_short']}, restored after a scribble {ok['restored']}")
sys.exit(0 if ok["done"] and (args.plain or (ok["spoken"] and ok["finished_short"] and ok["restored"])) else 1)
