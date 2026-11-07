"""The whole thing: the player teaches Buddy a game it has never seen, Buddy draws the board and plays it on the
paper. Scripted, no audio, simulation as fast as it goes. Prints every pipeline step.

    .venv/bin/python tools/teach_full_dry_run.py            # Footsteps, taught in two sentences; exits 0 at GAME_OVER
    .venv/bin/python tools/teach_full_dry_run.py --misere   # "three in a row and you lose"
    .venv/bin/python tools/teach_full_dry_run.py --new      # a game Buddy has never seen: it asks for a name and remembers it

The scripted player wakes Buddy, asks to teach a game, explains it (answering Buddy's question), says yes, picks
crosses, then plays legal moves drawn by the simulated hand; one move is asked for by voice. Buddy learns the
rules with Gemma 4, solves the game, draws the grid, and plays with the generic solver. Needs Ollama with
gemma3:4b (intents) and gemma4 (teaching).
"""
import argparse
import os
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("BUDDY_GAMES_FILE", os.path.join(tempfile.mkdtemp(), "games.json"))   # do not touch runtime/games.json
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buddy.app import App                                              # noqa: E402
from buddy.game.rules import cell_name, result                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--misere", action="store_true")
ap.add_argument("--new", action="store_true", help="a game not in the library, so Buddy asks for its name")
args = ap.parse_args()

if args.misere:
    LESSON = ["It's like noughts and crosses but backwards: if you make three in a row, you lose.", "Anywhere on the grid."]
elif args.new:
    LESSON = ["Each mark has to touch the last mark played, up, down or sideways, and three in a row wins."]
else:
    LESSON = ["It's a different game. Each mark has to go next to the last mark that was played, up, down or sideways.",
              "If you can't go anywhere, you lose."]
NAME = "Let's call it Chains"

app = App(voice=False, mic=False)
b = app.brain
rng = random.Random(5)
outcome = {"ok": False}


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
    say("Buddy, let me teach you a game I know.")
    wait(lambda: b.state == "TEACH", "the invitation")
    for line in LESSON:
        say(line)
        if b.state in ("OFFER_TAUGHT", "NAME_GAME"):
            break
    wait(lambda: b.state in ("OFFER_TAUGHT", "NAME_GAME"), "Buddy to learn the game")
    if b.state == "NAME_GAME":
        say(NAME)
        wait(lambda: b.state == "OFFER_TAUGHT", "Buddy to remember the game")
    say("Yes please.")
    wait(lambda: b.state == "ASK_SIDE", "the board")
    say("Crosses please.")
    wait(lambda: b.state in ("PLAYER_TURN", "ROBOT_TURN"), "the game to start")
    move = 0
    while b.state != "GAME_OVER":
        if b.state == "PLAYER_TURN" and not b.busy:
            legal = b.legal_cells()
            if not legal:
                time.sleep(0.2)
                continue
            cell = rng.choice(legal)
            move += 1
            if move == 2:
                say(f"put my cross in the {cell_name(b.spec, cell)} please")
            else:
                app.trace("system", f"(the player draws a cross in the {cell_name(b.spec, cell)})")
                app.scene_actions.put(("player_mark", cell))
            wait(lambda: b.state != "PLAYER_TURN", "Buddy to see my move", 60)
        elif b.state == "CONFIRM_MARK":
            say("yes")
            wait(lambda: b.state != "CONFIRM_MARK", "the confirmation", 30)
        elif b.state == "ROBOT_TURN":
            wait(lambda: b.state != "ROBOT_TURN", "Buddy's move", 120)
        else:
            time.sleep(0.1)
    wait(lambda: not b.busy, "the end of the game", 120)
    time.sleep(1.0)
    outcome["ok"] = True
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
print("\nfinal board (as read from the paper):")
for r in range(3):
    print("   " + " | ".join(b.board[3 * r + c] or " " for c in range(3)))
res = result(b.spec, b.gstate) if (b.spec and b.gstate) else None
print(f"game: {b.game_name} | rules: {b.rules_text()} | library: {b.library.names()} ({b.library.path})")
print(f"result: {res} | {time.time() - t0:.0f} s wall, {app.sim_time:.0f} s simulated | difficulty now {100 * b.difficulty.strength:.0f}%")
sys.exit(0 if outcome["ok"] else 1)
