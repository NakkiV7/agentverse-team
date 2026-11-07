"""A whole game against Buddy, scripted, no audio, simulation as fast as it goes. Prints every pipeline step.

    .venv/bin/python tools/dry_run.py              # exits 0 when the game reaches its end

The scripted player: wakes Buddy, agrees to noughts and crosses, chooses crosses, then plays. Most moves are drawn on
the paper by the simulated hand; one is spoken ("put my cross in the ..."), so Buddy draws it for them. During
Buddy's first move a hand reaches over the paper (safety stop); during its second the sheet is slid (re-plan).
Buddy only ever learns the player's moves by reading the paper with the camera.
"""
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buddy.app import App                                              # noqa: E402
from buddy.game.tictactoe import NAMES, winner                         # noqa: E402

app = App(voice=False, mic=False)
b = app.brain
rng = random.Random(3)
result = {"ok": False}


def wait(cond, what, timeout=240):
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"waiting for {what}; state {b.state}")
        time.sleep(0.1)


def say(text):
    app.trace("system", f"(player says) {text}")
    b.hear(text)


def player():
    time.sleep(2.0)
    say("Buddy, let's play a game")
    wait(lambda: b.state == "CHOOSE_GAME", "the offer")
    say("yes please")
    wait(lambda: b.state == "ASK_SIDE", "the board")
    say("crosses please")
    wait(lambda: b.state == "PLAYER_TURN", "my turn")
    move, robot_moves = 0, 0
    while b.state != "GAME_OVER":
        if b.state == "PLAYER_TURN" and not b.busy:
            cell = rng.choice(b.free_cells())
            move += 1
            if move == 2:
                say(f"put my cross in the {NAMES[cell]} please")
            else:
                app.trace("system", f"(the player draws a cross in the {NAMES[cell]})")
                app.scene_actions.put(("player_mark", cell))
            wait(lambda: b.state != "PLAYER_TURN", "Buddy to see my move", 60)
        elif b.state == "CONFIRM_MARK":
            say("yes")
            wait(lambda: b.state != "CONFIRM_MARK", "the confirmation", 30)
        elif b.state == "ROBOT_TURN":
            robot_moves += 1
            if robot_moves in (1, 2):
                wait(lambda: app.ctl.busy and not app.ctl.at_home, "Buddy to start drawing", 60)
                time.sleep(2.5)
                if robot_moves == 1:
                    app.trace("system", "(a hand reaches over the paper while Buddy draws)")
                    app.scene_actions.put(("reach", None))
                else:
                    app.trace("system", "(the player slides the sheet while Buddy draws)")
                    app.scene_actions.put(("slide", None))
            wait(lambda: b.state != "ROBOT_TURN", "Buddy's move", 120)
        else:
            time.sleep(0.1)
    wait(lambda: not b.busy, "the end of the game", 120)
    time.sleep(1.0)
    result["ok"] = True
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
w = winner(b.board)
print("\nfinal board (as read from the paper):")
for r in range(3):
    print("   " + " | ".join(b.board[3 * r + c] or " " for c in range(3)))
print(f"result: {'three in a row for ' + w[0] if w else 'draw'} | {time.time() - t0:.0f} s wall, {app.sim_time:.0f} s "
      f"simulated | difficulty now {100 * b.difficulty.strength:.0f}%")
sys.exit(0 if result["ok"] else 1)
