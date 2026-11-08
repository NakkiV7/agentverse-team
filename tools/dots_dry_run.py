"""Dots and boxes on the paper, scripted: Buddy draws the dot grid, the simulated hand draws the player's lines,
Buddy reads every line back from the camera, answers with its own, claims the boxes it closes, until every line is
drawn. No audio. The game pieces (buddy/game/dotsboxes.py, marks.py, perception/edges.py) are driven directly from
here; the brain states are set by name so the page and the scene actions behave as they will once brain.py owns
the loop (see docs/STATUS.md).

    .venv/bin/python tools/dots_dry_run.py              # 2x2 boxes (3x3 dots); exits 0 when the game completes
    .venv/bin/python tools/dots_dry_run.py --boxes 3    # 3x3 boxes (4x4 dots)
    .venv/bin/python tools/dots_dry_run.py --level easy
"""
import argparse
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buddy.app import App                                                           # noqa: E402
from buddy.game import dotsboxes as D                                               # noqa: E402
from buddy.game import marks as M                                                   # noqa: E402
from buddy.game.tictactoe import Difficulty                                         # noqa: E402
from buddy.perception.board import rectify                                          # noqa: E402
from buddy.perception.edges import drawn_mask, read_dots, read_edges                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--boxes", default="2", help="boxes per side, or ROWSxCOLS such as 3x4")
ap.add_argument("--level", default="impossible", choices=("easy", "medium", "hard", "impossible"))
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--first", default="player", choices=("player", "buddy"))
args = ap.parse_args()

N = int(args.boxes) if "x" not in args.boxes else tuple(int(v) for v in args.boxes.lower().split("x"))
PLAYER, BUDDY = "P", "B"
app = App(voice=False, mic=False)
b = app.brain
rng = random.Random(args.seed)
diff = Difficulty()
diff.set_level(args.level)
solver = D.Solver(N)
stats = {"done": False, "read_back": 0, "unsure": 0, "min_conf": 1.0, "solver_s": 0.0, "retries": 0, "dots_s": 0.0}


def say(text):
    b.say(text)


def wait(cond, what, timeout=120):
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"waiting for {what}; state {b.state}")
        time.sleep(0.1)


def read_board():
    """The edges as the camera sees them now (arm parked, hand away, sheet steady), or None."""
    look = app.look(timeout=20)
    if look is None or look.track is None or not look.track.visible:
        return None, None
    rect = rectify(look.rgb, app.cam, look.track.pose)
    res = read_edges(rect, N)
    stats["min_conf"] = min(stats["min_conf"], min(r.confidence for r in res))
    return res, rect


def wait_for_new_edges(known: int, timeout=90):
    """Watch the paper until at least one new line is there, every edge reads sure, and two looks agree."""
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        res, _ = read_board()
        if res is None:
            time.sleep(0.3)
            continue
        mask = drawn_mask(res)
        new = [e for e in range(D.n_edges(N)) if mask >> e & 1 and not known >> e & 1]
        unsure = [e for e, r in enumerate(res) if not r.sure]
        if unsure:
            stats["unsure"] += 1
            app.trace("see", "not sure about " + ", ".join(f"{D.edge_name(N, e)} ({100 * res[e].coverage:.0f}%)"
                                                            for e in unsure) + ": looking again")
        elif new and mask == last:
            return new, res
        last = mask
        time.sleep(0.3)
    raise TimeoutError("the player's line never showed up on the camera")


def setup():
    b.set_state("DOTS_SETUP", f"dry run: dots and boxes, {N}x{N} boxes, level {diff.level}")
    say("Lovely. Let me have a look at the paper.")
    look = app.look(timeout=20)
    sheet = look.sheet if look else None
    if sheet is None or sheet.confidence < 0.6 or sheet.cut_off:
        raise RuntimeError("no usable sheet in view")
    ok, why = app.dots_reachable(sheet.as_tuple(), N)
    app.trace("plan", f"{D.size_words(N)} boxes, dots {M.dot_spacing(N):.0f} mm apart"
                      + (f", {M.dot_shift(N):.0f} mm toward me" if M.dot_shift(N) else ", centred on the sheet") + f": {why}")
    if not ok:
        raise RuntimeError(why)
    say(D.describe(N))
    dots = M.dot_grid(N)
    t0 = time.time()
    for attempt in range(2):
        todo = list(range(len(dots)))
        if attempt == 1:
            seen = read_dots(read_board()[1], N)
            todo = [i for i, v in enumerate(seen) if v < 0.3]
            if not todo:
                break
            app.trace("decide", f"dot(s) {', '.join(str(i + 1) for i in todo)} did not come out: drawing them again")
        if attempt == 0:
            say("Let me draw the dots.")
        job = app.draw("the dots", [dots[i] for i in todo],
                       on_stroke=lambda k, n, todo=todo: app.trace("act", f"dot {todo[k] + 1}/{len(dots)}"))
        job.done.wait()
        if not job.ok:
            raise RuntimeError(f"drawing the dots failed: {job.error}")
    stats["dots_s"] = time.time() - t0
    seen = read_dots(read_board()[1], N)
    app.trace("check", "ink at each dot: " + ", ".join(f"{100 * v:.0f}%" for v in seen) + " (a dot counts above 30%)")
    if min(seen) < 0.3:
        raise RuntimeError("some dots are missing after two tries")
    say("There we are. You go first: join any two dots that are next to each other." if args.first == "player"
        else "There we are. I'll start.")


def player_turn(state):
    b.set_state("DOTS_PLAYER_TURN", "the player's turn")
    e = D.greedy_player(state, rng)
    app.trace("system", f"(the player draws {D.edge_name(N, e)})")
    app.scene_actions.put(("player_edge", (N, e)))
    new, res = wait_for_new_edges(state.edges)
    app.trace("see", "new line(s) on the paper: " + ", ".join(f"{D.edge_name(N, x)} ({100 * res[x].coverage:.0f}%)"
                                                              for x in new))
    stats["read_back"] += len(new)
    if len(new) > 1:
        say(f"I see {len(new)} new lines there. We take one each, but I'll let you have them.")
    for x in new:
        boxes = D.completed_by(N, state.edges, x)
        state = D.play(state, x)
        if boxes:
            names = " and ".join(D.box_name(N, k) for k in boxes)
            app.trace("decide", f"the player closed the {names} box: they go again")
            say(f"That closes the {names} box. It's yours, mark it, and have another go.")
            for k in boxes:
                app.scene_actions.put(("player_box", (N, k)))
            wait(lambda: not app.world.busy_animating() and app.scene_actions.empty(), "the player's mark", 60)
    return state


def robot_turn(state):
    b.set_state("DOTS_ROBOT_TURN", "Buddy's turn")
    ch = solver.choose(state, diff, rng)
    stats["solver_s"] = max(stats["solver_s"], solver.last_seconds)
    shown = ", ".join(f"{D.edge_name(N, e)} {s:+d}" for e, s in sorted(ch.scores.items(), key=lambda kv: -kv[1])[:3])
    how = "full search" if solver.last_depth == 0 else f"{solver.last_depth} moves deep"
    app.trace("plan", f"solver ({how}, {1000 * solver.last_seconds:.0f} ms), net boxes after each line: {shown} -> "
                      f"{D.edge_name(N, ch.cell)}: {ch.reason}")
    say(f"I'll take {D.edge_name(N, ch.cell)}.")
    for attempt in range(2):
        job = app.draw("my line", M.edge_stroke(N, ch.cell))
        job.done.wait()
        if not job.ok:
            raise RuntimeError(f"drawing failed: {job.error}")
        res, _ = read_board()
        got = res[ch.cell] if res else None
        app.trace("check", f"{D.edge_name(N, ch.cell)} now reads {'drawn' if got and got.drawn else 'missing'} "
                           f"({100 * got.coverage:.0f}% of it inked)" if got else "could not read the board")
        if got is not None and got.drawn and got.sure:
            stats["read_back"] += 1
            break
        stats["retries"] += 1
        app.trace("decide", "my line did not come out clearly: drawing it again")
    else:
        raise RuntimeError("my line never read back from the camera")
    boxes = D.completed_by(N, state.edges, ch.cell)
    state = D.play(state, ch.cell)
    if boxes:
        names = " and ".join(D.box_name(N, k) for k in boxes)
        app.trace("decide", f"that closes the {names} box: Buddy goes again")
        say(f"That closes the {names} box, so it's mine. I'll put my B in it and go again.")
        for k in boxes:
            app.draw("my B", M.box_claim(N, k)).done.wait()
    return state


def finish(state):
    p, q = state.score(PLAYER), state.score(BUDDY)
    b.set_state("DOTS_GAME_OVER", f"all lines drawn: player {p}, Buddy {q}")
    r = D.result(state)
    if r[0] == "draw":
        say(f"Every line is drawn and we have {p} boxes each. A draw!")
    elif r[1] == BUDDY:
        say(f"Every line is drawn. {q} boxes to me, {p} to you. My win, but well played!")
    else:
        say(f"Every line is drawn. {p} boxes to you, {q} to me. You win, well done!")
    app.trace("decide", "difficulty: " + diff.update("draw" if r[0] == "draw" else ("loss" if r[1] == BUDDY else "win")))


def player():
    time.sleep(1.0)
    setup()
    state = D.State.start(N, (PLAYER, BUDDY), first=PLAYER if args.first == "player" else BUDDY)
    while D.result(state) is None:
        state = player_turn(state) if state.to_move == PLAYER else robot_turn(state)
    finish(state)
    stats["state"] = state
    stats["done"] = True
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
total = D.n_edges(N)
s = stats.get("state")
print(f"\ndots and boxes {N}x{N}, level {diff.level}: "
      + (f"player {s.score(PLAYER)} boxes, Buddy {s.score(BUDDY)} | " if s else "did not finish | ")
      + f"{stats['read_back']}/{total} lines read back from the camera, {stats['retries']} redraws, "
      f"{stats['unsure']} unsure looks, lowest edge confidence {stats['min_conf']:.2f} | "
      f"dots {stats['dots_s']:.0f} s, slowest solver move {1000 * stats['solver_s']:.0f} ms | {time.time() - t0:.0f} s wall")
sys.exit(0 if stats["done"] and stats["read_back"] >= total else 1)
