"""Arthur meets Penny, learns Margaret's Footsteps, and plays two turns on the simulator.

    .venv/bin/python tools/residents_dry_run.py

Uses temporary game/rating files, the real planner, camera, arm and paper; no microphone or voice.
"""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

storage = tempfile.TemporaryDirectory(prefix="penny-residents-")
os.environ["BUDDY_GAMES_FILE"] = str(Path(storage.name) / "games.json")
os.environ["BUDDY_PROFILE_FILE"] = str(Path(storage.name) / "profile.json")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buddy.app import App  # noqa: E402
from buddy.game.rules import cell_name  # noqa: E402

app = App(voice=False, mic=False)
b = app.brain
ok = {"done": False}


def wait(cond, what, timeout=180):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise TimeoutError(f"waiting for {what}; state {b.state}; last reply {b.last_said!r}")
        time.sleep(0.1)


def say(text):
    app.trace("system", f"(player says) {text}")
    previous = len(b.events)
    b.hear(text)
    wait(lambda: any(e.kind == "said" for e in list(b.events)[previous:]), "Penny's reply", 60)


def player():
    time.sleep(2)
    say("Penny, I'm Arthur")
    assert b.resident.name == "Arthur" and b.difficulty.level == "hard" and b.level_asked
    assert b.last_said.startswith("Hello Arthur.")
    say("What games do you know?")
    wait(lambda: b.state == "CHOOSE_GAME", "the menu")
    assert "Footsteps" in b.last_said and "Margaret" in b.last_said
    say("Let's play Footsteps.")
    wait(lambda: b.state == "OFFER_TAUGHT", "Margaret's credit")
    assert b.last_said == "Margaret taught me this one. Would you like to learn it?"
    offered = len(b.events)
    say("Yes, I'd like to learn it.")
    wait(lambda: b.state == "ASK_SIDE", "the board")
    lesson = [e.text for e in list(b.events)[offered:] if e.kind == "said"]
    assert b.rules_text() in lesson, lesson
    assert lesson.index(b.rules_text()) < lesson.index("Lovely. Let me have a look at the paper.")
    assert not any(e.kind == "said" and e.text == b.LEVEL_QUESTION for e in b.events)
    say("Crosses please.")
    wait(lambda: b.state == "PLAYER_TURN" and not b.busy, "Arthur's first turn")
    for turn in range(2):
        before = sum(bool(mark) for mark in b.board)
        cell = b.legal_cells()[0]
        app.trace("system", f"(Arthur draws a cross in the {cell_name(b.spec, cell)})")
        app.scene_actions.put(("player_mark", cell))
        wait(lambda: b.state != "PLAYER_TURN", "Penny to read Arthur's mark", 60)
        if b.state == "CONFIRM_MARK":
            say("yes")
        wait(lambda: b.state == "PLAYER_TURN" and not b.busy and
             sum(bool(mark) for mark in b.board) >= before + 2, f"Penny's reply to turn {turn + 1}")
    state = app.state()
    assert state["caption"] == "Penny · playing Footsteps, taught by Margaret · with Arthur · level best"
    assert sum(e.kind == "move" and e.who == "player" for e in b.events) == 2
    out = Path(__file__).resolve().parent.parent / "runtime" / "residents.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(state, indent=2))
    app.trace("system", f"resident dry run passed: {state['caption']}; snapshot {out}")
    ok["done"] = True


def runner():
    try:
        player()
    except Exception as exc:
        app.trace("system", f"resident dry run stopped: {type(exc).__name__}: {exc}")
    finally:
        app.running = False


started = time.monotonic()
threading.Thread(target=runner, daemon=True).start()
app.run(until=lambda a: not a.running, realtime=False)
print(f"\nresident dry run: {'PASS' if ok['done'] else 'FAIL'} | {time.monotonic() - started:.1f} s wall")
app.close()
storage.cleanup()
sys.exit(0 if ok["done"] else 1)
