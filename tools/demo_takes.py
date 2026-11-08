"""The demo video's takes, scripted in the real simulator, with the time of every line (docs/demo-plan.md).

    .venv/bin/python tools/demo_takes.py teach      # situation 2: an under-explained game, two questions, read-back, play
    .venv/bin/python tools/demo_takes.py mark       # situation 1: a mark Penny cannot read, then an impossible spoken move

Typed input only, voice off, so the timings are the brain's and the arm's (add about 0.05 s to start each spoken
line with Piper, and the length of the speech itself). Prints a timeline and writes runtime/demo_take_<name>.json.
"""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEACH_LINES = ["My gran taught me a game. We take turns drawing a cross or a nought in the squares.",
               "You have to go right next to the last mark.",
               "Three in a row wins."]


def main():
    take = sys.argv[1] if len(sys.argv) > 1 else "teach"
    with tempfile.TemporaryDirectory(prefix="penny-take-") as tmp:
        os.environ["BUDDY_GAMES_FILE"] = str(Path(tmp) / "games.json")
        os.environ["BUDDY_PROFILE_FILE"] = str(Path(tmp) / "profile.json")
        from buddy import config as C
        from buddy.app import App

        app = App(voice=False, mic=False)
        b = app.brain
        t0 = time.monotonic()
        timeline, result = [], {"ok": False, "take": take}
        seen = [0]

        def log_new():
            for ev in list(b.events)[seen[0]:]:
                if ev.kind in ("said", "heard", "move", "game_start", "learned"):
                    timeline.append((round(time.monotonic() - t0, 1), ev.kind, ev.text))
            seen[0] = len(b.events)

        def wait(cond, what, timeout=240):
            end = time.monotonic() + timeout
            while not cond():
                log_new()
                if time.monotonic() > end:
                    raise TimeoutError(f"waiting for {what}; state={b.state}, last said {b.last_said!r}")
                time.sleep(0.05)
            log_new()

        def say(text, timeout=90):
            n = len(b.events)
            b.hear(text)
            wait(lambda: any(ev.kind == "said" for ev in list(b.events)[n:]), f"reply to {text[:40]!r}", timeout)

        def mark(t_label):
            timeline.append((round(time.monotonic() - t0, 1), "MARK", t_label))

        def teach():
            wait(lambda: app.tracker.track is not None, "the camera to find the sheet")
            say("Penny, let me teach you a game.")
            wait(lambda: b.state == "TEACH", "the teaching state")
            mark("UNCUT SEGMENT STARTS: the first, incomplete explanation")
            for line in TEACH_LINES:
                say(line)
            wait(lambda: b.state in ("NAME_GAME", "OFFER_TAUGHT"), "the read-back")
            mark("UNCUT SEGMENT ENDS: rules read back")
            if b.state == "NAME_GAME":
                say("Gran's game.")
            wait(lambda: b.state == "OFFER_TAUGHT", "the offer to play")
            say("Yes please.")
            wait(lambda: b.state in ("ASK_LEVEL", "SETUP", "WAIT_SHEET"), "setup")
            if b.state == "ASK_LEVEL":
                say("fair please")
            mark("grid drawing starts (cut or speed up here)")
            wait(lambda: b.state == "ASK_SIDE", "the grid", 240)
            mark("grid drawn and checked")
            say("Crosses please.")
            wait(lambda: b.state == "PLAYER_TURN", "the player's turn")
            app.scene_actions.put(("player_mark", 4))
            mark("the player draws a cross in the centre on the pad")
            wait(lambda: b.state == "ROBOT_TURN", "the mark read back", 60)
            wait(lambda: b.state in ("PLAYER_TURN", "GAME_OVER") and not b.busy, "Penny's move", 120)
            mark("Penny has moved next to the last mark")

        def mark_take():
            wait(lambda: app.tracker.track is not None, "the camera to find the sheet")
            say("Penny, let's play noughts and crosses.")
            wait(lambda: b.state in ("ASK_LEVEL", "SETUP"), "setup")
            if b.state == "ASK_LEVEL":
                say("fair please")
            wait(lambda: b.state == "ASK_SIDE", "the grid", 240)
            say("Crosses please.")
            wait(lambda: b.state == "PLAYER_TURN", "the player's turn")
            mark("UNCUT SEGMENT A STARTS: the player draws an unreadable mark")
            app.scene_actions.put(("player_eight", 2) if "player_eight" in getattr(app, "SCENE_ACTIONS", ()) else
                                  ("strokes", __import__("buddy.game.marks", fromlist=["x"]).eight_mark(2))
                                  if hasattr(__import__("buddy.game.marks", fromlist=["x"]), "eight_mark") else
                                  ("player_mark", 2))
            wait(lambda: b.state not in ("PLAYER_TURN",) or "read" in b.last_said.lower(), "Penny's question", 60)
            say("Yes.")
            wait(lambda: b.state in ("PLAYER_TURN", "ROBOT_TURN", "WAIT_SHEET", "SETUP") and not b.busy, "the answer handled", 180)
            mark("UNCUT SEGMENT A ENDS")
            wait(lambda: b.state == "PLAYER_TURN" and not b.busy, "the player's turn again", 180)
            mark("UNCUT SEGMENT B STARTS: an impossible spoken move")
            say("Put my cross in square ten.")
            say("Put my cross in the top right.")
            mark("UNCUT SEGMENT B ENDS")

        def player():
            try:
                teach() if take == "teach" else mark_take()
                result["ok"] = True
            except Exception as e:
                result["error"] = str(e)
            finally:
                app.running = False

        threading.Thread(target=player, daemon=True).start()
        try:
            app.run(until=lambda a: not a.running, realtime=True)
        finally:
            app.close()
        result["timeline"] = timeline
        Path(C.RUNTIME / f"demo_take_{take}.json").write_text(json.dumps(result, indent=2) + "\n")
        for t, kind, text in timeline:
            print(f"{t:7.1f}s  {kind:10} {text}")
        print(json.dumps({k: v for k, v in result.items() if k != "timeline"}))
        return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
