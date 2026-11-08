"""Buddy learns dots and boxes from the player, from scratch, then plays it: the demo flow in the real simulator.

    .venv/bin/python tools/learn_dots_dry_run.py

Dots and boxes is hidden at the start (BUDDY_HIDE_GAMES). The player asks what there is to play (no dots and
boxes), asks to teach a new game, pastes an explanation in the style of a chat assistant, answers Buddy's two
questions, draws one example line on the paper for the camera, says yes, picks the level and the size, puts a fresh
sheet down, and plays one line each way. Exits 0 when the menu did not name the game before, Buddy asked exactly two
questions, saw the example, the game joined the menu, and the first lines were played.
"""
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXPLANATION = ("Dots and Boxes is a classic pencil-and-paper game for two players. You start with a rectangular grid of "
               "dots. Players take turns drawing a single horizontal or vertical line between two adjacent dots. When a "
               "player draws the line that completes the fourth side of a small box, they claim that box by writing "
               "their initial inside it, and they must take another turn. The game ends when all possible lines have "
               "been drawn, and the player who has claimed the most boxes wins.")

ANSWERS = {   # whatever Buddy asks, the player has an answer in the spirit of the explanation
    "turn": "Yes, you go again straight away, and you keep going as long as you keep closing boxes.",
    "end": "When every line is drawn. Whoever has more boxes wins, and it can be a draw.",
    "grid": "Any size you like. Let's start small, two boxes by two.",
    "start": "You can go first, or I will, it doesn't matter.",
    "write": "Your initial, so B for you.",
}


def answer_for(question: str) -> str:
    q = question.lower()
    if re.search(r"again|another turn|extra|next|after", q):
        return ANSWERS["turn"]
    if re.search(r"end|finish|over|win|wins|winner|most", q):
        return ANSWERS["end"]
    if re.search(r"size|big|how many|grid|dots across|rows", q):
        return ANSWERS["grid"]
    if re.search(r"start|first|begin", q):
        return ANSWERS["start"]
    if re.search(r"write|mark|initial|letter", q):
        return ANSWERS["write"]
    return ANSWERS["end"]


def main():
    with tempfile.TemporaryDirectory(prefix="buddy-learn-") as tmp:
        os.environ["BUDDY_GAMES_FILE"] = str(Path(tmp) / "games.json")
        os.environ["BUDDY_PROFILE_FILE"] = str(Path(tmp) / "profile.json")
        os.environ["BUDDY_HIDE_GAMES"] = "dots and boxes"
        from buddy import config as C                       # noqa: F401  (reads BUDDY_HIDE_GAMES)
        from buddy.app import App

        app = App(voice=False, mic=False)
        b = app.brain
        result = {"ok": False, "questions": [], "answers": []}
        t0 = time.monotonic()

        def wait(condition, what, timeout=240):
            deadline = time.monotonic() + timeout
            while not condition():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"waiting for {what}; state={b.state}, busy={b.busy}, last said {b.last_said!r}")
                time.sleep(.05)

        def say(text, timeout=90):
            previous = len(b.events)
            app.trace("system", f"(player types) {text[:90]}{'...' if len(text) > 90 else ''}")
            b.hear(text)
            wait(lambda: any(ev.kind == "said" for ev in list(b.events)[previous:]), f"reply to {text[:40]!r}", timeout)
            return [ev.text for ev in list(b.events)[previous:] if ev.kind == "said"]

        def player():
            try:
                wait(lambda: app.tracker.track is not None, "camera to find the sheet")
                menu = " ".join(say("Buddy, what games can we play?"))
                result["menu"] = menu
                assert "dots" not in menu.lower(), f"dots and boxes was on the menu already: {menu!r}"
                invite = " ".join(say("I want to teach you a new game and play it with you."))
                wait(lambda: b.state == "TEACH", "the teaching state")
                result["invite"] = invite
                replies = say(EXPLANATION, timeout=120)
                for _ in range(2):
                    question = replies[-1]
                    assert question.endswith("?"), f"expected a question, got {question!r}"
                    result["questions"].append(question)
                    reply = answer_for(question)
                    result["answers"].append(reply)
                    replies = say(reply, timeout=120)
                    if b.awaiting_example or not replies[-1].endswith("?"):
                        break
                assert len(result["questions"]) == 2, f"asked {len(result['questions'])} questions"
                wait(lambda: b.awaiting_example, "the request for an example move", 60)
                result["summary"] = next((t for t in replies if t.lower().startswith("so")), replies[0])
                time.sleep(2.0)                                   # the watcher takes its baseline look first
                app.trace("system", "(the player joins two dots with a line, bottom left of the sheet)")
                app.scene_actions.put(("strokes", [np.array([(-80.0, -60.0), (-79.0, -60.0)]),
                                                   np.array([(-40.0, -60.0), (-39.0, -60.0)]),
                                                   np.array([(-78.0, -60.0), (-42.0, -60.0)])]))
                wait(lambda: b.state == "OFFER_TAUGHT", "Buddy to see the example and offer the game", 120)
                result["example_seen"] = "see it" in b.last_said.lower() or any(
                    "see it" in ev.text.lower() for ev in list(b.events)[-4:] if ev.kind == "said")
                assert b.library.find("dots and boxes") is not None, "the game did not join the menu"
                say("Yes please.")
                wait(lambda: b.state in ("ASK_LEVEL", "DOTS_SIZE", "DOTS_SETUP"), "the level question")
                if b.state == "ASK_LEVEL":
                    say("authentic please")
                wait(lambda: b.state in ("DOTS_SIZE", "DOTS_SETUP"), "the size question")
                if b.state == "DOTS_SIZE":
                    say("two by two")
                wait(lambda: b.state == "DOTS_SETUP", "the dots setup")
                wait(lambda: "fresh" in b.last_said.lower() or "blank" in b.last_said.lower()
                     or b.state == "DOTS_PLAYER_TURN", "the fresh-sheet request", 90)
                if b.state != "DOTS_PLAYER_TURN":
                    app.trace("system", "(the player puts a fresh sheet down)")
                    app.scene_actions.put(("new_sheet", None))
                wait(lambda: b.state == "DOTS_PLAYER_TURN", "the dots to be drawn", 300)
                say("Could you draw my line on the top side of the top left box?")
                wait(lambda: b.state == "DOTS_ROBOT_TURN", "the player's line to be read back", 120)
                wait(lambda: b.state == "DOTS_PLAYER_TURN" and not b.busy, "Buddy's line", 120)
                result.update(ok=True, menu_after=b.library.spoken_list(),
                              lines_played=sum(1 for ev in b.events if ev.kind == "line"))
            except Exception as error:
                result["error"] = str(error)
                app.trace("system", f"learn dry run failed: {error}")
            finally:
                app.running = False

        thread = threading.Thread(target=player, daemon=True)
        thread.start()
        try:
            app.run(until=lambda a: not a.running, realtime=False)
            thread.join(timeout=5)
        finally:
            app.close()
        result["wall_seconds"] = round(time.monotonic() - t0, 1)
        Path(C.RUNTIME / "learn_dots.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
