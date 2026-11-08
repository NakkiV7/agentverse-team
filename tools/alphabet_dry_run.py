"""The round-3 goal, end to end through the real path: the player's own sentence goes to the planner (Gemma 4 picks
custom_request), the compiler turns it into a routine, the paper routines write the alphabet in a free corner and
cross out the wrong letters so far, and a standing rule crosses out every wrong letter from then on. Questions
about the game are answered from the facts along the way.

    .venv/bin/python tools/alphabet_dry_run.py

Typed input only, no audio. Saves runtime/alphabet_overhead.png, runtime/alphabet_sheet.png and runtime/alphabet.json.
Exits 0 only when the camera sees the alphabet with exactly the wrong letters crossed and the questions were
answered with the right letters and numbers.
"""
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REQUEST = ("i want to see visually the alphabet as i play this. on the corner, write every letter of the alphabet "
           "and cross every letter i have got wrong from now on out and cross out the previous wrong letters too")


def main():
    with tempfile.TemporaryDirectory(prefix="buddy-alphabet-") as tmp:
        os.environ["BUDDY_GAMES_FILE"] = str(Path(tmp) / "games.json")
        os.environ["BUDDY_PROFILE_FILE"] = str(Path(tmp) / "profile.json")
        import cv2

        from buddy import config as C
        from buddy.agent import brain as brain_module
        from buddy.app import App
        from buddy.game.hangman import Hangman
        from buddy.paper.routines import GLYPH_MIN_INK, MARK_HALF_WIDTH_MM
        from buddy.perception.board import rectify
        from buddy.perception.sheet import check_lines

        class ScriptedHangman(Hangman):
            @classmethod
            def for_difficulty(cls, difficulty, rng=None):
                return cls("APPLE", level=difficulty)     # two P slots, and B C D are all wrong

        brain_module.Hangman = ScriptedHangman
        app = App(voice=False, mic=False)
        b = app.brain
        p = b.paper
        assert p is not None, "the paper routines are not installed in the brain"
        output = C.RUNTIME / "alphabet"
        result = {"ok": False, "answers": {}}
        t0 = time.monotonic()

        def wait(condition, what, timeout=600):
            deadline = time.monotonic() + timeout
            while not condition():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"waiting for {what}; state={b.state}, busy={b.busy}, paper pending={p.pending}, "
                                       f"last said {b.last_said!r}")
                time.sleep(.05)

        def say(text, timeout=90):
            previous = len(b.events)
            app.trace("system", f"(player types) {text}")
            b.hear(text)
            wait(lambda: any(ev.kind == "said" for ev in list(b.events)[previous:]), f"reply to {text!r}", timeout)
            return [ev.text for ev in list(b.events)[previous:] if ev.kind == "said"]

        def settled():
            return not b.busy and not p.pending

        def guess(letter):
            wait(settled, "drawing to finish")
            say(letter)
            wait(lambda: letter in b.hangman.guessed and settled(), f"checked guess and any rule for {letter}")

        WORDS = {"3": "three", "4": "four", "5": "five", "6": "six"}

        def has(word, text):          # a number may be said as a word ("four misses left")
            return any(re.search(rf"\b{re.escape(w)}\b", text, re.I) for w in {word, WORDS.get(word, word)})

        def ask(question, must, must_not=()):
            wait(settled, "drawing to finish")
            replies = " ".join(say(question))
            ok = all(has(m, replies) for m in must) and not any(has(m, replies) for m in must_not)
            result["answers"][question] = {"reply": replies, "ok": ok}
            app.trace("system", f"(answer check {'ok' if ok else 'WRONG'}: {replies!r})")
            return ok

        def player():
            try:
                wait(lambda: app.tracker.track is not None, "camera to find the sheet")
                say("Buddy, let's play hangman.")
                wait(lambda: b.state in ("ASK_LEVEL", "HANGMAN_TURN"), "the level question")
                if b.state == "ASK_LEVEL":
                    say("authentic please")
                wait(lambda: b.state == "HANGMAN_TURN" and not b.busy, "the checked dashes")
                assert b.hangman.word == "APPLE"
                for letter in "BC":
                    guess(letter)
                answers = [ask("what letters have I asked so far", ["B", "C"], ["D", "P"]),
                           ask("how many misses do I have left", ["4"], ["5", "6"])]
                # the player's own sentence, through the planner and the compiler
                tool_before = len(b.events)
                say(REQUEST, timeout=120)
                wait(lambda: "alphabet" in p.items or any("sorry" in ev.text.lower() or "can't" in ev.text.lower()
                                                           for ev in list(b.events)[tool_before:] if ev.kind == "said"),
                     "the compiler to produce the alphabet routine", 90)
                assert "alphabet" in p.items, "no alphabet item: " + " | ".join(
                    ev.text for ev in list(b.events)[tool_before:] if ev.kind == "said")
                result["rules"] = [r.get("name") for r in p.rules]
                wait(settled, "the alphabet to be written and the previous wrong letters crossed", 1200)
                result["write_seconds"] = round(time.monotonic() - t0, 1)
                for letter in "DP":
                    guess(letter)
                answers += [ask("which letters have I got wrong", ["B", "C", "D"], ["P"]),
                            ask("what have you written in the corner", ["alphabet"])]
                look = p._look()
                for suffix, rgb in (("overhead", look.rgb), ("sheet", rectify(look.rgb, app.cam, look.track.pose))):
                    if not cv2.imwrite(f"{output}_{suffix}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError(f"could not save {suffix} image")
                alphabet = p.items["alphabet"]
                glyphs = {c.char: p.glyph_ink(look, app, c) for c in alphabet.characters}
                diagonals = {c.char: check_lines(look.rgb, app.cam, look.track.pose, c.cross(),
                                                 half_width_mm=MARK_HALF_WIDTH_MM)[0] for c in alphabet.characters}
                crossed = {c for c, cov in diagonals.items() if cov >= C.LINE_OK_COVERAGE}
                result.update(guessed=sorted(b.hangman.guessed), crossed=sorted(crossed),
                              minimum_glyph_ink=min(glyphs.values()),
                              minimum_wrong_coverage=min(diagonals[c] for c in "BCD"),
                              maximum_unmarked_coverage=max(v for c, v in diagonals.items() if c not in "BCD"),
                              right_letter_coverage=diagonals["P"], description=p.describe(),
                              questions_ok=sum(answers), questions=len(answers))
                assert min(glyphs.values()) >= GLYPH_MIN_INK, "a written letter is missing from the camera image"
                assert crossed == set("BCD"), f"camera cross-outs are {sorted(crossed)}, expected B C D"
                assert sum(answers) >= 3, "too many questions answered wrongly"
                result["ok"] = True
            except Exception as error:
                result["error"] = str(error)
                app.trace("system", f"alphabet dry run failed: {error}")
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
        Path(f"{output}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
