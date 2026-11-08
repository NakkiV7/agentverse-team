"""Play hangman on paper through the real typed-input and local Gemma path.

    .venv/bin/python tools/hangman_dry_run.py         # win after five misses
    .venv/bin/python tools/hangman_dry_run.py --lose  # all six gallows parts

Checks a hint, ambiguous input, repeated letters, duplicate guesses and the saved
result. Saves overhead and straightened-sheet images under runtime/. No audio.
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



def _wait_level(b, timeout=20):
    """Buddy asks how hard to play before the first game; give it a moment to ask."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if b.state == "ASK_LEVEL":
            return True
        if b.state not in ("CHOOSE_GAME", "OFFER_TAUGHT", "CHAT", "IDLE"):
            return False
        time.sleep(0.1)
    return False

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lose", action="store_true", help="Play six misses to check the loss ending.")
    args = parser.parse_args()
    # Keep the player's real library history intact during this scripted check.
    with tempfile.TemporaryDirectory(prefix="buddy-hangman-") as tmp:
        os.environ["BUDDY_GAMES_FILE"] = str(Path(tmp) / "games.json")
        import cv2

        from buddy import config as C
        from buddy.app import App
        from buddy.game.library import Library
        from buddy.game.marks import stroke_segments
        from buddy.perception.board import rectify
        from buddy.perception.sheet import check_lines

        app = App(voice=False, mic=False)
        b = app.brain
        b.rng.seed(139)  # APPLE: two P slots exercise repeated-letter drawing.
        expected = "loss" if args.lose else "win"
        output = C.RUNTIME / f"hangman_{expected}"
        result = {"ok": False, "expected": expected}
        t0 = time.monotonic()

        def wait(condition, what, timeout=180):
            end = time.monotonic() + timeout
            while not condition():
                if time.monotonic() > end:
                    raise TimeoutError(f"Waiting for {what}; state={b.state}; Buddy said {b.last_said!r}")
                time.sleep(.1)

        def say(text):
            previous = b.last_said
            b.hear(text)
            wait(lambda: b.last_said != previous, f"reply to {text!r}", 60)

        def guess(c):
            wait(lambda: not b.busy, "the paper check")
            say(c)
            wait(lambda: c in b.hangman.guessed and not b.busy, f"verified letter {c}")

        def player():
            try:
                wait(lambda: app.tracker.track is not None, "camera to find the sheet")
                say("Buddy, let's play hangman.")
                if _wait_level(b):
                    say("authentic please")
                wait(lambda: b.state == "HANGMAN_TURN" and not b.busy, "the checked dashes")
                assert b.hangman.word == "APPLE"
                say("Can I have a hint?")
                wait(lambda: not b.busy, "hint")
                assert not b.hangman.guessed
                say("A or E")
                assert not b.hangman.guessed
                guess("P")
                assert b.hangman.pattern == "_PP__"
                say("P")
                assert b.hangman.guessed == {"P"} and "already" in b.last_said
                for c in ("BCDFGH" if args.lose else "BCDFGALE"):
                    guess(c)
                wait(lambda: b.state == "GAME_OVER" and not b.busy, "recorded result")
                assert b.hangman.result() == expected
                saved = Library(b.library.path).find("hangman")
                assert saved.plays == 1
                assert (saved.robot_wins if args.lose else saved.player_wins) == 1
                look = b._hangman_look()
                assert look is not None
                coverage = check_lines(look.rgb, app.cam, look.track.pose, stroke_segments(b._hangman_strokes))
                assert min(coverage) >= C.LINE_OK_COVERAGE
                for suffix, rgb in (("overhead", look.rgb), ("sheet", rectify(look.rgb, app.cam, look.track.pose))):
                    path = f"{output}_{suffix}.png"
                    if not cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError(f"Could not save {path}")
                result.update(ok=True, word=b.hangman.word, misses=b.hangman.misses,
                              guessed=sorted(b.hangman.guessed), minimum_coverage=min(coverage),
                              plays=saved.plays, player_wins=saved.player_wins, robot_wins=saved.robot_wins)
            except Exception as error:
                result["error"] = str(error)
                app.trace("system", f"hangman dry run failed: {error}")
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
