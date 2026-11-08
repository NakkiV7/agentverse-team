"""Real arm/camera checks for a paper routine, with no voice or microphone.

    .venv/bin/python tools/paper_dry_run.py
    .venv/bin/python tools/paper_dry_run.py --game dots

The JSON below is hand-written in the shared compiler contract. No compiler or
brain source changes are needed. The dots run places the sheet 30 mm closer to
the arm: at the default pose its bottom edge is outside the pen's lift reach.
The alphabet must finish in under four minutes of simulated arm time, measured
from its first arm job to its last glyph check (also reported in wall seconds).
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALPHABET_ROUTINE = {
    "say": "Of course. I'll write the alphabet in the top right corner and cross out each wrong letter.",
    "now": [
        {"op": "write_text", "id": "alphabet", "text": "ABCDEFGHIJKLM\nNOPQRSTUVWXYZ",
         "place": "top right corner", "size_mm": 6},
        {"op": "cross_out", "target": {"id": "alphabet", "chars": "{wrong}"}},
    ],
    "rules": [{"name": "cross out wrong letters", "when": {"kind": "guess", "who": "player", "correct": False},
               "do": [{"op": "cross_out", "target": {"id": "alphabet", "chars": "{letter}"}}]}],
    "cancel": [],
}
SCORE_ROUTINE = {
    "say": "I'll keep the scores along the bottom, yours first and mine second.",
    "now": [{"op": "write_text", "id": "score", "text": "0:0", "place": "bottom edge", "size_mm": 5}],
    "rules": [{"name": "score after each box", "when": {"kind": "box"},
               "do": [{"op": "append_text", "id": "score", "text": "/{score_player}:{score_buddy}"}]}],
    "cancel": [],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=("hangman", "dots"), default="hangman")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="buddy-paper-") as tmp:
        os.environ["BUDDY_GAMES_FILE"] = str(Path(tmp) / "games.json")
        os.environ["BUDDY_PROFILE_FILE"] = str(Path(tmp) / "profile.json")
        import cv2
        from buddy import config as C
        from buddy.app import App
        from buddy.agent import brain as brain_module
        from buddy.game.hangman import Hangman
        from buddy.game import dotsboxes as D, marks as M
        from buddy.paper import Paper, context_from
        from buddy.paper.routines import GLYPH_MIN_INK, MARK_HALF_WIDTH_MM
        from buddy.perception.board import rectify
        from buddy.perception.sheet import check_lines

        class ScriptedHangman(Hangman):
            @classmethod
            def for_difficulty(cls, difficulty, rng=None):
                return cls("APPLE", level=difficulty)

        brain_module.Hangman = ScriptedHangman
        app = App(voice=False, mic=False)
        b = app.brain
        writing_starts, writing_checks = [], []
        trace = app.trace

        def timed_trace(stage, message):
            trace(stage, message)
            if message.startswith("paper alphabet:"):
                stamp = (time.time() - app.t0, app.sim_time)
                if stage == "act" and message.endswith("moving the pen above the sheet"):
                    writing_starts.append(stamp)
                elif stage == "paper" and "glyph ink" in message:
                    writing_checks.append(stamp)
        app.trace = timed_trace
        b.rng = random.Random(3)
        p = b.paper                      # installed by the brain (paper_glue); a bare brain would need Paper(b)
        if p is None:
            p = b.paper = Paper(b)
            b.listeners.append(p.on_event)
        # Demonstrate the coordinator's one-line guard without editing brain.py.
        hangman_act = b._hangman_act

        def act(u):
            if not p.defer_guess(u):
                hangman_act(u)
        b._hangman_act = act
        if args.game == "dots":
            app.world.move_sheet(.18, 0, C.SHEET_POSE0[2])
            app.trace("system", "placed the sheet 30 mm nearer the arm for the bottom score strip")
        result = {"ok": False, "game": args.game}
        output = C.RUNTIME / ("paper" if args.game == "hangman" else "paper_dots")
        started = time.monotonic()

        def wait(condition, what, timeout=600):
            deadline = time.monotonic() + timeout
            while not condition():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"waiting for {what}; state={b.state}, paper={p.last_error!r}")
                time.sleep(.05)

        def say(text):
            previous = len(b.events)
            app.trace("system", f"(player types) {text}")
            b.hear(text)
            wait(lambda: any(ev.kind == "said" for ev in list(b.events)[previous:]), f"reply to {text}", 90)

        def guess(letter):
            wait(lambda: not b.busy and not p.pending, "drawing to finish")
            say(letter)
            wait(lambda: letter in b.hangman.guessed and not b.busy and not p.pending,
                 f"checked guess and rule for {letter}")

        def routine(spec):
            b.say(spec["say"])
            for name in spec["cancel"]:
                p.cancel(name)
            for rule in spec["rules"]:
                p.add_rule(rule)
            error = p.run(spec["now"], context_from(b))
            if error:
                raise AssertionError(error)

        def save(look):
            for suffix, rgb in (("overhead", look.rgb), ("sheet", rectify(look.rgb, app.cam, look.track.pose))):
                if not cv2.imwrite(f"{output}_{suffix}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"could not save {suffix} image")

        def setup(game, final_state):
            wait(lambda: app.tracker.track is not None, "camera sheet detection")
            say(f"Buddy, let's play {game}.")
            wait(lambda: b.state in ("ASK_LEVEL", "DOTS_SIZE", final_state), "level question")
            if b.state == "ASK_LEVEL":
                say("authentic please")
            if args.game == "dots":
                wait(lambda: b.state in ("DOTS_SIZE", final_state), "board size question")
                if b.state == "DOTS_SIZE":
                    say("two by two please")
            wait(lambda: b.state == final_state and not b.busy, "the checked game setup")

        def hangman():
            setup("hangman", "HANGMAN_TURN")
            assert b.hangman.word == "APPLE"
            for letter in "BC":
                guess(letter)
            routine(ALPHABET_ROUTINE)
            for letter in "DFP":
                guess(letter)
            look = p._look()
            save(look)
            alphabet = p.items["alphabet"]
            glyphs = {c.char: p.glyph_ink(look, app, c) for c in alphabet.characters}
            diagonals = {c.char: check_lines(look.rgb, app.cam, look.track.pose, c.cross(),
                                           half_width_mm=MARK_HALF_WIDTH_MM)[0] for c in alphabet.characters}
            crossed = {c for c, coverage in diagonals.items() if coverage >= C.LINE_OK_COVERAGE}
            result.update(guessed=sorted(b.hangman.guessed), wrong=sorted(b.hangman.guessed - set(b.hangman.word)),
                          glyph_ink=glyphs, diagonal_coverage=diagonals, crossed=sorted(crossed),
                          minimum_wrong_coverage=min(diagonals[c] for c in "BCDF"),
                          maximum_unmarked_coverage=max(v for c, v in diagonals.items() if c not in "BCDF"),
                          right_letter_coverage=diagonals["P"],
                          rows=len({c.box.y0 for c in alphabet.characters}), description=p.describe())
            assert writing_starts and writing_checks, "missing alphabet drawing timing in the trace"
            write_sim_seconds = writing_checks[-1][1] - writing_starts[0][1]
            result.update(alphabet_jobs=len(writing_starts),
                          alphabet_first_job_at=round(writing_starts[0][0], 2),
                          alphabet_last_check_at=round(writing_checks[-1][0], 2),
                          alphabet_write_seconds=round(writing_checks[-1][0] - writing_starts[0][0], 2),
                          alphabet_write_sim_seconds=round(write_sim_seconds, 2))
            assert write_sim_seconds < 240, f"alphabet took {write_sim_seconds:.2f} s of arm time, expected under 240 s"
            assert min(glyphs.values()) >= GLYPH_MIN_INK, "a written glyph is absent from the camera"
            assert crossed == set("BCDF"), f"camera cross-outs are {crossed}, expected BCDF"
            assert b.hangman.guessed == set("BCDFP")
            assert {c.char for c in alphabet.characters if "cross_out" in c.marks} == crossed
            assert p.last_error is None

        def dots():
            setup("dots and boxes", "DOTS_PLAYER_TURN")
            routine(SCORE_ROUTINE)
            scores = []
            settled = threading.Event()
            settled.set()
            accept = b.dots._accept

            def checked_accept(*args):
                # A board update precedes its final camera look and turn
                # transition. Observe completion instead of racing that gap.
                settled.clear()
                try:
                    return accept(*args)
                finally:
                    settled.set()
            b.dots._accept = checked_accept

            def record(ev):
                if ev.kind == "box":
                    scores.append((b.dots.state.score("P"), b.dots.state.score("B")))
            b.listeners.append(record)
            rng = random.Random(7)
            while b.state != "GAME_OVER":
                if (b.state == "DOTS_PLAYER_TURN" and b.dots.state.to_move == "P"
                        and settled.is_set() and not b.busy and not p.pending):
                    state = b.dots.state
                    edge = D.greedy_player(state, rng)
                    app.scene_actions.put(("player_edge", (2, edge)))
                    wait(lambda: b.dots.state.edges != state.edges, "player line read from camera")
                    wait(lambda: settled.is_set() and not b.busy and not p.pending,
                         "box rules and Buddy's turn")
                else:
                    time.sleep(.05)
            wait(lambda: not b.busy and not p.pending, "final score written")
            expected = "0:0" + "".join(f"/{a}:{z}" for a, z in scores)
            item = p.items["score"]
            look = p._look()
            save(look)
            ink = [p.glyph_ink(look, app, c) for c in item.characters]
            coverage = check_lines(look.rgb, app.cam, look.track.pose,
                                   M.stroke_segments([s for c in item.characters for s in c.strokes()]))
            result.update(score_events=scores, expected=expected, written=item.text,
                          minimum_glyph_ink=min(ink), minimum_segment_coverage=min(coverage),
                          boxes=b.dots.state.score("P") + b.dots.state.score("B"),
                          sheet_pose=list(look.track.pose), description=p.describe())
            assert item.text == expected and len(scores) >= 2
            assert min(ink) >= GLYPH_MIN_INK
            assert result["boxes"] == 4 and p.last_error is None

        def player():
            try:
                (hangman if args.game == "hangman" else dots)()
                result["ok"] = True
            except Exception as exc:
                import traceback
                traceback.print_exc()
                result["error"] = str(exc)
                app.trace("system", f"paper dry run failed: {exc}")
            finally:
                app.running = False

        thread = threading.Thread(target=player, daemon=True)
        thread.start()
        try:
            app.run(until=lambda a: not a.running, realtime=False)
            thread.join(timeout=5)
        finally:
            p.close()
            app.close()
        result["wall_seconds"] = round(time.monotonic() - started, 1)
        Path(f"{output}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
