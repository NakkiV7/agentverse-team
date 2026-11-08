"""Dots and boxes in conversation: the game loop that tools/dots_dry_run.py drove by hand, now owned by the brain.

States: DOTS_SETUP (find the sheet, draw the dot grid, check it) -> DOTS_PLAYER_TURN (watch the paper for the
player's new line; closing a box gives them another go) <-> DOTS_ROBOT_TURN (solver picks a line, Buddy draws it,
reads it back, claims boxes) -> GAME_OVER. The player draws on the real paper, or on the demo page's straightened
sheet with the mouse (the simulated hand copies the strokes). Every line, theirs and Buddy's, is read from the camera.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .. import config as C
from ..game import dotsboxes as D
from ..game import marks as M
from ..perception.board import rectify
from ..perception.edges import drawn_mask, read_dots, read_edges
from ..perception.sheet import ink_fraction

PLAYER, BUDDY = "P", "B"
STATES = ("DOTS_SETUP", "DOTS_PLAYER_TURN", "DOTS_ROBOT_TURN")


class DotsPlay:
    def __init__(self, brain, n: int = 2):
        self.b = brain
        self.app = brain.app
        self.n = n
        self.solver = D.Solver(n)
        self.state: D.State | None = None
        self.running = False
        self._last_mask = None

    # -- helpers ----------------------------------------------------------------------------------
    def say(self, text: str):
        self.b.say(text)

    def read_board(self):
        look = self.app.look(timeout=20)
        if look is None or look.track is None or not look.track.visible:
            return None, None
        rect = rectify(look.rgb, self.app.cam, look.track.pose)
        return read_edges(rect, self.n), rect

    def names(self, edges) -> str:
        return ", ".join(D.edge_name(self.n, e) for e in edges)

    # -- setup --------------------------------------------------------------------------------------
    def setup(self):
        """Worker: find a blank sheet within reach, draw the dots, check them, hand over to the player."""
        app, b, n = self.app, self.b, self.n
        b.set_state("DOTS_SETUP", f"dots and boxes, {n}x{n} boxes")
        asked = None
        while True:
            look = app.look(timeout=20)
            sheet = look.sheet if look else None
            if sheet is not None and sheet.cut_off:
                msg, key = "I can see the paper, but it's too far away. Could you slide it closer to me?", "cut"
            elif sheet is None or sheet.confidence < 0.6:
                msg, key = "I can't see a sheet of A4 paper. Could you put a blank sheet in front of me?", "none"
            else:
                ok, why = app.dots_reachable(sheet.as_tuple(), n)
                app.trace("plan", f"{n + 1}x{n + 1} dots, {M.dot_spacing(n):.0f} mm apart: {why}")
                inked = ink_fraction(look.rgb, app.cam, sheet.as_tuple(), C.GRID_SIZE_MM / 2 + 5)
                if not ok:
                    msg, key = "I can't quite reach the paper. Could you slide it a little closer to me?", "reach"
                elif inked > 0.01:
                    msg, key = "This sheet already has drawing on it. Could you give me a fresh one?", "used"
                else:
                    break
            if asked != key:
                self.say(msg)
                asked = key
            if b.state != "DOTS_SETUP":
                return
            time.sleep(1.0)
        if asked:
            self.say("Thank you, that's perfect.")
        self.say(D.describe(n))
        dots = M.dot_grid(n)
        for attempt in range(2):
            todo = list(range(len(dots)))
            if attempt == 1:
                _, rect = self.read_board()
                seen = read_dots(rect, n) if rect is not None else [0.0] * len(dots)
                todo = [i for i, v in enumerate(seen) if v < 0.3]
                if not todo:
                    break
                app.trace("decide", f"dot(s) {', '.join(str(i + 1) for i in todo)} did not come out: drawing them again")
            else:
                self.say("Let me draw the dots.")
            job = app.draw("the dots", [dots[i] for i in todo],
                           on_stroke=lambda k, m, todo=todo: app.trace("act", f"dot {todo[k] + 1}/{len(dots)}"))
            job.done.wait()
            if not job.ok:
                b.set_state("CHAT", "drawing the dots failed")
                self.say("I'm sorry, something went wrong while I was drawing the dots.")
                return
        _, rect = self.read_board()
        seen = read_dots(rect, n) if rect is not None else []
        if seen:
            app.trace("check", "ink at each dot: " + ", ".join(f"{100 * v:.0f}%" for v in seen) + " (a dot counts above 30%)")
        app.trace("decide", "difficulty: " + b.difficulty.start_game(b.game_name, b.rng))
        self.state = D.State.start(n, (PLAYER, BUDDY), first=PLAYER)
        self._last_mask = None
        self.running = True
        threading.Thread(target=self._watch, daemon=True, name="dots-watch").start()
        b.set_state("DOTS_PLAYER_TURN", "the dots are drawn; the player goes first")
        self.say("There we are. You go first: join any two dots that are next to each other, on the paper or on the "
                 "page. Closing a box gives you another go.")

    # -- the player's turn: watch the paper ------------------------------------------------------------
    def _watch(self):
        while self.running and self.b.state in STATES:
            time.sleep(0.4)
            if self.b.state == "DOTS_PLAYER_TURN" and not self.b.busy:
                try:
                    self._poll()
                except Exception as e:  # noqa: BLE001
                    self.app.trace("decide", f"error reading the lines: {e}")

    def _poll(self):
        res, _ = self.read_board()
        if res is None:
            return
        mask = drawn_mask(res)
        known = self.state.edges
        new = [e for e in range(D.n_edges(self.n)) if mask >> e & 1 and not known >> e & 1]
        if not new:
            self._last_mask = mask
            return
        unsure = [e for e, r in enumerate(res) if not r.sure]
        if unsure or mask != self._last_mask:          # wait for two agreeing, sure looks
            self._last_mask = mask
            return
        self.app.trace("see", "new line(s) on the paper: " + ", ".join(
            f"{D.edge_name(self.n, x)} ({100 * res[x].coverage:.0f}%)" for x in new))
        if len(new) > 1:
            self.say(f"I see {len(new)} new lines there. We take one each, but I'll let you have them.")
        again = False
        for x in new:
            boxes = D.completed_by(self.n, self.state.edges, x)
            self.state = D.play(self.state, x)
            if boxes:
                again = True
                names = " and ".join(D.box_name(self.n, k) for k in boxes)
                self.app.trace("decide", f"the player closed the {names} box: they go again")
                self.say(f"That closes the {names} box. It's yours, mark it, and have another go.")
        if D.result(self.state) is not None:
            self.finish()
        elif self.state.to_move == BUDDY:
            self.b.set_state("DOTS_ROBOT_TURN", "the player has drawn a line")
            self.b._worker(self.robot_turn)
        elif not again:
            self.say("Your turn again.")

    # -- Buddy's turn ----------------------------------------------------------------------------------
    def robot_turn(self):
        app, b, n = self.app, self.b, self.n
        while self.state.to_move == BUDDY and D.result(self.state) is None and b.state == "DOTS_ROBOT_TURN":
            ch = self.solver.choose(self.state, b.difficulty, b.rng)
            shown = ", ".join(f"{D.edge_name(n, e)} {s:+d}" for e, s in sorted(ch.scores.items(), key=lambda kv: -kv[1])[:3])
            how = "full search" if self.solver.last_depth == 0 else f"{self.solver.last_depth} moves deep"
            app.trace("plan", f"solver ({how}, {1000 * self.solver.last_seconds:.0f} ms), net boxes after each line: "
                              f"{shown} -> {D.edge_name(n, ch.cell)}: {ch.reason}")
            self.say(f"I'll take {D.edge_name(n, ch.cell)}.")
            drawn = False
            for attempt in range(2):
                job = app.draw("my line", M.edge_stroke(n, ch.cell))
                job.done.wait()
                if not job.ok:
                    self.say("I'm sorry, I couldn't draw that.")
                    return
                res, _ = self.read_board()
                got = res[ch.cell] if res else None
                app.trace("check", f"{D.edge_name(n, ch.cell)} now reads {'drawn' if got and got.drawn else 'missing'}"
                                   + (f" ({100 * got.coverage:.0f}% inked)" if got else ""))
                if got is not None and got.drawn and got.sure:
                    drawn = True
                    break
                app.trace("decide", "my line did not come out clearly: drawing it again")
            if not drawn:
                self.say("My line isn't showing up on the paper. Let me try once more later.")
            boxes = D.completed_by(n, self.state.edges, ch.cell)
            self.state = D.play(self.state, ch.cell)
            if boxes:
                names = " and ".join(D.box_name(n, k) for k in boxes)
                app.trace("decide", f"that closes the {names} box: Buddy goes again")
                self.say(f"That closes the {names} box, so it's mine. I'll put my B in it and go again.")
                for k in boxes:
                    app.draw("my B", M.box_claim(n, k)).done.wait()
        if D.result(self.state) is not None:
            self.finish()
        elif b.state == "DOTS_ROBOT_TURN":
            self._last_mask = None
            b.set_state("DOTS_PLAYER_TURN", "Buddy has drawn its line")
            self.say("Your turn.")

    # -- the end ----------------------------------------------------------------------------------
    def finish(self):
        self.running = False
        p, q = self.state.score(PLAYER), self.state.score(BUDDY)
        r = D.result(self.state)
        self.b.set_state("GAME_OVER", f"dots and boxes: all lines drawn, player {p}, Buddy {q}")
        if r[0] == "draw":
            self.say(f"Every line is drawn and we have {p} boxes each. A draw!")
            result = "draw"
        elif r[1] == BUDDY:
            self.say(f"Every line is drawn. {q} boxes to me, {p} to you. My win, but well played!")
            result = "loss"
        else:
            self.say(f"Every line is drawn. {p} boxes to you, {q} to me. You win, well done!")
            result = "win"
        self.app.trace("decide", "difficulty: " + self.b.difficulty.update(result))
        self.b._record(result)
        self.say("Shall we play another? Pop a fresh sheet down and say yes.")

    # -- words during the game -------------------------------------------------------------------------
    def act(self, u, text, sit):
        intent = u.intent
        if intent == "stop":
            self.running = False
            self.b.end_session()
        elif intent == "done":
            self._last_mask = None            # look again right now
            self.say("Let me look.")
        elif intent == "explain_rules":
            self.say(D.describe(self.n))
        elif intent in ("smalltalk", "unclear", "choose_game", "play_game", "move"):
            self.b._chat_reply(text, sit)
