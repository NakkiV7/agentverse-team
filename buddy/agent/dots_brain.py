"""Dots and boxes in conversation: the game loop that tools/dots_dry_run.py drove by hand, now owned by the brain.

States: DOTS_SIZE (which board; "three by four", up to 5 down by 6 across) -> DOTS_SETUP (find the sheet, draw the
dot grid, check it) -> DOTS_PLAYER_TURN (watch the paper for the player's new line; closing a box gives them
another go) <-> DOTS_ROBOT_TURN (solver picks a line, Buddy draws it, reads it back, claims boxes) -> GAME_OVER.

The player draws on the real paper or on the demo page's straightened sheet, or says a line ("the top of the top
left box") and Buddy draws it for them. A line that is too short to count is finished by Buddy. Ink that is not a
line between two neighbouring dots (a scribble, a diagonal, a line through a dot) means a fresh sheet: Buddy asks for
one and redraws the dots, every line drawn so far and every claimed box, then play goes on from the last proper move.
Talking is fine at any point: questions, difficulty, the rules, small talk.
"""
from __future__ import annotations

import re
import threading
import time

import numpy as np

from .. import config as C
from ..game import dotsboxes as D
from ..game import marks as M
from ..perception.board import rectify
from ..perception.edges import DRAWN_COVERAGE, drawn_mask, read_dots, read_edges
from ..perception.sheet import ink_fraction

PLAYER, BUDDY = "P", "B"
STATES = ("DOTS_SIZE", "DOTS_SETUP", "DOTS_PLAYER_TURN", "DOTS_ROBOT_TURN")
SHORT_COVERAGE = 0.3           # a line with at least this much of its length inked is "started": Buddy finishes it
MESS_LOOKS = 4                 # unsure or unexplained ink seen this many looks in a row (about 2 s) = not a proper line
MESS_EXTRA_INK = 0.012         # ink in the board area beyond what dots, lines and marks explain, as a fraction of the area


class DotsPlay:
    def __init__(self, brain, n=2):
        self.b = brain
        self.app = brain.app
        self.n = n
        self.solver = D.Solver(n)
        self.state: D.State | None = None
        self.running = False
        self._last_mask = None
        self._odd_looks = 0          # consecutive looks that did not make sense
        self._baseline_ink = 0.0     # ink fraction in the board area right after the last proper move

    # -- helpers ----------------------------------------------------------------------------------
    def say(self, text: str):
        self.b.say(text)

    def set_board(self, n):
        self.n = n
        self.solver = D.Solver(n)

    def look(self):
        look = self.app.look(timeout=20)
        if look is None or look.track is None or not look.track.visible:
            return None, None, None
        rect = rectify(look.rgb, self.app.cam, look.track.pose)
        return look, rect, read_edges(rect, self.n)

    def board_ink(self, rect) -> float:
        """Fraction of inked pixels in the board's area of the straightened sheet (dots, lines, marks and mess)."""
        from ..perception.edges import _ink
        from ..perception.board import mm_to_rect_px
        pts = np.array(M.dot_positions_mm(self.n))
        pad = M.dot_spacing(self.n) * 0.35
        lo = mm_to_rect_px([(pts[:, 0].min() - pad, pts[:, 1].max() + pad)])[0]
        hi = mm_to_rect_px([(pts[:, 0].max() + pad, pts[:, 1].min() - pad)])[0]
        ink = _ink(rect)
        y0, y1 = sorted((int(lo[1]), int(hi[1]))); x0, x1 = sorted((int(lo[0]), int(hi[0])))
        region = ink[max(0, y0):y1, max(0, x0):x1]
        return float(region.mean()) if region.size else 0.0

    # -- the size -----------------------------------------------------------------------------------
    def ask_size(self):
        self.b.set_state("DOTS_SIZE", "asking how big a board")
        self.say(f"How big a board would you like? Say the number of boxes across and down, for example three by "
                 f"three, up to {D.MAX_COLS} across and {D.MAX_ROWS} down. Or just say the usual, which is two by two.")

    def parse_size_text(self, text: str):
        t = text.lower()
        n = D.parse_size(t)
        if n is None and re.search(r"\b(usual|normal|default|small|two by two|don't mind|you choose|whatever)\b", t):
            n = (2, 2)
        return n

    def take_size(self, text: str) -> bool:
        """Read a size from what the player said; True when setup can start."""
        t = text.lower()
        n = D.parse_size(t)
        if n is None and re.search(r"\b(usual|normal|default|small|two by two|don't mind|you choose|whatever)\b", t):
            n = (2, 2)
        if n is None:
            return False
        ok, why = D.size_ok(n)
        if not ok:
            self.say(f"I'm sorry, {why}. What size shall we do?")
            return False
        self.set_board(n)
        return True

    # -- setup --------------------------------------------------------------------------------------
    def setup(self, restore: D.State | None = None):
        """Worker: find a blank sheet within reach, draw the dots (and, when restoring, every line and mark so
        far), check them, then hand over to whoever is to move."""
        app, b, n = self.app, self.b, self.n
        b.set_state("DOTS_SETUP", f"dots and boxes, {D.size_words(n)}" + (" (redrawing after a fresh sheet)" if restore else ""))
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
                app.trace("plan", f"{D.size_words(n)} boxes, dots {M.dot_spacing(n):.0f} mm apart: {why}")
                inked = ink_fraction(look.rgb, app.cam, sheet.as_tuple(), C.GRID_SIZE_MM / 2 + 5)
                if not ok:
                    if sheet.x > 0.19 and asked != "reach":
                        msg, key = "I can't quite reach the paper. Could you slide it a little closer to me?", "reach"
                    else:
                        b.set_state("DOTS_SIZE", "the board does not fit within reach")
                        self.say(f"I'm sorry, {D.size_words(n)} doesn't fit within my reach on this sheet. "
                                 "Shall we try a smaller board? Say a size, or the usual.")
                        return
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
        if restore is None:
            self.say(D.describe(n))
        dots = M.dot_grid(n)
        for attempt in range(2):
            todo = list(range(len(dots)))
            if attempt == 1:
                _, rect, _ = self.look()
                seen = read_dots(rect, n) if rect is not None else [0.0] * len(dots)
                todo = [i for i, v in enumerate(seen) if v < 0.3]
                if not todo:
                    break
                app.trace("decide", f"dot(s) {', '.join(str(i + 1) for i in todo)} did not come out: drawing them again")
            else:
                self.say("Let me draw the dots." if restore is None else "Let me draw the dots again.")
            job = app.draw("the dots", [dots[i] for i in todo],
                           on_stroke=lambda k, m, todo=todo: app.trace("act", f"dot {todo[k] + 1}/{len(dots)}"))
            job.done.wait()
            if not job.ok:
                b.set_state("CHAT", "drawing the dots failed")
                self.say("I'm sorry, something went wrong while I was drawing the dots.")
                return
        _, rect, _ = self.look()
        seen = read_dots(rect, n) if rect is not None else []
        if seen:
            app.trace("check", "ink at each dot: " + ", ".join(f"{100 * v:.0f}%" for v in seen) + " (a dot counts above 30%)")
        if restore is not None:
            self.state = restore
            edges = restore.drawn_edges()
            if edges:
                self.say(f"Now the {len(edges)} line{'s' if len(edges) != 1 else ''} we had drawn.")
                app.draw("the lines so far", [M.edge_stroke(n, e)[0] for e in edges],
                         on_stroke=lambda k, m, edges=edges: app.trace("act", f"line {k + 1}/{len(edges)}: {D.edge_name(n, edges[k])}")).done.wait()
            for k, owner in enumerate(restore.owners):
                if owner == BUDDY:
                    app.draw("my B", M.box_claim(n, k)).done.wait()
                elif owner == PLAYER:
                    app.draw("your box", M.box_claim(n, k, style="circle")).done.wait()
            app.trace("decide", f"restored the position: {len(edges)} lines, player {restore.score(PLAYER)} boxes, "
                                f"Buddy {restore.score(BUDDY)}; {'the player' if restore.to_move == PLAYER else 'Buddy'} to move")
        else:
            app.trace("decide", "difficulty: " + b.difficulty.start_game(b.game_name, b.rng))
            self.state = D.State.start(n, (PLAYER, BUDDY), first=PLAYER)
        _, rect, _ = self.look()
        self._baseline_ink = self.board_ink(rect) if rect is not None else 0.0
        self._last_mask = None
        self._odd_looks = 0
        if not self.running:
            self.running = True
            threading.Thread(target=self._watch, daemon=True, name="dots-watch").start()
        if self.state.to_move == BUDDY:
            b.set_state("DOTS_ROBOT_TURN", "restored; Buddy to move")
            self.say("There we are, back where we were. My turn.")
            b._worker(self.robot_turn)
        else:
            b.set_state("DOTS_PLAYER_TURN", "the dots are drawn; the player goes first" if restore is None else "restored; the player to move")
            self.say("There we are. You go first: join any two dots that are next to each other, on the paper, on the "
                     "page, or just tell me which line. Closing a box gives you another go." if restore is None
                     else "There we are, back where we were. Your turn.")

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
        _, rect, res = self.look()
        if res is None:
            return
        mask = drawn_mask(res)
        known = self.state.edges
        new = [e for e in range(D.n_edges(self.n)) if mask >> e & 1 and not known >> e & 1]
        unsure = [e for e, r in enumerate(res) if not r.sure and not known >> e & 1]
        started = [e for e, r in enumerate(res) if not known >> e & 1 and SHORT_COVERAGE <= r.coverage < DRAWN_COVERAGE]
        extra_ink = self.board_ink(rect) - self._baseline_ink
        if not new and not unsure and not started and extra_ink < MESS_EXTRA_INK:
            self._last_mask, self._odd_looks = mask, 0
            return
        if new and not unsure and mask == self._last_mask and extra_ink < MESS_EXTRA_INK + 0.004 * len(new):
            self._odd_looks = 0
            return self._accept(new, res)
        # something is not a clean line yet: a half line, a scribble, or ink still arriving
        self._last_mask = mask
        self._odd_looks += 1
        if self._odd_looks < MESS_LOOKS:
            return
        self._odd_looks = 0
        if started and not new and extra_ink < MESS_EXTRA_INK:
            return self._finish_short(started[0], res[started[0]].coverage)
        self.app.trace("see", f"ink that is not a line between two dots: {len(new)} new, {len(unsure)} unsure, "
                              f"extra ink {100 * extra_ink:.1f}% of the board area")
        self._mess()

    def _accept(self, new, res):
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
                self.app.trace("decide", f"the player closed the {names}: they go again")
                self.say(f"That closes the {names}. It's yours, mark it, and have another go.")
        _, rect, _ = self.look()
        if rect is not None:
            self._baseline_ink = self.board_ink(rect) + (0.004 * sum(1 for x in new if D.completed_by(self.n, self.state.edges & ~(1 << x), x)))
        self._after_player(again)

    def _after_player(self, again: bool):
        if D.result(self.state) is not None:
            self.finish()
        elif self.state.to_move == BUDDY:
            self.b.set_state("DOTS_ROBOT_TURN", "the player has drawn a line")
            self.b._worker(self.robot_turn)
        elif not again:
            self.say("Your turn again.")

    def _finish_short(self, e: int, coverage: float):
        """A line that was started but is too short to count: Buddy completes it and counts it as the player's."""
        self.app.trace("decide", f"{D.edge_name(self.n, e)} is only {100 * coverage:.0f}% drawn: finishing it for the player")
        self.say(f"That line looks a little short. Let me finish it for you.")
        self.b._worker(self._draw_for_player, e)

    def _draw_for_player(self, e: int):
        job = self.app.draw("the player's line", M.edge_stroke(self.n, e))
        job.done.wait()
        _, rect, res = self.look()
        if res is None or not res[e].drawn:
            self.say("Hmm, that line still isn't clear. Let me try once more.")
            self.app.draw("the player's line", M.edge_stroke(self.n, e)).done.wait()
            _, rect, res = self.look()
        if res is not None and res[e].drawn:
            self._accept([e], res)
        else:
            self._mess()

    def _mess(self):
        """Ink that is not a proper line: a fresh sheet, then everything so far is redrawn."""
        self.b.set_state("DOTS_SETUP", "the paper has marks that are not lines between dots: fresh sheet")
        self.say("I'm not sure what that mark is, it isn't a line between two dots next to each other. "
                 "Let's take a fresh sheet; I'll draw everything again up to your last proper move.")
        self.b._worker(self.setup, self.state)

    # -- spoken lines ------------------------------------------------------------------------------
    def parse_edge(self, text: str):
        """'the top of the top left box', 'left side of the centre', 'bottom row, second box, right side'."""
        t = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        t = re.sub(r"\bsquare\b", "box", t)
        for w, v in D.NUMBER_WORDS.items():
            t = re.sub(rf"\b{w}\b", str(v), t)
        best, best_score = None, 0
        for e in range(D.n_edges(self.n)):
            if self.state.edges >> e & 1:
                continue
            words = [w for w in re.sub(r"[^a-z0-9 ]+", " ", D.edge_name(self.n, e).lower()).split() if w not in ("the", "of", "side", "box", "in")]
            if all(re.search(rf"\b{w}\b", t) for w in words):
                score = len(words)
                if score > best_score:
                    best, best_score = e, score
        return best

    def _spoken_line(self, text: str):
        e = self.parse_edge(text)
        if e is None:
            self.say("Which line? Say the side and the box, for example the top side of the top left box, or draw it.")
            return
        self.app.trace("decide", f"the player asked me to draw {D.edge_name(self.n, e)} for them")
        self.say(f"{D.edge_name(self.n, e).capitalize()}. Here it comes.")
        self.b._worker(self._draw_for_player, e)

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
                _, rect, res = self.look()
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
                app.trace("decide", f"that closes the {names}: Buddy goes again")
                self.say(f"That closes the {names}, so it's mine. I'll put my B in it and go again.")
                for k in boxes:
                    app.draw("my B", M.box_claim(n, k)).done.wait()
        _, rect, _ = self.look()
        if rect is not None:
            self._baseline_ink = self.board_ink(rect)
        if D.result(self.state) is not None:
            self.finish()
        elif b.state == "DOTS_ROBOT_TURN":
            self._last_mask = None
            self._odd_looks = 0
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
        intent, st = u.intent, self.b.state
        if intent == "stop":
            self.running = False
            self.b.end_session()
            return
        if st == "DOTS_SIZE":
            if intent in ("no",) or (intent == "choose_game" and u.game_name and "dots" not in u.game_name.lower()):
                self.b.set_state("CHOOSE_GAME", "the player changed their mind about dots and boxes")
                self.say(f"No problem. Which would you like instead: {self.b.library.spoken_list()}?")
            elif self.take_size(text):
                self.say(f"{D.size_words(self.n).capitalize()} it is.")
                self.b._worker(self.setup)
            elif intent in ("smalltalk", "unclear", "explain_rules"):
                if intent == "explain_rules":
                    self.say(D.describe(self.n))
                self.say("How many boxes across and down? Two by two is the usual.")
            return
        if intent == "explain_rules":
            self.say(D.describe(self.n) + (" " + self._score_line() if self.state else ""))
        elif intent == "done":
            self._last_mask = None
            self.say("Let me look.")
        elif st == "DOTS_PLAYER_TURN" and (intent == "move" or re.search(r"\b(side|top|bottom|left|right)\b", text.lower())
                                             and re.search(r"\b(box|square|row|column|line)\b", text.lower())):
            self._spoken_line(text)
        elif intent in ("yes", "no") and st == "DOTS_PLAYER_TURN":
            self.say("It's your line. Draw it, or tell me which one, like the left side of the top right box.")
        else:
            self.b._chat_reply(text, sit + " " + self._score_line())

    def _score_line(self) -> str:
        if not self.state:
            return ""
        return (f"Score so far: the player {self.state.score(PLAYER)} boxes, Buddy {self.state.score(BUDDY)}; "
                f"{bin(self.state.edges).count('1')} of {D.n_edges(self.n)} lines drawn.")
