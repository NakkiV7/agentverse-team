"""Buddy's brain: the conversation state machine and the game setup behaviour.

Every input goes through the same visible steps, each written to the trace shown on the demo page:

  HEAR        what the microphone (Whisper) or the keyboard gave us
  UNDERSTAND  intent from the phrase rules (instant) or from Gemma (when the rules do not match)
  DECIDE      the state machine picks what to do; the language model never picks actions
  SEE / PLAN / ACT / SENSE / CHECK   the setup behaviour: find the sheet, plan the grid on it, draw, verify
  SAY         what Buddy says (fixed lines for the scripted steps, Gemma's reply for small talk)

States: IDLE -> CHAT -> CHOOSE_GAME -> SETUP (with WAIT_SHEET while there is no usable sheet) -> ASK_SIDE -> READY
"""
from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from .. import config as C
from ..motion.controller import Job
from ..motion.drawing import grid_lines_mm, sheet_to_world
from ..perception.sheet import check_lines, find_sheet, ink_fraction
from . import rules
from .llm import understand

if TYPE_CHECKING:
    from ..app import App

OFFER = "I'd love to. I know noughts and crosses. Shall we play that?"
ONLY_TTT = "I'm afraid I only know noughts and crosses for now. Shall we play that?"
RULES = ("Here's how it works. We take turns putting our mark in one of the nine squares. One of us draws crosses, "
         "the other noughts. The first to get three in a row, across, down or corner to corner, wins.")
SITUATION = {
    "IDLE": "You were resting; the player just said your name.",
    "CHAT": "You are chatting. You would like to offer a game of noughts and crosses.",
    "CHOOSE_GAME": "You asked whether they would like to play noughts and crosses.",
    "WAIT_SHEET": "You are waiting for the player to put a sheet of paper where you can reach it.",
    "SETUP": "You are drawing the noughts and crosses board on the paper.",
    "ASK_SIDE": "The board is drawn. You asked whether they want crosses or noughts; crosses go first.",
    "READY": "The board is ready and the player knows whose turn it is.",
}


class Brain:
    def __init__(self, app: "App"):
        self.app = app
        self.state = "IDLE"
        self.inputs: "queue.Queue" = queue.Queue()
        self.history: list[tuple[str, str]] = []
        self.last_said = ""
        self.lines_mm = grid_lines_mm()
        self.plan_pose = None
        self.coverage = None
        self.player_side = None
        threading.Thread(target=self._loop, daemon=True, name="brain").start()

    # ------------------------------------------------------------------ helpers
    def set_state(self, new: str, why: str):
        if new != self.state:
            self.app.trace("decide", f"{self.state} -> {new}: {why}")
            self.state = new

    def say(self, text: str, wait: bool = False):
        self.last_said = text
        self.history.append(("buddy", text))
        self.app.transcript_add("Buddy", text)
        done = self.app.say(text)
        if wait and done is not None:
            done.wait(timeout=30)

    def hear(self, text: str, source: str = "typed", stt_s: float = 0.0):
        self.inputs.put(("text", text, source, stt_s))

    # ------------------------------------------------------------------ main loop
    def _loop(self):
        while True:
            kind, *rest = self.inputs.get()
            if kind == "text":
                try:
                    self._on_text(*rest)
                except Exception as e:                      # keep the brain alive whatever happens
                    self.app.trace("decide", f"error handling input: {e}")

    def _understand(self, text: str):
        r = rules.match(text, self.state)
        if r is not None:
            intent, game, side = r
            self.app.trace("understand", f"phrase rules: intent={intent}" + (f", game={game}" if game != "none" else "")
                           + (f", side={side}" if side != "none" else "") + " (instant)")
            return intent, game, side, ""
        u = understand(text, SITUATION.get(self.state, ""), self.history)
        self.app.trace("understand", f"Gemma 3 4B: intent={u.intent}, game={u.game}, side={u.side} ({u.seconds:.1f} s)")
        return u.intent, u.game, u.side, u.reply

    def _on_text(self, text: str, source: str, stt_s: float):
        self.app.trace("hear", f"{'microphone' if source == 'mic' else 'keyboard'}: \"{text}\""
                       + (f" (Whisper {stt_s:.1f} s)" if source == "mic" else ""))
        if self.state == "IDLE":
            if not rules.has_wake_word(text):
                self.app.trace("decide", "ignored: I only listen after my name, 'Buddy'")
                return
            self.app.transcript_add("Player", text)
            self.history.append(("player", text))
            rest = text.lower()
            for w in C.WAKE_WORDS:
                rest = rest.replace(w, " ")
            if not rest.strip(" ,.!?"):
                self.set_state("CHAT", "woken up by name")
                self.say("Hello, I'm Buddy. Would you like to play a game?")
                return
            self.set_state("CHAT", "woken up by name")
            text_for_intent = rest
        else:
            self.app.transcript_add("Player", text)
            self.history.append(("player", text))
            text_for_intent = text
        intent, game, side, reply = self._understand(text_for_intent)
        self._act(intent, game, side, reply)

    def _act(self, intent, game, side, reply):
        st = self.state
        if intent == "repeat" and self.last_said:
            self.say(self.last_said)
            return
        if intent == "explain_rules":
            self.say(RULES + (" Shall we play?" if st in ("CHAT", "CHOOSE_GAME") else ""))
            if st in ("CHAT",):
                self.set_state("CHOOSE_GAME", "explained the rules, offered a game")
            return
        if st in ("CHAT", "CHOOSE_GAME"):
            if intent == "choose_game" and game == "tic_tac_toe":
                self.start_setup("the player chose noughts and crosses")
            elif intent == "yes" and st == "CHOOSE_GAME":
                self.start_setup("the player agreed to noughts and crosses")
            elif intent in ("play_game", "yes"):
                self.set_state("CHOOSE_GAME", "the player wants to play")
                self.say(OFFER)
            elif intent == "choose_game" and game == "other":
                self.set_state("CHOOSE_GAME", "asked for a game I don't know")
                self.say(ONLY_TTT)
            elif intent in ("no", "stop"):
                self.set_state("IDLE", "the player does not want to play now")
                self.say("No problem at all. Just say Buddy whenever you fancy a game.")
            else:
                self.say(reply or "Would you like a game of noughts and crosses?")
            return
        if st in ("SETUP", "WAIT_SHEET"):
            if intent == "stop":
                self.say("Alright. I'll finish what I'm doing and then rest.")
            else:
                self.say(reply or "Just a moment, I'm getting the board ready.")
            return
        if st == "ASK_SIDE":
            if intent == "choose_side" or side in ("crosses", "noughts"):
                self.player_side = side
                self.set_state("READY", f"the player chose {side}")
                if side == "crosses":
                    self.say("Crosses for you, so you go first. Draw your cross in any square when you're ready.")
                else:
                    self.say("Noughts for you, so I'll go first. Here's a little secret: I like the middle square.")
            else:
                self.say((reply + " " if reply else "") + "Would you like to be crosses or noughts?")
            return
        if st == "READY":
            if intent in ("stop", "no"):
                self.set_state("IDLE", "the player wants to stop")
                self.say("Thank you, that was lovely. Just say Buddy whenever you want to play again.")
            else:
                self.say(reply or "Whenever you're ready.")

    # ------------------------------------------------------------------ the setup behaviour
    def start_setup(self, why: str):
        self.set_state("SETUP", why)
        self.say("Lovely. Let me have a look at the paper.")
        threading.Thread(target=self._setup, daemon=True, name="setup").start()

    def _wait_for_view(self, timeout=10.0):
        """Fresh perception from a camera frame taken with the arm out of the way."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.app.ctl.at_home and not self.app.ctl.busy:
                rgb, t = self.app.latest_frame()
                if rgb is not None and t > t0 + 0.05:
                    return rgb
            time.sleep(0.05)
        return None

    def _reachable(self, pose) -> tuple[bool, str]:
        ik = self.app.ik_planner
        corners = sheet_to_world(pose, [(x, y) for x in (-C.GRID_SIZE_MM / 2, C.GRID_SIZE_MM / 2)
                                         for y in (-C.GRID_SIZE_MM / 2, C.GRID_SIZE_MM / 2)])
        seed = np.array([0.0, 0.0, 0.5, 1.2, 0.0])
        worst = 0.0
        for x, y in corners:
            for z in (C.SHEET_THICK, C.SHEET_THICK + C.HOVER_MM / 1000):
                r = ik.solve([x, y, z], seed)
                worst = max(worst, r.tilt_deg)
                if not r.ok:
                    dist = 100 * np.hypot(x - 0.039, y)
                    return False, f"grid corner at ({100 * x:.1f}, {100 * y:.1f}) cm, {dist:.1f} cm from my shoulder, is out of reach"
        return True, f"all 4 grid corners reachable with the pen upright (worst lean {worst:.1f} deg)"

    def _setup(self):
        app = self.app
        asked = None
        while True:                                            # 1. SEE: find a usable sheet
            rgb = self._wait_for_view()
            sheet = find_sheet(rgb, app.cam) if rgb is not None else None
            if sheet is not None and sheet.cut_off:
                app.trace("see", f"a sheet, but part of it is outside my view (I see {sheet.size_mm[0]:.0f}x"
                                 f"{sheet.size_mm[1]:.0f} mm of it)")
                if asked != "cut":
                    self.set_state("WAIT_SHEET", "the sheet is partly out of view")
                    self.say("I can see the paper, but it's too far away. Could you slide it closer to me?")
                    asked = "cut"
                time.sleep(1.0)
                continue
            if sheet is None or sheet.confidence < 0.6:
                app.trace("see", "no A4 sheet in view" if sheet is None else
                          f"white shape {sheet.size_mm[0]:.0f}x{sheet.size_mm[1]:.0f} mm does not look like A4 "
                          f"(confidence {sheet.confidence:.2f})")
                if asked != "none":
                    self.set_state("WAIT_SHEET", "no sheet of paper")
                    self.say("I can't see a sheet of A4 paper. Could you put a blank sheet in front of me?")
                    asked = "none"
                time.sleep(1.0)
                continue
            app.trace("see", f"sheet at x={100 * sheet.x:.1f} cm, y={round(100 * sheet.y, 1) + 0.0:.1f} cm, rotated "
                             f"{np.degrees(sheet.yaw):.0f} deg; size {sheet.size_mm[0]:.0f}x{sheet.size_mm[1]:.0f} mm "
                             f"(A4 is 297x210), confidence {sheet.confidence:.2f}")
            ok, why = self._reachable(sheet.as_tuple())          # 2. PLAN: can the grid be drawn there?
            app.trace("plan", f"grid {C.GRID_SIZE_MM / 10:.1f} cm centred on the sheet: {why}")
            if not ok:
                if asked != "reach":
                    self.set_state("WAIT_SHEET", "the sheet is out of reach")
                    self.say("I can't quite reach the paper. Could you slide it a little closer to me?")
                    asked = "reach"
                time.sleep(1.0)
                continue
            inked = ink_fraction(rgb, app.cam, sheet.as_tuple(), C.GRID_SIZE_MM / 2 + 5)
            app.trace("see", f"ink in the grid area: {100 * inked:.1f}% of it (a drawn grid is about 4%)")
            if inked > 0.01:
                if asked != "used":
                    self.set_state("WAIT_SHEET", "the sheet is not blank")
                    self.say("This sheet already has drawing on it. Could you give me a fresh one?")
                    asked = "used"
                time.sleep(1.0)
                continue
            break
        if self.state == "WAIT_SHEET":
            self.set_state("SETUP", "found a usable sheet")
            self.say("Thank you, that's perfect.")
        self.plan_pose = sheet.as_tuple()
        app.set_plan(self.plan_pose, self.lines_mm)
        todo = list(range(len(self.lines_mm)))
        for attempt in range(2):                                 # 3. ACT, then 4. CHECK (redraw once if needed)
            strokes = [sheet_to_world(self.plan_pose, self.lines_mm[i]) for i in todo]
            names = ["left vertical", "right vertical", "top horizontal", "bottom horizontal"]

            def on_stroke(k, n, todo=todo):
                app.trace("act", f"drawing line {todo[k] + 1}/4 ({names[todo[k]]})")

            if attempt == 0:
                self.say("Let me draw our board.")
            job = app.ctl.submit(Job("grid", strokes=strokes, home=True, on_stroke=on_stroke))
            if attempt == 0:
                time.sleep(2.5)
                self.say(RULES)
            job.done.wait()
            if not job.ok:
                app.trace("act", f"drawing failed: {job.error}")
                self.set_state("CHAT", "drawing failed")
                self.say("I'm sorry, something went wrong while I was drawing.")
                return
            rgb = self._wait_for_view()
            now = find_sheet(rgb, app.cam)
            pose = now.as_tuple() if now is not None else self.plan_pose
            self.coverage = check_lines(rgb, app.cam, pose, self.lines_mm)
            app.set_coverage(self.coverage)
            app.trace("check", "ink found along each line: " + ", ".join(f"{100 * c:.0f}%" for c in self.coverage)
                      + f" (a line counts above {100 * C.LINE_OK_COVERAGE:.0f}%)")
            todo = [i for i, c in enumerate(self.coverage) if c < C.LINE_OK_COVERAGE]
            if not todo:
                break
            app.trace("decide", f"redrawing line(s) {', '.join(str(i + 1) for i in todo)}")
        if todo:
            self.say("One of my lines didn't come out well, but we can still play.")
        app.trace("check", "board ready" if not todo else "board ready with a faint line")
        self.say("There we are. Would you like to be crosses or noughts? Crosses go first.")
        self.set_state("ASK_SIDE", "the board is drawn and checked")
