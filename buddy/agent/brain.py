"""Buddy's brain: conversation, game setup and the game itself.

Every step is written to the trace shown on the demo page:
  HEAR        the microphone (Whisper) or the keyboard
  UNDERSTAND  Gemma 3 4B turns the words into a structured intent (and a reply, only for small talk)
  DECIDE      the state machine picks what to do; the language model never picks actions or moves
  SEE         the overhead camera: the sheet, the board read by the mark classifier
  PLAN        reach checks, the solver's choice with its scores
  ACT / SENSE the arm draws; the pen-tip touch sensor finds the paper
  CHECK       the camera verifies what was drawn
  ADAPT       re-planning when the scene changes; SAFETY: the reflex layer
  SAY         Kokoro

The game board is never kept from memory alone: every move, the player's and Buddy's, is read from the paper by the
camera and only then committed.

States: IDLE -> CHAT -> CHOOSE_GAME -> SETUP (WAIT_SHEET) -> ASK_SIDE -> PLAYER_TURN <-> ROBOT_TURN -> GAME_OVER
        PLAYER_TURN -> CONFIRM_MARK when a new mark is unclear or the wrong symbol
"""
from __future__ import annotations

import queue
import random
import re
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from .. import config as C
from ..game import marks as M
from ..game.tictactoe import NAMES, Difficulty, choose, full, other, winner
from ..motion.drawing import grid_lines_mm, sheet_to_world
from ..perception.sheet import check_lines, ink_fraction
from .llm import Conversation, reply

if TYPE_CHECKING:
    from ..app import App

SYM = {"X": "cross", "O": "nought"}
OFFER = "I'd love to. I know noughts and crosses. Shall we play that?"
RULES = ("Here's how it works. We take turns putting our mark in one of the nine squares. One of us draws crosses, "
         "the other noughts. The first to get three in a row, across, down or corner to corner, wins.")
STABLE_S = 1.2                 # a new mark must look the same for this long, with no hand over the paper


def has_wake_word(text: str) -> bool:
    return any(w in C.WAKE_WORDS for w in re.findall(r"[a-z']+", text.lower()))


class Brain:
    def __init__(self, app: "App"):
        self.app = app
        self.state = "IDLE"
        self.inputs: "queue.Queue" = queue.Queue()
        self.last_said = ""
        self.player_sym = self.robot_sym = None
        self.board = [""] * 9
        self.ignored: set[int] = set()
        self.pending = None
        self.busy = False                       # a worker (setup / Buddy's move / drawing for the player) runs
        self.difficulty = Difficulty()
        self.rng = random.Random()
        self.games = 0
        self._seen, self._since = None, 0.0
        self.conv = Conversation()
        threading.Thread(target=self._loop, daemon=True, name="brain").start()
        threading.Thread(target=self._watch, daemon=True, name="board-watch").start()

    # ------------------------------------------------------------------ helpers
    def set_state(self, new: str, why: str):
        if new != self.state:
            self.app.trace("decide", f"{self.state} -> {new}: {why}")
            self.state = new

    def say(self, text: str, wait: bool = False):
        self.last_said = text
        self.app.transcript_add("Buddy", text)
        done = self.app.say(text)
        if wait and done is not None:
            done.wait(timeout=60)

    def hear(self, text: str, source: str = "typed", stt_s: float = 0.0):
        self.inputs.put((text, source, stt_s))

    def free_cells(self):
        return [i for i in range(9) if not self.board[i] and i not in self.ignored]

    def situation(self) -> str:
        st = self.state
        if st in ("PLAYER_TURN", "CONFIRM_MARK", "ROBOT_TURN"):
            free = ", ".join(NAMES[i] for i in self.free_cells()) or "none"
            base = (f"A game is on. The player is {SYM[self.player_sym]}es, Buddy is {SYM[self.robot_sym]}s. "
                    f"Free squares: {free}. ")
            if st == "PLAYER_TURN":
                return base + "It is the player's turn: they draw their mark, or say a square for Buddy to draw it."
            if st == "CONFIRM_MARK":
                return base + f"Buddy just asked: {self.last_said}"
            return base + "Buddy is taking its turn."
        return {
            "IDLE": "Buddy was resting; the player just said its name.",
            "CHAT": "Chatting. Buddy would like to offer a game of noughts and crosses.",
            "CHOOSE_GAME": "Buddy asked whether they would like to play noughts and crosses.",
            "WAIT_SHEET": f"Buddy is waiting for a usable sheet of paper. It said: {self.last_said}",
            "SETUP": "Buddy is drawing the noughts and crosses board on the paper.",
            "ASK_SIDE": "The board is drawn. Buddy asked: crosses or noughts? Crosses go first.",
            "GAME_OVER": "The game has ended. Buddy asked whether they would like another game.",
        }.get(st, "")

    def _worker(self, fn, *args):
        def run():
            self.busy = True
            try:
                fn(*args)
            except Exception as e:
                self.app.trace("decide", f"error: {e}")
            finally:
                self.busy = False
        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------ words
    def _loop(self):
        while True:
            text, source, stt_s = self.inputs.get()
            try:
                self._on_text(text, source, stt_s)
            except Exception as e:
                self.app.trace("decide", f"error handling input: {e}")

    def _on_text(self, text: str, source: str, stt_s: float):
        self.app.trace("hear", f"{'microphone' if source == 'mic' else 'keyboard'}: \"{text}\""
                       + (f" (Whisper {stt_s:.1f} s)" if source == "mic" else ""))
        if self.state == "IDLE":
            if not has_wake_word(text):
                self.app.trace("decide", "ignored: I only listen after my name, 'Buddy'")
                return
            self.set_state("CHAT", "woken up by name")
        self.app.transcript_add("Player", text)
        sit = self.situation()
        u = self.conv.understand(text, sit)
        extra = f", cell={NAMES[u.cell]}" if u.cell is not None else ""
        self.app.trace("understand", f"Gemma 3 4B: intent={u.intent}" + (f", game={u.game}" if u.game != "none" else "")
                       + (f", side={u.side}" if u.side != "none" else "") + extra + f" ({u.seconds:.1f} s)")
        self._act(u, text, sit)

    def _chat_reply(self, text, sit, fallback_prompt=""):
        r, dt = reply(text, sit)
        self.app.trace("understand", f"Gemma 3 4B wrote a reply ({dt:.1f} s)")
        self.say(r + (" " + fallback_prompt if fallback_prompt else ""))

    def _act(self, u, text, sit):
        st, intent = self.state, u.intent
        if intent == "repeat" and self.last_said:
            self.say(self.last_said)
            return
        if intent == "explain_rules":
            self.say(RULES + (" Shall we play?" if st == "CHAT" else ""))
            if st == "CHAT":
                self.set_state("CHOOSE_GAME", "explained the rules and offered a game")
            return
        if st in ("CHAT", "CHOOSE_GAME"):
            if (intent == "choose_game" and u.game == "tic_tac_toe") or (intent == "yes" and st == "CHOOSE_GAME") \
                    or intent == "new_game":
                self.start_setup("the player wants noughts and crosses")
            elif intent in ("play_game", "yes"):
                self.set_state("CHOOSE_GAME", "the player wants to play")
                self.say(OFFER)
            elif intent == "choose_game" and u.game == "other":
                self.set_state("CHOOSE_GAME", "asked for a game I don't know")
                self._chat_reply(text, sit + " They asked for a game Buddy doesn't know.",
                                 "Shall we play noughts and crosses?")
            elif intent in ("no", "stop"):
                self.set_state("IDLE", "the player does not want to play now")
                self.say("No problem at all. Just say Buddy whenever you fancy a game.")
            else:
                self._chat_reply(text, sit)
            return
        if st in ("SETUP", "WAIT_SHEET", "ROBOT_TURN"):
            if intent == "stop":
                self.say("Alright. Let me finish this, then we can stop.")
            elif intent in ("smalltalk", "unclear", "choose_game"):
                self._chat_reply(text, sit)
            return
        if st == "ASK_SIDE":
            if intent == "choose_side" and u.side in ("crosses", "noughts"):
                self.start_game("X" if u.side == "crosses" else "O")
            elif intent in ("smalltalk", "unclear"):
                self._chat_reply(text, sit, "Would you like to be crosses or noughts?")
            else:
                self.say("Would you like to be crosses or noughts? Crosses go first.")
            return
        if st == "PLAYER_TURN":
            if intent == "move" and u.cell is not None:
                self._voice_move(u.cell)
            elif intent == "done":
                self._seen = None                          # look again right now
                self._evaluate(force=True)
            elif intent == "stop":
                self.end_session()
            elif intent in ("smalltalk", "unclear", "choose_game", "play_game"):
                self._chat_reply(text, sit)
            return
        if st == "CONFIRM_MARK":
            cell, sym = self.pending
            if intent == "yes":
                self.app.trace("decide", f"the player confirmed: {SYM[self.player_sym]} in the {NAMES[cell]}")
                self.pending = None
                self._commit_player([cell])
            elif intent == "no":
                self.ignored.add(cell)
                self.pending = None
                self.set_state("PLAYER_TURN", "the player says that mark is not their move")
                self.say("Alright, I'll leave that mark out. Your turn whenever you're ready.")
            else:
                self.say(self.last_said)
            return
        if st == "GAME_OVER":
            if intent in ("yes", "new_game", "play_game", "choose_game"):
                self.start_setup("the player wants another game")
            elif intent in ("no", "stop"):
                self.end_session()
            else:
                self._chat_reply(text, sit, "Shall we play another?")

    def end_session(self):
        self.set_state("IDLE", "the player wants to stop")
        self.conv.reset()
        self.say("Thank you, that was lovely. Just say Buddy whenever you want to play again.")

    # ------------------------------------------------------------------ setup: find the sheet, draw the board
    def start_setup(self, why: str):
        self.set_state("SETUP", why)
        self.say("Lovely. Let me have a look at the paper.")
        self._worker(self._setup)

    def _setup(self):
        app = self.app
        asked = None
        while True:
            look = app.look(timeout=20)
            sheet = look.sheet if look else None
            if sheet is not None and sheet.cut_off:
                app.trace("see", f"a sheet, but part of it is outside my view ({sheet.size_mm[0]:.0f}x"
                                 f"{sheet.size_mm[1]:.0f} mm visible)")
                msg, key = "I can see the paper, but it's too far away. Could you slide it closer to me?", "cut"
            elif sheet is None or sheet.confidence < 0.6:
                app.trace("see", "no A4 sheet in view" if sheet is None else
                          f"white shape {sheet.size_mm[0]:.0f}x{sheet.size_mm[1]:.0f} mm does not look like A4 "
                          f"(confidence {sheet.confidence:.2f})")
                msg, key = "I can't see a sheet of A4 paper. Could you put a blank sheet in front of me?", "none"
            else:
                app.trace("see", f"sheet at x={100 * sheet.x:.1f} cm, y={round(100 * sheet.y, 1) + 0.0:.1f} cm, "
                                 f"rotated {np.degrees(sheet.yaw):.0f} deg; size {sheet.size_mm[0]:.0f}x"
                                 f"{sheet.size_mm[1]:.0f} mm (A4 is 297x210), confidence {sheet.confidence:.2f}")
                ok, why = app.grid_reachable(sheet.as_tuple())
                app.trace("plan", f"grid {C.GRID_SIZE_MM / 10:.1f} cm centred on the sheet: {why}")
                inked = ink_fraction(look.rgb, app.cam, sheet.as_tuple(), C.GRID_SIZE_MM / 2 + 5)
                app.trace("see", f"ink in the grid area: {100 * inked:.1f}% of it (a drawn grid is about 4%)")
                if not ok:
                    msg, key = "I can't quite reach the paper. Could you slide it a little closer to me?", "reach"
                elif inked > 0.01:
                    msg, key = "This sheet already has drawing on it. Could you give me a fresh one?", "used"
                else:
                    break
            if asked != key:
                self.set_state("WAIT_SHEET", {"cut": "the sheet is partly out of view", "none": "no sheet of paper",
                                              "reach": "the sheet is out of reach", "used": "the sheet is not blank"}[key])
                self.say(msg)
                asked = key
            time.sleep(1.0)
        if self.state == "WAIT_SHEET":
            self.set_state("SETUP", "found a usable sheet")
            self.say("Thank you, that's perfect.")
        lines = grid_lines_mm()
        names = ["left vertical", "right vertical", "top horizontal", "bottom horizontal"]
        todo = list(range(4))
        for attempt in range(2):
            if attempt == 0:
                self.say("Let me draw our board.")
            job = app.draw("the board", [lines[i] for i in todo],
                           on_stroke=lambda k, n, todo=todo: app.trace("act", f"drawing line {todo[k] + 1}/4 "
                                                                              f"({names[todo[k]]})"))
            if attempt == 0:
                time.sleep(2.0)
                self.say(RULES)
            job.done.wait()
            if not job.ok:
                self.set_state("CHAT", "drawing failed")
                self.say("I'm sorry, something went wrong while I was drawing.")
                return
            look = app.look(timeout=20)
            cov = check_lines(look.rgb, app.cam, look.track.pose, lines)
            app.set_overlay_lines(lines, cov)
            app.trace("check", "ink found along each line: " + ", ".join(f"{100 * c:.0f}%" for c in cov)
                      + f" (a line counts above {100 * C.LINE_OK_COVERAGE:.0f}%)")
            todo = [i for i, c in enumerate(cov) if c < C.LINE_OK_COVERAGE]
            if not todo:
                break
            app.trace("decide", f"redrawing line(s) {', '.join(str(i + 1) for i in todo)}")
        self.board = [""] * 9
        self.ignored = set()
        self.say("There we are. Would you like to be crosses or noughts? Crosses go first.")
        self.set_state("ASK_SIDE", "the board is drawn and checked")

    # ------------------------------------------------------------------ the game
    def start_game(self, player_sym: str):
        self.player_sym, self.robot_sym = player_sym, other(player_sym)
        self.games += 1
        self.app.trace("decide", f"game {self.games}: player {SYM[player_sym]}es, Buddy {SYM[self.robot_sym]}s, "
                                 f"Buddy plays the best move {100 * self.difficulty.strength:.0f}% of the time")
        if player_sym == "X":
            self.set_state("PLAYER_TURN", "the player chose crosses and goes first")
            self.say("Crosses for you, so you go first. Draw your cross in any square, or tell me the square and "
                     "I'll draw it for you.")
        else:
            self.set_state("ROBOT_TURN", "the player chose noughts, so Buddy goes first")
            self.say("Noughts for you, so I'll start.")
            self._worker(self._robot_move)

    def _watch(self):
        """Turn-taking: in PLAYER_TURN, watch the board the camera reads until a new mark has been steady for
        STABLE_S seconds with the arm parked and no hand over the paper."""
        while True:
            time.sleep(0.2)
            if self.state == "PLAYER_TURN" and not self.busy:
                try:
                    self._evaluate()
                except Exception as e:
                    self.app.trace("decide", f"error reading the board: {e}")

    def _evaluate(self, force: bool = False):
        snap = self.app.board_snapshot()
        if snap is None or not snap.usable:
            self._seen = None
            if force:
                self.say("Just a moment, I can't see the board clearly yet.")
            return
        marks = [c.mark for c in snap.cells]
        if marks != self._seen:
            self._seen, self._since = marks, time.time()
            if not force:
                return
        if not force and time.time() - self._since < STABLE_S:
            return
        new = [i for i in self.free_cells() if marks[i] != "empty"]
        if not new:
            if force:
                self.say(f"I don't see a new mark yet. Draw your {SYM[self.player_sym]} in any square, or tell me "
                         "where and I'll draw it for you.")
            return
        conf = ", ".join(f"{NAMES[i]}: {snap.cells[i].mark} {100 * snap.cells[i].confidence:.0f}%" for i in new)
        self.app.trace("see", f"new mark(s) on the paper: {conf}")
        unclear = [i for i in new if marks[i] == "unclear"]
        wrong = [i for i in new if marks[i] == self.robot_sym]
        if unclear or (len(new) == 1 and wrong):
            cell = (unclear or wrong)[0]
            p = snap.cells[cell].p
            if unclear:
                self.app.trace("decide", f"not sure about the {NAMES[cell]} (X {100 * p['X']:.0f}%, O "
                                         f"{100 * p['O']:.0f}%, empty {100 * p['empty']:.0f}%): asking")
                q = f"I can't quite read the mark in the {NAMES[cell]}. Is that your {SYM[self.player_sym]}?"
            else:
                self.app.trace("decide", f"the {NAMES[cell]} reads as a {SYM[self.robot_sym]}, but the player is "
                                         f"{SYM[self.player_sym]}es: asking")
                q = (f"That looks like a {SYM[self.robot_sym]} in the {NAMES[cell]}, but you're "
                     f"{SYM[self.player_sym]}es. Shall I count it as your {SYM[self.player_sym]}?")
            self.pending = (cell, self.player_sym)
            self.set_state("CONFIRM_MARK", "a new mark needs checking")
            self.say(q)
            return
        if len(new) > 1:
            self.app.trace("decide", f"{len(new)} new marks in one turn")
            self.say(f"I see {len(new)} new marks there! We usually take turns, but I'll let you have them.")
        self._commit_player(new)

    def _commit_player(self, cells):
        for c in cells:
            self.board[c] = self.player_sym
        self.app.trace("decide", "the player played the " + " and ".join(NAMES[c] for c in cells))
        self._seen = None
        if not self._game_over():
            self.set_state("ROBOT_TURN", "the player has played")
            self._worker(self._robot_move)

    def _voice_move(self, cell: int):
        if cell not in self.free_cells():
            self.say(f"The {NAMES[cell]} is already taken. Which other square would you like?")
            return
        self.app.trace("decide", f"the player asked me to draw their {SYM[self.player_sym]} in the {NAMES[cell]}")
        self.say(f"Your {SYM[self.player_sym]} in the {NAMES[cell]}. Here it comes.")
        self._worker(self._draw_for_player, cell)

    def _draw_for_player(self, cell: int):
        job = self.app.draw(f"the player's {SYM[self.player_sym]}", M.robot_mark(self.player_sym, cell))
        job.done.wait()
        cells = self._read_fresh()
        got = cells[cell] if cells else None
        self.app.trace("check", f"the {NAMES[cell]} now reads {got.mark} ({100 * got.confidence:.0f}%)" if got
                       else "could not read the board")
        self._commit_player([cell])

    def _read_fresh(self):
        snap = self.app.board_snapshot(fresh=True, timeout=15)
        return snap.cells if snap else None

    def _robot_move(self):
        ch = choose(self.board, self.robot_sym, self.difficulty, self.rng)
        shown = ", ".join(f"{NAMES[i]} {s:+d}" for i, s in sorted(ch.scores.items(), key=lambda kv: -kv[1])[:4])
        self.app.trace("plan", f"minimax scores (win +, loss -): {shown} -> {NAMES[ch.cell]}: {ch.reason}")
        self.say(f"I'll take the {NAMES[ch.cell]}.")
        for attempt in range(2):
            job = self.app.draw(f"my {SYM[self.robot_sym]}", M.robot_mark(self.robot_sym, ch.cell))
            job.done.wait()
            if not job.ok:
                self.say("I'm sorry, I couldn't draw that.")
                return
            cells = self._read_fresh()
            got = cells[ch.cell] if cells else None
            self.app.trace("check", f"the {NAMES[ch.cell]} now reads {got.mark} ({100 * got.confidence:.0f}%)" if got
                           else "could not read the board")
            if got is not None and got.mark == self.robot_sym:
                break
            self.app.trace("decide", "my mark did not come out clearly: drawing it again")
        self.board[ch.cell] = self.robot_sym
        self._seen = None
        if not self._game_over():
            self.set_state("PLAYER_TURN", "Buddy has played")
            self.say("Your turn.")

    def _game_over(self) -> bool:
        w = winner(self.board)
        if not w and not full(self.board):
            return False
        self.set_state("GAME_OVER", f"{'three in a row: ' + w[0] if w else 'the board is full'}")
        self._worker(self._finish, w)
        return True

    def _finish(self, w):
        if w:
            sym, cells = w
            self.app.trace("plan", f"three in a row: {', '.join(NAMES[c] for c in cells)}: drawing the winning line")
            job = self.app.draw("the winning line", M.win_line(cells))
            if sym == self.robot_sym:
                self.say("Three in a row for me! Well played though, that was a good game.")
                result = "loss"
            else:
                self.say("Three in a row! Well done, you beat me fair and square.")
                result = "win"
            job.done.wait()
        else:
            self.say("It's a draw! Neither of us gave an inch.")
            result = "draw"
        self.app.trace("decide", "difficulty: " + self.difficulty.update(result))
        self.say("Shall we play another? Pop a fresh sheet down and say yes.")
