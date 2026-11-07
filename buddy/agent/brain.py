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
        CHAT / CHOOSE_GAME -> TEACH (the player explains a game of their own; buddy/agent/teach.py runs the
        interview with Gemma 4) -> NAME_GAME (if they did not name it) -> OFFER_TAUGHT (Buddy reads the rules
        back, says what the solver found, offers a game) -> SETUP ... the same loop, with buddy/game/solver.py
        choosing Buddy's moves. Games live in buddy/game/library.py (built in + taught, kept in runtime/games.json);
        "easy / medium / hard / impossible / adapt to me" can be said in any state.
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
from ..game.library import Library
from ..game.rules import State, legal_moves
from ..game.rules import result as game_result
from ..game.solver import analyse
from ..game.tictactoe import LEVELS, NAMES, Difficulty, choose, full, other, winner
from ..motion.drawing import grid_lines_mm, sheet_to_world
from ..perception.sheet import check_lines, ink_fraction
from .llm import Conversation, reply
from .teach import Teacher

if TYPE_CHECKING:
    from ..app import App

SYM = {"X": "cross", "O": "nought"}
PLURAL = {"X": "crosses", "O": "noughts"}
OFFER = "I'd love to. I know noughts and crosses, or you could teach me a game of yours. Shall we play noughts and crosses?"  # replaced by Brain.offer_text()
TEACH_INVITE = "I'd love to learn it. Tell me the rules in your own words, a sentence or two at a time."
CANCEL_WORDS = ("never mind", "nevermind", "forget it", "stop", "cancel", "leave it")
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
        self.spec = None                        # a game the player taught (GameSpec); None = noughts and crosses
        self.solver = None                      # buddy/game/solver.py Solver for self.spec
        self.gstate = None                      # rules.State for the taught game, kept in step with self.board
        self.teacher = None                     # the running teaching conversation
        self.library = Library()                # built-in games plus the ones players taught (runtime/games.json)
        self.entry = self.library.find("noughts and crosses")   # the library entry of the current game
        self.pending_spec = None                # a taught game waiting for its name
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

    def legal_cells(self):
        """Where the player may put a mark now: any free square, or only next to the last mark in a taught game."""
        if self.spec is not None and self.gstate is not None:
            return [c for c in legal_moves(self.spec, self.gstate) if c not in self.ignored]
        return self.free_cells()

    @property
    def game_name(self) -> str:
        if self.entry is not None:
            return self.entry.name
        return (self.spec.name or "the game you taught me") if self.spec else "noughts and crosses"

    def offer_text(self) -> str:
        usual = (self.entry or self.library.find("noughts and crosses")).name
        return (f"I'd love to. I know {self.library.spoken_list()}. Shall we play {usual}, or another one? "
                "You can also teach me a new game, and say easy, hard or impossible at any time.")

    def _select(self, entry):
        """Make a library entry the current game."""
        self.entry = entry
        self.spec = entry.spec
        self.solver = analyse(entry.spec)[0] if entry.spec is not None else None

    def _offer_taught(self, entry):
        self._select(entry)
        _, a = analyse(entry.spec)
        self.app.trace("plan", f"solved {entry.name}: {a.nodes} positions in {a.seconds * 1000:.0f} ms, outcome with "
                               f"perfect play: {a.outcome}" + (", winning first squares: "
                               + ", ".join(NAMES[c] for c in a.winning_first_moves) if a.winning_first_moves else ""))
        self.set_state("OFFER_TAUGHT", f"ready to play {entry.name}")
        self.say("Let me think... " + a.speak(entry.spec) + f" Shall we play {entry.name}?")

    def rules_text(self) -> str:
        return self.spec.describe() if self.spec else RULES

    def situation(self) -> str:
        st = self.state
        if st in ("PLAYER_TURN", "CONFIRM_MARK", "ROBOT_TURN"):
            free = ", ".join(NAMES[i] for i in self.legal_cells()) or "none"
            game = f"The game is {self.game_name}: {self.spec.describe()} " if self.spec else ""
            base = (f"{game}A game is on. The player is {SYM[self.player_sym]}es, Buddy is {SYM[self.robot_sym]}s. "
                    f"Squares the player may use: {free}. ")
            if st == "PLAYER_TURN":
                return base + "It is the player's turn: they draw their mark, or say a square for Buddy to draw it."
            if st == "CONFIRM_MARK":
                return base + f"Buddy just asked: {self.last_said}"
            return base + "Buddy is taking its turn."
        known = f" Games Buddy knows: {self.library.spoken_list()}."
        return {
            "IDLE": "Buddy was resting; the player just said its name.",
            "NAME_GAME": "Buddy asked what the game the player just taught is called. Their next words are probably its name.",
            "CHAT": "Chatting. Buddy would like to offer a game of noughts and crosses.",
            "CHOOSE_GAME": "Buddy asked which game they would like to play, or whether they want to teach it one.",
            "TEACH": "The player is teaching Buddy a new game.",
            "OFFER_TAUGHT": f"Buddy learned the player's game and asked whether they want to play it. It said: {self.last_said}",
            "WAIT_SHEET": f"Buddy is waiting for a usable sheet of paper. It said: {self.last_said}",
            "SETUP": f"Buddy is drawing the board for {self.game_name} on the paper.",
            "ASK_SIDE": "The board is drawn. Buddy asked: crosses or noughts? Crosses go first.",
            "GAME_OVER": "The game has ended. Buddy asked whether they would like another game.",
        }.get(st, "") + known

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
        if self.state == "TEACH":
            if any(w in text.lower() for w in CANCEL_WORDS):
                self.spec = None
                self.set_state("CHOOSE_GAME", "the player stopped teaching")
                self.say(f"No problem. Which would you like instead: {self.library.spoken_list()}?")
            else:
                self._teach(text)
            return
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
            self.say(self.rules_text() + (" Shall we play?" if st == "CHAT" else ""))
            if st == "CHAT":
                self.set_state("CHOOSE_GAME", "explained the rules and offered a game")
            return
        if intent == "choose_difficulty" and u.level in (*LEVELS, "adaptive"):
            words = self.difficulty.set_level(u.level)
            self.app.trace("decide", f"difficulty set to {u.level} (strength {100 * self.difficulty.strength:.0f}%)")
            self.say(f"Alright, {words}." + (" " + self.last_said if st in ("CHOOSE_GAME", "ASK_SIDE") else ""))
            return
        if intent == "list_games":
            self.say(f"I know {self.library.spoken_list()}, and you can teach me a new one. Which would you like?")
            if st in ("CHAT", "GAME_OVER"):
                self.set_state("CHOOSE_GAME", "listed the games")
            return
        if st == "NAME_GAME":
            name = (u.game_name or re.sub(r"[^A-Za-z0-9' -]+", "", text)).strip()[:40]
            if intent in ("no", "stop") or not name:
                name = ""
            entry = self.library.remember(self.pending_spec, name)
            self.pending_spec = None
            self.app.trace("decide", f"remembered the game as {entry.name!r} ({self.library.path})")
            self.say(f"{entry.name}. I'll remember it.")
            self._offer_taught(entry)
            return
        if st in ("CHAT", "CHOOSE_GAME"):
            if intent == "teach_game" and not re.search(r"\b(teach|show you|explain|my game|a game i know|different game|rules|turns)\b", text.lower()):
                self.app.trace("decide", "teach_game without any teaching words: treating it as play_game")
                intent = "play_game"
            entry = None
            if intent == "choose_game":
                entry = self.library.find("noughts and crosses") if u.game == "tic_tac_toe" else \
                    (self.library.find(u.game_name) or self.library.find(text))
            if intent == "teach_game" or (intent == "choose_game" and entry is None):
                self.start_teaching("the player wants to teach me a game" if intent == "teach_game"
                                    else f"asked for a game I don't know ({u.game_name or text!r}): asking to be taught")
            elif entry is not None or intent == "new_game":
                self._select(entry or self.entry or self.library.find("noughts and crosses"))
                self.start_setup(f"the player chose {self.game_name}")
            elif intent == "yes" and st == "CHOOSE_GAME":
                self._select(self.entry or self.library.find("noughts and crosses"))   # "yes" = the usual game
                self.start_setup(f"the player said yes: {self.game_name}")
            elif intent in ("play_game", "yes"):
                self.set_state("CHOOSE_GAME", "the player wants to play")
                self.say(self.offer_text())
            elif intent in ("no", "stop"):
                self.set_state("IDLE", "the player does not want to play now")
                self.say("No problem at all. Just say Buddy whenever you fancy a game.")
            else:
                self._chat_reply(text, sit)
            return
        if st == "OFFER_TAUGHT":
            if intent in ("yes", "play_game", "new_game") or (intent == "choose_game" and u.game == "other"):
                self.start_setup(f"the player wants to play {self.game_name}")
            elif intent == "teach_game":
                self.start_teaching("the player wants to teach me another game")
            elif intent in ("no", "stop"):
                self.set_state("CHOOSE_GAME", "the player does not want to play the taught game")
                self.say(f"That's fine. Which would you like instead: {self.library.spoken_list()}?")
            elif intent == "explain_rules":
                self.say(self.rules_text() + " Shall we play it?")
            else:
                self._chat_reply(text, sit, "Shall we play it?")
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
            entry = None
            if intent == "choose_game":
                entry = self.library.find("noughts and crosses") if u.game == "tic_tac_toe" else \
                    (self.library.find(u.game_name) or self.library.find(text))
            if intent == "teach_game" or (intent == "choose_game" and entry is None):
                self.start_teaching("the player wants to teach me a game")
            elif intent in ("yes", "new_game", "play_game", "choose_game"):
                if entry is not None:
                    self._select(entry)
                self.start_setup(f"the player wants another game of {self.game_name}")
            elif intent in ("no", "stop"):
                self.end_session()
            else:
                self._chat_reply(text, sit, "Shall we play another?")

    def end_session(self):
        self.set_state("IDLE", "the player wants to stop")
        self.conv.reset()
        self.say("Thank you, that was lovely. Just say Buddy whenever you want to play again.")

    # ------------------------------------------------------------------ teaching: learn a game from the player
    def start_teaching(self, why: str):
        self.set_state("TEACH", why)
        self.teacher = Teacher()
        self.say(TEACH_INVITE)

    def _teach(self, text: str):
        turn = self.teacher.hear(text)
        d = turn.draft
        self.app.trace("understand", f"Gemma 4 ({self.teacher.model}): grid {d.rows}x{d.cols}, placement {d.placement}, "
                                     f"end {d.end}" + (f" k={d.k}" if d.k else "") + f", first {d.first_player}"
                                     + (f"; asks: {d.question!r}" if d.question and not turn.done else "")
                                     + f" ({turn.seconds:.1f} s)")
        if not turn.done:
            self.say(turn.say)
            return
        if turn.spec is None:
            self.set_state("CHOOSE_GAME", "could not learn that game")
            self.say(turn.say + f" Shall we play one I know: {self.library.spoken_list()}?")
            return
        spec = turn.spec
        if (spec.rows, spec.cols) != (3, 3):
            self.set_state("CHOOSE_GAME", f"the taught game needs a {spec.rows}x{spec.cols} board")
            self.say("I've understood it, but for now I can only draw a three by three board. "
                     f"Shall we play one I know: {self.library.spoken_list()}?")
            return
        known = self.library.same_rules(spec)
        if known is not None:
            self.app.trace("decide", f"these are the rules of {known.name}, which Buddy already knows")
            self.say(f"Oh, I know this one! Those are the rules of {known.name}.", wait=True)
            self._offer_taught(known)
            return
        self.say(turn.say, wait=True)                      # "Let me make sure I have it. ..."
        if spec.name:
            entry = self.library.remember(spec, spec.name)
            self.app.trace("decide", f"remembered the game as {entry.name!r} ({self.library.path})")
            self._offer_taught(entry)
            return
        self.pending_spec = spec
        self.set_state("NAME_GAME", "the game needs a name to be remembered by")
        self.say("What shall we call it?")

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
                self.say(self.rules_text())
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
        self.gstate = State.start(self.spec) if self.spec else None
        self.say("There we are. Would you like to be crosses or noughts?" + (" Crosses go first." if not self.spec else ""))
        self.set_state("ASK_SIDE", "the board is drawn and checked")

    # ------------------------------------------------------------------ the game
    def start_game(self, player_sym: str):
        self.player_sym, self.robot_sym = player_sym, other(player_sym)
        self.games += 1
        self.app.trace("decide", f"game {self.games} ({self.game_name}): player {SYM[player_sym]}es, Buddy "
                                 f"{SYM[self.robot_sym]}s, level {self.difficulty.level}, Buddy plays the best move "
                                 f"{100 * self.difficulty.strength:.0f}% of the time")
        player_first = (self.spec.first_player == "player") if self.spec else (player_sym == "X")
        if self.spec:
            self.gstate = State.start(self.spec, player_sym if player_first else self.robot_sym)
        if player_first:
            self.set_state("PLAYER_TURN", f"the player chose {SYM[player_sym]}es and goes first")
            self.say(f"{PLURAL[player_sym].capitalize()} for you, and you go first. Draw your {SYM[player_sym]} in any "
                     "square, or tell me the square and I'll draw it for you.")
        else:
            self.set_state("ROBOT_TURN", f"the player chose {SYM[player_sym]}es; Buddy goes first")
            self.say(f"{PLURAL[player_sym].capitalize()} for you, so I'll start.")
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
        if self.spec is not None and self.gstate is not None:
            bad = [c for c in cells if c not in legal_moves(self.spec, self.gstate)]
            if bad:
                self.app.trace("decide", f"the {NAMES[bad[0]]} is not allowed by the rules ({self.spec.placement}): "
                                         "pointing it out kindly and carrying on from there")
                self.say(f"Remember, in this game the mark has to go next to the last one. I'll let it go this once.")
        for c in cells:
            self.board[c] = self.player_sym
        if self.spec is not None:
            self.gstate = State(tuple(self.board), cells[-1], self.robot_sym)
        self.app.trace("decide", "the player played the " + " and ".join(NAMES[c] for c in cells))
        self._seen = None
        if not self._game_over():
            self.set_state("ROBOT_TURN", "the player has played")
            self._worker(self._robot_move)

    def _voice_move(self, cell: int):
        if cell not in self.free_cells():
            self.say(f"The {NAMES[cell]} is already taken. Which other square would you like?")
            return
        if cell not in self.legal_cells():
            allowed = ", ".join(NAMES[c] for c in self.legal_cells()) or "none"
            self.say(f"In this game it has to go next to the last mark, so the {NAMES[cell]} isn't allowed. "
                     f"You could use the {allowed}.")
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
        if self.spec is not None:
            ch = self.solver.choose(self.gstate, self.difficulty, self.rng)
        else:
            ch = choose(self.board, self.robot_sym, self.difficulty, self.rng)
        shown = ", ".join(f"{NAMES[i]} {s:+d}" for i, s in sorted(ch.scores.items(), key=lambda kv: -kv[1])[:4])
        self.app.trace("plan", f"{'solver' if self.spec else 'minimax'} scores (win +, loss -): {shown} -> "
                               f"{NAMES[ch.cell]}: {ch.reason}")
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
        if self.spec is not None:
            self.gstate = State(tuple(self.board), ch.cell, self.player_sym)
        self._seen = None
        if not self._game_over():
            self.set_state("PLAYER_TURN", "Buddy has played")
            self.say("Your turn.")

    def _game_over(self) -> bool:
        if self.spec is not None:
            r = game_result(self.spec, self.gstate)
            if r is None:
                return False
            self.set_state("GAME_OVER", "draw" if r[0] == "draw" else
                           f"{PLURAL[r[1]]} win" + (" with a line" if r[2] else ": the other side cannot move"))
            self._worker(self._finish_taught, r)
            return True
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
        self._record(result)
        self.say("Shall we play another? Pop a fresh sheet down and say yes.")

    def _record(self, result: str):
        e = self.entry or self.library.find("noughts and crosses")
        self.library.record(e, result)
        self.app.trace("decide", f"{e.name}: {e.plays} played, you {e.player_wins}, me {e.robot_wins}, drawn {e.draws}")

    def _finish_taught(self, r):
        kind, sym, line = r
        job = None
        if line:
            self.app.trace("plan", f"{self.spec.k} in a row: {', '.join(NAMES[c] for c in line)}: drawing the line")
            job = self.app.draw("the line", M.win_line(list(line)))
        if kind == "draw":
            self.say("It's a draw! Neither of us gave an inch.")
            result = "draw"
        else:
            buddy_won = sym == self.robot_sym
            result = "loss" if buddy_won else "win"
            if self.spec.end == "no_legal_move_loses":
                self.say("You can't move anywhere, so that's my win! Well played though." if buddy_won
                         else "I'm stuck, I can't move anywhere. You win, well done!")
            elif self.spec.end == "k_in_a_row_loses":
                self.say(f"{self.spec.k} in a row, and in this game that means you lose. My win, but a good game!"
                         if buddy_won else f"Oh no, I made {self.spec.k} in a row. That's your win, well done!")
            else:
                self.say(f"{self.spec.k} in a row for me! Well played though, that was a good game." if buddy_won
                         else f"{self.spec.k} in a row! Well done, you beat me fair and square.")
        if job is not None:
            job.done.wait()
        self.app.trace("decide", "difficulty: " + self.difficulty.update(result))
        self._record(result)
        self.say("Shall we play another? Pop a fresh sheet down and say yes.")
