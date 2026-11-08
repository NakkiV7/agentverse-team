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
        Hangman uses HANGMAN_SETUP -> HANGMAN_TURN -> GAME_OVER, checking known strokes without reading handwriting.
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
from ..game.hangman import Hangman, WORDS, letter as hangman_letter
from .dots_brain import STATES as DOTS_STATES, DotsPlay
from copy import deepcopy
from ..game import dotsboxes as D
from . import tools
from .toolcall import Planner, ToolCall

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
        self.hangman = None
        self.dots = None                        # the running dots-and-boxes game (buddy/agent/dots_brain.py)
        self.last_heard = ""
        self.level_asked = False                # the difficulty question is asked once per session, before the first game
        self.switch_to = None                   # a game change or stop asked for while the arm was busy; done when it is free
        self.last_choice = None                 # the solver's last Choice (cell/edge, scores, reason), for "why did you play that?"
        self._setup_after_level = None
        self._hangman_strokes = []
        self._hangman_pending = None
        self.planner = Planner()                # Gemma 4 picks tools from what the player said (buddy/agent/toolcall.py)
        self._size_hint = None                  # a dots board size said before the size question (new_board tool)
        self._side_hint = None                  # a side said before the board is drawn (choose_side tool)
        threading.Thread(target=self._loop, daemon=True, name="brain").start()
        threading.Thread(target=self._watch, daemon=True, name="board-watch").start()
        threading.Thread(target=self._warm_planner, daemon=True, name="planner-warm-up").start()

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
        self.difficulty.select_game(entry.name)

    def _offer_taught(self, entry):
        self._select(entry)
        _, a = analyse(entry.spec)
        self.app.trace("plan", f"solved {entry.name}: {a.nodes} positions in {a.seconds * 1000:.0f} ms, outcome with "
                               f"perfect play: {a.outcome}" + (", winning first squares: "
                               + ", ".join(NAMES[c] for c in a.winning_first_moves) if a.winning_first_moves else ""))
        self.set_state("OFFER_TAUGHT", f"ready to play {entry.name}")
        self.say("Let me think... " + a.speak(entry.spec) + f" Shall we play {entry.name}?")

    def rules_text(self) -> str:
        if self.entry is not None and self.entry.kind == "hangman":
            return self.entry.describe()
        if self.entry is not None and self.entry.kind == "dots":
            return D.describe(self.dots.n) if self.dots is not None else self.entry.describe()
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
            "ASK_LEVEL": "Buddy asked how hard it should play: easy, medium, hard, impossible, or authentic (adaptive: it learns the player's level over a few games). Their answer is a choose_difficulty with the level; 'authentic', 'realistic', 'match me' mean adaptive.",
            "TEACH": "The player is teaching Buddy a new game.",
            "OFFER_TAUGHT": f"Buddy learned the player's game and asked whether they want to play it. It said: {self.last_said}",
            "WAIT_SHEET": f"Buddy is waiting for a usable sheet of paper. It said: {self.last_said}",
            "SETUP": f"Buddy is drawing the board for {self.game_name} on the paper.",
            "ASK_SIDE": "The board is drawn. Buddy asked: crosses or noughts? Crosses go first.",
            "GAME_OVER": "The game has ended. Buddy asked whether they would like another game.",
            "DOTS_SIZE": "Buddy asked how big a dots-and-boxes board the player wants (boxes across and down, up to 6 across and 5 down).",
            "DOTS_SETUP": "Buddy is drawing the dots for dots and boxes.",
            "DOTS_PLAYER_TURN": "Dots and boxes: it is the player's turn to join two neighbouring dots with a line, on the paper or on the page, or by naming it ('the top side of the top left box', intent move). Buddy reads the paper with the camera.",
            "DOTS_ROBOT_TURN": "Dots and boxes: Buddy is drawing its line.",
            "HANGMAN_SETUP": "Buddy is preparing hangman on paper. Say yes to retry, or stop to finish.",
            "HANGMAN_TURN": "The player is guessing a hangman word. Say one letter, or ask for a hint. "
                            "Do not choose a letter for the player.",
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
        self.last_heard = text
        if self.state == "TEACH":
            if any(w in text.lower() for w in CANCEL_WORDS):
                self.spec = None
                self.set_state("CHOOSE_GAME", "the player stopped teaching")
                self.say(f"No problem. Which would you like instead: {self.library.spoken_list()}?")
            else:
                self._teach(text)
            return
        sit = self.situation()
        calls = self._plan_tools(text, sit)      # Gemma 4 picks tools; None = model off or unsure, use the old path
        if calls is not None:
            self.run_tools(calls, text)
            return
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
        if st == "DOTS_SIZE" and self.dots is not None and (self.dots.parse_size_text(text) or intent != "choose_difficulty"):
            self.dots.act(u, text, sit)          # a size answer wins over anything the model guessed
            return
        if intent == "choose_difficulty" and u.level in (*LEVELS, "adaptive"):
            words = self.difficulty.set_level(u.level)
            self.level_asked = True
            self.app.trace("decide", f"difficulty set to {u.level} (strength {100 * self.difficulty.strength:.0f}%)")
            if st == "ASK_LEVEL":
                self.say(f"Alright, {words}.")
                why, self._setup_after_level = self._setup_after_level, None
                self._start_setup(why or "the level is chosen")
                return
            self.say(f"Alright, {words}." + (" " + self.last_said if st in ("CHOOSE_GAME", "ASK_SIDE") else ""))
            return
        if st == "ASK_LEVEL":
            t = text.lower()
            if re.search(r"\b(authentic|realistic|adapt|match|learn|you (choose|decide|pick)|don't mind|whatever|normal)\b", t):
                u.level = "adaptive"
            elif re.search(r"\b(impossible|perfect|best|hardest|no mercy)\b", t):
                u.level = "impossible"
            elif re.search(r"\b(hard|tough|difficult)\b", t):
                u.level = "hard"
            elif re.search(r"\b(medium|middle|moderate)\b", t):
                u.level = "medium"
            elif re.search(r"\b(easy|gentle|simple|beginner)\b", t):
                u.level = "easy"
            if u.level in (*LEVELS, "adaptive"):
                words = self.difficulty.set_level(u.level)
                self.level_asked = True
                self.app.trace("decide", f"difficulty set to {u.level} from the words (strength {100 * self.difficulty.strength:.0f}%)")
                self.say(f"Alright, {words}.")
                why, self._setup_after_level = self._setup_after_level, None
                self._start_setup(why or "the level is chosen")
            elif intent in ("no", "stop"):
                self.level_asked = True
                self.say("I'll adapt to you, then.")
                why, self._setup_after_level = self._setup_after_level, None
                self._start_setup(why or "default level")
            else:
                self.say("Easy, medium, hard, impossible, or authentic?")
            return
        if intent == "list_games":
            self.say(f"I know {self.library.spoken_list()}, and you can teach me a new one. Which would you like?")
            if st in ("CHAT", "GAME_OVER"):
                self.set_state("CHOOSE_GAME", "listed the games")
            return
        if st not in ("IDLE", "CHAT", "CHOOSE_GAME", "GAME_OVER", "TEACH", "NAME_GAME", "OFFER_TAUGHT", "DOTS_SIZE"):
            # a game change, a new game to teach, or stop, asked for in the middle of things
            if intent == "stop" and st != "PLAYER_TURN":
                self.request_switch("stop", "the player wants to stop")
                return
            if self._wants_to_teach(u, text):
                self.request_switch("teach", "the player wants to teach me a game")
                return
            if intent == "teach_game":
                intent = "choose_game"
            if intent in ("choose_game", "new_game") or (intent == "play_game" and self._find_game(u, text) is not None):
                entry = self._find_game(u, text)
                if intent == "choose_game" and entry is None and u.game == "other":
                    self.request_switch("teach", f"asked for a game I don't know ({u.game_name or text!r})")
                    return
                if entry is not None and entry is not self.entry:
                    self.request_switch(entry, f"the player asked for {entry.name}")
                    return
                if entry is not None and entry is self.entry and st in ("ASK_SIDE", "ASK_LEVEL", "SETUP", "WAIT_SHEET"):
                    self.say(f"Yes, {entry.name} is what we're setting up." + (" " + self.last_said if st in ("ASK_SIDE", "ASK_LEVEL") else ""))
                    return
                if intent == "new_game" and st in ("PLAYER_TURN", "ROBOT_TURN", "CONFIRM_MARK", "DOTS_PLAYER_TURN", "DOTS_ROBOT_TURN", "HANGMAN_TURN"):
                    self.request_switch(self.entry, "the player wants to start again")
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
            if intent == "teach_game" and not self._wants_to_teach(u, text):
                self.app.trace("decide", "teach_game without teaching words: treating it as choosing or asking for a game")
                intent = "choose_game" if self._find_game(u, text) is not None else "play_game"
            entry = self._find_game(u, text) if intent in ("choose_game", "play_game", "new_game") else None
            if intent == "play_game" and entry is not None and entry is self.library.find("noughts and crosses") \
                    and not re.search(r"noughts|crosses|tic.?tac|x'?s and o'?s", text.lower()):
                entry = None                      # "let's play a game" is not a request for noughts and crosses
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
            if intent in ("smalltalk", "unclear", "choose_game", "play_game", "yes", "no"):
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
        if st in ("HANGMAN_SETUP", "HANGMAN_TURN"):
            self._hangman_act(u)
        if st in DOTS_STATES and self.dots is not None:
            self.dots.act(u, text, sit)

    BUSY_STATES = ("SETUP", "ROBOT_TURN", "DOTS_SETUP", "DOTS_ROBOT_TURN", "HANGMAN_SETUP")
    TEACH_WORDS = re.compile(r"\b(teach|show you|explain|my game|a game i know|different game|game of mine|invented|made up|the rules are|you (take|put|write) turns|we take turns)\b")

    def _wants_to_teach(self, u, text) -> bool:
        """teach_game only counts when the words are about teaching; a known game named in the sentence wins."""
        if u.intent != "teach_game":
            return False
        if self._find_game(u, text) is not None and not self.TEACH_WORDS.search(text.lower()):
            return False
        return True

    def _find_game(self, u, text):
        """The game the player named, if any: by game_name, by the words, or noughts and crosses by the model's flag."""
        if u.game == "tic_tac_toe":
            return self.library.find("noughts and crosses")
        return self.library.find(u.game_name) or self.library.find(text)

    def request_switch(self, target, why: str):
        """Change game (an Entry), teach a new one ("teach"), or stop ("stop"), from any state. When the arm is in
        the middle of something the request waits until that stroke is done."""
        name = target if isinstance(target, str) else target.name
        if self.busy or self.state in self.BUSY_STATES:
            self.switch_to = target
            self.app.trace("decide", f"{why}: will switch to {name} as soon as this stroke is finished")
            self.say("Alright. Let me finish this stroke, then we'll " + ("stop." if target == "stop" else
                     "start on the game you want to teach me." if target == "teach" else f"change to {name}."))
            return
        self._do_switch(target, why)

    def _maybe_switch(self) -> bool:
        """Called by workers when they finish a stroke: carry out a waiting switch. True if one was done."""
        if self.switch_to is None:
            return False
        target, self.switch_to = self.switch_to, None
        self._do_switch(target, "the player asked while I was busy")
        return True

    def _do_switch(self, target, why: str):
        if self.dots is not None:
            self.dots.running = False
        self.hangman = None
        self.pending = None
        if target == "stop":
            self.end_session()
            return
        if target == "teach":
            self.start_teaching(why)
            return
        self._select(target)
        self.app.trace("decide", f"{why}: switching to {target.name}")
        self.say(f"Let's play {target.name} instead." if self.games else f"Let's play {target.name}.")
        self.start_setup(why)

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
    LEVEL_QUESTION = ("How hard shall I play? Easy, medium, hard or impossible. Or say authentic, and I'll learn your "
                      "level over a few games and play to match it.")

    def start_setup(self, why: str):
        """Every game starts here. The first time in a session Buddy asks how hard to play, unless a level was
        already chosen; then the game's own setup runs."""
        if not self.level_asked:
            self._setup_after_level = why
            self.set_state("ASK_LEVEL", "asking how hard to play before the first game")
            self.say(self.LEVEL_QUESTION)
            return
        self._start_setup(why)

    def _start_setup(self, why: str):
        if self.entry is not None and self.entry.kind == "dots":
            self.dots = DotsPlay(self)
            said, self._size_hint = self._size_hint, None
            if self.dots.take_size(said or self.last_heard):
                self.say(f"Lovely, {self.dots.n[1] if isinstance(self.dots.n, tuple) else self.dots.n} across. Let me have a look at the paper.")
                self._worker(self.dots.setup)
            else:
                self.dots.ask_size()
            return
        self.set_state("SETUP", why)
        self.say("Lovely. Let me have a look at the paper.")
        self._worker(self._setup)

    def _setup(self):
        if self.entry is not None and self.entry.kind == "hangman":
            self._hangman_setup()
            return
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
        if self._maybe_switch():
            return
        if self._side_hint:                              # the player already said crosses or noughts
            sym, self._side_hint = self._side_hint, None
            self.say("There we are.")
            self.start_game(sym, inline=True)
            return
        self.say("There we are. Would you like to be crosses or noughts?" + (" Crosses go first." if not self.spec else ""))
        self.set_state("ASK_SIDE", "the board is drawn and checked")

    # ------------------------------------------------------------------ the game
    def start_game(self, player_sym: str, inline: bool = False):
        """inline: already inside a worker (the setup), so Buddy's first move runs here rather than in a new one."""
        self.player_sym, self.robot_sym = player_sym, other(player_sym)
        self.games += 1
        self.app.trace("decide", "difficulty: " + self.difficulty.start_game(self.game_name, self.rng))
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
            if inline:
                self._robot_move()
            else:
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
        self.last_choice = ch
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
        if self._maybe_switch():
            return
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

    # ------------------------------------------------------------------ hangman: spoken letters, checked ink
    def _hangman_act(self, u):
        if u.intent == "stop":
            self.end_session()
            return
        if self.busy:
            self.say("Let me finish checking the paper first.")
            return
        if self.state == "HANGMAN_SETUP":
            if u.intent in ("yes", "done"):
                self.busy = True
                self._worker(self._hangman_setup, True)
            else:
                self.say("Please clear the paper, then say yes to try again, or stop to finish.")
            return
        if u.intent == "hangman_hint":
            self.busy = True
            self._worker(self._hangman_hint)
            return
        c = hangman_letter(u.letter) if u.intent == "say_letter" else None
        if c is None:
            self.say("I did not catch one letter. Please say a single letter, such as A for apple.")
        elif c in self.hangman.guessed:
            self.say(f"You have already tried {c}. Please choose another letter.")
        elif self._hangman_pending is not None and c != self._hangman_pending[0]:
            self.say(f"I still need to check {self._hangman_pending[0]}. Please say that letter again, or stop.")
        else:
            self.busy = True
            self._worker(self._hangman_turn, c)

    def _hangman_look(self):
        look = self.app.look(timeout=20)
        if (look is None or look.sheet is None or look.sheet.cut_off or look.sheet.confidence < .6
                or look.track is None or not look.track.visible or not look.track.stable
                or not self.app.view_clear()):
            return None
        return look

    def _hangman_coverage(self, look, strokes):
        lines = M.stroke_segments(strokes)
        coverage = check_lines(look.rgb, self.app.cam, look.track.pose, lines)
        self.app.trace("check", f"hangman: {len(lines)} segments, minimum ink coverage "
                       f"{min(coverage, default=1):.0%}")
        return all(c >= C.LINE_OK_COVERAGE for c in coverage)

    def _hangman_draw_checked(self, name, strokes):
        for attempt in range(2):
            if self.state not in ("HANGMAN_SETUP", "HANGMAN_TURN"):
                return False
            job = self.app.draw(name, strokes)
            job.done.wait()
            if not job.ok:
                self.app.trace("check", f"hangman drawing failed: {job.error}")
                return False
            look = self._hangman_look()
            if look is None:
                return False
            if self._hangman_coverage(look, self._hangman_strokes + strokes):
                return self.state in ("HANGMAN_SETUP", "HANGMAN_TURN")
            self.app.trace("check", "hangman ink was unclear; retrying the drawing once")
        return False

    def _hangman_setup(self, retry=False):
        self.set_state("HANGMAN_SETUP", "preparing the word on paper")
        if not retry:
            self.hangman = Hangman(self.rng.choice(WORDS))
            self._hangman_strokes = []
            self._hangman_pending = None
        look = self._hangman_look()
        if self.state != "HANGMAN_SETUP":
            return
        if look is None:
            self.say("I cannot see a clear sheet. Please place the paper in view, then say yes.")
            return
        reachable, why = self.app.grid_reachable(look.track.pose)
        self.app.trace("plan", f"hangman drawing area: {why}")
        if not reachable:
            self.say("I cannot reach the paper. Please slide it closer, then say yes.")
            return
        dashes = M.hangman_dashes(len(self.hangman.word))
        # A failed setup may have drawn some dashes; retry on a fresh sheet to avoid accepting unrelated ink.
        if ink_fraction(look.rgb, self.app.cam, look.track.pose, C.GRID_SIZE_MM / 2 + 5) > .001:
            self.say("Please put down a fresh sheet, then say yes.")
            return
        self.say(self.rules_text() + f" The word has {len(self.hangman.word)} letters.")
        if not self._hangman_draw_checked("the word dashes", dashes):
            if self.state == "HANGMAN_SETUP":
                self.say("I could not check the dashes. Please put down a fresh sheet, then say yes.")
            return
        self._hangman_strokes = dashes
        self.games += 1
        self.set_state("HANGMAN_TURN", "the camera checked the word dashes")
        self.say("The dashes are ready. Say one letter whenever you are ready.")

    def _hangman_paper_ready(self):
        look = self._hangman_look()
        if self.state != "HANGMAN_TURN":
            return False
        if look is None or not self._hangman_coverage(look, self._hangman_strokes):
            self.say("I cannot check our drawing. Please clear the view and try again, or say stop.")
            return False
        reachable, _ = self.app.grid_reachable(look.track.pose)
        if not reachable:
            self.say("I cannot reach the paper. Please slide it closer and try again.")
            return False
        return True

    def _hangman_hint(self):
        if self._hangman_pending is not None:
            self.say(f"Please say {self._hangman_pending[0]} again so I can check its drawing first.")
        elif self._hangman_paper_ready():
            self.say(f"You could try {self.hangman.hint()}. It is common in words that fit our clues.")

    def _hangman_turn(self, c):
        if not self._hangman_paper_ready():
            return
        if self._hangman_pending is None:
            candidate = deepcopy(self.hangman)
            positions = candidate.guess(c)
            strokes = ([s for i in positions for s in M.hangman_letter(c, i, len(candidate.word))]
                       if positions else M.gallows_part(candidate.misses))
            self._hangman_pending = (c, candidate, strokes)
        _, candidate, strokes = self._hangman_pending
        self.say(f"Let me draw {c}." if c in candidate.word else
                 f"There is no {c}. Let me draw the next gallows part.")
        if not self._hangman_draw_checked(f"hangman letter {c}", strokes):
            if self.state == "HANGMAN_TURN":
                self.say(f"I could not check that drawing. Please clear the view, then say {c} again, or stop.")
            return
        self.hangman = candidate
        self._hangman_strokes.extend(strokes)
        self._hangman_pending = None
        self.app.trace("decide", f"hangman: {candidate.pattern}, {candidate.misses}/6 misses")
        result = candidate.result()
        if result is not None:
            self._record(result)
            self.set_state("GAME_OVER", f"hangman: {result}")
            self.say(f"You found it. The word is {candidate.word}. Well done." if result == "win" else
                     f"That is six misses. The word was {candidate.word}. Thank you for playing.")
            self.say("Shall we play another? Pop a fresh sheet down and say yes.")
        else:
            found = f"{c} is in the word. " if c in candidate.word else ""
            noun = "miss" if candidate.remaining == 1 else "misses"
            self.say(f"{found}{candidate.remaining} {noun} left. Say another letter.")

    # ------------------------------------------------------------------ tools: the player asks in their own words
    # Gemma 4 (buddy/agent/toolcall.py) turns what the player said into tool calls from buddy/agent/tools.py. Nothing
    # here trusts the model: every tool is checked against the state it was offered in, then against the live state,
    # the board, the rules and the reach checks by the same methods the intent path uses. The model never picks a
    # move: draw_my_mark / draw_my_line / say_letter only carry what the player named, and _voice_move / _spoken_line /
    # _hangman_act check it.
    IN_GAME = ("PLAYER_TURN", "ROBOT_TURN", "CONFIRM_MARK", "DOTS_PLAYER_TURN", "DOTS_ROBOT_TURN", "HANGMAN_TURN")
    RESTING = ("IDLE", "CHAT", "CHOOSE_GAME", "GAME_OVER", "OFFER_TAUGHT")
    SETTING_UP = ("ASK_LEVEL", "SETUP", "WAIT_SHEET", "ASK_SIDE", "DOTS_SIZE", "DOTS_SETUP", "HANGMAN_SETUP")

    def _warm_planner(self):
        """Process the planner's instructions once at start-up so the first real request is quick."""
        try:
            self.planner.plan("hello", "Buddy is starting up.", "CHAT")
            self.planner.reset()
            self.app.trace("system", f"tool planner ready ({self.planner.model} via Ollama, warmed up in {self.planner.seconds:.1f} s)")
        except Exception as e:  # noqa: BLE001
            self.app.trace("system", f"tool planner not available ({e}); using the intent path")

    def _plan_tools(self, text: str, sit: str):
        """Ask the planner. A size answer to the size question wins without a model call (the old precedence rule)."""
        if self.state in ("TEACH", "NAME_GAME"):
            return None
        if self.state == "DOTS_SIZE" and self.dots is not None and self.dots.parse_size_text(text) is not None:
            self.app.trace("understand", f"tools: new_board(size={text!r}) (a size answer wins; no model call)")
            return [ToolCall("new_board", {"size": text})]
        full = sit + f" Rules of {self.game_name}: {self.rules_text()}"
        if self.dots is not None and self.dots.state is not None and self.state in DOTS_STATES:
            full += " " + self.dots._score_line()
        if self.last_said:
            full += f' Buddy last said: "{self.last_said}"'
        calls = self.planner.plan(text, full, self.state)
        if calls is None:
            self.app.trace("understand", f"tools: no answer from {self.planner.model} ({self.planner.seconds:.1f} s); using the intent path")
            return None
        self.app.trace("understand", "tools: " + ", ".join(str(c) for c in calls) + f" ({self.planner.seconds:.1f} s)")
        return calls

    def _u(self, intent: str, **kw) -> Understanding:
        """A synthetic Understanding, to reuse the state machine's own handling of yes / no / done / a letter."""
        return Understanding(intent, "none", "none", None, 0.0, **kw)

    def run_tools(self, calls, text: str = ""):
        """Carry out the planner's choices in order. Availability is checked against the state the request came
        in (the one the model was told about); each action then checks the live state and the board itself."""
        shown = self.state
        offered = {t.name for t in tools.available(shown)}
        for call in calls:
            name, args = call.name, dict(call.args or {})
            if name not in tools.BY_NAME:
                self.app.trace("decide", f"tool {name}: not one of mine, ignored")
                continue
            if name not in offered and not (name == "answer" and shown == "CHAT"):
                self.app.trace("decide", f"tool {name}: does not fit {shown}, refused")
                self.say(self._refusal(name))
                continue
            fn = getattr(self, f"_tool_{name}")
            try:
                fn(args, text)
            except Exception as e:  # noqa: BLE001
                self.app.trace("decide", f"tool {name}: error: {e}")
                self.say("I'm sorry, I couldn't do that just now.")

    def _refusal(self, name: str) -> str:
        st = self.state
        if name == "answer":
            return {"PLAYER_TURN": "It's your turn: draw your mark, or tell me the square and I'll draw it.",
                    "DOTS_PLAYER_TURN": "It's your line. Draw it, or tell me which one, like the left side of the top right box.",
                    "HANGMAN_TURN": "Say a letter whenever you're ready."}.get(st, "Alright.")
        if st in self.RESTING:
            return "There's no game on just now. Shall we start one?"
        if st in self.SETTING_UP:
            return "Let me get the board ready first."
        if st in ("ROBOT_TURN", "DOTS_ROBOT_TURN"):
            return "Let me finish my go first."
        return "I can't do that just now."

    # -- choosing what to play
    def _tool_switch_game(self, args, text):
        name = str(args.get("name") or "").strip()
        entry = self.library.find(name) or self.library.find(text)
        st = self.state
        if entry is None:
            why = f"asked for a game I don't know ({name or text!r}): asking to be taught"
            if st in self.RESTING:
                self.start_teaching(why)
            else:
                self.request_switch("teach", why)
            return
        if entry is self.entry:
            if st in self.SETTING_UP:
                self.say(f"Yes, {entry.name} is what we're setting up." + (" " + self.last_said if st in ("ASK_SIDE", "ASK_LEVEL", "DOTS_SIZE") else ""))
                return
            if st in self.IN_GAME or st == "GAME_OVER":
                self._tool_start_again(args, text)
                return
        if st in self.RESTING:
            self._select(entry)
            self.start_setup(f"the player chose {entry.name}")
        else:
            self.request_switch(entry, f"the player asked for {entry.name}")

    def _tool_teach_game(self, args, text):
        if self.state in self.RESTING:
            self.start_teaching("the player wants to teach me a game")
        else:
            self.request_switch("teach", "the player wants to teach me a game")

    def _tool_list_games(self, args, text):
        self.say(f"I know {self.library.spoken_list()}, and you can teach me a new one. Which would you like?")
        if self.state in ("CHAT", "GAME_OVER"):
            self.set_state("CHOOSE_GAME", "listed the games")

    def _tool_set_difficulty(self, args, text):
        level = str(args.get("level") or "").strip().lower()
        level = {"authentic": "adaptive", "adapt": "adaptive", "realistic": "adaptive", "adaptive": "adaptive"}.get(level, level)
        if level not in (*LEVELS, "adaptive"):
            self.say("Easy, medium, hard, impossible, or authentic?")
            return
        words = self.difficulty.set_level(level)
        self.level_asked = True
        self.app.trace("decide", f"difficulty set to {level} (strength {100 * self.difficulty.strength:.0f}%)")
        if self.state == "ASK_LEVEL":
            self.say(f"Alright, {words}.")
            why, self._setup_after_level = self._setup_after_level, None
            self._start_setup(why or "the level is chosen")
            return
        self.say(f"Alright, {words}." + (" " + self.last_said if self.state in ("CHOOSE_GAME", "ASK_SIDE") else ""))

    def _tool_new_board(self, args, text):
        size = str(args.get("size") or "").strip()
        st = self.state
        dots_entry = self.library.find("dots and boxes")
        if st == "DOTS_SIZE" and self.dots is not None:
            said = size if self.dots.parse_size_text(size) is not None else text
            if self.dots.parse_size_text(said) is None:
                self.say("How many boxes across and down? Two by two is the usual.")
            elif self.dots.take_size(said):               # says why when the size does not fit
                self.say(f"{D.size_words(self.dots.n).capitalize()} it is.")
                self._worker(self.dots.setup)
            return
        self._size_hint = size or text
        if st in ("DOTS_PLAYER_TURN", "DOTS_ROBOT_TURN"):
            self._tool_start_again(args, text)              # a new size mid-game: a fresh board of that size
        elif st in self.RESTING or self.entry is not dots_entry:
            self._select(dots_entry)
            self.start_setup("the player wants a dots and boxes board")
        # otherwise the size is kept for when the setup starts (for example after the level question)

    def _tool_start_again(self, args, text):
        entry = self.entry or self.library.find("noughts and crosses")
        if self.busy or self.state in self.BUSY_STATES:
            self.request_switch(entry, "the player wants to start again")
            return
        if self.dots is not None:
            self.dots.running = False
        self.hangman = None
        self.pending = None
        self.app.trace("decide", f"the player wants to start {entry.name} again")
        self.say(f"Another game of {entry.name}, then." if self.state == "GAME_OVER" else "Let's start again.")
        self.start_setup("the player wants to start again")

    def _tool_stop(self, args, text):
        if self.state in self.RESTING and not self.busy:
            self.set_state("IDLE", "the player does not want to play now")
            self.say("No problem at all. Just say Buddy whenever you fancy a game.")
        else:
            self.request_switch("stop", "the player wants to stop")

    # -- answers to Buddy's questions
    def _tool_answer(self, args, text):
        yes = args.get("yes", True)
        if isinstance(yes, str):
            yes = yes.strip().lower() in ("true", "yes", "1")
        self._act(self._u("yes" if yes else "no"), text, self.situation())

    def _tool_choose_side(self, args, text):
        side = str(args.get("side") or "").strip().lower()
        sym = "X" if "cross" in side or side == "x" else "O" if "nought" in side or side in ("o", "zero", "circle") else None
        if sym is None:
            self.say("Crosses or noughts?")
            return
        if self.state == "ASK_SIDE" and not self.busy:
            self.start_game(sym)
            return
        self._side_hint = sym
        self.say(f"{PLURAL[sym].capitalize()} for you, then.")

    def _tool_fresh_sheet_ready(self, args, text):
        if self.state in ("HANGMAN_SETUP", "GAME_OVER"):
            self._act(self._u("yes"), text, self.situation())
        else:
            self.say("Thank you. Let me have a look.")

    # -- the player's own move, said rather than drawn
    def _cell_index(self, words: str):
        t = re.sub(r"[^a-z ]+", " ", words.lower())
        t = re.sub(r"\bcenter\b", "centre", t)
        t = re.sub(r"\bmiddle\b", "centre", t) if re.fullmatch(r"\s*(the\s+)?middle\s*(one|square)?\s*", t) else t
        if t.strip() in tools.CELLS:
            return tools.CELLS.index(t.strip())
        rows = {"top": 0, "centre": 1, "middle": 1, "bottom": 2}
        cols = {"left": 0, "centre": 1, "middle": 1, "right": 2}
        r = next((v for k, v in rows.items() if re.search(rf"\b{k}\b", t)), None)
        c = next((v for k, v in cols.items() if re.search(rf"\b{k}\b", t) and not (k in ("centre", "middle") and r == 1 and t.count(k) < 2)), None)
        if r is None and c is None:
            return None
        return 3 * (1 if r is None else r) + (1 if c is None else c)

    def _tool_draw_my_mark(self, args, text):
        cell = str(args.get("cell") or "").strip().lower()
        idx = tools.CELLS.index(cell) if cell in tools.CELLS else self._cell_index(cell or text)
        if idx is None:
            self.say("Which square? Say the row and the column, like the top left, or the centre.")
            return
        if self.state == "CONFIRM_MARK":
            self.say(self.last_said)
            return
        if self.state != "PLAYER_TURN" or self.busy:
            self.say("Let me finish my go first.")
            return
        self._voice_move(idx)                               # checks free and legal, then draws

    def _tool_draw_my_line(self, args, text):
        if self.state != "DOTS_PLAYER_TURN" or self.dots is None or self.dots.state is None:
            self.say(self._refusal("draw_my_line"))
            return
        if self.busy:
            self.say("Let me finish this first.")
            return
        where = str(args.get("where") or "").strip()
        if self.dots.parse_edge(where) is None:
            if self.dots.parse_edge(text) is not None:
                where = text
            elif self.dots.parse_edge(where or text, include_drawn=True) is not None:
                self.say("That line is already drawn. Which other one would you like?")
                return
        self.dots._spoken_line(where or text)              # finds the line, checks it is free, then draws

    def _tool_say_letter(self, args, text):
        if self.state != "HANGMAN_TURN" or self.hangman is None:
            self.say(self._refusal("say_letter"))
            return
        if re.search(r"\b(?:or|and)\b|\b[A-Za-z]\s*[,/]\s*[A-Za-z]\b", text, re.I):
            self.say("One letter at a time, please. Which would you like?")
            return
        self._hangman_act(self._u("say_letter", letter=hangman_letter(str(args.get("letter") or ""))))

    def _tool_finish_my_line(self, args, text):
        if self.state != "DOTS_PLAYER_TURN" or self.dots is None or self.dots.state is None:
            self.say(self._refusal("finish_my_line"))
            return
        if self.busy:
            self.say("Let me finish this first.")
            return
        self.dots.finish_started()

    def _tool_i_am_done(self, args, text):
        self._act(self._u("done"), text, self.situation())

    # -- about the drawing
    def _tool_redraw_last_stroke(self, args, text):
        last = getattr(self.app, "last_strokes", None)
        if not last:
            self.say("There's nothing for me to draw again yet.")
            return
        if self.busy:
            self.say("Let me finish this first, then I'll go over it.")
            return
        name, strokes = last
        self.app.trace("decide", f"drawing {name} again, as the player asked")
        self.say("Of course. Let me go over that again.")
        self._worker(self._redraw, name, strokes)

    def _redraw(self, name, strokes):
        job = self.app.draw(name, strokes)
        job.done.wait()
        self._seen = None
        self._maybe_switch()

    def _tool_wait(self, args, text):
        self.say("Of course, take your time.")

    # -- talking about the game
    def _narrate(self, what: str, fallback):
        """buddy/agent/narration.py (another agent's) when it exists, else a short built-in line."""
        try:
            from . import narration
            fn = getattr(narration, what, None)
            if fn is not None:
                out = fn(self)
                if out:
                    return out
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            self.app.trace("decide", f"narration.{what} failed ({e}); using the built-in line")
        return fallback()

    def _last_move_name(self) -> str | None:
        ch = self.last_choice
        if ch is None:
            return None
        if self.entry is not None and self.entry.kind == "dots" and self.dots is not None:
            return D.edge_name(self.dots.n, ch.cell)
        return "the " + NAMES[ch.cell]

    def _score_fallback(self) -> str:
        e = self.entry or self.library.find("noughts and crosses")
        record = (f" Over all our games of {e.name}, you've won {e.player_wins} and I've won {e.robot_wins}." if e.plays else "")
        if self.dots is not None and self.dots.state is not None and self.state in DOTS_STATES:
            p, q = self.dots.state.score("P"), self.dots.state.score("B")
            turn = "your turn" if self.state == "DOTS_PLAYER_TURN" else "my turn"
            return f"You have {p} box{'es' if p != 1 else ''} and I have {q}. It's {turn}."
        if self.hangman is not None and self.state == "HANGMAN_TURN":
            spelled = " ".join("blank" if c == "_" else c for c in self.hangman.pattern)
            return f"So far the word reads {spelled}, with {self.hangman.remaining} misses left."
        if self.state in ("PLAYER_TURN", "CONFIRM_MARK"):
            return "No score yet in this game; it's your turn." + record
        if self.state == "ROBOT_TURN":
            return "No score yet in this game; it's my turn." + record
        return ("There's no game on just now." + record) if not record else record.strip()

    def _tool_tell_score(self, args, text):
        self.say(self._narrate("score_text", self._score_fallback))

    def _tool_explain_last_move(self, args, text):
        def fallback():
            name = self._last_move_name()
            if name is None:
                return "I haven't played a move yet this game."
            return f"I took {name} because it was {self.last_choice.reason}."
        self.say(self._narrate("explain_last_move", fallback))

    def _hint_fallback(self) -> str:
        best = Difficulty(level="impossible")
        if self.hangman is not None and self.state == "HANGMAN_TURN":
            self._hangman_act(self._u("hangman_hint"))
            return ""
        if self.dots is not None and self.dots.state is not None and self.state == "DOTS_PLAYER_TURN":
            ch = self.dots.solver.choose(self.dots.state, best, self.rng)
            return f"You could try {D.edge_name(self.dots.n, ch.cell)}."
        if self.state == "PLAYER_TURN":
            if self.spec is not None and self.gstate is not None:
                ch = self.solver.choose(self.gstate, best, self.rng)
            else:
                ch = choose(self.board, self.player_sym, best, self.rng)
            return f"You could try the {NAMES[ch.cell]}."
        return "I can only give a hint on your turn."

    def _tool_give_hint(self, args, text):
        out = self._narrate("hint_text", self._hint_fallback)
        if out:
            self.say(out)

    def _tool_explain_rules(self, args, text):
        self.say(self._narrate("rules_text", self.rules_text) + (" Shall we play?" if self.state == "CHAT" else ""))
        if self.state == "CHAT":
            self.set_state("CHOOSE_GAME", "explained the rules and offered a game")

    def _tool_repeat(self, args, text):
        self.say(self.last_said or "I haven't said anything yet.")

    def _tool_say(self, args, text):
        reply_text = str(args.get("text") or "").strip().strip('"')
        self.say(reply_text[:300] if reply_text else "I'm listening.")
