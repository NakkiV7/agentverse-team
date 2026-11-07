"""Owner: Brain. The deliberate layer: one state machine, driven by intents, percepts and game events.

States: IDLE -> GREET -> CHOOSE_GAME -> SETUP (draw the grid while explaining the rules)
        -> PLAYER_TURN <-> ROBOT_TURN -> GAME_OVER -> GREET or IDLE
Interrupts from any state (PDF p.6): hand in area (say why, resume when clear), sheet moved
(stop, re-find, continue), unclear mark (ask), two marks / out of turn (point it out kindly),
sheet out of reach (ask to move it closer).

Hour 7 milestone: a full typed game start to finish (typed intents from the ui, voice comes at hour 11).
"""
from __future__ import annotations

import enum

from buddy.bus import Bus
from buddy.contracts import Intent, MotionCommand, Say
from buddy.game.tictactoe import Game
from buddy.motion.planner import grid_strokes, x_strokes


class State(enum.Enum):
    IDLE = "idle"; GREET = "greet"; CHOOSE_GAME = "choose_game"; SETUP = "setup"
    PLAYER_TURN = "player_turn"; ROBOT_TURN = "robot_turn"; GAME_OVER = "game_over"


class Agent:
    def __init__(self, bus: Bus, llm=None):
        self.bus = bus
        self.llm = llm
        self.state = State.IDLE
        self.game = Game()

    def say(self, text: str, urgent: bool = False):
        self.bus.say.put(Say(text, "urgent" if urgent else "normal"))
        self.bus.transcript.put(f"Buddy: {text}")

    def on_intent(self, it: Intent):
        """TODO Brain: full table. This is enough to start a game and draw the grid."""
        if it.kind == "stop":
            self.bus.motion_cmds.put(MotionCommand(home=True)); self.state = State.IDLE
            self.say("All right, we can stop there. It was lovely playing with you.")
        elif self.state in (State.IDLE, State.GREET) and it.kind in ("play", "yes", "choose_game"):
            self.state = State.SETUP
            self.say("Let's play noughts and crosses. I'll draw the grid. You are O, I am X, and you go first.")
            self.bus.motion_cmds.put(MotionCommand(draw=grid_strokes()))
        elif self.state == State.PLAYER_TURN and it.kind == "move" and it.cell is not None:
            self.say(f"You'd like cell {it.cell}. Please draw your O there and I'll take my turn.")
        elif it.kind == "repeat":
            self.say("Of course. Three in a row wins. You are O and you go first.")
        elif it.kind == "chat" and it.text:
            self.say(it.text)

    def run(self):
        while not self.bus.stop.is_set():
            it = self.bus.intents.get(timeout=0.1)
            if it is not None:
                self.bus.transcript.put(f"Player: {it.text or it.kind}")
                self.on_intent(it)
            for done in self.bus.motion_done.drain():
                if done == "drawn" and self.state == State.SETUP:
                    self.state = State.PLAYER_TURN
                    self.say("There we are. Your move: draw an O in any square.")
            # TODO Brain: watch bus.percept for the player's mark (game.check_player), pick the robot's
            # move (game.robot_move), send MotionCommand(draw=x_strokes(cell)), handle results and interrupts.
