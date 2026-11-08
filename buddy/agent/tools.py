"""Buddy's abilities as tools the language model may choose from.

The player says anything; Gemma 4 (via Ollama tool calling, see toolcall.py) picks one or more of these tools with
arguments; the brain checks each one against the rules, the state and the reach limits and runs it (brain.run_tools).
The model never chooses a game move: draw_my_mark / draw_my_line only carry out a square or a line the PLAYER named,
and the solver decides Buddy's own moves. Safety (the hand freeze) is below all of this and unaffected.

Three roles share this file:
  - tool descriptions (the `description` and parameter texts) are owned by the evaluation work (tools/tool_eval.py):
    they are prompt engineering and get tuned against measured accuracy;
  - the executor (brain.run_tools) maps names to brain methods; adding a tool means adding both;
  - `states` says where a tool makes sense; tools outside the current state are not offered to the model.

Keep names stable: they appear in traces, tests and the demo page.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ANY = ("*",)
GAME_ON = ("PLAYER_TURN", "ROBOT_TURN", "CONFIRM_MARK", "DOTS_PLAYER_TURN", "DOTS_ROBOT_TURN", "HANGMAN_TURN")
GRID_TURN = ("PLAYER_TURN", "CONFIRM_MARK")
DOTS_TURN = ("DOTS_PLAYER_TURN",)
PENDING_QUESTION = ("ASK_LEVEL", "ASK_SIDE", "CONFIRM_MARK", "OFFER_TAUGHT", "DOTS_SIZE", "WAIT_SHEET", "HANGMAN_SETUP", "GAME_OVER", "CHOOSE_GAME")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params: dict = field(default_factory=dict)      # JSON schema "properties"
    required: tuple = ()
    states: tuple = ANY                             # where it is offered; ANY = everywhere outside TEACH / NAME_GAME
    confirm: bool = False                           # ask "yes?" before doing it when a game is in progress

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": self.params, "required": list(self.required)}}}


LEVELS = ["easy", "medium", "hard", "impossible", "authentic"]
SIDES = ["crosses", "noughts"]
CELLS = ["top left", "top middle", "top right", "middle left", "centre", "middle right",
         "bottom left", "bottom middle", "bottom right"]

TOOLS: list[Tool] = [
    # --- choosing what to play -------------------------------------------------------------------------
    Tool("switch_game", "Play a different game that Buddy knows (the situation lists them), or the same game from "
         "the start when the player wants a fresh game. Use it whenever the player names a game to play, even in "
         "the middle of another game.", {"name": {"type": "string", "description": "the game's name as the player said it"}}, ("name",)),
    Tool("teach_game", "The player wants to teach Buddy a game of their own, or starts explaining rules of a game "
         "Buddy does not know."),
    Tool("list_games", "The player asks which games Buddy knows."),
    Tool("set_difficulty", "The player asks Buddy to play easier or harder, or names a level. 'authentic' means "
         "adapt to the player over time.", {"level": {"type": "string", "enum": LEVELS}}, ("level",)),
    Tool("new_board", "Dots and boxes: the player says how big a board they want, e.g. 'three by four', "
         "'five across and two down', 'the usual'.", {"size": {"type": "string"}}, ("size",), ("DOTS_SIZE", "DOTS_PLAYER_TURN", "DOTS_ROBOT_TURN", "CHOOSE_GAME")),
    Tool("start_again", "Restart the current game from the beginning (a fresh sheet).", states=GAME_ON + ("GAME_OVER",)),
    Tool("stop", "The player wants to stop playing, says goodbye, or wants a rest."),
    # --- answers to Buddy's questions --------------------------------------------------------------------
    Tool("answer", "A plain yes or no to the question Buddy just asked (crosses or noughts, is that your mark, shall "
         "we play it, fresh sheet ready).", {"yes": {"type": "boolean"}}, ("yes",), PENDING_QUESTION),
    Tool("choose_side", "The player picks crosses or noughts.", {"side": {"type": "string", "enum": SIDES}}, ("side",), ("ASK_SIDE", "SETUP", "CHOOSE_GAME")),
    Tool("fresh_sheet_ready", "The player says they have put a new or blank sheet of paper down.", states=("WAIT_SHEET", "DOTS_SETUP", "HANGMAN_SETUP", "GAME_OVER", "SETUP")),
    # --- the player's own move, said rather than drawn -------------------------------------------------------
    Tool("draw_my_mark", "Noughts and crosses and the other grid games: the player says which square THEY want, and "
         "Buddy draws their mark there for them.", {"cell": {"type": "string", "enum": CELLS}}, ("cell",), GRID_TURN),
    Tool("draw_my_line", "Dots and boxes: the player says which line THEY want and Buddy draws it for them, e.g. "
         "'the top side of the top left box', 'row two column three, left side'.", {"where": {"type": "string"}}, ("where",), DOTS_TURN),
    Tool("say_letter", "Hangman: the player guesses a letter ('E', 'the letter S', 'bee' is B, 'double you' is W).",
         {"letter": {"type": "string", "description": "one capital letter A-Z"}}, ("letter",), ("HANGMAN_TURN",)),
    Tool("finish_my_line", "Dots and boxes: the player asks Buddy to complete or tidy a line they started.", states=DOTS_TURN),
    Tool("i_am_done", "The player says they have finished their turn / drawn their mark ('your go', 'done', 'look').",
         states=("PLAYER_TURN", "DOTS_PLAYER_TURN")),
    # --- about the drawing -------------------------------------------------------------------------------
    Tool("redraw_last_stroke", "Draw Buddy's most recent stroke again because it came out faint, broken or wrong.",
         states=GAME_ON + ("ASK_SIDE", "GAME_OVER")),
    Tool("wait", "The player asks Buddy to wait, hold on, pause, or give them a moment."),
    # --- talking about the game ----------------------------------------------------------------------------
    Tool("tell_score", "Say the score or whose turn it is.", states=GAME_ON + ("GAME_OVER",)),
    Tool("explain_rules", "Explain how the current game works, or repeat the rules."),
    Tool("explain_last_move", "Explain why Buddy played its last move.", states=GAME_ON + ("GAME_OVER",)),
    Tool("give_hint", "Suggest a move or a letter to the player, without playing it.", states=GAME_ON),
    Tool("repeat", "Say the last thing again; the player did not catch it."),
    Tool("say", "Reply in words only, when nothing above fits (small talk, a question about Buddy, thanks). One or "
         "two short, warm sentences, plain British English, no emoji.", {"text": {"type": "string"}}, ("text",)),
]

BY_NAME = {t.name: t for t in TOOLS}


def available(state: str) -> list[Tool]:
    """The tools that make sense in this state (TEACH and NAME_GAME have their own flows and use none)."""
    if state in ("TEACH", "NAME_GAME"):
        return []
    return [t for t in TOOLS if t.states == ANY or state in t.states]


def schemas(state: str) -> list[dict]:
    return [t.schema() for t in available(state)]
