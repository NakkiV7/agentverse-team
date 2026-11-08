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
    Tool("switch_game", "Only a request to play a named game NOW. Past memories ('we used to play hangman at school') "
         "or family mentions are say. Dots game = dots and boxes; word game = hangman; trail game = Footsteps. "
         "Choose the requested game, not the rejected one. An unnamed new round is start_again.",
         {"name": {"type": "string", "description": "The requested game's full name from Games Buddy knows. Tic-tac-toe / X's and O's means noughts and crosses."}}, ("name",)),
    Tool("teach_game", "The player wants to teach Buddy a game of their own, or starts explaining rules of a game "
         "Buddy does not know."),
    Tool("list_games", "The player asks which games Buddy knows."),
    Tool("set_difficulty", "Change playing strength. Go easy / gentler = easy; a bit harder = hard; don't hold back / "
         "play your best = impossible; authentic / match my level / adapt to me = authentic. 'Something easier than "
         "this' without naming a game or strength is unclear: ask with say.",
         {"level": {"type": "string", "enum": LEVELS, "description": "The requested strength; authentic is adaptive."}}, ("level",)),
    Tool("new_board", "Dots and boxes board SIZE: dimensions such as three by four, five across and two down, or "
         "the usual. A game name alone is switch_game. Pass even oversized requests for the executor to check.",
         {"size": {"type": "string", "description": "Copy the player's exact words, including the size. Do not change numbers or across/down."}}, ("size",), ("DOTS_SIZE", "DOTS_PLAYER_TURN", "DOTS_ROBOT_TURN", "CHOOSE_GAME")),
    Tool("start_again", "Restart the current game: same again, another round, start over, begin from scratch. "
         "A bare yes to an offer is answer. A different named game is switch_game.", states=GAME_ON + ("GAME_OVER",)),
    Tool("stop", "The player wants to stop playing, says goodbye, or wants a rest."),
    # --- answers to Buddy's questions --------------------------------------------------------------------
    Tool("answer", "Answer Buddy's last question: yes / go on then = true; no / no thanks = false. "
         "Use a more specific tool for a named game, side, level or explicit report of new paper.",
         {"yes": {"type": "boolean", "description": "True for agreement, false for refusal."}}, ("yes",), PENDING_QUESTION),
    Tool("choose_side", "Choose the player's marks: crosses or noughts (the round ones).",
         {"side": {"type": "string", "enum": SIDES, "description": "The player's chosen mark, not Buddy's."}}, ("side",), ("ASK_SIDE", "SETUP", "CHOOSE_GAME")),
    Tool("fresh_sheet_ready", "The player explicitly reports a new, clean or blank sheet is down or ready. "
         "This asks the camera to check it; it does not prove the paper is usable. A bare yes is answer.", states=("WAIT_SHEET", "DOTS_SETUP", "HANGMAN_SETUP", "GAME_OVER", "SETUP")),
    # --- the player's own move, said rather than drawn -------------------------------------------------------
    Tool("draw_my_mark", "Draw the player's mark in the exact square THEY chose, e.g. 'stick mine in the corner, "
         "top left'. Never choose a square. 'Would top left be good?' asks for give_hint, not drawing.",
         {"cell": {"type": "string", "enum": CELLS, "description": "The player's named square. The middle / centre means centre."}}, ("cell",), GRID_TURN),
    Tool("draw_my_line", "Dots and boxes: the player says which line THEY want and Buddy draws it for them, e.g. "
         "'the top side of the top left box', 'row two column three, left side'. Never choose the line yourself.",
         {"where": {"type": "string", "description": "Copy the player's exact words naming the line. The executor resolves the edge."}}, ("where",), DOTS_TURN),
    Tool("say_letter", "The player chooses a hangman letter. Examples: E means letter E; Why means letter Y; "
         "En means letter N; Eye means letter I. Always submit a single spoken letter name here, not as chat. "
         "Multiple letters need clarification.",
         {"letter": {"type": "string", "description": "One capital A-Z. A for apple=A; bee=B; sea=C; aitch=H; eye=I; em=M; en=N; queue=Q; tee=T; double you=W; why=Y; zed=Z."}}, ("letter",), ("HANGMAN_TURN",)),
    Tool("finish_my_line", "Dots and boxes: the player asks Buddy to complete or tidy a line they started.", states=DOTS_TURN),
    Tool("i_am_done", "The player reports their mark or line is drawn: 'your turn now', 'your go', 'done', 'have a "
         "look'. A statement handing over the turn uses this tool; a question about whose turn uses tell_score.",
         states=("PLAYER_TURN", "DOTS_PLAYER_TURN")),
    # --- about the drawing -------------------------------------------------------------------------------
    Tool("redraw_last_stroke", "Draw Buddy's last stroke again: it came out faint, broken, or did not show. "
         "'Do that again, it came out faint' means this, not repeat (speech) or start_again (a new game).",
         states=GAME_ON + ("ASK_SIDE", "GAME_OVER")),
    Tool("wait", "Pause briefly: wait, hang on, hold on a sec, give me a moment, don't move yet. "
         "Keep the game. Ending the session is stop."),
    # --- talking about the game ----------------------------------------------------------------------------
    Tool("tell_score", "Report the score, who won, who is winning, or whose go it is.", states=GAME_ON + ("GAME_OVER",)),
    Tool("explain_rules", "Explain how the current game works, or repeat the rules."),
    Tool("explain_last_move", "Explain why Buddy played its last move.", states=GAME_ON + ("GAME_OVER",)),
    Tool("give_hint", "Ask the solver for advice, a clue, or a good move/letter. Includes 'would top left be good?' "
         "and 'pick a good corner'. Do not draw or choose any move yourself.", states=GAME_ON),
    Tool("repeat", "Say the last thing again; the player did not catch it."),
    Tool("say", "Speak through this tool for small talk, thanks, or one question when unsure. Memories about games "
         "are conversation. Questions ABOUT the paper ('what have you written in the corner', 'which letters are "
         "crossed out') are say; requests to change or keep marking the paper are custom_request. "
         "Use specific tools for clear requests, including spoken letter names in hangman. "
         "All replies must be native tool calls; put conversation in this tool's text argument.",
         {"text": {"type": "string", "description": "One or two short, warm sentences in plain British English, no emoji. Ask when unsure."}}, ("text",)),
    Tool("custom_request", "The player asks Buddy to write, draw, mark or keep doing something on the paper "
         "that is not a game move or an existing tool: write the alphabet and cross out wrong letters, keep a "
         "tally, write the score, underline letters, write a name, or stop crossing letters out. Pass even "
         "unsupported drawing requests for the paper compiler to check. Questions ABOUT what is already "
         "written or crossed out are say. A chosen game square/line/letter, faint stroke, pause or new board "
         "uses its existing tool. Stopping a paper rule is custom_request; stopping play is stop.",
         {"request": {"type": "string", "description": "Copy the player's complete words verbatim, including all requested actions and timing. Do not summarise or rephrase."}},
         ("request",), GAME_ON + ("CHAT", "CHOOSE_GAME", "ASK_LEVEL", "ASK_SIDE", "OFFER_TAUGHT", "WAIT_SHEET",
                                 "SETUP", "DOTS_SIZE", "DOTS_SETUP", "HANGMAN_SETUP", "GAME_OVER")),
]

BY_NAME = {t.name: t for t in TOOLS}


def available(state: str) -> list[Tool]:
    """The tools that make sense in this state (TEACH and NAME_GAME have their own flows and use none)."""
    if state in ("TEACH", "NAME_GAME"):
        return []
    return [t for t in TOOLS if t.states == ANY or state in t.states]


def schemas(state: str) -> list[dict]:
    return [t.schema() for t in available(state)]
