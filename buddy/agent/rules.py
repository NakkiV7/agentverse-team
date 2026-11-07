"""Fast path for understanding: plain phrase rules for the clear, frequent answers ("yes", "crosses", "how do I
play"). They answer instantly and are easy to explain; anything they do not match goes to the language model.
Each rule returns (intent, game, side) or None."""
from __future__ import annotations

import re

from .. import config as C

_YES = r"\b(yes|yeah|yep|sure|ok|okay|go on|alright|all right|of course|please do|why not|let'?s do it|lovely)\b"
_NO = r"\b(no|nope|not now|not today|maybe later|later|another time)\b"
_RULES = r"\b(how (do|does|to) (i|you|we|it)?\s*(play|work)|what are the rules|rules|explain)\b"
_REPEAT = r"\b(say (that|it) again|repeat|pardon|sorry\?|what did you say)\b"
_STOP = r"\b(stop|goodbye|bye|that'?s enough|i'?m done)\b"
_PLAY = r"\b(play|game)\b"
_TTT = r"(tic[\s-]?tac[\s-]?toe|noughts?|naughts?|crosses|x'?s and o'?s)"
_OTHER = (r"\b(chess|draughts|checkers|cards?|scrabble|sudoku|hangman|dominoes|bingo|poker|bridge|connect four|"
          r"crossword|battleships?|snakes and ladders|monopoly)\b")


def has_wake_word(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    return any(w in C.WAKE_WORDS for w in words)


def match(text: str, state: str):
    t = text.lower()
    if state == "ASK_SIDE":
        if re.search(r"\b(cross|crosses|x)\b", t) and not re.search(r"\bnoughts?\b", t):
            return ("choose_side", "none", "crosses")
        if re.search(r"\b(nought|noughts|naught|naughts|o|circles?)\b", t):
            return ("choose_side", "none", "noughts")
    if re.search(_STOP, t):
        return ("stop", "none", "none")
    if re.search(_RULES, t):
        return ("explain_rules", "none", "none")
    if re.search(_REPEAT, t):
        return ("repeat", "none", "none")
    if re.search(_TTT, t) and state != "ASK_SIDE":
        return ("choose_game", "tic_tac_toe", "none")
    if re.search(_OTHER, t):
        return ("choose_game", "other", "none")
    if re.search(_NO, t):
        return ("no", "none", "none")
    if re.search(_YES, t):
        return ("yes", "none", "none")
    if re.search(_PLAY, t):
        return ("play_game", "none", "none")
    return None
