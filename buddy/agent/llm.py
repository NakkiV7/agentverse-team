"""Language understanding with Gemma 3 4B (Ollama, on the laptop).

Every utterance becomes one structured call: the reply is forced into a JSON schema (Ollama structured outputs) with
the intent, the game, the side, the cell for a spoken move ("put mine in the top left") and, only for small talk,
a short reply. The state machine acts on these fields; the model never chooses moves, positions or arm motions.
The situation (what Buddy just asked, whose turn it is, which squares are free) is given with every call.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass

from .. import config as C

INTENTS = ["play_game", "choose_game", "yes", "no", "explain_rules", "repeat", "stop", "smalltalk", "choose_side",
           "move", "done", "new_game", "teach_game", "choose_difficulty", "list_games", "unclear"]
INTENTS += ["say_letter", "hangman_hint"]
GAMES = ["tic_tac_toe", "other", "none"]
SIDES = ["crosses", "noughts", "none"]
LEVELS = ["easy", "medium", "hard", "impossible", "adaptive", "none"]
CELLS = ["top_left", "top_middle", "top_right", "middle_left", "centre", "middle_right", "bottom_left",
         "bottom_middle", "bottom_right", "none"]

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "game": {"type": "string", "enum": GAMES},
        "side": {"type": "string", "enum": SIDES},
        "cell": {"type": "string", "enum": CELLS},
        "game_name": {"type": "string"},
        "level": {"type": "string", "enum": LEVELS},
    },
    "required": ["intent", "game", "side", "cell", "game_name", "level"],
}
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["none"]
SCHEMA["properties"]["letter"] = {"type": "string", "enum": LETTERS}
SCHEMA["required"].append("letter")

SYSTEM = """You are the language understanding of Buddy, a friendly robot arm that plays noughts and crosses \
(tic-tac-toe) on paper with older adults. Read what the player said, in the situation given, and return JSON.

intent:
- play_game: wants to play a game. choose_game: names a game (set game, and copy the game's name into game_name). \
new_game: wants another game. list_games: asks which games Buddy knows.
- choose_difficulty: asks Buddy to play easier or harder, or names a level (set level: easy, medium, hard, impossible, \
or adaptive for "adjust to me"). "go easy on me" is easy, "play your best" / "don't hold back" is impossible.
- teach_game: ONLY when the player offers to teach or explain a game of THEIR OWN ("let me teach you one", "I know a \
different game", "here's how mine works") or starts stating rules ("we take turns putting..."). "Let's play a game" \
or "shall we play" is play_game, never teach_game.
- yes / no: agrees or declines what Buddy just asked.
- explain_rules: asks how to play. repeat: wants Buddy to say it again. stop: wants to stop or says goodbye.
- choose_side: picks crosses or noughts (set side).
- move: says where THEIR mark should go (set cell); Buddy will draw it for them.
- done: says they have finished their turn ("your turn", "done", "go on").
- smalltalk: anything social. unclear: cannot tell.
game: tic_tac_toe for noughts and crosses / tic-tac-toe / X's and O's; other for any other game; else none.
game_name: the game's name as the player said it (e.g. "footsteps", "the trail game"), else "". The situation lists \
the games Buddy knows; a known game is still intent choose_game with game "other" and its name in game_name.
level: only with choose_difficulty, else none.
side: crosses or noughts only with choose_side, else none.
cell: the square as the player sees the paper: top/middle/bottom row, left/middle/right column; the very middle \
is centre. "top" alone is top_middle, "left" alone is middle_left. none unless intent is move.

Examples:
"how does it work again" -> {"intent":"explain_rules","game":"none","side":"none","cell":"none","game_name":"","level":"none"}
"go on then" -> {"intent":"yes","game":"none","side":"none","cell":"none","game_name":"","level":"none"}
"Buddy, let's play a game" -> {"intent":"play_game","game":"none","side":"none","cell":"none","game_name":"","level":"none"}
"I'll be noughts" -> {"intent":"choose_side","game":"none","side":"noughts","cell":"none","game_name":"","level":"none"}
"put mine in the top left please" -> {"intent":"move","game":"none","side":"none","cell":"top_left","game_name":"","level":"none"}
"the middle one" -> {"intent":"move","game":"none","side":"none","cell":"centre","game_name":"","level":"none"}
"your go" -> {"intent":"done","game":"none","side":"none","cell":"none","game_name":"","level":"none"}
"can we play chess" -> {"intent":"choose_game","game":"other","side":"none","cell":"none","game_name":"","level":"none"}
"let me teach you a game my dad taught me" -> {"intent":"teach_game","game":"other","side":"none","cell":"none","game_name":"","level":"none"}
"it's a different one, you put each mark next to the last" -> {"intent":"teach_game","game":"other","side":"none","cell":"none","game_name":"","level":"none"}
"how are you today" -> {"intent":"smalltalk","game":"none","side":"none","cell":"none","game_name":"","level":"none"}
"let's play footsteps" -> {"intent":"choose_game","game":"other","side":"none","cell":"none","game_name":"footsteps","level":"none"}
"what games do you know" -> {"intent":"list_games","game":"none","side":"none","cell":"none","game_name":"","level":"none"}
"go easy on me" -> {"intent":"choose_difficulty","game":"none","side":"none","cell":"none","game_name":"","level":"easy"}
"play your best, no holding back" -> {"intent":"choose_difficulty","game":"none","side":"none","cell":"none","game_name":"","level":"impossible"}"""


SYSTEM += """\nHangman additions:
- say_letter: in hangman, the player names one letter. Set letter to its capital A to Z.
  Examples: "E" or "the letter e please" -> say_letter, letter E; "bee" -> B; "sea" -> C;
  "why" -> Y; "eye" -> I; "queue" -> Q; "double you" -> W; "A for apple" -> A.
  Multiple letters or an uncertain letter -> unclear, letter none. Never choose a letter yourself.
- hangman_hint: asks for a hint in hangman. Set letter none; the game code chooses the hint.
- "let's play hangman" -> choose_game, game other, game_name hangman, letter none.
Always include letter, using none except for say_letter. All earlier examples have letter none.
"""

REPLY_SYSTEM = """You are Buddy, a friendly robot arm that plays noughts and crosses on paper with older adults. \
Answer the player in one warm sentence of plain British English, at most 20 words, no slang, no emoji. You only \
know noughts and crosses. Never describe actions or moves that the situation does not mention."""


@dataclass
class Understanding:
    intent: str
    game: str
    side: str
    cell: int | None          # 0..8 row by row from the player's top-left
    seconds: float
    game_name: str = ""
    level: str = "none"
    letter: str | None = None


def _chat(body: dict) -> str:
    req = urllib.request.Request(f"{C.OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["message"]["content"]


class Conversation:
    """One running conversation with the model. Each call appends to it, so Ollama only has to process the newest
    message (about 0.4 s) instead of the whole instruction prompt (about 2.2 s). It restarts when it gets long."""
    MAX_TURNS = 24

    def __init__(self):
        self.reset()

    def reset(self):
        self.msgs = [{"role": "system", "content": SYSTEM}]

    def understand(self, utterance: str, situation: str) -> Understanding:
        if len(self.msgs) > 2 * self.MAX_TURNS:
            self.reset()
        self.msgs.append({"role": "user", "content": f"Situation: {situation}\nPlayer said: \"{utterance}\""})
        body = {"model": C.LLM_MODEL, "stream": False, "format": SCHEMA, "keep_alive": "60m",
                "options": {"temperature": 0.0, "num_predict": 96}, "messages": self.msgs}
        t0 = time.time()
        content = _chat(body)
        self.msgs.append({"role": "assistant", "content": content})
        d = json.loads(content)
        cell = d.get("cell", "none")
        if d.get("intent") == "say_letter" and re.search(r"\b(?:or|and)\b|\b[A-Za-z]\s*[,/]\s*[A-Za-z]\b", utterance, re.I):
            d["intent"], d["letter"] = "unclear", "none"
        return Understanding(d.get("intent", "unclear"), d.get("game", "none"), d.get("side", "none"),
                             CELLS.index(cell) if cell in CELLS[:9] else None, time.time() - t0,
                             str(d.get("game_name") or "").strip(), d.get("level") or "none",
                             d.get("letter") if d.get("letter") in LETTERS[:26] else None)


def reply(utterance: str, situation: str) -> tuple[str, float]:
    """A short spoken answer, only for small talk and other free conversation."""
    body = {"model": C.LLM_MODEL, "stream": False, "keep_alive": "60m",
            "options": {"temperature": 0.6, "num_predict": 45},
            "messages": [{"role": "system", "content": REPLY_SYSTEM},
                         {"role": "user", "content": f"Situation: {situation}\nPlayer said: \"{utterance}\""}]}
    t0 = time.time()
    text = _chat(body).strip().strip('"')
    return text, time.time() - t0


def warm_up(conv: Conversation) -> float:
    """Load the model and process the instructions once, so the first real question is fast."""
    t0 = time.time()
    conv.understand("hello", "Buddy is starting up.")
    reply("hello", "Buddy is starting up.")
    return time.time() - t0
