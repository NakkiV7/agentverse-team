"""The language model (Gemma 3 4B via Ollama, local): understands what the player said and writes a short reply.

It is never asked to choose moves, positions or arm motions. Its answer is forced into a JSON schema (Ollama's
structured outputs), so the state machine always gets a valid intent it can act on.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass

from .. import config as C

INTENTS = ["play_game", "choose_game", "yes", "no", "explain_rules", "repeat", "stop", "smalltalk", "unclear"]
GAMES = ["tic_tac_toe", "other", "none"]
SIDES = ["crosses", "noughts", "none"]

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "game": {"type": "string", "enum": GAMES},
        "side": {"type": "string", "enum": SIDES},
        "reply": {"type": "string"},
    },
    "required": ["intent", "game", "side", "reply"],
}

SYSTEM = """You are Buddy, a friendly robot arm that plays pen-and-paper games with older adults. You speak plain, \
warm British English: one or two short sentences, at most 25 words, no slang, no lists, no emoji.

You only know one game: noughts and crosses (tic-tac-toe). You draw the board yourself on a sheet of paper.

For every message from the player, return JSON:
- intent: play_game (they want to play), choose_game (they name a game), yes, no, explain_rules (they ask how to \
play or how it works), repeat (say that again), stop (stop, later, goodbye), smalltalk (anything social), unclear.
- game: tic_tac_toe for noughts and crosses / tic-tac-toe, other for any other game (chess, cards...), else none.
- side: crosses or noughts only if they choose one right now, else none.
- reply: a natural answer that fits the situation you are given. Never promise or describe steps that the \
situation does not mention.

Examples:
Player: how does it work again -> {"intent": "explain_rules", "game": "none", "side": "none", "reply": "Of course."}
Player: can we play chess -> {"intent": "choose_game", "game": "other", "side": "none", "reply": "I'm afraid I only \
know noughts and crosses."}
Player: go on then -> {"intent": "yes", "game": "none", "side": "none", "reply": "Lovely."}
Player: how are you today -> {"intent": "smalltalk", "game": "none", "side": "none", "reply": "I'm very well, \
thank you. Fancy a game of noughts and crosses?"}"""


@dataclass
class Understanding:
    intent: str
    game: str
    side: str
    reply: str
    seconds: float
    raw: str


def understand(utterance: str, situation: str, history: list[tuple[str, str]] | None = None) -> Understanding:
    msgs = [{"role": "system", "content": SYSTEM}]
    for who, text in (history or [])[-6:]:
        msgs.append({"role": "user" if who == "player" else "assistant", "content": text})
    msgs.append({"role": "user", "content": f"[Situation: {situation}]\nPlayer: {utterance}"})
    body = {"model": C.LLM_MODEL, "messages": msgs, "stream": False, "format": SCHEMA,
            "options": {"temperature": 0.3, "num_predict": 120}, "keep_alive": "30m"}
    t0 = time.time()
    req = urllib.request.Request(f"{C.OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        content = json.loads(r.read())["message"]["content"]
    dt = time.time() - t0
    try:
        d = json.loads(content)
        intent = d.get("intent") if d.get("intent") in INTENTS else "unclear"
        return Understanding(intent, d.get("game", "none"), d.get("side", "none"), str(d.get("reply", "")).strip(),
                             dt, content)
    except json.JSONDecodeError:
        return Understanding("unclear", "none", "none", "", dt, content)


def warm_up() -> float:
    """Load the model into memory so the first real question is fast."""
    t0 = time.time()
    understand("hello", "warming up")
    return time.time() - t0
