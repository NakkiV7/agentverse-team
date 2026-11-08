"""Learning a new game from the player, in conversation.

The language model does language only: it turns what the player said into a GameSpec draft and, when
something is missing or ambiguous, ONE clarifying question. It never decides a move; the solver does.
Gemma 4 E4B is used here (measured 7/8 first-turn extractions and 2/2 interview completions on the eval
in tools/teach_eval.py); Gemma 3 4B guessed rules when unsure, so it stays on the in-game intents.
Gemma 4 runs on Ollama's MLX backend without grammar support, so JSON is asked for in the prompt and
checked here field by field.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field

from buddy import config as C
from buddy.game.rules import ENDS, PLACEMENTS, GameSpec

NUM_CTX = 8192   # the same context size as the planner (toolcall.NUM_CTX): Ollama reloads a model whenever a request asks
                 # for a different one, which cost several seconds between a teaching call and a planner call

TEACH_MODEL = getattr(C, "TEACH_MODEL", "gemma4")
MAX_QUESTIONS = 3

SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {"type": "integer", "enum": [3, 4]},
        "cols": {"type": "integer", "enum": [3, 4]},
        "placement": {"type": "string", "enum": list(PLACEMENTS) + ["unspecified"]},
        "end": {"type": "string", "enum": list(ENDS) + ["unsupported", "unspecified"]},
        "k": {"type": "integer", "enum": [0, 3, 4]},
        "first_player": {"type": "string", "enum": ["player", "robot", "unspecified"]},
        "question": {"type": "string"},
        "name": {"type": "string"},
    },
    "required": ["rows", "cols", "placement", "end", "k", "first_player", "question", "name"],
}

SYSTEM = """You help Buddy, a robot that plays pen-and-paper games on a 3x3 or 4x4 grid where two players take turns writing X or O.
The player is explaining the rules of a game. Fill in the game description ONLY from what they actually said.

Fields:
- rows, cols: grid size; 3 and 3 unless they say otherwise.
- placement: "anywhere" (any empty square), "adjacent_to_last_mark" (only up/down/left/right of the most recent mark), "adjacent_to_last_mark_or_diagonal" (next to the most recent mark, diagonals included), or "unspecified".
- end: "k_in_a_row_wins" (a line of k wins), "k_in_a_row_loses" (a line of k loses), "no_legal_move_loses" (the player who cannot move loses), "unspecified", or "unsupported".
- k: the line length for k_in_a_row rules, else 0.
- first_player: who starts, if said; else "unspecified".
- question: ONE short, friendly clarifying question, or "" if nothing is needed.
- name: the game's name if the player said one ("it's called Footsteps", "we call it Trails"), else "".

Rules for filling it in:
1. If the player did not say how the game ENDS, end = "unspecified" and ask about it.
2. If the player did not say WHERE marks may go, placement = "unspecified" and ask about it.
3. "next to each other" / "touching" without saying whether it means the LAST mark or ANY mark, or without saying whether diagonals count, is ambiguous: ask.
4. If the game is not two players taking turns writing X or O on a grid (for example it uses numbers, words, cards or dice), end = "unsupported" and say so kindly in the question.
5. Never invent a rule. An unspecified field with an empty question is an error.
Answer with one JSON object only, no prose, no code fence, integers as numbers, matching this JSON schema: """ + json.dumps(SCHEMA)


@dataclass
class Draft:
    rows: int = 3
    cols: int = 3
    placement: str = "unspecified"
    end: str = "unspecified"
    k: int = 0
    first_player: str = "unspecified"
    question: str = ""
    name: str = ""

    @property
    def unsupported(self) -> bool:
        return self.end == "unsupported"

    @property
    def complete(self) -> bool:
        return self.placement != "unspecified" and self.end not in ("unspecified", "unsupported") and \
            (self.end == "no_legal_move_loses" or self.k in (3, 4))

    def to_spec(self, name: str = "") -> GameSpec:
        return GameSpec(rows=self.rows, cols=self.cols, placement=self.placement, end=self.end,
                        k=self.k if self.end != "no_legal_move_loses" else 3,
                        first_player="robot" if self.first_player == "robot" else "player", name=name)


@dataclass
class TeachTurn:
    draft: Draft
    say: str                         # what Buddy says next (a question, or the rules read back)
    done: bool                       # True when the spec is complete or the game cannot be learned
    spec: GameSpec | None = None
    seconds: float = 0.0


@dataclass
class Teacher:
    """One teaching conversation. Call hear() with each thing the player says."""
    model: str = TEACH_MODEL
    msgs: list = field(default_factory=lambda: [{"role": "system", "content": SYSTEM}])
    questions: int = 0
    draft: Draft = field(default_factory=Draft)

    def _chat(self) -> str:
        body = {"model": self.model, "messages": self.msgs, "stream": False, "think": False, "keep_alive": "30m",
                "options": {"temperature": 0.0, "num_predict": 160, "num_ctx": NUM_CTX}}
        req = urllib.request.Request(f"{C.OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["message"]["content"]

    @staticmethod
    def _parse(txt: str) -> Draft:
        i, j = txt.find("{"), txt.rfind("}")
        d = json.loads(txt[i:j + 1]) if i >= 0 else {}
        def num(v, allowed, default):
            try:
                v = int(v)
            except (TypeError, ValueError):
                return default
            return v if v in allowed else default
        placement = d.get("placement") if d.get("placement") in PLACEMENTS else "unspecified"
        end = d.get("end") if d.get("end") in ENDS or d.get("end") == "unsupported" else "unspecified"
        first = d.get("first_player") if d.get("first_player") in ("player", "robot") else "unspecified"
        return Draft(num(d.get("rows"), (3, 4), 3), num(d.get("cols"), (3, 4), 3), placement, end,
                     num(d.get("k"), (0, 3, 4), 0), first, str(d.get("question") or "").strip(),
                     str(d.get("name") or "").strip()[:40])

    def hear(self, text: str, name: str = "") -> TeachTurn:
        t0 = time.perf_counter()
        if self.questions:
            text = f'Buddy asked: "{self.draft.question}"\nThe player answered: "{text}"\nUpdate the game description with everything said so far.'
        else:
            text = f'The player said: "{text}"'
        self.msgs.append({"role": "user", "content": text})
        raw = self._chat()
        self.msgs.append({"role": "assistant", "content": raw})
        self.draft = self._parse(raw)
        dt = time.perf_counter() - t0
        if self.draft.unsupported:
            q = self.draft.question or "I can only play games where we take turns writing X and O on a grid."
            return TeachTurn(self.draft, q, True, None, dt)
        if self.draft.complete:
            spec = self.draft.to_spec(name or self.draft.name)
            return TeachTurn(self.draft, "Let me make sure I have it. " + spec.describe(), True, spec, dt)
        if self.questions >= MAX_QUESTIONS:
            return TeachTurn(self.draft, "I'm sorry, I haven't quite understood that one. Shall we play noughts and crosses instead?", True, None, dt)
        self.questions += 1
        q = self.draft.question or _default_question(self.draft)
        return TeachTurn(self.draft, q, False, None, dt)


def warm_up(model: str = TEACH_MODEL) -> float:
    """Load the teaching model once at start-up so the first lesson is not slowed by a cold start (~6 s)."""
    t0 = time.perf_counter()
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": False, "think": False,
            "keep_alive": "30m", "options": {"num_predict": 1, "num_ctx": NUM_CTX}}
    req = urllib.request.Request(f"{C.OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120):
        pass
    return time.perf_counter() - t0


def _default_question(d: Draft) -> str:
    if d.placement == "unspecified":
        return "Where can a mark go: any empty square, or only next to the last one played?"
    if d.end == "unspecified":
        return "And how does the game end: a line of three, or when someone can't move?"
    return "Is it three in a row, or four?"
