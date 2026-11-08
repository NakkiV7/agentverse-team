"""Learning a game from the player's own explanation, when it is one Penny can already play.

For the demo Penny starts with dots and boxes hidden (config.HIDE_GAMES): it is not on the menu, not in the facts,
and "let's play dots and boxes" leads to "teach me". When the player then explains the game, recognise() spots which
hidden game it is from the words, and a Relearn interview asks two short clarifying questions through Gemma 4
(grounded in what was said, never about what was already explained), reads the rules back in Penny's words, asks
the player to draw one example move on the paper so the camera can see it, and reveals the game. From then on the
ordinary dots-and-boxes engine plays it.

Everything the model produces is validated; a question or summary that fails validation falls back to a fixed one,
so the flow never stalls on the model.
"""
from __future__ import annotations

import difflib
import json
import re
import time
import urllib.request

from .. import config as C

MODEL = getattr(C, "TEACH_MODEL", "gemma4")
NUM_CTX = 8192
TIMEOUT_S = 30
MAX_QUESTIONS = 2
MIN_HITS = 4                      # distinct hint words needed before a description is taken for a hidden game

HINTS = {
    "dots and boxes": {"dot", "dots", "box", "boxes", "line", "lines", "grid", "square", "squares", "join", "joining",
                       "joins", "connect", "connecting", "connects", "initial", "initials", "claim", "claims",
                       "complete", "completes", "completing", "fourth", "side", "sides", "adjacent", "horizontal",
                       "vertical"},
}

FALLBACK_QUESTIONS = {
    "dots and boxes": ["When someone draws the fourth side of a box, do they take another turn straight away?",
                       "How does it end: when every line is drawn, and whoever has more boxes wins?"],
}

FALLBACK_SUMMARY = {
    "dots and boxes": ("So, if I've understood: we take turns joining two dots next to each other with a line. "
                       "Whoever draws the fourth side of a box writes their initial in it and goes again. When every "
                       "line is drawn, the one with more boxes wins."),
}

SYSTEM = """You are Penny, a friendly robot arm learning a pen-and-paper game from an older adult's explanation. \
You will play it on real paper, so you need the rules exactly. Reply with JSON only, no other text."""

QUESTION_PROMPT = """From the explanation and the answers so far, ask ONE short, warm clarifying question (under 22 \
words, plain British English) about a rule you genuinely cannot infer yet: who moves next after a box is completed, \
how the game ends and who wins, how big the grid of dots is, who starts, or what to write in a claimed box. Never \
ask about something already explained or already answered, and never ask the same thing twice.
Reply exactly: {"question": "..."}"""

SUMMARY_PROMPT = """Now read the rules back to the player in two or three short sentences, starting "So, if I've \
understood:", using only what they said and answered. No questions, no emoji.
Reply exactly: {"summary": "..."}"""


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def recognise(library, text: str):
    """The hidden game this explanation describes, or None. Only hidden games are considered."""
    words = _words(text)
    best, best_hits = None, 0
    for entry in library.entries:
        if not library.is_hidden(entry):
            continue
        hits = len(words & HINTS.get(entry.name.lower(), set()))
        if hits >= MIN_HITS and hits > best_hits:
            best, best_hits = entry, hits
    return best


class Relearn:
    """One interview about a hidden game. ask() twice, answer() after each, then summary()."""

    def __init__(self, entry, description: str, model: str = MODEL):
        self.entry = entry
        self.model = model
        self.notes: list[tuple[str, str]] = [("player", description)]
        self.asked: list[str] = []
        self.questions = 0
        self.seconds = 0.0
        self.used_fallback = 0

    @property
    def done_asking(self) -> bool:
        return self.questions >= MAX_QUESTIONS

    def ask(self) -> str:
        fallback = FALLBACK_QUESTIONS.get(self.entry.name.lower(), ["Could you tell me a little more about how a turn goes?",
                                                                     "And how does the game end?"])
        q = self._model(QUESTION_PROMPT, "question")
        if not self._good_question(q):
            q = next((f for f in fallback if f not in self.asked), fallback[-1])
            self.used_fallback += 1
        self.asked.append(q)
        self.notes.append(("buddy", q))
        self.questions += 1
        return q

    def answer(self, text: str):
        self.notes.append(("player", text))

    def summary(self) -> str:
        s = self._model(SUMMARY_PROMPT, "summary")
        if not isinstance(s, str) or not (40 <= len(s.strip()) <= 420) or "?" in s:
            self.used_fallback += 1
            return FALLBACK_SUMMARY.get(self.entry.name.lower(), f"I think I have the rules of {self.entry.name} now.")
        s = s.strip()
        return s if s.lower().startswith("so") else "So, if I've understood: " + s[0].lower() + s[1:]

    # -- the model ---------------------------------------------------------------------------------------------
    def _transcript(self) -> str:
        return "\n".join(("The player explained: " if i == 0 else "Penny asked: " if who == "buddy" else "The player answered: ")
                         + f'"{text}"' for i, (who, text) in enumerate(self.notes))

    def _model(self, prompt: str, key: str):
        body = {"model": self.model, "stream": False, "think": False, "keep_alive": "30m",
                "options": {"temperature": 0.0, "num_predict": 120, "num_ctx": NUM_CTX},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": self._transcript() + "\n\n" + prompt}]}
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(f"{C.OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                content = json.loads(r.read())["message"]["content"]
            i, j = content.find("{"), content.rfind("}")
            value = json.loads(content[i:j + 1]).get(key) if i >= 0 else None
        except Exception:
            value = None
        self.seconds += time.perf_counter() - t0
        return value

    def _good_question(self, q) -> bool:
        if not isinstance(q, str):
            return False
        q = q.strip()
        if not (12 <= len(q) <= 160) or not q.endswith("?") or len(q.split()) > 26:
            return False
        for earlier in self.asked:
            if difflib.SequenceMatcher(None, earlier.lower(), q.lower()).ratio() > 0.75:
                return False
        return True
