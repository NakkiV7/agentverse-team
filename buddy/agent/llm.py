"""Gemma 4 via Ollama, local, JSON-schema constrained. The model does language only:
speech -> Intent, situation -> a short spoken line. It never picks moves or arm motions."""
from __future__ import annotations

import base64
import json
import time

import requests

from buddy import config
from buddy.contracts import Intent

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["play", "choose_game", "move", "explain_rules", "repeat", "yes", "no", "stop", "chat"]},
        "cell": {"type": ["integer", "null"], "minimum": 0, "maximum": 8},
        "text": {"type": "string"},
    },
    "required": ["kind", "cell", "text"],
}

SYSTEM_INTENT = (
    "You are the ear of Buddy, a robot arm that plays tic-tac-toe on paper with an older adult. "
    "Classify what the player just said into one intent. Cells are numbered 0 to 8, row by row, "
    "top-left first: 'top left' is 0, 'middle' or 'centre' is 4, 'bottom right' is 8. "
    "If they name a cell, kind is 'move'. 'play', 'let's play', 'a game' -> 'play'. "
    "'again', 'say that again', 'pardon' -> 'repeat'. 'stop', 'enough', 'bye' -> 'stop'. "
    "Anything else is 'chat' with a short, warm reply in 'text' (one sentence, plain words)."
)

SYSTEM_SPEAK = (
    "You are the voice of Buddy, a kind robot that plays tic-tac-toe on paper with an older adult. "
    "Write exactly what Buddy says next: one or two short sentences, plain words, no lists, "
    "no emoji, unhurried, never patronising. Use the facts given; invent nothing."
)


class LLM:
    def __init__(self, model: str = config.LLM_MODEL, url: str = config.OLLAMA_URL, keep_alive: str = "30m"):
        self.model, self.url, self.keep_alive = model, url, keep_alive
        self.grammar_ok = True        # flips to False if the server cannot do constrained decoding

    def _chat(self, messages, fmt=None, images=None, num_predict=120) -> tuple[str, float, float]:
        """One chat call. `fmt` is a JSON schema. Ollama builds without the grammar library (the Homebrew
        build on Apple silicon, Nov 2026) answer 501 to `format`; then we ask for JSON in the prompt
        instead and validate it ourselves (see _parse_json)."""
        if images:
            messages[-1]["images"] = [base64.b64encode(b).decode() for b in images]
        body = {"model": self.model, "messages": messages, "stream": False, "keep_alive": self.keep_alive,
                "think": False,            # Gemma 4 thinks by default; that costs seconds and empties `content`
                "options": {"temperature": 0.2, "num_predict": num_predict}}
        if fmt and self.grammar_ok:
            body["format"] = fmt
        elif fmt:
            messages[0] = {"role": "system", "content": messages[0]["content"] + "\nAnswer with one JSON object only, "
                           "no prose, no code fence, matching this JSON schema: " + json.dumps(fmt)}
        t0 = time.perf_counter()
        r = requests.post(f"{self.url}/api/chat", json=body, timeout=60)
        if r.status_code == 501 and fmt and self.grammar_ok:
            self.grammar_ok = False
            return self._chat(messages, fmt, None, num_predict)
        r.raise_for_status()
        j = r.json()
        total = time.perf_counter() - t0
        first = (j.get("prompt_eval_duration", 0) + j.get("load_duration", 0)) / 1e9
        return j["message"]["content"], first, total

    @staticmethod
    def _parse_json(txt: str, schema: dict) -> dict:
        """Pull the first JSON object out of the text and check enums / types against the schema."""
        txt = txt.strip()
        if txt.startswith("```"):
            txt = txt.strip("`").removeprefix("json").strip()
        start, end = txt.find("{"), txt.rfind("}")
        j = json.loads(txt[start:end + 1])
        for k, spec in schema["properties"].items():
            if "enum" in spec and j.get(k) not in spec["enum"]:
                raise ValueError(f"{k}={j.get(k)!r} not in {spec['enum']}")
        return j

    def intent(self, utterance: str) -> tuple[Intent, dict]:
        txt, first, total = self._chat(
            [{"role": "system", "content": SYSTEM_INTENT}, {"role": "user", "content": utterance}], fmt=INTENT_SCHEMA)
        j = self._parse_json(txt, INTENT_SCHEMA)
        cell = j.get("cell")
        cell = int(cell) if cell not in (None, "", "null") and str(cell).lstrip("-").isdigit() and 0 <= int(cell) <= 8 else None
        return Intent(kind=j["kind"], cell=cell, text=str(j.get("text") or "")), {"first_s": first, "total_s": total}

    def speak(self, situation: str) -> tuple[str, dict]:
        txt, first, total = self._chat(
            [{"role": "system", "content": SYSTEM_SPEAK}, {"role": "user", "content": situation}], num_predict=60)
        return txt.strip(), {"first_s": first, "total_s": total}

    def read_mark(self, cell_png: bytes) -> tuple[str, dict]:
        """Backup for low-confidence cells: ask the vision model what is drawn. Returns X | O | empty | unclear."""
        schema = {"type": "object", "properties": {"mark": {"type": "string", "enum": ["X", "O", "empty", "unclear"]}}, "required": ["mark"]}
        txt, first, total = self._chat(
            [{"role": "user", "content": "This is one cell of a hand-drawn tic-tac-toe grid. What is in it?"}],
            fmt=schema, images=[cell_png], num_predict=20)
        return self._parse_json(txt, schema)["mark"], {"first_s": first, "total_s": total}

    def warm(self) -> None:
        self._chat([{"role": "user", "content": "hi"}], num_predict=1)
