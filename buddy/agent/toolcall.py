"""The player asks for anything in their own words; Gemma 4 picks which of Buddy's tools do it.

Ollama's native tool calling (POST /api/chat with "tools") is used with the model in config.TEACH_MODEL (Gemma 4).
The model only picks tool names and arguments from buddy/agent/tools.py; the brain (brain.run_tools) checks every
one against the state, the rules and the reach limits and carries it out. The model never chooses a game move.

plan(text, situation, state, facts) returns a list of ToolCall, or None when the model could not be asked (server
down, timeout, bad answer) so the caller falls back to the Gemma 3 intent path in llm.py. A plain-text answer with no
tool call becomes say(text). One running conversation is kept, like llm.Conversation, so Ollama only has to process
the newest message; the oldest exchanges are dropped when it gets long, and it restarts after a failure.

The facts (brain.facts(): the board, the letters tried, the score, the history, the last turns of conversation) ride
in the newest user message only, under "Facts:", so the system prompt and the worked examples stay a stable,
cached prefix. Questions are answered with say from those facts; the model is told never to invent one. Gemma 4's
context here is 4096 tokens and the prefix is about 2560, so the stored history keeps only the player's words and
the model's calls, never the facts, and at most MAX_TURNS exchanges.

BUDDY_TOOLS=0 in the environment switches the planner off (every call returns None), for comparing the two paths.

Measured on this Mac (M5) with gemma4 through Ollama, 21 requests over the whole session (menu, size, a spoken line,
wait, score, why, redraw, hint, a switch, a letter, stop, level, side, a spoken square, yes, done, sheet, chat):
21 of 21 picked the expected tool, 1.8 s a call on average (0.7 to 1.9 s when the prompt cache is warm).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field

from .. import config as C
from . import tools

MODEL = getattr(C, "TEACH_MODEL", "gemma4")
ENABLED = os.environ.get("BUDDY_TOOLS", "1") != "0"
TIMEOUT_S = 20.0
NUM_PREDICT = 160             # a tool call or three short sentences; the 4096-token context is shared with the prompt
NUM_CTX = 8192                # Ollama's default context is 4096 and the facts block took the prompt to 3660 tokens before the
                              # custom_request tool joined the list; a per-request window twice that leaves room (measured: same speed)
MAX_TURNS = 3                 # exchanges kept in the running conversation (the facts carry the last eight turns anyway):
                              # measured in the simulator, prefix 3126 + facts and situation up to ~600 + history ~70 each
# Ollama caches the prompt prefix, and the tool list is part of it. Offering only the state's tools (tools.schemas)
# changes the prefix at every state change, so the whole prompt is processed again: measured 5.7 s a call on
# average over 21 requests, against 1.8 s with the full list every time, with the same 21 of 21 right. So the model
# sees every tool and brain.run_tools refuses the ones that do not fit the state.
OFFER_ALL_TOOLS = True

SYSTEM = """You choose what Buddy does next. Buddy is a friendly robot arm that plays pen-and-paper games with \
older adults: it draws on a sheet of paper and talks. The player just said something; the situation says what is \
going on, which game is being played and its rules, the score, which squares are free and what Buddy said last.

Call the tool or tools that do what the player asked, in the order they asked for them. Most requests need one tool.
- A plain yes, no, "go on", "please do", "not that one" is always the answer tool. Never turn a yes into a side, \
a square or a letter.
- A game named by the player is switch_game, even in the middle of another game. Wanting to play without naming \
a game, or asking what there is to play, is list_games.
- Never invent a move: draw_my_mark, draw_my_line and say_letter only carry a square, line or letter the player \
themselves named, passed as they said it. If they did not name one, do not call them.
- A board size is new_board. A level, or playing easier or harder, is set_difficulty. Asking Buddy to hold on is \
wait. Asking Buddy to say it again is repeat. Saying their turn is done is i_am_done. Stopping or goodbye is stop.
- Use say for small talk, thanks and any QUESTION about the game, the session or the paper: one or two short, \
warm sentences of plain British English, no emoji.

Questions: the message carries Facts, everything true right now (board, word so far, letters tried right and \
wrong, misses left, score, level, rating, the moves of this game, games played, the last turns of conversation, what \
is written on the paper). Answer with say from the Facts only, quoting letters, squares and numbers exactly, up to \
three sentences. Never invent a fact; if the Facts do not say, say "I don't know" kindly. The level, strength, \
rating, record, earlier games, who has which marks, names and what was said are Facts: answer them with say. Questions about what is \
written on the paper are answered with say, never a drawing tool. "What did I say" or "what did I ask" is a \
question about the conversation in the Facts: answer it with say, never by repeating Buddy's own words. Use the player's name now and then if the Facts give it. "What's the score", "why did \
you play that", "give me a hint" and "what are the rules" keep their own tools."""

KNOWN = "Games Buddy knows: noughts and crosses, hangman and dots and boxes."


def _seed(situation: str, text: str, *calls: tuple[str, dict], facts: str = "") -> list[dict]:
    """One worked example as real conversation turns, so the model answers with native tool calls."""
    body = f"Situation: {situation} {KNOWN}\n" + (f"Facts:\n{facts}\n" if facts else "") + f"Player said: \"{text}\""
    msgs = [{"role": "user", "content": body},
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls]}]
    msgs += [{"role": "tool", "tool_name": n, "content": "done"} for n, _ in calls]
    return msgs


SEEDS = (
    _seed("Chatting. Buddy would like to offer a game.", "Buddy, let's play a game", ("list_games", {}))
    + _seed("Buddy asked which game they would like to play.", "yes please", ("answer", {"yes": True}))
    + _seed("Buddy asked which game they would like to play.", "let's do dots and boxes, a small one, and go easy on me",
            ("switch_game", {"name": "dots and boxes"}), ("new_board", {"size": "small"}), ("set_difficulty", {"level": "easy"}))
    + _seed("A game of noughts and crosses is on. Buddy just asked: I can't quite read the mark in the centre. Is that your cross?",
            "yes", ("answer", {"yes": True}))
    + _seed("Dots and boxes: it is the player's turn. Buddy last said: Your turn.", "how are you keeping today",
            ("say", {"text": "Very well, thank you. It's lovely to have a game with you."}))
    + _seed("The player is guessing a hangman word.", "which letters have I tried so far?",
            ("say", {"text": "You've tried four letters so far: P, T, E and S. P and E were right; T and S were wrong."}),
            facts="Game: hangman, on.\nWord so far: _ P P _ E (5 letters). Letters tried so far: P, T, E, S (4 letters). "
                  "Right letters: E, P. Wrong letters: S, T.\nMisses: 2 of 6 used, 4 misses left.")
    + _seed("A game of noughts and crosses is on; it is the player's turn.", "how hard are you playing?",
            ("say", {"text": "I'm on easy just now, so I play my best move about a third of the time. You've won one of our two games and I've won the other."}),
            facts="Game: noughts and crosses, on, game 3 of this session.\nLevel: easy; Buddy plays its best move 30% of the time. "
                  "Record at noughts and crosses: 2 played, the player won 1, Buddy won 1, 0 drawn.")
)



@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)

    def __str__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.name}({inner})"


def _post(body: dict, timeout: float) -> dict:
    req = urllib.request.Request(f"{C.OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else {}
        except ValueError:
            return {}
    return {}


CALL_TEXT = re.compile(r"^\s*([a-z_]+)\s*[\(\{](.*?)[\)\}]\s*(?:<tool_call\|>)?\s*$", re.M)
KWARG = re.compile(r"([a-z_]+)\s*[=:]\s*(?:<\|\"\|>(.*?)<\|\"\|>|\"((?:[^\"\\\\]|\\\\.)*)\"|'((?:[^'\\\\]|\\\\.)*)'|(true|false|[^,]+))")


def _kwargs(text: str) -> dict:
    out = {}
    for m in KWARG.finditer(text):
        v = next(g for g in m.groups()[1:] if g is not None).strip()
        if v in ("true", "false"):
            v = v == "true"
        out[m.group(1)] = v
    return out


def _named_in_text(content: str) -> ToolCall | None:
    """Content that is a tool name, optionally followed by one argument in words: the argument goes to the tool's
    first parameter. Seen from Gemma 4 now and then instead of a native call."""
    m = re.match(r"^\s*[`'\"]?([a-z_]+)[`'\"]?\s*[:=\-]?\s*(.*?)\s*[.`'\"]*\s*$", content, re.S)
    if not m or m.group(1) not in tools.BY_NAME:
        return None
    tool, arg = tools.BY_NAME[m.group(1)], m.group(2).strip().strip("\"'`()")
    if not tool.params:
        return ToolCall(tool.name) if not arg else None  # "wait" alone, not "wait, I need to think about that"
    if not arg:
        return ToolCall(tool.name)
    if "\n" in arg or len(arg) > 80 or len(arg.split()) > 6:
        return None                                     # a sentence, not an argument
    key = (tool.required or tuple(tool.params))[0]
    kind = tool.params[key].get("type")
    value: object = arg
    if kind == "boolean":
        value = arg.lower() in ("true", "yes", "1")
    return ToolCall(tool.name, {key: value})


def parse_message(message: dict, state: str) -> list[ToolCall] | None:
    """Tool calls from an Ollama chat message. Unknown tool names are dropped; plain content becomes say(text);
    nothing usable -> None."""
    calls = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = str(fn.get("name") or "").strip()
        if name in tools.BY_NAME:
            calls.append(ToolCall(name, _args(fn.get("arguments"))))
    content = str(message.get("content") or "").strip()
    if not calls and content:
        for m in CALL_TEXT.finditer(content):           # "switch_game(name=\"hangman\")" written as text
            name = m.group(1)
            if name in tools.BY_NAME:
                calls.append(ToolCall(name, _kwargs(m.group(2))))
        if calls:
            return calls
        named = _named_in_text(content)                         # "stop", "switch_game hangman", "say_letter: E"
        if named is not None:
            return [named]
        # some answers come back as JSON text instead of a tool call
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                name = d.get("name") or (d.get("function") or {}).get("name")
                if name in tools.BY_NAME:
                    calls.append(ToolCall(name, _args(d.get("arguments") or d.get("parameters") or
                                                      (d.get("function") or {}).get("arguments"))))
            except ValueError:
                pass
        if not calls:
            calls.append(ToolCall("say", {"text": content.strip('"')}))
    return calls or None


class Planner:
    """One running conversation with the model."""

    def __init__(self, model: str = MODEL):
        self.model = model
        self.seconds = 0.0            # how long the last plan() took
        self.facts = ""               # brain.facts() for the next plan() when it is not passed as an argument
        self.tokens = 0               # prompt tokens of the last call (the context is 4096)
        self.reset()

    def reset(self):
        self.msgs = [{"role": "system", "content": SYSTEM}] + list(SEEDS)

    def _trim(self):
        """Drop the oldest exchanges past MAX_TURNS, keeping the system prompt and the worked examples."""
        base = 1 + len(SEEDS)
        while True:
            starts = [i for i in range(base, len(self.msgs)) if self.msgs[i]["role"] == "user"]
            if len(starts) <= MAX_TURNS:
                return
            end = starts[1] if len(starts) > 1 else len(self.msgs)
            del self.msgs[starts[0]:end]

    @staticmethod
    def user_message(text: str, situation: str, facts: str = "") -> str:
        return f"Situation: {situation}\n" + (f"Facts:\n{facts.strip()}\n" if facts else "") + f"Player said: \"{text}\""

    def plan(self, text: str, situation: str, state: str, facts: str | None = None) -> list[ToolCall] | None:
        if not ENABLED:
            return None
        if not tools.available(state):          # TEACH and NAME_GAME have their own flows
            return None
        if facts is None:
            facts, self.facts = self.facts, ""
        offered = [t.schema() for t in tools.TOOLS] if OFFER_ALL_TOOLS else tools.schemas(state)
        self._trim()
        msgs = self.msgs + [{"role": "user", "content": self.user_message(text, situation, facts)}]
        body = {"model": self.model, "stream": False, "think": False, "keep_alive": "60m",
                "tools": offered, "options": {"temperature": 0.0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX}, "messages": msgs}
        t0 = time.time()
        try:
            r = _post(body, TIMEOUT_S)
            message = r.get("message") or {}
            self.tokens = int(r.get("prompt_eval_count") or 0)
            calls = parse_message(message, state)
        except Exception:
            self.reset()
            self.seconds = time.time() - t0
            return None
        self.seconds = time.time() - t0
        if calls is None:
            return None
        # remember the exchange: the player's words (not the facts or the situation, which are stale next time and
        # would fill the 4096-token context), the assistant's calls and a short result for each
        self.msgs = msgs[:-1] + [{"role": "user", "content": f"Player said: \"{text}\""}]
        self.msgs.append({"role": "assistant", "content": message.get("content") or "",
                          **({"tool_calls": message["tool_calls"]} if message.get("tool_calls") else {})})
        for tc in message.get("tool_calls") or []:
            self.msgs.append({"role": "tool", "tool_name": (tc.get("function") or {}).get("name", ""), "content": "done"})
        self._trim()
        return calls


_default: Planner | None = None


def plan(text: str, situation: str, state: str, facts: str = "") -> list[ToolCall] | None:
    """Module-level convenience with one shared conversation."""
    global _default
    if _default is None:
        _default = Planner()
    return _default.plan(text, situation, state, facts)


def last_seconds() -> float:
    return _default.seconds if _default is not None else 0.0
