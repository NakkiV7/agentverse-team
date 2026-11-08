"""The games Penny knows: a few built in, plus every game a player has taught it, remembered on disk
(runtime/games.json, gitignored) so they are offered again next session.

A library entry is a GameSpec with a name and a little history. Noughts and crosses is special: it has
no GameSpec here (spec None) because brain.py plays it with Hakim's original code and win-line drawing.
Hangman also has spec None; its kind selects the separate spoken-letter game.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from buddy import config as C
from buddy.game.rules import FOOTSTEPS, FOOTSTEPS_DIAG, MISERE, GameSpec

GAMES_FILE = Path(os.environ.get("BUDDY_GAMES_FILE", str(C.RUNTIME / "games.json")))


@dataclass
class Entry:
    name: str
    spec: GameSpec | None               # None = a built-in with its own rules, selected by kind.
    aliases: tuple[str, ...] = ()
    builtin: bool = False
    taught_at: float = 0.0
    plays: int = 0
    player_wins: int = 0
    robot_wins: int = 0
    draws: int = 0
    kind: str = "grid"                  # Non-grid built-ins also have spec None.

    def describe(self) -> str:
        if self.kind == "dots":
            return ("I draw a grid of dots. We take turns joining two dots that are next to each other with a line. "
                    "Whoever draws the fourth side of a box wins it, marks it, and goes again. Most boxes wins.")
        if self.kind == "hangman":
            return ("I choose a familiar word. Say or type one letter at a time. I write every matching letter. "
                    "A wrong letter adds one gallows part. Find the word before six misses. You can ask for a hint.")
        if self.spec is None:
            return ("We take turns putting our mark in one of the nine squares. The first to get three in a row, "
                    "across, down or corner to corner, wins.")
        return self.spec.describe()


BUILTIN = [
    Entry("noughts and crosses", None, ("tic tac toe", "tic-tac-toe", "x's and o's", "xs and os", "naughts and crosses"), True),
    Entry("Footsteps", FOOTSTEPS, ("footsteps", "the trail game", "trails"), True),
    Entry("backwards noughts and crosses", MISERE, ("misere", "misère", "reverse noughts and crosses", "backwards tic tac toe", "inverse tic tac toe"), True),
    Entry("Footsteps with diagonals", FOOTSTEPS_DIAG, ("diagonal footsteps", "footsteps diagonals"), True),
    Entry("hangman", None, ("hang man", "the word game"), True, kind="hangman"),
    Entry("dots and boxes", None, ("dots", "boxes", "dots & boxes", "the dots game", "squares"), True, kind="dots"),
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


from .. import config as C


class Library:
    def __init__(self, path: Path = GAMES_FILE, hidden=None):
        self.path = Path(path)
        self.entries: list[Entry] = [Entry(**{**asdict(e), "spec": e.spec}) for e in BUILTIN]
        # Games Penny pretends not to know until taught (config.HIDE_GAMES): left out of the menu, the facts and
        # name matching, so "let's play dots and boxes" leads to "teach me", not to a game. reveal() ends that.
        self.hidden: set[str] = set(_norm(n) for n in (C.HIDE_GAMES if hidden is None else hidden))
        self.load()

    def is_hidden(self, entry: Entry) -> bool:
        return _norm(entry.name) in self.hidden

    def visible(self) -> list[Entry]:
        return [e for e in self.entries if not self.is_hidden(e)]

    def reveal(self, entry: Entry) -> Entry:
        """The player has taught Penny this game: it joins the menu and counts as taught today."""
        self.hidden.discard(_norm(entry.name))
        entry.taught_at = time.strftime("%Y-%m-%d %H:%M")
        return entry

    # -- persistence -------------------------------------------------------------------------------
    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for d in data.get("games", []):
            spec = GameSpec(**d["spec"]) if d.get("spec") else None
            e = Entry(d["name"], spec, tuple(d.get("aliases", ())), False, d.get("taught_at", 0.0),
                      d.get("plays", 0), d.get("player_wins", 0), d.get("robot_wins", 0), d.get("draws", 0))
            e.kind = d.get("kind", "grid")
            self._upsert(e)
        for d in data.get("builtin_history", []):
            e = self.find(d["name"])
            if e is not None and e.builtin:
                e.plays, e.player_wins, e.robot_wins, e.draws = d["plays"], d["player_wins"], d["robot_wins"], d["draws"]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        learned = [{**{k: v for k, v in asdict(e).items() if k not in ("spec", "builtin")},
                    "spec": asdict(e.spec) if e.spec else None} for e in self.entries if not e.builtin]
        hist = [{"name": e.name, "plays": e.plays, "player_wins": e.player_wins, "robot_wins": e.robot_wins, "draws": e.draws}
                for e in self.entries if e.builtin and e.plays]
        self.path.write_text(json.dumps({"games": learned, "builtin_history": hist}, indent=2))

    # -- lookup ------------------------------------------------------------------------------------
    def _upsert(self, e: Entry) -> Entry:
        for i, x in enumerate(self.entries):
            if _norm(x.name) == _norm(e.name):
                self.entries[i] = e
                return e
        self.entries.append(e)
        return e

    def names(self) -> list[str]:
        return [e.name for e in self.visible()]

    def learned(self) -> list[Entry]:
        return [e for e in self.entries if not e.builtin]

    def find(self, text: str, include_hidden: bool = False) -> Entry | None:
        """Match what the player said to a game: exact name or alias, then a contained name, then a close match."""
        t = _norm(text)
        if not t:
            return None
        entries = self.entries if include_hidden else self.visible()
        for e in entries:
            if t == _norm(e.name) or t in (_norm(a) for a in e.aliases):
                return e
        for e in sorted(entries, key=lambda e: -len(e.name)):
            if _norm(e.name) in t or any(_norm(a) in t for a in e.aliases):
                return e
        keys = {_norm(e.name): e for e in entries}
        for e in entries:
            for a in e.aliases:
                keys.setdefault(_norm(a), e)
        close = difflib.get_close_matches(t, list(keys), n=1, cutoff=0.75)
        return keys[close[0]] if close else None

    def same_rules(self, spec: GameSpec) -> Entry | None:
        for e in self.entries:
            if e.spec is not None and e.spec.with_(name="") == spec.with_(name=""):
                return e
        return None

    # -- changes -----------------------------------------------------------------------------------
    def remember(self, spec: GameSpec, name: str = "") -> Entry:
        """Keep a taught game. An existing game with the same rules is returned as is."""
        known = self.same_rules(spec)
        if known is not None:
            return known
        name = name.strip() or f"your game {len(self.learned()) + 1}"
        e = self._upsert(Entry(name, spec.with_(name=name), (), False, time.time()))
        self.save()
        return e

    def record(self, entry: Entry, result: str) -> None:
        """result from the player's side: 'win' | 'loss' | 'draw'."""
        entry.plays += 1
        if result == "win":
            entry.player_wins += 1
        elif result == "loss":
            entry.robot_wins += 1
        else:
            entry.draws += 1
        self.save()

    def spoken_list(self) -> str:
        names = self.names()
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:-1]) + " and " + names[-1]
