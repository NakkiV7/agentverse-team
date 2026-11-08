"""Resident preferences, shared by the conversation, game library and video caption."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from buddy import config as C

RESIDENTS_DIR = C.ROOT / "data" / "residents"
LEVELS = {"gentle": "easy", "fair": "medium", "best": "hard"}
VOICE_SPEEDS = {"slow": 1.1, "normal": 0.95, "brisk": 0.85}


@dataclass(frozen=True)
class Resident:
    name: str
    age: int
    way_to_play: str
    level: str
    voice_speed: str
    favourite_game: str
    taught_games: list[str]
    captions: bool = False
    short_sessions: bool = False

    def __post_init__(self):
        if self.way_to_play not in ("write_for_me", "i_write", "mix"):
            raise ValueError(f"Unknown way to play: {self.way_to_play}")
        if self.level not in LEVELS or self.voice_speed not in VOICE_SPEEDS:
            raise ValueError("Unknown resident level or voice speed")

    @property
    def difficulty(self) -> str:
        return LEVELS[self.level]

    @property
    def length_scale(self) -> float:
        return VOICE_SPEEDS[self.voice_speed]

    def public_state(self) -> dict:
        return {"name": self.name, "way_to_play": self.way_to_play,
                "level": self.level, "captions": self.captions}

    def greeting(self) -> str:
        game = self.favourite_game[0].upper() + self.favourite_game[1:]
        if self.way_to_play == "write_for_me":
            offer = f"Shall I write for you again? {game}, as usual?"
        elif self.way_to_play == "i_write":
            pace = {"gentle": "gently", "fair": "at a fair level", "best": "at your best"}[self.level]
            offer = f"{game}, {pace}?"
        else:
            offer = f"Shall we share the writing? {game}, as usual?"
        return f"Hello {self.name}. {offer}"

    def facts(self) -> str:
        return (f"Resident: {self.name}, age {self.age}; way_to_play={self.way_to_play}; "
                f"preferred level={self.level}; voice_speed={self.voice_speed}; "
                f"favourite_game={self.favourite_game}; taught_games={', '.join(self.taught_games) or 'none'}; "
                f"captions={str(self.captions).lower()}; short_sessions={str(self.short_sessions).lower()}.")


def residents(directory: Path | None = None) -> list[Resident]:
    """Read the committed profiles; names never become filesystem paths."""
    return [Resident(**json.loads(path.read_text()))
            for path in sorted((directory or RESIDENTS_DIR).glob("*.json"))]


def load_resident(name: str, directory: Path | None = None) -> Resident | None:
    return next((r for r in residents(directory) if r.name.casefold() == name.strip().casefold()), None)


def context_caption(brain) -> str:
    """One line describing what is happening now, with the actual playing level."""
    state = brain.state
    if state == "IDLE":
        return "Penny · resting"
    resident = getattr(brain, "resident", None)
    name = resident.name if resident else getattr(brain, "player_name", None)
    if state in ("TEACH", "TEACH_MODEL", "NAME_GAME"):
        return "Penny · learning a new game" + (f" from {name}" if name else "")
    entry = getattr(brain, "entry", None)
    if state in ("CHAT", "CHOOSE_GAME"):
        activity = "choosing a game"
    else:
        action = "finished" if state == "GAME_OVER" else "learning" if state == "OFFER_TAUGHT" else "playing"
        activity = f"{action} {brain.game_name}"
        if entry and entry.taught_by:
            activity += f", taught by {entry.taught_by}"
    level = {v: k for k, v in LEVELS.items()}.get(brain.difficulty.level, brain.difficulty.level)
    return f"Penny · {activity}" + (f" · with {name}" if name else "") + f" · level {level}"
