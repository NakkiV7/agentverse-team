"""Local Elo ratings for each player and game, and Penny at each strength.

Results are from the player's side. A draw scores half a point. Strengths are
grouped in steps of 0.05, so repeated games can calibrate the same opponent.
Set BUDDY_PROFILE_FILE to move runtime/profile.json and BUDDY_PLAYER_ID to use
another player's history. There is no player identification from the camera.
"""
from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from buddy import config as C

INITIAL_RATING = 1000.0
STRENGTHS = tuple(i / 20 for i in range(21))
SCORES = {"win": 1.0, "draw": 0.5, "loss": 0.0}
ADAPTIVE_MIN, ADAPTIVE_MAX = 0.10, 0.95    # the range the adaptive policy may choose from


def expected_score(player: float, opponent: float) -> float:
    """Elo expectation: probability of a win plus half the probability of a draw."""
    gap = max(-4000.0, min(4000.0, opponent - player))
    return 1.0 / (1.0 + 10.0 ** (gap / 400.0))


def strength_key(strength: float) -> str:
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be between zero and one")
    return f"{round(strength * 20) / 20:.2f}"


@dataclass
class Record:
    rating: float = INITIAL_RATING
    games: int = 0
    recent: list[str] = field(default_factory=list)

    @classmethod
    def read(cls, data: dict, initial: float = INITIAL_RATING) -> Record:
        try:
            record = cls(**{"rating": initial, **data})
            if (not math.isfinite(record.rating) or not isinstance(record.games, int)
                    or record.games < 0 or not isinstance(record.recent, list)
                    or any(r not in SCORES for r in record.recent)):
                raise ValueError("invalid rating record")
            return record
        except (TypeError, ValueError):
            return cls(rating=initial)

    def update(self, result: str, expectation: float, scale: float = 1.0) -> None:
        # The first five results establish a rating quickly. Penny learns more
        # slowly, keeping the scale steady while the player's skill is measured.
        k = (64.0 if self.games < 5 else 24.0) * scale
        self.rating += k * (SCORES[result] - expectation)
        self.games += 1
        self.recent = (self.recent + [result])[-20:]


class Rating:
    def __init__(self, game: str = "noughts and crosses", player: str | None = None,
                 path: Path | None = None):
        self.game = " ".join(game.casefold().split())
        self.player_id = player if player is not None else os.environ.get("BUDDY_PLAYER_ID", "default")
        self.path = Path(path if path is not None else os.environ.get("BUDDY_PROFILE_FILE", C.RUNTIME / "profile.json"))
        self.load()

    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
            if (not isinstance(data, dict) or not isinstance(data.get("players"), dict)
                    or not isinstance(data.get("buddy"), dict)):
                raise ValueError("invalid profile")
        except (OSError, ValueError):
            data = {"version": 1, "players": {}, "buddy": {}}
        self.data = data
        player_games = data["players"].setdefault(self.player_id, {})
        buddy_game = data["buddy"].setdefault(self.game, {})
        self.player = Record.read(player_games.get(self.game, {}))
        # Priors express ordering, not measured human skill. Results calibrate
        # each game separately, including games taught by the player.
        self.buddy = {strength_key(s): Record.read(buddy_game.get(strength_key(s), {}), 600.0 + 800.0 * s)
                      for s in STRENGTHS}

    def expected(self, buddy_strength: float) -> float:
        return expected_score(self.player.rating, self.buddy[strength_key(buddy_strength)].rating)

    def update(self, result: str, buddy_strength: float) -> float:
        """Record one completed game, persist both ratings, return the player's rating."""
        if result not in SCORES:
            raise ValueError(f"unknown result: {result}")
        key = strength_key(buddy_strength)
        # Other game/player objects may have saved since this object was opened.
        self.load()
        opponent = self.buddy[key]
        expectation = expected_score(self.player.rating, opponent.rating)
        self.player.update(result, expectation)
        opponent.update({"win": "loss", "loss": "win", "draw": "draw"}[result], 1 - expectation, scale=0.25)
        self.save()
        return self.player.rating

    def save(self) -> None:
        self.data["players"][self.player_id][self.game] = asdict(self.player)
        self.data["buddy"][self.game] = {s: asdict(r) for s, r in self.buddy.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Replace atomically so an interrupted write leaves the last profile intact.
        with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as f:
            temp = Path(f.name)
            try:
                json.dump(self.data, f, indent=2, allow_nan=False)
                f.write("\n")
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        try:
            temp.replace(self.path)
        finally:
            temp.unlink(missing_ok=True)


@dataclass(frozen=True)
class Selection:
    strength: float
    target_strength: float
    expected_score: float
    reason: str


def select_strength(rating: Rating, rng=random) -> Selection:
    """Aim for 45% player wins, with small, deliberate variation each game.

    Elo predicts score, so add half the recent draw rate to the win target.
    Strength stays between ADAPTIVE_MIN and ADAPTIVE_MAX: never worse than the
    "easy" level, never perfect play. A target is an aim, not a promised outcome:
    forced results and the solvers' safety rules can make it unreachable.
    """
    recent = rating.player.recent
    draw_rate = recent.count("draw") / len(recent) if recent else 0.0
    # Elo scores a draw as half a point, so in a draw-heavy game (noughts and crosses) the target must rise with the
    # draw rate or the player never actually wins. The floor keeps Penny from playing almost at random after a run
    # of draws (it reached 5% strength before the floor).
    target_score = min(0.95, 0.45 + 0.5 * draw_rate)
    strengths = [v for v in STRENGTHS if ADAPTIVE_MIN <= v <= ADAPTIVE_MAX]
    target = min(strengths, key=lambda s: abs(rating.expected(s) - target_score))
    nearby = [s for s in strengths if abs(s - target) <= 0.100001]
    strength = rng.choice(nearby)
    if recent[-3:] == ["win"] * 3:
        reason = "You have won the last three games, so I will try a bit harder."
    elif recent[-3:] == ["loss"] * 3:
        reason = "I have won the last three games, so I will make things a little gentler."
    elif recent and recent[-1] == "win":
        reason = "You won, so I am adjusting the challenge and mixing up my play."
    elif recent and recent[-1] == "loss":
        reason = "I won, so I am easing the challenge and mixing up my play."
    elif recent:
        reason = "We drew, so I am allowing for draws and mixing up my play."
    else:
        reason = "I will start with a varied challenge while I learn how you play."
    return Selection(strength, target, rating.expected(strength), reason)
