"""Noughts and crosses: rules, a perfect solver, and adaptive difficulty.

Board: 9 cells, row by row from the player's top-left, each "" / "X" / "O". The board the game uses is always the
one perception read from the paper, never an internal copy.

Solver: full minimax over the game tree (memoised), scoring every legal move: +10 - depth for a win, 0 for a draw,
-10 + depth for a loss, so it prefers quick wins and slow losses.

Adaptive difficulty: a persistent player rating chooses a varied strength each game. Buddy plays the best move
with that probability; otherwise it plays a friendly move that does not let the player win on the very next turn.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache

from buddy.game.rating import Rating, select_strength

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
NAMES = ["top left", "top middle", "top right", "middle left", "centre", "middle right",
         "bottom left", "bottom middle", "bottom right"]


def winner(board) -> tuple[str, tuple] | None:
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a], (a, b, c)
    return None


def full(board) -> bool:
    return all(board)


def other(sym: str) -> str:
    return "O" if sym == "X" else "X"


@lru_cache(maxsize=None)
def _minimax(board: tuple, to_move: str, me: str, depth: int) -> int:
    w = winner(board)
    if w:
        return (10 - depth) if w[0] == me else (depth - 10)
    if full(board):
        return 0
    scores = []
    for i in range(9):
        if not board[i]:
            nb = board[:i] + (to_move,) + board[i + 1:]
            scores.append(_minimax(nb, other(to_move), me, depth + 1))
    return max(scores) if to_move == me else min(scores)


def move_scores(board, me: str) -> dict[int, int]:
    b = tuple(board)
    return {i: _minimax(b[:i] + (me,) + b[i + 1:], other(me), me, 1) for i in range(9) if not b[i]}


def _gives_immediate_loss(board, i, me) -> bool:
    nb = list(board)
    nb[i] = me
    opp = other(me)
    return any(winner(nb[:j] + [opp] + nb[j + 1:]) for j in range(9) if not nb[j])


@dataclass
class Choice:
    cell: int
    best: list[int]
    scores: dict
    played_best: bool
    reason: str


LEVELS = {"easy": 0.3, "medium": 0.6, "hard": 0.85, "impossible": 1.0}
LEVEL_WORDS = {"easy": "easy, I'll make plenty of mistakes", "medium": "medium", "hard": "hard",
               "impossible": "impossible: I'll play perfectly, so the best you can hope for is a draw",
               "adaptive": "adapting to you: harder when you win, gentler when you lose"}


@dataclass
class Difficulty:
    """How well Buddy plays. `level` is a fixed setting (easy / medium / hard / impossible) or "adaptive",
    where a per-player, per-game rating sets the challenge. Explicit strengths still work for solver callers;
    start_game selects adaptive strength when a real game begins."""
    strength: float = 0.75
    level: str = "adaptive"
    history: list = field(default_factory=list)
    rating: Rating = field(default_factory=Rating, repr=False)
    rng: random.Random = field(default_factory=random.Random, repr=False)
    reason: str = field(default="", init=False)
    _prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.level in LEVELS:
            self.strength = LEVELS[self.level]
        elif self.level != "adaptive":
            raise ValueError(self.level)

    def select_game(self, game: str) -> None:
        if " ".join(game.casefold().split()) != self.rating.game:
            self.rating = Rating(game, self.rating.player_id, self.rating.path)
            self.history = []
            self.reason = ""
            self._prepared = False

    def _adapt(self) -> None:
        selection = select_strength(self.rating, self.rng)
        self.strength, self.reason = selection.strength, selection.reason
        self._prepared = True

    def start_game(self, game: str | None = None, rng=None) -> str:
        if rng is not None:
            self.rng = rng
        if game is not None:
            self.select_game(game)
        if self.level == "adaptive":
            if not self._prepared:
                self._adapt()
            self._prepared = False
            return self.reason
        return f"level {self.level}, strength stays {self.strength:.0%}"

    def set_level(self, level: str) -> str:
        if level not in LEVELS and level != "adaptive":
            raise ValueError(level)
        self.level = level
        self._prepared = False
        if level in LEVELS:
            self.strength = LEVELS[level]
            self.reason = f"I will keep playing at {level} until you ask me to change."
        else:
            self._adapt()
        return LEVEL_WORDS[level]

    def update(self, result: str) -> str:
        """result from the PLAYER's side: 'win' / 'loss' / 'draw'."""
        old = self.strength
        player_rating = self.rating.update(result, old)
        self.history.append(result)
        if self.level != "adaptive":
            return (f"player {result}: rating {player_rating:.0f}; level {self.level}, "
                    f"strength stays {100 * self.strength:.0f}%")
        self._adapt()
        return (f"player {result}: rating {player_rating:.0f}; strength {100 * old:.0f}% -> "
                f"{100 * self.strength:.0f}%. {self.reason}")


def choose(board, me: str, diff: Difficulty, rng=random) -> Choice:
    scores = move_scores(board, me)
    top = max(scores.values())
    best = [i for i, s in scores.items() if s == top]
    if rng.random() < diff.strength or len(scores) == 1:
        i = rng.choice(best)
        return Choice(i, best, scores, True, f"best move ({diff.strength:.0%} strength)")
    safe = [i for i in scores if i not in best and not _gives_immediate_loss(board, i, me)]
    if not safe:
        i = rng.choice(best)
        return Choice(i, best, scores, True, "every other move would lose at once, so the best one")
    i = rng.choice(safe)
    return Choice(i, best, scores, False, f"a friendlier move ({1 - diff.strength:.0%} of the time)")
