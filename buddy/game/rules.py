"""Teachable grid games: a small family of rules that covers noughts and crosses, Footsteps and their
variants, chosen so that everything else in Buddy stays the same (a 3x3 or 4x4 grid, X and O marks,
the same drawing and the same mark reader).

A game is a GameSpec. The player teaches it in conversation (buddy/agent/teach.py fills the spec from
what they say, asking when something is missing); buddy/game/solver.py then plays it perfectly from the
very first move. Cells are numbered row by row from the player's top-left, as everywhere else in Buddy.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal

Placement = Literal["anywhere", "adjacent_to_last_mark", "adjacent_to_last_mark_or_diagonal"]
End = Literal["k_in_a_row_wins", "k_in_a_row_loses", "no_legal_move_loses"]

PLACEMENTS = ("anywhere", "adjacent_to_last_mark", "adjacent_to_last_mark_or_diagonal")
ENDS = ("k_in_a_row_wins", "k_in_a_row_loses", "no_legal_move_loses")


@dataclass(frozen=True)
class GameSpec:
    rows: int = 3
    cols: int = 3
    placement: Placement = "anywhere"
    end: End = "k_in_a_row_wins"
    k: int = 3                      # line length for the k_in_a_row rules; ignored otherwise
    first_player: Literal["player", "robot"] = "player"
    name: str = ""

    def __post_init__(self):
        if not (2 <= self.rows <= 4 and 2 <= self.cols <= 4):
            raise ValueError("Buddy draws 3x3 or 4x4 boards")
        if self.placement not in PLACEMENTS or self.end not in ENDS:
            raise ValueError("unknown rule")
        if self.end != "no_legal_move_loses" and not (2 <= self.k <= max(self.rows, self.cols)):
            raise ValueError("k must fit on the board")

    @property
    def n(self) -> int:
        return self.rows * self.cols

    def with_(self, **kw) -> "GameSpec":
        return replace(self, **kw)

    def describe(self) -> str:
        """The rules in the plain words Buddy says back to the player."""
        size = f"{self.rows} by {self.cols}" if (self.rows, self.cols) != (3, 3) else "three by three"
        where = {
            "anywhere": "You can put your mark in any empty square.",
            "adjacent_to_last_mark": "Each mark must go next to the last mark played: up, down, left or right.",
            "adjacent_to_last_mark_or_diagonal": "Each mark must touch the last mark played, diagonals included.",
        }[self.placement]
        how = {
            "k_in_a_row_wins": f"{_num(self.k).capitalize()} in a row wins.",
            "k_in_a_row_loses": f"If you make {_num(self.k)} in a row, you lose.",
            "no_legal_move_loses": "If you can't make a move, you lose.",
        }[self.end]
        first = "You go first." if self.first_player == "player" else "I go first."
        return f"A {size} grid, we take turns writing X and O. {where} {how} {first}"


def _num(k: int) -> str:
    return {2: "two", 3: "three", 4: "four"}.get(k, str(k))


# --- known games, for tests and for the demo page ---------------------------------------------------
TIC_TAC_TOE = GameSpec(name="noughts and crosses")
MISERE = GameSpec(end="k_in_a_row_loses", name="misère noughts and crosses")
FOOTSTEPS = GameSpec(placement="adjacent_to_last_mark", end="no_legal_move_loses", name="Footsteps")
FOOTSTEPS_DIAG = GameSpec(placement="adjacent_to_last_mark_or_diagonal", end="no_legal_move_loses", name="Footsteps with diagonals")


# --- state and rules --------------------------------------------------------------------------------
@dataclass(frozen=True)
class State:
    board: tuple[str, ...]           # "" / "X" / "O", row by row
    last: int | None                 # cell of the most recent mark
    to_move: str                     # "X" or "O"

    @staticmethod
    def start(spec: GameSpec, first: str = "X") -> "State":
        return State(("",) * spec.n, None, first)


def other(mark: str) -> str:
    return "O" if mark == "X" else "X"


@lru_cache(maxsize=None)
def neighbours(spec: GameSpec, cell: int, diagonals: bool) -> tuple[int, ...]:
    r, c = divmod(cell, spec.cols)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) == (0, 0) or (not diagonals and dr and dc):
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < spec.rows and 0 <= cc < spec.cols:
                out.append(rr * spec.cols + cc)
    return tuple(out)


def legal_moves(spec: GameSpec, s: State) -> list[int]:
    empty = [i for i, v in enumerate(s.board) if not v]
    if spec.placement == "anywhere" or s.last is None:
        return empty
    allowed = set(neighbours(spec, s.last, spec.placement == "adjacent_to_last_mark_or_diagonal"))
    return [i for i in empty if i in allowed]


@lru_cache(maxsize=None)
def lines(spec: GameSpec) -> tuple[tuple[int, ...], ...]:
    """Every straight line of k cells on the board."""
    k, out = spec.k, []
    for r in range(spec.rows):
        for c in range(spec.cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + i * dr, c + i * dc) for i in range(k)]
                if all(0 <= rr < spec.rows and 0 <= cc < spec.cols for rr, cc in cells):
                    out.append(tuple(rr * spec.cols + cc for rr, cc in cells))
    return tuple(out)


@lru_cache(maxsize=None)
def lines_through(spec: GameSpec, cell: int) -> tuple[tuple[int, ...], ...]:
    return tuple(ln for ln in lines(spec) if cell in ln)


def line_of(spec: GameSpec, board: tuple[str, ...]) -> tuple[str, tuple[int, ...]] | None:
    """(mark, cells) of a completed line, if any."""
    for ln in lines(spec):
        m = board[ln[0]]
        if m and all(board[i] == m for i in ln):
            return m, ln
    return None


def result(spec: GameSpec, s: State) -> tuple[str, str | None, tuple[int, ...] | None] | None:
    """None while the game goes on; else ("win", mark, line_or_None) or ("draw", None, None)."""
    if spec.end in ("k_in_a_row_wins", "k_in_a_row_loses"):
        if s.last is not None:
            m = s.board[s.last]
            for cells in lines_through(spec, s.last):
                if all(s.board[i] == m for i in cells):
                    return ("win", m if spec.end == "k_in_a_row_wins" else other(m), cells)
        if not legal_moves(spec, s):
            return ("draw", None, None)
        return None
    # no_legal_move_loses
    if not legal_moves(spec, s):
        return ("win", other(s.to_move), None)
    return None


def play(spec: GameSpec, s: State, cell: int) -> State:
    if cell not in legal_moves(spec, s):
        raise ValueError(f"illegal move {cell}")
    b = list(s.board)
    b[cell] = s.to_move
    return State(tuple(b), cell, other(s.to_move))


def cell_name(spec: GameSpec, cell: int) -> str:
    """'top left', 'centre', 'row two, column three'... as the player sees the board."""
    r, c = divmod(cell, spec.cols)
    if (spec.rows, spec.cols) == (3, 3):
        return ["top left", "top middle", "top right", "middle left", "centre", "middle right",
                "bottom left", "bottom middle", "bottom right"][cell]
    rows = ["top", "second", "third", "bottom"][: spec.rows]
    cols = ["left", "second", "third", "right"][: spec.cols]
    if spec.rows == 4:
        rows[-1] = "bottom"
    if spec.cols == 4:
        cols[-1] = "right"
    return f"{rows[r]} row, {cols[c]} column"
