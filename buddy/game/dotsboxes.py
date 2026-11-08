"""Dots and boxes: rules and a solver for small boards (2x2 boxes = 3x3 dots, 3x3 boxes = 4x4 dots).

Board: n x n boxes. Edges are numbered: first the horizontal ones row by row from the player's top-left
(h(r, c) = r * n + c for r in 0..n, c in 0..n-1), then the vertical ones (v(r, c) = n * (n + 1) + r * (n + 1) + c
for r in 0..n-1, c in 0..n). Boxes are numbered row by row from the top-left, like the tic-tac-toe cells.

State: the set of drawn edges (a bitmask), who owns each box, and whose turn it is. A player who completes a box
moves again. The state the game uses is always built from the edges the camera read on the paper.

Solver: the value of a position is the net number of boxes the player to move will still gain with best play.
It depends only on which edges are left, so it is memoised on the edge mask alone. 2x2 boxes (12 edges) is searched
completely. 3x3 boxes (24 edges) is searched completely once few enough edges are left; before that a depth-limited
negamax with alpha-beta pruning, a chain-aware heuristic at the cut-off, a node budget and a deadline keeps every
move under about a second.

Difficulty reuses tictactoe.Difficulty: Buddy plays the best edge with probability `strength`, otherwise a
friendlier edge that does not hand the player a box.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from functools import lru_cache

from .tictactoe import Choice, Difficulty

BOX_NAMES = {
    2: ["top left", "top right", "bottom left", "bottom right"],
    3: ["top left", "top middle", "top right", "middle left", "centre", "middle right",
        "bottom left", "bottom middle", "bottom right"],
}


# ------------------------------------------------------------------ geometry
def n_edges(n: int) -> int:
    return 2 * n * (n + 1)


def h_edge(n: int, r: int, c: int) -> int:
    """Horizontal edge above box row r (r = n is the bottom border), in column c."""
    return r * n + c


def v_edge(n: int, r: int, c: int) -> int:
    """Vertical edge left of box column c (c = n is the right border), in row r."""
    return n * (n + 1) + r * (n + 1) + c


@lru_cache(maxsize=None)
def box_edges(n: int) -> tuple:
    """For each box: (top, bottom, left, right) edge indices."""
    out = []
    for r in range(n):
        for c in range(n):
            out.append((h_edge(n, r, c), h_edge(n, r + 1, c), v_edge(n, r, c), v_edge(n, r, c + 1)))
    return tuple(out)


@lru_cache(maxsize=None)
def box_masks(n: int) -> tuple:
    return tuple(sum(1 << e for e in edges) for edges in box_edges(n))


@lru_cache(maxsize=None)
def edge_boxes(n: int) -> tuple:
    """For each edge: the boxes it borders (one on the outside border, two inside)."""
    out = [[] for _ in range(n_edges(n))]
    for b, edges in enumerate(box_edges(n)):
        for e in edges:
            out[e].append(b)
    return tuple(tuple(x) for x in out)


def edge_dots(n: int, e: int) -> tuple:
    """The two dots an edge joins, as (row, col) in the (n+1) x (n+1) dot grid, row 0 at the top."""
    if e < n * (n + 1):
        r, c = divmod(e, n)
        return (r, c), (r, c + 1)
    r, c = divmod(e - n * (n + 1), n + 1)
    return (r, c), (r + 1, c)


def edge_name(n: int, e: int) -> str:
    """Spoken name: 'the top side of the top left box'."""
    names = BOX_NAMES.get(n) or [f"box {i + 1}" for i in range(n * n)]
    if e < n * (n + 1):
        r, c = divmod(e, n)
        if r < n:
            return f"the top side of the {names[r * n + c]} box"
        return f"the bottom side of the {names[(n - 1) * n + c]} box"
    r, c = divmod(e - n * (n + 1), n + 1)
    if c < n:
        return f"the left side of the {names[r * n + c]} box"
    return f"the right side of the {names[r * n + n - 1]} box"


def box_name(n: int, b: int) -> str:
    names = BOX_NAMES.get(n) or [f"box {i + 1}" for i in range(n * n)]
    return names[b]


# ------------------------------------------------------------------ rules
@dataclass(frozen=True)
class State:
    n: int
    edges: int                      # bitmask of drawn edges
    owners: tuple                   # per box: "" or the symbol of the player who completed it
    to_move: str
    players: tuple = ("A", "B")

    @staticmethod
    def start(n: int, players=("A", "B"), first: str | None = None) -> "State":
        return State(n, 0, ("",) * (n * n), first or players[0], tuple(players))

    def other(self, sym: str | None = None) -> str:
        sym = sym or self.to_move
        return self.players[1] if sym == self.players[0] else self.players[0]

    def drawn(self, e: int) -> bool:
        return bool(self.edges >> e & 1)

    def drawn_edges(self) -> list[int]:
        return [e for e in range(n_edges(self.n)) if self.edges >> e & 1]

    def score(self, sym: str) -> int:
        return sum(1 for o in self.owners if o == sym)

    def full(self) -> bool:
        return self.edges == (1 << n_edges(self.n)) - 1


def legal_moves(s: State) -> list[int]:
    return [e for e in range(n_edges(s.n)) if not s.edges >> e & 1]


def completed_by(n: int, edges_before: int, e: int) -> list[int]:
    """Boxes that edge e completes when added to edges_before."""
    after = edges_before | (1 << e)
    masks = box_masks(n)
    return [b for b in edge_boxes(n)[e] if after & masks[b] == masks[b]]


def play(s: State, e: int) -> State:
    if s.edges >> e & 1:
        raise ValueError(f"edge {e} is already drawn")
    done = completed_by(s.n, s.edges, e)
    owners = list(s.owners)
    for b in done:
        owners[b] = s.to_move
    return State(s.n, s.edges | (1 << e), tuple(owners), s.to_move if done else s.other(), s.players)


def result(s: State):
    """None while edges are left; otherwise ('win', symbol) or ('draw', None)."""
    if not s.full():
        return None
    a, b = s.score(s.players[0]), s.score(s.players[1])
    if a == b:
        return ("draw", None)
    return ("win", s.players[0] if a > b else s.players[1])


def third_side_boxes(n: int, mask: int, e: int) -> list[int]:
    """Boxes that would have exactly 3 sides after e is drawn (a gift to the opponent)."""
    after = mask | (1 << e)
    out = []
    for b in edge_boxes(n)[e]:
        if bin(after & box_masks(n)[b]).count("1") == 3:
            out.append(b)
    return out


def is_safe(n: int, mask: int, e: int) -> bool:
    """Drawing e neither completes a box nor leaves one with three sides."""
    return not completed_by(n, mask, e) and not third_side_boxes(n, mask, e)


# ------------------------------------------------------------------ solver
class _OutOfTime(Exception):
    pass


class Solver:
    """Values are net boxes for the player to move, from the given edges to the end of the game."""

    def __init__(self, n: int, node_budget: int = 150_000, time_limit: float = 0.8):
        self.n = n
        self.full_mask = (1 << n_edges(n)) - 1
        self.exact: dict[int, int] = {}
        self.budget = node_budget
        self.time_limit = time_limit
        self.nodes = 0
        self.last_depth = 0            # how the last move_scores was found: 0 = exact, else the search depth
        self.last_seconds = 0.0

    # -- exact search (memo on the edge mask) ------------------------------------------------------
    def value(self, mask: int) -> int:
        if mask == self.full_mask:
            return 0
        v = self.exact.get(mask)
        if v is not None:
            return v
        self.nodes += 1
        if self.nodes > self.budget:
            raise _OutOfTime()
        n, best = self.n, -10 ** 6
        masks, eb = box_masks(n), edge_boxes(n)
        for e in range(n_edges(n)):
            if mask >> e & 1:
                continue
            child = mask | (1 << e)
            gained = sum(1 for b in eb[e] if child & masks[b] == masks[b])
            v = gained + self.value(child) if gained else -self.value(child)
            if v > best:
                best = v
        self.exact[mask] = best
        return best

    # -- bounded search for the big board ----------------------------------------------------------
    def heuristic(self, mask: int) -> int:
        """Net boxes for the mover, estimated: take every box that can be taken; if a safe edge is left the position
        is even; otherwise the mover must open the shortest chain, the opponent takes it and opens the next, and so
        on (chains alternate). Chains are runs of two-sided boxes joined through undrawn edges."""
        n = self.n
        masks, eb = box_masks(n), edge_boxes(n)
        gained, m = 0, mask
        taking = True
        while taking:
            taking = False
            for e in range(n_edges(n)):
                if m >> e & 1:
                    continue
                child = m | (1 << e)
                g = sum(1 for b in eb[e] if child & masks[b] == masks[b])
                if g:
                    gained, m, taking = gained + g, child, True
                    break
        if m == self.full_mask:
            return gained
        if any(is_safe(n, m, e) for e in range(n_edges(n)) if not m >> e & 1):
            return gained
        sizes = sorted(self._chain_sizes(m))
        v = gained
        for i, size in enumerate(sizes):
            v += size if i % 2 else -size
        return v

    def _chain_sizes(self, mask: int) -> list[int]:
        n = self.n
        masks = box_masks(n)
        two = {b for b in range(n * n) if bin(mask & masks[b]).count("1") == 2}
        seen, sizes = set(), []
        for start in two:
            if start in seen:
                continue
            size, stack = 0, [start]
            seen.add(start)
            while stack:
                b = stack.pop()
                size += 1
                for e in box_edges(n)[b]:
                    if mask >> e & 1:
                        continue
                    for nb in edge_boxes(n)[e]:
                        if nb != b and nb in two and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
            sizes.append(size)
        return sizes

    def _ordered(self, mask: int) -> list[tuple[int, int, int]]:
        """(edge, child mask, boxes gained), captures first, then safe edges, then sacrifices."""
        n = self.n
        masks, eb = box_masks(n), edge_boxes(n)
        moves = []
        for e in range(n_edges(n)):
            if mask >> e & 1:
                continue
            child = mask | (1 << e)
            gained = sum(1 for b in eb[e] if child & masks[b] == masks[b])
            rank = 0 if gained else (1 if not third_side_boxes(n, mask, e) else 2)
            moves.append((rank, e, child, gained))
        moves.sort()
        return [(e, child, gained) for _, e, child, gained in moves]

    def bounded(self, mask: int, depth: int, alpha: int, beta: int, deadline: float) -> int:
        if mask == self.full_mask:
            return 0
        v = self.exact.get(mask)
        if v is not None:
            return v
        if depth == 0:
            return self.heuristic(mask)
        self.nodes += 1
        if self.nodes & 255 == 0 and time.perf_counter() > deadline:
            raise _OutOfTime()
        best = -10 ** 6
        for e, child, gained in self._ordered(mask):
            if gained:
                v = gained + self.bounded(child, depth, alpha - gained, beta - gained, deadline)
            else:
                v = -self.bounded(child, depth - 1, -beta, -alpha, deadline)
            if v > best:
                best = v
            if v > alpha:
                alpha = v
            if alpha >= beta:
                break
        return best

    # -- what Buddy uses ---------------------------------------------------------------------------
    def move_scores(self, s: State) -> dict[int, int]:
        """Net boxes for the mover after each legal edge (higher is better)."""
        t0 = time.perf_counter()
        n, mask = s.n, s.edges
        remaining = n_edges(n) - bin(mask).count("1")
        moves = self._ordered(mask)
        self.nodes = 0
        if remaining <= 14 or n <= 2:
            try:
                out = {e: (g + self.value(c) if g else -self.value(c)) for e, c, g in moves}
                self.last_depth, self.last_seconds = 0, time.perf_counter() - t0
                return out
            except _OutOfTime:
                pass                                   # what was proven stays in self.exact
        deadline = t0 + self.time_limit
        out = {e: (g + self.heuristic(c) if g else -self.heuristic(c)) for e, c, g in moves}
        depth = 1
        while depth < remaining:
            depth += 1
            try:
                trial = {}
                for e, c, g in moves:
                    if g:
                        trial[e] = g + self.bounded(c, depth, -10 ** 6, 10 ** 6, deadline)
                    else:
                        trial[e] = -self.bounded(c, depth - 1, -10 ** 6, 10 ** 6, deadline)
                out = trial
                self.last_depth = depth
            except _OutOfTime:
                break
            if time.perf_counter() > deadline:
                break
        self.last_seconds = time.perf_counter() - t0
        return out

    def trappiest(self, s: State, best: list[int]) -> list[int]:
        """Among equally good edges, the ones after which the player has the fewest replies that keep their
        value: a lost position is still lost against perfect play, but most people are not perfect. Only when the
        search is exact (the child positions are then already in the memo)."""
        if len(best) < 2 or self.last_depth != 0:
            return best
        counts = {}
        for e in best:
            child = play(s, e)
            if child.to_move == s.to_move:
                counts[e] = 0
            else:
                sc = self.move_scores(child)
                top = max(sc.values())
                counts[e] = sum(1 for v in sc.values() if v == top)
        fewest = min(counts.values())
        return [e for e in best if counts[e] == fewest]

    def choose(self, s: State, diff: Difficulty, rng=random) -> Choice:
        scores = self.move_scores(s)
        top = max(scores.values())
        best = self.trappiest(s, [e for e, v in scores.items() if v == top])
        if rng.random() < diff.strength or len(scores) == 1:
            e = rng.choice(best)
            return Choice(e, best, scores, True, f"best edge ({diff.strength:.0%} strength)")
        safe = [e for e in scores if e not in best and is_safe(s.n, s.edges, e)]
        if not safe:
            e = rng.choice(best)
            return Choice(e, best, scores, True, "every other edge would hand you a box, so the best one")
        e = rng.choice(safe)
        return Choice(e, best, scores, False, f"a friendlier edge ({1 - diff.strength:.0%} of the time)")


def greedy_player(s: State, rng=random) -> int:
    """A plain opponent for dry runs and tests: takes a box when it can, else a safe edge, else any edge."""
    moves = legal_moves(s)
    taking = [e for e in moves if completed_by(s.n, s.edges, e)]
    if taking:
        return rng.choice(taking)
    safe = [e for e in moves if is_safe(s.n, s.edges, e)]
    return rng.choice(safe or moves)


def describe(n: int) -> str:
    dots = n + 1
    return (f"Here's how it works. I'll draw a grid of {dots} by {dots} dots. We take turns joining two dots next to "
            "each other with a line, across or down. Whoever draws the fourth side of a little box wins that box, "
            "puts their mark in it, and goes again. When every line is drawn, the one with more boxes wins.")


__all__ = ["State", "Solver", "Choice", "Difficulty", "legal_moves", "play", "result", "completed_by", "is_safe",
           "third_side_boxes", "n_edges", "h_edge", "v_edge", "box_edges", "edge_boxes", "edge_dots", "edge_name",
           "box_name", "greedy_player", "describe"]
