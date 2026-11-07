"""Plays any GameSpec game perfectly from the first move: full negamax with a memo, and the analysis
Buddy speaks after learning a game ("if you go first you can always win, but only from a corner or the
middle"). For boards too big to search completely (4x4 with marks anywhere) it looks a few moves ahead
instead and says so.

Scores are from the point of view of the player to move: +WIN - d for a forced win in d plies, d - WIN
for a forced loss, 0 for a draw. Quick wins and slow losses are preferred, as in tictactoe.py.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from buddy.game.rules import GameSpec, State, cell_name, legal_moves, other, play, result
from buddy.game.tictactoe import Choice, Difficulty

WIN = 100


class Solver:
    def __init__(self, spec: GameSpec, node_budget: int = 400_000, lookahead: int = 6):
        self.spec = spec
        self.memo: dict = {}
        self.nodes = 0
        self.budget = node_budget
        self.lookahead = lookahead
        self.exhausted = False           # the budget ran out: values below are estimates, not proofs

    # -- exact search -----------------------------------------------------------------------------
    def _key(self, s: State):
        if self.spec.end == "no_legal_move_loses":        # only occupancy matters, not who wrote what
            return tuple(bool(v) for v in s.board), s.last, s.to_move
        return s.board, s.last, s.to_move

    def value(self, s: State) -> int:
        key = self._key(s)
        if key in self.memo:
            return self.memo[key]
        self.nodes += 1
        r = result(self.spec, s)
        if r:
            v = 0 if r[0] == "draw" else (WIN if r[1] == s.to_move else -WIN)
            self.memo[key] = v
            return v
        if self.nodes > self.budget:
            self.exhausted = True
            return self._heuristic(s)
        best = -10 ** 9
        for m in legal_moves(self.spec, s):
            vc = self.value(play(self.spec, s, m))
            v = -(vc - 1) if vc > 0 else -(vc + 1) if vc < 0 else 0     # one ply further from the end
            best = max(best, v)
        if not self.exhausted:
            self.memo[key] = best
        return best

    # -- bounded search for big boards -------------------------------------------------------------
    def _heuristic(self, s: State) -> int:
        """Rough static score for the player to move: open lines for k-in-a-row games, mobility otherwise."""
        me, opp = s.to_move, other(s.to_move)
        if self.spec.end == "no_legal_move_loses":
            return min(len(legal_moves(self.spec, s)), 8)
        from buddy.game.rules import lines
        score = 0
        for ln in lines(self.spec):
            marks = {s.board[i] for i in ln} - {""}
            if marks == {me}:
                score += 1
            elif marks == {opp}:
                score -= 1
        return score if self.spec.end == "k_in_a_row_wins" else -score

    def bounded(self, s: State, depth: int, alpha: int = -10 ** 9, beta: int = 10 ** 9) -> int:
        """Depth-limited negamax with alpha-beta pruning; immediate wins are tried first."""
        r = result(self.spec, s)
        if r:
            return 0 if r[0] == "draw" else (WIN if r[1] == s.to_move else -WIN)
        if depth == 0:
            return self._heuristic(s)
        moves = legal_moves(self.spec, s)
        children = [(m, play(self.spec, s, m)) for m in moves]
        children.sort(key=lambda mc: 0 if result(self.spec, mc[1]) else 1)
        best = -10 ** 9
        for m, child in children:
            vc = self.bounded(child, depth - 1, -beta, -alpha)
            v = -(vc - 1) if vc > 0 else -(vc + 1) if vc < 0 else 0
            if v > best:
                best = v
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best

    def depth_for(self, s: State) -> int:
        """How far ahead to look on a board too big to solve: keeps the search around 10^5 nodes."""
        empties = sum(1 for v in s.board if not v)
        if empties <= 7:
            return empties                  # solve the endgame exactly
        return {8: 6, 9: 5, 10: 5, 11: 4, 12: 4}.get(empties, 3)

    # -- what Buddy uses --------------------------------------------------------------------------
    def move_scores(self, s: State) -> dict[int, int]:
        """Score of every legal move for the player to move (higher is better)."""
        out = {}
        for m in legal_moves(self.spec, s):
            child = play(self.spec, s, m)
            vc = self.value(child) if not self.exhausted else self.bounded(child, self.depth_for(child))
            out[m] = -(vc - 1) if vc > 0 else -(vc + 1) if vc < 0 else 0
        return out

    def ends_game_for_opponent_next(self, s: State, m: int) -> bool:
        """After my move m, can the opponent finish the game in their favour at once?"""
        child = play(self.spec, s, m)
        if result(self.spec, child):
            return False
        for om in legal_moves(self.spec, child):
            r = result(self.spec, play(self.spec, child, om))
            if r and r[0] == "win" and r[1] == child.to_move:
                return True
        return False

    def choose(self, s: State, diff: Difficulty, rng=random) -> Choice:
        scores = self.move_scores(s)
        top = max(scores.values())
        best = [m for m, v in scores.items() if v == top]
        if rng.random() < diff.strength or len(scores) == 1:
            m = rng.choice(best)
            return Choice(m, best, scores, True, f"best move ({diff.strength:.0%} strength)")
        safe = [m for m in scores if m not in best and not self.ends_game_for_opponent_next(s, m)]
        if not safe:
            m = rng.choice(best)
            return Choice(m, best, scores, True, "every other move would lose at once, so the best one")
        m = rng.choice(safe)
        return Choice(m, best, scores, False, f"a friendlier move ({1 - diff.strength:.0%} of the time)")


@dataclass
class Analysis:
    outcome: str                      # "first" | "second" | "draw" | "unknown"
    winning_first_moves: list[int]
    drawing_first_moves: list[int]
    nodes: int
    seconds: float
    complete: bool

    def speak(self, spec: GameSpec) -> str:
        """The line Buddy says after thinking the game through."""
        n = spec.n
        if not self.complete:
            return "That board is too big for me to work out completely, so I'll play by looking a few moves ahead."
        if self.outcome == "first":
            if len(self.winning_first_moves) == n:
                return "Interesting! Whoever goes first can always win, from any square."
            names = _join([cell_name(spec, c) for c in sorted(self.winning_first_moves)])
            return f"Interesting! Whoever goes first can always win, but only by starting in the {names}."
        if self.outcome == "second":
            return "Interesting! Whoever goes second can always win, if they play it perfectly."
        if self.outcome == "draw":
            return "With perfect play it's always a draw, so the first slip decides it."
        return "I couldn't work that one out completely."


def _join(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " or " + names[-1]


def analyse(spec: GameSpec, first: str = "X", node_budget: int = 400_000) -> tuple[Solver, Analysis]:
    """Solve the game from the empty board. Returns the solver (with its memo, ready to play) and the verdict."""
    t0 = time.perf_counter()
    solver = Solver(spec, node_budget=node_budget)
    s = State.start(spec, first)
    scores = solver.move_scores(s)
    wins = sorted(m for m, v in scores.items() if v > 0)
    draws = sorted(m for m, v in scores.items() if v == 0)
    complete = not solver.exhausted
    outcome = "unknown" if not complete else "first" if wins else "draw" if draws else "second"
    return solver, Analysis(outcome, wins, draws, solver.nodes, time.perf_counter() - t0, complete)
