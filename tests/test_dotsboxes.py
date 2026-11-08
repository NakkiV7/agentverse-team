"""Dots and boxes rules and solver (pure Python, no simulator).

    .venv/bin/python -m pytest -q tests/test_dotsboxes.py
"""
import random
import time
from functools import lru_cache

import pytest

from buddy.game import dotsboxes as D
from buddy.game.tictactoe import Difficulty


def brute_force(n: int):
    """An independent, unoptimised solver: net boxes for the mover from an edge set, by trying everything."""
    boxes = D.box_edges(n)

    @lru_cache(maxsize=None)
    def value(drawn: frozenset) -> int:
        if len(drawn) == D.n_edges(n):
            return 0
        best = None
        for e in range(D.n_edges(n)):
            if e in drawn:
                continue
            after = drawn | {e}
            gained = sum(1 for edges in boxes if e in edges and all(x in after for x in edges))
            v = gained + value(after) if gained else -value(after)
            best = v if best is None else max(best, v)
        return best
    return value


def test_geometry():
    assert D.n_edges(2) == 12 and D.n_edges(3) == 24
    assert D.box_edges(2)[0] == (0, 2, 6, 7)                  # top-left box: top, bottom, left, right
    assert D.box_edges(2)[3] == (3, 5, 10, 11)
    assert D.edge_boxes(2)[2] == (0, 2)                       # the edge between the two left boxes
    assert D.edge_boxes(2)[0] == (0,)                         # a border edge has one box
    assert D.edge_dots(2, 0) == ((0, 0), (0, 1))
    assert D.edge_dots(2, 6) == ((0, 0), (1, 0))
    assert D.edge_name(2, 0) == "the top side of the top left box"
    assert D.edge_name(2, 5) == "the bottom side of the bottom right box"
    assert D.edge_name(2, 11) == "the right side of the bottom right box"


def test_legal_moves_and_extra_turn():
    s = D.State.start(2, ("P", "B"))
    assert D.legal_moves(s) == list(range(12))
    for e in (0, 6, 7):
        s = D.play(s, e)
    assert len(D.legal_moves(s)) == 9 and 7 not in D.legal_moves(s)
    assert s.to_move == "B"                                    # three plain moves: P, B, P -> B to move
    assert D.completed_by(2, s.edges, 2) == [0]
    s = D.play(s, 2)                                           # B closes the top-left box...
    assert s.owners[0] == "B" and s.to_move == "B"             # ...and moves again
    with pytest.raises(ValueError):
        D.play(s, 2)
    assert D.result(s) is None
    assert D.third_side_boxes(2, s.edges, 3) == [] and D.is_safe(2, s.edges, 3)
    s = D.play(s, 1)                                           # B's extra move: top of the top-right box
    assert s.to_move == "P"
    assert D.third_side_boxes(2, s.edges, 8) == [1]            # its right side would leave it with 3 sides
    assert not D.is_safe(2, s.edges, 8) and D.is_safe(2, s.edges, 5)


def test_known_results_on_small_boards():
    assert D.Solver(1).value(0) == -1                          # 1x1: whoever starts loses the only box
    sol = D.Solver(2)
    assert sol.value(0) == 2                                   # 2x2: the first player wins 3 boxes to 1
    ref = brute_force(2)
    assert ref(frozenset()) == 2
    rng = random.Random(3)
    for _ in range(40):                                        # random positions agree with the brute force
        drawn = frozenset(e for e in range(12) if rng.random() < 0.5)
        mask = sum(1 << e for e in drawn)
        assert sol.value(mask) == ref(drawn), sorted(drawn)
    s = D.State.start(2)
    for e in (0, 6, 7):
        s = D.play(s, e)
    scores = sol.move_scores(s)
    assert max(scores, key=scores.get) == 2                    # take the box that is on offer
    assert scores[2] > max(v for e, v in scores.items() if e != 2)


def play_out(n, buddy, rng, strength=1.0, opponent="random", exact=None):
    """Buddy (the solver) against a random or greedy opponent. With `exact` (a Solver with the full memo), also
    reports whether the opponent ever made a mistake, measured against the exact values."""
    sol = D.Solver(n)
    s = D.State.start(n, ("P", "B"), first="P" if buddy == "second" else "B")
    worst, mistake = 0.0, False
    while D.result(s) is None:
        if s.to_move == "B":
            e = sol.choose(s, Difficulty(strength), rng).cell
            worst = max(worst, sol.last_seconds)
        else:
            e = D.greedy_player(s, rng) if opponent == "greedy" else rng.choice(D.legal_moves(s))
            if exact is not None:
                sc = exact.move_scores(s)
                mistake = mistake or sc[e] < max(sc.values())
        s = D.play(s, e)
    return s, worst, mistake


def test_solver_never_loses_to_random_on_2x2():
    """2x2 is a forced 3-1 win for the first player (test_known_results_on_small_boards). Going first, the solver
    never loses. Going second it is in a lost position, so it can only lose when the random player happens to play
    every move perfectly; measured: once in 2,000 games."""
    rng = random.Random(0)
    exact = D.Solver(2)
    for _ in range(150):
        s, _, _ = play_out(2, "first", rng)
        assert s.score("B") > s.score("P"), s.owners
    for _ in range(300):
        s, _, mistake = play_out(2, "second", rng, exact=exact)
        if s.score("B") < s.score("P"):
            assert not mistake, ("lost to a random player who made a mistake", s.owners)


def test_solver_beats_greedy_on_2x2_from_the_right_side():
    rng = random.Random(1)
    wins = sum(play_out(2, "first", rng, opponent="greedy")[0].score("B") > 2 for _ in range(30))
    assert wins == 30                                          # a 3-1 win is forced for the first player


def test_3x3_moves_stay_under_a_second():
    rng = random.Random(2)
    s, worst, _ = play_out(3, "first", rng, opponent="greedy")
    assert D.result(s) is not None
    assert worst < 1.2, worst
    sol = D.Solver(3)
    t0 = time.perf_counter()
    sol.move_scores(D.State.start(3))
    assert time.perf_counter() - t0 < 1.2 and sol.last_depth >= 3


def test_friendly_move_does_not_give_a_box():
    rng = random.Random(5)
    sol = D.Solver(2)
    s = D.State.start(2, ("P", "B"), first="B")
    gifts = 0
    for _ in range(40):
        ch = sol.choose(s, Difficulty(0.0), rng)
        if D.third_side_boxes(2, s.edges, ch.cell):
            gifts += 1
        assert not ch.played_best or "every other" in ch.reason
    assert gifts == 0
