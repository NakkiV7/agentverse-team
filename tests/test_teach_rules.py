"""The teachable-games engine: rules, solver and (when Ollama is up) the Gemma interview.

    .venv/bin/python -m pytest -q tests/test_teach_rules.py
"""
import random
import urllib.request

import pytest

from buddy.game import tictactoe as ttt
from buddy.game.rules import FOOTSTEPS, FOOTSTEPS_DIAG, MISERE, TIC_TAC_TOE, GameSpec, State, legal_moves, line_of, play, result
from buddy.game.solver import Solver, analyse


def random_game(spec, rng, solver=None, me="X", strength=1.0):
    s = State.start(spec)
    diff = ttt.Difficulty(strength=strength)
    while not result(spec, s):
        if solver and s.to_move == me:
            c = solver.choose(s, diff, rng).cell
        else:
            c = rng.choice(legal_moves(spec, s))
        s = play(spec, s, c)
    return s, result(spec, s)


def test_tictactoe_spec_agrees_with_hakims_rules():
    rng = random.Random(0)
    for _ in range(300):
        s, r = random_game(TIC_TAC_TOE, rng)
        w = ttt.winner(list(s.board))
        if w:
            assert r == ("win", w[0], tuple(w[1])) or (r[0] == "win" and r[1] == w[0])
        else:
            assert r[0] == "draw" and ttt.full(list(s.board))


def test_footsteps_results_match_the_rules_sheet():
    _, a = analyse(FOOTSTEPS)
    assert a.complete and a.outcome == "first" and a.winning_first_moves == [0, 2, 4, 6, 8]
    _, a = analyse(FOOTSTEPS_DIAG)
    assert a.outcome == "first" and a.winning_first_moves == list(range(9))
    _, a = analyse(GameSpec(rows=3, cols=4, placement="adjacent_to_last_mark", end="no_legal_move_loses"))
    assert a.outcome == "second"
    _, a = analyse(GameSpec(rows=4, cols=4, placement="adjacent_to_last_mark", end="no_legal_move_loses"))
    assert a.outcome == "second"


def test_known_solved_games():
    _, a = analyse(TIC_TAC_TOE)
    assert a.outcome == "draw" and a.drawing_first_moves == list(range(9))
    _, a = analyse(MISERE)
    assert a.outcome == "draw" and a.drawing_first_moves == [4]       # misère: only the centre holds the draw


@pytest.mark.parametrize("spec", [TIC_TAC_TOE, MISERE, FOOTSTEPS, FOOTSTEPS_DIAG])
def test_perfect_solver_never_loses_as_either_side(spec):
    rng = random.Random(1)
    for me in ("X", "O"):
        solver, a = analyse(spec)
        for _ in range(60):
            _, r = random_game(spec, rng, solver, me)
            if r[0] == "win":
                assert r[1] == me or (me == "O" and a.outcome == "first") or (me == "X" and a.outcome == "second"), (spec.name, me, r)


def test_footsteps_first_player_always_wins_from_a_corner():
    solver, _ = analyse(FOOTSTEPS)
    rng = random.Random(2)
    for _ in range(100):
        s = play(FOOTSTEPS, State.start(FOOTSTEPS), rng.choice([0, 2, 4, 6, 8]))
        while not result(FOOTSTEPS, s):
            c = solver.choose(s, ttt.Difficulty(1.0), rng).cell if s.to_move == "X" else rng.choice(legal_moves(FOOTSTEPS, s))
            s = play(FOOTSTEPS, s, c)
        assert result(FOOTSTEPS, s) == ("win", "X", None)


def test_big_board_falls_back_to_lookahead_quickly():
    import time
    spec = GameSpec(rows=4, cols=4, k=4)
    solver, a = analyse(spec, node_budget=100_000)
    assert not a.complete and a.outcome == "unknown"
    t0 = time.perf_counter()
    ch = solver.choose(State.start(spec), ttt.Difficulty(1.0))
    assert ch.cell in range(16) and time.perf_counter() - t0 < 5


def test_describe_and_names():
    assert "next to the last mark" in FOOTSTEPS.describe()
    assert "three in a row, you lose" in MISERE.describe().lower()
    assert line_of(TIC_TAC_TOE, ("X", "X", "X", "", "O", "O", "", "", "")) == ("X", (0, 1, 2))


def _ollama_has(model):
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as r:
            import json
            return any(m["name"].startswith(model) for m in json.loads(r.read())["models"])
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _ollama_has("gemma4"), reason="needs Ollama with gemma4")
def test_gemma_learns_footsteps_from_one_sentence():
    from buddy.agent.teach import Teacher
    t = Teacher()
    turn = t.hear("We take turns putting our mark next to the last one, up, down or sideways. If you can't go anywhere, you lose.")
    assert turn.done and turn.spec is not None
    assert turn.spec.placement == "adjacent_to_last_mark" and turn.spec.end == "no_legal_move_loses"


@pytest.mark.skipif(not _ollama_has("gemma4"), reason="needs Ollama with gemma4")
def test_gemma_asks_when_the_ending_is_missing():
    from buddy.agent.teach import Teacher
    t = Teacher()
    turn = t.hear("Each mark has to be next to the one before it, no diagonals.")
    assert not turn.done and turn.say.endswith("?")
    turn = t.hear("If you can't place a mark, you've lost.")
    assert turn.done and turn.spec and turn.spec.end == "no_legal_move_loses"
