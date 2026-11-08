"""Spoken facts and suggestions from checked state, without a robot or model."""
from copy import deepcopy
import random
import re
from types import SimpleNamespace

import numpy as np
import pytest

from buddy.agent.narration import explain_last_move, hint_text, score_text
from buddy.game import dotsboxes as dots
from buddy.game import rules, tictactoe
from buddy.game.hangman import Hangman
from buddy.game.solver import Solver


def grid_brain(board=None, **overrides):
    values = dict(state="PLAYER_TURN", board=list(board or [""] * 9), player_sym="X", robot_sym="O",
                  spec=None, gstate=None, solver=None, last_choice=None, ignored=set(), busy=False,
                  rng=random.Random(7), entry=SimpleNamespace(kind="grid"))
    values.update(overrides)
    return SimpleNamespace(**values)


def dots_brain(game_state=None, n=2, **overrides):
    game = game_state or dots.State.start(n, ("P", "B"))
    values = dict(state="DOTS_PLAYER_TURN", dots=SimpleNamespace(n=n, state=game),
                  last_choice=None, entry=SimpleNamespace(kind="dots"))
    values.update(overrides)
    return SimpleNamespace(**values)


def hangman_brain(game=None, **overrides):
    values = dict(state="HANGMAN_TURN", hangman=game or Hangman(),
                  entry=SimpleNamespace(kind="hangman"), _hangman_pending=None)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("state", ["IDLE", "CHAT", "CHOOSE_GAME", "TEACH", "NAME_GAME", "OFFER_TAUGHT",
                                 "ASK_LEVEL", "SETUP", "ASK_SIDE", "WAIT_SHEET", "DOTS_SIZE",
                                 "DOTS_SETUP", "HANGMAN_SETUP"])
def test_outside_play_ignores_old_games_and_choices(state):
    brain = grid_brain(state=state, hangman=Hangman(), dots=dots_brain().dots,
                       last_choice=tictactoe.Choice(4, [4], {4: 0}, True, "old game"))
    for function in (score_text, explain_last_move, hint_text):
        assert function(brain) == "We aren't playing a game just now."


@pytest.mark.parametrize("state, turn", [("PLAYER_TURN", "your turn"), ("ROBOT_TURN", "my turn"),
                                        ("CONFIRM_MARK", "check your last mark")])
def test_grid_score_uses_marks_and_actual_turn(state, turn):
    brain = grid_brain(["X", "O", "X", "", "", "", "", "", ""], state=state)
    text = score_text(brain)
    assert "two marks" in text and "one mark" in text and turn in text
    assert "winning" not in text  # The number of marks is not a lead.


@pytest.mark.parametrize("board, player, ending", [
    (["X", "X", "X", "O", "O", "", "", "", ""], "X", "You won"),
    (["X", "X", "X", "O", "O", "", "", "", ""], "O", "I won"),
    (list("XOXXOOOXX"), "X", "a draw"),
])
def test_grid_score_recognises_results(board, player, ending):
    assert ending in score_text(grid_brain(board, state="GAME_OVER", player_sym=player))


def test_taught_game_result_uses_its_rules_and_checked_board():
    board = ("X", "X", "X", "O", "O", "", "", "", "")
    brain = grid_brain(board, state="GAME_OVER", spec=rules.MISERE,
                       gstate=rules.State(("",) * 9, 2, "O"))
    assert "I won" in score_text(brain)  # Making three loses in this taught game.
    board = ("X", "X", "", "O", "", "", "", "", "")
    brain = grid_brain(board, state="GAME_OVER", spec=rules.FOOTSTEPS,
                       gstate=rules.State(board, 0, "O"))
    assert "You won" in score_text(brain)  # No legal neighbouring square remains.


def test_dots_score_counts_boxes_and_lines_and_extra_turn():
    game = dots.State.start(2, ("P", "B"), first="P")
    for edge in dots.box_edges(2)[0]:
        game = dots.play(game, edge)
    brain = dots_brain(game)
    text = score_text(brain)
    assert "zero boxes" in text and "one box" in text
    assert "four of twelve lines" in text and "my turn" in text


@pytest.mark.parametrize("owners, ending", [(tuple("PPBB"), "a draw"), (tuple("PPPB"), "you won"),
                                           (tuple("PBBB"), "i won")])
def test_dots_final_score(owners, ending):
    game = dots.State(2, (1 << dots.n_edges(2)) - 1, owners, "B", ("P", "B"))
    assert ending in score_text(dots_brain(game, state="GAME_OVER"))


def test_hangman_score_counts_repeated_letters_without_revealing_answer():
    game = Hangman()
    game.guess("P")
    game.guess("Z")
    text = score_text(hangman_brain(game))
    assert "two of five letters" in text and "five misses left" in text and "your turn" in text
    assert "APPLE" not in text


@pytest.mark.parametrize("letters, ending", [("APLE", "You found the word"), ("BCDFGH", "game is over")])
def test_hangman_score_after_game_over(letters, ending):
    game = Hangman()
    for letter in letters:
        game.guess(letter)
    text = score_text(hangman_brain(game, state="GAME_OVER"))
    assert ending in text and "your turn" not in text


def choice_brain(scores, chosen=4, played_best=True, **kwargs):
    best = [c for c, v in scores.items() if v == max(scores.values())]
    board = [""] * 9
    board[chosen] = "O"
    return grid_brain(board, last_choice=tictactoe.Choice(chosen, best, scores, played_best,
                                                        "unsupported claim about a fork"), **kwargs)


def test_explanation_uses_scores_without_inventing_a_tactical_reason():
    text = explain_last_move(choice_brain({0: -4, 4: 0, 8: -2}))
    assert "centre" in text and "only way to avoid losing" in text
    assert "fork" not in text and "three in a row" not in text
    text = explain_last_move(choice_brain({0: -4, 4: -2, 8: -3}))
    assert "you could still win" in text and "avoid losing" not in text


@pytest.mark.parametrize("scores, expected", [({0: 0, 4: 0}, "keep the game level"),
                                             ({0: 1, 4: 3}, "way to win")])
def test_explanation_of_best_equal_or_winning_move(scores, expected):
    assert expected in explain_last_move(choice_brain(scores))


def test_friendly_and_bounded_moves_are_explained_honestly():
    assert "gentler move than my best" in explain_last_move(choice_brain({0: 0, 4: -2}, played_best=False))
    brain = choice_brain({0: -3, 4: 0}, solver=SimpleNamespace(exhausted=True))
    assert "moves I could check" in explain_last_move(brain)
    assert "only way" not in explain_last_move(brain)
    # A tied score is not a gentler move, even with an inconsistent flag.
    assert "gentler" not in explain_last_move(choice_brain({0: 0, 4: 0}, played_best=False))


def test_no_explanation_for_missing_unchecked_or_stale_move():
    assert "haven't chosen" in explain_last_move(grid_brain())
    brain = choice_brain({4: 0}, busy=True)
    assert "finish checking" in explain_last_move(brain)
    brain.busy = False
    brain.board[4] = ""
    assert "haven't played a checked move" in explain_last_move(brain)
    brain.board[4] = "O"
    brain.last_choice.scores = {}
    assert "enough detail" in explain_last_move(brain)


def test_dots_explanation_names_edge_and_does_not_turn_future_boxes_into_a_win():
    n = (2, 3)
    game = dots.play(dots.State.start(n, ("P", "B"), first="B"), 0)
    choice = tictactoe.Choice(0, [0], {0: 1, 1: -2}, True, "best edge")
    brain = dots_brain(game, n=n, last_choice=choice)
    text = explain_last_move(brain)
    assert "top side of the box in row one, column one" in text and "gaining boxes" in text
    assert "win" not in text and "avoid losing" not in text
    brain.last_choice = tictactoe.Choice(0, [1], {0: 0, 1: 2}, False, "friendlier edge")
    assert "gentler" in explain_last_move(brain)


def test_hangman_explains_checked_reveals_misses_and_pending_ink():
    game = Hangman()
    brain = hangman_brain(game)
    assert "dashes" in explain_last_move(brain)
    game.guess("P")
    assert "every P" in explain_last_move(brain)
    game.guess("Z")
    assert "gallows part because Z isn't in the word" in explain_last_move(brain)
    brain._hangman_pending = ("E", None, [])
    assert "still need to check" in explain_last_move(brain)


def test_hangman_old_instances_can_use_the_saved_drawing_without_guessing():
    game = Hangman()
    game.guess("P")
    game.last_guess = None
    brain = hangman_brain(game, app=SimpleNamespace(last_strokes=("hangman letter P", [])))
    assert "every P" in explain_last_move(brain)
    brain.app.last_strokes = ("hangman letter Z", [])
    assert "can't explain" in explain_last_move(brain)


def test_grid_hint_uses_players_best_move_and_excludes_ignored_squares():
    brain = grid_brain(["X", "X", "", "O", "", "", "", "O", ""])
    assert hint_text(brain) == "You could try the top right, if you like."
    brain.ignored.add(2)
    assert "top right" not in hint_text(brain)
    brain.ignored.update(range(9))
    assert "isn't a free square" in hint_text(brain)


def test_taught_hint_respects_adjacency_and_does_not_mutate_solver_or_brain():
    state = rules.play(rules.FOOTSTEPS, rules.State.start(rules.FOOTSTEPS, "O"), 0)
    solver = Solver(rules.FOOTSTEPS)
    brain = grid_brain(state.board, spec=rules.FOOTSTEPS, gstate=state, solver=solver)
    snapshot = deepcopy(solver.__dict__)
    rng = brain.rng.getstate()
    board = brain.board[:]
    scores = deepcopy(solver).move_scores(state)
    best = max(sorted(scores), key=scores.get)
    assert rules.cell_name(rules.FOOTSTEPS, best) in hint_text(brain)
    assert best in (1, 3)
    assert solver.__dict__ == snapshot and brain.rng.getstate() == rng
    assert brain.board == board and brain.gstate is state and brain.last_choice is None


def test_taught_hint_uses_camera_board_rather_than_stale_state_board():
    state = rules.State(("O", "X", "", "X", "", "", "", "", ""), 0, "X")
    brain = grid_brain(state.board, spec=rules.FOOTSTEPS, gstate=rules.State(("",) * 9, 0, "X"),
                       solver=Solver(rules.FOOTSTEPS))
    assert "isn't a move" in hint_text(brain)


def test_dots_hint_takes_a_box_before_suggesting_a_safe_edge():
    edges = dots.box_edges(2)[0]
    mask = sum(1 << e for e in edges[:-1])
    game = dots.State(2, mask, ("",) * 4, "P", ("P", "B"))
    brain = dots_brain(game)
    text = hint_text(brain)
    assert "take a box" in text and dots.edge_name(2, edges[-1]) in text
    assert "if you like" in text and brain.dots.state is game
    text = hint_text(dots_brain())
    assert "won't leave a box ready for me" in text


def test_dots_hint_admits_when_every_line_gives_away_a_box():
    n = 2
    mask = next(mask for mask in range(1 << dots.n_edges(n))
                if mask != (1 << dots.n_edges(n)) - 1
                and all(not dots.completed_by(n, mask, e) and not dots.is_safe(n, mask, e)
                        for e in range(dots.n_edges(n)) if not mask >> e & 1))
    game = dots.State(n, mask, ("",) * 4, "P", ("P", "B"))
    assert "can't see a line" in hint_text(dots_brain(game))


@pytest.mark.parametrize("brain", [grid_brain(state="CONFIRM_MARK"), grid_brain(busy=True),
                                 grid_brain(state="ROBOT_TURN"), dots_brain(state="DOTS_ROBOT_TURN"),
                                 hangman_brain(_hangman_pending=("P", None, [])),
                                 grid_brain(state="GAME_OVER")])
def test_hint_waits_for_a_checked_player_turn(brain):
    assert "You could" not in hint_text(brain)


@pytest.mark.parametrize("level", ["easy", "medium", "hard", "impossible"])
def test_hangman_hint_uses_current_dial_without_replacing_the_word(level):
    game = Hangman()
    brain = hangman_brain(game, difficulty=SimpleNamespace(level=level, strength=.5))
    before = deepcopy(game.__dict__)
    text = hint_text(brain)
    if level == "easy":
        assert "try A, if you like" in text
    elif level == "medium":
        assert f"try {game.hint()}, if you like" in text
    elif level == "hard":
        assert "try a vowel" in text or "try a consonant" in text
    else:
        assert "no hints on impossible" in text
    assert game.__dict__ == before


def test_spoken_text_is_short_and_avoids_numeric_scores_and_hidden_words():
    brains = [grid_brain(), choice_brain({0: -2, 4: 0}), dots_brain(n=(5, 6)), hangman_brain()]
    for brain in brains:
        for function in (score_text, explain_last_move, hint_text):
            text = function(brain)
            assert not re.search(r"\d|[%+]", text)
            assert 1 <= text.count(".") <= 2 and len(text.split()) <= 40
            assert "APPLE" not in text


def test_redraw_submits_fresh_copies_with_current_sheet_and_preserves_source():
    from buddy.app import App

    class Controller:
        def __init__(self):
            self.jobs = []
            self.paused = True

        def submit(self, job):
            self.jobs.append(job)
            return job

    app = App.__new__(App)
    app.ctl = Controller()
    app.tracker = SimpleNamespace(track=object())
    app.last_strokes = None
    assert app.redraw_last() is None and not app.ctl.jobs
    strokes = [np.array([[0., 0.], [10., 10.]]), np.array([[0., 10.], [10., 0.]])]
    original = app.draw("my cross", strokes)
    original.done.set()
    original.ok = True
    again = app.redraw_last()
    assert app.ctl.jobs == [original, again]
    assert again.name == "again: my cross" and again.home
    assert not again.done.is_set() and not again.ok and app.ctl.paused
    for source, copied in zip(strokes, again.strokes_mm):
        np.testing.assert_array_equal(source, copied)
        assert not np.shares_memory(source, copied)
    app.tracker.track = object()
    assert again.sheet() is app.tracker.track
    assert app.redraw_last().name == "again: my cross"
    assert app.redraw_last("faint cross").name == "again: faint cross"
    again.strokes_mm[0][0] = 999
    np.testing.assert_array_equal(app.last_strokes[1][0], strokes[0])
