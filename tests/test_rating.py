"""Persistent skill, deliberate variation, and real games against a fixed player."""
import json
import random
import threading
from collections import Counter
from unittest.mock import Mock

import pytest

from buddy.game.rating import Rating, expected_score, select_strength
from buddy.game.tictactoe import LEVELS, Difficulty, choose, full, other, winner


@pytest.mark.parametrize("result,player,buddy", [("win", 1032, 992), ("loss", 968, 1008), ("draw", 1000, 1000)])
def test_rating_updates_both_sides_and_reloads(result, player, buddy):
    rating = Rating()
    assert rating.expected(0) > rating.expected(.5) > rating.expected(1)
    assert expected_score(1200, 1000) == pytest.approx(0.7597469)
    rating.update(result, .5)
    saved = Rating()
    assert saved.player.rating == player
    assert saved.buddy["0.50"].rating == buddy
    assert saved.player.games == saved.buddy["0.50"].games == 1
    assert saved.player.recent == [result]
    assert saved.buddy["0.30"].games == 0


def test_draw_moves_rating_toward_the_opponent():
    rating = Rating()
    rating.update("draw", 1)
    assert rating.player.rating > 1000
    assert rating.buddy["1.00"].rating < 1400


def test_larger_k_applies_to_exactly_the_first_five_games():
    rating = Rating()
    for _ in range(4):
        rating.update("draw", .5)
    rating.update("win", .5)
    assert rating.player.rating == 1032
    expectation = rating.expected(.5)
    rating.update("win", .5)
    assert rating.player.rating == pytest.approx(1032 + 24 * (1 - expectation))


def test_players_games_and_old_objects_do_not_overwrite_other_records(tmp_path):
    path = tmp_path / "profiles" / "profile.json"
    first = Rating("Footsteps", "pat", path)
    second = Rating("noughts and crosses", "pat", path)
    third = Rating("Footsteps", "sam", path)
    first.update("win", .5)
    second.update("loss", .3)
    third.update("draw", .5)
    first.update("win", .5)
    assert Rating(" footsteps ", "pat", path).player.recent == ["win", "win"]
    assert Rating("Footsteps", "sam", path).player.recent == ["draw"]
    assert Rating("noughts and crosses", "pat", path).player.recent == ["loss"]
    assert Rating("Footsteps", "pat", path).buddy["0.50"].games == 3
    assert len(json.loads(path.read_text())["players"]) == 2


def test_profile_and_player_environment_overrides(monkeypatch, tmp_path):
    path = tmp_path / "another.json"
    monkeypatch.setenv("BUDDY_PROFILE_FILE", str(path))
    monkeypatch.setenv("BUDDY_PLAYER_ID", "pat")
    rating = Rating()
    rating.update("win", .6)
    assert path.exists() and Rating().player.games == 1
    monkeypatch.setenv("BUDDY_PLAYER_ID", "sam")
    assert Rating().player.games == 0


def test_unreadable_profile_starts_with_priors_and_bad_results_do_not_write():
    rating = Rating()
    rating.path.write_text("{truncated")
    rating = Rating()
    assert rating.player.rating == 1000 and rating.buddy["1.00"].rating == 1400
    for result, strength in [("unknown", .5), ("win", -1), ("loss", float("nan"))]:
        with pytest.raises(ValueError):
            rating.update(result, strength)
    assert rating.path.read_text() == "{truncated"


def test_policy_tracks_skill_accounts_for_draws_and_varies():
    rating = Rating()
    low = select_strength(rating, random.Random(0))
    rating.player.rating += 200
    high = select_strength(rating, random.Random(0))
    assert high.target_strength > low.target_strength
    rating.player.recent = ["draw"] * 10
    draws = select_strength(rating, random.Random(0))
    assert draws.target_strength < high.target_strength
    rng = random.Random(4)
    picks = [select_strength(rating, rng) for _ in range(30)]
    assert len({pick.strength for pick in picks}) > 1
    assert all(.05 <= pick.strength <= .95 and abs(pick.strength - pick.target_strength) <= .100001
               for pick in picks)
    rating.player.recent = ["win"] * 3
    assert "won the last three" in select_strength(rating).reason
    rating.player.recent = ["loss"] * 3
    assert "gentler" in select_strength(rating).reason


@pytest.mark.parametrize("skill", [.7, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_target_win_rate_in_200_actual_games(skill, seed):
    """Play legal tic-tac-toe moves, alternate who starts, and count wins separately from draws.

    The scripted player's strength stays fixed for all 200 games. It uses the
    same solver as Penny, with its own constant chance of a friendly move.
    """
    rng = random.Random(seed)
    adaptive = Difficulty(rng=rng)
    scripted = Difficulty(strength=skill)
    counts, strengths, friendly = Counter(), set(), 0
    for game in range(200):
        adaptive.start_game()
        strengths.add(adaptive.strength)
        board, turn = [""] * 9, "X"
        player = "X" if game % 2 else "O"
        while not winner(board) and not full(board):
            choice = choose(board, turn, scripted if turn == player else adaptive, rng)
            friendly += turn != player and not choice.played_best
            board[choice.cell] = turn
            turn = other(turn)
        won = winner(board)
        result = "draw" if not won else "win" if won[0] == player else "loss"
        counts[result] += 1
        adaptive.update(result)
    print(f"skill={skill}, seed={seed}: {dict(counts)}, {len(strengths)} strengths, {friendly} friendly moves")
    assert .40 <= counts["win"] / 200 <= .50
    assert counts["draw"] > 0 and friendly > 0 and len(strengths) > 3
    assert Rating().player.games == 200


@pytest.mark.parametrize("level,strength", LEVELS.items())
def test_fixed_levels_keep_strength_and_move_choices_but_still_learn(level, strength):
    rng = random.Random(19)
    before = rng.getstate()
    difficulty = Difficulty(level=level, rng=rng)
    for result in ("win", "win", "loss", "draw"):
        difficulty.set_level(level)
        difficulty.start_game()
        difficulty.update(result)
        assert difficulty.strength == strength
    assert rng.getstate() == before
    assert Rating().player.games == 4
    board = ["X", "", "", "", "O", "", "", "", ""]
    for seed in range(20):
        assert choose(board, "X", difficulty, random.Random(seed)) == choose(
            board, "X", Difficulty(strength), random.Random(seed))


def test_difficulty_remembers_games_and_prepares_one_strength_per_game():
    difficulty = Difficulty(rng=random.Random(4))
    difficulty.start_game("Footsteps")
    difficulty.update("win")
    prepared = difficulty.strength
    rng_state = difficulty.rng.getstate()
    difficulty.start_game()
    assert difficulty.strength == prepared and difficulty.rng.getstate() == rng_state
    difficulty.select_game("noughts and crosses")
    assert difficulty.rating.player.games == 0
    difficulty.select_game("Footsteps")
    assert difficulty.rating.player.recent == ["win"]
    restarted = Difficulty()
    restarted.start_game("Footsteps", random.Random(4))
    assert restarted.rating.player.rating == difficulty.rating.player.rating
    restarted.set_level("impossible")
    restarted.set_level("adaptive")
    assert .05 <= restarted.strength <= .95


@pytest.mark.parametrize("taught", [False, True])
@pytest.mark.parametrize("result", ["win", "loss", "draw"])
def test_both_brain_finish_paths_record_rating_and_explanation(taught, result, tmp_path):
    from buddy.agent.brain import Brain
    from buddy.app import App
    from buddy.game.library import Library

    brain = Brain.__new__(Brain)              # no worker threads, model calls or simulated arm
    app = brain.app = Mock()
    app.say.return_value = None
    brain.library = Library(tmp_path / "games.json")
    brain.difficulty = Difficulty(rng=random.Random(1))
    brain._select(brain.library.find("Footsteps" if taught else "noughts and crosses"))
    brain.player_sym, brain.robot_sym = "X", "O"
    brain.difficulty.start_game()
    sym = "X" if result == "win" else "O"
    if taught:
        brain._finish_taught(("draw" if result == "draw" else "win", sym, None))
    else:
        brain._finish(None if result == "draw" else (sym, (0, 1, 2)))
    saved = Rating(brain.game_name)
    assert saved.player.games == brain.entry.plays == 1
    assert saved.player.recent == [result]
    assert any("rating " in call.args[1] and brain.difficulty.reason in call.args[1]
               for call in app.trace.call_args_list)

    # Check the actual state endpoint without constructing the MuJoCo world.
    app.brain, app.tracker.track, app.board_snap = brain, None, None
    app._lock, app._trace, app._transcript = threading.Lock(), [], []
    app.speaker, app.listener = None, None
    app.intrusion.max_height_cm, app.world.touch.return_value, app.sim_time = 0, 0, 0
    brain.state, brain.board, brain.games = "GAME_OVER", [""] * 9, 1
    state = App.state(app)
    assert state["game"]["rating"] == round(saved.player.rating)
    assert state["game"]["rating_games"] == 1
    assert state["game"]["difficulty_reason"] == brain.difficulty.reason
