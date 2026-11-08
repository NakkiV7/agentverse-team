"""The game library and difficulty levels."""
import json
import os

from buddy.game.library import Library
from buddy.game.rules import FOOTSTEPS, GameSpec
from buddy.game.tictactoe import LEVELS, Difficulty


def test_builtins_and_matching(tmp_path):
    L = Library(tmp_path / "games.json")
    assert L.names()[0] == "noughts and crosses" and "Footsteps" in L.names()
    assert L.find("let's play footsteps").name == "Footsteps"
    assert L.find("tic tac toe").name == "noughts and crosses"
    assert L.find("Footstepz").name == "Footsteps"                       # close match
    assert L.find("the backwards one") is None


def test_remember_and_reload(tmp_path):
    p = tmp_path / "games.json"
    L = Library(p)
    spec = GameSpec(placement="adjacent_to_last_mark_or_diagonal", end="k_in_a_row_wins", k=3)
    e = L.remember(spec, "Corners")
    L.record(e, "win"); L.record(e, "draw")
    assert L.remember(spec, "Another name") is e                       # same rules: not duplicated
    assert L.remember(FOOTSTEPS, "x").builtin                           # built-in rules are recognised
    L2 = Library(p)
    got = L2.find("corners")
    assert got and got.spec == spec.with_(name="Corners") and got.plays == 2 and got.player_wins == 1
    assert json.loads(p.read_text())["games"][0]["name"] == "Corners"


def test_unnamed_game_gets_a_name(tmp_path):
    L = Library(tmp_path / "g.json")
    e = L.remember(GameSpec(end="k_in_a_row_loses", placement="adjacent_to_last_mark"), "")
    assert e.name == "your game 1"


def test_difficulty_levels():
    d = Difficulty()
    assert d.level == "adaptive"
    d.set_level("impossible"); assert d.strength == 1.0
    d.update("loss"); assert d.strength == 1.0                          # fixed levels do not move
    d.set_level("adaptive"); d.strength = 0.75; d.update("loss"); assert d.strength < 0.75
    assert set(LEVELS) == {"easy", "medium", "hard", "impossible"}


def test_a_broken_games_file_is_set_aside_and_bad_entries_skipped(tmp_path):
    from buddy.game.library import Library
    path = tmp_path / "games.json"
    path.write_text('{"games": [{"name": "Chains", "spec"')            # half-written
    lib = Library(path, hidden=())
    assert lib.find("hangman") is not None and not path.exists()
    assert list(tmp_path.glob("games.json.broken-*"))
    path.write_text('{"games": [{"spec": null}, {"name": "Chains", "spec": null}], "builtin_history": [{"plays": 2}]}')
    lib = Library(path, hidden=())
    assert lib.find("Chains") is not None                              # the good entry survives the bad one


def test_save_leaves_no_temporary_file(tmp_path):
    from buddy.game.library import Library
    lib = Library(tmp_path / "games.json", hidden=())
    lib.record(lib.find("hangman"), "win")
    assert [p.name for p in tmp_path.iterdir()] == ["games.json"]
