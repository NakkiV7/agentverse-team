"""Learning a hidden game from the player's explanation: recognition, the fixed fallbacks, and the library's menu."""
from pathlib import Path

from buddy.agent import relearn
from buddy.game.library import Library

EXPLANATION = ("You start with a grid of dots. Players take turns drawing a line between two adjacent dots. When a "
               "player completes the fourth side of a box they claim it with their initial and take another turn. "
               "When all lines are drawn, the player with the most boxes wins.")


def library(tmp_path: Path, hidden=("dots and boxes",)) -> Library:
    return Library(tmp_path / "games.json", hidden=hidden)


def test_hidden_game_is_off_the_menu_until_revealed(tmp_path):
    lib = library(tmp_path)
    assert "dots" not in lib.spoken_list()
    assert lib.find("let's play dots and boxes") is None
    assert lib.find("dots and boxes", include_hidden=True).name == "dots and boxes"
    assert lib.find("hangman").name == "hangman"
    entry = lib.reveal(lib.find("dots and boxes", include_hidden=True))
    assert lib.find("dots and boxes") is entry and "dots and boxes" in lib.spoken_list() and entry.taught_at


def test_nothing_hidden_by_default(tmp_path):
    lib = Library(tmp_path / "games.json", hidden=())
    assert lib.find("dots and boxes").name == "dots and boxes"


def test_recognise_the_explanation_only_when_hidden(tmp_path):
    lib = library(tmp_path)
    assert relearn.recognise(lib, EXPLANATION).name == "dots and boxes"
    assert relearn.recognise(lib, "We take turns putting crosses and noughts on a grid of nine squares.") is None
    lib.reveal(lib.find("dots and boxes", include_hidden=True))
    assert relearn.recognise(lib, EXPLANATION) is None          # known games are not re-learned


def test_interview_falls_back_to_fixed_questions_without_a_model(tmp_path, monkeypatch):
    lib = library(tmp_path)
    r = relearn.Relearn(lib.find("dots and boxes", include_hidden=True), EXPLANATION)
    monkeypatch.setattr(r, "_model", lambda prompt, key: None)
    q1 = r.ask()
    r.answer("Yes, you go again.")
    q2 = r.ask()
    assert q1 != q2 and q1.endswith("?") and q2.endswith("?") and r.done_asking
    r.answer("Most boxes wins.")
    s = r.summary()
    assert s.startswith("So, if I've understood") and "?" not in s
    assert r.used_fallback == 3


def test_model_questions_are_checked(tmp_path, monkeypatch):
    lib = library(tmp_path)
    r = relearn.Relearn(lib.find("dots and boxes", include_hidden=True), EXPLANATION)
    answers = iter(["Who goes first?", "Who goes first, then?", {"not": "a string"}])
    monkeypatch.setattr(r, "_model", lambda prompt, key: next(answers))
    assert r.ask() == "Who goes first?"
    r.answer("You can.")
    second = r.ask()                       # a near repeat is rejected in favour of a fixed question
    assert second != "Who goes first, then?" and second.endswith("?")
    assert r.used_fallback == 1


def test_name_alone_and_accumulated_text_recognise_the_hidden_game(tmp_path):
    lib = library(tmp_path)
    named = lib.find("dots and boxes", include_hidden=True)
    assert named is not None and lib.is_hidden(named)
    first, second = "Thank you.", "You join two dots with a line; the fourth side of a box claims it with your initial."
    assert relearn.recognise(lib, first) is None
    assert relearn.recognise(lib, first + " " + second).name == "dots and boxes"
