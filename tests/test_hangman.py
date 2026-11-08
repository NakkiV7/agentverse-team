import pytest

from buddy.game.hangman import DICTIONARY, WORDS, Hangman, letter


def test_dictionary_and_letter_validation():
    assert len(WORDS) == len(DICTIONARY) >= 80
    assert all(4 <= len(w) <= 7 and w.isascii() and w.isalpha() for w in WORDS)
    assert letter(" a ") == "A"
    for value in (None, "", "AB", "1", "é", "ß", "A please", "none"):
        assert letter(value) is None
    with pytest.raises(ValueError):
        Hangman("notaword")
    assert Hangman().word in DICTIONARY


def test_repeated_letters_and_win():
    game = Hangman("apple")
    assert game.pattern == "_____"
    assert game.guess("p") == (1, 2)
    assert game.pattern == "_PP__" and game.misses == 0
    assert not game.legal_letter("P")
    with pytest.raises(ValueError):
        game.guess("P")
    for c in "ALE":
        game.guess(c)
    assert game.result() == "win" and game.remaining == 6
    assert not game.legal_letter("Z") and game.hint() is None


def test_six_misses_lose_without_changing_on_bad_input():
    game = Hangman("apple")
    for n, c in enumerate("BCDFGH", 1):
        assert game.guess(c) == ()
        assert game.misses == n and game.remaining == 6 - n
        assert game.result() == ("loss" if n == 6 else None)
    with pytest.raises(ValueError):
        game.guess("A")
    assert game.pattern == "_____" and game.hint() is None


def test_hint_uses_public_pattern_and_word_frequency():
    a, b = Hangman("apple"), Hangman("chair")
    assert a.hint() == b.hint()  # Same length, same public clues.
    hint = a.hint()
    votes = {c: sum(c in w for w in WORDS if len(w) == 5) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    assert votes[hint] == max(votes.values())
    a.guess(hint)
    assert a.hint() not in a.guessed
    game = Hangman("apple")
    for c in "APLE":
        assert game.hint() not in game.guessed
        game.guess(c)
