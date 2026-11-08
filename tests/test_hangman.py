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


@pytest.mark.parametrize("level, low, high", [("easy", 4, 5), ("medium", 4, 7),
                                            ("hard", 6, 7), ("impossible", 7, 7)])
def test_word_pools_fit_difficulty_and_existing_paper(level, low, high):
    import random
    from buddy.game.hangman import WORDS_BY_LEVEL, word_for_level

    pool = WORDS_BY_LEVEL[level]
    assert pool and len(pool) == len(set(pool))
    assert all(low <= len(w) <= high and w.isascii() and w.isalpha() and w.isupper() for w in pool)
    if level == "impossible":
        assert not set(pool) & DICTIONARY
    first, second = random.Random(7), random.Random(7)
    assert [word_for_level(level, first) for _ in range(20)] == [word_for_level(level, second) for _ in range(20)]
    for word in pool:
        assert Hangman(word, level=level).word == word
    assert word_for_level(level, first) in pool
    assert Hangman().word == "APPLE" and Hangman().level == "medium"


@pytest.mark.parametrize("name", ["adaptive", "authentic"])
@pytest.mark.parametrize("strength, expected", [(0, "easy"), (.3999, "easy"), (.4, "medium"),
                                                (.6999, "medium"), (.7, "hard"), (.8999, "hard"),
                                                (.9, "impossible"), (1, "impossible")])
def test_adaptive_word_and_hint_level_boundaries(name, strength, expected):
    import random
    from types import SimpleNamespace
    from buddy.game.hangman import WORDS_BY_LEVEL, level_for_difficulty, word_for_level

    difficulty = SimpleNamespace(level=name, strength=strength)
    assert level_for_difficulty(difficulty) == expected
    assert word_for_level(difficulty, random.Random(5)) in WORDS_BY_LEVEL[expected]
    game = Hangman.for_difficulty(difficulty, random.Random(5))
    assert game.level == expected and game.word in WORDS_BY_LEVEL[expected]
    assert difficulty.level == name and difficulty.strength == strength


@pytest.mark.parametrize("level", ["easy", "medium", "hard", "impossible"])
def test_fixed_difficulty_ignores_adaptive_strength(level):
    from types import SimpleNamespace
    from buddy.game.hangman import level_for_difficulty

    assert level_for_difficulty(SimpleNamespace(level=level, strength=.01)) == level


def test_easy_hint_names_an_unused_answer_letter_until_finished():
    from buddy.game.hangman import WORDS_BY_LEVEL

    for word in WORDS_BY_LEVEL["easy"]:
        game = Hangman(word, level="easy")
        while game.result() is None:
            hint = game.hint()
            assert hint in word and hint not in game.guessed
            game.guess(hint)
        assert game.result() == "win" and game.hint() is None


def test_hard_hints_only_give_categories_and_impossible_never_gives_hints():
    hard = Hangman("BASKET", level="hard")
    for c in "BASKET":
        assert hard.hint() in ("a vowel", "a consonant")
        hard.guess(c)
    assert hard.hint() is None
    game = Hangman("QUETZAL", level="impossible")
    for c in "QUETZAL":
        assert game.hint() is None
        game.guess(c)
    assert game.hint() is None


def test_letter_category_uses_public_clues_and_difficulty_can_change_midgame():
    from types import SimpleNamespace

    a, b = Hangman("BASKET", level="hard"), Hangman("BOTTLE", level="hard")
    assert a.hint() == b.hint()
    game = Hangman()
    game.guess("P")
    before = (game.word, game.level, game.pattern, game.guessed.copy(), game.last_guess)
    assert game.hint(SimpleNamespace(level="easy", strength=1)) in "ALE"
    assert game.hint("hard") in ("a vowel", "a consonant")
    assert game.hint("impossible") is None
    assert before == (game.word, game.level, game.pattern, game.guessed, game.last_guess)


def test_last_guess_is_preserved_when_a_letter_is_invalid():
    game = Hangman()
    assert game.last_guess is None
    game.guess("p")
    assert game.last_guess == "P"
    for bad in ("P", "AB", "?"):
        with pytest.raises(ValueError):
            game.guess(bad)
        assert game.last_guess == "P"


def test_unknown_level_is_rejected_and_bare_adaptive_names_default_to_medium():
    import random
    from buddy.game.hangman import level_for_difficulty, word_for_level

    with pytest.raises(ValueError):
        word_for_level("unknown", random.Random())
    with pytest.raises(ValueError):
        Hangman(level="unknown")
    assert level_for_difficulty("adaptive") == "medium"
    assert level_for_difficulty("authentic") == "medium"
