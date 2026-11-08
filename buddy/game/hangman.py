"""Local hangman rules. Words and hints come from this list, never a language model."""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field


# Familiar nouns, four to seven letters. Keep the original medium hint dictionary.
WORDS = tuple(sorted("""
apple apron basket beach beans bedroom bicycle biscuit blanket bottle bread
bridge broom butter button cabbage candle carrot carpet castle chair cheese
cherry clock cloud coffee cushion daisy dinner donkey drawer dress flower
forest garden glove grapes hammer harbour horse house island jacket kettle
kitchen ladder lemon letter library market meadow mirror monkey mother
needle orange orchard packet palace paper peach pencil pepper
piano picnic pillow plate pocket potato rabbit radio ribbon river robin
salad school sheep shirt shoe shop sister spoon spring square station
stone street sugar summer table teapot ticket tomato towel tractor train
tree tulip turtle uncle valley village violin wallet water window winter
""".upper().split()))
DICTIONARY = frozenset(WORDS)


WORDS_BY_LEVEL = {
    "easy": tuple(w for w in WORDS if 4 <= len(w) <= 5),
    "medium": WORDS,
    "hard": tuple(w for w in WORDS if 6 <= len(w) <= 7),
    # Seven letters meet the hardest level and still fit the existing paper layout.
    "impossible": ("AXOLOTL", "AZIMUTH", "JACAMAR", "QUETZAL", "XYLITOL", "ZEPHYRS", "ZYZZYVA"),
}
_ALL_WORDS = frozenset(w for words in WORDS_BY_LEVEL.values() for w in words)


def level_for_difficulty(level, strength: float | None = None) -> str:
    """Resolve a level name or brain.difficulty without changing its rating.

    Pass the difficulty object for adaptive play so its current strength is used.
    A bare adaptive/authentic name uses medium until a strength is available.
    """
    strength = getattr(level, "strength", .6 if strength is None else strength)
    name = getattr(level, "level", level)
    if name in ("adaptive", "authentic"):
        return "easy" if strength < .4 else "medium" if strength < .7 else "hard" if strength < .9 else "impossible"
    if name not in WORDS_BY_LEVEL:
        raise ValueError(f"Unknown hangman difficulty: {name}")
    return name


def word_for_level(level, rng) -> str:
    """Choose locally; level may be a fixed name or brain.difficulty."""
    return rng.choice(WORDS_BY_LEVEL[level_for_difficulty(level)])


def letter(value: str | None) -> str | None:
    """Return one ASCII capital, or None. Do not guess a letter from a phrase."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value.upper() if len(value) == 1 and value in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" else None


@dataclass
class Hangman:
    word: str = "APPLE"
    level: str = "medium"
    guessed: set[str] = field(default_factory=set, init=False)
    max_misses: int = field(default=6, init=False)
    last_guess: str | None = field(default=None, init=False)

    def __post_init__(self):
        self.word = self.word.upper()
        self.level = level_for_difficulty(self.level)
        if self.word not in _ALL_WORDS:
            raise ValueError("Choose a word from the built-in dictionary.")

    @classmethod
    def for_difficulty(cls, difficulty, rng=random) -> "Hangman":
        """Start with the brain's chosen level; keep that word through drawing retries."""
        level = level_for_difficulty(difficulty)
        return cls(word_for_level(level, rng), level=level)

    @property
    def misses(self) -> int:
        return len(self.guessed - set(self.word))

    @property
    def remaining(self) -> int:
        return self.max_misses - self.misses

    @property
    def pattern(self) -> str:
        return "".join(c if c in self.guessed else "_" for c in self.word)

    def legal_letter(self, value: str | None) -> bool:
        c = letter(value)
        return c is not None and c not in self.guessed and self.result() is None

    def guess(self, value: str) -> tuple[int, ...]:
        """Apply a legal guess and return every matching position (zero based)."""
        if not self.legal_letter(value):
            raise ValueError("Say one unused letter from A to Z while the game is on.")
        c = letter(value)
        self.guessed.add(c)
        self.last_guess = c
        return tuple(i for i, got in enumerate(self.word) if got == c)

    def result(self) -> str | None:
        """Result from the player's side: win, loss, or still playing."""
        if set(self.word) <= self.guessed:
            return "win"
        if self.misses >= self.max_misses:
            return "loss"
        return None

    def hint(self, difficulty=None) -> str | None:
        """A letter, a vowel/consonant category, or no hint at the chosen level.

        Medium keeps the original public-clue frequency hint. Easy names a letter
        in the answer; hard gives only a category; impossible offers no hints.
        An optional brain.difficulty applies a spoken level change immediately.
        """
        if self.result() is not None:
            return None
        level = level_for_difficulty(self.level if difficulty is None else difficulty)
        if level == "impossible":
            return None
        if level == "easy":
            return min(set(self.word) - self.guessed)
        vocabulary = WORDS if level == "medium" else _ALL_WORDS
        candidates = [w for w in vocabulary if len(w) == len(self.word)
                      and "".join(c if c in self.guessed else "_" for c in w) == self.pattern]
        counts = Counter(c for w in candidates for c in set(w) - self.guessed)
        hint = min(counts, key=lambda c: (-counts[c], c)) if counts else None
        if hint is not None and level == "hard":
            return "a vowel" if hint in "AEIOU" else "a consonant"
        return hint
