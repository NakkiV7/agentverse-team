"""Local hangman rules. Words and hints come from this list, never a language model."""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field


# Familiar nouns, four to seven letters. This is also the allowed dictionary.
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


def letter(value: str | None) -> str | None:
    """Return one ASCII capital, or None. Do not guess a letter from a phrase."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value.upper() if len(value) == 1 and value in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" else None


@dataclass
class Hangman:
    word: str = field(default_factory=lambda: random.choice(WORDS))
    guessed: set[str] = field(default_factory=set, init=False)
    max_misses: int = field(default=6, init=False)

    def __post_init__(self):
        self.word = self.word.upper()
        if self.word not in DICTIONARY:
            raise ValueError("Choose a word from the built-in dictionary.")

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
        return tuple(i for i, got in enumerate(self.word) if got == c)

    def result(self) -> str | None:
        """Result from the player's side: win, loss, or still playing."""
        if set(self.word) <= self.guessed:
            return "win"
        if self.misses >= self.max_misses:
            return "loss"
        return None

    def hint(self) -> str | None:
        """Most frequent unused letter among words fitting the public clues.

        Each word votes once per letter. Ties use alphabetical order. The secret
        word is used only for its already revealed pattern, never to pick a hint.
        """
        if self.result() is not None:
            return None
        candidates = [w for w in WORDS if len(w) == len(self.word)
                      and "".join(c if c in self.guessed else "_" for c in w) == self.pattern]
        counts = Counter(c for w in candidates for c in set(w) - self.guessed)
        return min(counts, key=lambda c: (-counts[c], c)) if counts else None
