"""Penny wakes only when her name starts a sentence (a resident or carer may also be called Penny)."""
from buddy.agent.brain import has_wake_word


def test_name_at_the_start_wakes():
    for t in ("Penny", "Penny, let's play a game", "Hey Penny", "Okay, Penny, what can we play?",
              "Right. Penny, are you there?", "Hello there. Penny!", "Buddy, let's play"):
        assert has_wake_word(t), t


def test_name_inside_a_sentence_does_not_wake():
    for t in ("Tell Penny I said hello", "I was talking to Penny earlier", "My sister Penny is visiting",
              "Well, that's the pitch outline", "Everyone cheers."):
        assert not has_wake_word(t), t
