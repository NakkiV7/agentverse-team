"""Hangman accepts spoken letters only after known ink is checked by the camera."""
import json
import threading
from types import SimpleNamespace

import pytest

from buddy.agent import brain as brain_module
from buddy.agent import llm
from buddy.agent.brain import Brain
from buddy.agent.llm import Understanding
from buddy.game.hangman import Hangman
from buddy.game.library import Library


class Paper:
    def __init__(self):
        self.lines = []
        self.messages = []
        self.cam = None
        self.visible = True
        self.coverage = 1.0
        self.reachable = True
        self.motion_ok = True
        self.draw_hook = None
        self.inked = 0.0

    def trace(self, *args):
        pass

    def transcript_add(self, who, text):
        self.messages.append(text)

    def say(self, text):
        pass

    def look(self, **kwargs):
        if not self.visible:
            return None
        return SimpleNamespace(rgb=None, sheet=SimpleNamespace(cut_off=False, confidence=1),
                               track=SimpleNamespace(pose=(.21, 0, 1.5708), visible=True, stable=True))

    def view_clear(self):
        return self.visible

    def grid_reachable(self, pose):
        return self.reachable, "test reach check"

    def draw(self, name, strokes):
        self.lines.append((name, strokes))
        if self.draw_hook:
            self.draw_hook()
        done = threading.Event()
        done.set()
        return SimpleNamespace(done=done, ok=self.motion_ok, error="test motion failure")


@pytest.fixture
def brain(tmp_path, monkeypatch):
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    b = Brain(Paper())
    b.level_asked = True          # these tests start games directly; the difficulty question is covered elsewhere
    b.library = Library(tmp_path / "games.json")
    b._select(b.library.find("hangman"))
    b.rng = SimpleNamespace(choice=lambda words: "APPLE")

    def run(fn, *args):
        try:
            fn(*args)
        finally:
            b.busy = False

    b._worker = run
    monkeypatch.setattr(brain_module, "check_lines", lambda rgb, cam, pose, lines: [b.app.coverage] * len(lines))
    monkeypatch.setattr(brain_module, "ink_fraction", lambda *args: b.app.inked)
    return b


def act(b, intent="say_letter", letter="A"):
    b._act(Understanding(intent, "none", "none", None, 0, letter=letter), letter or "", b.situation())


def setup_apple(b):
    b.start_setup("test")
    b.hangman = Hangman("APPLE")  # Same five dashes; a deterministic test answer.


def test_setup_routes_to_hangman_and_rules(brain):
    brain.start_setup("test")
    assert brain.state == "HANGMAN_TURN"
    assert len(brain.app.lines[0][1]) == len(brain.hangman.word)
    assert "six misses" in brain.rules_text()
    assert "one letter" in brain.situation()
    assert brain.spec is None and brain.solver is None


def test_reveal_all_matches_duplicate_hint_and_record_win(brain):
    setup_apple(brain)
    act(brain, letter="P")
    assert brain.hangman.pattern == "_PP__"
    assert len(brain.app.lines[-1][1]) == 2  # P is one polyline, drawn in two slots.
    drawn = len(brain.app.lines)
    act(brain, letter="P")
    act(brain, letter=None)
    act(brain, "hangman_hint", None)
    assert len(brain.app.lines) == drawn and "common" in brain.last_said
    for c in "ALE":
        act(brain, letter=c)
    assert brain.state == "GAME_OVER" and brain.entry.player_wins == 1
    assert Library(brain.library.path).find("hangman").player_wins == 1
    assert any("The word is APPLE" in s for s in brain.app.messages)


def test_six_verified_misses_record_loss(brain):
    setup_apple(brain)
    for c in "BCDFGH":
        act(brain, letter=c)
    assert brain.hangman.misses == 6 and brain.state == "GAME_OVER"
    assert len(brain.app.lines) == 7 and brain.entry.robot_wins == 1
    assert any("The word was APPLE" in s for s in brain.app.messages)


@pytest.mark.parametrize("problem", ["visible", "reachable", "coverage"])
def test_no_guess_when_paper_is_uncertain(brain, problem):
    setup_apple(brain)
    setattr(brain.app, problem, False if problem != "coverage" else .4)
    act(brain, letter="A")
    assert not brain.hangman.guessed and len(brain.app.lines) == 1


def test_partial_drawing_retries_same_letter_without_committing(brain):
    setup_apple(brain)
    brain.app.draw_hook = lambda: setattr(brain.app, "coverage", .2)
    act(brain, letter="P")
    assert not brain.hangman.guessed and brain._hangman_pending[0] == "P"
    assert len(brain.app.lines) == 3  # Two attempts after setup.
    act(brain, letter="A")
    assert "still need to check P" in brain.last_said
    brain.app.draw_hook = None
    brain.app.coverage = 1.0
    act(brain, letter="P")
    assert brain.hangman.pattern == "_PP__" and brain._hangman_pending is None


def test_motion_failure_and_stop_do_not_count_a_guess(brain):
    setup_apple(brain)
    brain.app.motion_ok = False
    act(brain, letter="B")
    assert brain.hangman.misses == 0 and brain.entry.plays == 0
    brain.app.motion_ok = True
    brain.app.draw_hook = brain.end_session
    act(brain, letter="B")
    assert brain.state == "IDLE" and not brain.hangman.guessed
    assert brain.entry.plays == 0


def test_setup_asks_for_clear_blank_reachable_sheet(brain):
    brain.app.visible = False
    brain.start_setup("test")
    assert brain.state == "HANGMAN_SETUP" and not brain.app.lines
    brain.app.visible = True
    brain.app.inked = .02
    act(brain, "yes")
    assert "fresh sheet" in brain.last_said and not brain.app.lines
    brain.app.inked = 0
    act(brain, "yes")
    assert brain.state == "HANGMAN_TURN"


@pytest.mark.parametrize("value, expected", [("B", "B"), ("none", None), ("AB", None), (None, None)])
def test_schema_and_letter_parsing(monkeypatch, value, expected):
    monkeypatch.setattr(llm, "_chat", lambda body: json.dumps({"intent": "say_letter", "letter": value}))
    u = llm.Conversation().understand("bee", "Hangman turn")
    assert u.letter == expected
    assert "say_letter" in llm.INTENTS and "letter" in llm.SCHEMA["required"]
    assert llm.SCHEMA["properties"]["letter"]["enum"] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["none"]


@pytest.mark.parametrize("text", ["A or E", "bee and sea", "A, E", "B/C"])
def test_model_cannot_choose_between_multiple_letters(monkeypatch, text):
    monkeypatch.setattr(llm, "_chat", lambda body: json.dumps({"intent": "say_letter", "letter": "A"}))
    u = llm.Conversation().understand(text, "Hangman turn")
    assert u.intent == "unclear" and u.letter is None


def test_library_keeps_existing_grid_paths_and_hangman_history(tmp_path):
    lib = Library(tmp_path / "games.json")
    assert lib.find("noughts and crosses").kind == "grid"
    assert "nine squares" in lib.find("noughts and crosses").describe()
    entry = lib.find("let's play hangman")
    assert entry.spec is None and entry.kind == "hangman"
    lib.record(entry, "loss")
    assert Library(lib.path).find("hang man").robot_wins == 1
