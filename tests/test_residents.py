"""Resident introductions, live voice pace, teacher memory and the video's state contract."""
import json
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from buddy.agent.brain import Brain
from buddy.agent.residents import context_caption, load_resident, residents
from buddy.app import App
from buddy.game.library import Library
from buddy.game.rules import GameSpec
from buddy.voice.tts import PiperEngine, Speaker


@pytest.fixture
def brain(tmp_path, monkeypatch):
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    monkeypatch.setenv("BUDDY_GAMES_FILE", str(tmp_path / "games.json"))
    app = Mock()
    app.say.return_value = None
    b = Brain(app)
    b._plan_tools = lambda *args: []
    return b


@pytest.mark.parametrize("name,level,scale", [("Margaret", "easy", 1.1), ("Arthur", "hard", 0.95),
                                              ("Doreen", "medium", 0.95)])
@pytest.mark.parametrize("phrasing", ["Penny, I'm {}", "Penny, It's {}", "Penny, My name is {}", "Penny, I’m {}"])
def test_introductions_apply_profile_before_greeting(brain, name, level, scale, phrasing):
    brain._on_text(phrasing.format(name), "typed", 0)
    assert brain.player_name == brain.resident.name == name
    assert brain.difficulty.level == level and brain.level_asked
    brain.app.speaker.set_length_scale.assert_called_once_with(scale)
    assert brain.last_said.startswith(f"Hello {name}.")
    assert brain.game_name == brain.resident.favourite_game
    assert brain.state == "CHOOSE_GAME"
    facts = brain.facts()
    assert f"Resident: {name}, age {brain.resident.age}" in facts
    assert f"way_to_play={brain.resident.way_to_play}" in facts
    assert len(facts) <= brain.FACTS_LIMIT


@pytest.mark.parametrize("name", ["Margaret", "Arthur", "Doreen"])
@pytest.mark.parametrize("question", ["Who am I talking to?", "What's your name?", "What shall I call you?"])
def test_bare_name_after_name_question(brain, name, question):
    brain.state = "CHAT"
    brain.say(question)
    brain._on_text(name.lower(), "mic", 0)
    assert brain.resident.name == name
    assert brain.last_said.startswith(f"Hello {name}.")


def test_names_do_not_bypass_wake_or_game_answers(brain):
    brain._on_text("I'm Margaret", "mic", 0)
    assert brain.resident is None and brain.state == "IDLE"
    brain.state = "CHAT"
    brain.say("Which game would you like?")
    brain._on_text("Arthur", "typed", 0)
    assert brain.resident is None


def test_unknown_name_keeps_existing_conversation(brain):
    brain._on_text("Penny, I'm Dorothy", "typed", 0)
    assert brain.player_name == "Dorothy" and brain.resident is None
    assert brain.state == "CHAT" and not brain.level_asked
    assert brain.difficulty.level == "adaptive"
    brain.app.speaker.set_length_scale.assert_not_called()


def test_greetings_and_remaining_preferences(brain):
    brain._on_text("Penny, I'm Margaret", "typed", 0)
    assert brain.last_said == "Hello Margaret. Shall I write for you again? Noughts and crosses, as usual?"
    brain._on_text("It's Arthur", "typed", 0)
    assert brain.last_said == "Hello Arthur. Dots and boxes, at your best?"
    assert brain.resident.captions
    brain._on_text("My name is Doreen", "typed", 0)
    assert brain.last_said == "Hello Doreen. Shall we share the writing? Hangman, as usual?"
    assert brain.resident.short_sessions
    assert "short_sessions=true" in brain.facts()


def test_greeting_yes_uses_favourite_without_level_question(brain):
    brain._on_text("Penny, I'm Margaret", "typed", 0)
    brain._start_setup = Mock()
    brain._tool_answer({"yes": True}, "yes")
    brain._start_setup.assert_called_once()
    assert brain.game_name == "noughts and crosses" and brain.state != "ASK_LEVEL"


@pytest.mark.parametrize("speed,scale", [("slow", 1.1), ("normal", 0.95), ("brisk", 0.85)])
def test_speaker_changes_piper_configuration_live(speed, scale):
    from piper import SynthesisConfig
    engine = PiperEngine.__new__(PiperEngine)
    engine.cfg = SynthesisConfig(length_scale=0.9)
    old_config = engine.cfg
    engine.voice = Mock()
    engine.voice.synthesize.return_value = [SimpleNamespace(audio_float_array=np.ones(4, dtype=np.float32))]
    speaker = Speaker.__new__(Speaker)
    speaker.engine = engine
    profile = replace(load_resident("Margaret"), voice_speed=speed)
    speaker.set_length_scale(profile.length_scale)
    speaker.synth("Hello Margaret.")
    assert engine.voice.synthesize.call_args.args[1].length_scale == scale
    assert old_config.length_scale == 0.9  # in-flight synthesis keeps its own config


def choose_footsteps(brain, name="Arthur", tool=True):
    brain._on_text(f"Penny, I'm {name}", "typed", 0)
    if tool:
        brain._tool_switch_game({"name": "Footsteps"}, "Let's play Footsteps")
    else:
        brain._act(brain._u("choose_game", game_name="Footsteps"), "Footsteps", brain.situation())


@pytest.mark.parametrize("tool", [False, True])
def test_non_teacher_gets_credit_then_rules_before_setup(brain, tool):
    choose_footsteps(brain, tool=tool)
    assert brain.state == "OFFER_TAUGHT"
    assert brain.last_said == "Margaret taught me this one. Would you like to learn it?"
    order = []
    brain.app.say.side_effect = lambda text: order.append(text)
    brain._start_setup = lambda why: order.append("setup")
    brain._tool_answer({"yes": True}, "yes")
    assert order == [brain.rules_text(), "setup"]
    assert "Margaret taught Penny Footsteps" in brain.facts()


def test_teacher_is_welcomed_back_to_their_game(brain):
    choose_footsteps(brain, "Margaret")
    assert brain.last_said == "Your game! Shall we play it?"
    brain._start_setup = Mock()
    brain.app.say.reset_mock()
    brain._tool_answer({"yes": True}, "yes")
    brain._start_setup.assert_called_once()
    brain.app.say.assert_not_called()


def test_declining_lesson_does_not_start_game(brain):
    choose_footsteps(brain)
    brain._start_setup = Mock()
    brain._tool_answer({"yes": False}, "no")
    assert brain.state == "CHOOSE_GAME"
    brain._start_setup.assert_not_called()
    brain._tool_switch_game({"name": "Footsteps"}, "Footsteps")
    assert "Margaret taught me" in brain.last_said


@pytest.mark.parametrize("answer", ["yes", "Yes please.", "Yes, I'd like to learn it.", "Sure, let’s play it."])
def test_accepting_a_lesson_does_not_depend_on_model_routing(brain, answer):
    choose_footsteps(brain)
    brain._start_setup = Mock()
    brain._plan_tools = Mock(side_effect=AssertionError("An explicit lesson answer needs no planner"))
    brain._on_text(answer, "typed", 0)
    brain._start_setup.assert_called_once()
    brain._plan_tools.assert_not_called()


def test_taught_offer_does_not_swallow_other_requests(brain):
    choose_footsteps(brain)
    brain._plan_tools = Mock(return_value=[])
    brain._on_text("Yes, but I want to teach you another game", "typed", 0)
    brain._plan_tools.assert_called_once()


def test_menu_and_builtin_credit_come_from_profiles(brain):
    assert load_resident("margaret").taught_games == ["Footsteps"]
    assert len(residents()) == 3
    assert brain.library.find("Footsteps").taught_by == "Margaret"
    assert "Footsteps (taught by Margaret)" in brain.offer_text()
    brain._tool_list_games({}, "menu")
    assert "Footsteps (taught by Margaret)" in brain.last_said


@pytest.mark.parametrize("named", [False, True])
def test_teaching_records_loaded_resident_and_survives_reload(brain, named):
    brain._on_text("Penny, I'm Margaret", "typed", 0)
    spec = GameSpec(placement="adjacent_to_last_mark", end="k_in_a_row_loses", name="Corners" if named else "")
    brain.state = "TEACH"
    draft = SimpleNamespace(**spec.__dict__, question="")
    turn = SimpleNamespace(draft=draft, done=True, spec=spec, say="I've understood your rules.", seconds=0)
    brain.teacher = SimpleNamespace(hear=lambda text: turn, model="test")
    brain._teach("We lose if we make a line of three next to the last mark.")
    if not named:
        assert brain.state == "NAME_GAME"
        brain._act(brain._u("smalltalk"), "Corners", brain.situation())
    assert brain.entry.name == "Corners" and brain.entry.taught_by == "Margaret"
    assert brain.last_said == "Your game! Shall we play it?"
    saved = json.loads(brain.library.path.read_text())["games"][0]
    assert saved["taught_by"] == "Margaret" and saved["taught_at"]
    reloaded = Library(brain.library.path)
    assert reloaded.find("Corners").taught_by == "Margaret"
    reloaded.remember(spec, "Another name", taught_by="Arthur")
    assert reloaded.find("Corners").taught_by == "Margaret"


def test_revealed_builtin_keeps_teacher_and_history(brain):
    brain._on_text("Penny, I'm Margaret", "typed", 0)
    brain.library.hidden.add("dots and boxes")
    brain._offer_relearned(brain.library.find("dots and boxes", include_hidden=True))
    assert brain.entry.taught_by == "Margaret"
    reloaded = Library(brain.library.path, hidden=["dots and boxes"])
    assert reloaded.find("dots and boxes").taught_by == "Margaret"
    assert reloaded.find("dots and boxes").taught_at


def app_state(brain):
    app = brain.app
    app.brain, app.tracker.track, app.board_snap = brain, None, None
    app._lock, app._trace, app._transcript = threading.Lock(), [], []
    app.speaker, app.listener = None, None
    app.ctl.paused, app.ctl.at_home, app.ctl.busy = False, True, False
    app.intrusion.pixels, app.intrusion.max_height_cm = 0, 0
    app.world.touch.return_value, app.sim_time = 0, 0
    return App.state(app)


def test_caption_and_resident_state_contract(brain):
    state = app_state(brain)
    assert state["caption"] == "Penny · resting" and state["resident"] is None
    choose_footsteps(brain)
    brain.difficulty.set_level("medium")
    brain.state = "PLAYER_TURN"
    state = app_state(brain)
    assert state["resident"] == {"name": "Arthur", "way_to_play": "i_write", "level": "best", "captions": True}
    assert state["caption"] == "Penny · playing Footsteps, taught by Margaret · with Arthur · level fair"
    brain._note_name("I'm Margaret")
    brain.state = "TEACH"
    assert app_state(brain)["caption"] == "Penny · learning a new game from Margaret"
    brain.state = "IDLE"
    assert context_caption(brain) == "Penny · resting"
