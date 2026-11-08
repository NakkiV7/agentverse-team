"""The tool executor (brain.run_tools) with a fake app: each tool reaches the right method, is refused outside its
state, an unknown game asks to be taught, a taken square is refused. Plus the planner's parsing and, when Ollama and
gemma4 are there, one live plan() call."""
import json
import threading
import urllib.request
from types import SimpleNamespace

import pytest

from buddy import config as C
from buddy.agent import brain as brain_module
from buddy.agent import toolcall
from buddy.agent.brain import Brain
from buddy.agent.toolcall import ToolCall
from buddy.game.library import Library
from buddy.game.hangman import Hangman


class Paper:
    """The parts of App the brain touches; drawing succeeds at once and nothing is read from a camera."""

    def __init__(self):
        self.lines = []
        self.messages = []
        self.traces = []
        self.cam = None
        self.last_strokes = None

    def trace(self, stage, msg):
        self.traces.append(f"{stage}: {msg}")

    def transcript_add(self, who, text):
        self.messages.append((who, text))

    def say(self, text):
        return None

    def look(self, **kwargs):
        return SimpleNamespace(rgb=None, sheet=SimpleNamespace(cut_off=False, confidence=1),
                               track=SimpleNamespace(pose=(.21, 0, 1.5708), visible=True, stable=True))

    def view_clear(self):
        return True

    def grid_reachable(self, pose):
        return True, "test reach check"

    def board_snapshot(self, **kwargs):
        return None

    def draw(self, name, strokes, **kwargs):
        self.lines.append((name, strokes))
        self.last_strokes = (name, strokes)
        done = threading.Event()
        done.set()
        return SimpleNamespace(done=done, ok=True, error="")


@pytest.fixture
def brain(tmp_path, monkeypatch):
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(toolcall, "ENABLED", False)       # no model in these tests
    b = Brain(Paper())
    b.level_asked = True
    b.library = Library(tmp_path / "games.json")
    b._select(b.library.find("noughts and crosses"))

    def run(fn, *args):
        try:
            fn(*args)
        finally:
            b.busy = False

    b._worker = run
    monkeypatch.setattr(brain_module, "check_lines", lambda rgb, cam, pose, lines: [1.0] * len(lines))
    monkeypatch.setattr(brain_module, "ink_fraction", lambda *args: 0.0)
    b.state = "CHAT"
    return b


def run(b, tool, text="", **args):
    b.run_tools([ToolCall(tool, args)], text)


def in_grid_game(b, player="X", board=None):
    b.player_sym, b.robot_sym = player, "O" if player == "X" else "X"
    b.board = board or [""] * 9
    b.state = "PLAYER_TURN"


# ---------------------------------------------------------------- routing
def test_list_games_offers_the_menu(brain):
    run(brain, "list_games")
    assert brain.state == "CHOOSE_GAME" and "noughts and crosses" in brain.last_said


def test_switch_game_starts_the_setup_of_a_known_game(brain, monkeypatch):
    started = []
    monkeypatch.setattr(brain, "start_setup", lambda why: started.append(why))
    run(brain, "switch_game", "can we play hangman", name="hangman")
    assert brain.entry.name == "hangman" and started


def test_switch_game_without_a_game_in_the_words_offers_the_menu(brain):
    """"Buddy, let's play a game": the model may guess a game, the brain offers the menu as it always did."""
    run(brain, "switch_game", "Buddy, let's play a game", name="noughts and crosses")
    assert brain.state == "CHOOSE_GAME" and "Shall we play" in brain.last_said
    brain.state = "GAME_OVER"
    brain.games = 1
    brain.app.traces.clear()
    run(brain, "switch_game", "let's play again", name="noughts and crosses")
    assert any("start noughts and crosses again" in t for t in brain.app.traces)


def test_draw_my_line_tells_rows_and_columns_apart_and_knows_corners(brain, monkeypatch):
    from buddy.game import dotsboxes as D
    brain._select(brain.library.find("dots and boxes"))
    brain.dots = brain_module.DotsPlay(brain, (2, 3))
    brain.dots.state = D.State.start((2, 3), ("P", "B"), first="P")
    brain.state = "DOTS_PLAYER_TURN"
    drawn = []
    monkeypatch.setattr(brain.dots, "_draw_for_player", lambda e: drawn.append(D.edge_name((2, 3), e)))
    run(brain, "draw_my_line", "put my line on the left side of the box in row 1, column 3",
        where="left side of the box in row 1, column 3")
    run(brain, "draw_my_line", "the top side of the top left box", where="top side of the top left box")
    run(brain, "draw_my_line", "the bottom of the bottom right box", where="bottom side of the bottom right box")
    assert drawn == ["the left side of the box in row 1, column 3", "the top side of the box in row 1, column 1",
                     "the bottom side of the box in row 2, column 3"]
    brain.dots.state = D.play(brain.dots.state, 0)              # the top side of row 1, column 1 is now drawn
    run(brain, "draw_my_line", where="the top side of the top left box")
    assert "already drawn" in brain.last_said and len(drawn) == 3


def test_unknown_game_asks_to_be_taught(brain):
    run(brain, "switch_game", "let's play snakes and ladders", name="snakes and ladders")
    assert brain.state == "TEACH" and brain.teacher is not None


def test_switch_game_trusts_the_pick_over_the_rejected_game_in_the_words(brain):
    """While noughts and crosses is being set up, "not noughts and crosses, dots and boxes instead" changes game."""
    brain.entry = brain.library.find("noughts and crosses")
    brain.state = "SETUP"
    brain.busy = True
    run(brain, "switch_game", "No wait, let's not play noughts and crosses, let's play dots and boxes instead.",
        name="dots and boxes")
    assert brain.switch_to is not None and brain.switch_to.name == "dots and boxes"
    brain.busy = False
    brain.switch_to = None
    run(brain, "switch_game", "let's play noughts and crosses", name="noughts and crosses")
    assert "what we're setting up" in brain.last_said


def test_switch_mid_game_waits_for_the_stroke(brain):
    in_grid_game(brain)
    brain.busy = True
    run(brain, "switch_game", name="dots and boxes")
    assert brain.switch_to is not None and brain.switch_to.name == "dots and boxes"
    assert "finish this stroke" in brain.last_said


def test_set_difficulty_authentic_is_adaptive(brain):
    run(brain, "set_difficulty", level="authentic")
    assert brain.difficulty.level == "adaptive" and brain.level_asked
    run(brain, "set_difficulty", level="hard")
    assert brain.difficulty.level == "hard"
    run(brain, "set_difficulty", level="silly")
    assert "Easy, medium" in brain.last_said and brain.difficulty.level == "hard"


def test_set_difficulty_answers_the_level_question(brain, monkeypatch):
    started = []
    monkeypatch.setattr(brain, "_start_setup", lambda why: started.append(why))
    brain.state = "ASK_LEVEL"
    brain._setup_after_level = "the player chose noughts and crosses"
    run(brain, "set_difficulty", level="easy")
    assert started == ["the player chose noughts and crosses"] and brain.difficulty.level == "easy"


def test_new_board_takes_the_size(brain, monkeypatch):
    brain._select(brain.library.find("dots and boxes"))
    brain.start_setup("test")                           # no size said yet: asks
    assert brain.state == "DOTS_SIZE"
    monkeypatch.setattr(brain.dots, "setup", lambda: setattr(brain, "state", "DOTS_SETUP"))
    run(brain, "new_board", size="three by two")
    assert brain.dots.n == (2, 3) and brain.state == "DOTS_SETUP"


def test_new_board_too_big_is_declined(brain):
    brain._select(brain.library.find("dots and boxes"))
    brain.start_setup("test")
    run(brain, "new_board", size="ten by ten")
    assert brain.state == "DOTS_SIZE" and "sorry" in brain.last_said.lower()


def test_size_and_level_said_with_the_game_are_kept_for_the_setup(brain, monkeypatch):
    """"Let's do dots and boxes, a small one, and go easy": three tools from CHOOSE_GAME, the size kept across the
    level step."""
    brain.state = "CHOOSE_GAME"
    brain.level_asked = False
    monkeypatch.setattr(brain_module.DotsPlay, "setup", lambda self: None)
    brain.run_tools([ToolCall("switch_game", {"name": "dots and boxes"}), ToolCall("new_board", {"size": "small"}),
                     ToolCall("set_difficulty", {"level": "easy"})], "let's do dots and boxes, a small one, and go easy")
    assert brain.entry.name == "dots and boxes" and brain.difficulty.level == "easy"
    assert brain.dots.n == (2, 2) and "2 across" in brain.last_said and brain._size_hint is None


def test_answer_routes_through_the_state_machine(brain, monkeypatch):
    brain.state = "CHOOSE_GAME"
    started = []
    monkeypatch.setattr(brain, "start_setup", lambda why: started.append(why))
    run(brain, "answer", yes=True)
    assert started
    brain.state = "CHOOSE_GAME"
    run(brain, "answer", yes=False)
    assert brain.state == "IDLE"


def test_answer_confirms_a_mark(brain, monkeypatch):
    in_grid_game(brain)
    brain.state = "CONFIRM_MARK"
    brain.pending = (4, "X")
    monkeypatch.setattr(brain, "_robot_move", lambda: None)
    run(brain, "answer", yes="true")
    assert brain.board[4] == "X" and brain.pending is None


def test_choose_side_starts_the_game_or_is_remembered(brain, monkeypatch):
    brain.state = "ASK_SIDE"
    monkeypatch.setattr(brain, "_robot_move", lambda: None)
    run(brain, "choose_side", side="noughts")
    assert brain.player_sym == "O" and brain.state == "ROBOT_TURN"
    brain.state = "SETUP"
    run(brain, "choose_side", side="crosses")
    assert brain._side_hint == "X" and "Crosses for you" in brain.last_said


def test_draw_my_mark_draws_a_free_square(brain, monkeypatch):
    in_grid_game(brain)
    monkeypatch.setattr(brain, "_draw_for_player", lambda cell: brain.app.lines.append(("mark", cell)))
    run(brain, "draw_my_mark", cell="top left")
    assert brain.app.lines == [("mark", 0)]
    run(brain, "draw_my_mark", "put it in the bottom right", cell="somewhere")
    assert brain.app.lines[-1] == ("mark", 8)


def test_draw_my_mark_refuses_a_taken_square(brain, monkeypatch):
    in_grid_game(brain, board=["O", "", "", "", "", "", "", "", ""])
    monkeypatch.setattr(brain, "_draw_for_player", lambda cell: brain.app.lines.append(("mark", cell)))
    run(brain, "draw_my_mark", cell="top left")
    assert not brain.app.lines and "already taken" in brain.last_said


def test_draw_my_mark_refuses_an_illegal_square_in_a_taught_game(brain, monkeypatch):
    brain._select(brain.library.find("footsteps"))
    in_grid_game(brain, board=["X", "", "", "", "", "", "", "", ""])
    from buddy.game.rules import State
    brain.gstate = State(tuple(brain.board), 0, "X")
    monkeypatch.setattr(brain, "_draw_for_player", lambda cell: brain.app.lines.append(("mark", cell)))
    run(brain, "draw_my_mark", cell="bottom right")
    assert not brain.app.lines and "next to the last mark" in brain.last_said


def test_tools_outside_their_state_are_refused_kindly(brain, monkeypatch):
    monkeypatch.setattr(brain, "_voice_move", lambda cell: brain.app.lines.append(cell))
    for name, args in [("draw_my_mark", {"cell": "centre"}), ("say_letter", {"letter": "E"}),
                       ("draw_my_line", {"where": "top side of the top left box"}), ("tell_score", {}),
                       ("give_hint", {}), ("start_again", {}), ("i_am_done", {})]:
        brain.state = "CHAT"
        brain.last_said = ""
        run(brain, name, **args)
        assert "no game on" in brain.last_said, name
    assert not brain.app.lines
    assert any("refused" in t for t in brain.app.traces)


def test_say_letter_goes_through_hangman(brain, monkeypatch):
    brain._select(brain.library.find("hangman"))
    brain.hangman = Hangman("APPLE")
    brain.state = "HANGMAN_TURN"
    brain._hangman_strokes = []
    monkeypatch.setattr(brain, "_hangman_paper_ready", lambda: True)
    monkeypatch.setattr(brain, "_hangman_draw_checked", lambda name, strokes: True)
    run(brain, "say_letter", "the letter P", letter="p")
    assert brain.hangman.pattern == "_PP__"
    run(brain, "say_letter", "A or E", letter="A")
    assert "One letter at a time" in brain.last_said and "A" not in brain.hangman.guessed


def test_redraw_last_stroke(brain):
    in_grid_game(brain)
    run(brain, "redraw_last_stroke")
    assert "nothing" in brain.last_said and not brain.app.lines
    brain.app.last_strokes = ("my nought", [[(0, 0), (1, 1)]])
    run(brain, "redraw_last_stroke")
    assert brain.app.lines == [("my nought", [[(0, 0), (1, 1)]])]


def test_wait_repeat_and_say(brain):
    brain.state = "CHAT"
    brain.run_tools([ToolCall("say", {"text": "Lovely to see you."})])
    assert brain.last_said == "Lovely to see you."
    run(brain, "wait")
    assert brain.last_said == "Of course, take your time."
    brain.last_said = "Shall we play?"
    run(brain, "repeat")
    assert brain.last_said == "Shall we play?"


def test_score_hint_and_why_have_built_in_lines(brain):
    in_grid_game(brain, board=["", "", "", "", "O", "", "", "", ""])
    run(brain, "tell_score")
    assert "your turn" in brain.last_said
    run(brain, "explain_last_move")
    assert "haven't" in brain.last_said      # the narration module's line, or the built-in one
    brain.last_choice = SimpleNamespace(cell=4, reason="best move (100% strength)")
    run(brain, "explain_last_move")
    assert "the centre" in brain.last_said and "best move" in brain.last_said
    run(brain, "give_hint")
    assert "You could try the" in brain.last_said
    assert brain.board.count("X") == 0                  # a hint never plays


def test_stop_from_the_menu_and_from_a_game(brain):
    brain.state = "CHOOSE_GAME"
    run(brain, "stop")
    assert brain.state == "IDLE"
    in_grid_game(brain)
    run(brain, "stop")
    assert brain.state == "IDLE" and "lovely" in brain.last_said.lower()


def test_two_tools_run_in_order(brain, monkeypatch):
    brain.state = "CHOOSE_GAME"
    started = []
    monkeypatch.setattr(brain, "start_setup", lambda why: started.append(why))
    brain.run_tools([ToolCall("set_difficulty", {"level": "easy"}), ToolCall("switch_game", {"name": "hangman"})],
                    "go easy on me and let's play hangman")
    assert brain.difficulty.level == "easy" and brain.entry.name == "hangman" and started
    assert any("tool" in t or "difficulty" in t for t in brain.app.traces)


# ---------------------------------------------------------------- the hook
def test_hear_uses_the_planner_first_and_falls_back(brain, monkeypatch):
    brain.state = "CHAT"
    monkeypatch.setattr(brain.planner, "plan", lambda text, sit, state: [ToolCall("list_games")])
    brain._on_text("what can we play", "typed", 0.0)
    assert brain.state == "CHOOSE_GAME"
    monkeypatch.setattr(brain.planner, "plan", lambda text, sit, state: None)
    understood = []
    monkeypatch.setattr(brain.conv, "understand", lambda text, sit: understood.append(text) or
                        brain._u("list_games"))
    brain._on_text("what can we play", "typed", 0.0)
    assert understood == ["what can we play"]


def test_size_answer_wins_without_a_model_call(brain, monkeypatch):
    brain._select(brain.library.find("dots and boxes"))
    brain.start_setup("test")
    assert brain.state == "DOTS_SIZE"
    monkeypatch.setattr(brain.planner, "plan", lambda *a: pytest.fail("the model was asked"))
    monkeypatch.setattr(brain.dots, "setup", lambda: setattr(brain, "state", "DOTS_SETUP"))
    brain._on_text("three by two please", "typed", 0.0)
    assert brain.dots.n == (2, 3) and brain.state == "DOTS_SETUP"


# ---------------------------------------------------------------- the planner's parsing
def test_parse_native_text_and_garbled_calls():
    p = toolcall.parse_message
    assert p({"tool_calls": [{"function": {"name": "switch_game", "arguments": {"name": "hangman"}}}]}, "CHAT") \
        == [ToolCall("switch_game", {"name": "hangman"})]
    assert p({"tool_calls": [{"function": {"name": "answer", "arguments": '{"yes": true}'}}]}, "CHAT") \
        == [ToolCall("answer", {"yes": True})]
    assert p({"content": 'switch_game(name="dots and boxes")\nnew_board(size="small")'}, "CHOOSE_GAME") \
        == [ToolCall("switch_game", {"name": "dots and boxes"}), ToolCall("new_board", {"size": "small"})]
    assert p({"content": 'new_board{size:<|"|>three by two<|"|>}<tool_call|>'}, "DOTS_SIZE") \
        == [ToolCall("new_board", {"size": "three by two"})]
    assert p({"content": "Hello there, shall we play?"}, "CHAT") == [ToolCall("say", {"text": "Hello there, shall we play?"})]
    assert p({"content": "stop"}, "CHAT") == [ToolCall("stop")]
    assert p({"content": "switch_game hangman"}, "CHAT") == [ToolCall("switch_game", {"name": "hangman"})]
    assert p({"content": "say_letter: E"}, "HANGMAN_TURN") == [ToolCall("say_letter", {"letter": "E"})]
    assert p({"content": "answer yes"}, "CHOOSE_GAME") == [ToolCall("answer", {"yes": True})]
    assert p({"content": "wait, I need to think about that one."}, "CHAT") == [ToolCall("say", {"text": "wait, I need to think about that one."})]
    assert p({"tool_calls": [{"function": {"name": "launch_rocket", "arguments": {}}}]}, "CHAT") is None
    assert p({"content": ""}, "CHAT") is None


def test_planner_returns_none_when_the_server_is_down(monkeypatch):
    monkeypatch.setattr(toolcall, "ENABLED", True)
    monkeypatch.setattr(C, "OLLAMA_URL", "http://127.0.0.1:9")
    pl = toolcall.Planner()
    assert pl.plan("hello", "Chatting.", "CHAT") is None
    assert pl.plan("hello", "Teaching.", "TEACH") is None


def _gemma4_up() -> bool:
    try:
        with urllib.request.urlopen(f"{C.OLLAMA_URL}/api/tags", timeout=2) as r:
            names = [m["name"] for m in json.loads(r.read())["models"]]
        return any(n.split(":")[0] == toolcall.MODEL.split(":")[0] for n in names)
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _gemma4_up(), reason="needs Ollama with gemma4")
def test_live_plan_picks_switch_game():
    calls = toolcall.Planner().plan("let's play hangman instead",
                                    "Dots and boxes: it is the player's turn. Games Buddy knows: noughts and crosses, "
                                    "hangman and dots and boxes.", "DOTS_PLAYER_TURN")
    assert calls and calls[0].name == "switch_game" and "hangman" in str(calls[0].args.get("name", "")).lower()
