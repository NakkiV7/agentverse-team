"""brain.facts(): the full plain-sentence picture the language model answers questions from, built with a fake app
and no model. Also the planner's user message (facts in the newest message only, compact history) and the name
memory. The scenario builders here are also what tests/data/questions.jsonl was generated from
(tools/question_eval.py --dump-facts prints them again)."""
import threading
import time
from types import SimpleNamespace

import pytest

from buddy.agent import brain as brain_module
from buddy.agent import toolcall
from buddy.agent.brain import Brain
from buddy.agent.toolcall import ToolCall
from buddy.game import dotsboxes as D
from buddy.game.hangman import Hangman
from buddy.game.library import Library
from buddy.game.tictactoe import Choice


class Paper:
    """The parts of App the brain touches; nothing is drawn or read."""

    def __init__(self):
        self.messages, self.traces = [], []
        self.cam = None
        self.last_strokes = None

    def trace(self, stage, msg):
        self.traces.append(f"{stage}: {msg}")

    def transcript_add(self, who, text):
        self.messages.append((who, text))

    def say(self, text):
        return None

    def grid_reachable(self, pose):
        return True, "test"

    def draw(self, name, strokes, **kwargs):
        done = threading.Event()
        done.set()
        return SimpleNamespace(done=done, ok=True, error="")


def make_brain(tmp_path, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr(threading.Thread, "start", lambda self: None)
        monkeypatch.setattr(toolcall, "ENABLED", False)
    b = Brain(Paper())
    b.level_asked = True
    b.library = Library(tmp_path / "games.json")
    b._select(b.library.find("noughts and crosses"))
    b.state = "CHAT"
    return b


# ---------------------------------------------------------------- scenarios (shared with the question eval)
def talk(b, player, buddy):
    b.emit("heard", "player", player)
    b.last_heard = player
    b.say(buddy)


def finished_game(b, game, result, number):
    """A game that ended earlier this session, as the events would record it."""
    b.emit("game_start", "buddy", f"game {number} of {game} started", game=game, number=number)
    e = b.library.find(game)
    b.library.record(e, result)
    words = {"win": "the player won", "loss": "Buddy won", "draw": "a draw"}[result]
    b.emit("game_over", "buddy", f"{game} over: {words}", game=game, result=result, plays=e.plays,
           player_wins=e.player_wins, robot_wins=e.robot_wins, draws=e.draws)


def hangman_scene(b, word="APPLE", guesses="PTESN", name="Margaret", earlier=True, minutes=4):
    """Game 2 of the session: APPLE with P, T, E, S, N tried (_PP_E, 3 misses), after a drawn noughts and crosses."""
    if earlier:
        finished_game(b, "noughts and crosses", "draw", 1)
    b._select(b.library.find("hangman"))
    b.difficulty.set_level("adaptive")
    b.difficulty.strength = 0.62
    b.hangman = Hangman(word)
    b.games = 2 if earlier else 1
    b.emit("game_start", "buddy", f"game {b.games} of hangman started", game="hangman", number=b.games)
    b.set_state("HANGMAN_TURN", "test")
    b.game_started_at = time.time() - 60 * minutes
    b.say("The dashes are ready. Say one letter whenever you are ready.")
    if name:
        talk(b, f"I'm {name}, by the way", f"Lovely to meet you, {name}. Say a letter when you're ready.")
        b.player_name = name
    for c in guesses:
        b.emit("heard", "player", c)
        positions = b.hangman.guess(c)
        right = bool(positions)
        b.emit("guess", "player", f"the player guessed {c}: {'right' if right else 'wrong'} (word so far "
               f"{' '.join(b.hangman.pattern)}, {b.hangman.misses} of 6 misses)", letter=c, correct=right,
               pattern=b.hangman.pattern, misses=b.hangman.misses, guessed=sorted(b.hangman.guessed),
               wrong=sorted(b.hangman.guessed - set(word)))
        if c == guesses[-2]:
            talk(b, "I think it might be a fruit", "You may be right. Say another letter.")
        b.say(f"{c} is in the word. {b.hangman.remaining} misses left. Say another letter." if right else
              f"There is no {c}. {b.hangman.remaining} misses left. Say another letter.")
    return b


def hangman_fresh_scene(b):
    """A fresh hard game: a seven-letter word, nothing guessed, no name known."""
    b._select(b.library.find("hangman"))
    b.difficulty.set_level("hard")
    b.hangman = Hangman("BICYCLE", level="hard")
    b.games = 1
    b.emit("game_start", "buddy", "game 1 of hangman started", game="hangman", number=1)
    b.set_state("HANGMAN_TURN", "test")
    b.game_started_at = time.time() - 20
    b.say("The dashes are ready. The word has 7 letters. Say one letter whenever you are ready.")
    return b


def grid_scene(b, state="PLAYER_TURN"):
    """Game 3 of the session, noughts and crosses on easy: X top left, O centre, X bottom right, O top right."""
    finished_game(b, "noughts and crosses", "loss", 1)
    finished_game(b, "noughts and crosses", "win", 2)
    b.difficulty.set_level("easy")
    b.player_sym, b.robot_sym = "X", "O"
    b.games = 3
    b.emit("game_start", "buddy", "game 3 of noughts and crosses started", game="noughts and crosses", number=3)
    b.set_state(state, "test")
    b.game_started_at = time.time() - 90
    b.say("Crosses for you, and you go first.")
    b.board = ["X", "", "O", "", "O", "", "", "", "X"]
    for who, sym, cell, reason in (("player", "cross", "top left", None), ("buddy", "nought", "centre", "the strongest square"),
                                   ("player", "cross", "bottom right", None), ("buddy", "nought", "top right", "blocking your line")):
        b.emit("move", who, (f"Buddy put {sym} in the {cell} ({reason})" if who == "buddy" else f"the player put {sym} in the {cell}"),
               cells=[cell], symbol=sym, board=list(b.board), **({"reason": reason} if reason else {}))
    b.last_choice = Choice(2, [2], {2: 0, 1: -1}, True, "blocking your line")
    talk(b, "you're not making this easy", "I'm playing gently, I promise. Your turn.")
    return b


def grid_over_scene(b):
    """The player just won game 3 with three crosses along the top."""
    grid_scene(b)
    b.board = ["X", "X", "X", "O", "O", "", "", "", ""]
    b.last_choice = Choice(4, [4], {4: 0}, True, "the strongest square")
    e = b.library.find("noughts and crosses")
    b.library.record(e, "win")
    b.set_state("GAME_OVER", "test")
    b.emit("game_over", "buddy", "noughts and crosses over: the player won", game="noughts and crosses", result="win",
           plays=e.plays, player_wins=e.player_wins, robot_wins=e.robot_wins, draws=e.draws)
    b.say("Three in a row! Well done, you beat me fair and square. Shall we play another?")
    return b


def dots_scene(b):
    """A 2 by 2 game: five lines drawn, the player owns the top left box, the player to move, Buddy drew last."""
    b._select(b.library.find("dots and boxes"))
    b.difficulty.set_level("medium")
    b.dots = brain_module.DotsPlay(b, 2)
    s = D.State.start(2, ("P", "B"), first="P")
    b.games = 1
    b.emit("game_start", "buddy", "game 1 of dots and boxes started", game="dots and boxes", number=1)
    b.set_state("DOTS_PLAYER_TURN", "test")
    b.game_started_at = time.time() - 150
    # edges of the top left box: top 0, bottom 2, left 6, right 7; Buddy's safe line is the right side of the top right box (8)
    for who, e in (("player", 0), ("buddy", 2), ("player", 6), ("buddy", 8), ("player", 7)):
        boxes = D.completed_by(2, s.edges, e)
        s = D.play(s, e)
        text = f"{'the player' if who == 'player' else 'Buddy'} drew {D.edge_name(2, e)}"
        b.emit("line", who, text, edge=e, name=D.edge_name(2, e), boxes=list(boxes), score={"player": s.score("P"), "buddy": s.score("B")})
        if boxes:
            b.emit("box", who, f"{'the player' if who == 'player' else 'Buddy'} closed the {D.box_name(2, boxes[0])}", boxes=list(boxes))
    # make the position as described: the player closed the top left box with its right side (7) and Buddy answered with 8
    b.dots.state = D.State(2, s.edges, ("P", "", "", ""), "P", ("P", "B"))
    b.last_choice = Choice(8, [8], {8: 0}, True, "a safe line that gives nothing away")
    b.say("Your turn.")
    return b


def dots_big_scene(b):
    """A 3 across by 2 down game just started: two lines, no boxes, Buddy to move."""
    b._select(b.library.find("dots and boxes"))
    b.dots = brain_module.DotsPlay(b, (2, 3))
    s = D.State.start((2, 3), ("P", "B"), first="P")
    s = D.play(s, 0)
    s = D.play(s, 4)
    b.dots.state = D.State((2, 3), s.edges, s.owners, "B", ("P", "B"))
    b.games = 1
    b.set_state("DOTS_ROBOT_TURN", "test")
    b.game_started_at = time.time() - 30
    b.emit("line", "player", f"the player drew {D.edge_name((2, 3), 0)}", edge=0, name=D.edge_name((2, 3), 0), boxes=[])
    b.emit("line", "buddy", f"Buddy drew {D.edge_name((2, 3), 4)}", edge=4, name=D.edge_name((2, 3), 4), boxes=[])
    b.say("Let me think about my line.")
    return b


def hangman_paper_scene(b):
    """The hangman scene with a paper routine attached: the alphabet in the corner, wrong letters crossed out."""
    hangman_scene(b)
    b.paper = SimpleNamespace(describe=lambda: "Written on the paper: the alphabet in the top right corner, with N, S and T "
                                               "crossed out. Standing rule: cross out each wrong letter.")
    return b


SCENES = {"hangman": hangman_scene, "hangman_fresh": hangman_fresh_scene, "hangman_paper": hangman_paper_scene,
          "grid": grid_scene, "grid_over": grid_over_scene, "dots": dots_scene, "dots_big": dots_big_scene}


@pytest.fixture
def brain(tmp_path, monkeypatch):
    return make_brain(tmp_path, monkeypatch)


# ---------------------------------------------------------------- the facts
def test_hangman_facts_tell_the_letters_right_wrong_and_misses(brain):
    hangman_scene(brain)
    f = brain.facts()
    assert "Game: hangman, on, game 2 of this session, started 4 minutes ago" in f
    assert "Word so far: _ P P _ E (5 letters, 3 found, 2 still blank)" in f
    assert "Letters tried so far: P, T, E, S, N (5 letters). Right letters: E, P. Wrong letters: N, S, T." in f
    assert "Misses: 3 of 6 used, 3 misses left. Gallows parts drawn: the upright post, the beam, the rope." in f
    assert "Whose turn: the player's." in f
    assert "The player's name is Margaret." in f
    assert "APPLE" not in f, "the hidden word must never reach the model while the game is on"
    assert "Games this session: 1 finished (game 1: noughts and crosses, a draw); this hangman is game 2, still going." in f
    assert "Level: authentic (adaptive" in f and "62% of the time" in f and "rating is" in f
    assert "Moves this game, in order: 1. the player guessed P: right" in f
    assert "Conversation so far" in f and 'Player: "N"' in f
    assert 'Buddy last said: "There is no N. 3 misses left. Say another letter."' in f
    assert len(f) <= Brain.FACTS_LIMIT


def test_fresh_hangman_says_none_tried_and_the_length(brain):
    hangman_fresh_scene(brain)
    f = brain.facts()
    assert "Letters tried so far: none yet." in f and "(7 letters, 0 found, 7 still blank)" in f
    assert "Misses: 0 of 6 used, 6 misses left. Gallows parts drawn: none." in f
    assert "Level: hard" in f and "name" not in f.lower().split("conversation")[0]
    assert "Games this session: this hangman is the first, still going." in f


def test_grid_facts_name_every_square(brain):
    grid_scene(brain)
    f = brain.facts()
    assert "The player is crosses (X), Buddy is noughts (O)." in f
    assert "Board: top left cross; top right nought; centre nought; bottom right cross." in f
    assert "Free squares: top middle, middle left, middle right, bottom left, bottom middle." in f
    assert "Marks: the player 2, Buddy 2." in f
    assert "Buddy's last move: the top right (blocking your line)." in f
    assert "Level: easy; Buddy plays its best move 30% of the time." in f
    assert "Record at noughts and crosses: 2 played, the player won 1, Buddy won 1, 0 drawn." in f
    assert "Games this session: 2 finished (game 1: noughts and crosses, Buddy won; game 2: noughts and crosses, the player won); this noughts and crosses is game 3, still going." in f
    assert "4. Buddy put nought in the top right (blocking your line)" in f


def test_grid_over_facts_give_the_result(brain):
    grid_over_scene(brain)
    f = brain.facts()
    assert "Game: noughts and crosses, just finished" in f
    assert "Result: the player won with three in a row." in f
    assert "Whose turn: nobody's, the game is over." in f
    assert "Games this session: 3 finished" in f and "game 3: noughts and crosses, the player won" in f


def test_dots_facts_count_lines_and_boxes(brain):
    dots_scene(brain)
    f = brain.facts()
    assert "Dots board: 2 by 2 boxes (4 boxes, 12 lines). Lines drawn: 5 of 12, 7 to go." in f
    assert "Score: the player 1, Buddy 0, 3 boxes still open. The player's boxes: top left box. Buddy's boxes: none yet." in f
    assert "Buddy's last line: the right side of the top right box (a safe line that gives nothing away)." in f
    assert "the player closed the top left box" in f


def test_dots_big_board_and_buddys_turn(brain):
    dots_big_scene(brain)
    f = brain.facts()
    assert "Dots board: 2 by 3 boxes (6 boxes, 17 lines). Lines drawn: 2 of 17" in f
    assert "Whose turn: Buddy's." in f


def test_no_game_facts_still_describe_the_session(brain):
    f = brain.facts()
    assert f.startswith("No game is on.") and "Games this session: none yet." in f


def test_current_turn_is_left_out_of_the_conversation(brain):
    hangman_scene(brain)
    brain.emit("heard", "player", "what letters have I asked so far")
    with_it = brain.facts()
    without = brain.facts(current="what letters have I asked so far")
    assert "what letters have I asked so far" in with_it
    assert "what letters have I asked so far" not in without
    assert 'Player: "N" | Buddy: "There is no N.' in without


def test_facts_stay_under_the_limit_with_a_long_history(brain):
    hangman_scene(brain)
    for i in range(40):
        talk(brain, f"player line number {i} with a few extra words to make it long", f"Buddy reply number {i}, also padded out a bit")
        brain.emit("guess", "player", f"the player guessed Q: wrong (word so far _ P P _ E, 3 of 6 misses) padding {i}")
    f = brain.facts()
    assert len(f) <= Brain.FACTS_LIMIT
    assert "Buddy reply number 39" in f and "Moves this game" in f
    assert "player line number 0 " not in f


def test_paper_description_is_included_when_present(brain):
    hangman_scene(brain)
    brain.paper = SimpleNamespace(describe=lambda: "Written on the paper: the alphabet in the top right corner, with N, S and T crossed out.")
    assert "alphabet in the top right corner" in brain.facts()
    brain.paper = SimpleNamespace(describe=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert "could not be described" in brain.facts()


# ---------------------------------------------------------------- the name
@pytest.mark.parametrize("said, name", [
    ("I'm Margaret, by the way", "Margaret"), ("my name is arthur", "Arthur"), ("Call me Peg", "Peg"),
    ("I'm fine thanks", None), ("I am done", None), ("it's E", None), ("I'm not sure", None), ("this is fun", None),
])
def test_name_memory(brain, said, name):
    brain._note_name(said)
    assert brain.player_name == name


def test_hearing_a_name_in_a_turn_remembers_it(brain):
    brain.state = "CHAT"
    brain._plan_tools = lambda text, sit: []
    brain._on_text("Hello, I'm Dorothy", "typed", 0.0)
    assert brain.player_name == "Dorothy"


# ---------------------------------------------------------------- the planner's message
def test_user_message_carries_the_facts_after_the_situation():
    m = toolcall.Planner.user_message("which were wrong?", "The player is guessing.", "Wrong letters: S, T.")
    assert m == 'Situation: The player is guessing.\nFacts:\nWrong letters: S, T.\nPlayer said: "which were wrong?"'
    assert toolcall.Planner.user_message("hi", "Chatting.") == 'Situation: Chatting.\nPlayer said: "hi"'


def test_history_keeps_the_words_not_the_facts_and_trims(monkeypatch):
    monkeypatch.setattr(toolcall, "ENABLED", True)
    answers = iter([{"message": {"content": "", "tool_calls": [{"function": {"name": "say", "arguments": {"text": f"answer {i}"}}}]}}
                    for i in range(10)])
    monkeypatch.setattr(toolcall, "_post", lambda body, timeout: next(answers))
    p = toolcall.Planner()
    base = len(p.msgs)
    for i in range(7):
        calls = p.plan(f"question {i}", "sit", "HANGMAN_TURN", facts=f"FACTS {i}")
        assert calls and calls[0].name == "say"
    history = p.msgs[base:]
    assert not any("FACTS" in m.get("content", "") for m in history)
    users = [m for m in history if m["role"] == "user"]
    assert len(users) == toolcall.MAX_TURNS
    first = 7 - toolcall.MAX_TURNS
    assert users[0]["content"] == f'Player said: "question {first}"' and users[-1]["content"] == 'Player said: "question 6"'
    assert p.msgs[:base] == toolcall.Planner().msgs[:base]


def test_what_did_i_say_is_answered_from_the_log_even_through_repeat(brain):
    hangman_scene(brain)
    brain.emit("heard", "player", "what did I say a minute ago?")
    brain.run_tools([ToolCall("repeat", {})], "what did I say a minute ago?")
    assert brain.last_said == 'A moment ago you said "N", and before that "I think it might be a fruit".'
    brain.run_tools([ToolCall("repeat", {})], "say that again please")
    assert brain.last_said == 'A moment ago you said "N", and before that "I think it might be a fruit".'   # a real repeat repeats
    fresh = brain.__class__.__new__(brain.__class__)
    fresh.app, fresh.events = brain.app, []
    assert fresh.what_player_said() == "You haven't said anything to me yet."


def test_say_keeps_three_sentences(brain):
    long = "One sentence here. " * 40
    brain.run_tools([ToolCall("say", {"text": long})], "")
    assert len(brain.last_said) == 500


def test_exact_question_tools_still_answer(brain):
    hangman_scene(brain)
    brain.run_tools([ToolCall("tell_score", {})], "what's the score")
    assert "three misses left" in brain.last_said
    brain.run_tools([ToolCall("explain_last_move", {})], "why")
    assert "N" in brain.last_said and "gallows" in brain.last_said
    brain.run_tools([ToolCall("explain_rules", {})], "rules")
    assert "letter" in brain.last_said
