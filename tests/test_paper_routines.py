import threading
import time
from types import SimpleNamespace

import pytest

from buddy.game.hangman import Hangman
from buddy.paper import Paper, context_from
from buddy.paper.routines import fill_template, matches


def wait(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "paper worker did not finish"
        time.sleep(.005)


@pytest.fixture
def paper(monkeypatch):
    traces, said, draws = [], [], []
    app = SimpleNamespace(trace=lambda *args: traces.append(args), running=True,
                          points_reachable=lambda pose, pts: (True, "reachable"))
    b = SimpleNamespace(app=app, busy=False, game_name="hangman", hangman=Hangman("APPLE"),
                        games=1, dots=None, say=said.append, switch_to=None)
    p = Paper(b)
    monkeypatch.setattr(p, "_look", lambda: SimpleNamespace(track=SimpleNamespace(pose=(.21, 0, 1.5708))))
    monkeypatch.setattr(p, "_draw_checked", lambda name, strokes, generation, cell=None: draws.append((name, strokes)))
    p.traces, p.said, p.draws = traces, said, draws
    yield p
    p.close()


def write(p, text="ABC", id="alphabet", place="top right corner"):
    return p.run([{"op": "write_text", "id": id, "text": text, "place": place}])


def event(kind="guess", who="player", **data):
    return SimpleNamespace(kind=kind, who=who, data=data)


def rule(name="wrong letters", op="cross_out"):
    return {"name": name, "when": {"kind": "guess", "who": "player", "correct": False},
            "do": [{"op": op, "target": {"id": "alphabet", "chars": "{letter}"}}]}


def test_templates_use_public_state_and_event_values(paper):
    paper.brain.hangman.guess("B")
    paper.brain.player_name = "Margaret"
    context = context_from(paper.brain)
    assert "word" not in context
    assert fill_template("{wrong}/{player_name}/{number}", context) == "B/Margaret/1"
    assert fill_template("{correct}: {guessed}", {"correct": False, "guessed": ["B", "P"]}) == "FALSE: B P"
    for template in ("{unknown}", "{result}", "{number.real}", "{number!r}", "{number:04}"):
        with pytest.raises(ValueError):
            fill_template(template, context)
    assert write(paper) is None
    paper.add_rule(rule())
    paper.on_event(event(letter="C", correct=False))
    wait(lambda: not paper.pending)
    assert paper.items["alphabet"].characters[2].marks == {"cross_out"}


def test_event_matching_is_exact_kind_who_and_json_type():
    when = rule()["when"]
    assert matches(when, event(correct=False))
    for ev in (event(correct=True), event(who="buddy", correct=False), event(kind="line", correct=False),
               event(correct=0), event()):
        assert not matches(when, ev)


def test_synchronous_busy_emit_does_not_deadlock_and_cancel_drops_queue(paper):
    assert write(paper) is None
    paper.add_rule(rule())
    paper.brain.busy = True
    before = time.monotonic()
    paper.on_event(event(letter="B", correct=False))
    assert time.monotonic() - before < .1
    time.sleep(.04)
    assert not paper.items["alphabet"].characters[1].marks
    paper.brain.busy = False
    wait(lambda: not paper.pending)
    assert paper.items["alphabet"].characters[1].marks == {"cross_out"}
    paper.brain.busy = True
    paper.on_event(event(letter="C", correct=False))
    assert paper.cancel("cross_out") == 1
    paper.brain.busy = False
    wait(lambda: not paper.pending)
    assert not paper.items["alphabet"].characters[2].marks


@pytest.mark.parametrize("selector", ["wrong letters", "guess", "cross_out", {"kind": "guess", "op": "cross_out"}])
def test_cancel_by_name_event_or_operation(paper, selector):
    paper.add_rule(rule())
    assert paper.cancel(selector) == 1
    assert paper.rules == []


def test_cross_all_occurrences_is_idempotent_and_append_preserves_boxes(paper):
    assert write(paper, "ABBA") is None
    mark = {"op": "cross_out", "target": {"id": "alphabet", "chars": "B"}}
    assert paper.run([mark]) is None
    count = len(paper.draws)
    assert paper.run([mark]) is None
    assert len(paper.draws) == count
    boxes = paper.items["alphabet"].boxes[:]
    assert paper.run([{"op": "append_text", "id": "alphabet", "text": " 2:0"}]) is None
    assert paper.items["alphabet"].boxes[:4] == boxes
    assert paper.items["alphabet"].characters[1].marks == {"cross_out"}
    assert "B crossed out" in paper.describe()


def test_refusal_suggests_a_fitting_place_and_draws_nothing(paper):
    error = write(paper, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", place="bottom edge")
    assert "within reach" in error and "would work" in error
    assert not paper.draws and not paper.items
    assert "at least 5" in paper.run([{"op": "write_text", "id": "tiny", "text": "A",
                                      "place": "top right corner", "size_mm": 4}])


def test_unknown_template_skips_only_that_action_with_trace(paper):
    error = paper.run([{"op": "write_text", "id": "missing", "text": "{name}", "place": "top right corner"},
                       {"op": "write_text", "id": "ok", "text": "A", "place": "top right corner"}])
    assert "template {name}" in error
    assert set(paper.items) == {"ok"}
    assert any("skipped" in msg for _, msg in paper.traces)


def test_new_sheet_keeps_rules_and_missing_target_requests_rewrite(paper):
    assert write(paper) is None
    paper.add_rule(rule())
    paper.on_event(event(kind="switch"))
    assert not paper.items and len(paper.rules) == 1
    assert paper.needs_rewrite("alphabet")
    paper.on_event(event(letter="B", correct=False))
    wait(lambda: not paper.pending)
    assert not paper.said
    assert write(paper) is None
    assert not paper.needs_rewrite("alphabet")


def test_event_score_snapshot_is_preserved_while_waiting(paper):
    paper.brain.dots = SimpleNamespace(state=SimpleNamespace(score=lambda who: 1 if who == "P" else 0))
    assert write(paper, "0:0", id="score") is None
    paper.add_rule({"name": "scores", "when": {"kind": "box"},
                    "do": [{"op": "append_text", "id": "score", "text": " {score_player}:{score_buddy}"}]})
    paper.brain.busy = True
    paper.on_event(event(kind="box"))
    paper.brain.dots.state = SimpleNamespace(score=lambda who: 2)
    paper.brain.busy = False
    wait(lambda: not paper.pending)
    assert paper.items["score"].text == "0:0 1:0"


def test_defer_guess_is_drained_at_character_boundary(paper):
    guesses = []
    paper.brain._hangman_act = lambda u: guesses.append(u.letter)
    u = SimpleNamespace(intent="say_letter", letter="P")
    assert not paper.defer_guess(u)
    paper.drawing = True
    assert paper.defer_guess(u)
    assert not guesses
    paper.drawing = False
    paper._boundary(paper._generation)
    assert guesses == ["P"]


def test_a_said_rule_failure_does_not_generate_another_said_event(paper):
    paper.add_rule({"name": "notes on speech", "when": {"kind": "said"},
                    "do": [{"op": "write_text", "id": "speech", "text": "{name}",
                            "place": "top right corner"}]})
    paper.on_event(event(kind="said", who="buddy"))
    wait(lambda: not paper.pending)
    assert "template {name}" in paper.last_error
    assert not paper.said and not paper.items


@pytest.mark.parametrize("op", ["underline", "circle", "tick"])
def test_other_annotations_target_each_matching_character(paper, op):
    assert write(paper, "ABA") is None
    assert paper.run([{"op": op, "target": {"id": "alphabet", "chars": "A"}}]) is None
    assert [c.marks for c in paper.items["alphabet"].characters] == [{op}, set(), {op}]
    assert paper.draws[-1][0] == f"paper {op}: A"


def test_unsupported_operation_is_refused_without_ink(paper):
    assert "cannot erase" in paper.run([{"op": "erase_item", "id": "alphabet"}]).lower()
    assert not paper.draws


def test_context_has_last_move_and_result_but_does_not_cross_game_boundaries(paper):
    b = paper.brain
    b.hangman, b.game_name, b.state = None, "noughts and crosses", "PLAYER_TURN"
    b.events = [event(kind="game_start"), event(kind="move", cells=[4])]
    assert context_from(b)["cells"] == [4]
    b.events.append(event(kind="switch"))
    assert "cells" not in context_from(b)
    b.state = "GAME_OVER"
    b.events.append(event(kind="game_over", game=b.game_name, result="draw"))
    assert context_from(b)["result"] == "draw"
    b.state = "PLAYER_TURN"
    assert "result" not in context_from(b)


def test_a_reserved_robot_turn_is_not_mistaken_for_an_idle_worker(paper):
    assert write(paper) is None
    paper.add_rule(rule())
    paper.brain.state = "DOTS_ROBOT_TURN"
    paper.on_event(event(letter="B", correct=False))
    time.sleep(.05)
    assert not paper.items["alphabet"].characters[1].marks
    paper.brain.state = "DOTS_PLAYER_TURN"
    wait(lambda: not paper.pending)
    assert paper.items["alphabet"].characters[1].marks == {"cross_out"}


def test_full_item_refuses_append_with_a_workable_place_for_another_note(paper):
    assert write(paper, "A") is None
    item = paper.items["alphabet"]
    item.region = item.characters[0].box
    before = len(paper.draws)
    error = paper.run([{"op": "append_text", "id": "alphabet", "text": "B"}])
    assert "would work for a new note" in error
    assert item.text == "A" and len(paper.draws) == before
