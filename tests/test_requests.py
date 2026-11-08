"""Saved-answer parser and validator checks; this file never requires Ollama."""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
import urllib.error

import pytest

from buddy.agent import requests as R
from buddy.agent.tools import TOOLS, available
from tools import request_eval as E

# Saved Gemma 4 answer from the first live C3 smoke check (8 Nov 2026).
# Literal ET is valid, though the exact structural eval requires {wrong}.
SAVED_ALPHABET = '''{
  "say": "Of course. I'll write the alphabet in the top right corner and cross out your wrong letters as we play.",
  "now": [
    {"op": "write_text", "id": "alphabet", "text": "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "place": "top right corner", "size_mm": 6},
    {"op": "cross_out", "target": {"id": "alphabet", "chars": "ET"}}
  ],
  "rules": [{"name": "cross out wrong letters", "when": {"kind": "guess", "who": "player", "correct": false},
             "do": [{"op": "cross_out", "target": {"id": "alphabet", "chars": "{letter}"}}]}],
  "cancel": []
}'''


@pytest.fixture
def answer():
    return json.loads(SAVED_ALPHABET)


def test_saved_answer_and_whitespace():
    result = R.parse(" \n" + SAVED_ALPHABET + "\t")
    assert isinstance(result, R.Routine)
    assert result.now[1]["target"]["chars"] == "ET"
    assert result.rules[0]["when"] == {"kind": "guess", "who": "player", "correct": False}


@pytest.mark.parametrize("prefix,suffix", [
    ("Here is your JSON: ", ""), ("", " I'll do that now."), ("```json\n", "\n```"),
    ("", "{}"), ("[", "]"), ("I cannot do that. ", ""),
])
def test_rejects_json_wrapped_in_prose(prefix, suffix):
    result = R.parse(prefix + SAVED_ALPHABET + suffix)
    assert isinstance(result, R.Refusal)
    assert "wrapped" in result.reason


@pytest.mark.parametrize("raw", ["", "yes", "[]", "null", "{broken", '{"say":"a","say":"b"}',
                                  '{"now":NaN}', '{"now":Infinity}', "{" * 1100, None])
def test_bad_json_is_a_refusal(raw):
    assert isinstance(R.parse(raw), R.Refusal)


@pytest.mark.parametrize("path,value", [
    (("now", 0, "op"), "erase_item"),
    (("now", 0, "place"), "centre"),
    (("now", 0, "place"), ["top edge"]),
    (("now", 0, "text"), "{vowels}"),
    (("now", 1, "target", "chars"), "{right}"),
    (("now", 0, "text"), "{letter.upper()}"),
    (("now", 0, "text"), "{{letter}}"),
    (("now", 0, "text"), "hello"),
    (("now", 0, "text"), "A_B"),
    (("now", 0, "text"), "CAT 🐱"),
    (("now", 0, "size_mm"), 4.99),
    (("now", 0, "size_mm"), True),
    (("now", 0, "size_mm"), "6"),
    (("now", 0, "size_mm"), float("inf")),
    (("now", 0, "id"), "../arm"),
    (("now", 1, "target"), {"box": 0}),
    (("now", 1, "target"), {"id": "alphabet", "x": 10, "y": 20}),
    (("rules", 0, "when", "kind"), "wrong_guess"),
    (("rules", 0, "when", "kind"), []),
    (("rules", 0, "when", "who"), "robot"),
    (("rules", 0, "when", "correct"), "false"),
    (("rules", 0, "when", "correct"), 0),
    (("rules", 0, "when"), {"kind": "guess", "score_player": 1}),
    (("rules", 0, "when"), {"kind": "box", "correct": False}),
    (("rules", 0, "do", 0, "op"), "draw_picture"),
    (("rules", 0, "do", 0, "target", "chars"), "{vowels}"),
    (("rules", 0, "do"), []),
    (("cancel",), [{"op": "erase_item"}]),
    (("cancel",), [{"kind": "won_box"}]),
    (("cancel",), [{"kind": []}]),
    (("cancel",), [{"op": "cross_out", "kind": "guess"}]),
    (("cancel",), [False]),
    (("cancel",), "all"),
    (("say",), 123),
    (("rules",), {}),
    (("now",), {}),
])
def test_rejects_invalid_vocabulary_and_types_atomically(answer, path, value):
    target = answer
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    result = R.validate(answer)
    assert isinstance(result, R.Refusal)
    assert result.reason
    assert "tally" in result.say and "written item" in result.say
    assert not hasattr(result, "now")


@pytest.mark.parametrize("path,key", [((), "motion"), (("now", 0), "strokes"),
                                      (("rules", 0), "repeat"), (("rules", 0, "when"), "wrong_letter")])
def test_rejects_extra_keys(answer, path, key):
    target = answer
    for part in path:
        target = target[part]
    target[key] = True
    assert isinstance(R.validate(answer), R.Refusal)


def test_no_partial_success_and_no_empty_routine(answer):
    answer["now"].append({"op": "move_arm"})
    assert isinstance(R.validate(answer), R.Refusal)
    assert isinstance(R.validate({"say": "Of course.", "now": [], "rules": [], "cancel": []}), R.Refusal)


def test_duplicate_rule_names_and_nested_duplicate_json_keys(answer):
    answer["rules"] *= 2
    assert isinstance(R.validate(answer), R.Refusal)
    raw = SAVED_ALPHABET.replace('"op": "write_text"', '"op": "erase_item", "op": "write_text"')
    assert isinstance(R.parse(raw), R.Refusal)


def test_valid_answer_is_detached_from_input(answer):
    result = R.validate(answer)
    answer["now"][0]["op"] = "move_arm"
    assert result.now[0]["op"] == "write_text"


@pytest.mark.parametrize("entry", ["cross out wrong letters", {"op": "cross_out"}, {"kind": "guess"}])
def test_all_cancel_forms(entry):
    result = R.validate({"say": "I'll stop those marks.", "now": [], "rules": [], "cancel": [entry]})
    assert isinstance(result, R.Routine)
    assert result.cancel == [entry]


@pytest.mark.parametrize("place", R.PLACES)
def test_every_place_and_default_size(answer, place):
    answer["now"][0]["place"] = place
    answer["now"][0].pop("size_mm")
    assert isinstance(R.validate(answer), R.Routine)


@pytest.mark.parametrize("op", R.OPS)
def test_every_operation(answer, op):
    action = {"op": op, "target": {"id": "alphabet"}}
    if op == "append_text":
        action = {"op": op, "id": "alphabet", "text": " 10.2:/-"}
    if op == "write_text":
        action = {"op": op, "id": "score", "text": "SCORE\n10:2", "place": "bottom edge", "size_mm": 5}
    answer["now"] = [action]
    assert isinstance(R.validate(answer), R.Routine)


@pytest.mark.parametrize("template", R.TEMPLATES)
def test_every_template_in_text_and_chars(answer, template):
    answer["now"][0]["text"] = "{" + template + "}"
    answer["now"][1]["target"]["chars"] = "{" + template + "}"
    assert isinstance(R.validate(answer), R.Routine)


@pytest.mark.parametrize("kind,keys", R.EVENT_KEYS.items())
def test_every_event_kind_and_data_key(answer, kind, keys):
    # Event data is matched by exact JSON value, not interpreted as a predicate.
    answer["rules"][0]["when"] = {"kind": kind, **{key: False if key == "correct" else "VALUE" for key in keys}}
    assert isinstance(R.validate(answer), R.Routine)


def test_all_prompt_examples_are_valid():
    assert len(R.EXAMPLES) == 10
    assert R.EXAMPLES[0][0] == R.ALPHABET_REQUEST
    for _, _, answer in R.EXAMPLES:
        result = R.validate(answer)
        assert isinstance(result, R.Routine) or set(answer) == {"reason", "say"}


def test_refusal_must_not_smuggle_actions():
    answer = {"reason": "No box target", "say": "I can keep a written tally instead."}
    assert R.parse(json.dumps(answer)) == R.Refusal(**answer)
    answer["now"] = [{"op": "erase_item"}]
    assert isinstance(R.validate(answer), R.Refusal)
    assert R.validate(answer).reason != answer["reason"]


def test_compile_uses_saved_answer(monkeypatch):
    monkeypatch.setattr(R, "_chat", lambda request, facts, state: SAVED_ALPHABET)
    assert isinstance(R.compile("write the alphabet", "Nothing written yet.", "CHAT"), R.Routine)


def test_http_contract(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, limit):
            assert limit == R.MAX_RESPONSE + 1
            return json.dumps({"done": True, "message": {"content": SAVED_ALPHABET}}).encode()

    def urlopen(req, timeout):
        seen.update(body=json.loads(req.data), url=req.full_url, timeout=timeout)
        return Response()

    monkeypatch.setattr(R.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(R.C, "OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(R.C, "TEACH_MODEL", "test-gemma")
    request = "write my name; it's Margaret"
    assert isinstance(R.compile(request, "Known facts", "CHAT"), R.Routine)
    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    assert seen["timeout"] == 60
    body = seen["body"]
    assert body["model"] == "test-gemma" and body["think"] is False and body["stream"] is False
    assert body["options"]["temperature"] == 0
    assert body["options"]["num_ctx"] == 8192
    assert "format" not in body
    assert body["messages"][0]["content"] == R.SYSTEM
    assert json.loads(body["messages"][1]["content"]) == {"request": request, "facts": "Known facts", "state": "CHAT"}


@pytest.mark.parametrize("url", ["https://example.com", "http://192.168.1.2:11434", "http://localhost/api",
                                 "http://localhost?other=1", "http://user:secret@localhost"])
def test_remote_or_ambiguous_endpoints_refuse(monkeypatch, url):
    monkeypatch.setattr(R.C, "OLLAMA_URL", url)
    assert isinstance(R.compile("write A", "Paper is blank.", "CHAT"), R.Refusal)


@pytest.mark.parametrize("error", [TimeoutError("slow"), urllib.error.URLError("offline"), ValueError("bad payload")])
def test_connection_errors_refuse(monkeypatch, error):
    def fail(*_):
        raise error
    monkeypatch.setattr(R, "_chat", fail)
    assert isinstance(R.compile("write A", "Paper is blank.", "CHAT"), R.Refusal)


def test_total_deadline_discards_late_model_answer(monkeypatch):
    release = threading.Event()
    finished = threading.Event()

    def slow(*_):
        release.wait(1)
        finished.set()
        return SAVED_ALPHABET

    monkeypatch.setattr(R, "TIMEOUT", .02)
    monkeypatch.setattr(R, "_chat", slow)
    start = time.perf_counter()
    result = R.compile("write A", "Paper is blank.", "CHAT")
    try:
        assert isinstance(result, R.Refusal) and "timed out" in result.reason
        assert time.perf_counter() - start < .5
    finally:
        release.set()
        assert finished.wait(1)
    assert isinstance(result, R.Refusal)


def test_invalid_inputs_do_not_call_model(monkeypatch):
    def fail(*_):
        pytest.fail("model should not be called")
    monkeypatch.setattr(R, "_chat", fail)
    for args in [("", "facts", "CHAT"), (None, "facts", "CHAT"), ("hello", {}, "CHAT"), ("hello", "facts", "")]:
        assert isinstance(R.compile(*args), R.Refusal)


def test_custom_tool_is_last_available_in_setup_and_game_states():
    tool = TOOLS[-1]
    assert tool.name == "custom_request" and tool.required == ("request",)
    for state in ("CHAT", "CHOOSE_GAME", "ASK_LEVEL", "ASK_SIDE", "WAIT_SHEET", "SETUP", "OFFER_TAUGHT",
                  "DOTS_SIZE", "DOTS_SETUP", "DOTS_PLAYER_TURN", "DOTS_ROBOT_TURN", "HANGMAN_SETUP",
                  "HANGMAN_TURN", "PLAYER_TURN", "ROBOT_TURN", "CONFIRM_MARK", "GAME_OVER"):
        assert tool in available(state)
    for state in ("TEACH", "NAME_GAME", "IDLE"):
        assert tool not in available(state)


def test_corpus_and_three_planner_cases():
    root = Path(__file__).resolve().parents[1]
    cases = E.load_cases(root / "tests/data/requests.jsonl")
    assert len(cases) >= 25
    assert R.ALPHABET_REQUEST in [c["request"] for c in cases]
    for request in ("erase the gallows", "draw a picture of a cat", "change the rules so I win"):
        assert next(c for c in cases if c["request"] == request)["expect"] == {"refusal": True}
    planner = [json.loads(line) for line in (root / "tests/data/tool_eval.jsonl").read_text().splitlines()]
    custom = [c for c in planner if c["category"] == "custom_request"]
    assert len(planner) == 137 and len(custom) == 3
    assert all(c["args"]["custom_request"]["request"] == c["said"] for c in custom)


@pytest.mark.parametrize("change", ["op", "place", "template", "event", "data_key", "data_value", "extra_action", "cancel"])
def test_scorer_requires_exact_structure(answer, change):
    result = R.validate(answer)
    case = {"expect": E.structure(result)}
    changed = deepcopy(result)
    if change == "op": changed.now[1]["op"] = "underline"
    elif change == "place": changed.now[0]["place"] = "bottom edge"
    elif change == "template": changed.now[1]["target"]["chars"] = "{wrong}"
    elif change == "event": changed.rules[0]["when"]["kind"] = "box"
    elif change == "data_key": changed.rules[0]["when"].pop("correct")
    elif change == "data_value": changed.rules[0]["when"]["correct"] = True
    elif change == "extra_action": changed.now.append(deepcopy(changed.now[1]))
    elif change == "cancel": changed.cancel.append({"op": "cross_out"})
    assert not E.check(case, result, SAVED_ALPHABET)
    assert E.check(case, changed, None)


def test_scorer_does_not_credit_transport_or_validation_failure_as_model_refusal():
    case = {"expect": {"refusal": True}}
    for raw in (None, "nonsense", '{"op":"erase_item"}', '{"reason":"x","say":42}'):
        assert E.check(case, R.parse(raw), raw)
    raw = '{"reason":"No box targets","say":"I can keep a written tally instead."}'
    assert not E.check(case, R.parse(raw), raw)


def test_scorer_checks_literal_selection():
    result = R.parse(SAVED_ALPHABET)
    case = {"expect": E.structure(result), "literals": [{"action": 1, "field": "chars", "value": "AE"}]}
    assert E.check(case, result, SAVED_ALPHABET)


def test_refusal_written_with_the_refusal_key_keeps_its_sentence():
    """The model sometimes answers {"refusal": ...}: the kind sentence is kept, the shape is not an error."""
    from buddy.agent.requests import Refusal, validate
    out = validate({"refusal": "Ink cannot be erased.", "say": "I can't rub anything out, but I can cross it through."})
    assert isinstance(out, Refusal) and out.reason == "Ink cannot be erased." and "cross it through" in out.say
    out = validate({"refusal": "No picture drawing."})
    assert isinstance(out, Refusal) and out.say
    assert isinstance(validate({"refusal": "x", "now": []}), Refusal)   # a refusal never carries actions
