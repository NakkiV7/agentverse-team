import threading
from types import SimpleNamespace

import numpy as np
import pytest

from buddy.paper import Paper
from buddy.paper import layout as L
from buddy.paper import routines as R


@pytest.fixture
def drawing(monkeypatch):
    calls, traces = [], []
    done = threading.Event()
    done.set()
    app = SimpleNamespace(running=True, cam=None, trace=lambda *args: traces.append(args),
                          points_reachable=lambda pose, pts: (True, "reachable"),
                          draw=lambda name, strokes: (calls.append((name, strokes))
                                                     or SimpleNamespace(done=done, ok=True)))
    b = SimpleNamespace(app=app, busy=False, switch_to=None)

    def worker(fn):
        def run():
            try:
                fn()
            finally:
                b.busy = False
        threading.Thread(target=run).start()
    b._worker = worker
    p = Paper(b)
    monkeypatch.setattr(p, "_look", lambda: SimpleNamespace(rgb=None, track=SimpleNamespace(pose=(.21, 0, 1.57))))
    p.calls, p.traces = calls, traces
    yield p
    p.close()


def test_faint_mark_retries_once_and_success_releases_brain(drawing, monkeypatch):
    coverages = iter([[.3], [1]])
    monkeypatch.setattr(R, "check_lines", lambda *args, **kwargs: next(coverages))
    drawing._draw_checked("cross out", [np.array([[0, 0], [5, 10]])], 0)
    assert len(drawing.calls) == 2
    assert not drawing.brain.busy and not drawing.drawing


def test_two_faint_attempts_refuse_and_never_retry_forever(drawing, monkeypatch):
    monkeypatch.setattr(R, "check_lines", lambda *args, **kwargs: [.3])
    with pytest.raises(ValueError, match="after one retry"):
        drawing._draw_checked("cross out", [np.array([[0, 0], [5, 10]])], 0)
    assert len(drawing.calls) == 2
    assert not drawing.brain.busy


def test_glyph_ink_check_handles_tiny_segments(drawing, monkeypatch):
    monkeypatch.setattr(R, "check_lines", lambda *args, **kwargs: [.4])
    monkeypatch.setattr(drawing, "glyph_ink", lambda *args: .15)
    cell, = L.layout_text(".", L.PAGE)
    drawing._draw_checked("period", cell.strokes(), 0, cell=cell)
    assert len(drawing.calls) == 1


def test_reach_checks_every_point_before_any_ink(drawing, monkeypatch):
    checked = []

    def reachable(pose, points):
        checked.extend(points)
        return False, "middle point is out of reach"
    drawing.app.points_reachable = reachable
    stroke = np.array([[0, 0], [3, 8], [5, 10]])
    with pytest.raises(ValueError, match="middle point"):
        drawing._draw_checked("line", [stroke], 0)
    assert checked == [(0, 0), (3, 8), (5, 10)]
    assert not drawing.calls


def test_switch_stops_before_the_next_character(drawing):
    drawing.brain.switch_to = "stop"

    def switch():
        drawing.brain.switch_to = None
        drawing.new_sheet()
    drawing.brain._maybe_switch = switch
    with pytest.raises(ValueError, match="sheet or game changed"):
        drawing._draw_checked("A", [np.array([[0, 0], [5, 10]])], 0)
    assert not drawing.calls


def test_points_reachable_rejects_bad_and_off_sheet_points_without_touching_world():
    from buddy.app import App
    app = object.__new__(App)
    for points in ([1, 2], [[float("nan"), 0]], [[149, 0]], [[0, -106]]):
        assert not app.points_reachable((.21, 0, 1.57), points)[0]
    assert app.points_reachable((.21, 0, 1.57), [])[0]


def test_group_checks_one_frame_and_redraws_only_each_failed_character(drawing, monkeypatch):
    cells = L.layout_text("ABCD", L.PAGE)
    seen = []
    looks = []

    def look():
        frame = SimpleNamespace(rgb=None, track=SimpleNamespace(pose=(.21, 0, 1.57)))
        looks.append(frame)
        return frame

    def ink(frame, app, cell):
        seen.append((cell.char, frame))
        return 0 if len(drawing.calls) == 1 and cell.char in "BD" else .2

    monkeypatch.setattr(drawing, "_look", look)
    monkeypatch.setattr(drawing, "glyph_ink", ink)
    monkeypatch.setattr(R, "check_lines", lambda *args, **kwargs: [1])
    drawing._draw_checked("paper note: ABCD", [s for c in cells for s in c.strokes()], 0, cells=cells)
    assert len(drawing.calls) == 3  # the row, B alone, then D alone
    assert [c for c, _ in seen] == list("ABCDBD")
    assert all(frame is seen[0][1] for _, frame in seen[:4])
    assert seen[4][1] is not seen[0][1] and seen[5][1] is not seen[4][1]
    assert len(looks) == 6  # one preflight and one verification frame per job
    for call, cell in zip(drawing.calls[1:], (cells[1], cells[3])):
        assert len(call[1]) == len(cell.strokes())
        assert all(np.array_equal(actual, expected) for actual, expected in zip(call[1], cell.strokes()))
    assert all(c.written for c in cells)


def test_partial_retry_failure_keeps_successful_characters_checked(drawing, monkeypatch):
    cells = L.layout_text("ABC", L.PAGE)
    monkeypatch.setattr(drawing, "glyph_ink", lambda look, app, cell: 0 if cell.char == "B" else .2)
    monkeypatch.setattr(R, "check_lines", lambda *args, **kwargs: [1])
    with pytest.raises(ValueError, match="after one retry"):
        drawing._draw_checked("paper note: ABC", [s for c in cells for s in c.strokes()], 0, cells=cells)
    assert len(drawing.calls) == 2
    assert [c.written for c in cells] == [True, False, True]
    assert not drawing.brain.busy and not drawing.drawing


@pytest.mark.parametrize("faint", [False, True])
def test_guess_is_served_after_its_job_even_before_a_retry_or_at_the_end(drawing, monkeypatch, faint):
    cells = L.layout_text("AB", L.PAGE)
    order = []
    submit = drawing.app.draw
    guess = SimpleNamespace(intent="say_letter", letter="P")

    def draw(name, strokes):
        order.append(name)
        if not drawing.calls:
            assert drawing.defer_guess(guess)
            assert len(order) == 1
        return submit(name, strokes)

    def ink(look, app, cell):
        return 0 if faint and len(drawing.calls) == 1 and cell.char == "B" else .2

    drawing.app.draw = draw
    drawing.brain._hangman_act = lambda u: order.append(f"guess {u.letter}")
    monkeypatch.setattr(drawing, "glyph_ink", ink)
    monkeypatch.setattr(R, "check_lines", lambda *args, **kwargs: [1])
    drawing._draw_checked("paper note: AB", [s for c in cells for s in c.strokes()], 0, cells=cells)
    assert order[:2] == ["paper note: AB", "guess P"]
    assert len(order) == (3 if faint else 2)
    if faint:
        assert "retry B" in order[2]
    assert not drawing._guesses
