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
