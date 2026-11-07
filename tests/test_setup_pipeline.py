"""Perception + drawing on the simulator, without speech.

    .venv/bin/python -m pytest -q tests

The robot only learns where the sheet is through the camera: drawing the grid in the right place on a shifted,
rotated (and, in one test, moving) sheet proves the behaviour is driven by perception.
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from buddy import config as C
from buddy.game.tictactoe import Difficulty, choose, full, other, winner
from buddy.motion.controller import ArmController, Job
from buddy.motion.drawing import grid_lines_mm
from buddy.motion.ik import PenIK
from buddy.perception import safety
from buddy.perception.sheet import CameraModel, SheetTracker, check_lines, find_sheet
from buddy.sim.world import World

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


@pytest.fixture(scope="module")
def world():
    w = World()
    yield w
    w.close()


def run_job(w, cam, strokes_mm, events=None, max_s=150):
    """Run a drawing job with the camera feeding the sheet tracker at 10 Hz, like the app does."""
    tracker = SheetTracker()
    ik = PenIK(w.model, w.pen.tip_body, w.pen.axis_body, World.ARM)
    ctl = ArmController(ik, w.arm_q(), lambda kind, msg: None)
    spt = int(round(1 / C.CONTROL_HZ / C.TIMESTEP))
    t, forces = 0.0, []
    for _ in range(3):
        t += 0.1
        tracker.update(find_sheet(w.render_overhead(), cam), t)
    job = ctl.submit(Job("test", strokes_mm=strokes_mm, sheet=lambda: tracker.track, home=True))
    for k in range(int(max_s * C.CONTROL_HZ)):
        w.command(ctl.tick(w.touch()))
        w.step(spt)
        t += 1 / C.CONTROL_HZ
        forces.append(w.touch())
        if k % (C.CONTROL_HZ // C.CAMERA_HZ) == 0:
            tracker.update(find_sheet(w.render_overhead(), cam), t)
        if events and k in events:
            events[k](w)
        if job.done.is_set():
            break
    return job, np.array(forces), tracker


def test_finds_sheet_anywhere(world):
    cam = CameraModel(*world.camera_model())
    for pose in [(0.21, 0.0, math.pi / 2), (0.225, 0.03, math.pi / 2 + 0.25), (0.20, -0.025, math.pi / 2 - 0.15)]:
        world.move_sheet(*pose)
        world.step(2)
        s = find_sheet(world.render_overhead(), cam)
        assert s is not None
        assert abs(s.x - pose[0]) < 0.003 and abs(s.y - pose[1]) < 0.003
        assert abs(math.degrees(s.yaw - pose[2])) < 1.0
        assert s.confidence > 0.8


def test_draws_and_verifies_grid(world):
    cam = CameraModel(*world.camera_model())
    world.clear_ink()
    world.move_sheet(0.215, 0.02, math.pi / 2 + 0.12)
    job, forces, tracker = run_job(world, cam, grid_lines_mm())
    assert job.ok, job.error
    pressing = forces[forces > 0.05]
    assert np.median(pressing) == pytest.approx(C.TOUCH_TARGET, abs=0.2)
    assert forces.max() < C.TOUCH_MAX + 1.0
    rgb = world.render_overhead()
    coverage = check_lines(rgb, cam, find_sheet(rgb, cam).as_tuple(), grid_lines_mm())
    assert min(coverage) >= C.LINE_OK_COVERAGE, coverage


def test_replans_when_the_sheet_moves(world):
    """The sheet is slid 3 cm while the second line is being drawn: the arm must lift, wait, and redraw it where
    the paper is now, so every line still reads as drawn."""
    cam = CameraModel(*world.camera_model())
    world.clear_ink()
    world.move_sheet(0.21, 0.0, math.pi / 2)
    job, _, _ = run_job(world, cam, grid_lines_mm(), events={600: lambda w: w.slide_sheet(0.015, -0.025, 0.0)})
    assert job.ok, job.error
    assert job.replans >= 1
    rgb = world.render_overhead()
    coverage = check_lines(rgb, cam, find_sheet(rgb, cam).as_tuple(), grid_lines_mm())
    assert min(coverage) >= C.LINE_OK_COVERAGE, coverage


def test_safety_sees_a_hand_but_not_the_arm(world):
    cam = CameraModel(*world.camera_model())
    cam_d = safety.depth_camera(cam)
    world.move_sheet(0.21, 0.0, math.pi / 2)
    pose = (0.21, 0.0, math.pi / 2)
    ik = PenIK(world.model, world.pen.tip_body, world.pen.axis_body, World.ARM)
    r = ik.solve([0.21, 0.0, 0.03], np.array([0.0, 0.0, 0.5, 1.2, 0.0]), max_tilt_deg=20)
    world.set_arm_qpos(r.q)                                        # the arm right over the paper
    world.step(1)
    assert not safety.check(*world.render_depth(), cam_d, pose).present
    world.reach_in(3.0)
    for _ in range(75):
        world.step(10)
    hit = safety.check(*world.render_depth(), cam_d, pose)
    assert hit.present and hit.max_height_cm > 3
    for _ in range(150):
        world.step(10)
    world.set_arm_qpos(np.array(C.HOME_Q))


def test_solver_never_loses():
    rng = random.Random(0)
    perfect = Difficulty(1.0)
    for start in ("X", "O"):
        for _ in range(200):
            b, turn = [""] * 9, "X"
            while not winner(b) and not full(b):
                i = choose(b, turn, perfect, rng).cell if turn == start else rng.choice([j for j in range(9) if not b[j]])
                b[i] = turn
                turn = other(turn)
            w = winner(b)
            assert w is None or w[0] == start


def test_mark_classifier_on_fresh_boards():
    """Boards the network never saw (different random seed), through the full camera pipeline."""
    from progress import Progress
    import train_marks
    from buddy.perception.board import MarkReader
    X, Y, _ = train_marks.generate(40, seed=12345, prog=Progress("test"))
    import torch
    reader = MarkReader()
    with torch.no_grad():
        p = torch.softmax(reader.net(torch.from_numpy(X[:, None])), 1).numpy()
    assert (p.argmax(1) == Y).mean() >= 0.99
