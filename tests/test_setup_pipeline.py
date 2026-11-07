"""End-to-end check of the game setup, without speech: see the sheet, plan the grid on it, draw it, verify it.

    .venv/bin/python -m pytest -q tests/test_setup_pipeline.py

The sheet is put somewhere else than the default (shifted and rotated): the arm only learns where it is through
the camera, so drawing the grid in the right place proves the setup is driven by perception.
"""
import math

import numpy as np
import pytest

from buddy import config as C
from buddy.motion.controller import ArmController, Job
from buddy.motion.drawing import grid_lines_mm, sheet_to_world
from buddy.motion.ik import PenIK
from buddy.perception.sheet import CameraModel, check_lines, find_sheet
from buddy.sim.world import World


@pytest.fixture(scope="module")
def world():
    w = World()
    yield w
    w.close()


def run_until_done(w, ctl, job, max_s=120):
    spt = int(round(1 / C.CONTROL_HZ / C.TIMESTEP))
    forces = []
    for _ in range(int(max_s * C.CONTROL_HZ)):
        w.command(ctl.tick(w.touch()))
        w.step(spt)
        forces.append(w.touch())
        if job.done.is_set():
            break
    return np.array(forces)


def test_finds_sheet_anywhere(world):
    cam = CameraModel(*world.camera_model())
    for pose in [(0.21, 0.0, math.pi / 2), (0.225, 0.03, math.pi / 2 + 0.25), (0.20, -0.025, math.pi / 2 - 0.15)]:
        world.move_sheet(*pose)
        world.step(2)
        s = find_sheet(world.render_overhead()[0], cam)
        assert s is not None
        assert abs(s.x - pose[0]) < 0.003 and abs(s.y - pose[1]) < 0.003
        assert abs(math.degrees(s.yaw - pose[2])) < 1.0
        assert s.confidence > 0.8


def test_draws_and_verifies_grid(world):
    cam = CameraModel(*world.camera_model())
    world.clear_ink()
    world.move_sheet(0.215, 0.02, math.pi / 2 + 0.12)
    world.step(2)
    seen = find_sheet(world.render_overhead()[0], cam)
    lines = grid_lines_mm()
    ik = PenIK(world.model, world.pen.tip_body, world.pen.axis_body, World.ARM)
    ctl = ArmController(ik, world.arm_q(), lambda kind, msg: None)
    job = ctl.submit(Job("grid", strokes=[sheet_to_world(seen.as_tuple(), l) for l in lines], home=True))
    forces = run_until_done(world, ctl, job)
    assert job.ok, job.error
    pressing = forces[forces > 0.05]
    assert np.median(pressing) == pytest.approx(C.TOUCH_TARGET, abs=0.2)
    assert forces.max() < C.TOUCH_MAX + 1.0
    after = find_sheet(world.render_overhead()[0], cam)
    coverage = check_lines(world.render_overhead()[0], cam, after.as_tuple(), lines)
    assert min(coverage) >= C.LINE_OK_COVERAGE, coverage
