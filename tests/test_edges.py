"""Reading dots-and-boxes edges from images the simulator makes, through the same camera, sheet-finding and
straightening pipeline as at run time (like tools/train_marks.py builds its boards).

    .venv/bin/python -m pytest -q tests/test_edges.py
"""
import math

import numpy as np
import pytest

from buddy import config as C
from buddy.game import dotsboxes as D
from buddy.game import marks as M
from buddy.perception.board import rectify
from buddy.perception.edges import DRAWN_COVERAGE, drawn_mask, read_dots, read_edges
from buddy.perception.sheet import CameraModel, SheetTracker, find_sheet
from buddy.sim.world import World

INKS = [C.INK_RGB, C.PLAYER_INK_RGB, (20, 20, 120), (60, 60, 60)]


def paint(world, strokes, rgb, radius):
    for st in strokes:
        st = np.asarray(st, float)
        for a, b in zip(st[:-1], st[1:]):
            world.paint(a, b, rgb=rgb, radius_mm=radius)


def random_board(world, rng, n):
    """Dots, a random half of the edges (Penny's straight or the player's wobbly, any pen), some claimed boxes;
    the sheet anywhere in reach, random lighting and camera noise. Returns the truth mask."""
    world.clear_ink()
    world.move_sheet(rng.uniform(0.19, 0.235), rng.uniform(-0.035, 0.035), math.pi / 2 + rng.uniform(-0.35, 0.35))
    world.step(2)
    world.model.light_diffuse[:] = world.light0 * rng.uniform(0.75, 1.2)
    world.noise = rng.uniform(1.0, 6.0)
    paint(world, M.dot_grid(n), C.INK_RGB, C.INK_RADIUS_MM)
    truth = 0
    for e in range(D.n_edges(n)):
        if rng.random() < 0.5:
            truth |= 1 << e
            if rng.random() < 0.5:
                paint(world, M.edge_stroke(n, e), INKS[rng.integers(4)], rng.uniform(0.5, 0.9))
            else:
                paint(world, M.hand_edge(n, e, rng), INKS[rng.integers(4)], rng.uniform(0.5, 1.0))
    for b in range(n * n):
        u = rng.random()
        if u < 0.3:
            paint(world, M.box_claim(n, b, "B" if rng.random() < 0.7 else "circle"), C.INK_RGB, C.INK_RADIUS_MM)
        elif u < 0.6:
            paint(world, M.hand_box_claim(n, b, rng), C.PLAYER_INK_RGB, 0.8)
    return truth


@pytest.fixture(scope="module")
def world():
    w = World()
    w.light0 = w.model.light_diffuse.copy()
    yield w
    w.close()


def test_reads_edges_and_dots_on_fresh_boards(world):
    cam = CameraModel(*world.camera_model())
    rng = np.random.default_rng(2026)
    boards, edges_checked = 0, 0
    for k in range(16):
        n = 2 if k % 2 == 0 else 3
        truth = random_board(world, rng, n)
        rgb = world.render_overhead()
        sheet = find_sheet(rgb, cam)
        if sheet is None or sheet.confidence < SheetTracker.MIN_CONF:
            continue                                    # the app would not read a board from such a sighting either
        rect = rectify(rgb, cam, sheet.as_tuple())
        res = read_edges(rect, n)
        assert drawn_mask(res) == truth, [(e, r) for e, r in enumerate(res) if r.drawn != bool(truth >> e & 1)]
        assert all(r.sure for r in res), min(r.confidence for r in res)
        assert all(r.coverage >= 0.9 for e, r in enumerate(res) if truth >> e & 1)
        assert all(r.coverage <= 0.1 for e, r in enumerate(res) if not truth >> e & 1)
        assert min(read_dots(rect, n)) >= 0.3
        boards += 1
        edges_checked += len(res)
    world.model.light_diffuse[:] = world.light0
    assert boards >= 10 and edges_checked >= 150


def test_blank_board_reads_nothing(world):
    cam = CameraModel(*world.camera_model())
    world.clear_ink()
    world.move_sheet(0.21, 0.0, math.pi / 2)
    world.step(2)
    world.model.light_diffuse[:] = world.light0
    world.noise = C.CAMERA_NOISE
    paint(world, M.dot_grid(2), C.INK_RGB, C.INK_RADIUS_MM)
    rgb = world.render_overhead()
    rect = rectify(rgb, cam, find_sheet(rgb, cam).as_tuple())
    res = read_edges(rect, 2)
    assert drawn_mask(res) == 0 and max(r.coverage for r in res) < DRAWN_COVERAGE / 3
    assert min(read_dots(rect, 2)) >= 0.5
    world.clear_ink()
    assert max(read_dots(rectify(world.render_overhead(), cam, find_sheet(world.render_overhead(), cam).as_tuple()), 2)) < 0.1
