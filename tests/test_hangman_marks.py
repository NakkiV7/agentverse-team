import numpy as np
import pytest

from buddy import config as C
from buddy.game import marks as M
from buddy.motion.drawing import sheet_to_world
from tools.draw_alphabet import alphabet_strokes


def test_font_and_centred_seven_letter_row():
    assert set(M.CAPITAL_STROKES) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.:-/")
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        pts = np.concatenate(M.capital_letter(c))
        assert np.ptp(pts[:, 1]) == 25
        assert pts[:, 0].min() >= 0 and pts[:, 0].max() <= 14
    for n in range(4, 8):
        pts = np.concatenate(M.hangman_dashes(n))
        assert len(M.hangman_dashes(n)) == n
        assert pts[:, 0].min() == -pts[:, 0].max()
        assert np.ptp(pts[:, 0]) <= 128


def test_all_strokes_within_grid_and_shoulder_reach_band():
    strokes = alphabet_strokes() + M.hangman_dashes(7)
    strokes += [s for i in range(7) for c in M.CAPITAL_STROKES for s in M.hangman_letter(c, i, 7)]
    strokes += [s for i in range(1, 7) for s in M.gallows_part(i)]
    pts = np.concatenate(strokes)
    assert np.abs(pts).max() <= C.GRID_SIZE_MM / 2
    # Sample whole segments: the nearest point to the shoulder can be inside one.
    samples = np.concatenate([np.linspace(a, b, 50) for a, b in M.stroke_segments(strokes)])
    world = sheet_to_world(C.SHEET_POSE0, samples)
    radii = np.linalg.norm(world - [0.039, 0], axis=1)
    assert radii.min() >= C.REACH_MIN and radii.max() <= C.REACH_MAX


def test_invalid_layout_requests():
    for value in (0, 3, 8):
        with pytest.raises(ValueError):
            M.hangman_dashes(value)
    for value in (0, 7):
        with pytest.raises(ValueError):
            M.gallows_part(value)
    with pytest.raises(ValueError):
        M.hangman_letter("A", -1, 4)
    with pytest.raises(ValueError):
        M.capital_letter("a")
