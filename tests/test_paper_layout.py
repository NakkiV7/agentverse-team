from types import SimpleNamespace

import numpy as np
import pytest

from buddy import config as C
from buddy.game import marks as M
from buddy.game.hangman import Hangman
from buddy.paper import layout as L


def brain(game, n=2):
    return SimpleNamespace(game_name=game, hangman=Hangman("APPLE") if game == "hangman" else None,
                           dots=SimpleNamespace(n=n) if game == "dots and boxes" else None)


@pytest.mark.parametrize("game,n", [("hangman", 2), ("noughts and crosses", 2),
                                    ("dots and boxes", 2), ("dots and boxes", (3, 5))])
def test_all_places_are_on_sheet_and_outside_complete_game(game, n):
    b = brain(game, n)
    regions = L.places(b)
    assert set(regions) == set(L.PLACES)
    assert any(regions.values())
    for rects in regions.values():
        for r in rects:
            assert L.SHEET.contains(r)
            assert not any(r.overlaps(g) for g in L.footprint(b))


def test_dots_reserves_camera_scribble_window_too():
    b = brain("dots and boxes")
    box, = L.footprint(b)
    pts = np.array(M.dot_positions_mm(2))
    pad = M.dot_spacing(2) * .35
    assert box.contains(L.Rect(*((pts.min(axis=0) - pad)), *((pts.max(axis=0) + pad))))


def test_wrap_reflows_advisory_newlines_and_keeps_every_character():
    cells = L.layout_text("ABCDEFGHIJKLM\nNOPQRSTUVWXYZ", L.Rect(2, 8, 64, 62), 6)
    assert "".join(c.char for c in cells) == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert len({c.box.y0 for c in cells}) == 4
    assert all(not a.box.overlaps(b.box) for i, a in enumerate(cells) for b in cells[i + 1:])
    assert all(c.box.contains(c.glyph) for c in cells)
    # An overlong explicit row breaks at the place boundary.
    assert cells[7].box.y1 <= cells[0].box.y0
    # Newlines may be reflowed when insisting on them would waste a row.
    assert len(L.layout_text("A\nB\nC", L.Rect(0, 0, 30, 12))) == 3


@pytest.mark.parametrize("size", [4.99, 0, -1, float("nan"), float("inf"), True, "6"])
def test_refuses_unreadable_sizes(size):
    with pytest.raises(ValueError, match="at least 5"):
        L.layout_text("A", L.PAGE, size)


def test_spaces_newlines_and_font_are_distinct_from_unsupported_symbols():
    cells = L.layout_text("A 2\nB", L.PAGE)
    assert [c.index for c in cells] == [0, 1, 2, 4]
    assert cells[1].strokes() == []
    for text in ("a", "é", "cat!", "A\tB"):
        with pytest.raises(ValueError, match="capitals"):
            L.layout_text(text, L.PAGE)


def test_digits_and_punctuation_are_finite_drawable_polylines():
    for letter in "0123456789.:-/":
        assert letter in M.CAPITAL_STROKES
        for stroke in M.capital_letter(letter):
            assert len(stroke) >= 2
            assert np.isfinite(stroke).all()
            assert all(np.linalg.norm(b - a) > 0 for a, b in zip(stroke, stroke[1:]))


def test_named_regions_filter_ik_and_existing_items():
    b = brain("hangman")
    occupied = L.Rect(2, 8, 132, 98)
    places = L.places(b, reachable=lambda pts: (all(x < 0 for x, y in pts), "left only"), occupied=[occupied])
    assert not places["top right corner"]
    assert places["top left corner"]
    assert all(not r.overlaps(occupied) for rs in places.values() for r in rs)


def test_reach_patches_follow_translated_sheet():
    old, new = L.reach_patches(), L.reach_patches((C.SHEET_POSE0[0] - .03, 0, C.SHEET_POSE0[2]))
    assert new[0].y0 == pytest.approx(old[0].y0 - 30, abs=.001)
    assert new[0].y1 == pytest.approx(old[0].y1 - 30, abs=.001)


def test_annotation_stays_in_the_reserved_character_box():
    cell, = L.layout_text("X", L.PAGE)
    for op in ("cross_out", "underline", "circle", "tick"):
        for stroke in L.mark_strokes(op, cell.box):
            for x, y in stroke:
                assert cell.box.x0 - 1e-8 <= x <= cell.box.x1 + 1e-8
                assert cell.box.y0 - 1e-8 <= y <= cell.box.y1 + 1e-8
    assert len(cell.cross()) == 1


def test_selected_game_footprint_ignores_old_hangman_and_dots_objects():
    b = brain("noughts and crosses")
    expected = L.footprint(b)
    b.hangman = Hangman("APPLE")
    b.dots = SimpleNamespace(n=2)
    assert L.footprint(b) == expected
    b.entry = SimpleNamespace(kind="grid")
    b.game_name = "a taught game with dots in its name"
    assert L.footprint(b) == expected
