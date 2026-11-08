"""Letter strips Buddy writes beside a game (buddy/game/notes.py): what a request names, how 26 letters fit a
corner or a strip along the bottom, and that every stroke stays inside the free area."""
import numpy as np

from buddy.game import notes as N


def test_letters_of_reads_the_alphabet_and_named_letters():
    assert N.letters_of("the alphabet") == N.ALPHABET
    assert N.letters_of("every letter") == N.ALPHABET
    assert N.letters_of("A, E and P") == "AEP"
    assert N.letters_of("apple") == "APLE"
    assert N.letters_of("the letters a and b please") == "AB"
    assert N.letters_of("123") == ""


def _inside(cells, free, slack=1.5):
    pts = np.concatenate([np.concatenate(c.strokes()) for c in cells])
    return (pts[:, 0] >= free.x0 - slack).all() and (pts[:, 0] <= free.x1 + slack).all() \
        and (pts[:, 1] >= free.y0 - slack).all() and (pts[:, 1] <= free.y1 + slack).all()


def test_alphabet_fits_a_hangman_corner_in_four_rows():
    free = N.Rect(2, 8, 64, 62)               # the top-right quarter beside the gallows
    cells = N.layout("the alphabet", free)
    assert "".join(c.letter for c in cells) == N.ALPHABET
    assert len({round(c.origin[1], 2) for c in cells}) == 4
    assert cells[0].width == N.WIDTH_MM and _inside(cells, free)
    assert cells[0].origin[1] > cells[-1].origin[1]          # rows read downwards from the top


def test_alphabet_fits_a_strip_along_the_bottom_in_two_rows():
    free = N.Rect(-64, -62, 64, -34)
    cells = N.layout("the alphabet", free, start="bottom")
    assert len(cells) == 26 and len({round(c.origin[1], 2) for c in cells}) == 2 and _inside(cells, free)


def test_letters_shrink_for_a_small_area_and_give_up_when_there_is_no_room():
    small = N.Rect(0, 0, 30, 60)
    cells = N.layout("the alphabet", small)
    assert cells and cells[0].width < N.WIDTH_MM and _inside(cells, small)
    assert N.layout("the alphabet", N.Rect(0, 0, 10, 10)) == []


def test_cross_goes_corner_to_corner_through_the_letter():
    cell = N.layout("A", N.Rect(0, 0, 20, 20))[0]
    (a, b), = cell.cross()
    assert a[0] < cell.origin[0] < cell.origin[0] + cell.width < b[0]
    assert a[1] < cell.origin[1] < cell.origin[1] + cell.height < b[1]


def test_describe():
    cells = N.layout("the alphabet", N.Rect(2, 8, 64, 62))
    assert N.describe(cells, "top right corner") == "the alphabet in 4 rows in the top right corner"
    assert N.describe(N.layout("AEP", N.Rect(2, 8, 64, 62)), "top right corner") == "AEP in 1 row in the top right corner"
