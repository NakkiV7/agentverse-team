"""Rectangular dots-and-boxes boards, the size parser, and spoken line names."""
import random

from buddy.game import dotsboxes as D
from buddy.game import marks as M
from buddy.perception.edges import read_edges


def test_rectangle_geometry():
    n = (3, 4)
    assert D.n_edges(n) == 31 and D.n_boxes(n) == 12
    assert len(M.dot_positions_mm(n)) == 20
    # every box has 4 distinct edges, every edge borders 1 or 2 boxes, and all edges are used
    used = set()
    for edges in D.box_edges(n):
        assert len(set(edges)) == 4
        used |= set(edges)
    assert used == set(range(31))
    assert all(1 <= len(bs) <= 2 for bs in D.edge_boxes(n))
    assert D.edge_name(n, 0) == "the top side of the box in row 1, column 1"
    assert D.edge_name((1, 2), 0) == "the top side of the left box"
    assert D.edge_name(2, 0) == "the top side of the top left box"


def test_size_parser_and_limits():
    assert D.parse_size("three by four") == (4, 3)              # across by down -> (rows, cols)
    assert D.parse_size("4x3 please") == (3, 4)
    assert D.parse_size("five across and three down") == (3, 5)
    assert D.parse_size("3 down and 5 across") == (3, 5)
    assert D.parse_size("a six grid") == (6, 6)
    assert D.parse_size("dots and boxes") is None
    assert D.size_ok((2, 2))[0] and D.size_ok((5, 6))[0]
    assert not D.size_ok((6, 6))[0] and not D.size_ok((1, 1))[0]
    assert D.size_words((3, 4)) == "4 across and 3 down" and D.size_words(2) == "2 by 2"


def test_rectangle_game_plays_out():
    n = (2, 3)
    sol = D.Solver(n)
    rng = random.Random(0)
    s = D.State.start(n, ("P", "B"))
    while D.result(s) is None:
        e = sol.choose(s, D.Difficulty(1.0), rng).cell if s.to_move == "B" else D.greedy_player(s, rng)
        s = D.play(s, e)
    assert s.score("P") + s.score("B") == 6


def test_edges_read_on_a_rectangle():
    """A synthetic straightened sheet with a 2x3 board: dots plus three drawn edges read back exactly."""
    import numpy as np
    import cv2
    from buddy.perception.board import mm_to_rect_px
    n = (2, 3)
    rect = np.full((630, 891, 3), 240, np.uint8)
    for p in M.dot_positions_mm(n):
        x, y = np.round(mm_to_rect_px([p])[0]).astype(int)
        cv2.circle(rect, (int(x), int(y)), 4, (30, 30, 30), -1)
    drawn = [0, 7, 12]
    for e in drawn:
        a, b = M.edge_ends_mm(n, e)
        pa, pb = np.round(mm_to_rect_px([a, b])).astype(int)
        cv2.line(rect, tuple(int(v) for v in pa), tuple(int(v) for v in pb), (30, 30, 30), 3)
    res = read_edges(rect, n)
    assert [e for e, r in enumerate(res) if r.drawn] == drawn
    assert all(r.sure for r in res)
