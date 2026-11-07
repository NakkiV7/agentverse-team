"""Tic-tac-toe: exact minimax with adaptive difficulty. The robot plays X, the player O.

Board: list of 9 marks "X" | "O" | "" (row by row). Cells 0..8.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(board: list[str]) -> str | None:
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


def legal_moves(board: list[str]) -> list[int]:
    return [i for i, v in enumerate(board) if not v]


def minimax(board: list[str], me: str, turn: str) -> int:
    w = winner(board)
    if w:
        return 1 if w == me else -1
    moves = legal_moves(board)
    if not moves:
        return 0
    other = "O" if turn == "X" else "X"
    scores = []
    for m in moves:
        board[m] = turn
        scores.append(minimax(board, me, other))
        board[m] = ""
    return max(scores) if turn == me else min(scores)


def best_move(board: list[str], me: str = "X") -> int:
    other = "O" if me == "X" else "X"
    best, best_s = None, -2
    for m in legal_moves(board):
        board[m] = me
        s = minimax(board, me, other)
        board[m] = ""
        if s > best_s:
            best, best_s = m, s
    assert best is not None
    return best


@dataclass
class Difficulty:
    """Plays perfectly with probability 1 - p, a random legal move with probability p.
    p rises after a robot win and falls after a player win so games stay close."""
    p: float = 0.5
    step: float = 0.15

    def pick(self, board: list[str], me: str = "X", rng: random.Random | None = None) -> int:
        rng = rng or random
        if rng.random() < self.p:
            return rng.choice(legal_moves(board))
        return best_move(board, me)

    def after_game(self, result: str) -> None:
        if result == "robot":
            self.p = min(0.9, self.p + self.step)
        elif result == "player":
            self.p = max(0.0, self.p - self.step)


@dataclass
class Game:
    """Game state driven by Percepts: the board is whatever the camera sees, memory only spots differences."""
    robot: str = "X"
    player: str = "O"
    board: list[str] = field(default_factory=lambda: [""] * 9)
    turn: str = "O"                       # the player goes first
    difficulty: Difficulty = field(default_factory=Difficulty)

    def diff(self, seen: list[str]) -> dict:
        """Compare the camera's board with memory. Returns {"new": [(cell, mark)], "gone": [cell]}."""
        new = [(i, s) for i, (s, b) in enumerate(zip(seen, self.board)) if s and s != b]
        gone = [i for i, (s, b) in enumerate(zip(seen, self.board)) if b and not s]
        return {"new": new, "gone": gone}

    def check_player(self, seen: list[str]) -> str:
        """'ok' | 'none' | 'cheat' (two marks, wrong mark, or a mark during the robot's turn)."""
        d = self.diff(seen)
        if d["gone"]:
            return "cheat"
        if len(d["new"]) == 0:
            return "none"
        if len(d["new"]) > 1 or d["new"][0][1] != self.player or self.turn != self.player:
            return "cheat"
        self.board[d["new"][0][0]] = self.player
        self.turn = self.robot
        return "ok"

    def robot_move(self) -> int:
        assert self.turn == self.robot
        cell = self.difficulty.pick(self.board, self.robot)
        self.board[cell] = self.robot
        self.turn = self.player
        return cell

    def result(self) -> str | None:
        w = winner(self.board)
        if w == self.robot:
            return "robot"
        if w == self.player:
            return "player"
        if not legal_moves(self.board):
            return "draw"
        return None
