"""Short, factual speech from the checked game state, without taking a turn.

The executor checks the camera before asking for a hint. These helpers only read
the brain's committed board and solver records; they never draw or call a model.
"""
from __future__ import annotations

from copy import deepcopy
import re

from ..game import dotsboxes as dots
from ..game import rules, tictactoe
from ..game.hangman import level_for_difficulty


_GRID_STATES = {"PLAYER_TURN", "ROBOT_TURN", "CONFIRM_MARK"}
_SMALL = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
          "seventeen", "eighteen", "nineteen")


def _number(n: int) -> str:
    if n < 20:
        return _SMALL[n]
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    return tens[n // 10] + (" " + _SMALL[n % 10] if n % 10 else "")


def _count(n: int, noun: str, plural: str | None = None) -> str:
    return f"{_number(n)} {noun if n == 1 else plural or noun + 's'}"


def _kind(brain) -> str | None:
    state = getattr(brain, "state", "IDLE")
    if state in _GRID_STATES:
        return "grid"
    if state in ("DOTS_PLAYER_TURN", "DOTS_ROBOT_TURN"):
        return "dots"
    if state == "HANGMAN_TURN":
        return "hangman"
    if state == "GAME_OVER":
        kind = getattr(getattr(brain, "entry", None), "kind", None)
        if kind:
            return "dots" if kind == "dots_and_boxes" else kind
        if getattr(brain, "hangman", None) is not None:
            return "hangman"
        if getattr(brain, "dots", None) is not None:
            return "dots"
        return "grid"
    return None


def _grid_state(brain):
    """Use the checked board, with the taught game's recorded last square."""
    state = getattr(brain, "gstate", None)
    if state is None:
        return None
    return rules.State(tuple(brain.board), state.last, state.to_move)


def _grid_result(brain):
    spec = getattr(brain, "spec", None)
    if spec is not None:
        state = _grid_state(brain)
        return rules.result(spec, state) if state is not None else None
    won = tictactoe.winner(brain.board)
    return ("win", won[0], won[1]) if won else ("draw", None, None) if tictactoe.full(brain.board) else None


def _ending(result, player) -> str:
    if result is None:
        return "The game is over."
    if result[0] == "draw":
        return "The game was a draw."
    return "You won this game." if result[1] == player else "I won this game."


def _cell_name(brain, cell: int) -> str:
    spec = getattr(brain, "spec", None)
    return rules.cell_name(spec, cell) if spec is not None else tictactoe.NAMES[cell]


def _edge_name(n, edge: int) -> str:
    return re.sub(r"\d+", lambda m: _number(int(m[0])), dots.edge_name(n, edge))


def score_text(brain) -> str:
    """Describe the score or revealed letters and whose turn it is."""
    kind = _kind(brain)
    state = getattr(brain, "state", "IDLE")
    if kind == "hangman" and getattr(brain, "hangman", None) is not None:
        game = brain.hangman
        found = len(game.pattern) - game.pattern.count("_")
        score = (f"You've found {_number(found)} of {_count(len(game.word), 'letter')}, "
                 f"with {_count(game.remaining, 'miss', 'misses')} left.")
        ending = {"win": "You found the word, well done.", "loss": "The game is over."}.get(game.result())
        return score + " " + (ending or "It's your turn to choose a letter.")
    if kind == "dots" and getattr(getattr(brain, "dots", None), "state", None) is not None:
        game = brain.dots.state
        player, robot = game.players
        score = f"You have {_count(game.score(player), 'box', 'boxes')} and I have {_count(game.score(robot), 'box', 'boxes')}."
        progress = f"We've drawn {_number(game.edges.bit_count())} of {_number(dots.n_edges(brain.dots.n))} lines"
        if state == "GAME_OVER" or dots.result(game) is not None:
            return score + " " + progress + "; " + _ending(dots.result(game), player).lower()
        turn = "your" if game.to_move == player else "my"
        return score + " " + progress + f", and it's {turn} turn."
    if kind == "grid" and getattr(brain, "player_sym", None) in ("X", "O"):
        player = brain.player_sym
        score = (f"You have {_count(brain.board.count(player), 'mark')} and "
                 f"I have {_count(brain.board.count(tictactoe.other(player)), 'mark')}.")
        result = _grid_result(brain)
        if state == "GAME_OVER" or result is not None:
            return score + " " + _ending(result, player)
        turn = "I need to check your last mark first." if state == "CONFIRM_MARK" else (
            "It's your turn." if state == "PLAYER_TURN" else "It's my turn.")
        return score + " " + turn
    return "We aren't playing a game just now."


def explain_last_move(brain) -> str:
    """Explain only what the recorded choice or checked hangman guess supports."""
    kind = _kind(brain)
    if kind is None:
        return "We aren't playing a game just now."
    if kind == "hangman":
        game = getattr(brain, "hangman", None)
        if game is None:
            return "I haven't drawn a word yet."
        if getattr(brain, "_hangman_pending", None) is not None:
            return "I still need to check the drawing for your last letter."
        last = getattr(game, "last_guess", None)
        drawing = getattr(getattr(brain, "app", None), "last_strokes", None)
        if last is None and drawing:
            match = re.fullmatch(r"(?:again: )*hangman letter ([A-Z])", drawing[0])
            last = match[1] if match and match[1] in game.guessed else None
        if last in game.guessed:
            if last in game.word:
                return f"I filled in every {last} because you found that letter in the word."
            return f"I added a gallows part because {last} isn't in the word."
        if not game.guessed:
            return "The dashes leave a space for each letter of the word."
        return "I haven't kept which letter I drew last, so I can't explain that drawing for certain."
    choice = getattr(brain, "last_choice", None)
    if choice is None:
        return "I haven't chosen a move in this game yet."
    if getattr(brain, "busy", False):
        return "Let me finish checking my drawing first."
    scores = choice.scores
    if choice.cell not in scores:
        return "I haven't kept enough detail to explain that move for certain."
    top = max(scores.values())
    value = scores[choice.cell]
    if not choice.played_best and value < top:
        return "I played a gentler move than my best there, to keep it close."
    if kind == "dots":
        name = _edge_name(brain.dots.n, choice.cell)
        return f"I chose {name} because it looked best for gaining boxes." if value == top else (
            "That line wasn't the strongest one I found.")
    name = "the " + _cell_name(brain, choice.cell)
    if value != top:
        return f"{name.capitalize()} wasn't the strongest square I found."
    if getattr(getattr(brain, "solver", None), "exhausted", False):
        return f"I chose {name} because it looked strongest from the moves I could check."
    others = [v for c, v in scores.items() if c != choice.cell]
    if value >= 0 and others and all(v < 0 for v in others):
        return f"{name.capitalize()} was my only way to avoid losing with best play."
    if value > 0:
        return f"I chose {name} because it gave me a way to win with best play."
    if value == 0:
        return f"I chose {name} because it could keep the game level with best play."
    return f"{name.capitalize()} was the best square I found, though you could still win with best play."


def hint_text(brain) -> str:
    """Suggest a legal move without playing, drawing, or altering the solver."""
    kind = _kind(brain)
    if kind is None:
        return "We aren't playing a game just now."
    if brain.state == "GAME_OVER":
        return "That game is finished, so there isn't another move to suggest."
    if brain.state == "CONFIRM_MARK" or getattr(brain, "_hangman_pending", None) is not None:
        return "Let's check your last mark before choosing another."
    if getattr(brain, "busy", False):
        return "Let me finish checking the paper first."
    if brain.state in ("ROBOT_TURN", "DOTS_ROBOT_TURN"):
        return "It's my turn just now, so let's look at your next move afterwards."
    if kind == "hangman":
        game = getattr(brain, "hangman", None)
        difficulty = getattr(brain, "difficulty", None)
        if game is not None and game.result() is not None:
            return "That game is finished, so there isn't another letter to suggest."
        if game is not None and level_for_difficulty(difficulty if difficulty is not None else game.level) == "impossible":
            return "There are no hints on impossible, but you can ask for an easier level."
        hint = game.hint(difficulty) if game is not None else None
        return f"You could try {hint}, if you like." if hint else "I haven't a letter to suggest just now."
    if kind == "dots":
        play = getattr(brain, "dots", None)
        if play is None or play.state is None:
            return "I need to check the board before suggesting a line."
        moves = dots.legal_moves(play.state)
        captures = [e for e in moves if dots.completed_by(play.n, play.state.edges, e)]
        safe = [e for e in moves if dots.is_safe(play.n, play.state.edges, e)]
        if captures:
            return f"You could take a box with {_edge_name(play.n, captures[0])}, if you like."
        if safe:
            return f"You could try {_edge_name(play.n, safe[0])}, if you like. It won't leave a box ready for me."
        return "I can't see a line that avoids leaving a box for me, so it's your choice."
    if getattr(brain, "player_sym", None) not in ("X", "O") or _grid_result(brain) is not None:
        return "There isn't a move to suggest just now."
    if getattr(brain, "spec", None) is not None:
        state = _grid_state(brain)
        solver = getattr(brain, "solver", None)
        if state is None or solver is None or state.to_move != brain.player_sym:
            return "I need to check the board before suggesting a square."
        scores = deepcopy(solver).move_scores(state)
    else:
        scores = tictactoe.move_scores(tuple(brain.board), brain.player_sym)
    scores = {c: v for c, v in scores.items() if not brain.board[c] and c not in getattr(brain, "ignored", ())}
    if not scores:
        return "There isn't a free square I can suggest just now."
    cell = max(sorted(scores), key=scores.get)
    return f"You could try the {_cell_name(brain, cell)}, if you like."
