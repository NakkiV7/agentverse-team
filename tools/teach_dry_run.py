"""Teach Buddy a game it has never seen, then watch it play from the first move. Typed, no arm, no audio.

    .venv/bin/python tools/teach_dry_run.py                 # Footsteps, taught in two sentences
    .venv/bin/python tools/teach_dry_run.py --vague         # a vague explanation: Buddy has to ask first
    .venv/bin/python tools/teach_dry_run.py --misere        # "three in a row and you lose"
    .venv/bin/python tools/teach_dry_run.py -i              # interactive: explain your own game, then play it by typing cells

Needs Ollama with gemma4 (TEACH_MODEL). Exits 0 when a game is learned and played to the end.
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buddy.agent.teach import Teacher                                                    # noqa: E402
from buddy.game.rules import State, cell_name, legal_moves, play, result                 # noqa: E402
from buddy.game.solver import analyse                                                    # noqa: E402
from buddy.game.tictactoe import Difficulty                                              # noqa: E402

SCRIPTS = {
    "footsteps": ["New game, Buddy. We take turns putting our mark next to the last one, up, down or sideways.",
                  "If you can't go anywhere, you lose."],
    "vague": ["You just put marks next to each other and try not to get stuck.",
              "Next to the last mark that was played, and only up, down, left or right.",
              "The one who can't move loses."],
    "misere": ["It's like noughts and crosses, but backwards: if you make three in a row, you lose.",
               "Anywhere on the grid."],
}

ap = argparse.ArgumentParser()
ap.add_argument("--vague", action="store_true")
ap.add_argument("--misere", action="store_true")
ap.add_argument("-i", "--interactive", action="store_true", help="type the explanation and your moves yourself")
args = ap.parse_args()
script = SCRIPTS["vague" if args.vague else "misere" if args.misere else "footsteps"]


def typed_lines():
    print("Explain the game to Buddy in your own words (one or two sentences at a time; empty line to give up).")
    while True:
        try:
            line = input("You: ").strip()
        except EOFError:
            return
        if not line:
            return
        yield line


def typed_move(spec, s):
    """Ask the human for a cell: a number (0-based, row by row) or a name like 'top left' / 'centre'."""
    legal = legal_moves(spec, s)
    names = {cell_name(spec, c): c for c in range(spec.n)}
    while True:
        try:
            raw = input(f"Your move ({', '.join(str(c) for c in legal)} or a name like '{cell_name(spec, legal[0])}'): ").strip().lower()
        except EOFError:
            print("\nBye."); sys.exit(0)
        c = names.get(raw)
        if c is None and raw.isdigit():
            c = int(raw)
        if c in legal:
            return c
        print("  that square isn't allowed right now")

print("== TEACH ==")
teacher = Teacher()
spec = None
for line in (typed_lines() if args.interactive else script):
    if not args.interactive:
        print(f"Player: {line}")
    turn = teacher.hear(line, name="the player's game")
    print(f"Buddy ({turn.seconds:.1f}s): {turn.say}")
    if turn.done:
        spec = turn.spec
        break
if spec is None:
    print("Buddy could not learn the game"); sys.exit(1)

print("\n== THINK ==")
solver, a = analyse(spec)
print(f"Buddy: Let me think... ({a.nodes} positions in {a.seconds * 1000:.0f} ms) {a.speak(spec)} Want to try?")

print("\n== PLAY ==  " + ("(you are X, type your moves; Buddy = solver at 75% strength)" if args.interactive
                        else "(player = random legal moves, Buddy = solver at 75% strength)"))
rng = random.Random(7)
human, robot = ("X", "O")
s = State.start(spec, human if spec.first_player == "player" else robot)
diff = Difficulty(strength=0.75)
while not result(spec, s):
    if s.to_move == human:
        c = typed_move(spec, s) if args.interactive else rng.choice(legal_moves(spec, s))
        print(f"Player writes X in the {cell_name(spec, c)}")
    else:
        ch = solver.choose(s, diff, rng)
        print(f"Buddy: I'll take the {cell_name(spec, ch.cell)}.   [{ch.reason}; scores {ch.scores}]")
        c = ch.cell
    s = play(spec, s, c)
    for r in range(spec.rows):
        print("    " + " | ".join(s.board[r * spec.cols + c] or " " for c in range(spec.cols)))
r = result(spec, s)
who = "Buddy" if r[1] == robot else "the player"
if r[0] == "draw":
    why = "draw"
elif spec.end == "k_in_a_row_loses":
    why = f"{who} wins: the other side made {spec.k} in a row {r[2]} and loses"
elif r[2]:
    why = f"{who} wins with the line {r[2]}"
else:
    why = f"{who} wins: the other side cannot move"
print("\nResult:", why)
sys.exit(0)
