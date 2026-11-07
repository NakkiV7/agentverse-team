"""Teach Buddy a game it has never seen, then watch it play from the first move. Typed, no arm, no audio.

    .venv/bin/python tools/teach_dry_run.py                 # Footsteps, taught in two sentences
    .venv/bin/python tools/teach_dry_run.py --vague         # a vague explanation: Buddy has to ask first
    .venv/bin/python tools/teach_dry_run.py --misere        # "three in a row and you lose"

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
args = ap.parse_args()
script = SCRIPTS["vague" if args.vague else "misere" if args.misere else "footsteps"]

print("== TEACH ==")
teacher = Teacher()
spec = None
for line in script:
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

print("\n== PLAY ==  (player = random legal moves, Buddy = solver at 75% strength)")
rng = random.Random(7)
s = State.start(spec, "X")                          # the player is X and goes first
diff = Difficulty(strength=0.75)
human, robot = "X", "O"
while not result(spec, s):
    if s.to_move == human:
        c = rng.choice(legal_moves(spec, s))
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
