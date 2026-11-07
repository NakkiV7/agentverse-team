from buddy.game.tictactoe import Difficulty, Game, best_move, winner


def test_blocks_and_wins():
    assert best_move(["X", "X", "", "O", "O", "", "", "", ""]) == 2       # take the win
    assert best_move(["O", "O", "", "X", "", "", "", "", ""]) == 2        # block
    assert winner(["X", "X", "X", "", "", "", "", "", ""]) == "X"


def test_perfect_player_never_loses():
    import random
    rng = random.Random(0)
    for _ in range(200):
        b = [""] * 9
        turn = "O"
        while winner(b) is None and "" in b:
            if turn == "X":
                b[best_move(b)] = "X"
            else:
                b[rng.choice([i for i, v in enumerate(b) if not v])] = "O"
            turn = "O" if turn == "X" else "X"
        assert winner(b) != "O"


def test_cheat_detection():
    g = Game()
    assert g.check_player(["O", "", "", "", "", "", "", "", ""]) == "ok"
    assert g.turn == "X"
    assert g.check_player(["O", "O", "", "", "", "", "", "", ""]) == "cheat"   # out of turn
    g.robot_move()
    seen = list(g.board); seen[8] = "O"; seen[7] = "O"
    assert g.check_player(seen) == "cheat"                                      # two marks


def test_difficulty_adapts():
    d = Difficulty(p=0.5)
    d.after_game("robot"); assert d.p > 0.5
    d.after_game("player"); d.after_game("player"); assert d.p < 0.5
