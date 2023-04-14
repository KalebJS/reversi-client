import unittest

import numpy as np

from reversi_bot import ReversiBot
from state import ReversiGameState


class TestReversiGameState(unittest.TestCase):
    def setUp(self):
        # Initialize a ReversiGameState object to use in the tests
        board = np.zeros((8, 8), dtype=int)
        board[3, 3:5] = board[4, 4:6] = 1
        board[3, 4] = board[4, 3] = 2
        self.state = ReversiGameState(board, 1)
        self.bot = ReversiBot(1)

    def test_heuristic_caching_individual_state(self):
        start = self.bot.evaluations
        self.bot.heuristic(self.state)
        # regenerate states with the same board
        for _ in range(1000):
            state = ReversiGameState(self.state.board, self.state.turn)
            self.bot.heuristic(state)

        assert self.bot.evaluations == start + 1

    def test_heuristic_caching_multiple_states(self):
        start = self.bot.evaluations
        s = {ReversiGameState(np.random.randint(0, 2, size=(8, 8)), 1) for _ in range(1000000)}
        for state in s:
            self.bot.heuristic(state)

        for state in s:
            self.bot.heuristic(state)

        assert self.bot.evaluations == start + len(s)


if __name__ == '__main__':
    unittest.main()
