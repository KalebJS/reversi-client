import unittest

import numpy as np

from state import ReversiGameState


class State(unittest.TestCase):
    def test_state_moves_left(self):
        board_zeroes = np.zeros((8, 8), dtype=int)
        state = ReversiGameState(board_zeroes, 1)

        assert state.turns_remaining() == 64

        board_mix = np.zeros((8, 8), dtype=int)
        board_mix[3, 3:5] = board_mix[4, 4:6] = 1
        board_mix[3, 4] = board_mix[4, 3] = 2
        state = ReversiGameState(board_mix, 1)

        assert state.turns_remaining() == 60


if __name__ == '__main__':
    unittest.main()
