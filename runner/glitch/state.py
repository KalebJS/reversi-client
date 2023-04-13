from functools import lru_cache
import itertools
from typing import Tuple
import numpy as np


class ReversiGameState:
    def __init__(self, board, turn):
        self.board_dim = 8  # Reversi is played on an 8x8 board
        self.board = board
        self.turn = turn  # Whose turn is it

    def __capture_will_occur(self, row, col, xdir, ydir, could_capture=0):
        # We shouldn't be able to leave the board
        if not self.__space_is_on_board(row, col):
            return False

        # If we're on a space associated with our turn and we have pieces
        # that could be captured return True. If there are no pieces that
        # could be captured that means we have consecutive bot pieces.
        if self.board[row, col] == self.turn:
            return could_capture != 0

        if self.__space_is_unoccupied(row, col):
            return False

        return self.__capture_will_occur(row + ydir, col + xdir, xdir, ydir, could_capture + 1)

    def __space_is_on_board(self, row, col):
        return 0 <= row < self.board_dim and 0 <= col < self.board_dim

    def __space_is_unoccupied(self, row, col):
        return self.board[row, col] == 0

    def __space_is_available(self, row, col):
        return self.__space_is_on_board(row, col) and self.__space_is_unoccupied(row, col)

    def _is_valid_move(self, row, col):
        if self.__space_is_available(row, col):
            # A valid move results in capture
            for xdir, ydir in itertools.product(range(-1, 2), range(-1, 2)):
                if xdir == ydir == 0:
                    continue
                if self.__capture_will_occur(row + ydir, col + xdir, xdir, ydir):
                    return True

    def get_player_pieces(self, player_id: int):
        # get a sum of all pieces on board that == player_id
        return np.sum(self.board == player_id)

    @lru_cache
    def get_valid_moves(self):
        valid_moves = []

        # If the middle four squares aren't taken the remaining ones are all
        # that is available
        if 0 in self.board[3:5, 3:5]:
            valid_moves.extend(
                (row, col) for row, col in itertools.product(range(3, 5), range(3, 5)) if self.board[row, col] == 0
            )
        else:
            for row in range(self.board_dim):
                valid_moves.extend((row, col) for col in range(self.board_dim) if self._is_valid_move(row, col))
        return valid_moves

    def simulate_move(self, move: Tuple[int, int]) -> "ReversiGameState":
        """
        This function should take a move and return a new state that is the
        result of making that move. It should not modify the current state.
        """
        new_board = np.copy(self.board)
        for xdir, ydir in itertools.product(range(-1, 2), range(-1, 2)):
            if xdir == ydir == 0:
                continue
            for i in itertools.count(1):
                row, col = move[0] + i * ydir, move[1] + i * xdir
                if not self.__space_is_on_board(row, col):
                    break
                if self.__space_is_unoccupied(row, col):
                    break
                if self.board[row, col] == self.turn:
                    for j in range(1, i):
                        new_board[move[0] + j * ydir, move[1] + j * xdir] = self.turn
                    break
        new_board[move] = self.turn
        new_state = ReversiGameState(new_board, 1 if self.turn == 2 else 2)
        if not new_state.get_valid_moves():
            new_state.turn = 1 if new_state.turn == 2 else 2
        return new_state

    def turns_remaining(self):
        # TODO - this is a bit of a hack, but it works for now
        return np.sum(self.board == 0)

    def __hash__(self):
        return hash(self.board.tostring())
