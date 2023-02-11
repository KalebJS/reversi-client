import itertools
import random as rand
from functools import lru_cache
from typing import Tuple, Union

from state import ReversiGameState

MAX_DEPTH = 8
CORNER_PIECE_VALUE = 5
EDGE_PIECE_VALUE = 2


class ReversiBot:
    def __init__(self, move_num):
        self.move_num = move_num
        self.own_id = None
        self.opponent_id = None
        self.evaluations = 0

    @lru_cache()
    def evaluate_board(self, state: ReversiGameState):
        """
        This function should take a board and return a score for the board.
        The score should be a number that is higher if the board is better
        for the player whose turn it is and lower if the board is better
        for the other player.
        """
        self.evaluations += 1
        own_pieces = state.get_player_pieces(self.own_id)
        opponent_pieces = state.get_player_pieces(self.opponent_id)

        # Corner pieces are worth 10 points
        own_corner_pieces = 0
        opponent_corner_pieces = 0
        for row, col in itertools.product(range(state.board_dim), range(state.board_dim)):
            if state.board[row, col] == self.own_id:
                if row in [0, state.board_dim - 1] and col in [0, state.board_dim - 1]:
                    own_corner_pieces += CORNER_PIECE_VALUE
            elif state.board[row, col] == self.opponent_id:
                if row in [0, state.board_dim - 1] and col in [0, state.board_dim - 1]:
                    opponent_corner_pieces += CORNER_PIECE_VALUE

        # Edge pieces are worth 5 points
        own_edge_pieces = 0
        opponent_edge_pieces = 0
        for row, col in itertools.product(range(state.board_dim), range(state.board_dim)):
            if state.board[row, col] == self.own_id:
                if row in [0, state.board_dim - 1] and col in [0, state.board_dim - 1]:
                    own_corner_pieces += EDGE_PIECE_VALUE
                elif row == 0 or row == state.board_dim - 1 or col == 0 or col == state.board_dim - 1:
                    own_edge_pieces += EDGE_PIECE_VALUE
            elif state.board[row, col] == self.opponent_id:
                if row in [0, state.board_dim - 1] and col in [0, state.board_dim - 1]:
                    opponent_corner_pieces += EDGE_PIECE_VALUE
                elif row == 0 or row == state.board_dim - 1 or col == 0 or col == state.board_dim - 1:
                    opponent_edge_pieces += EDGE_PIECE_VALUE

        # TODO: add a heuristic for the number of pieces that can be flipped
        # TODO: add a heuristic for the number of edge pieces that can be flipped
        # TODO: adjust weights of the heuristics depending on the number of moves remaining

        return (
            own_pieces
            - opponent_pieces
            + own_corner_pieces
            - opponent_corner_pieces
            + own_edge_pieces
            - opponent_edge_pieces
        )

    def minimax(
        self,
        state: ReversiGameState,
        move: Tuple[int, int],
        depth: int,
        is_maximizing: bool,
        alpha: Union[float, int],
        beta: Union[float, int],
    ) -> int:
        """
        This function should implement the minimax algorithm. It should
        return the best possible score for the current state. It should
        also update the self.best_move variable to be the move that
        leads to the best possible score.

        :param beta:
        :param alpha:
        :param move:
        :param state: The current game state
        :param depth: How many moves ahead to look
        :param is_maximizing: boolean that is true if the current player is the maximizing player
        :return: Score for the current state
        """

        move_state = state.simulate_move(move)
        # NOTE: the second condition may be problematic since the bot won't be able to see past
        # situations where it has no valid moves, which could be winning games.
        if depth == 0 or not move_state.get_valid_moves():
            return self.evaluate_board(move_state)

        if is_maximizing:
            best_score = float("-inf")
            f = max
        else:
            best_score = float("inf")
            f = min

        for new_move in move_state.get_valid_moves():
            score = self.minimax(move_state, new_move, depth - 1, not is_maximizing, alpha, beta)
            # current_score = self.evaluate_board(move_state)
            # score = current_score * 0.55 + score * 0.45
            best_score = f(score, best_score)

            if is_maximizing:
                alpha = max(alpha, best_score)
            else:
                beta = min(beta, best_score)

            if beta <= alpha:
                break

        return best_score

    def make_move(self, state: ReversiGameState):
        """
        This is the only function that needs to be implemented for the lab!
        The bot should take a game state and return a move.

        The parameter "state" is of type ReversiGameState and has two useful
        member variables. The first is "board", which is an 8x8 numpy array
        of 0s, 1s, and 2s. If a spot has a 0 that means it is unoccupied. If
        there is a 1 that means the spot has one of player 1's stones. If
        there is a 2 on the spot that means that spot has one of player 2's
        stones. The other useful member variable is "turn", which is 1 if it's
        player 1's turn and 2 if it's player 2's turn.

        ReversiGameState objects have a nice method called get_valid_moves.
        When you invoke it on a ReversiGameState object a list of valid
        moves for that state is returned in the form of a list of tuples.

        Move should be a tuple (row, col) of the move you want the bot to make.
        """
        self.own_id = state.turn
        self.opponent_id = 1 if state.turn == 2 else 2

        depth = min(MAX_DEPTH, self.move_num // 5 + 1)

        moves = [
            (self.minimax(state, move, depth, is_maximizing=True, alpha=float("-inf"), beta=float("inf")), move)
            for move in state.get_valid_moves()
        ]
        _, move = max(moves, key=lambda x: x[0])

        print(self.evaluations)

        self.move_num += 1
        return move

    def random_move(self, state: ReversiGameState):
        valid_moves = state.get_valid_moves()
        self.move_num += 1
        return rand.choice(valid_moves)
