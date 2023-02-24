import itertools
import random as rand
from functools import lru_cache
from typing import Tuple, Union

from state import ReversiGameState


class ReversiBot:
    def __init__(self, move_num):
        self.move_num = move_num
        self.own_id = None
        self.opponent_id = None
        self.evaluations = 0

    @lru_cache
    def evaluate_board(self, state: ReversiGameState) -> int:
        ...

    def heuristic(self, state: ReversiGameState):
        moves_left = state.turns_remaining()

        """
        Possible Heuristics:
            Disc Count
            Disc parity (difference in disc quantity between players)
            Available moves
            Corners captured
            Edges captured
            Unflippable Discs/Stability matrix (calculate the stability of each disc in a range from 0 to 1)
            Danger zones/Frontiers (Close to edge/close to corner if those edges/corners are not yet taken, or on the outside of a cluster of discs)
            Control center of the board at the beginning, edges at the end?
            Connectivity/Full sides captured (from corner to corner or other long chains of discs)
            Density? (clusters of discs, especially if they are all unflippable)
            Flippable discs? (which of my tokens can get flipped)
            Threat blocking? (when the enemy has the option to flip many of our tokens, we neutralize the threat if possible to avoid losing discs)
            Sacrifice potential (sacrificing a disc or several discs may be worth it to get an advantages position on the board)    
            Initiative/momentum (calculates which player has the momentum in the game, that forces )
        """

    def minimax(
        self,
        state: ReversiGameState,
        move: Tuple[int, int],
        depth: int,
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
        is_maximizing = move_state.turn == self.own_id
        if depth == 0 or not move_state.turns_remaining():
            return self.evaluate_board(move_state)

        if is_maximizing:
            best_score = float("-inf")
            f = max
        else:
            best_score = float("inf")
            f = min

        for new_move in move_state.get_valid_moves():
            score = self.minimax(move_state, new_move, depth - 1, alpha, beta)
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
        valid_moves = state.get_valid_moves()
        for move in valid_moves:
            self.minimax(state, move, 3, state.turn == self.own_id, float("-inf"), float("inf"))
        self.move_num += 1

        return rand.choice(valid_moves)
