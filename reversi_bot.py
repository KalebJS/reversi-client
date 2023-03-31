import itertools
import numpy as np
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

    
    def discCount(state: ReversiGameState):
        # Get total discs for the current player
        return np.sum(state.board == state.turn)
    def discParity(state: ReversiGameState):
        # Get the difference in discs from the perspective of the current player (positive if more than enemy, negative if less than enemy)
        return np.sum(state.board == state.turn) - np.sum(state.board == (3 - state.turn))
    def availableMoves(state: ReversiGameState):
        # How many moves can the current player make from this position (how many can the enemy make as well?)
        ...
    def cornersCaptured(state: ReversiGameState):
        # Get the number of corners captured by the current player
        totalCorners = 0
        corners = [(0,0),(0,7),(7,0),(7,7)]
        for corner in corners:
            if state.board[corner[0]][corner[1]] == state.turn:
                totalCorners += 1
        return totalCorners
    def edgesCaptured(state: ReversiGameState):
        # Get the number of edges captured by the current player
        totalEdges = 0
        for point in state.board:
            if (point[0] == 0 or point[0] == 7) and (point[1] < 7 and point[1] > 0) and (point == state.turn):
                totalEdges += 1
            elif (point[0] != 0 and point[0] != 7) and (point[1] == 0 or point[1] == 7) and ([point == state.turn]):
                totalEdges += 1
        return totalEdges
    def fullEdgesCaptured(state: ReversiGameState):
        # Get the number of complete edges captured by the current player
        totalSides = 0
        ...
    def permanantUnflippableDiscs(state: ReversiGameState):
        # Get the number of unflippable discs
        ...
    def tempUnflippableDiscs(state: ReversiGameState):
        # Get the number of unflippable discs
        ...
    def dangerZones(state: ReversiGameState):
        # Spaces that allow the enemy to capture crucial locations (within 1 of corners and edges), maybe make multiple of these
        # equations if the weighted values are different between edges and corners
        ...
    def boardControl(state: ReversiGameState):
        # Depending on the state of the game, who has more board control, this may just be the same as the availableMoves() function
        ...
    def discDensity(state: ReversiGameState):
        # Finds clusters of discs, specifically if they are unflippable
        ...
    
    # Main heuristic function that calls the other various heuristic functions and applies multipliers to them
    def heuristic(self, state: ReversiGameState):
        # We may change multiplier values based on the number of turns left
        '''
        TODO: DON'T FORGET TO CHANGE THE HEURISTIC SUCH THAT IT IS ALWAYS MAXIMIZING OUR PLAYER VALUES, NOT BOTH PLAYER VALUES
        TODO: DON'T FORGET TO CHANGE THE HEURISTIC SUCH THAT IT IS ALWAYS MAXIMIZING OUR PLAYER VALUES, NOT BOTH PLAYER VALUES
        TODO: DON'T FORGET TO CHANGE THE HEURISTIC SUCH THAT IT IS ALWAYS MAXIMIZING OUR PLAYER VALUES, NOT BOTH PLAYER VALUES
        TODO: DON'T FORGET TO CHANGE THE HEURISTIC SUCH THAT IT IS ALWAYS MAXIMIZING OUR PLAYER VALUES, NOT BOTH PLAYER VALUES
        TODO: DON'T FORGET TO CHANGE THE HEURISTIC SUCH THAT IT IS ALWAYS MAXIMIZING OUR PLAYER VALUES, NOT BOTH PLAYER VALUES
        TODO: DON'T FORGET TO CHANGE THE HEURISTIC SUCH THAT IT IS ALWAYS MAXIMIZING OUR PLAYER VALUES, NOT BOTH PLAYER VALUES
        '''
        moves_left = state.turns_remaining()

        discs = discCount(state)
        parity = discParity(state)
        mobility = availableMoves(state)
        corners = cornersCaptured(state)
        edges = edgesCaptured(state)
        sides
        permanentUnflippable
        tempUnflippable
        riskySpace
        control
        density



        return 0


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
