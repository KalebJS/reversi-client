import itertools
import pickle
from functools import cache, wraps
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from state import ReversiGameState


def dump_board(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            with open("board.pkl", "wb") as f:
                pickle.dump(args[0].board, f)
            raise e

    return wrapper


class ReversiBot:
    def __init__(self, move_num):
        self.move_num = move_num
        self.own_id = move_num
        self.opponent_id = 3 - move_num
        self.evaluations = 0
        self.stability_matrix: np.ndarray = np.array(
            [
                [120, -10, 10, 5, 5, 10, -10, 120],
                [-10, -40, 1, 1, 1, 1, -40, -10],
                [10, 1, 5, 2, 2, 5, 1, 10],
                [5, 1, 2, 10, 10, 2, 1, 5],
                [5, 1, 2, 10, 10, 2, 1, 5],
                [10, 1, 5, 2, 2, 5, 1, 10],
                [-10, -40, 1, 1, 1, 1, -40, -10],
                [120, -10, 10, 5, 5, 10, -10, 120]
            ]
        )
        # parity
        # corners
        # edges
        # value_weight
        # unflippable
        self.weights = [
            [10, 10, 35, 30, 15],   # 64
            [10, 15, 30, 25, 20],   # 50
            [12, 20, 25, 18, 25],   # 40
            [20, 13, 25, 7, 35],    # 30
            [65, 0, 10, 3, 22],    # 20
            [85, 0, 0, 0, 15],     # 15
            [100, 0, 0, 0, 0],      # 10
        ]

    def disc_parity(self, state: ReversiGameState):
        """
        Get the difference in discs from the perspective of the bot
        """
        return np.sum(state.board == self.own_id) - np.sum(state.board == self.opponent_id)

    def corners_captured(self, state: ReversiGameState):
        """
        Get the number of corners captured by the current player
        """
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        return sum(state.board[corner[0]][corner[1]] == self.own_id for corner in corners)

    def edges_captured(self, state: ReversiGameState):
        """
        Get the number of edges captured by the current player
        """
        player_mask: np.ndarray = state.board == self.own_id

        total_edges = 0

        # Check top and bottom rows
        for j in range(1, 7):
            if player_mask[0, j]:
                total_edges += 1
            if player_mask[7, j]:
                total_edges += 1

        # Check left and right columns
        for i in range(1, 7):
            if player_mask[i, 0]:
                total_edges += 1
            if player_mask[i, 7]:
                total_edges += 1

        return total_edges

    def permanent_unflippable_discs(self, state: ReversiGameState):
        board = state.board
        unflippableTable = np.empty((8, 8), dtype="U")
        unflippableTable.fill("")

        # Define the edges of the board
        edges = [(0, 0),(0, 1),(0, 2),(0, 3),(0, 4),(0, 5),(0, 6),(0, 7),(1, 0),(1, 7),(2, 0),(2, 7),(3, 0),(3, 7),
            (4, 0),(4, 7),(5, 0),(5, 7),(6, 0),(6, 7),(7, 0),(7, 1),(7, 2),(7, 3),(7, 4),(7, 5),(7, 6),(7, 7)]

        # Define the four corners of the board
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]

        # If a corner is owned, mark it as unflipabble
        for corner in corners:
            if board[corner[0]][corner[1]] == self.own_id:
                unflippableTable[corner[0]][corner[1]] = "u"

        counter = 0
        tempState = np.copy(unflippableTable)
        while True:
            # Calculate which edges are unflippable
            for edge in edges:
                row, col = edge
                if unflippableTable[row][col] == "u":
                    continue
                if board[row][col] == self.own_id:
                    for i in [-1, 0, 1]:
                        for j in [-1, 0, 1]:
                            if (
                                (i == 0 and j == 0)
                                or (i == 1 and j == 1)
                                or (i == -1 and j == -1)
                                or (i == 1 and j == -1)
                                or (i == -1 and j == 1)
                            ):
                                continue
                            adj_row = row + i
                            adj_col = col + j
                            if 0 <= adj_row < 8 and 0 <= adj_col < 8 and unflippableTable[adj_row][adj_col] in ["u"]:
                                unflippableTable[row][col] = "u"
            if np.array_equal(tempState, unflippableTable) or counter >= 7:
                break
            else:
                counter += 1
                tempState = np.copy(unflippableTable)
        return [
            (x, y)
            for x, y in itertools.product(range(unflippableTable.shape[0]), range(unflippableTable.shape[1]))
            if unflippableTable[x][y] == "u"
        ]

    def stability(self, state: ReversiGameState):
        """
        calculates value based on stability matrix
        :param state:
        :return:
        """
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]

        stability_matrix = self.stability_matrix.copy()

        for corner in corners:
            if state.board[corner[0]][corner[1]] == 1:
                if corner == (0, 0):
                    stability_matrix[0][1] = 200
                    stability_matrix[1][1] = 80
                    stability_matrix[1][0] = 200
                    stability_matrix[2][0] = 80
                    stability_matrix[0][2] = 80
                    stability_matrix[1][2] = 40
                    stability_matrix[2][1] = 40
                    stability_matrix[3][0] = 60
                    stability_matrix[0][3] = 60
                elif corner == (0, 7):
                    stability_matrix[0][6] = 200
                    stability_matrix[1][6] = 80
                    stability_matrix[1][7] = 200
                    stability_matrix[0][5] = 80
                    stability_matrix[2][7] = 80
                    stability_matrix[1][5] = 40
                    stability_matrix[2][6] = 40
                    stability_matrix[3][7] = 60
                    stability_matrix[0][4] = 60
                elif corner == (7, 0):
                    stability_matrix[6][0] = 200
                    stability_matrix[6][1] = 80
                    stability_matrix[7][1] = 200
                    stability_matrix[5][0] = 80
                    stability_matrix[7][2] = 80
                    stability_matrix[5][1] = 40
                    stability_matrix[6][2] = 40
                    stability_matrix[4][0] = 60
                    stability_matrix[7][3] = 60
                elif corner == (7, 7):
                    stability_matrix[6][7] = 200
                    stability_matrix[6][6] = 80
                    stability_matrix[7][6] = 200
                    stability_matrix[5][7] = 80
                    stability_matrix[7][5] = 80
                    stability_matrix[5][6] = 40
                    stability_matrix[6][5] = 40
                    stability_matrix[4][7] = 60
                    stability_matrix[7][4] = 60

        return sum(
            stability_matrix[i][j] for i, j in itertools.product(range(8), range(8)) if state.board[i][j] == self.own_id
        )

    def get_weights(self, moves_left):
        weights = self.weights

        if moves_left <= 2:
            return (
                100,
                0,
                0,
                0,
                0,
            )
        if moves_left <= 10:
            return [*weights[6]]
        if moves_left <= 15:
            return [*weights[5]]
        if moves_left <= 20:
            return [*weights[4]]
        if moves_left <= 30:
            return [*weights[3]]
        if moves_left <= 40:
            return [*weights[2]]
        if moves_left <= 50:
            return [*weights[1]]
        if moves_left <= 64:
            return [*weights[0]]

    @cache
    def heuristic(self, state: ReversiGameState):
        # Main heuristic function that calls the other various heuristic functions and applies multipliers to them
        # We may change multiplier values based on the number of turns left
        self.evaluations += 1
        moves_left = state.turns_remaining()

        parity = self.disc_parity(state)
        corners = self.corners_captured(state)
        edges = self.edges_captured(state)
        unflippable = len(self.permanent_unflippable_discs(state))
        value = self.stability(state)

        # normalize each heuristic value
        normalized_parity = parity / np.sum(state.board != 0)

        corner_count = np.sum([state.board[c] != 0 for c in [(0, 0), (0, 7), (7, 0), (7, 7)]])
        normalized_corners = corners / corner_count if corner_count != 0 else 0

        edge_count = np.sum(
            [state.board[r][c] != 0 for r in range(1, 7) for c in range(1, 7) if r in (1, 6) or c in (1, 6)]
        )
        normalized_edges = edges / edge_count if edge_count != 0 else 0

        # normalized_sides = (sides - min_sides) / (max_sides - min_sides)
        normalized_unflippable = (
            (unflippable / (edge_count + corner_count)) if edge_count != 0 or corner_count != 0 else 0
        )

        normalized_stability = value / np.sum(state.board != 0)

        # Set initial weights
        (parity_weight, corners_weight, edges_weight, unflippable_weight, value_weight) = self.get_weights(moves_left)

        return (
            (normalized_parity * parity_weight)
            + (normalized_corners * corners_weight)
            + (normalized_edges * edges_weight)
            + (normalized_unflippable * unflippable_weight)
            + (normalized_stability * value_weight)
        )

    @dump_board
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
        :return: Score for the current state
        """

        move_state = state.simulate_move(move)

        is_maximizing = move_state.turn == self.own_id
        if depth == 0 or not move_state.turns_remaining():
            return self.heuristic(move_state)

        if is_maximizing:
            best_score = float("-inf")
            f = max
        else:
            best_score = float("inf")
            f = min

        moves = move_state.get_valid_moves()
        l = [(self.heuristic(move_state.simulate_move(moves)), move) for moves in moves]
        l.sort(key=lambda x: x[0], reverse=True)
        for _, new_move in l[:6]:
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
        assert valid_moves, "No valid moves"
        depth = 8
        if self.move_num > 30:
            depth = 11
        best, res = -1 * float("inf"), None
        for move in valid_moves:
            score = self.minimax(state, move, depth, float("-inf"), float("inf"))
            if score > best:
                best = score
                res = move
        self.move_num += 1

        print(self.evaluations)
        return res if res else [0, 0]


if __name__ == "__main__":
    bot = ReversiBot(1)
    # create 8x8 board numpy array
    path = Path.cwd() / "board.pkl"
    if not path.exists():
        board = np.zeros((8, 8), dtype=int)
    else:
        with open("board.pkl", "rb") as f:
            board = pickle.load(f)
    state = ReversiGameState(board, 1)
    bot.make_move(state)
