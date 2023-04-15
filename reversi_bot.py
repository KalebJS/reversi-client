import pickle
from functools import cache, wraps
from pathlib import Path
from typing import List, Tuple, Union

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

    def count_discs(self, state: ReversiGameState):
        # Get total discs for the current player
        return np.sum(state.board == self.own_id)

    def disc_parity(self, state: ReversiGameState):
        """
        Get the difference in discs from the perspective of the current
        player (positive if more than enemy, negative if less than enemy)
        """
        return np.sum(state.board == self.own_id) - np.sum(state.board == (3 - self.own_id))

    def corners_captured(self, state: ReversiGameState):
        """
        Get the number of corners captured by the current player
        """
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        return sum(
            state.board[corner[0]][corner[1]] == self.own_id for corner in corners
        )

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

    def full_edges_captured(self, state: ReversiGameState):
        # Get the number of complete edges captured by the current player
        player_board = np.array(state.board == self.own_id, dtype=np.int8)

        top_row = player_board[0]
        bottom_row = player_board[7]
        left_col = player_board[:, 0]
        right_col = player_board[:, 7]

        full_edges = 0

        # Check top and bottom rows
        if np.all(top_row) or np.all(top_row[::-1]):
            full_edges += 1
        if np.all(bottom_row) or np.all(bottom_row[::-1]):
            full_edges += 1

        # Check left and right columns
        if np.all(left_col) or np.all(left_col[::-1]):
            full_edges += 1
        if np.all(right_col) or np.all(right_col[::-1]):
            full_edges += 1

        return full_edges

    def permanent_unflippable_discs(self, state: ReversiGameState):
        board = state.board
        unflippableTable = np.empty((8, 8), dtype="U")
        unflippableTable.fill("")

        # Define the edges of the board
        edges = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 0),
            (1, 7),
            (2, 0),
            (2, 7),
            (3, 0),
            (3, 7),
            (4, 0),
            (4, 7),
            (5, 0),
            (5, 7),
            (6, 0),
            (6, 7),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 7),
        ]

        # Define the four corners of the board
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]

        # If a corner is owned, mark it as unflipabble
        for corner in corners:
            if board[corner[0]][corner[1]] == self.own_id:
                unflippableTable[corner[0]][corner[1]] = "u"

        top_row = board[0]
        bottom_row = board[7]
        left_col = board[:, 0]
        right_col = board[:, 7]
        occupied_edges = []

        # Check top and bottom rows
        if np.array_equal(np.unique(top_row), np.array([1, 2])):
            occupied_edges.append("top")
        if np.array_equal(np.unique(bottom_row), np.array([1, 2])):
            occupied_edges.append("bottom")

        # Check left and right columns
        if np.array_equal(np.unique(left_col), np.array([1, 2])):
            occupied_edges.append("left")
        if np.array_equal(np.unique(right_col), np.array([1, 2])):
            occupied_edges.append("right")

        if occupied_edges:
            for side in occupied_edges:
                if side == "top":
                    for i in range(0, 7):
                        if board[0][i] == self.own_id:
                            unflippableTable[0][i] = "u"
                elif side == "bottom":
                    for i in range(0, 7):
                        if board[7][i] == self.own_id:
                            unflippableTable[7][i] = "u"
                elif side == "left":
                    for i in range(0, 7):
                        if board[i][0] == self.own_id:
                            unflippableTable[i][0] = "u"
                elif side == "right":
                    for i in range(0, 7):
                        if board[i][7] == self.own_id:
                            unflippableTable[i][7] = "u"

        iterate = 0
        # Change this to update while there was a change in the previous loop
        while iterate < 8:
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

            directions = [0, 1, 2, 3, 4, 5, 6, 7]  # up, up-right, right, down-right, down, down-left, left, up-left
            coordinates = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

            for x in range(1, 7):
                for y in range(1, 7):
                    surroundingUnflippables = 0
                    unflippableList = []
                    if board[x][y] == 0:
                        continue
                    for i in range(len(directions)):
                        if unflippableTable[x + coordinates[i][0]][y + coordinates[i][1]] == "u":
                            surroundingUnflippables += 1
                            unflippableList.append(i)
                    if len(unflippableList) >= 4:
                        numbersInARow = 0
                        orderedList1 = []
                        orderedList2 = []
                        currentIndex = 0
                        for i in range(len(unflippableList)):
                            if unflippableList[i] == 0:
                                orderedList1.append(unflippableList[i])
                                continue
                            elif unflippableList[i] - unflippableList[i - 1] == self.own_id:
                                orderedList1.append(unflippableList[i])
                            elif unflippableList[i] - unflippableList[i - 1] != self.own_id:
                                orderedList2.append(unflippableList[i])
                                currentIndex = i
                                break
                        for i in range(currentIndex + 1, len(unflippableList)):
                            if unflippableList[i] - unflippableList[i - 1] == self.own_id:
                                orderedList2.append(unflippableList[i])
                        finalOrderedList = orderedList2 + orderedList1
                        for i in range(len(finalOrderedList)):
                            if i == 0:
                                continue
                            if (finalOrderedList[i] == finalOrderedList[i - 1] + 1) or (
                                finalOrderedList[i] == finalOrderedList[i - 1] - 7
                            ):
                                if i == 1:
                                    numbersInARow += 1
                                numbersInARow += 1
                                continue

                        if numbersInARow >= 4:
                            unflippableTable[x][y] = "u"
            iterate += 1
        unflippable_coords = []
        for x in range(unflippableTable.shape[0]):
            for y in range(unflippableTable.shape[1]):
                if unflippableTable[x][y] == "u":
                    unflippable_coords.append((x, y))
        return unflippable_coords

    def corner_danger_zones(self, state: ReversiGameState):
        danger_coords = [(0, 1), (1, 0), (1, 1), (0, 6), (1, 6), (1, 7), (6, 0), (6, 1), (7, 1), (6, 6), (6, 7), (7, 6)]
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        danger_count = 0

        unflippable_coords = self.permanent_unflippable_discs(state)

        for corner in corners:
            row, col = corner
            if state.board[row][col] != 0:  # Skip if corner is already occupied
                continue
            if corner == (0, 0):
                for i in range(3):
                    if state.board[danger_coords[i][0]][danger_coords[i][1]] == self.own_id and (
                        danger_coords[i] not in unflippable_coords
                    ):
                        danger_count += 1
                    elif state.board[danger_coords[i][0]][danger_coords[i][1]] == self.own_id and (
                        danger_coords[i] in unflippable_coords
                    ):
                        continue
            if corner == (0, 7):
                for i in range(3, 6):
                    if state.board[danger_coords[i][0]][danger_coords[i][1]] == self.own_id and (
                        danger_coords[i] not in unflippable_coords
                    ):
                        danger_count += 1
                    elif state.board[danger_coords[i][0]][danger_coords[i][1]] == self.own_id and (
                        danger_coords[i] in unflippable_coords
                    ):
                        continue
            if corner == (7, 0):
                for i in range(6, 9):
                    if state.board[danger_coords[i][0]][danger_coords[i][1]] == self.own_id and (
                        danger_coords[i] not in unflippable_coords
                    ):
                        danger_count += 1
                    elif state.board[danger_coords[i][0]][danger_coords[i][1]] == self.own_id and (
                        danger_coords[i] in unflippable_coords
                    ):
                        continue
            if corner == (7, 7):
                for i in range(9, 12):
                    if state.board[danger_coords[i][0]][danger_coords[i][1]] == self.own_id and (
                        danger_coords[i] not in unflippable_coords
                    ):
                        danger_count += 1
                    elif state.board[danger_coords[i][0]][danger_coords[i][1]] == self.own_id and (
                        danger_coords[i] in unflippable_coords
                    ):
                        continue

        return danger_count

    def valid_moves(self, state: ReversiGameState):
        hypothetical_state = ReversiGameState(state.board, self.own_id)
        num_moves = len(hypothetical_state.get_valid_moves())
        return num_moves

    @cache
    def heuristic(self, state: ReversiGameState):
        # Main heuristic function that calls the other various heuristic functions and applies multipliers to them
        # We may change multiplier values based on the number of turns left
        self.evaluations += 1
        moves_left = state.turns_remaining()

        discs = self.count_discs(state)
        parity = self.disc_parity(state)
        corners = self.corners_captured(state)
        edges = self.edges_captured(state)
        sides = self.full_edges_captured(state)
        unflippable = len(self.permanent_unflippable_discs(state))
        corner_danger_zone = self.corner_danger_zones(state)
        mobility = self.valid_moves(state)

        # max and min values for each heuristic function
        max_discs = 64
        min_discs = 0

        max_parity = 64
        min_parity = -64

        max_corners = 4
        min_corners = 0

        max_edges = 24
        min_edges = 0

        max_sides = 4
        min_sides = 0

        max_unflippable = np.sum(state.board != 0)
        min_unflippable = 0

        max_corner_danger_zone = 0
        min_corner_danger_zone = -12

        max_mobility = 15
        min_mobility = 0

        # normalize each heuristic value
        # normalized_discs = (discs - min_discs) / (max_discs - min_discs)
        # if the below is ever 1 this is a winning state
        normalized_discs = discs / np.sum(state.board != 0)
        # normalized_parity = (parity - min_parity) / (max_parity - min_parity) * 2 - 1  # scale to range from -1 to 1
        # if the below is ever 1 this is a winning state
        normalized_parity = parity / np.sum(state.board != 0)
        normalized_corners = (corners - min_corners) / (max_corners - min_corners)
        normalized_edges = (edges - min_edges) / (max_edges - min_edges)
        normalized_sides = (sides - min_sides) / (max_sides - min_sides)
        normalized_unflippable = (unflippable - min_unflippable) / (max_unflippable - min_unflippable)
        normalized_corner_danger_zone = (corner_danger_zone - max_corner_danger_zone) / (
            min_corner_danger_zone - max_corner_danger_zone
        )  # scale to range from -1 to 0
        normalized_mobility = (mobility - min_mobility) / (max_mobility - min_mobility)

        # Set initial weights
        discs_weight = 0
        parity_weight = 0
        corners_weight = 0
        edges_weight = 0
        sides_weight = 0
        unflippable_weight = 0
        corner_danger_zone_weight = 0
        mobility_weight = 0

        if moves_left <= 64:
            discs_weight = 5
            parity_weight = 5
            corners_weight = 20
            edges_weight = 20
            sides_weight = 10
            unflippable_weight = 15
            corner_danger_zone_weight = 25
            # mobility_weight = 20
        if moves_left <= 50:
            discs_weight = 5
            parity_weight = 5
            corners_weight = 20
            edges_weight = 20
            sides_weight = 10
            unflippable_weight = 15
            corner_danger_zone_weight = 25
            # mobility_weight = 15
        if moves_left <= 40:
            discs_weight = 5
            parity_weight = 7
            corners_weight = 20
            edges_weight = 15
            sides_weight = 10
            unflippable_weight = 20
            corner_danger_zone_weight = 23
            # mobility_weight = 5
        if moves_left <= 30:
            discs_weight = 10
            parity_weight = 15
            corners_weight = 20
            edges_weight = 10
            sides_weight = 15
            unflippable_weight = 20
            corner_danger_zone_weight = 15
            # mobility_weight = 5
        if moves_left <= 20:
            discs_weight = 10
            parity_weight = 35
            corners_weight = 5
            edges_weight = 5
            sides_weight = 15
            unflippable_weight = 20
            corner_danger_zone_weight = 10
            # mobility_weight = 0
        if moves_left <= 15:
            discs_weight = 0
            parity_weight = 47
            corners_weight = 0
            edges_weight = 0
            sides_weight = 10
            unflippable_weight = 38
            corner_danger_zone_weight = 5
            # mobility_weight = 0
        if moves_left <= 10:
            discs_weight = 0
            parity_weight = 100
            corners_weight = 0
            edges_weight = 0
            sides_weight = 0
            unflippable_weight = 0
            corner_danger_zone_weight = 0
            # mobility_weight = 0

        heuristicScore = (
            (normalized_discs * discs_weight)
            + (normalized_parity * parity_weight)
            + (normalized_corners * corners_weight)
            + (normalized_edges * edges_weight)
            + (normalized_sides * sides_weight)
            + (normalized_unflippable * unflippable_weight)
            + (normalized_corner_danger_zone * corner_danger_zone_weight)
        )  # + (normalized_mobility * mobility_weight)

        return heuristicScore

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
        # more moves as it gets deeper into the game
        depth = 1 + self.move_num // 10
        best, res = float("-inf"), None
        for move in valid_moves:
            score = self.minimax(state, move, depth, float("-inf"), float("inf"))
            if score > best:
                best = score
                res = move
        self.move_num += 1

        print(self.evaluations)
        return res


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
