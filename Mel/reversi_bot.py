import math
import numpy as np
import random as rand
import reversi
import itertools


class ReversiBot:
    def __init__(self, move_num):
        self.move_num = move_num

    def make_move(self, state):
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

        move = rand.choice(valid_moves)  # Moves randomly...for now
        self.player = state.turn

        # if a corner is available, always take it
        if (0, 0) in valid_moves:
            return (0, 0)
        elif (0, 7) in valid_moves:
            return (0, 7)
        elif (7, 0) in valid_moves:
            return (7, 0)
        elif (7, 7) in valid_moves:
            return (7, 7)

        # print("New Round")
        depth = 6
        tree_root = self.generate_tree(state, depth)
        move = tree_root.winning_child.move_made

        return move

    def generate_tree(self, current_state, depth):
        maximizer = self.player
        minimizer = self.player % 2 + 1
        root = GameNode(current_state, (), [])
        root.generate_child_states(depth, maximizer, minimizer, 0, math.inf)
        return root


class GameNode:
    def __init__(self, state, move_made, children):
        self.move_made = move_made
        self.state = state
        self.children = children
        self.winning_child = None

    def generate_child_states(self, depth, maximizer, minimizer, alpha, beta):
        valid_moves = self.state.get_valid_moves()
        for move in valid_moves:
            child_state = self.generate_state_for_move(self.state, move)
            child = GameNode(child_state, move, [])
            # print(alpha, beta)
            # print("move")
            # print(move)

            # if beta <= alpha:
            #     break

            if depth > 1:
                if self.winning_child == None:
                    child.generate_child_states(depth - 1, maximizer, minimizer, max(alpha, 0), min(beta, math.inf))
                else:
                    child.generate_child_states(
                        depth - 1,
                        maximizer,
                        minimizer,
                        max(self.winning_child.get_score(maximizer), alpha),
                        min(self.winning_child.get_score(maximizer), beta),
                    )
            if self.state.turn == maximizer:
                if child.get_score(maximizer) > beta:
                    # print("max")
                    # print("pruned")
                    # print(child.move_made)
                    # print(beta)
                    # print(child.get_score(minimizer))
                    return
                if self.winning_child == None or child.get_score(maximizer) > self.winning_child.get_score(maximizer):
                    self.winning_child = child
            else:
                if child.get_score(maximizer) < alpha:
                    # print("min")
                    # print("pruned")
                    # print(child.move_made)
                    # print(alpha)
                    # print(child.get_score(maximizer))
                    return
                if self.winning_child == None or child.get_score(minimizer) > self.winning_child.get_score(minimizer):
                    self.winning_child = child
            self.children.append(child)

    def get_score(self, player):
        if self.winning_child != None:
            return self.winning_child.get_score(player)
        count = 0
        for row in range(len(self.state.board)):
            for col in range(len(self.state.board[row])):
                if self.state.board[row, col] == player:
                    count += 1

        CORNER_PIECE_VALUE = 5
        EDGE_PIECE_VALUE = 2

        own_corner_pieces = 0
        opponent_corner_pieces = 0
        for row, col in itertools.product(range(self.state.board_dim), range(self.state.board_dim)):
            if self.state.board[row, col] == player:
                if row in [0, self.state.board_dim - 1] and col in [0, self.state.board_dim - 1]:
                    own_corner_pieces += CORNER_PIECE_VALUE

        own_edge_pieces = 0
        opponent_edge_pieces = 0
        for row, col in itertools.product(range(self.state.board_dim), range(self.state.board_dim)):
            if self.state.board[row, col] == player:
                if row in [0, self.state.board_dim - 1] and col in [0, self.state.board_dim - 1]:
                    own_corner_pieces += EDGE_PIECE_VALUE
                elif row == 0 or row == self.state.board_dim - 1 or col == 0 or col == self.state.board_dim - 1:
                    own_edge_pieces += EDGE_PIECE_VALUE

        return count + own_corner_pieces + own_edge_pieces
        # return count

    def generate_state_for_move(self, current_state, move_to_make):
        # valid_moves = current_state.get_valid_moves()

        # if move_to_make not in valid_moves:
        #     return None

        new_board = current_state.board.copy()
        new_board[move_to_make[0], move_to_make[1]] = current_state.turn

        def space_is_on_board(row, col, dim):
            return 0 <= row < dim and 0 <= col < dim

        def space_is_unoccupied(self, row, col):
            return current_state.board[row, col] == 0

        def capture(self, board, row, col, xdir, ydir, could_capture=0):
            # We shouldn't be able to leave the board
            if not space_is_on_board(row, col, current_state.board_dim):
                return False, None

            # If we're on a space associated with our turn and we have pieces
            # that could be captured return True. If there are no pieces that
            # could be captured that means we have consecutive bot pieces.
            if current_state.board[row, col] == current_state.turn:
                return could_capture != 0, board

            if space_is_unoccupied(self, row, col):
                return False, None

            board[row, col] = current_state.turn
            return capture(self, board, row + ydir, col + xdir, xdir, ydir, could_capture + 1)

        for xdir in range(-1, 2):
            for ydir in range(-1, 2):
                if xdir == ydir == 0:
                    continue
                tmp_board = new_board.copy()

                success, tmp_board = capture(
                    self, tmp_board, move_to_make[0] + ydir, move_to_make[1] + xdir, xdir, ydir
                )
                if success:
                    new_board = tmp_board

        # next turn is the other player
        next_turn = current_state.turn % 2 + 1

        new_state = reversi.ReversiGameState(new_board, next_turn)
        # unless there are no moves that can be made
        if len(new_state.get_valid_moves()) == 0:
            current_state.turn

            new_state = reversi.ReversiGameState(new_board, next_turn)

        return new_state
