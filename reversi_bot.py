import numpy as np
from functools import cache
from typing import Tuple, Union

from state import ReversiGameState


class ReversiBot:
    def __init__(self, move_num):
        self.move_num = move_num
        self.own_id = move_num
        self.opponent_id = 3 - move_num
        self.evaluations = 0

    def discCount(self, state: ReversiGameState):
        # Get total discs for the current player
        return np.sum(state.board == state.turn)

    def discParity(self, state: ReversiGameState):
        # Get the difference in discs from the perspective of the current player (positive if more than enemy, negative if less than enemy)
        return np.sum(state.board == state.turn) - np.sum(state.board == (3 - state.turn))

    def availableMoves(self, state: ReversiGameState):
        # How many moves can the current player make from this position (how many can the enemy make as well?)
        ...

    def cornersCaptured(self, state: ReversiGameState):
        # Get the number of corners captured by the current player
        totalCorners = 0
        corners = [(0,0),(0,7),(7,0),(7,7)]
        for corner in corners:
            if state.board[corner[0]][corner[1]] == state.turn:
                totalCorners += 1
        return totalCorners

    def edgesCaptured(self, state: ReversiGameState):
        # Get the number of edges captured by the current player
        player_mask = state.board == state.turn
        row_mask = (1 << 8) - 1

        top_row = player_mask[0]
        bottom_row = player_mask[7]
        left_col = player_mask[:, 0]
        right_col = player_mask[:, 7]

        total_edges = 0

        # Check top and bottom rows
        if (top_row[1:-1] & ~row_mask[1:-1]).sum() == 0:
            total_edges += 1
        if (bottom_row[1:-1] & ~row_mask[1:-1]).sum() == 0:
            total_edges += 1

        # Check left and right columns
        left_col_mask = (1 << 56) | (1 << 48) | (1 << 40) | (1 << 32) | (1 << 24) | (1 << 16) | (1 << 8) | 1
        right_col_mask = left_col_mask << 7

        if (left_col[1:-1] & ~left_col_mask).sum() == 0:
            total_edges += 1
        if (right_col[1:-1] & ~right_col_mask).sum() == 0:
            total_edges += 1

        return total_edges

    def fullEdgesCaptured(self, state: ReversiGameState):
        # Get the number of complete edges captured by the current player
        player_mask = state.board == state.turn
        row_mask = (1 << 8) - 1

        top_row = player_mask[0]
        bottom_row = player_mask[7]
        left_col = player_mask[:, 0]
        right_col = player_mask[:, 7]

        full_edges = 0

        # Check top and bottom rows
        if (top_row & row_mask) == row_mask:
            full_edges += 1
        if (bottom_row & row_mask) == row_mask:
            full_edges += 1

        # Check left and right columns
        left_col_mask = (1 << 56) | (1 << 48) | (1 << 40) | (1 << 32) | (1 << 24) | (1 << 16) | (1 << 8) | 1
        right_col_mask = left_col_mask << 7

        if (left_col & left_col_mask) == left_col_mask:
            full_edges += 1
        if (right_col & right_col_mask) == right_col_mask:
            full_edges += 1

        return full_edges
        
    def permanantUnflippableDiscs(self, state: ReversiGameState):
        # Get the number of unflippable discs
        player_mask = state.board == state.turn

        # Define masks for edge and corner squares
        corner_mask = np.zeros((8, 8), dtype=bool)
        corner_mask[0, 0] = corner_mask[0, 7] = corner_mask[7, 0] = corner_mask[7, 7] = True

        edge_mask = np.zeros((8, 8), dtype=bool)
        edge_mask[0, :] = edge_mask[7, :] = edge_mask[:, 0] = edge_mask[:, 7] = True
        edge_mask = edge_mask & (~corner_mask)

        # Define masks for squares that can be permanently unflippable
        unflippable_mask = np.zeros((8, 8), dtype=bool)

        # Check corners
        if corner_mask[0, 0]:
            if state.board[1, 0] == (3 - state.turn) and state.board[0, 1] == (3 - state.turn):
                unflippable_mask[0, 0] = True

        if corner_mask[0, 7]:
            if state.board[0, 6] == (3 - state.turn) and state.board[1, 7] == (3 - state.turn):
                unflippable_mask[0, 7] = True

        if corner_mask[7, 0]:
            if state.board[6, 0] == (3 - state.turn) and state.board[7, 1] == (3 - state.turn):
                unflippable_mask[7, 0] = True

        if corner_mask[7, 7]:
            if state.board[6, 7] == (3 - state.turn) and state.board[7, 6] == (3 - state.turn):
                unflippable_mask[7, 7] = True

        # Check edges
        for i in range(1, 7):
            if edge_mask[0, i]:
                if (state.board[0, i-1] == (3 - state.turn) and state.board[0, i+1] == (3 - state.turn) and
                        state.board[1, i] == (3 - state.turn)):
                    unflippable_mask[0, i] = True

            if edge_mask[7, i]:
                if (state.board[7, i-1] == (3 - state.turn) and state.board[7, i+1] == (3 - state.turn) and
                        state.board[6, i] == (3 - state.turn)):
                    unflippable_mask[7, i] = True

            if edge_mask[i, 0]:
                if (state.board[i-1, 0] == (3 - state.turn) and state.board[i+1, 0] == (3 - state.turn) and
                    state.board[i, 1] == (3 - state.turn)):
                    unflippable_mask[i, 0] = True
                    if edge_mask[i, 7]:
                        if (state.board[i-1, 7] == (3 - state.turn) and state.board[i+1, 7] == (3 - state.turn) and
                                state.board[i, 6] == (3 - state.turn)):
                            unflippable_mask[i, 7] = True

        # Count the number of unflippable discs
        unflippable_discs = np.sum(unflippable_mask & player_mask)

        return unflippable_discs

    def tempUnflippableDiscs(self, state: ReversiGameState):
        # Define a mask for the current player's discs
        player_mask = state.board == state.turn
        
        # Define a mask for the opponent's discs
        opp_mask = state.board == (3 - state.turn)
        
        # Define a mask for empty squares
        empty_mask = (state.board == 0)
        
        # Get the set of legal moves for the current player
        legal_moves = state.get_legal_actions()
        
        # Define a mask for squares that can be temporarily unflippable
        unflippable_mask = np.zeros((8, 8), dtype=bool)
        
        # Iterate over each legal move
        for move in legal_moves:
            # Get the coordinates of the move
            row, col = move
            
            # Define a mask for the squares surrounding the move
            surround_mask = np.zeros((8, 8), dtype=bool)
            surround_mask[max(0, row-1):min(7, row+1)+1, max(0, col-1):min(7, col+1)+1] = True
            surround_mask[row, col] = False
            
            # Check if any of the surrounding squares contain an opponent's disc
            if np.any(opp_mask & surround_mask):
                # Get the direction of the opponent's disc
                opp_direction = np.zeros((8, 8), dtype=bool)
                opp_direction[row, col] = True
                
                # Check in each direction from the opponent's disc for an empty square
                for drow, dcol in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    cur_row, cur_col = row + drow, col + dcol
                    cur_direction = np.zeros((8, 8), dtype=bool)
                    cur_direction[cur_row, cur_col] = True
                    
                    # Keep going in the same direction while we're on the board and we're on empty squares
                    while (cur_row >= 0 and cur_row <= 7 and cur_col >= 0 and cur_col <= 7 and 
                        empty_mask[cur_row, cur_col] and np.any(cur_direction & surround_mask)):
                        cur_row += drow
                        cur_col += dcol
                        cur_direction[cur_row, cur_col] = True
                    
                    # If we've reached a player's disc, mark the squares between the opponent's disc and the player's disc as unflippable
                    if (cur_row >= 0 and cur_row <= 7 and cur_col >= 0 and cur_col <= 7 and 
                        player_mask[cur_row, cur_col] and np.any(cur_direction & surround_mask)):
                        unflippable_mask[opp_direction] = True
            
        # Count the number of temporarily unflippable discs
        unflippable_discs = np.sum(unflippable_mask & player_mask)
        
        return unflippable_discs

    def cornerDangerZones(self, state: ReversiGameState):
        # Spaces that allow the enemy to capture corners (within 1 diagonal space of an unclaimed corner)
        ...

    def edgeDangerZones(self, state: ReversiGameState):
        # Spaces that allow the enemy to capture edges (within 1 space of an unclaimed edge)
        ...

    
    @cache
    def heuristic(self, state: ReversiGameState):
        # Main heuristic function that calls the other various heuristic functions and applies multipliers to them
        # We may change multiplier values based on the number of turns left
        '''
      ___________________   ________   ________         ________   ________    _______ /\___________    ___________________  __________   ________ ______________________    ___________________       _________    ___ ___     _____    _______     ________ ___________    ___________ ___ ___ ___________      ___ ___ ___________ ____ ___ __________ .___   ____________________.___ _________        _________ ____ ___ _________    ___ ___      ___________ ___ ___     _____ ___________    .___ ___________    .___   _________       _____   .____    __      __   _____  _____.___.  _________       _____      _____   ____  ___.___    _____   .___ __________.___  _______     ________     ________    ____ ___ __________     __________ .____        _____  _____.___._____________________     ____   ____ _____   .____      ____ ___ ___________  _________       _______   ________ ___________    __________ ________ ___________ ___ ___      __________ .____        _____  _____.___._____________________     ____   ____ _____   .____      ____ ___ ___________  _________ 
      \__    ___/\_____  \  \______ \  \_____  \  /\    \______ \  \_____  \   \      \\(\__    ___/    \_   _____/\_____  \ \______   \ /  _____/ \_   _____/\__    ___/    \__    ___/\_____  \      \_   ___ \  /   |   \   /  _  \   \      \   /  _____/ \_   _____/    \__    ___//   |   \\_   _____/     /   |   \\_   _____/|    |   \\______   \|   | /   _____/\__    ___/|   |\_   ___ \      /   _____/|    |   \\_   ___ \  /   |   \     \__    ___//   |   \   /  _  \\__    ___/    |   |\__    ___/    |   | /   _____/      /  _  \  |    |  /  \    /  \ /  _  \ \__  |   | /   _____/      /     \    /  _  \  \   \/  /|   |  /     \  |   |\____    /|   | \      \   /  _____/     \_____  \  |    |   \\______   \    \______   \|    |      /  _  \ \__  |   |\_   _____/\______   \    \   \ /   //  _  \  |    |    |    |   \\_   _____/ /   _____/       \      \  \_____  \\__    ___/    \______   \\_____  \\__    ___//   |   \     \______   \|    |      /  _  \ \__  |   |\_   _____/\______   \    \   \ /   //  _  \  |    |    |    |   \\_   _____/ /   _____/ 
        |    |    /   |   \  |    |  \  /   |   \ \/     |    |  \  /   |   \  /   |   \   |    |        |    __)   /   |   \ |       _//   \  ___  |    __)_   |    |         |    |    /   |   \     /    \  \/ /    ~    \ /  /_\  \  /   |   \ /   \  ___  |    __)_       |    |  /    ~    \|    __)_     /    ~    \|    __)_ |    |   / |       _/|   | \_____  \   |    |   |   |/    \  \/      \_____  \ |    |   //    \  \/ /    ~    \      |    |  /    ~    \ /  /_\  \ |    |       |   |  |    |       |   | \_____  \      /  /_\  \ |    |  \   \/\/   //  /_\  \ /   |   | \_____  \      /  \ /  \  /  /_\  \  \     / |   | /  \ /  \ |   |  /     / |   | /   |   \ /   \  ___      /   |   \ |    |   / |       _/     |     ___/|    |     /  /_\  \ /   |   | |    __)_  |       _/     \   Y   //  /_\  \ |    |    |    |   / |    __)_  \_____  \        /   |   \  /   |   \ |    |        |    |  _/ /   |   \ |    |  /    ~    \     |     ___/|    |     /  /_\  \ /   |   | |    __)_  |       _/     \   Y   //  /_\  \ |    |    |    |   / |    __)_  \_____  \  
        |    |   /    |    \ |    `   \/    |    \/\     |    `   \/    |    \/    |    \  |    |        |     \   /    |    \|    |   \\    \_\  \ |        \  |    |         |    |   /    |    \    \     \____\    Y    //    |    \/    |    \\    \_\  \ |        \      |    |  \    Y    /|        \    \    Y    /|        \|    |  /  |    |   \|   | /        \  |    |   |   |\     \____     /        \|    |  / \     \____\    Y    /      |    |  \    Y    //    |    \|    |       |   |  |    |       |   | /        \    /    |    \|    |___\        //    |    \\____   | /        \    /    Y    \/    |    \ /     \ |   |/    Y    \|   | /     /_ |   |/    |    \\    \_\  \    /    |    \|    |  /  |    |   \     |    |    |    |___ /    |    \\____   | |        \ |    |   \      \     //    |    \|    |___ |    |  /  |        \ /        \      /    |    \/    |    \|    |        |    |   \/    |    \|    |  \    Y    /     |    |    |    |___ /    |    \\____   | |        \ |    |   \      \     //    |    \|    |___ |    |  /  |        \ /        \ 
        |____|   \_______  //_______  /\_______  /\/    /_______  /\_______  /\____|__  /  |____|        \___  /   \_______  /|____|_  / \______  //_______  /  |____|         |____|   \_______  /     \______  / \___|_  / \____|__  /\____|__  / \______  //_______  /      |____|   \___|_  //_______  /     \___|_  //_______  /|______/   |____|_  /|___|/_______  /  |____|   |___| \______  /    /_______  /|______/   \______  / \___|_  /       |____|   \___|_  / \____|__  /|____|       |___|  |____|       |___|/_______  /    \____|__  /|_______ \\__/\  / \____|__  // ______|/_______  /    \____|__  /\____|__  //___/\  \|___|\____|__  /|___|/_______ \|___|\____|__  / \______  /    \_______  /|______/   |____|_  /     |____|    |_______ \\____|__  // ______|/_______  / |____|_  /       \___/ \____|__  /|_______ \|______/  /_______  //_______  //\    \____|__  /\_______  /|____|        |______  /\_______  /|____|   \___|_  /      |____|    |_______ \\____|__  // ______|/_______  / |____|_  /       \___/ \____|__  /|_______ \|______/  /_______  //_______  / 
                        \/         \/         \/               \/         \/         \/                     \/            \/        \/         \/         \/                                   \/             \/        \/          \/         \/         \/         \/                      \/         \/            \/         \/                   \/              \/                         \/             \/                   \/        \/                       \/          \/                                               \/             \/         \/     \/          \/ \/               \/             \/         \/       \_/             \/              \/             \/         \/             \/                   \/                        \/        \/ \/               \/         \/                      \/         \/                  \/         \/ )/            \/         \/                      \/         \/                \/                         \/        \/ \/               \/         \/                      \/         \/                  \/         \/  
        '''
        self.evaluations += 1
        # moves_left = state.turns_remaining()

        # discs = self.discCount(state)
        # parity = self.discParity(state)
        # mobility = self.availableMoves(state)
        # corners = self.cornersCaptured(state)
        # edges = self.edgesCaptured(state)
        # sides = self.fullEdgesCaptured(state)
        # unflippable = self.permanentUnflippableDiscs(state)
        # tempUnflippable = self.tempUnflippableDiscs(state)
        # riskySpace = self.riskySpace(state)
        
        return state.get_player_pieces(self.own_id)

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
        DEPTH = 5
        best, res = float("-inf"), None
        for move in valid_moves:
            score = self.minimax(state, move, DEPTH, float("-inf"), float("inf"))
            if score > best:
                best = score
                res = move
        self.move_num += 1

        print(self.evaluations)
        return res 
