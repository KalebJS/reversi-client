from pathlib import Path
import numpy as np
from functools import cache, wraps
from typing import Tuple, Union

from state import ReversiGameState
import pickle


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

    def discCount(self, state: ReversiGameState):
        # Get total discs for the current player
        return np.sum(state.board == self.own_id)

    def discParity(self, state: ReversiGameState):
        # Get the difference in discs from the perspective of the current player (positive if more than enemy, negative if less than enemy)
        return np.sum(state.board == self.own_id) - np.sum(state.board == (3 - self.own_id))

    def cornersCaptured(self, state: ReversiGameState):
        # Get the number of corners captured by the current player
        totalCorners = 0
        corners = [(0,0),(0,7),(7,0),(7,7)]
        for corner in corners:
            if state.board[corner[0]][corner[1]] == self.own_id:
                totalCorners += 1
        return totalCorners

    def edgesCaptured(self, state: ReversiGameState):
        # Get the number of edges captured by the current player
        player_mask = state.board == self.own_id

        total_edges = 0

        # Check top and bottom rows
        if np.all(player_mask[0, :]) or np.all(player_mask[7, :]):
            total_edges += 1

        # Check left and right columns
        if np.all(player_mask[:, 0]) or np.all(player_mask[:, 7]):
            total_edges += 1

        return total_edges

    def fullEdgesCaptured(self, state: ReversiGameState):
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
        
    def permanentUnflippableDiscs(self, state: ReversiGameState):
        # Get the number of unflippable discs
        player_mask = state.board == self.own_id

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
            if state.board[1, 0] == (3 - self.own_id) and state.board[0, 1] == (3 - self.own_id):
                unflippable_mask[0, 0] = True

        if corner_mask[0, 7]:
            if state.board[0, 6] == (3 - self.own_id) and state.board[1, 7] == (3 - self.own_id):
                unflippable_mask[0, 7] = True

        if corner_mask[7, 0]:
            if state.board[6, 0] == (3 - self.own_id) and state.board[7, 1] == (3 - self.own_id):
                unflippable_mask[7, 0] = True

        if corner_mask[7, 7]:
            if state.board[6, 7] == (3 - self.own_id) and state.board[7, 6] == (3 - self.own_id):
                unflippable_mask[7, 7] = True

        # Check edges
        for i in range(1, 7):
            if edge_mask[0, i]:
                if (state.board[0, i-1] == (3 - self.own_id) and state.board[0, i+1] == (3 - self.own_id) and
                        state.board[1, i] == (3 - self.own_id)):
                    unflippable_mask[0, i] = True

            if edge_mask[7, i]:
                if (state.board[7, i-1] == (3 - self.own_id) and state.board[7, i+1] == (3 - self.own_id) and
                        state.board[6, i] == (3 - self.own_id)):
                    unflippable_mask[7, i] = True

            if edge_mask[i, 0]:
                if (state.board[i-1, 0] == (3 - self.own_id) and state.board[i+1, 0] == (3 - self.own_id) and
                    state.board[i, 1] == (3 - self.own_id)):
                    unflippable_mask[i, 0] = True
                    if edge_mask[i, 7]:
                        if (state.board[i-1, 7] == (3 - self.own_id) and state.board[i+1, 7] == (3 - self.own_id) and
                                state.board[i, 6] == (3 - self.own_id)):
                            unflippable_mask[i, 7] = True

        # Count the number of unflippable discs
        unflippable_discs = np.sum(unflippable_mask & player_mask)

        return unflippable_discs

    def tempUnflippableDiscs(self, state: ReversiGameState):
        # Define a mask for the current player's discs
        player_mask = state.board == self.own_id
        
        # Define a mask for the opponent's discs
        opp_mask = state.board == (3 - self.own_id)
        
        # Define a mask for empty squares
        empty_mask = (state.board == 0)
        
        # Get the set of legal moves for the current player
        legal_moves = state.get_valid_moves()
        
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
                    if cur_row < 0 or cur_row > 7 or cur_col < 0 or cur_col > 7:
                        continue
                    cur_direction = np.zeros((8, 8), dtype=bool)
                    cur_direction[cur_row, cur_col] = True
                    
                    # Keep going in the same direction while we're on the board and we're on empty squares
                    while (cur_row >= 0 and cur_row <= 7 and cur_col >= 0 and cur_col <= 7 and 
                        empty_mask[cur_row, cur_col] and np.any(cur_direction & surround_mask)):
                        cur_direction[cur_row, cur_col] = True
                        cur_row += drow
                        cur_col += dcol
                    
                    # If we've reached a player's disc, mark the squares between the opponent's disc and the player's disc as unflippable
                    if (cur_row >= 0 and cur_row <= 7 and cur_col >= 0 and cur_col <= 7 and 
                        player_mask[cur_row, cur_col] and np.any(cur_direction & surround_mask)):
                        unflippable_mask[opp_direction] = True
            
        # Count the number of temporarily unflippable discs
        unflippable_discs = np.sum(unflippable_mask & player_mask)
        
        return unflippable_discs

    def cornerDangerZones(self, state: ReversiGameState):
        dangerCoords = [(1,1), (1,6), (6,1), (6,6), (0,1), (1,0), (0,6), (1,7), (6,0), (7,1), (6,7), (7,6)]
        corners = [(0,0), (0,7), (7,0), (7,7)]
        dangerCount = 0
        for corner in corners:
            row, col = corner
            if state.board[row][col] != 0: # Skip if corner is already occupied
                continue
            for coord in dangerCoords:
                row_offset, col_offset = coord
                danger_row = row + row_offset
                danger_col = col + col_offset
                if (0 <= danger_row < 8) and (0 <= danger_col < 8) and (state.board[danger_row][danger_col] == self.own_id):
                    # Check if the disc in the danger zone is flippable
                    flip_row = danger_row + row_offset
                    flip_col = danger_col + col_offset
                    while (0 <= flip_row < 8) and (0 <= flip_col < 8) and (state.board[flip_row][flip_col] == -self.own_id):
                        flip_row += row_offset
                        flip_col += col_offset
                    if (0 <= flip_row < 8) and (0 <= flip_col < 8) and (state.board[flip_row][flip_col] == self.own_id):
                        dangerCount += 1
                elif (0 <= danger_row < 8) and (0 <= danger_col < 8) and (state.board[danger_row][danger_col] == -self.own_id):
                    # Check if the disc in the danger zone is unflippable
                    flip_row = danger_row + row_offset
                    flip_col = danger_col + col_offset
                    while (0 <= flip_row < 8) and (0 <= flip_col < 8) and (state.board[flip_row][flip_col] == -self.own_id):
                        flip_row += row_offset
                        flip_col += col_offset
                    if (0 <= flip_row < 8) and (0 <= flip_col < 8) and (state.board[flip_row][flip_col] == self.own_id):
                        pass
                    else:
                        dangerCount += 1
        return dangerCount

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
        moves_left = state.turns_remaining()

        discs = self.discCount(state)
        parity = self.discParity(state)
        corners = self.cornersCaptured(state)
        edges = self.edgesCaptured(state)
        sides = self.fullEdgesCaptured(state)
        unflippable = self.permanentUnflippableDiscs(state)
        tempUnflippable = self.tempUnflippableDiscs(state)
        cornerDangerZone = self.cornerDangerZones(state)
        # mobility = self.availableMoves(state)
      
        # Set initial weights
        discs_weight = 0
        parity_weight = 0
        corners_weight = 0
        edges_weight = 0
        sides_weight = 0
        unflippable_weight = 0
        tempUnflippable_weight = 0
        cornerDangerZone_weight = 0
        # mobility_weight = 0

        if moves_left <= 64:
            discs_weight = 1
            parity_weight = 1
            corners_weight = 80
            edges_weight = 50
            sides_weight = 50
            unflippable_weight = 8
            tempUnflippable = 5
            cornerDangerZone_weight = -15
        if moves_left <= 50:
            ...
        if moves_left <= 40:
            unflippable_weight = 50
        if moves_left <= 30:
            ...
        if moves_left <= 20:
            discs_weight = 45
            tempUnflippable_weight = 8
            corners_weight = 100
        if moves_left <= 15:
            ...
        if moves_left <= 10:
            discs_weight = 100
            parity_weight = 30
            corners_weight = 10
            edges_weight = 10
            sides_weight = 30
            unflippable_weight = 15
            tempUnflippable = 10
            cornerDangerZone_weight = -5
        if moves_left <= 5:
            # discs_weight = .9
            # parity_weight = .9
            ...
        if moves_left == 1:
            discs_weight = 100
            parity_weight = 0
            corners_weight = 0
            edges_weight = 0
            sides_weight = 0
            unflippable_weight = 0
            tempUnflippable = 0
            cornerDangerZone_weight = 0
        
        heuristicScore = (discs * discs_weight) + (parity * parity_weight) + (corners * corners_weight) + \
                        (edges * edges_weight) + (sides * sides_weight) + (unflippable * unflippable_weight) + \
                        (tempUnflippable * tempUnflippable_weight) + (cornerDangerZone * cornerDangerZone_weight)
        
        return heuristicScore
        
        # return state.get_player_pieces(self.own_id)

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
        DEPTH = 1
        best, res = float("-inf"), None
        for move in valid_moves:
            score = self.minimax(state, move, DEPTH, float("-inf"), float("inf"))
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
