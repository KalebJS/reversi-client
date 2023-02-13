import numpy as np
import random as rand
import time
import reversi
import shelve
import random

memo = {}
moveNumber = 0

class ReversiBot:
    def __init__(self, move_num):
        global moveNumber
        print(move_num)
        self.move_num = move_num
        moveNumber = move_num

    def make_move(self, state):
        '''
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
        '''

        validMoves = state.get_valid_moves()
        startingMoves = [(3, 3), (3, 4), (4, 3), (4, 4)]
        playerNum = state.turn
        nextPlayer = 2 if playerNum == 1 else 1

        if (playerNum == 1) and (validMoves == startingMoves):
            move = (3,3)
            return move
        withinStartingMoves = False

        for move in validMoves:
            if (move == (3,3)) or (move == (3,4)) or (move == (4,3)) or (move == (4,4)):
                withinStartingMoves = True
        
        if withinStartingMoves == True:
            move = rand.choice(validMoves) # Moves randomly at the beginning
            return move
        

        '''
        UpdateBoard function updates the next turn in the game for the minimax algorithm to predict what will happen next
        '''
        def UpdateBoard(board, x, y, playerNum):
            # Define the 8 directions to check for captured pieces
            directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
            
            for dx, dy in directions:
                i = x + dx
                j = y + dy
                myCoin = 0
                theirCoin = 0
                if playerNum == 1:
                    myCoin = 1
                    theirCoin = 2
                else:
                    myCoin = 2
                    theirCoin = 1
                # Check if there is a coin of the opposite color in this direction
                if (0 <= i < 8) and (0 <= j < 8) and (board[i][j] == theirCoin):
                    # Keep moving in this direction until a coin of the same color is found
                    while (0 <= i < 8) and (0 <= j < 8) and (board[i][j] == theirCoin):
                        i += dx
                        j += dy
                    # If a coin of the same color was found, flip all the coins in between
                    if (0 <= i < 8) and (0 <= j < 8) and board[i][j] == myCoin:
                        i -= dx
                        j -= dy
                        while board[i][j] == theirCoin:
                            board[i][j] = myCoin
                            i -= dx
                            j -= dy
            return board
            
        def heuristic_evaluation(board, player):

            opponent = 3 - player
    
            me = reversi.ReversiGameState(board, player)
            enemy = reversi.ReversiGameState(board, opponent)
            my_moves = len(me.get_valid_moves())
            opp_moves = len(enemy.get_valid_moves())
            my_disks = np.sum(board == player)
            opp_disks = np.sum(board == opponent)

            disk_parity = 100 * (my_disks - opp_disks) / (my_disks + opp_disks + 1)
            corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
            my_corners = np.sum([board[x, y] == player for x, y in corners])
            opp_corners = np.sum([board[x, y] == opponent for x, y in corners])
            corner_occupancy = 25 * (my_corners - opp_corners) / 4
            edge_indices = [(0, i) for i in range(1, 7)] + [(7, i) for i in range(1, 7)] + [(i, 0) for i in range(1, 7)] + [(i, 7) for i in range(1, 7)]
            my_edges = np.sum([board[x, y] == player for x, y in edge_indices])
            opp_edges = np.sum([board[x, y] == opponent for x, y in edge_indices])
            edge_discs = -12.5 * (my_edges - opp_edges) / 32
            mobility = 100 * (my_moves - opp_moves) / (my_moves + opp_moves + 1)

            unflippable_disks = 0
            for i in range(8):
                for j in range(8):
                    if board[i, j] == player:
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            if 0 <= i + dx <= 7 and 0 <= j + dy <= 7:
                                if board[i + dx, j + dy] == opponent:
                                    unflippable_disks += 1
                                    break
            unflippable_disks = -unflippable_disks
            corner_closeness = 0
            adj_corners = [(0, 1), (1, 0), (1, 1), (0, 6), (1, 7), (1, 6), (6, 0), (7, 1), (6, 1), (7, 6), (6, 7), (6, 6)]
            for i, j in adj_corners:
                if board[i, j] == player:
                    corner_closeness -= 2
            center_indices = [(i, j) for i in range(2, 6) for j in range(2, 6)]
            my_center = np.sum([board[x, y] == player for x, y in center_indices])
            opp_center = np.sum([board[x, y] == opponent for x, y in center_indices])
            center = (5 - (my_center - opp_center)) / 5
            return 1 * disk_parity + 10 * corner_occupancy + 6 *edge_discs +  4 * mobility +  6 * unflippable_disks +  -8 * corner_closeness + 3 * center


        def save_memo():
            with shelve.open("memo_dict") as db:
                db["memo"] = memo

        def alpha_beta_reversi(board, moves, player, depth, alpha=-np.inf, beta=np.inf):
            # print(type(board))
            if depth == 0 or len(moves) == 0:
                return (None, heuristic_evaluation(board, player))

            # Check if the state has been computed before
            state = (board.tobytes(), player, depth)
            if state in memo:
                return memo[state]

            # Initialize the best move and value
            best_move = None
            move_values = []
            if player == 1:
                # For each move, simulate the move, get the value from the next iteration
                # and update the move values array
                for move in moves:
                    x, y = move
                    nextGameBoard = np.copy(board)
                    nextGameBoard[x][y] = player
                    nextGameBoard = UpdateBoard(nextGameBoard, x, y, player)
                    nextState = reversi.ReversiGameState(nextGameBoard, nextPlayer)
                    nextPlayerValidMoves = nextState.get_valid_moves()

                    
                    # board[x, y] = player
                    _, value = alpha_beta_reversi(nextGameBoard, nextPlayerValidMoves, 2, depth - 1, alpha, beta)
                    # board[x, y] = 0
                    move_values.append((move, value))
                    alpha = max(alpha, value)
                    # If alpha is greater than or equal to beta, the branch can be pruned
                    if alpha >= beta:
                        break
                # Choose the move with the best value
                best_move, best_value = max(move_values, key=lambda x: x[1])
                # Store the result in the memoization dictionary and return
                memo[state] = (best_move, best_value)
                return (best_move, best_value)
            else:
                # For each move, simulate the move, get the value from the next iteration
                # and update the move values array
                for move in moves:
                    x, y = move
                    # board[x, y] = player
                    nextGameBoard = np.copy(board)
                    nextGameBoard[x][y] = player
                    nextGameBoard = UpdateBoard(nextGameBoard, x, y, player)
                    nextState = reversi.ReversiGameState(nextGameBoard, nextPlayer)
                    nextPlayerValidMoves = nextState.get_valid_moves()

                    _, value = alpha_beta_reversi(board, moves, 1, depth - 1, alpha, beta)
                    
                    # board[x, y] = 0
                    move_values.append((move, value))
                    beta = min(beta, value)
                    # If alpha is greater than or equal to beta, the branch can be pruned
                    if alpha >= beta:
                        break
                # Choose the move with the best value
                best_move, best_value = min(move_values, key=lambda x: x[1])
                # Store the result in the memoization dictionary and return
                memo[state] = (best_move, best_value)
                return (best_move, best_value)


        print("------------------------------------------------------------------------")
        maxDepth = 8
        # finalCoinValues = []
        
        time0 = time.time()
        
        if moveNumber == 2:
            randomVariable = random.randint(0,100)
            if randomVariable < 35:
                print("RANDOM")
                return rand.choice(validMoves)

        finalCoinValues = alpha_beta_reversi(state.board, validMoves, playerNum, maxDepth)
        time1 = time.time()
        print("Final Move = " + str(finalCoinValues))
        print("Time for turn: " + str(time1 - time0))

        save_memo()

        return move
