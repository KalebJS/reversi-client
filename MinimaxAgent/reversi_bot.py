import numpy as np
import random as rand
import time
import reversi

class ReversiBot:
    def __init__(self, move_num):
        print(move_num)
        self.move_num = move_num

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
        
        def Minimax2(currentGameBoard, possibleMoves, player, depth, maxDepth, isEnemyTurn):
            # Assign player values according to the player variable
            currentPlayer = 1 if player == 1 else 2
            nextPlayer = 2 if player == 1 else 1

            # Find the number of empty spaces. If there are less than maxDepth spaces left, adjust maxDepth
            numTurnsLeft = 0
            for i in range(len(currentGameBoard[0])):
                for j in range(len(currentGameBoard[1])):
                    if currentGameBoard[i][j] == 0:
                        numTurnsLeft += 1
            if (numTurnsLeft > 1) and (numTurnsLeft < maxDepth * 2):
                
                # print("NUMTURNSLEFT IS TOO SMALL")
                # print(numTurnsLeft)
                # print(maxDepth)
                maxDepth = int(np.floor(numTurnsLeft / 2))
                if depth > maxDepth:
                    depth = maxDepth
            elif numTurnsLeft == 1:
                return possibleMoves
            # print(maxDepth)

            # Loop through the possible game board trees 
            branchTotals = []
            nextBranchValues = []
            currentBranch = []
            for move in possibleMoves:
                x, y = move
                nextGameBoard = np.copy(currentGameBoard)
                nextGameBoard[x][y] = currentPlayer
                nextGameBoard = UpdateBoard(nextGameBoard, x, y, currentPlayer)
                nextState = reversi.ReversiGameState(nextGameBoard, nextPlayer)
                nextPlayerValidMoves = nextState.get_valid_moves()
                if len(nextPlayerValidMoves) == 0:
                    # print("Next valid moves = 0")
                    # print(possibleMoves)
                    continue

                if ((depth == maxDepth) and (isEnemyTurn)) or (numTurnsLeft == 1):
                    currentPlayerPoints = 0
                    nextPlayerPoints = 0
                    for i in range(len(nextGameBoard[0])):
                        for j in range(len(nextGameBoard[1])):
                            if nextGameBoard[i][j] == currentPlayer:
                                currentPlayerPoints += 1
                            elif nextGameBoard[i][j] == nextPlayer:
                                nextPlayerPoints += 1
                    newBranchTotal = (x, y, nextPlayerPoints)
                    branchTotals.append(newBranchTotal)
                else:
                    if isEnemyTurn:
                        nextBranchValues.append(Minimax2(nextGameBoard, nextPlayerValidMoves, nextPlayer, depth + 1, maxDepth, False))
                    else:
                        nextBranchValues.append(Minimax2(nextGameBoard, nextPlayerValidMoves, nextPlayer, depth, maxDepth, True))
                    currentBranch.append((x, y))


            if (depth == maxDepth) and (isEnemyTurn):
                minValue = 0
                minInt = 0
                try:
                    # print("branch totals = " + str(branchTotals))
                    for i in range(len(branchTotals)):
                        if i == 0:
                            minValue = branchTotals[i][2]
                            minInt = i
                        else:
                            if branchTotals[i][2] < minValue:
                                minValue = branchTotals[i][2]
                                minInt = i
                except:
                        return rand.choice(possibleMoves)
                try:
                    bestChoice = branchTotals[minInt]
                except:
                    return rand.choice(possibleMoves)
                return bestChoice
            else:
                valueIndex = -1
                if isEnemyTurn:
                    minValue = 0
                    minInt = 0
                    try:
                        for i in range(len(nextBranchValues)):
                            if i == 0:
                                minValue = nextBranchValues[i][2]
                                minInt = i
                            else:
                                if nextBranchValues[i][2] < minValue:
                                    minValue = nextBranchValues[i][2]
                                    minInt = i
                        valueIndex = minInt
                    except:
                        return rand.choice(possibleMoves)
                else:
                    maxValue = 0
                    maxInt = 0
                    try:
                        for i in range(len(nextBranchValues)):
                            if i == 0:
                                maxValue = nextBranchValues[i][2]
                                maxInt = i
                            else:
                                if nextBranchValues[i][2] > maxValue:
                                    maxValue = nextBranchValues[i][2]
                                    maxInt = i
                        valueIndex = maxInt
                    except:
                        return rand.choice(possibleMoves)
                    
                # print("CurrentValue = " + str(currentBranch))
                # print("nextBranchValues = " + str(nextBranchValues))
                # print("possible moves = " + str(possibleMoves))
                try:
                    finalValue = (currentBranch[valueIndex][0], currentBranch[valueIndex][1], nextBranchValues[valueIndex][2])
                except:
                    return rand.choice(possibleMoves)
                
                return finalValue

    

        print("------------------------------------------------------------------------")
        maxDepth = 2
        finalCoinValues = []

        time0 = time.time()
        finalCoinValues.append(Minimax2(state.board, validMoves, playerNum, 1, maxDepth, False))
        time1 = time.time()

        print("Final Move = " + str(finalCoinValues))
        print("Time for turn: " + str(time1 - time0))
        if finalCoinValues[0] == 0:
            move = rand.choice(validMoves)
        else:
            try:
                move = (finalCoinValues[0][0], finalCoinValues[0][1])
            except:
                move = (finalCoinValues[0][0][0], finalCoinValues[0][0][1])
        return move
