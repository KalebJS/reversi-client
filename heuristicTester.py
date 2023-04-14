import numpy as np

from state import ReversiGameState

own_id = 1


def discCount(state: ReversiGameState):
    # Get total discs for the current player
    return np.sum(state.board == own_id)


def discParity(state: ReversiGameState):
    # Get the difference in discs from the perspective of the current player (positive if more than enemy, negative if less than enemy)
    return np.sum(state.board == own_id) - np.sum(state.board == (3 - own_id))


def cornersCaptured(state: ReversiGameState):
    # Get the number of corners captured by the current player
    totalCorners = 0
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    for corner in corners:
        if state.board[corner[0]][corner[1]] == own_id:
            totalCorners += 1
    return totalCorners


def edgesCaptured(state: ReversiGameState):
    # Get the number of edges captured by the current player
    player_mask = state.board == own_id

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


def fullEdgesCaptured(state: ReversiGameState):
    # Get the number of complete edges captured by the current player
    player_board = np.array(state.board == own_id, dtype=np.int8)

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


def permanentUnflippableDiscs(state: ReversiGameState):
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
        if board[corner[0]][corner[1]] == own_id:
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
            print(side)
            if side == "top":
                for i in range(0, 7):
                    if board[0][i] == own_id:
                        unflippableTable[0][i] = "u"
            elif side == "bottom":
                for i in range(0, 7):
                    if board[7][i] == own_id:
                        unflippableTable[7][i] = "u"
            elif side == "left":
                for i in range(0, 7):
                    if board[i][0] == own_id:
                        unflippableTable[i][0] = "u"
            elif side == "right":
                for i in range(0, 7):
                    if board[i][7] == own_id:
                        unflippableTable[i][7] = "u"

    iterate = 0
    # Change this to update while there was a change in the previous loop
    while iterate < 15:
        # Calculate which edges are unflippable
        for edge in edges:
            row, col = edge
            if unflippableTable[row][col] == "u":
                continue
            if board[row][col] == own_id:
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
                        elif unflippableList[i] - unflippableList[i - 1] == own_id:
                            orderedList1.append(unflippableList[i])
                        elif unflippableList[i] - unflippableList[i - 1] != own_id:
                            orderedList2.append(unflippableList[i])
                            currentIndex = i
                            break
                    for i in range(currentIndex + 1, len(unflippableList)):
                        if unflippableList[i] - unflippableList[i - 1] == own_id:
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
    unflippabeCoords = []
    for x in range(unflippableTable.shape[0]):
        for y in range(unflippableTable.shape[1]):
            if unflippableTable[x][y] == "u":
                unflippabeCoords.append((x, y))
    return unflippabeCoords


def cornerDangerZones(state: ReversiGameState):
    dangerCoords = [(0, 1), (1, 0), (1, 1), (0, 6), (1, 6), (1, 7), (6, 0), (6, 1), (7, 1), (6, 6), (6, 7), (7, 6)]
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    dangerCount = 0

    unflippableCoords = permanentUnflippableDiscs(state)

    for corner in corners:
        row, col = corner
        if state.board[row][col] != 0:  # Skip if corner is already occupied
            continue
        if corner == (0, 0):
            for i in range(3):
                if state.board[dangerCoords[i][0]][dangerCoords[i][1]] == own_id and (
                    dangerCoords[i] not in unflippableCoords
                ):
                    dangerCount += 1
                elif state.board[dangerCoords[i][0]][dangerCoords[i][1]] == own_id and (
                    dangerCoords[i] in unflippableCoords
                ):
                    continue
        if corner == (0, 7):
            for i in range(3, 6):
                if state.board[dangerCoords[i][0]][dangerCoords[i][1]] == own_id and (
                    dangerCoords[i] not in unflippableCoords
                ):
                    dangerCount += 1
                elif state.board[dangerCoords[i][0]][dangerCoords[i][1]] == own_id and (
                    dangerCoords[i] in unflippableCoords
                ):
                    continue
        if corner == (7, 0):
            for i in range(6, 9):
                if state.board[dangerCoords[i][0]][dangerCoords[i][1]] == own_id and (
                    dangerCoords[i] not in unflippableCoords
                ):
                    dangerCount += 1
                elif state.board[dangerCoords[i][0]][dangerCoords[i][1]] == own_id and (
                    dangerCoords[i] in unflippableCoords
                ):
                    continue
        if corner == (7, 7):
            for i in range(9, 12):
                if state.board[dangerCoords[i][0]][dangerCoords[i][1]] == own_id and (
                    dangerCoords[i] not in unflippableCoords
                ):
                    dangerCount += 1
                elif state.board[dangerCoords[i][0]][dangerCoords[i][1]] == own_id and (
                    dangerCoords[i] in unflippableCoords
                ):
                    continue

    return dangerCount


def validMoves(state: ReversiGameState):
    hypotheticalState = ReversiGameState(state.board, 2)
    numMoves = hypotheticalState.get_valid_moves()
    return numMoves, state.get_valid_moves()


myMap = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 2, 2, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
myTurn = 1

state = ReversiGameState(myMap, myTurn)

print("Disc Count = " + str(discCount(state)))
print("Disc Parity = " + str(discParity(state)))
print("Corners Captured =  " + str(cornersCaptured(state)))
print("Edges Captured =  " + str(edgesCaptured(state)))
print("Sides Captured =  " + str(fullEdgesCaptured(state)))
print("Unflippable Discs =  " + str(len(permanentUnflippableDiscs(state))))
print("Danger Zones = " + str(cornerDangerZones(state)))
print("NumMoves: " + str(validMoves(state)))
