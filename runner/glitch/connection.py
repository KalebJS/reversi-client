import socket

import numpy as np

from state import ReversiGameState


class ReversiServerConnection:
    def __init__(self, host, bot_move_num):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (host, 3333 + bot_move_num)
        self.sock.connect(server_address)
        self.sock.recv(1024)

    def get_game_state(self):
        server_msg = self.sock.recv(1024).decode("utf-8").split("\n")

        turn = int(server_msg[0])

        # If the game is over
        if turn == -999:
            return ReversiGameState(None, turn)

        # Flip is necessary because of the way the server does indexing
        board = np.flip(np.array([int(x) for x in server_msg[4:68]]).reshape(8, 8), 0)

        return ReversiGameState(board, turn)

    def send_move(self, move):
        # The 7 - bit is necessary because of the way the server does indexing
        move_str = str(7 - move[0]) + "\n" + str(move[1]) + "\n"
        self.sock.send(move_str.encode("utf-8"))
