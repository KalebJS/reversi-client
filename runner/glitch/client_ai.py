import sys

from game import ReversiGame

if __name__ == "__main__":
    server_address = sys.argv[1]
    bot_move_number = int(sys.argv[2])

    reversi_game = ReversiGame(server_address, bot_move_number)
    reversi_game.play()
