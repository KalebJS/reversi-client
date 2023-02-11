import sys
import time

import reversi_bot
from connection import ReversiServerConnection


class ReversiGame:
    def __init__(self, host, bot_move_num, random: bool = False):
        self.bot_move_num = bot_move_num
        self.server_conn = ReversiServerConnection(host, bot_move_num)
        self.bot = reversi_bot.ReversiBot(bot_move_num)
        self.random = random

    def play(self):
        while True:
            state = self.server_conn.get_game_state()

            # If the game is over
            if state.turn == -999:
                time.sleep(1)
                sys.exit()

            # If it is the bot's turn
            if state.turn == self.bot_move_num:
                if self.random:
                    move = self.bot.random_move(state)
                else:
                    move = self.bot.make_move(state)
                self.server_conn.send_move(move)
