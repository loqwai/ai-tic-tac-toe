#!/usr/bin/env python3

import numpy as np

# A class that represents a tic tac toe game


class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.player = 1
        self.winner = None
        self.game_over = False

    def __str__(self):
        return f"""
            {self.board[0]} | {self.board[1]} | {self.board[2]}
            {self.board[3]} | {self.board[4]} | {self.board[5]}
            {self.board[6]} | {self.board[7]} | {self.board[8]}
        """

    def play(self, position: int):
        if self.game_over:
            raise Exception("Game is over")
        if self.board[position] != 0:
            raise Exception("Position is already taken")
        self.board[position] = self.player
        self._check_for_winner()
        self._switch_player()

    def list_valid_actions(self, player):
        valid_actions = []
        for i in range(9):
            if self.board[i] is None:
                valid_actions.append(i)
        return valid_actions

    def _switch_player(self):
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def _check_for_winner(self):
        winning_combinations = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]
        for combination in winning_combinations:
            if self.board[combination[0]] == self.board[combination[1]] == self.board[combination[2]] == self.player:
                self.winner = self.player
                self.game_over = True
                return
        if 0 not in self.board:
            self.game_over = True

    def is_there_a_winner(self):
        return self.winner is not None


if __name__ == "__main__":
    game = TicTacToe()
    while not game.game_over:
        print(game)
        print(f"Player {game.player}'s turn")
        print("Valid actions: ", game.list_valid_actions(game.player))
        position = int(input("Enter position: "))
        game.play(position)
    print(game)

    if game.winner == '-':
        print("It's a tie")
    else:
        print(f"Winner is {game.winner}")
