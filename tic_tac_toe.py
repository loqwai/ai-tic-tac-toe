#!/usr/bin/env python3

from typing import Any, Optional
import numpy as np
import numpy.typing as npt

# A class that represents a tic tac toe game

BoardTuple = tuple[int, int, int, int, int, int, int, int, int]
Board = npt.NDArray[np.int_]


class TicTacToe:
    def __init__(self):
        self._board = np.zeros(9, dtype=int)
        self._current_player = 1
        self._winner = None
        self._game_over = False

    def __str__(self):
        return f"""
            {self._board[0]} | {self._board[1]} | {self._board[2]}
            {self._board[3]} | {self._board[4]} | {self._board[5]}
            {self._board[6]} | {self._board[7]} | {self._board[8]}
        """

    def play(self, position: int):
        if self._game_over:
            return
        if self._board[position] != 0:
            self._game_over = True
            return
        self._board[position] = self._current_player
        self._check_for_winner()
        self._switch_player()

    def list_valid_actions(self, player):
        valid_actions = []
        for i in range(9):
            if self._board[i] is None:
                valid_actions.append(i)
        return valid_actions

    def observe(self) -> Board:
        return self._board

    def terminated(self):
        self._check_for_winner()
        return self._game_over

    def _switch_player(self):
        if self._current_player == 1:
            self._current_player = 2
        else:
            self._current_player = 1

    def _check_for_winner(self):
        if self._game_over:
            return

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
            if self._board[combination[0]] == self._board[combination[1]] == self._board[combination[2]] == self._current_player:
                self._winner = self._current_player
                self._game_over = True
                return

        if 0 not in self._board:
            self._game_over = True

    def winner(self) -> Optional[int]:
        self._check_for_winner()
        return self._winner


if __name__ == "__main__":
    game = TicTacToe()
    while not game._game_over:
        print(game)
        print(f"Player {game._current_player}'s turn")
        print("Valid actions: ", game.list_valid_actions(game._current_player))
        position = int(input("Enter position: "))
        game.play(position)
    print(game)

    if game.winner == '-':
        print("It's a tie")
    else:
        print(f"Winner is {game.winner}")
