#!/usr/bin/env python3

from typing import Any, Optional
import numpy as np
import numpy.typing as npt

# A class that represents a tic tac toe game

BoardTuple = tuple[int, int, int, int, int, int, int, int, int]
Board = npt.NDArray[np.int_]


def board_to_string(board):
    return f"""
        {board[0]} | {board[1]} | {board[2]}
        {board[3]} | {board[4]} | {board[5]}
        {board[6]} | {board[7]} | {board[8]}
    """


class TicTacToe:
    def __init__(self):
        self._board = np.zeros(9, dtype=int)
        self._current_player = 1
        self._winner = None
        self._game_over = False

    def __str__(self):
        return board_to_string(self._board)

    def play(self, position: int):
        if self._game_over:
            return
        if self._board[position] != 0:
            self._game_over = True
            return
        self._board[position] = self._current_player
        self._stabilize_board()
        self._check_for_winner()
        self._switch_player()

    def list_valid_actions(self):
        valid_actions = []
        for i in range(9):
            if self._board[i] == 0:
                valid_actions.append(i)
        return valid_actions

    def observe(self) -> Board:
        return self._board

    def _stabilize_board(self) -> None:
        """Rotates and flips the board to orient it consistently.

        For example, a board with only an X in the top left is the same as a board with only an X any other corner.
        """
        self._board = np.asarray(min(self._symmetric_combinations()))

    def _symmetric_combinations(self):
        combinations = list()

        board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        for y in range(3):
            for x in range(3):
                value = self._board[y * 3 + x]
                board[y][x] = value

        for _ in range(4):
            board = np.rot90(board)
            for _ in range(2):
                board = np.flip(board, axis=0)
                for _ in range(2):
                    board = np.flip(board, axis=1)
                    combinations.append(self._flatten(board))

        return combinations

    def _flatten(self, board):
        flattened = []

        for row in board:
            for cell in row:
                flattened.append(cell)

        return tuple(flattened)

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
        print("Valid actions: ", game.list_valid_actions())
        position = int(input("Enter position: "))
        game.play(position)
    print(game)

    if game.winner == '-':
        print("It's a tie")
    else:
        print(f"Winner is {game.winner}")
