#!/usr/bin/env python3

# A class that represents a tic tac toe game
class TicTacToe:
    def __init__(self):
        self.board = ["-"] * 9
        self.player = "X"
        self.winner = None
        self.game_over = False

    def __str__(self):
        return f"""
            {self.board[0]} | {self.board[1]} | {self.board[2]}
            {self.board[3]} | {self.board[4]} | {self.board[5]}
            {self.board[6]} | {self.board[7]} | {self.board[8]}
        """

    def play(self, position):
        if self.game_over:
            raise Exception("Game is over")
        if self.board[position] is not None:
            raise Exception("Position is already taken")
        self.board[position] = self.player
        self._check_for_winner()
        self._switch_player()

    def _switch_player(self):
        if self.player == "X":
            self.player = "O"
        else:
            self.player = "X"

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
        if None not in self.board:
            self.game_over = True


if __name__ == "__main__":
    game = TicTacToe()
    while not game.game_over:
        print(game)
        position = int(input("Enter position: "))
        game.play(position)
    print(game)

    if game.winner is '-':
        print("It's a tie")
    else:
        print(f"Winner is {game.winner}")
