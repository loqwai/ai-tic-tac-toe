#!/usr/bin/env python

from tic_tac_toe import TicTacToe

# A class that trains a model to play tic tac toe
import torch
import random
import numpy as np

from policy import Reinforce


class TicTacToeModel:
    pass


class TicTacToeTrainerEnvironment:
    def __init__(self):
        self.game = TicTacToe()
        self.return_queue = []

    def step(self, action):
        # this function is responsible for taking an action and returning the
        # observation, reward, terminated, and info
        self.game.play(action)
        # observation is the board state
        observation = self.game.board
        # reward is the reward for the action
        reward = self.did_i_win()
        self.return_queue.append(reward)
        # terminated is a boolean that is true if the game is over
        terminated = self.is_the_game_over()

        return observation, reward, terminated

    def did_i_win(self):
        self.tic_tac_toe.check_for_winner()
        return self.tic_tac_toe.is_there_a_winner()

    def is_the_game_over(self):
        return self.tic_tac_toe.game_over


if __name__ == "__main__":
    seed = 1
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    agent = Reinforce(9, 9)
    env = TicTacToeTrainerEnvironment()

    observation = env.game.board
    terminated = False

    for _ in range(1000):
        done = False

        while not done:
            action = agent.sample_action(observation)
            observation, reward, terminated = env.step(action)
            agent.rewards.append(reward)

            done = terminated

        agent.update()

        # use hyperparameters to train the model here

        # break out of loop if game is terminated
        if terminated:
            break

    avg_reward = int(np.mean(env.return_queue))
    print("average reward: ", avg_reward)
