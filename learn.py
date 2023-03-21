#!/usr/bin/env python

from typing import Optional
import torch
import random
import numpy as np
from gymnasium import spaces

from tic_tac_toe import TicTacToe
from policy import Reinforce


class TicTacToeModel:
    pass


class TicTacToeTrainerEnvironment:
    def __init__(self):
        self.game = TicTacToe()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Discrete(9)
        self.return_queue = []

    def reset(self):
        self.game = TicTacToe()
        return self.game._board

    def step(self, action):
        # this function is responsible for taking an action and returning the
        # observation, reward, terminated, and info
        self.game.play(action)
        # observation is the board state
        observation = self.game._board
        # reward is the reward for the action
        reward = 1 if self.did_i_win() else 0
        self.return_queue.append(reward)
        # terminated is a boolean that is true if the game is over
        terminated = self.is_the_game_over()

        return observation, reward, terminated

    def did_i_win(self):
        return bool(self.game.winner())

    def is_the_game_over(self):
        return self.game._game_over


if __name__ == "__main__":
    seed = 1
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    agent = Reinforce(9, 1)
    env = TicTacToeTrainerEnvironment()

    for _ in range(1000):
        done = False
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation, env.action_space)
            observation, reward, terminated = env.step(action)
            agent.rewards.append(reward)

            if reward:
                assert terminated
                print(env.game)
                print("action: ", action)
                print("A winner!: ", action)

            done = terminated

        agent.update()

    avg_reward = int(np.mean(env.return_queue))
    print(f"win/num_games ratio: {env.return_queue.count(1)}/{len(env.return_queue)}",)
    print("average reward: ", avg_reward)
