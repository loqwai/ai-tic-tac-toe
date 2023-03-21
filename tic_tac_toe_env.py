from typing import Any, Literal, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from tic_tac_toe import Board, TicTacToe


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": [None, "ansi"], "render_fps": 4}
    action_space: spaces.Discrete

    def __init__(self, render_mode: Literal["ansi"] | None = None) -> None:
        super().__init__()

        # This should never fail if type checking is enabled
        assert render_mode in self.metadata["render_modes"]

        # Initialize our game
        self.game = TicTacToe()

        # There are 9 cells with 3 possible states per cell: 0-2
        self.observation_space = spaces.MultiDiscrete(9 * [3])

        # There are 9 possible actions: 0-8
        self.action_space = spaces.Discrete(9)

    def reset(self, seed: Optional[int] = None, options: Any = None) -> tuple[Board, dict]:
        super().reset(seed=seed)

        self.game = TicTacToe()

        observation = self.game.observe()
        info = {}  # Info is unused, but the gynmasium API expects reset() to return it

        return observation, info

    def step(self, action: int) -> tuple[Board, float, bool, bool, dict]:
        self.game.play(action)

        # If the action is invalid, the game is over
        observation = self.game.observe()
        reward = 1 if self.game.winner() else 0
        terminated = self.game.terminated()
        truncated = False  # This may get overwritted by gynmasium if the episode is too long (see max_episode_steps)
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        print(self.game)
