import os
from typing import Any, Literal, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from tic_tac_toe import Board, TicTacToe


def debug(*args: Any, **kwargs: Any) -> None:
    if "DEBUG" in os.environ:
        print(*args, **kwargs)


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": [None, "ansi"], "render_fps": 4}
    action_space: spaces.Discrete

    def __init__(self, render_mode: Literal["ansi"] | None = None, random_opponent: bool = False) -> None:
        super().__init__()

        # This should never fail if type checking is enabled
        assert render_mode in self.metadata["render_modes"]

        # Initialize our game
        self.game = TicTacToe()

        self.random_opponent = random_opponent

        # There are 9 cells with 3 possible states per cell: 0-2
        self.observation_space = spaces.MultiDiscrete(9 * [3])

        # There are 9 possible actions: 0-8
        self.action_space = spaces.Discrete(9)
        self.random = np.random.RandomState()

    def reset(self, seed: Optional[int] = None, options: Any = None) -> tuple[Board, dict]:
        super().reset(seed=seed)
        self.random = np.random.RandomState(seed=seed)

        self.game = TicTacToe()

        observation = self.game.observe()
        info = {}  # Info is unused, but the gynmasium API expects reset() to return it

        return observation, info

    def step(self, action: int) -> tuple[Board, float, bool, bool, dict]:
        self.game.play(action)

        if self.random_opponent and not self.game.terminated():
            # If the game is not over, then we need to choose an action for the other player
            valid_actions = self.game.list_valid_actions()
            self.game.play(self.random.choice(valid_actions))

        observation = self.game.observe()
        reward = self._reward()
        terminated = self.game.terminated()
        truncated = False  # This may get overwritted by gynmasium if the episode is too long (see max_episode_steps)
        info = {}

        debug("action: ", action)
        debug(self.game)
        debug("reward: ", reward)

        return observation, reward, terminated, truncated, info

    def _reward(self):
        if self.game.winner() == 1:
            return 10
        if self.game.winner() == 2:
            return -20
        if not self.game.terminated():
            return 0
        return -10

    def render(self) -> None:
        print(self.game)
