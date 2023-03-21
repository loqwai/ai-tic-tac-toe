from collections import defaultdict
import itertools
from typing import Any
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
import numpy as np
import numpy.typing as npt

from tic_tac_toe import Board, BoardTuple
from tic_tac_toe_env import TicTacToeEnv

# register("TicTacToe-v0", entry_point="tic_tac_toe_env:TicTacToeEnv", max_episode_steps=9)

# env = gym.make("TicTacToe-v0")

env = TicTacToeEnv()

Actions = npt.NDArray[np.int_]


def make_epsilon_greedy_policy(Q: defaultdict[BoardTuple, Actions], epsilon: float, nA: np.int64):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation: Board):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[tuple(observation)])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_learning(env: TicTacToeEnv, num_episodes: int, discount_factor: float = 1.0, alpha: float = 0.5, epsilon: float = 0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q: defaultdict[BoardTuple, Actions] = defaultdict(lambda: np.zeros(env.action_space.n, dtype=int))  # type: ignore

    # Keeps track of useful statistics
    stats = {'episode_lengths': np.zeros(num_episodes), 'episode_rewards': np.zeros(num_episodes)}

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state, _ = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, truncated, _ = env.step(action)

            # Update statistics
            stats['episode_rewards'][i_episode] += reward
            stats['episode_lengths'][i_episode] = t

            # TD Update
            hashable_state = tuple(state)
            hashable_next_state = tuple(next_state)
            best_next_action = np.argmax(Q[hashable_next_state])
            td_target = reward + discount_factor * Q[hashable_next_state][best_next_action]
            td_delta = td_target - Q[hashable_state][action]
            Q[hashable_state][action] += alpha * td_delta

            if done or truncated:
                break

            state = next_state

    return Q, stats


if __name__ == "__main__":
    Q, stats = q_learning(env, 500)
    print(stats)