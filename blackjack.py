from collections import defaultdict
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
from tic_tac_toe_env import TicTacToeEnv

# env = gym.make('Blackjack-v1')
seed = 0
gym.register("TicTacToe-v0", TicTacToeEnv)
env = gym.make('TicTacToe-v0', random_opponent=True, seed=seed)
torch.manual_seed(seed)

# Q is a dictionary of state-action values
# an example state would be (12, 2, False)
# where 12 is the sum of the cards, 2 is the dealer's card, and False means the player has no usable ace
# an example action would be 0 (hit)

# If you want to affect the learning rate, try changing the value of alpha
# the higher the value of alpha, the more the agent cares about new information

# If you want to affect the exploration rate, try changing the value of epsilon
# the higher the value of epsilon, the more likely the agent is to explore

# If you want to affect the discount rate, try changing the value of gamma
# the higher the value of gamma, the more the agent cares about future rewards

# If we had to ask Roy what he thought about the above parameters, he would say:
# "I think that the learning rate should be high, because I want to learn quickly.
# I think that the exploration rate should be low, because I want to exploit my knowledge.
# I think that the discount rate should be low, because I want to be a realist."
# But we don't have to ask Roy, because we know that he is a realist.
# And we know that he is a realist, because he is a realist.
# And we know that he is a realist, because he is a realist.
# And we know that he is a realist, because he is a realist.
# and we know realists are realists because they are realists.
# But realists are not realists because they are realists.
# they are realists because they are realists.
# And that is why realists are realists.
# - Roy
# But to be honest, Roy is not a realist. He is a realist.


def run_episode(env, Q, epsilon, n_action):

    state, _ = env.reset()
    rewards = []
    actions = []
    states = []
    is_done = False
    while not is_done:
        # probs is a vector of probabilities of taking each action
        probs = torch.ones(n_action) * epsilon / n_action
        # in this case, the best action is the one with the highest Q value
        best_action = torch.argmax(Q[state]).item()
        # we increase the probability of taking the best action
        probs[best_action] += 1.0 - epsilon
        # we sample an action from the distribution
        action = torch.multinomial(probs, 1).item()
        actions.append(action)
        states.append(state)
        state, reward, is_done, is_truncated, info = env.step(action)
        rewards.append(reward)
        if is_done or is_truncated:
            break

    return states, actions, rewards


def mc_control_epsilon_greedy(env, gamma, n_episode, epsilon):
    n_action = env.action_space.n
    G_sum = defaultdict(float)
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.empty(n_action))
    policy = {}

    episode_lengths = []

    for episode in tqdm(range(n_episode)):
        states_t, actions_t, rewards_t = run_episode(env, Q, epsilon, n_action)
        return_t = 0

        G = {}

        episode_lengths.append(len(states_t))
        # we iterate over the episode in reverse order
        for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):

            return_t = gamma * return_t + reward_t
            G[(state_t, action_t)] = return_t

        for state_action, return_t in G.items():
            state, action = state_action
            G_sum[state_action] += return_t
            N[state_action] += 1
            Q[state][action] = G_sum[state_action] / N[state_action]

        for state, actions in Q.items():
            policy[state] = torch.argmax(actions).item()

    print("average episode length: ", np.mean(episode_lengths))
    return Q, policy


def simulate_episode(env, policy) -> float:
    state, _ = env.reset()
    while True:
        if state not in policy:
            return 0
        action = policy[state]
        state, reward, is_done, is_truncated, info = env.step(action)
        if is_done or is_truncated:
            return reward


gamma = 1
n_episode = 10000
epsilon = 0.9

optimal_Q, optimal_policy = mc_control_epsilon_greedy(env, gamma, n_episode, epsilon)

# n_episode = 10000
n_episode = 5000
n_win_optimal = 0
n_loose_optimal = 0

running_total = 0

for _ in tqdm(range(n_episode)):
    reward = simulate_episode(env, optimal_policy)
    running_total += reward
    if reward > 0:
        n_win_optimal += 1
    elif reward < 0:
        n_loose_optimal += 1

print("running total: ", running_total)
print("win rate: ", n_win_optimal / n_episode)
