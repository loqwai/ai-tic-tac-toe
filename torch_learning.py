

from collections import deque, namedtuple
from itertools import count
import math

import numpy as np
import torch
from tqdm import tqdm
from tic_tac_toe_env import TicTacToeEnv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, random, capacity):
        self.random = random
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return self.random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def select_action(random, device, policy_net, steps_done, state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model(memory, device, optimizer, policy_net, target_net):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def torch_learning(env: TicTacToeEnv, num_episodes: int, seed: int | None = None) -> DQN:
    random = np.random.RandomState(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(env.observation_space.n, env.action_space.n).to(device)
    target_net = DQN(env.observation_space.n, env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(random, 10000)
    steps_done = 0

    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = select_action(random, device, policy_net, steps_done, state)
            steps_done += 1

            observation, reward, terminated, truncated, _ = env.step(int(action.item()))
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory, device, optimizer, policy_net, target_net)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break

    return policy_net


def evaluate_performance(env: TicTacToeEnv, policy_net: DQN, num_episodes: int, seed: int | None = None):
    stats = {'episode_lengths': [], 'episode_rewards': []}
    return stats


if __name__ == "__main__":
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 0
    env = TicTacToeEnv(random_opponent=False)

    print(f"Training using {device_name}...")
    policy_net = torch_learning(env, 10000, seed=seed)
    print("Evaluating...")
    stats = evaluate_performance(env, policy_net, num_episodes=1000, seed=seed)

    # print("number of unique states:", len(Q))
    print("longest episode", max(stats['episode_lengths']))
    print("highest reward", max(stats["episode_rewards"]))
    print("number of games won", sum([1 for x in stats["episode_rewards"] if x == 10]))
    print("number of games losses", sum([1 for x in stats["episode_rewards"] if x == -20]))
    print("number of games ending in an invalid move", sum([1 for x in stats["episode_rewards"] if x == -10]))
    print("win ratio", sum([1 for x in stats["episode_rewards"] if x == 10]) / len(stats["episode_rewards"]))
