#!/usr/bin/env python3
from typing import Tuple

import gym

from lib import wrappers

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5           # Mean reward that indicates of full game solution

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000                # History replay buffer size
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000          # Wait this amount of frames until syncing network and target network
REPLAY_START_SIZE = 10000          # Mandatory initial history buffer size

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


class DQN(nn.Module):
    """
    Implement the Q network architecture:

    DQN(
      (conv): Sequential(
        (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
        (1): ReLU()
        (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        (3): ReLU()
        (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        (5): ReLU()
      )
      (fc): Sequential(
        (0): Linear(in_features=XXXX, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=6, bias=True)
      )
    )

    Don't forget to calculate XXXX
    """

    def __init__(self, input_shape, n_actions):
        """
        :param input_shape: tuple
        :param num_actions: number of possible actions
        """
        super(DQN, self).__init__()

    def forward(self, x):
        pass
        # YOUR CODE HERE


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        """

        :param batch_size:
        :return: array of states, array of actions, array of rewards, array of done flags, array of next states
        """
        # YOUR CODE HERE


class Agent:
    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu") -> float:
        """
        :param net: Q value approximator
        :param epsilon: epsilon-greedy policy
        :param device: "cpu" or "gpu"
        :return: total episode reward if last step, and None if not.
        """
        episode_reward: float = None

        # YOUR CODE HERE
        # 1. calculate q_vals for current state
        # 2. get actions
        # 3. get next step
        # 4. accumulate reward
        # 5. handle end of episode

        return episode_reward


def calc_loss(batch, net: nn.Module, target_net: nn.Module, device="cpu") -> torch.Tensor:
    """
    :param batch: tuple of states, actions, rewards, done flags, and next states (as numpy arrays)
    :param net: Q network
    :param target_net: target Q network - weights shouldn't be updated.
    :param device: "cpu" or "gpu"
    :return: mean square error loss
    """
    states, actions, rewards, dones, next_states = batch

    states_tensor = torch.Tensor(states).to(device)
    next_states_tensor = torch.Tensor(next_states).to(device)
    actions_tensor = torch.LongTensor(actions).to(device)     # 1D tensor of indices of the chosen actions
    rewards_tensor = torch.Tensor(rewards).to(device)         # 1D tensor of rewards for (states, actions)
    done_mask_tensor = torch.ByteTensor(dones).to(device)     # indexes of final states (i.e. states where there's no next state)

    # 1. calculate qvals
    q_vals = ...               # hint: use torch.gather()
    q_vals_next_state = ...    # hint: use target_net

    # 2. handle terminal state action values here (Why does it require special care?)
    ...

    # 3. make target_net not trainable
    ...

    # 4. calculate TD target
    expected_q_vals = ...
    return nn.MSELoss()(q_vals, expected_q_vals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)

    net = ...         # Create DQN
    target_net = ...  # Create target DQN
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=str(device))
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward: float = np.mean(total_rewards[-100:])
            print(f"{frame_idx}: done {len(total_rewards)} games, mean reward {mean_reward}, eps {epsilon}, speed {speed} f/s")
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print(f"Best mean reward updated {best_mean_reward} -> {mean_reward}, model saved")
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print(f"Solved in {frame_idx} frames!")
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            target_net.load_state_dict(net.state_dict())

        # 1. zero out the optimizer's gradient
        # 2. sample a new batch
        # 3. compute loss (calc_loss())
        # 4. backpropagate
        # 5. update params
    writer.close()
