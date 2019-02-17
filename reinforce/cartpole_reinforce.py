#!/usr/bin/env python3
from typing import List

import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from lib.agents import PolicyAgent
from lib.experience import DiscountedExperienceSource

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()
        """
        PGN(
          (net): Sequential(
            (0): Linear(in_features=4, out_features=128, bias=True)
            (1): ReLU()
            (2): Linear(in_features=128, out_features=2, bias=True)
          )
        )

        """
        self.net = ...

    def forward(self, x):
        return self.net(x)


def calculate_mc_returns(rewards: List[float]) -> List[float]:
    """
    Accepts a list of rewards for the whole episode and
    calculates the discounted total reward for every step

    hint: calculate the reward from the end of the local reward list
    :param rewards: array of rewards
    :return: discounted cummulative rewards for each time step (MC returns)
    """
    ...


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-reinforce")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = PolicyAgent(net)

    # exp_source is a helper class that will handle interaction with the environment
    exp_source = DiscountedExperienceSource(env, agent, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_idx = 0
    completed_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_mc_returns = [], [], []
    cur_episode_rewards = []

    for step_idx, exp in enumerate(exp_source):  # exp is ('state', 'action', 'reward', 'last_state'))

        # 1. accumulate states, actions, and current episode rewards
        ...

        # 2. handle end of episode: calculate discounted rewards, reset relevant variables
        if exp.last_state is None:
            ...
            batch_episodes += 1

        # 3. report current progress and write metrics to TensorBoard
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            completed_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))

            print(f"{step_idx}: reward: {reward}, mean_100: {mean_rewards}, episodes: {completed_episodes}")
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", completed_episodes, step_idx)

            # solve criteria
            if mean_rewards > 195:
                print(f"Solved in {step_idx} steps and {completed_episodes} episodes!")
                break

        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        # 4. reset optimizer
        ...
        # 5. create relevant batch tensors (states, actions, mc_returns)
        ...
        # 6. calculate policy gradients (notice that the output of the PGN are just logits)
        ...
        # 7. backpropagate
        ...
        # 8. update parameters
        ...
        # 9. clear all relevant for next batch of training
        batch_episodes = 0
        ...

    writer.close()
