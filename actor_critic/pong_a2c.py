#!/usr/bin/env python3
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim

from lib import wrappers
from lib.agents import PolicyAgent
from lib.common import TBMeanTracker, RewardTracker
from lib.experience import DiscountedExperienceSource
from tensorboardX import SummaryWriter

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"


class AtariA2C(nn.Module):
    """
    The network will have outputs: one as a single output for V(s) approximation
    and another one for the policy distribution.

    Don't forget to figure out XXXX

    AtariA2C(
      (conv): Sequential(
        (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
        (1): ReLU()
        (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        (3): ReLU()
        (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        (5): ReLU()
      )
      (policy): Sequential(
        (0): Linear(in_features=XXXX, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=6, bias=True)
      )
      (value): Sequential(
        (0): Linear(in_features=XXXX, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=1, bias=True)
      )
    )
    """
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

    def forward(self, x):
        pass


def unpack_batch(batch, net: nn.Module, device='cpu'):
    """
    Convert experience batch into training tensors
    :param batch: experience batch of (s,a,r,s') - where r is discounted reward between s and s' (number of steps is REWARD_STEPS constant)
    :param net:
    :return: states tensor, actions tensor, qvals tensor
    """
    states = []        # s
    actions = []       # a
    rewards = []       # r
    not_done_idx = []  # indices of non terminal entries
    last_states = []   # s'
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)

        # episode not done
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    states_tensor = torch.FloatTensor(states).to(device)
    actions_tensor = torch.LongTensor(actions).to(device)

    # calculate Q vals
    ...
    qvals_tensor = ...
    return states_tensor, actions_tensor, qvals_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # make multiple environments to get more experience faster
    def make_env(): return wrappers.make_env(DEFAULT_ENV_NAME)
    envs = [make_env() for _ in range(NUM_ENVS)]

    # create summary writer
    writer = SummaryWriter(comment="-pong-a2c_" + args.name)

    # create network
    net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)

    agent = PolicyAgent(lambda x: net(x)[0], device=device)
    exp_source = DiscountedExperienceSource(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []

    # moving average for last 100 episodes reward of 18 considers the game as solved
    with RewardTracker(writer, stop_reward=18) as tracker:
        with TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                # handle new rewards - pop_total_rewards collects all rewards for fully played episodes until now
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                # keep collecting experience until batch size reached
                if len(batch) < BATCH_SIZE:
                    continue

                states_tensor, actions_tensor, qvals_tensor = unpack_batch(batch, net, device=str(device))
                batch.clear()

                # 1. reset optimizer gradients
                ...
                # 2. calculate value loss
                ...
                value_loss_tensor = ...
                # 3. calculate advantage
                ...
                advantage_tensor = ...
                # 4. calculate policy loss
                ...
                policy_loss_tensor = ...

                # 5. calculate policy gradients
                # NOTE: you will have to use .backward(retain_graph=True) since there are going to be multiple backprops
                # over the same dynamic graph - one for PG and another for V
                ...

                # 6. get all gradients for tracking
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                # 7. calculate value gradients
                ...

                # Optimization to make the network updates not too big - this improves training stability
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)

                # 8. optimization update
                ...

                # get full loss
                total_loss_tensor = policy_loss_tensor + value_loss_tensor

                tb_tracker.track("advantage", advantage_tensor, step_idx)
                tb_tracker.track("batch_rewards", qvals_tensor, step_idx)
                tb_tracker.track("loss_policy", policy_loss_tensor, step_idx)
                tb_tracker.track("loss_value", value_loss_tensor, step_idx)
                tb_tracker.track("loss_total", total_loss_tensor, step_idx)
                tb_tracker.track("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max", np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var", np.var(grads), step_idx)
