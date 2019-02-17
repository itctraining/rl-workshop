import torch
from torch.nn import functional as F

import numpy as np


class BaseAgent:
    """
    Abstract Agent interface
    """
    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


class PolicyAgent(BaseAgent):
    """
    Policy agent gets action probabilities from the model and samples actions from it
    """
    def __init__(self, model, device="cpu"):
        super().__init__()
        self.model = model
        self.device = device

    def __call__(self, states, agent_states=None):
        """
        Return actions from given list of states
        :param states list of states
        :param agent_states
        :return list of actions
        """
        if agent_states is None:
            agent_states = [None] * len(states)

        states = self.preprocess_states(states)
        states = states.to(self.device)
        probs_tensor = self.model(states)
        probs_tensor = F.softmax(probs_tensor, dim=1)
        probs = probs_tensor.data.cpu().numpy()
        actions = self.select_action(probs)
        return np.array(actions), agent_states

    def preprocess_states(self, states):
        """
        convert states to tensor
        """
        np_states = np.array(states, dtype=np.float32)
        return torch.Tensor(np_states)

    def select_action(self, probs: np.ndarray) -> np.array:
        """
        sample actions from probability distributions
        :param probs: array of probability distributions (each distribution is also an array)
        :return:
        """
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)
