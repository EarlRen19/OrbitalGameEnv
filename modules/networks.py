"""Neural networks for PPO on OGE."""

import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


class Policy(GaussianMixin, Model):
    """Actor network for PPO."""

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0, dv_max=0.05):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, self.num_actions),
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions) - 1.0)
        self.dv_max = dv_max

    def compute(self, inputs, role):
        mean = torch.tanh(self.net(inputs["states"]))
        return mean, self.log_std, {}


class Value(DeterministicMixin, Model):
    """Critic network for PPO."""

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}