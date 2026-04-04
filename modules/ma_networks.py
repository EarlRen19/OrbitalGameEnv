"""Neural networks for multi-agent PPO on OGE.

独立于 networks.py，不影响单智能体训练。
obs_dim 由 PursuerSharedPolicyWrapper 决定（13 或 17 维），
通过 observation_space 自动传入，无需手动指定。
"""

import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


class MAPursuerPolicy(GaussianMixin, Model):
    """追击者共享策略网络（参数共享 IPPO）。

    网络容量比单智能体略大，以应对多个追击者之间的博弈复杂度。
    obs_dim 由 observation_space 自动决定（13 dim 基础版 / 17 dim 含威胁感知）。
    """

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions),
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions) - 1.0)

    def compute(self, inputs, role):
        mean = torch.tanh(self.net(inputs["states"]))
        return mean, self.log_std, {}


class MAPursuerValue(DeterministicMixin, Model):
    """追击者共享价值网络（参数共享 IPPO）。"""

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
