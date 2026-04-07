"""蓝色侦照星第一阶段训练脚本
Phase 1: Blue Recon (RL) vs Red Escort (passive, zero thrust)
"""

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import torch
from copy import deepcopy
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import GaussianMixin, DeterministicMixin
from skrl.models.torch import Model
import torch.nn as nn

from modules.env_wrapper_escort_recon import EscortReconWrapper
from configs.escort_recon_cfg import (
    env_cfg, JD_EPOCH_ESCORT_RECON,
    RED_HV_OE, RED_ESC_OE,
    BLUE_DIST_MIN_KM, BLUE_DIST_MAX_KM,
    BLUE_SUN_ANGLE_MIN, BLUE_SUN_ANGLE_MAX,
    ESCORT_THREAT_DIST_KM,
)
from configs.ppo_cfg import ppo_cfg as base_ppo_cfg


# ── Networks ──────────────────────────────────────────────────────────────────

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, self.num_actions),
        )
        self.log_std = nn.Parameter(-0.5 * torch.ones(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int,   default=10_000_000)
    parser.add_argument("--rollouts",  type=int,   default=2048)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--name",      type=str,   default="blue_recon_phase1")
    parser.add_argument("--checkpoint",type=str,   default=None)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Steps: {args.timesteps:,}  |  Run: {args.name}")

    env = EscortReconWrapper(
        env_cfg=env_cfg,
        jd_epoch=JD_EPOCH_ESCORT_RECON,
        red_hv_oe=RED_HV_OE,
        red_esc_oe=RED_ESC_OE,
        blue_dist_range=(BLUE_DIST_MIN_KM, BLUE_DIST_MAX_KM),
        blue_sun_range=(BLUE_SUN_ANGLE_MIN, BLUE_SUN_ANGLE_MAX),
        train_blue=True,
        escort_intercept_dist=ESCORT_THREAT_DIST_KM,
        seed=args.seed,
    )

    ppo_cfg = deepcopy(base_ppo_cfg)
    ppo_cfg["learning_rate"]  = args.lr
    ppo_cfg["rollouts"]       = args.rollouts
    ppo_cfg["experiment"]["directory"]       = f"runs/{args.name}"
    ppo_cfg["experiment"]["experiment_name"] = args.name
    ppo_cfg["experiment"]["wandb_kwargs"] = {
        "project": "OGE-EscortRecon",
        "name":    args.name,
        "tags":    ["ppo", "blue_recon", "phase1"],
    }
    ppo_cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space.shape[0], "device": device}
    ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    policy = Policy(env.observation_space, env.action_space, device)
    value  = Value(env.observation_space,  env.action_space, device)

    memory = RandomMemory(memory_size=args.rollouts, num_envs=1, device=device)

    agent = PPO(
        models={"policy": policy, "value": value},
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)

    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={"timesteps": args.timesteps, "headless": False},
    )
    trainer.train()


if __name__ == "__main__":
    main()
