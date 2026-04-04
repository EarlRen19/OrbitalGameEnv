"""Multi-agent PPO training for OGE pursuit-evasion.

All pursuers share one policy (parameter-sharing IPPO).
Interceptors are scripted (proportional navigation).
HVT is passive.

Scenarios:
  4v1  : 4 pursuers vs 1 HVT         (--evaders 1 --pursuers 4)
  2v2  : 2 pursuers vs 1HVT+1intp    (--evaders 2 --pursuers 2)
  4v4  : 4 pursuers vs 1HVT+3intp    (--evaders 4 --pursuers 4)
"""

import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import torch
from copy import deepcopy
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from modules.env_wrapper_ma import PursuerSharedPolicyWrapper
from modules.ma_networks import MAPursuerPolicy, MAPursuerValue
from configs.env_cfg import env_cfg
from configs.ppo_cfg import ppo_cfg as base_ppo_cfg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaders",   type=int,   default=2,
                        help="蓝方总数：index 0 = HVT, 1+ = 拦截星  (默认 2 → 1HVT+1拦截)")
    parser.add_argument("--pursuers",  type=int,   default=1,
                        help="红方追击星数量  (默认 1)")
    parser.add_argument("--intercept", type=float, default=30.0,
                        help="拦截距离 km  (默认 30)")
    parser.add_argument("--threat",    action="store_true", default=True,
                        help="obs 中包含最近拦截星方向（仅当 evaders>1 时有效）")
    parser.add_argument("--timesteps", type=int,   default=5_000_000,
                        help="训练总步数  (默认 5M)")
    parser.add_argument("--rollouts",  type=int,   default=2048,
                        help="每次 rollout 步数  (默认 2048)")
    parser.add_argument("--lr",        type=float, default=3e-4,
                        help="学习率  (默认 3e-4)")
    parser.add_argument("--name",      type=str,   default=None,
                        help="实验名称  (默认自动生成)")
    parser.add_argument("--checkpoint",type=str,   default=None,
                        help="续训 checkpoint 路径")
    args = parser.parse_args()

    if args.name is None:
        args.name = f"ma_{args.evaders}v{args.pursuers}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Scenario: {args.pursuers} pursuers vs {args.evaders} evaders "
          f"(1 HVT + {args.evaders - 1} interceptors)")
    print(f"Obs dim: {13 + (4 if args.threat and args.evaders > 1 else 0)}")

    # ── Environment ──────────────────────────────────────────────────────────
    env = PursuerSharedPolicyWrapper(
        env_cfg=env_cfg,
        num_evaders=args.evaders,
        num_pursuers=args.pursuers,
        intercept_distance=args.intercept,
        threat_obs=args.threat,
    )

    # ── PPO config ───────────────────────────────────────────────────────────
    ppo_cfg = deepcopy(base_ppo_cfg)
    ppo_cfg["learning_rate"]  = args.lr
    ppo_cfg["rollouts"]       = args.rollouts
    ppo_cfg["experiment"]["directory"]       = f"runs/{args.name}"
    ppo_cfg["experiment"]["experiment_name"] = args.name
    ppo_cfg["experiment"]["wandb_kwargs"] = {
        "project": "OGE-MultiAgent",
        "name":    args.name,
        "tags":    ["ppo", "ippo", f"{args.pursuers}v{args.evaders}"],
    }
    ppo_cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space.shape[0], "device": device}
    ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # ── Networks (shared across all pursuers) ─────────────────────────────────
    policy = MAPursuerPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    value = MAPursuerValue(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    models = {"policy": policy, "value": value}

    # ── Memory: size × num_envs total transitions per update ─────────────────
    memory = RandomMemory(
        memory_size=args.rollouts,
        num_envs=env.num_envs,   # = num_pursuers
        device=device,
    )

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = PPO(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={"timesteps": args.timesteps, "headless": False},
    )
    print(f"Training {args.name}  ({args.timesteps:,} steps, lr={args.lr})")
    trainer.train()


if __name__ == "__main__":
    main()
