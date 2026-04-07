"""红色护卫星 Phase 2 训练脚本
蓝色侦照星加载 Phase 1 checkpoint，固定策略运行；
红色护卫星为 RL agent，目标是侦照蓝星（以蓝星为顶点，60° 内，20km，持续 200s）。
"""

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import argparse
import dataclasses
import torch
from copy import deepcopy
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler

from modules.env_wrapper_escort_recon import EscortReconWrapper
from scripts.train_blue_recon_phase1 import Policy, Value
from configs.escort_recon_cfg import (
    env_cfg, JD_EPOCH_ESCORT_RECON,
    RED_HV_OE, RED_ESC_OE,
    BLUE_DIST_MIN_KM, BLUE_DIST_MAX_KM,
    BLUE_SUN_ANGLE_MIN, BLUE_SUN_ANGLE_MAX,
    ESCORT_THREAT_DIST_KM,
)

# Phase 2 固定蓝星初始六根数（a km，角度 rad）
BLUE_OE_FIXED = dict(
    a    = 42169.502913,
    e    = 0.0,
    i    = 0.002287,
    raan = 1.592829,
    w    = 0.0,
    M    = 0.424435,
)
from configs.ppo_cfg import ppo_cfg as base_ppo_cfg

# Phase 2 默认蓝色 checkpoint
DEFAULT_BLUE_CKPT = (
    "runs/April_7_blue_recon_phase1/blue_recon_phase1/checkpoints/best_agent.pt"
)

ESC_DV_INIT    = 0.020   # 红护卫初始燃料 km/s（20 m/s）
ESC_DV_MAX     = 0.002   # 红护卫单步最大 km/s（2 m/s）


def load_blue_policy(checkpoint_path, obs_dim, act_dim, device):
    """加载蓝色侦照星策略（固定，不参与训练）。"""
    import gymnasium
    import numpy as np
    obs_space = gymnasium.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(obs_dim,), dtype="float32")
    act_space = gymnasium.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(act_dim,), dtype="float32")

    policy = Policy(obs_space, act_space, device)
    ckpt   = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    policy.to(device).eval()

    preprocessor = RunningStandardScaler(size=obs_dim, device=device)
    if "state_preprocessor" in ckpt:
        preprocessor.load_state_dict(ckpt["state_preprocessor"])
    preprocessor.eval()

    return policy, preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blue-checkpoint", type=str, default=DEFAULT_BLUE_CKPT,
                        help="蓝色侦照星 Phase 1 checkpoint")
    parser.add_argument("--timesteps", type=int,   default=10_000_000)
    parser.add_argument("--rollouts",  type=int,   default=2048)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--name",      type=str,   default="red_escort_phase2")
    parser.add_argument("--checkpoint",type=str,   default=None,
                        help="红护卫续训 checkpoint")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Steps: {args.timesteps:,}  |  Run: {args.name}")
    print(f"Blue checkpoint: {args.blue_checkpoint}")

    # Phase 2 C++ 配置：给 RedEsc（evader）分配燃料，C++ 才能正确 clip 动作
    env_cfg_p2 = dataclasses.replace(
        env_cfg,
        dv_init_blue         = ESC_DV_INIT,
        dv_max_per_step_blue = ESC_DV_MAX,
    )

    # 先加载蓝色策略（固定）
    blue_policy, blue_prep = load_blue_policy(
        args.blue_checkpoint,
        obs_dim=17, act_dim=3,
        device=device,
    )

    # 创建环境（train_blue=False → 红护卫为 RL）
    env = EscortReconWrapper(
        env_cfg=env_cfg_p2,
        jd_epoch=JD_EPOCH_ESCORT_RECON,
        red_hv_oe=RED_HV_OE,
        red_esc_oe=RED_ESC_OE,
        blue_dist_range=(BLUE_DIST_MIN_KM, BLUE_DIST_MAX_KM),
        blue_sun_range=(BLUE_SUN_ANGLE_MIN, BLUE_SUN_ANGLE_MAX),
        train_blue=False,
        blue_policy=blue_policy,
        blue_preprocessor=blue_prep,
        escort_intercept_dist=ESCORT_THREAT_DIST_KM,
        esc_dv_init=ESC_DV_INIT,
        blue_oe=BLUE_OE_FIXED,   # Phase 2：固定蓝星初始位置
        seed=args.seed,
    )

    # PPO 配置
    ppo_cfg = deepcopy(base_ppo_cfg)
    ppo_cfg["learning_rate"]  = args.lr
    ppo_cfg["rollouts"]       = args.rollouts
    ppo_cfg["experiment"]["directory"]       = f"runs/{args.name}"
    ppo_cfg["experiment"]["experiment_name"] = args.name
    ppo_cfg["experiment"]["wandb_kwargs"] = {
        "project": "OGE-EscortRecon",
        "name":    args.name,
        "tags":    ["ppo", "red_escort", "phase2"],
    }
    ppo_cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space.shape[0], "device": device}
    ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # 网络
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
        print(f"Loading red escort checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)

    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={"timesteps": args.timesteps, "headless": False},
    )
    trainer.train()


if __name__ == "__main__":
    main()
