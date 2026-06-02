"""
集群蓝方训练脚本
================
四种任务分开训练，每次只训练一颗蓝星的策略。

用法：
  python train_cluster_blue.py --task strike1   # 蓝1 打击
  python train_cluster_blue.py --task strike2   # 蓝2 打击
  python train_cluster_blue.py --task jam       # 蓝3 干扰
  python train_cluster_blue.py --task recon1    # 蓝4 侦照
  python train_cluster_blue.py --task recon2    # 蓝5 侦照
  python train_cluster_blue.py --task operate   # 蓝6 操控

  --checkpoint path/to/agent.pt   从 checkpoint 继续训练
  --timesteps   8000000           总训练步数（默认 8M）
"""

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import argparse
from copy import deepcopy

import torch
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler

from configs.cluster_escort_cfg import (
    env_cfg,
    RED_HV_OE,
    RED_ESC_OE_LIST,
    BLUE_REC_OE_LIST,
)
from modules.env_wrapper_cluster_blue import (
    BlueStrikeWrapper,
    BlueReconWrapper,
    BlueJamWrapper,
    BlueOperateWrapper,
    make_blue_wrapper,
)
from modules.networks import Policy, Value

# ── 任务名 → task_idx 映射 ────────────────────────────────────────────────────
TASK_MAP = {
    "strike1": 0,
    "strike2": 1,
    "jam":     2,
    "recon1":  3,
    "recon2":  4,
    "operate": 5,
}

TASK_DV_MAX = {
    "strike1": 0.002,
    "strike2": 0.002,
    "jam":     0.002,
    "recon1":  0.002,
    "recon2":  0.002,
    "operate": 0.002,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASK_MAP.keys()),
                        help="训练哪颗蓝星的任务")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="从 checkpoint 继续训练")
    parser.add_argument("--timesteps", type=int, default=8_000_000,
                        help="总训练步数")
    parser.add_argument("--run-name", type=str, default=None,
                        help="自定义 runs/ 下的文件夹名（默认 ppo_blue_<task>）")
    args = parser.parse_args()

    task_idx = TASK_MAP[args.task]
    dv_max   = TASK_DV_MAX[args.task]
    run_name = args.run_name if args.run_name else f"ppo_blue_{args.task}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Task: {args.task}  |  device: {device}  |  run: {run_name}")

    # ── 环境 ──────────────────────────────────────────────────────────────────
    env = make_blue_wrapper(
        task_idx       = task_idx,
        env_cfg        = env_cfg,
        red_hv_oe      = RED_HV_OE,
        red_esc_oe_list= RED_ESC_OE_LIST,
        blue_oe_list   = BLUE_REC_OE_LIST,
    )

    # ── PPO 配置 ───────────────────────────────────────────────────────────────
    ppo_cfg = {
        "rollouts": 2048,
        "learning_epochs": 8,
        "mini_batches": 4,
        "discount_factor": 0.99,
        "lambda": 0.95,
        "learning_rate": 3e-4,
        "learning_rate_scheduler": None,
        "learning_rate_scheduler_kwargs": {},
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "grad_norm_clip": 0.5,
        "entropy_loss_scale": 0.01,
        "value_loss_scale": 0.5,
        "kl_threshold": 0,
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_space.shape[0],
                                       "device": device},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},
        "random_timesteps": 0,
        "learning_starts": 0,
        "experiment": {
            "directory": f"runs/{run_name}",
            "experiment_name": run_name,
            "write_interval": 1000,
            "checkpoint_interval": 50000,
            "wandb": True,
            "wandb_kwargs": {
                "project": "OGE-ClusterBlue",
                "name": run_name,
                "tags": ["ppo", args.task, "cluster-blue"],
            },
        },
    }

    trainer_cfg = {
        "timesteps": args.timesteps,
        "headless": True,
    }

    # ── 网络 ───────────────────────────────────────────────────────────────────
    policy = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        dv_max=dv_max,
        clip_actions=False,
    )
    value = Value(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # ── Memory ─────────────────────────────────────────────────────────────────
    memory = RandomMemory(
        memory_size=ppo_cfg["rollouts"],
        num_envs=env.num_envs,
        device=device,
    )

    # ── Agent ──────────────────────────────────────────────────────────────────
    agent = PPO(
        models={"policy": policy, "value": value},
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)

    # ── 训练 ───────────────────────────────────────────────────────────────────
    trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)
    print(f"Starting training: {args.task}")
    trainer.train()


if __name__ == "__main__":
    main()
