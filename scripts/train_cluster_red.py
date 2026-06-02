"""
集群红方护卫训练脚本
====================
第二阶段：蓝方策略已固定，训练红方护卫星各自对对应蓝星执行侦照/拦截。

用法：
  python train_cluster_red.py --task esc1   # 红护卫1 对蓝1(打击)
  python train_cluster_red.py --task esc2   # 红护卫2 对蓝2(打击)
  python train_cluster_red.py --task esc3   # 红护卫3 对蓝3(干扰)
  python train_cluster_red.py --task esc4   # 红护卫4 对蓝4(侦照)
  python train_cluster_red.py --task esc5   # 红护卫5 对蓝5(侦照)
  python train_cluster_red.py --task esc6   # 红护卫6 对蓝6(操控)

  --checkpoint path/to/agent.pt   从 checkpoint 继续训练
  --timesteps   8000000           总训练步数（默认 8M）
  --run-name    my_exp            自定义 runs/ 下的文件夹名

蓝方 ckpt 路径在脚本顶部 BLUE_CKPT_PATHS 中配置，None = 零推力。
"""

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import argparse
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
from modules.env_wrapper_cluster_red import make_red_wrapper
from modules.networks import Policy, Value

# ── 蓝方 ckpt 路径配置 ────────────────────────────────────────────────────────
# 对应 6 颗蓝星（蓝1~蓝6），None = 该蓝星零推力（ckpt 未就绪时使用）
_RUNS = "/home/star/Downloads/oge_2.0/runs"
BLUE_CKPT_PATHS = [
    f"{_RUNS}/May31_blue_strike1/May31_blue_strike1/checkpoints/best_agent.pt",   # 蓝1
    f"{_RUNS}/May31_blue_strike2/May31_blue_strike2/checkpoints/best_agent.pt",   # 蓝2
    f"{_RUNS}/May31_blue_jam/May31_blue_jam/checkpoints/best_agent.pt",            # 蓝3
    f"{_RUNS}/May31_blue_recon1/May31_blue_recon1/checkpoints/best_agent.pt",     # 蓝4
    f"{_RUNS}/May31_blue_recon2/May31_blue_recon2/checkpoints/best_agent.pt",     # 蓝5
    f"{_RUNS}/May31_blue_operate/May31_blue_operate/checkpoints/best_agent.pt",   # 蓝6
]

# ── 任务名 → esc_idx 映射 ──────────────────────────────────────────────────────
TASK_MAP = {
    "esc1": 0,   # 红护卫1 → 蓝1（打击）
    "esc2": 1,   # 红护卫2 → 蓝2（打击）
    "esc3": 2,   # 红护卫3 → 蓝3（干扰）
    "esc4": 3,   # 红护卫4 → 蓝4（侦照）
    "esc5": 4,   # 红护卫5 → 蓝5（侦照）
    "esc6": 5,   # 红护卫6 → 蓝6（操控）
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASK_MAP.keys()),
                        help="训练哪颗红护卫")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="从 checkpoint 继续训练")
    parser.add_argument("--timesteps", type=int, default=8_000_000,
                        help="总训练步数")
    parser.add_argument("--run-name", type=str, default=None,
                        help="自定义 runs/ 下的文件夹名（默认 ppo_red_<task>）")
    args = parser.parse_args()

    esc_idx  = TASK_MAP[args.task]
    run_name = args.run_name if args.run_name else f"ppo_red_{args.task}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Task: {args.task}  |  device: {device}  |  run: {run_name}")

    # ── 检查蓝方 ckpt ──────────────────────────────────────────────────────────
    for i, p in enumerate(BLUE_CKPT_PATHS):
        if p is None:
            print(f"  蓝{i+1}: 零推力（无 ckpt）")
        elif os.path.exists(p):
            print(f"  蓝{i+1}: {p}")
        else:
            print(f"  蓝{i+1}: ckpt 不存在，将使用零推力 → {p}")

    # ── 环境 ──────────────────────────────────────────────────────────────────
    env = make_red_wrapper(
        esc_idx         = esc_idx,
        env_cfg         = env_cfg,
        red_hv_oe       = RED_HV_OE,
        red_esc_oe_list = RED_ESC_OE_LIST,
        blue_oe_list    = BLUE_REC_OE_LIST,
        blue_ckpt_paths = BLUE_CKPT_PATHS,
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
                "project": "OGE-ClusterRed",
                "name": run_name,
                "tags": ["ppo", args.task, "cluster-red"],
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
        dv_max=0.002,
        clip_actions=False,
    )
    value = Value(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    memory = RandomMemory(
        memory_size=ppo_cfg["rollouts"],
        num_envs=env.num_envs,
        device=device,
    )

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
