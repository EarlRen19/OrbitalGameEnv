"""
集群蓝方策略验证脚本
====================
加载 best_agent.pt，跑 N 个 episode，统计任务成功率和关键指标。

用法：
  python evaluate_cluster_blue.py --task strike1 --checkpoint path/to/best_agent.pt
  python evaluate_cluster_blue.py --task jam     --checkpoint path/to/best_agent.pt --episodes 50

支持任务：strike1, strike2, jam, recon1, recon2, operate
"""

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import argparse
import numpy as np
import torch

from configs.cluster_escort_cfg import (
    env_cfg,
    RED_HV_OE,
    RED_ESC_OE_LIST,
    BLUE_REC_OE_LIST,
)
from modules.env_wrapper_cluster_blue import (
    make_blue_wrapper,
    BlueStrikeWrapper,
    BlueJamWrapper,
    BlueReconWrapper,
    BlueOperateWrapper,
)
from modules.networks import Policy
from skrl.resources.preprocessors.torch import RunningStandardScaler

# ── 任务映射 ──────────────────────────────────────────────────────────────────
TASK_MAP = {
    "strike1": 0,
    "strike2": 1,
    "jam":     2,
    "recon1":  3,
    "recon2":  4,
    "operate": 5,
}

# ── 各任务成功判定阈值（与 wrapper 里保持一致） ───────────────────────────────
TASK_SUCCESS_CRITERIA = {
    "strike1": dict(dist_km=20.0, angle_deg=90.0, duration_s=40.0,   label="dist≤20km & solar≤90° & 持续≥40s"),
    "strike2": dict(dist_km=20.0, angle_deg=90.0, duration_s=40.0,   label="dist≤20km & solar≤90° & 持续≥40s"),
    "jam":     dict(dist_km=20.0, angle_deg=5.0,  duration_s=600.0,  label="dist≤20km & jam_angle≤5° & 持续≥600s"),
    "recon1":  dict(dist_km=20.0, angle_deg=60.0, duration_s=120.0,  label="dist≤20km & solar≤60° & 持续≥120s"),
    "recon2":  dict(dist_km=20.0, angle_deg=60.0, duration_s=120.0,  label="dist≤20km & solar≤60° & 持续≥120s"),
    "operate": dict(dist_km=2.0,  angle_deg=None, duration_s=None,   label="dist≤2km"),
}


def run_episode(env, policy, preprocessor, device, task: str, timestep: float):
    """
    跑一个 episode，返回统计字典。

    返回：
      success      : bool
      total_reward : float
      steps        : int
      min_dist_km  : float
      final_dv_ratio: float
      min_angle_deg: float | None  (operate 任务无角度指标)
      max_in_zone_s: float         (最长连续在区时间，operate 无此项)
    """
    obs, _ = env.reset()
    done = truncated = False
    total_reward = 0.0
    steps = 0
    success = False

    min_dist_km   = float("inf")
    min_angle_deg = float("inf")
    max_in_zone_s = 0.0
    cur_in_zone_s = 0.0
    final_dv_ratio = 1.0

    criteria = TASK_SUCCESS_CRITERIA[task]

    while not (done or truncated):
        with torch.no_grad():
            # 必须先经过 RunningStandardScaler 归一化，和训练时一致
            obs_norm = preprocessor(obs)
            action, _, _ = policy.act({"states": obs_norm}, role="policy")

        obs, rew, term, trunc, info = env.step(action)
        done      = bool(term.squeeze())
        truncated = bool(trunc.squeeze())
        total_reward += float(rew.squeeze())
        steps += 1

        # 从观测里读取当前状态
        obs_np = obs.squeeze().cpu().numpy()
        dist_km    = float(obs_np[6]) * 20.0
        dv_ratio   = float(obs_np[11])
        final_dv_ratio = dv_ratio

        min_dist_km = min(min_dist_km, dist_km)

        if task != "operate":
            angle_rad = float(obs_np[7]) * np.pi
            angle_deg = np.rad2deg(angle_rad)
            min_angle_deg = min(min_angle_deg, angle_deg)

            in_zone = (dist_km <= criteria["dist_km"] and angle_deg <= criteria["angle_deg"])
            if in_zone:
                cur_in_zone_s += timestep
                max_in_zone_s  = max(max_in_zone_s, cur_in_zone_s)
            else:
                cur_in_zone_s = 0.0
        else:
            # operate：只看距离
            if dist_km <= criteria["dist_km"]:
                success = True

    # 判定成功（通用：wrapper 内 done=True 且 reward>0 意味着成功）
    # 更可靠：直接用任务完成条件再判断一次
    if task != "operate":
        if max_in_zone_s >= criteria["duration_s"]:
            success = True

    return dict(
        success       = success,
        total_reward  = total_reward,
        steps         = steps,
        min_dist_km   = min_dist_km,
        min_angle_deg = min_angle_deg if task != "operate" else None,
        max_in_zone_s = max_in_zone_s if task != "operate" else None,
        final_dv_ratio= final_dv_ratio,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASK_MAP.keys()),
                        help="要验证的任务")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="best_agent.pt 路径")
    parser.add_argument("--episodes", type=int, default=20,
                        help="验证 episode 数量（默认 20）")
    parser.add_argument("--verbose", action="store_true",
                        help="逐 episode 打印详情")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"任务：{args.task}")
    print(f"成功条件：{TASK_SUCCESS_CRITERIA[args.task]['label']}")
    print(f"Checkpoint：{args.checkpoint}")
    print(f"Device：{device}  |  Episodes：{args.episodes}")
    print(f"{'='*60}")

    # ── 创建环境 ──────────────────────────────────────────────────────────────
    task_idx = TASK_MAP[args.task]
    env = make_blue_wrapper(
        task_idx        = task_idx,
        env_cfg         = env_cfg,
        red_hv_oe       = RED_HV_OE,
        red_esc_oe_list = RED_ESC_OE_LIST,
        blue_oe_list    = BLUE_REC_OE_LIST,
    )
    timestep = float(env_cfg.timestep)

    # ── 加载策略网络 ──────────────────────────────────────────────────────────
    policy = Policy(
        observation_space = env.observation_space,
        action_space      = env.action_space,
        device            = device,
        dv_max            = 0.002,
        clip_actions      = False,
    )
    # skrl checkpoint 格式：dict 含 "policy" key
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "policy" in ckpt:
        policy.load_state_dict(ckpt["policy"])
    else:
        policy.load_state_dict(ckpt)
    policy.to(device)
    policy.eval()

    # ── 加载 state_preprocessor（RunningStandardScaler） ──────────────────────
    obs_dim = env.observation_space.shape[0]
    preprocessor = RunningStandardScaler(size=obs_dim, device=device)
    preprocessor.load_state_dict(ckpt["state_preprocessor"])
    preprocessor.eval()
    print(f"策略 + 预处理器加载成功\n")

    # ── 跑多个 episode ────────────────────────────────────────────────────────
    results = []
    for ep in range(args.episodes):
        r = run_episode(env, policy, preprocessor, device, args.task, timestep)
        results.append(r)
        if args.verbose:
            status = "✓ 成功" if r["success"] else "✗ 失败"
            angle_str = (f"  min_angle={r['min_angle_deg']:.1f}°"
                         if r["min_angle_deg"] is not None else "")
            zone_str  = (f"  in_zone_max={r['max_in_zone_s']:.0f}s"
                         if r["max_in_zone_s"] is not None else "")
            print(f"  ep {ep+1:3d}: {status}  "
                  f"min_dist={r['min_dist_km']:.1f}km"
                  f"{angle_str}{zone_str}"
                  f"  dv_remain={r['final_dv_ratio']*100:.0f}%"
                  f"  reward={r['total_reward']:.1f}")

    # ── 汇总统计 ──────────────────────────────────────────────────────────────
    n_success = sum(r["success"] for r in results)
    success_rate = n_success / args.episodes * 100

    min_dists   = [r["min_dist_km"]    for r in results]
    rewards     = [r["total_reward"]   for r in results]
    dv_ratios   = [r["final_dv_ratio"] for r in results]

    print(f"\n{'─'*60}")
    print(f"成功率：{n_success}/{args.episodes} = {success_rate:.1f}%")
    print(f"最近距离：mean={np.mean(min_dists):.2f}km  "
          f"min={np.min(min_dists):.2f}km  max={np.max(min_dists):.2f}km")
    print(f"剩余燃料：mean={np.mean(dv_ratios)*100:.1f}%  "
          f"min={np.min(dv_ratios)*100:.1f}%")
    print(f"总奖励：  mean={np.mean(rewards):.1f}  "
          f"min={np.min(rewards):.1f}  max={np.max(rewards):.1f}")

    if args.task != "operate":
        angles    = [r["min_angle_deg"]  for r in results]
        in_zones  = [r["max_in_zone_s"]  for r in results]
        print(f"最小角度：mean={np.mean(angles):.1f}°  min={np.min(angles):.1f}°")
        print(f"最长在区：mean={np.mean(in_zones):.0f}s  max={np.max(in_zones):.0f}s")

    print(f"{'─'*60}\n")

    env.close()


if __name__ == "__main__":
    main()
