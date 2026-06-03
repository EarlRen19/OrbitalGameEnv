"""
集群 6v6 对抗验证脚本
=====================
同时加载 6 颗红护卫 + 6 颗蓝星的策略，跑完整 13 星场景，
输出每颗红护卫是否成功拦截（侦照）对应蓝星。

用法：
  python evaluate_cluster_6v6.py --episodes 20 --verbose
  python evaluate_cluster_6v6.py --episodes 50

ckpt 路径在脚本顶部 RED_CKPT_PATHS / BLUE_CKPT_PATHS 中配置。
"""

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import argparse
import numpy as np
import torch
import gymnasium

from dataclasses import asdict
from skrl.resources.preprocessors.torch import RunningStandardScaler

from configs.cluster_escort_cfg import (
    env_cfg,
    RED_HV_OE,
    RED_ESC_OE_LIST,
    BLUE_REC_OE_LIST,
)
from modules.networks import Policy
from modules.env_wrapper_cluster_red import (
    _coe2rv, _make_state,
    BLUE_IDXS, RED_ESC_IDXS, HV_IDX,
    BLUE_TASK_TYPES, BLUE_DV_INIT,
    RED_ESC_DV_INIT, RED_ESC_DV_LIST,
    TASK_RECON, TASK_STRIKE, TASK_JAM, TASK_OPERATE,
)

try:
    from oge_py._oge_py_ma import MultiAgentOGEEnv as _CppEnv
    from oge_py._oge_py import SatState
except ImportError:
    from _oge_py_ma import MultiAgentOGEEnv as _CppEnv
    from _oge_py import SatState

# ── ckpt 路径配置 ──────────────────────────────────────────────────────────────
_RUNS = "/home/star/Downloads/oge_2.0/runs"

RED_CKPT_PATHS = [
    f"{_RUNS}/June1_red_esc1/June1_red_esc1/checkpoints/best_agent.pt",
    f"{_RUNS}/June1_red_esc2/June1_red_esc2/checkpoints/best_agent.pt",
    f"{_RUNS}/June1_red_esc3/June1_red_esc3/checkpoints/best_agent.pt",
    f"{_RUNS}/June1_red_esc4/June1_red_esc4/checkpoints/best_agent.pt",
    f"{_RUNS}/June1_red_esc5/June1_red_esc5/checkpoints/best_agent.pt",
    f"{_RUNS}/June1_red_esc6/June1_red_esc6/checkpoints/best_agent.pt",
]

BLUE_CKPT_PATHS = [
    f"{_RUNS}/May31_blue_strike1/May31_blue_strike1/checkpoints/best_agent.pt",
    f"{_RUNS}/May31_blue_strike2/May31_blue_strike2/checkpoints/best_agent.pt",
    f"{_RUNS}/May31_blue_jam/May31_blue_jam/checkpoints/best_agent.pt",
    f"{_RUNS}/May31_blue_recon1/May31_blue_recon1/checkpoints/best_agent.pt",
    f"{_RUNS}/May31_blue_recon2/May31_blue_recon2/checkpoints/best_agent.pt",
    f"{_RUNS}/May31_blue_operate/May31_blue_operate/checkpoints/best_agent.pt",
]

# 红护卫成功条件（镜像蓝星任务类型）
ESC_SUCCESS = [
    dict(task=TASK_STRIKE,  dist_km=20.0, angle_deg=90.0, duration_s=200.0),  # 红1 → 蓝1打击
    dict(task=TASK_STRIKE,  dist_km=20.0, angle_deg=90.0, duration_s=200.0),  # 红2 → 蓝2打击
    dict(task=TASK_JAM,     dist_km=20.0, angle_deg=5.0,  duration_s=600.0),  # 红3 → 蓝3干扰
    dict(task=TASK_RECON,   dist_km=20.0, angle_deg=60.0, duration_s=200.0),  # 红4 → 蓝4侦照
    dict(task=TASK_RECON,   dist_km=20.0, angle_deg=60.0, duration_s=200.0),  # 红5 → 蓝5侦照
    dict(task=TASK_OPERATE, dist_km=2.0,  angle_deg=None, duration_s=200.0),  # 红6 → 蓝6操控
]


# ── 策略加载 ──────────────────────────────────────────────────────────────────

def load_policy(ckpt_path: str, obs_dim: int, device):
    """加载 Policy 网络 + RunningStandardScaler，和训练时完全一致。"""
    obs_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    policy = Policy(observation_space=obs_space, action_space=act_space,
                    device=device, dv_max=0.002, clip_actions=False)
    prep = RunningStandardScaler(size=obs_dim, device=device)

    ckpt = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    prep.load_state_dict(ckpt["state_preprocessor"])
    policy.to(device).eval()
    prep.eval()
    return policy, prep


# ── 推理工具 ──────────────────────────────────────────────────────────────────

def infer_action(policy, prep, obs_np: np.ndarray,
                 buf: torch.Tensor, dv_max: float) -> np.ndarray:
    """原地写入 buf，推理并返回 numpy action（km/s）。"""
    buf[0].copy_(torch.from_numpy(obs_np.astype(np.float32)))
    with torch.no_grad():
        obs_norm = prep(buf)
        act, _, _ = policy.act({"states": obs_norm}, role="policy")
    return (act.squeeze().cpu().numpy() * dv_max).astype(np.float64)


# ── 环境构建 ──────────────────────────────────────────────────────────────────

def build_env(device):
    """创建 C++ 13星环境并设置任务分配。"""
    cfg_dict = asdict(env_cfg)
    cfg_dict["jd_epoch"] = 2460264.770833   # BJT 2023-11-16 22:30

    oge = _CppEnv(cfg_dict, num_evaders=7, num_pursuers=6,
                  intercept_distance=0.1)

    # 任务分配：
    #   HV(0): 无任务
    #   红护卫(1~6): 各自侦照对应蓝星
    #   蓝星(7~12): 各自任务对红HV
    assignments = [
        {"task_type": TASK_RECON, "target_idx": 0, "threat_idx": -1},  # HV
    ]
    for k in range(6):
        assignments.append({
            "task_type":  BLUE_TASK_TYPES[k],  # 镜像蓝星任务类型
            "target_idx": BLUE_IDXS[k],
            "threat_idx": -1,
        })
    for k, t in enumerate(BLUE_TASK_TYPES):
        assignments.append({
            "task_type":  t,
            "target_idx": HV_IDX,
            "threat_idx": -1,
        })
    oge.set_task_assignment(assignments)
    return oge


def build_states():
    """构建 13 颗星的初始状态。"""
    r_hv, v_hv = _coe2rv(**RED_HV_OE)
    esc_states  = [_coe2rv(**oe) for oe in RED_ESC_OE_LIST]
    blue_states = [_coe2rv(**oe) for oe in BLUE_REC_OE_LIST]

    states = {"blue_sat_0": _make_state(r_hv, v_hv, 0.0)}
    for k, (r, v) in enumerate(esc_states):
        states[f"blue_sat_{k+1}"] = _make_state(r, v, RED_ESC_DV_LIST[k])
    for k, (r, v) in enumerate(blue_states):
        states[f"red_sat_{k}"] = _make_state(r, v, BLUE_DV_INIT[k])
    return states


# ── 单 episode ────────────────────────────────────────────────────────────────

def run_episode(oge, red_policies, red_preps, blue_policies, blue_preps,
                red_bufs, blue_bufs, device, timestep: float, verbose: bool):
    """
    跑一个完整 episode，返回每颗红护卫的结果字典列表（6个）。

    每个字典：
      success      : bool   是否完成任务
      min_dist_km  : float  最近距离
      min_angle_deg: float  最小角度（OPERATE 任务为 None）
      max_zone_s   : float  最长连续在区时间
      final_dv_ratio: float 剩余燃料比
    """
    oge.reset_with_states(build_states())
    raw = np.asarray(oge.get_task_observations(), dtype=np.float32)

    last_dist   = [float(raw[i][6]) * 20.0 for i in range(1, 7)]
    acc_time    = [0.0] * 6
    in_zone     = [False] * 6
    min_dist    = [float("inf")] * 6
    min_angle   = [float("inf")] * 6
    max_zone_s  = [0.0] * 6
    cur_zone_s  = [0.0] * 6
    success     = [False] * 6
    final_dv    = [1.0] * 6

    combined = np.zeros((13, 3), dtype=np.float64)

    while not (oge.is_terminal() or oge.is_truncated()):
        combined[:] = 0.0

        for k in range(6):
            gi = k + 1
            combined[gi] = infer_action(
                red_policies[k], red_preps[k], raw[gi], red_bufs[k], 0.002)

        for k in range(6):
            gi = BLUE_IDXS[k]
            combined[gi] = infer_action(
                blue_policies[k], blue_preps[k], raw[gi], blue_bufs[k], 0.002)

        oge.act(combined)
        raw = np.asarray(oge.get_task_observations(), dtype=np.float32)

        for k in range(6):
            if success[k]:
                continue

            gi  = k + 1
            obs = raw[gi]
            crit = ESC_SUCCESS[k]

            dist_km  = float(obs[6]) * 20.0
            dv_ratio = float(obs[11])

            final_dv[k] = dv_ratio
            min_dist[k] = min(min_dist[k], dist_km)

            if crit["task"] == TASK_OPERATE:
                if dist_km <= crit["dist_km"]:
                    acc_time[k] += timestep
                    max_zone_s[k] = max(max_zone_s[k], acc_time[k])
                    if acc_time[k] >= crit["duration_s"]:
                        success[k] = True
                else:
                    acc_time[k] = 0.0
            else:
                angle_rad = float(obs[7]) * np.pi
                angle_deg = np.rad2deg(angle_rad)
                min_angle[k] = min(min_angle[k], angle_deg)

                in_z = (dist_km <= crit["dist_km"] and angle_deg <= crit["angle_deg"])
                if in_z:
                    cur_zone_s[k] += timestep
                    max_zone_s[k]  = max(max_zone_s[k], cur_zone_s[k])
                    if cur_zone_s[k] >= crit["duration_s"]:
                        success[k] = True
                else:
                    cur_zone_s[k] = 0.0

            last_dist[k] = dist_km

        if all(success):
            break

    return [dict(
        success        = success[k],
        min_dist_km    = min_dist[k],
        min_angle_deg  = min_angle[k] if ESC_SUCCESS[k]["task"] != TASK_OPERATE else None,
        max_zone_s     = max_zone_s[k],
        final_dv_ratio = final_dv[k],
    ) for k in range(6)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestep = float(env_cfg.timestep)

    print(f"\n{'='*65}")
    print(f"集群 6v6 对抗验证  |  device: {device}  |  episodes: {args.episodes}")
    print(f"{'='*65}")

    # ── 加载策略 ──────────────────────────────────────────────────────────────
    print("\n加载 ckpt...")
    red_policies, red_preps = [], []
    for i, path in enumerate(RED_CKPT_PATHS):
        exists = os.path.exists(path)
        pol, prep = load_policy(path, 17, device)
        red_policies.append(pol)
        red_preps.append(prep)
        status = "✓" if exists else "⚠ (placeholder)"
        print(f"  红护卫{i+1}: {status}")

    blue_policies, blue_preps = [], []
    for i, path in enumerate(BLUE_CKPT_PATHS):
        pol, prep = load_policy(path, 17, device)
        blue_policies.append(pol)
        blue_preps.append(prep)
        print(f"  蓝星  {i+1}: ✓")

    # 预分配推理 buffer
    red_bufs  = [torch.zeros(1, 17, dtype=torch.float32, device=device)
                 for _ in range(6)]
    blue_bufs = [torch.zeros(1, 17, dtype=torch.float32, device=device)
                 for _ in range(6)]

    # ── 构建环境 ──────────────────────────────────────────────────────────────
    oge = build_env(device)

    # ── 跑 episodes ───────────────────────────────────────────────────────────
    # all_results[ep][k] = 第 ep 个 episode 第 k 颗红护卫的结果
    all_results = []

    for ep in range(args.episodes):
        ep_results = run_episode(
            oge, red_policies, red_preps,
            blue_policies, blue_preps,
            red_bufs, blue_bufs, device, timestep,
            verbose=args.verbose,
        )
        all_results.append(ep_results)

        if args.verbose:
            successes = [r["success"] for r in ep_results]
            n_ok = sum(successes)
            flags = " ".join("✓" if s else "✗" for s in successes)
            print(f"  ep {ep+1:3d}: [{flags}]  {n_ok}/6 成功")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    BLUE_TASK_NAMES = ["打击", "打击", "干扰", "侦照", "侦照", "操控"]
    ESC_TASK_NAMES  = ["打击", "打击", "干扰", "侦照", "侦照", "操控"]
    print(f"\n{'─'*70}")
    print(f"{'护卫':>4}  {'对应蓝星':>8}  {'成功率':>7}  "
          f"{'最近距离均值':>12}  {'最小角度均值':>12}  {'最长在区均值':>12}")
    print(f"{'─'*70}")

    for k in range(6):
        ep_k     = [all_results[ep][k] for ep in range(args.episodes)]
        n_ok     = sum(r["success"]        for r in ep_k)
        rate     = n_ok / args.episodes * 100
        avg_dist = np.mean([r["min_dist_km"]   for r in ep_k])
        avg_zone = np.mean([r["max_zone_s"]    for r in ep_k])
        crit     = ESC_SUCCESS[k]

        if crit["task"] != TASK_OPERATE:
            avg_ang = np.mean([r["min_angle_deg"] for r in ep_k])
            angle_str = f"{avg_ang:>10.1f}°"
        else:
            angle_str = f"{'—':>11}"

        print(f"  红{k+1}  蓝{k+1}({ESC_TASK_NAMES[k]:>2})  "
              f"{n_ok}/{args.episodes}={rate:5.1f}%  "
              f"{avg_dist:>10.2f}km  "
              f"{angle_str}  "
              f"{avg_zone:>10.0f}s")

    # 全局成功率（所有护卫全部成功才算全局成功）
    global_ok = sum(
        all(all_results[ep][k]["success"] for k in range(6))
        for ep in range(args.episodes)
    )
    print(f"{'─'*70}")
    print(f"全局成功（6/6 全拦截）：{global_ok}/{args.episodes} = "
          f"{global_ok/args.episodes*100:.1f}%\n")


if __name__ == "__main__":
    main()
