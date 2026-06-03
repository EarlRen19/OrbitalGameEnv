"""
集群 6v6 对抗可视化
==================
红方 6 颗护卫 vs 蓝方 6 颗星，各自执行任务。
护卫/蓝星标注编号，任务成功后隐藏对应的配对。

输出：
  visualizations/cluster_6v6/
    ├── trajectory.gif   — 动画
    └── summary.png      — 静态汇总
"""

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from dataclasses import asdict
from skrl.resources.preprocessors.torch import RunningStandardScaler

from configs.cluster_escort_cfg import (
    env_cfg, RED_HV_OE, RED_ESC_OE_LIST, BLUE_REC_OE_LIST,
)
from modules.env_wrapper_cluster_red import (
    _coe2rv, _make_state, BLUE_IDXS, HV_IDX,
    BLUE_TASK_TYPES, BLUE_DV_INIT, RED_ESC_DV_LIST,
    TASK_STRIKE, TASK_RECON, TASK_JAM, TASK_OPERATE,
)
from modules.networks import Policy

try:
    from oge_py._oge_py_ma import MultiAgentOGEEnv as _CppEnv
    from oge_py._oge_py import SatState
except ImportError:
    from _oge_py_ma import MultiAgentOGEEnv as _CppEnv
    from _oge_py import SatState

# ── 全局样式 ──────────────────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
})

# ── 路径配置 ──────────────────────────────────────────────────────────────────
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

OUT_DIR = os.path.join(_ROOT, "visualizations", "cluster_6v6")

# ── 成功条件（与 evaluate_cluster_6v6.py 一致） ──────────────────────────────
ESC_SUCCESS = [
    dict(task=TASK_STRIKE,  dist_km=20.0, angle_deg=90.0, duration_s=200.0),
    dict(task=TASK_STRIKE,  dist_km=20.0, angle_deg=90.0, duration_s=200.0),
    dict(task=TASK_JAM,     dist_km=20.0, angle_deg=5.0,  duration_s=600.0),
    dict(task=TASK_RECON,   dist_km=20.0, angle_deg=60.0, duration_s=200.0),
    dict(task=TASK_RECON,   dist_km=20.0, angle_deg=60.0, duration_s=200.0),
    dict(task=TASK_OPERATE, dist_km=2.0,  angle_deg=None, duration_s=200.0),
]

TIMESTEP = 200.0
C_RED_ESC = "red"
C_BLUE = "steelblue"
C_HV = "darkred"
C_SUCCESS = "limegreen"


# ── 策略加载 ──────────────────────────────────────────────────────────────────

def load_policy(ckpt_path, obs_dim, device):
    import gymnasium
    obs_space = gymnasium.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
    act_space = gymnasium.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
    policy = Policy(obs_space, act_space, device, dv_max=0.002, clip_actions=False)
    prep = RunningStandardScaler(size=obs_dim, device=device)

    ckpt = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    prep.load_state_dict(ckpt["state_preprocessor"])
    policy.to(device).eval()
    prep.eval()
    return policy, prep


def infer_action(policy, prep, obs_np, buf, dv_max):
    buf[0].copy_(torch.from_numpy(obs_np.astype(np.float32)))
    with torch.no_grad():
        obs_norm = prep(buf)
        act, _, _ = policy.act({"states": obs_norm}, role="policy")
    return (act.squeeze().cpu().numpy() * dv_max).astype(np.float64)


# ── 环境构建 ──────────────────────────────────────────────────────────────────

def build_env(device):
    cfg_dict = asdict(env_cfg)
    cfg_dict["jd_epoch"] = 2460264.770833
    oge = _CppEnv(cfg_dict, num_evaders=7, num_pursuers=6, intercept_distance=0.1)

    assignments = [
        {"task_type": TASK_RECON, "target_idx": 0, "threat_idx": -1},
    ]
    for k in range(6):
        assignments.append({
            "task_type": BLUE_TASK_TYPES[k],
            "target_idx": BLUE_IDXS[k],
            "threat_idx": -1,
        })
    for k, t in enumerate(BLUE_TASK_TYPES):
        assignments.append({
            "task_type": t,
            "target_idx": HV_IDX,
            "threat_idx": -1,
        })
    oge.set_task_assignment(assignments)
    return oge


def build_states():
    r_hv, v_hv = _coe2rv(**RED_HV_OE)
    esc_states = [_coe2rv(**oe) for oe in RED_ESC_OE_LIST]
    blue_states = [_coe2rv(**oe) for oe in BLUE_REC_OE_LIST]

    states = {"blue_sat_0": _make_state(r_hv, v_hv, 0.0)}
    for k, (r, v) in enumerate(esc_states):
        states[f"blue_sat_{k+1}"] = _make_state(r, v, RED_ESC_DV_LIST[k])
    for k, (r, v) in enumerate(blue_states):
        states[f"red_sat_{k}"] = _make_state(r, v, BLUE_DV_INIT[k])
    return states


# ── Episode 录制 ──────────────────────────────────────────────────────────────

def run_episode(device):
    print("Loading policies...")
    red_policies, red_preps = [], []
    for path in RED_CKPT_PATHS:
        pol, prep = load_policy(path, 17, device)
        red_policies.append(pol)
        red_preps.append(prep)

    blue_policies, blue_preps = [], []
    for path in BLUE_CKPT_PATHS:
        pol, prep = load_policy(path, 17, device)
        blue_policies.append(pol)
        blue_preps.append(prep)

    red_bufs = [torch.zeros(1, 17, dtype=torch.float32, device=device) for _ in range(6)]
    blue_bufs = [torch.zeros(1, 17, dtype=torch.float32, device=device) for _ in range(6)]

    oge = build_env(device)
    oge.reset_with_states(build_states())

    traj = {
        "r_hv": [],
        "r_esc": [[] for _ in range(6)],
        "r_blue": [[] for _ in range(6)],
        "success": [False] * 6,
        "success_step": [-1] * 6,
    }

    acc_time = [0.0] * 6
    cur_zone_s = [0.0] * 6

    combined = np.zeros((13, 3), dtype=np.float64)
    step = 0

    while not (oge.is_terminal() or oge.is_truncated()):
        raw = np.asarray(oge.get_task_observations(), dtype=np.float32)

        # 记录位置
        states = oge.get_sat_states()
        traj["r_hv"].append(states["blue_sat_0"].r_j2000.copy())
        for k in range(6):
            traj["r_esc"][k].append(states[f"blue_sat_{k+1}"].r_j2000.copy())
            traj["r_blue"][k].append(states[f"red_sat_{k}"].r_j2000.copy())

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

        # 判定成功
        for k in range(6):
            if traj["success"][k]:
                continue
            gi = k + 1
            obs = raw[gi]
            crit = ESC_SUCCESS[k]

            dist_km = float(obs[6]) * 20.0

            if crit["task"] == TASK_OPERATE:
                if dist_km <= crit["dist_km"]:
                    acc_time[k] += TIMESTEP
                    if acc_time[k] >= crit["duration_s"]:
                        traj["success"][k] = True
                        traj["success_step"][k] = step
                else:
                    acc_time[k] = 0.0
            else:
                angle_rad = float(obs[7]) * np.pi
                angle_deg = np.rad2deg(angle_rad)
                in_z = (dist_km <= crit["dist_km"] and angle_deg <= crit["angle_deg"])
                if in_z:
                    cur_zone_s[k] += TIMESTEP
                    if cur_zone_s[k] >= crit["duration_s"]:
                        traj["success"][k] = True
                        traj["success_step"][k] = step
                else:
                    cur_zone_s[k] = 0.0

        step += 1

    for k in ["r_hv"] + [f"r_esc_{i}" for i in range(6)] + [f"r_blue_{i}" for i in range(6)]:
        if k == "r_hv":
            traj[k] = np.array(traj[k])
        elif k.startswith("r_esc"):
            idx = int(k.split("_")[-1])
            traj[f"r_esc"][idx] = np.array(traj[f"r_esc"][idx])
        elif k.startswith("r_blue"):
            idx = int(k.split("_")[-1])
            traj[f"r_blue"][idx] = np.array(traj[f"r_blue"][idx])

    times = np.arange(step + 1) * TIMESTEP
    n_success = sum(traj["success"])
    outcome = f"{n_success}/6 Success"
    return traj, times, outcome


# ── GIF 生成 ──────────────────────────────────────────────────────────────────

def create_gif(traj, times, outcome, out_path):
    n_frames = len(times) - 1  # 最后一帧没有新数据，跳过
    r_hv = traj["r_hv"]

    fig, ax = plt.subplots(figsize=(14, 14), facecolor="white")

    # 固定坐标范围：200km（以 HV 为中心）
    span = 200.0
    lim = (-span, span)

    def update(frame):
        ax.clear()
        ax.set_facecolor("white")

        t_now = times[frame]
        center = r_hv[frame, :2]

        # HV
        ax.scatter(0, 0, c=C_HV, s=280, marker="*", zorder=8,
                   edgecolors="black", lw=1.2, label="Red HV")

        # 红护卫 + 蓝星
        for k in range(6):
            if traj["success"][k] and frame > traj["success_step"][k]:
                continue  # 成功后隐藏

            rel_esc = traj["r_esc"][k][frame, :2] - center
            rel_blue = traj["r_blue"][k][frame, :2] - center

            # 红护卫
            ax.scatter(rel_esc[0], rel_esc[1], c=C_RED_ESC, s=150, marker="^",
                       edgecolors="black", lw=1.0, zorder=7)
            ax.text(rel_esc[0], rel_esc[1] + 8, f"R{k+1}", fontsize=10,
                    ha="center", va="bottom", color=C_RED_ESC, fontweight="bold")

            # 蓝星
            ax.scatter(rel_blue[0], rel_blue[1], c=C_BLUE, s=150, marker="s",
                       edgecolors="black", lw=1.0, zorder=7)
            ax.text(rel_blue[0], rel_blue[1] - 8, f"B{k+1}", fontsize=10,
                    ha="center", va="top", color=C_BLUE, fontweight="bold")

            # 连线
            ax.plot([rel_esc[0], rel_blue[0]], [rel_esc[1], rel_blue[1]],
                    color="gray", lw=0.8, ls=":", alpha=0.4, zorder=4)

        # 历史轨迹
        if frame > 0:
            alphas = np.linspace(0.2, 0.7, frame + 1)
            for k in range(6):
                if traj["success"][k] and frame > traj["success_step"][k]:
                    fade_end = traj["success_step"][k]
                else:
                    fade_end = frame

                for i in range(fade_end):
                    a = alphas[i]
                    rel_esc_i = traj["r_esc"][k][i:i+2, :2] - r_hv[i:i+2, :2]
                    rel_blue_i = traj["r_blue"][k][i:i+2, :2] - r_hv[i:i+2, :2]
                    ax.plot(rel_esc_i[:, 0], rel_esc_i[:, 1],
                            color=C_RED_ESC, lw=1.2, alpha=a, zorder=3)
                    ax.plot(rel_blue_i[:, 0], rel_blue_i[:, 1],
                            color=C_BLUE, lw=1.2, alpha=a, zorder=3)

        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("ΔX (km) — relative to Red HV", fontsize=11)
        ax.set_ylabel("ΔY (km) — relative to Red HV", fontsize=11)
        ax.grid(True, alpha=0.35, color="gray", ls="--")

        success_str = ", ".join([f"R{k+1}" for k in range(6) if traj["success"][k]])
        ax.set_title(
            f"Cluster 6v6  |  Step {frame}/{n_frames-1}  |  t = {t_now/3600:.2f} h\n"
            f"Success: {success_str if success_str else 'None'}",
            fontsize=13, fontweight="bold", pad=10,
        )

        leg = [
            Line2D([0], [0], marker="*", color="none", markerfacecolor=C_HV,
                   markeredgecolor="black", markersize=14, label="Red HV"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor=C_RED_ESC,
                   markeredgecolor="black", markersize=11, label="Red Escort"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=C_BLUE,
                   markeredgecolor="black", markersize=11, label="Blue"),
        ]
        ax.legend(handles=leg, loc="upper right", framealpha=0.9, fontsize=10)

    fig.suptitle(f"Cluster 6v6 — {outcome}", fontsize=14, fontweight="bold")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=400)
    anim.save(out_path, writer=PillowWriter(fps=2.5))
    plt.close(fig)
    print(f"GIF saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    traj, times, outcome = run_episode(device)
    print(f"Steps: {len(times)-1} ({times[-1]/3600:.2f} h) | {outcome}")

    gif_path = os.path.join(OUT_DIR, "trajectory.gif")
    print("Creating trajectory.gif ...")
    create_gif(traj, times, outcome, gif_path)

    print(f"\nOutput saved to:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
