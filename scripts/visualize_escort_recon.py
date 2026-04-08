"""
护卫侦照场景轨迹可视化
=======================
场景：蓝色侦照星 → 追逐红色高价值星（Blue chases Red HV）
      红色护卫星 → 对蓝色侦照星进行反侦照（Red Escort counter-reconnoiters Blue）

输出：
  visualizations/blue_recon_redHV__escort_counterrecon/
    ├── trajectory.gif   — 动画（全程轨迹）
    └── summary.png      — 静态汇总图（4 子图）
"""

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import dataclasses
import numpy as np
import torch
import gymnasium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from skrl.resources.preprocessors.torch import RunningStandardScaler

from modules.env_wrapper_escort_recon import (
    EscortReconWrapper, _blk, _other_k, _base, _solar_pos_j2000,
)
from scripts.train_blue_recon_phase1 import Policy
from configs.escort_recon_cfg import (
    env_cfg, JD_EPOCH_ESCORT_RECON, ESCORT_THREAT_DIST_KM,
)

# ── 全局样式（与 visualize_fixed_init.py 对齐） ───────────────────────────────
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams.update({
    "font.size":        12,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})

# ── 颜色常量 ──────────────────────────────────────────────────────────────────
C_ESC    = "darkorange"     # 红护卫
C_BLUE   = "steelblue"      # 蓝侦照星
C_HV     = "red"            # 红高价值星
C_ZONE   = "green"          # 侦照区
C_INZONE = "limegreen"      # 入区高亮
C_SUN    = "goldenrod"      # 太阳

# ── 初始六根数 ────────────────────────────────────────────────────────────────
_OE = ["a", "e", "i", "raan", "w", "M"]
RED_HV_OE  = dict(zip(_OE, [42169.502913, 0.0, 0.002287, 1.592853, 0.0, 0.419833]))
RED_ESC_OE = dict(zip(_OE, [42169.502913, 0.0, 0.002287, 1.592853, 0.0, 0.421133]))
BLUE_OE    = dict(zip(_OE, [42169.502913, 0.0, 0.002287, 1.592829, 0.0, 0.424435]))

# ── 路径 ─────────────────────────────────────────────────────────────────────
BLUE_CKPT    = os.path.join(_ROOT, "runs/April_7_blue_recon_phase1/"
                             "blue_recon_phase1/checkpoints/best_agent.pt")
RED_ESC_CKPT = os.path.join(_ROOT, "runs/Apr_7_red_escort_phase2/"
                             "Apr_7_red_escort_phase2/checkpoints/best_agent.pt")
OUT_DIR = os.path.join(_ROOT, "visualizations",
                       "blue_recon_redHV__escort_counterrecon")

# ── 任务常数 ──────────────────────────────────────────────────────────────────
ESC_DV_INIT = 0.020
ESC_DV_MAX  = 0.002
TIMESTEP    = 200.0
RECON_DIST  = 20.0
RECON_ANG   = 60.0
RECON_DUR   = 200.0

N    = 3
BASE = _base(N)   # = 20

_ESC_K_BLUE = _other_k(1, 2, N)
_ESC_K_HV   = _other_k(1, 0, N)
_BLU_K_HV   = _other_k(2, 0, N)

IDX_ESC_DIST_BLUE = _blk(_ESC_K_BLUE) + 6
IDX_ESC_DIST_HV   = _blk(_ESC_K_HV)   + 6
IDX_BLU_DIST_HV   = _blk(_BLU_K_HV)   + 6


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _solar_cone_xy(center_xy, sun_dir_xy, half_deg, radius, n=80):
    """返回扇形多边形 (xs, ys)，以 center_xy 为顶点，朝 sun_dir_xy，半角 half_deg。"""
    cx, cy = center_xy
    base_ang = np.arctan2(sun_dir_xy[1], sun_dir_xy[0])
    half_rad = np.deg2rad(half_deg)
    thetas = np.linspace(base_ang - half_rad, base_ang + half_rad, n)
    xs = np.concatenate([[cx], cx + radius * np.cos(thetas), [cx]])
    ys = np.concatenate([[cy], cy + radius * np.sin(thetas), [cy]])
    return xs, ys


def load_policy(ckpt_path, obs_dim, act_dim, device):
    obs_sp = gymnasium.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
    act_sp = gymnasium.spaces.Box(-np.inf, np.inf, (act_dim,), dtype=np.float32)
    pol = Policy(obs_sp, act_sp, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pol.load_state_dict(ckpt["policy"])
    pol.to(device).eval()
    prep = RunningStandardScaler(size=obs_dim, device=device)
    if "state_preprocessor" in ckpt:
        prep.load_state_dict(ckpt["state_preprocessor"])
    prep.eval()
    return pol, prep


# ── Episode 录制 ──────────────────────────────────────────────────────────────

def run_episode(device):
    blue_pol, blue_prep = load_policy(BLUE_CKPT,    17, 3, device)
    esc_pol,  esc_prep  = load_policy(RED_ESC_CKPT, 17, 3, device)

    cfg2 = dataclasses.replace(env_cfg,
                                dv_init_blue=ESC_DV_INIT,
                                dv_max_per_step_blue=ESC_DV_MAX)
    env = EscortReconWrapper(
        env_cfg=cfg2, jd_epoch=JD_EPOCH_ESCORT_RECON,
        red_hv_oe=RED_HV_OE, red_esc_oe=RED_ESC_OE,
        blue_dist_range=(190.0, 195.0), blue_sun_range=(43.0, 47.0),
        train_blue=False, blue_policy=blue_pol, blue_preprocessor=blue_prep,
        escort_intercept_dist=ESCORT_THREAT_DIST_KM,
        esc_dv_init=ESC_DV_INIT, blue_oe=BLUE_OE, seed=42,
    )

    traj = {k: [] for k in [
        "r_hv", "r_esc", "r_blue",
        "dist_eb", "dist_bh", "dist_eh",
        "ang_esc", "ang_blue",
        "dv_esc", "dv_blue",
        "acc_esc", "acc_blue",
        "zone_esc", "zone_blue",
    ]}

    def snap(raw):
        traj["r_hv"].append(raw[0][0:3].copy())
        traj["r_esc"].append(raw[1][0:3].copy())
        traj["r_blue"].append(raw[2][0:3].copy())
        d_eb = float(raw[1][IDX_ESC_DIST_BLUE] * 20.0)
        d_bh = float(raw[2][IDX_BLU_DIST_HV]  * 20.0)
        d_eh = float(raw[1][IDX_ESC_DIST_HV]   * 20.0)
        ae   = float(np.rad2deg(raw[1][BASE + 0]))
        ab   = float(np.rad2deg(raw[2][BASE + 0]))
        traj["dist_eb"].append(d_eb);  traj["dist_bh"].append(d_bh)
        traj["dist_eh"].append(d_eh)
        traj["ang_esc"].append(ae);    traj["ang_blue"].append(ab)
        traj["dv_esc"].append(float(raw[1][BASE + 1]))
        traj["dv_blue"].append(float(raw[2][BASE + 1]))
        traj["acc_esc"].append(float(env._recon_acc))
        traj["acc_blue"].append(float(env._blue_recon_acc))
        traj["zone_esc"].append(d_eb <= RECON_DIST and ae <= RECON_ANG)
        traj["zone_blue"].append(d_bh <= RECON_DIST and ab <= RECON_ANG)

    obs, _ = env.reset()
    snap(env._last_raw)

    done = truncated = False
    while not done and not truncated:
        with torch.no_grad():
            action, _, _ = esc_pol.act({"states": esc_prep(obs)}, role="policy")
        obs, reward, term, trunc, _ = env.step(action)
        done, truncated = bool(term.item()), bool(trunc.item())
        snap(env._last_raw)

    if done:
        if env._recon_acc >= RECON_DUR:
            outcome = "SUCCESS — Red Escort reconned Blue"
        elif env._blue_recon_acc >= RECON_DUR:
            outcome = "FAILURE — Blue reconned Red HV first"
        else:
            outcome = "FAILURE — Escort fuel depleted"
    else:
        outcome = "TIMEOUT — 20 h elapsed"

    for k in traj:
        traj[k] = np.array(traj[k])
    times = np.arange(len(traj["dist_eb"])) * TIMESTEP
    return traj, times, outcome


# ── GIF 生成 ──────────────────────────────────────────────────────────────────

def create_gif(traj, times, outcome, out_path):
    n_frames = len(times)

    r_hv     = traj["r_hv"]
    rel_esc  = traj["r_esc"]  - r_hv
    rel_blue = traj["r_blue"] - r_hv

    all_xy = np.vstack([rel_esc[:, :2], rel_blue[:, :2]])
    pad    = 30.0
    span   = max(np.abs(all_xy).max() + pad, 60.0)
    lim    = (-span, span)

    def _sd(frame):
        jd = JD_EPOCH_ESCORT_RECON + times[frame] / 86400.0
        ps = _solar_pos_j2000(jd)
        d  = (ps - r_hv[frame])[:2]
        n  = np.linalg.norm(d)
        return d / n if n > 1e-6 else np.array([1.0, 0.0])

    # ── 图形布局（白底，大尺寸，高清） ───────────────────────────────────
    fig = plt.figure(figsize=(22, 10), facecolor="white")
    gs  = fig.add_gridspec(3, 2, width_ratios=[1.2, 1],
                           left=0.06, right=0.97, top=0.87, bottom=0.08,
                           hspace=0.55, wspace=0.35)
    ax_l  = fig.add_subplot(gs[:, 0])
    ax_r1 = fig.add_subplot(gs[0, 1])
    ax_r2 = fig.add_subplot(gs[1, 1])
    ax_r3 = fig.add_subplot(gs[2, 1])

    t_max = float(times[-1])

    # 右侧：固定曲线（全程） + 游标
    for ax, d1, d2, ylabel, thr, l1, l2 in [
        (ax_r1, traj["dist_eb"],  traj["dist_bh"],
         "Distance (km)", RECON_DIST,
         "Escort→Blue", "Blue→HV"),
        (ax_r2, traj["ang_esc"],  traj["ang_blue"],
         "Solar Angle (°)", RECON_ANG,
         "Escort solar∠ (Blue vtx)", "Blue solar∠ (HV vtx)"),
        (ax_r3, traj["acc_esc"],  traj["acc_blue"],
         "Recon Acc. (s)", RECON_DUR,
         "Escort recon acc.", "Blue recon acc."),
    ]:
        ax.plot(times, d1, color=C_ESC,  lw=2.0, label=l1)
        ax.plot(times, d2, color=C_BLUE, lw=2.0, label=l2)
        ax.axhline(thr, color=C_ZONE, ls="--", lw=1.5, alpha=0.85,
                   label=f"Threshold {thr}")
        ax.fill_between(times, 0, thr, alpha=0.06, color=C_ZONE)
        ax.set_xlim(0, t_max * 1.04)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.35, color="gray", ls="--")
        ax.set_facecolor("white")

    ax_r1.set_xticklabels([])
    ax_r2.set_xticklabels([])
    ax_r3.set_xlabel("Sim Time (s)")

    # 游标线（黑色）
    vl1 = ax_r1.axvline(0, color="black", lw=1.2, ls=":", alpha=0.75)
    vl2 = ax_r2.axvline(0, color="black", lw=1.2, ls=":", alpha=0.75)
    vl3 = ax_r3.axvline(0, color="black", lw=1.2, ls=":", alpha=0.75)

    def update(frame):
        ax_l.clear()
        ax_l.set_facecolor("white")

        t_now = float(times[frame])
        sd    = _sd(frame)

        # ── 太阳光照锥（HV 顶点，金色，半角 60°） ─────────────────────
        xs, ys = _solar_cone_xy((0, 0), sd, 60.0, span * 0.92)
        ax_l.fill(xs, ys, color=C_SUN, alpha=0.12, zorder=0)
        ax_l.plot(xs[1:-1], ys[1:-1], color=C_SUN, lw=1.0, alpha=0.50, zorder=0)
        arr = span * 0.28
        ax_l.annotate("",
                      xy=(sd[0] * arr, sd[1] * arr), xytext=(0.0, 0.0),
                      arrowprops=dict(arrowstyle="->", color=C_SUN, lw=2.0))
        ax_l.text(sd[0]*arr*1.15, sd[1]*arr*1.15,
                  "☀ Sun", color=C_SUN, fontsize=11,
                  ha="center", va="center", fontweight="bold")

        # ── Blue 侦照成功区（圆心=HV，半径20km，绿色虚线） ─────────────
        ax_l.add_patch(mpatches.Circle((0, 0), RECON_DIST,
                        color=C_ZONE, alpha=0.12, fill=True, zorder=1))
        ax_l.add_patch(mpatches.Circle((0, 0), RECON_DIST,
                        color=C_ZONE, fill=False, lw=1.8, ls="--",
                        alpha=0.75, zorder=2))

        # ── Escort 侦照成功区（圆心=Blue当前位置，半径20km，橙色点线） ──
        bx, by = float(rel_blue[frame, 0]), float(rel_blue[frame, 1])
        ax_l.add_patch(mpatches.Circle((bx, by), RECON_DIST,
                        color=C_ESC, alpha=0.12, fill=True, zorder=1))
        ax_l.add_patch(mpatches.Circle((bx, by), RECON_DIST,
                        color=C_ESC, fill=False, lw=1.8, ls=":",
                        alpha=0.75, zorder=2))

        # ── 历史轨迹（渐变透明度） ────────────────────────────────────
        if frame > 0:
            alphas = np.linspace(0.25, 0.85, frame + 1)
            for i in range(frame):
                a = alphas[i]
                ax_l.plot(rel_esc[i:i+2, 0],  rel_esc[i:i+2, 1],
                          color=C_ESC,  lw=1.8, alpha=a, zorder=3)
                ax_l.plot(rel_blue[i:i+2, 0], rel_blue[i:i+2, 1],
                          color=C_BLUE, lw=1.8, alpha=a, zorder=3)

        # ── 当前位置 ──────────────────────────────────────────────────
        ax_l.scatter(0, 0, c=C_HV, s=260, marker="*", zorder=7,
                     edgecolors="black", lw=1.0)
        bc = C_INZONE if traj["zone_blue"][frame] else C_BLUE
        ax_l.scatter(bx, by, c=bc, s=180, marker="s",
                     edgecolors="black", lw=1.0, zorder=7)
        ex, ey = float(rel_esc[frame, 0]), float(rel_esc[frame, 1])
        ec = C_INZONE if traj["zone_esc"][frame] else "red"
        ax_l.scatter(ex, ey, c=ec, s=180, marker="^",
                     edgecolors="black", lw=1.0, zorder=7)

        # 连线
        ax_l.plot([0, bx], [0, by],
                  color=C_BLUE, lw=1.0, ls=":", alpha=0.50, zorder=4)
        ax_l.plot([ex, bx], [ey, by],
                  color=C_ESC,  lw=1.0, ls=":", alpha=0.50, zorder=4)

        ax_l.set_xlim(lim); ax_l.set_ylim(lim)
        ax_l.set_aspect("equal")
        ax_l.set_xlabel("ΔX (km) — relative to Red HV", fontsize=12)
        ax_l.set_ylabel("ΔY (km) — relative to Red HV", fontsize=12)
        ax_l.grid(True, alpha=0.35, color="gray", ls="--")

        # 图例
        leg = [
            Line2D([0],[0], marker="*", color="none", markerfacecolor=C_HV,
                   markeredgecolor="black", markersize=13, label="Red HV (center)"),
            Line2D([0],[0], marker="^", color="none", markerfacecolor="red",
                   markeredgecolor="black", markersize=11, label="Red Escort"),
            Line2D([0],[0], marker="s", color="none", markerfacecolor=C_BLUE,
                   markeredgecolor="black", markersize=11, label="Blue Recon"),
            mpatches.Patch(fc=C_ZONE, alpha=0.60,
                           label=f"Blue recon zone (HV, 20 km)"),
            mpatches.Patch(fc=C_ESC,  alpha=0.50,
                           label=f"Escort recon zone (Blue, 20 km)"),
            mpatches.Patch(fc=C_SUN,  alpha=0.50,
                           label="Sun illum. cone (60°)"),
            Line2D([0],[0], marker="o", color="none", markerfacecolor=C_INZONE,
                   markeredgecolor="black", markersize=10, label="In zone ✓"),
        ]
        ax_l.legend(handles=leg, loc="lower left",
                    framealpha=0.90, fontsize=10)

        # 标题（含状态）
        es = "IN ZONE ✓" if traj["zone_esc"][frame]  else f"∠{traj['ang_esc'][frame]:.1f}°"
        bs = "IN ZONE ✓" if traj["zone_blue"][frame] else f"∠{traj['ang_blue'][frame]:.1f}°"
        ax_l.set_title(
            f"Step {frame:>2d}/{n_frames-1}  ·  t = {t_now/3600:.3f} h\n"
            f"Escort→Blue {traj['dist_eb'][frame]:.1f} km  [{es}]  "
            f"Acc {traj['acc_esc'][frame]:.0f}/{RECON_DUR:.0f} s    "
            f"Blue→HV {traj['dist_bh'][frame]:.1f} km  [{bs}]  "
            f"Acc {traj['acc_blue'][frame]:.0f}/{RECON_DUR:.0f} s",
            fontsize=15, fontweight="bold", color="red", pad=8,
        )

        # 游标更新
        for vl in (vl1, vl2, vl3):
            vl.set_xdata([t_now, t_now])

    fig.suptitle(
        "Blue Recon  →  Red HV   |   Red Escort  Counter-Reconnoiters  Blue\n"
        f"Fixed Init · BJT 2027-09-01 20:00   |   {outcome}",
        fontsize=14, fontweight="bold",
    )

    anim = FuncAnimation(fig, update, frames=n_frames, interval=400)
    anim.save(out_path, writer=PillowWriter(fps=2.5))
    plt.close(fig)
    print(f"GIF saved: {out_path}")


# ── 静态汇总 PNG ──────────────────────────────────────────────────────────────

def create_summary(traj, times, outcome, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Escort-Recon Scenario — Trajectory Summary\n"
        f"Fixed Init · BJT 2027-09-01 20:00 · {outcome}",
        fontsize=14, fontweight="bold",
    )

    r_hv     = traj["r_hv"]
    rel_esc  = traj["r_esc"]  - r_hv
    rel_blue = traj["r_blue"] - r_hv
    n        = len(times)
    alphas   = np.linspace(0.25, 1.0, n)

    # ── 子图 1：XY 相对轨迹（以 HV 为原点） ──────────────────────────
    ax = axes[0, 0]
    ps   = _solar_pos_j2000(JD_EPOCH_ESCORT_RECON)
    d    = (ps - r_hv[0])[:2]
    sd   = d / np.linalg.norm(d)
    span = max(np.abs(np.vstack([rel_esc[:, :2], rel_blue[:, :2]])).max() * 1.25,
               60.0)
    xs, ys = _solar_cone_xy((0, 0), sd, 60.0, span * 0.85)
    ax.fill(xs, ys, color=C_SUN, alpha=0.13, zorder=0)
    ax.plot(xs[1:-1], ys[1:-1], color=C_SUN, lw=0.8, alpha=0.45, zorder=0)

    for i in range(n - 1):
        a = alphas[i]
        ax.plot(rel_esc[i:i+2, 0],  rel_esc[i:i+2, 1],
                color=C_ESC,  lw=2.0, alpha=a)
        ax.plot(rel_blue[i:i+2, 0], rel_blue[i:i+2, 1],
                color=C_BLUE, lw=2.0, alpha=a)

    ax.scatter(0, 0,
               c=C_HV, s=260, marker="*", zorder=6,
               edgecolors="black", lw=1.0, label="Red HV ★")
    ax.scatter(rel_esc[0,0],  rel_esc[0,1],
               c="red", s=80, marker="^", alpha=0.45, zorder=5,
               edgecolors="black", lw=0.8)
    ax.scatter(rel_esc[-1,0], rel_esc[-1,1],
               c="red", s=160, marker="^", zorder=6,
               edgecolors="black", lw=1.0, label="Red Escort ▲")
    ax.scatter(rel_blue[0,0],  rel_blue[0,1],
               c=C_BLUE, s=80, marker="s", alpha=0.45, zorder=5,
               edgecolors="black", lw=0.8)
    ax.scatter(rel_blue[-1,0], rel_blue[-1,1],
               c=C_BLUE, s=160, marker="s", zorder=6,
               edgecolors="black", lw=1.0, label="Blue Recon ■")

    ax.add_patch(mpatches.Circle((0, 0), RECON_DIST, fill=False,
                                  color=C_ZONE, lw=1.8, ls="--",
                                  label=f"Recon zone {RECON_DIST:.0f} km"))
    arr = span * 0.50
    ax.annotate("", xy=(sd[0]*arr, sd[1]*arr), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_SUN, lw=2.0))
    ax.text(sd[0]*arr*1.12, sd[1]*arr*1.12,
            "☀ Sun", color=C_SUN, fontsize=11, ha="center", fontweight="bold")

    ax.set_xlim(-span, span); ax.set_ylim(-span, span)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.35, color="gray", ls="--")
    ax.set_xlabel("ΔX (km) relative to Red HV")
    ax.set_ylabel("ΔY (km) relative to Red HV")
    ax.set_title("XY Relative Trajectory (HV center)", fontweight="bold")
    ax.legend(fontsize=10)

    # ── 子图 2：距离时间序列 ──────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(times, traj["dist_eb"], color=C_ESC,  lw=2.5, label="Escort → Blue")
    ax.plot(times, traj["dist_bh"], color=C_BLUE, lw=2.5, label="Blue → HV")
    ax.axhline(RECON_DIST, color=C_ZONE, ls="--", lw=2.0,
               label=f"Recon dist {RECON_DIST:.0f} km")
    ax.fill_between(times, 0, RECON_DIST, alpha=0.08, color=C_ZONE)
    for i in range(n):
        if traj["zone_esc"][i]:
            ax.scatter(times[i], traj["dist_eb"][i],
                       c=C_INZONE, s=45, zorder=5, edgecolors="black", lw=0.5)
        if traj["zone_blue"][i]:
            ax.scatter(times[i], traj["dist_bh"][i],
                       c=C_INZONE, s=45, zorder=5, edgecolors="black", lw=0.5)
    ax.set_xlabel("Sim Time (s)"); ax.set_ylabel("Distance (km)")
    ax.set_title("Distance vs. Time", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.35, color="gray", ls="--")

    # ── 子图 3：太阳角时间序列 ────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(times, traj["ang_esc"],  color=C_ESC,  lw=2.5,
            label="Escort solar∠ (Blue vtx)")
    ax.plot(times, traj["ang_blue"], color=C_BLUE, lw=2.5,
            label="Blue solar∠ (HV vtx)")
    ax.axhline(RECON_ANG, color=C_ZONE, ls="--", lw=2.0,
               label=f"Threshold {RECON_ANG:.0f}°")
    ax.fill_between(times, 0, RECON_ANG, alpha=0.08, color=C_ZONE)
    ax.set_xlabel("Sim Time (s)"); ax.set_ylabel("Solar Angle (°)")
    ax.set_title("Solar Angle vs. Time", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.35, color="gray", ls="--")

    # ── 子图 4：侦照进度 + 剩余燃料（双纵轴） ────────────────────────
    ax  = axes[1, 1]
    ax.plot(times, traj["acc_esc"],  color=C_ESC,  lw=2.5,
            label="Escort recon acc. (s)")
    ax.plot(times, traj["acc_blue"], color=C_BLUE, lw=2.5,
            label="Blue recon acc. (s)")
    ax.axhline(RECON_DUR, color=C_ZONE, ls="--", lw=2.0,
               label=f"Success {RECON_DUR:.0f} s")
    ax.fill_between(times, 0, RECON_DUR, alpha=0.08, color=C_ZONE)

    ax2 = ax.twinx()
    ax2.plot(times, traj["dv_esc"]  * 1000, color=C_ESC,
             lw=1.5, ls=":", alpha=0.60, label="Escort ΔV (m/s)")
    ax2.plot(times, traj["dv_blue"] * 1000, color=C_BLUE,
             lw=1.5, ls=":", alpha=0.60, label="Blue ΔV (m/s)")
    ax2.set_ylabel("Remaining ΔV (m/s)")

    ax.set_xlabel("Sim Time (s)"); ax.set_ylabel("Recon Accumulator (s)")
    ax.set_title("Recon Progress & Fuel", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.35, color="gray", ls="--")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary PNG saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print("Running episode...")

    traj, times, outcome = run_episode(device)
    print(f"Steps  : {len(times)-1}  ({times[-1]/3600:.2f} h)  |  {outcome}")

    png_path = os.path.join(OUT_DIR, "summary.png")
    gif_path = os.path.join(OUT_DIR, "trajectory.gif")

    print("Creating summary.png ...")
    create_summary(traj, times, outcome, png_path)

    print("Creating trajectory.gif ...")
    create_gif(traj, times, outcome, gif_path)

    print(f"\nAll outputs saved to:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
