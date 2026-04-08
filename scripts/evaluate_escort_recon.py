#!/usr/bin/env python3
"""
验证脚本：固定初始六根数下，红色护卫星能否成功侦照蓝色星
=============================================================
红色高价值星 (Red HV)  ── 无机动
红色护卫星   (Red Esc) ── 加载 Phase 2 checkpoint
蓝色侦照星   (Blue)    ── 加载 Phase 1 checkpoint

成功条件（护卫视角）：
  dist(RedEsc, Blue) ≤ 20 km  AND  solar_angle(Blue 为顶点) ≤ 60°
  累计持续 ≥ 200 s

失败条件：
  1. 蓝色侦照星抢先对红高侦照成功
  2. 护卫星燃料耗尽
  3. 任务时间（20 h）耗尽
"""

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import dataclasses
import numpy as np
import torch
import gymnasium
from skrl.resources.preprocessors.torch import RunningStandardScaler

from modules.env_wrapper_escort_recon import (
    EscortReconWrapper, _blk, _other_k, _base,
)
from scripts.train_blue_recon_phase1 import Policy
from configs.escort_recon_cfg import (
    env_cfg, JD_EPOCH_ESCORT_RECON,
    ESCORT_THREAT_DIST_KM,
)

# ── 初始六根数（a: km，角度: rad） ────────────────────────────────────────────
_OE_KEYS = ["a", "e", "i", "raan", "w", "M"]

RED_HV_OE  = dict(zip(_OE_KEYS, [42169.502913, 0.0, 0.002287, 1.592853, 0.0, 0.419833]))
RED_ESC_OE = dict(zip(_OE_KEYS, [42169.502913, 0.0, 0.002287, 1.592853, 0.0, 0.421133]))
BLUE_OE    = dict(zip(_OE_KEYS, [42169.502913, 0.0, 0.002287, 1.592829, 0.0, 0.424435]))

# ── Checkpoint 路径 ───────────────────────────────────────────────────────────
BLUE_CKPT    = os.path.join(
    _ROOT, "runs/April_7_blue_recon_phase1/blue_recon_phase1/checkpoints/best_agent.pt")
RED_ESC_CKPT = os.path.join(
    _ROOT, "runs/Apr_7_red_escort_phase2/Apr_7_red_escort_phase2/checkpoints/best_agent.pt")

# ── 任务参数 ──────────────────────────────────────────────────────────────────
ESC_DV_INIT = 0.020   # km/s（20 m/s）
ESC_DV_MAX  = 0.002   # km/s（2 m/s per step）
TIMESTEP    = 200.0   # s/step
PRINT_EVERY = 10      # 每 N 步打印一次（终止时始终打印）


# ── 策略加载 ──────────────────────────────────────────────────────────────────

def load_policy(ckpt_path: str, obs_dim: int, act_dim: int, device):
    obs_sp = gymnasium.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
    act_sp = gymnasium.spaces.Box(-np.inf, np.inf, shape=(act_dim,), dtype=np.float32)

    policy = Policy(obs_sp, act_sp, device)
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    policy.to(device).eval()

    prep = RunningStandardScaler(size=obs_dim, device=device)
    if "state_preprocessor" in ckpt:
        prep.load_state_dict(ckpt["state_preprocessor"])
    prep.eval()
    return policy, prep


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"Blue  ckpt   : {BLUE_CKPT}")
    print(f"RedEsc ckpt  : {RED_ESC_CKPT}")
    print(f"JD epoch     : {JD_EPOCH_ESCORT_RECON}  (BJT 2027-09-01 20:00)")
    print()

    # 加载策略
    blue_policy, blue_prep       = load_policy(BLUE_CKPT,    17, 3, device)
    red_esc_policy, red_esc_prep = load_policy(RED_ESC_CKPT, 17, 3, device)
    print("Both policies loaded.\n")

    # 构建环境（train_blue=False → 红护卫为 RL agent）
    env_cfg_p2 = dataclasses.replace(
        env_cfg,
        dv_init_blue         = ESC_DV_INIT,
        dv_max_per_step_blue = ESC_DV_MAX,
    )

    env = EscortReconWrapper(
        env_cfg=env_cfg_p2,
        jd_epoch=JD_EPOCH_ESCORT_RECON,
        red_hv_oe=RED_HV_OE,
        red_esc_oe=RED_ESC_OE,
        blue_dist_range=(190.0, 195.0),
        blue_sun_range=(43.0, 47.0),
        train_blue=False,
        blue_policy=blue_policy,
        blue_preprocessor=blue_prep,
        escort_intercept_dist=ESCORT_THREAT_DIST_KM,
        esc_dv_init=ESC_DV_INIT,
        blue_oe=BLUE_OE,
        seed=42,
    )

    obs, _ = env.reset()
    done = truncated = False
    step         = 0
    total_reward = 0.0
    last_reward  = 0.0

    hdr = (f"{'Step':>5} {'SimTime(s)':>10} "
           f"{'Esc→Blue(km)':>13} {'SolAng(°)':>10} "
           f"{'EscRecon(s)':>12} {'BlueRecon(s)':>13} "
           f"{'EscDV(km/s)':>12} {'BlueDV(km/s)':>13} "
           f"{'Blue→HV(km)':>12}")
    print(hdr)
    print("-" * len(hdr))

    N    = env.N
    base = _base(N)

    while not done and not truncated:
        with torch.no_grad():
            obs_p  = red_esc_prep(obs)
            action, _, _ = red_esc_policy.act({"states": obs_p}, role="policy")

        obs, reward, term, trunc, _ = env.step(action)
        done         = bool(term.item())
        truncated    = bool(trunc.item())
        last_reward  = reward.item()
        total_reward += last_reward
        step         += 1

        if step % PRINT_EVERY == 0 or done or truncated:
            raw = env._last_raw

            # Red Escort (gi=1) 到 Blue Recon (gi=2)
            k_esc_to_blue  = _other_k(1, 2, N)
            dist_esc_blue  = raw[1][_blk(k_esc_to_blue) + 6] * 20.0
            solar_deg      = np.rad2deg(env._esc_solar_angle(raw))
            dv_esc         = raw[1][base + 1]   # 护卫剩余燃料 km/s

            # Blue Recon (gi=2) 到 Red HV (gi=0)
            k_blue_to_hv   = _other_k(2, 0, N)
            dist_blue_hv   = raw[2][_blk(k_blue_to_hv) + 6] * 20.0
            dv_blue        = raw[2][base + 1]   # 蓝星剩余燃料 km/s

            t_s = step * TIMESTEP
            print(f"{step:>5} {t_s:>10.0f} "
                  f"{dist_esc_blue:>13.2f} {solar_deg:>10.2f} "
                  f"{env._recon_acc:>12.1f} {env._blue_recon_acc:>13.1f} "
                  f"{dv_esc:>12.5f} {dv_blue:>13.5f} "
                  f"{dist_blue_hv:>12.2f}")

    print("-" * len(hdr))
    print(f"\n总步数: {step}  |  仿真时间: {step * TIMESTEP:.0f} s "
          f"({step * TIMESTEP / 3600:.2f} h)  |  累计奖励: {total_reward:.2f}")

    # ── 判定结果 ───────────────────────────────────────────────────────────────
    print()
    if done:
        if env._recon_acc >= env.RECON_DURATION:
            print("=" * 50)
            print("  [SUCCESS] 红色护卫星成功侦照蓝色星！")
            print("=" * 50)
        elif env._blue_recon_acc >= env.RECON_DURATION:
            print("=" * 50)
            print("  [FAILURE] 蓝色侦照星抢先完成对红高价值星的侦照，护卫任务失败。")
            print("=" * 50)
        else:
            # 燃料耗尽（last_reward ≈ -40）
            print("=" * 50)
            print(f"  [FAILURE] 护卫星燃料耗尽（last_reward={last_reward:.1f}）。")
            print("=" * 50)
    elif truncated:
        print("=" * 50)
        print("  [TIMEOUT] 任务时间（20 h）耗尽，未完成侦照。")
        raw = env._last_raw
        k   = _other_k(1, 2, N)
        print(f"  护卫侦照累计: {env._recon_acc:.1f} s / {env.RECON_DURATION:.1f} s")
        print(f"  蓝星侦照累计: {env._blue_recon_acc:.1f} s / {env.RECON_DURATION:.1f} s")
        print("=" * 50)


if __name__ == "__main__":
    main()
