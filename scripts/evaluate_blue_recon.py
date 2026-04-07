"""验证蓝色侦照星 checkpoint 在护卫侦照场景下的表现。

固定初始六根数（a km，角度 rad）：
  Red HV : [42169.502913, 0.0, 0.002287, 1.592853, 0.0, 0.419833]
  Red Esc: [42169.502913, 0.0, 0.002287, 1.592853, 0.0, 0.421133]
  Blue   : [42169.502913, 0.0, 0.002287, 1.592829, 0.0, 0.424435]

用法：
  python scripts/evaluate_blue_recon.py --checkpoint runs/blue_recon_phase1/.../best_agent.pt
  python scripts/evaluate_blue_recon.py --checkpoint ... --episodes 50
"""

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import argparse
import numpy as np
import torch
from skrl.resources.preprocessors.torch import RunningStandardScaler

from modules.env_wrapper_escort_recon import EscortReconWrapper, _coe2rv, _ma2ta
from scripts.train_blue_recon_phase1 import Policy
from configs.escort_recon_cfg import (
    env_cfg, JD_EPOCH_ESCORT_RECON,
    RED_HV_OE, RED_ESC_OE,
    BLUE_DIST_MIN_KM, BLUE_DIST_MAX_KM,
    BLUE_SUN_ANGLE_MIN, BLUE_SUN_ANGLE_MAX,
    ESCORT_THREAT_DIST_KM,
)

try:
    from oge_py._oge_py import SatState
except ImportError:
    from _oge_py import SatState

# ── 固定初始六根数 ─────────────────────────────────────────────────────────────
DEFAULT_RED_HV  = [42169.502913, 0.0, 0.002287, 1.592853, 0.0, 0.419833]
DEFAULT_RED_ESC = [42169.502913, 0.0, 0.002287, 1.592853, 0.0, 0.421133]
DEFAULT_BLUE    = [42169.502913, 0.0, 0.002287, 1.592829, 0.0, 0.424435]


def _oe_to_satstate(oe_list, dv_remain):
    a, e, i, raan, w, M = oe_list
    r, v = _coe2rv(a, e, i, raan, w, M)
    s = SatState()
    s.r_j2000   = r.astype(np.float64)
    s.v_j2000   = v.astype(np.float64)
    s.dv_remain = float(dv_remain)
    s.is_alive  = True
    return s


def build_fixed_states(dv_blue):
    return {
        "blue_sat_0": _oe_to_satstate(DEFAULT_RED_HV,  0.0),
        "blue_sat_1": _oe_to_satstate(DEFAULT_RED_ESC, 0.0),
        "red_sat_0":  _oe_to_satstate(DEFAULT_BLUE,    dv_blue),
    }


def run_episode(env, policy, preprocessor, device, fixed_states=None):
    if fixed_states is not None:
        env._oge.reset_with_states(fixed_states)
        import numpy as np_
        raw = np_.asarray(env._oge.get_observations(), dtype=np_.float64)
        env._last_raw = raw
        env._recon_acc = 0.0
        env._in_zone   = False
        from modules.env_wrapper_escort_recon import _blk, _other_k
        env._last_dist = raw[2][_blk(_other_k(2, 0, 3)) + 6] * 20.0
        obs = env._blue_obs(raw)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        obs_t, _ = env.reset()
        obs_t = obs_t.to(device)

    done = False
    total_reward = 0.0
    steps = 0
    success = False
    time_in_zone = 0.0
    entry_count  = 0
    prev_in_zone = False

    traj = {"dist": [], "solar_deg": [], "dist_esc": [], "reward": [], "in_zone": []}

    while not done:
        with torch.no_grad():
            norm_obs = preprocessor(obs_t)
            action   = policy.act({"states": norm_obs}, role="policy")[0]

        obs_t, reward, terminated, truncated, _ = env.step(action)
        obs_t = obs_t.to(device)

        obs_np = obs_t.squeeze().cpu().numpy()
        dist_km      = obs_np[6] * 20.0
        solar_deg    = np.rad2deg(obs_np[7] * np.pi)
        dist_esc_km  = obs_np[13] * 20.0

        in_zone = dist_km <= 20.0 and solar_deg <= 60.0
        if in_zone:
            time_in_zone += env._timestep
            if not prev_in_zone:
                entry_count += 1
        if time_in_zone >= env.RECON_DURATION:
            success = True

        prev_in_zone = in_zone
        done = terminated.item() or truncated.item()

        traj["dist"].append(dist_km)
        traj["solar_deg"].append(solar_deg)
        traj["dist_esc"].append(dist_esc_km)
        traj["reward"].append(reward.item())
        traj["in_zone"].append(in_zone)

        total_reward += reward.item()
        steps += 1

    return {
        "total_reward": total_reward,
        "success":      success,
        "time_in_zone": time_in_zone,
        "entry_count":  entry_count,
        "final_dist":   dist_km,
        "final_solar":  solar_deg,
        "final_dist_esc": dist_esc_km,
        "steps":        steps,
        "traj":         traj,
    }


def evaluate(checkpoint_path, num_episodes=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Checkpoint: {checkpoint_path}")

    env = EscortReconWrapper(
        env_cfg=env_cfg,
        jd_epoch=JD_EPOCH_ESCORT_RECON,
        red_hv_oe=RED_HV_OE,
        red_esc_oe=RED_ESC_OE,
        blue_dist_range=(BLUE_DIST_MIN_KM, BLUE_DIST_MAX_KM),
        blue_sun_range=(BLUE_SUN_ANGLE_MIN, BLUE_SUN_ANGLE_MAX),
        train_blue=True,
        escort_intercept_dist=ESCORT_THREAT_DIST_KM,
        seed=0,
    )

    policy = Policy(env.observation_space, env.action_space, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["policy"])
    policy.to(device).eval()

    preprocessor = RunningStandardScaler(
        size=env.observation_space.shape[0], device=device)
    if "state_preprocessor" in checkpoint:
        preprocessor.load_state_dict(checkpoint["state_preprocessor"])
    preprocessor.eval()

    dv_blue = float(env_cfg.dv_init_red)

    # ── 随机初始化 ───────────────────────────────────────────────────────────
    print(f"\n=== Random Init ({num_episodes} episodes) ===")
    print(f"  {'Ep':>4}  {'Result':>8}  {'Reward':>9}  "
          f"{'FinalDist':>10}  {'Solar°':>7}  {'EscDist':>8}  "
          f"{'InZone(s)':>10}  {'Steps':>6}")
    rand_results = []
    for ep in range(num_episodes):
        r = run_episode(env, policy, preprocessor, device, fixed_states=None)
        rand_results.append(r)
        status = "SUCCESS" if r["success"] else "FAILED"
        print(f"  {ep+1:4d}  {status:>8}  {r['total_reward']:9.2f}  "
              f"{r['final_dist']:10.2f}  {r['final_solar']:7.1f}  "
              f"{r['final_dist_esc']:8.2f}  "
              f"{r['time_in_zone']:10.1f}  {r['steps']:6d}")

    sr = sum(r["success"] for r in rand_results) / num_episodes * 100
    print(f"\n  成功率        : {sr:.1f}%")
    print(f"  平均奖励      : {np.mean([r['total_reward'] for r in rand_results]):.2f}")
    print(f"  平均最终距离  : {np.mean([r['final_dist'] for r in rand_results]):.2f} km")
    print(f"  平均区内时间  : {np.mean([r['time_in_zone'] for r in rand_results]):.1f} s")
    print(f"  平均进入次数  : {np.mean([r['entry_count'] for r in rand_results]):.1f}")

    # ── 固定初始化 ───────────────────────────────────────────────────────────
    print("\n=== Fixed Init ===")
    print(f"  RedHV : a={DEFAULT_RED_HV[0]} e={DEFAULT_RED_HV[1]} "
          f"i={DEFAULT_RED_HV[2]:.6f} raan={DEFAULT_RED_HV[3]:.6f} "
          f"w={DEFAULT_RED_HV[4]} M={DEFAULT_RED_HV[5]:.6f}")
    print(f"  RedEsc: a={DEFAULT_RED_ESC[0]} e={DEFAULT_RED_ESC[1]} "
          f"i={DEFAULT_RED_ESC[2]:.6f} raan={DEFAULT_RED_ESC[3]:.6f} "
          f"w={DEFAULT_RED_ESC[4]} M={DEFAULT_RED_ESC[5]:.6f}")
    print(f"  Blue  : a={DEFAULT_BLUE[0]} e={DEFAULT_BLUE[1]} "
          f"i={DEFAULT_BLUE[2]:.6f} raan={DEFAULT_BLUE[3]:.6f} "
          f"w={DEFAULT_BLUE[4]} M={DEFAULT_BLUE[5]:.6f}")

    fixed = build_fixed_states(dv_blue)
    r = run_episode(env, policy, preprocessor, device, fixed_states=fixed)
    status = "SUCCESS" if r["success"] else "FAILED"
    print(f"\n  结果 [{status}]:  Reward={r['total_reward']:.2f}  "
          f"FinalDist={r['final_dist']:.2f}km  Solar={r['final_solar']:.1f}°  "
          f"EscDist={r['final_dist_esc']:.2f}km  "
          f"InZone={r['time_in_zone']:.1f}s  Steps={r['steps']}")

    traj = r["traj"]
    print(f"\n  轨迹（每10步）:")
    print(f"  {'Step':>6}  {'Dist(km)':>10}  {'Solar(°)':>10}  "
          f"{'EscDist':>9}  {'InZone':>7}  {'Reward':>8}")
    for i in range(0, len(traj["dist"]), 10):
        print(f"  {i:6d}  {traj['dist'][i]:10.2f}  {traj['solar_deg'][i]:10.1f}  "
              f"{traj['dist_esc'][i]:9.2f}  "
              f"{'YES' if traj['in_zone'][i] else 'NO':>7}  "
              f"{traj['reward'][i]:8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes",   type=int, default=20)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.episodes)
