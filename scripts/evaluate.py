"""Evaluate trained PPO agent on OGE tasks."""

import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import torch
import numpy as np
import oge_py
from modules.env_wrapper import OGESingleEnvWrapper, OGESingleEnvWrapper_operate
from modules.networks import Policy
from configs.env_cfg import env_cfg

from skrl.resources.preprocessors.torch import RunningStandardScaler


def ma2ta(ma, ecc, tol=1e-10, max_iter=100):
    """平近点角 -> 真近点角（弧度）。用牛顿法解开普勒方程。"""
    # 解开普勒方程 E - e*sin(E) = M
    E = ma if ecc < 0.8 else np.pi
    for _ in range(max_iter):
        dE = (ma - E + ecc * np.sin(E)) / (1.0 - ecc * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    # 偏近点角 -> 真近点角
    ta = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(E / 2.0),
    )
    return ta % (2 * np.pi)


def coe2rv_py(sma, ecc, incl, raan, argp, ta):
    """轨道根数(km, rad) -> J2000 r(km), v(km/s)。"""
    MU = 398600.4418  # km^3/s^2
    h = np.sqrt(sma * MU * (1.0 - ecc ** 2))
    r_pf = (h ** 2 / MU) / (1.0 + ecc * np.cos(ta)) * np.array([np.cos(ta), np.sin(ta), 0.0])
    v_pf = (MU / h) * np.array([-np.sin(ta), ecc + np.cos(ta), 0.0])

    def Rz(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    def Rx(a): return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

    Q = Rz(raan) @ Rx(incl) @ Rz(argp)
    return Q @ r_pf, Q @ v_pf


# 指定验证用的初始轨道根数（单位 km, rad）
FIXED_INIT = {
    "red": {
        "sma": 42060.338261, "ecc": 0.003001, "incl": 0.002287,
        "raan": 1.592759, "argp": 3.303797, "ma": 1.033423,
    },
    "blue": {
        "sma": 42169.502913, "ecc": 0.0, "incl": 0.002287,
        "raan": 1.592829, "argp": 0.0, "ma": 4.345423,
    },
}


def build_fixed_states(dv_init_red=0.02, dv_init_blue=0.001):
    """将 FIXED_INIT 轨道根数转换为 SatState 字典。"""
    states = {}
    for name, oe in FIXED_INIT.items():
        ta = ma2ta(oe["ma"], oe["ecc"])
        r, v = coe2rv_py(oe["sma"], oe["ecc"], oe["incl"], oe["raan"], oe["argp"], ta)
        s = oge_py.SatState()
        s.r_j2000 = r
        s.v_j2000 = v
        s.dv_remain = dv_init_red if name == "red" else dv_init_blue
        s.is_alive = True
        agent_key = "red_sat" if name == "red" else "blue_sat"
        states[agent_key] = s
    return states


def run_episode_recon(env, policy, state_preprocessor, device, fixed_states=None):
    """运行侦照任务 episode。"""
    if fixed_states is not None:
        obs, info = env._env.reset(options={"states": fixed_states})
        env._last_obs = np.asarray(obs, dtype=np.float32)
        env._recon_time_accumulated = 0.0
        red_obs = env._last_obs[1]
        env._last_dist = np.linalg.norm(red_obs[6:9])
        refined_obs = env._get_refined_obs(env._last_obs)
        obs_t = torch.tensor(refined_obs, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        obs_t, info = env.reset()
        obs_t = obs_t.to(device)

    done = False
    total_reward = 0.0
    step_count = 0
    is_success = False
    trajectory = {"distances": [], "solar_angles_deg": [], "rewards": []}

    while not done:
        with torch.no_grad():
            normalized_obs = state_preprocessor(obs_t)
            action = policy.act({"states": normalized_obs}, role="policy")[0]
        obs_t, reward, terminated, truncated, info = env.step(action)
        obs_t = obs_t.to(device)

        # 精炼obs: [6]=dist/20, [7]=solar_angle/pi
        obs_np = obs_t.squeeze().cpu().numpy()
        distance = obs_np[6] * 20.0          # km
        solar_angle = obs_np[7] * np.pi      # rad
        solar_angle_deg = np.rad2deg(solar_angle)

        done = terminated.item() or truncated.item()

        if distance <= 20.0 and solar_angle <= np.deg2rad(60.0):
            is_success = True

        trajectory["distances"].append(distance)
        trajectory["solar_angles_deg"].append(solar_angle_deg)
        trajectory["rewards"].append(reward.item())

        total_reward += reward.item()
        step_count += 1

    return {
        "total_reward": total_reward,
        "success": is_success,
        "final_distance": distance,
        "final_solar_angle_deg": solar_angle_deg,
        "steps": step_count,
        "trajectory": trajectory,
    }


def run_episode_operate(env, policy, state_preprocessor, device, fixed_states=None):
    """运行操控任务 episode。"""
    if fixed_states is not None:
        obs, info = env._env.reset(options={"states": fixed_states})
        env._last_obs = np.asarray(obs, dtype=np.float32)
        refined_obs = env._get_refined_obs(env._last_obs)
        red_obs = env._last_obs[1]
        blue_obs = env._last_obs[0]
        env._last_dist = np.linalg.norm(red_obs[6:9])
        env._last_vel_ms = np.linalg.norm(red_obs[3:6] - blue_obs[3:6]) * 1000.0
        obs_t = torch.tensor(refined_obs, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        obs_t, info = env.reset()
        obs_t = obs_t.to(device)

    done = False
    total_reward = 0.0
    step_count = 0
    is_success = False
    trajectory = {"distances": [], "relative_vels_ms": [], "rewards": []}

    while not done:
        with torch.no_grad():
            normalized_obs = state_preprocessor(obs_t)
            action = policy.act({"states": normalized_obs}, role="policy")[0]
        obs_t, reward, terminated, truncated, info = env.step(action)
        obs_t = obs_t.to(device)

        obs_np = obs_t.squeeze().cpu().numpy()
        distance = np.linalg.norm(obs_np[0:3] * 100.0)

        blue_obs = env._last_obs[0]
        red_obs = env._last_obs[1]
        relative_vel_ms = np.linalg.norm(red_obs[3:6] - blue_obs[3:6]) * 1000.0

        done = terminated.item() or truncated.item()

        if distance <= 2.0:
            is_success = True

        trajectory["distances"].append(distance)
        trajectory["relative_vels_ms"].append(relative_vel_ms)
        trajectory["rewards"].append(reward.item())

        total_reward += reward.item()
        step_count += 1

    return {
        "total_reward": total_reward,
        "success": is_success,
        "final_distance": distance,
        "final_vel_ms": relative_vel_ms,
        "steps": step_count,
        "trajectory": trajectory,
    }


def evaluate(checkpoint_path, num_episodes=10, task="recon"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from oge_py import OGEEnv
    raw_env = OGEEnv(env_cfg)

    if task == "recon":
        env = OGESingleEnvWrapper(raw_env, env_cfg)
        dv_max = 0.002 / (3 ** 0.5)
        run_episode = run_episode_recon
    else:
        env = OGESingleEnvWrapper_operate(raw_env, env_cfg)
        dv_max = 0.002
        run_episode = run_episode_operate

    policy = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        dv_max=dv_max,
        clip_actions=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["policy"])
    policy.to(device)
    policy.eval()

    state_preprocessor = RunningStandardScaler(size=env.observation_space.shape[0], device=device)
    if "state_preprocessor" in checkpoint:
        state_preprocessor.load_state_dict(checkpoint["state_preprocessor"])
    state_preprocessor.eval()

    results = []

    # --- 随机初始化 episodes ---
    print(f"=== Random Init Episodes ({num_episodes}) - {task.upper()} Task ===")
    for ep in range(num_episodes):
        r = run_episode(env, policy, state_preprocessor, device, fixed_states=None)
        results.append(r)
        status = "SUCCESS" if r["success"] else "FAILED"

        if task == "recon":
            print(f"  Ep {ep+1:3d} [{status}]: "
                  f"Reward={r['total_reward']:8.2f}  "
                  f"Dist={r['final_distance']:7.2f}km  "
                  f"SolarAngle={r['final_solar_angle_deg']:6.1f}°  "
                  f"Steps={r['steps']}")
        else:
            print(f"  Ep {ep+1:3d} [{status}]: "
                  f"Reward={r['total_reward']:8.2f}  "
                  f"Dist={r['final_distance']:7.2f}km  "
                  f"Vel={r['final_vel_ms']:6.2f}m/s  "
                  f"Steps={r['steps']}")

    print("\n=== Summary (Random) ===")
    print(f"  Avg Reward      : {np.mean([r['total_reward'] for r in results]):.2f}")
    print(f"  Avg Final Dist  : {np.mean([r['final_distance'] for r in results]):.2f} km")
    if task == "recon":
        print(f"  Avg Solar Angle : {np.mean([r['final_solar_angle_deg'] for r in results]):.1f} °")
    else:
        print(f"  Avg Final Vel   : {np.mean([r['final_vel_ms'] for r in results]):.2f} m/s")
    print(f"  Success Rate    : {sum(r['success'] for r in results) / num_episodes * 100:.1f}%")

    # --- 指定初始状态验证 ---
    print("\n=== Fixed Init Episode ===")
    print("  Red : sma=42060.338261 e=0.003001 incl=0.002287 raan=1.592759 argp=3.303797 MA=1.033423")
    print("  Blue: sma=42169.502913 e=0.0      incl=0.002287 raan=1.592829 argp=0.0      MA=4.345423")

    fixed_states = build_fixed_states(
        dv_init_red=env_cfg.dv_init_red,
        dv_init_blue=env_cfg.dv_init_blue,
    )
    r = run_episode(env, policy, state_preprocessor, device, fixed_states=fixed_states)
    status = "SUCCESS" if r["success"] else "FAILED"

    if task == "recon":
        print(f"  Result [{status}]: "
              f"Reward={r['total_reward']:.2f}  "
              f"FinalDist={r['final_distance']:.2f}km  "
              f"FinalSolarAngle={r['final_solar_angle_deg']:.1f}°  "
              f"Steps={r['steps']}")

        traj = r["trajectory"]
        print("\n  Trajectory (every 20 steps):")
        print(f"  {'Step':>6}  {'Dist(km)':>10}  {'SolarAngle(°)':>14}  {'Reward':>8}")
        for i in range(0, len(traj["distances"]), 20):
            print(f"  {i:6d}  {traj['distances'][i]:10.2f}  "
                  f"{traj['solar_angles_deg'][i]:14.1f}  "
                  f"{traj['rewards'][i]:8.3f}")
    else:
        print(f"  Result [{status}]: "
              f"Reward={r['total_reward']:.2f}  "
              f"FinalDist={r['final_distance']:.2f}km  "
              f"FinalVel={r['final_vel_ms']:.2f}m/s  "
              f"Steps={r['steps']}")

        traj = r["trajectory"]
        print("\n  Trajectory (every 20 steps):")
        print(f"  {'Step':>6}  {'Dist(km)':>10}  {'Vel(m/s)':>10}  {'Reward':>8}")
        for i in range(0, len(traj["distances"]), 20):
            print(f"  {i:6d}  {traj['distances'][i]:10.2f}  "
                  f"{traj['relative_vels_ms'][i]:10.2f}  "
                  f"{traj['rewards'][i]:8.3f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--task", type=str, default="recon", choices=["recon", "operate"],
                        help="Task type: recon (reconnaissance) or operate (maneuver)")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.episodes, args.task)
