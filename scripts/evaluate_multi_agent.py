"""Evaluate trained multi-agent PPO (parameter-sharing IPPO) on OGE pursuit-evasion.

Metrics per episode
  - hvt_captured   : bool — at least one pursuer reached HVT
  - pursuers_alive : int  — how many pursuers survived (not intercepted / fuel-out)
  - avg_reward     : float — mean total reward across all pursuer slots
  - steps          : int

Summary over N episodes
  - Capture rate (%)
  - Interception rate (%) — fraction of episodes where ≥1 pursuer was intercepted
  - Avg alive pursuers
  - Avg reward per pursuer
  - Avg steps
"""

import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import torch
import numpy as np

from modules.env_wrapper_ma import PursuerSharedPolicyWrapper
from modules.ma_networks import MAPursuerPolicy
from configs.env_cfg import env_cfg

from skrl.resources.preprocessors.torch import RunningStandardScaler


def run_episode(env, policy, state_preprocessor, device):
    """Run one episode; return per-pursuer stats."""
    obs_t, _ = env.reset()
    obs_t = obs_t.to(device)

    num_p = env.num_envs
    total_rewards  = np.zeros(num_p, dtype=np.float64)
    alive_mask     = np.ones(num_p, dtype=bool)   # starts all alive
    was_intercepted = np.zeros(num_p, dtype=bool)

    steps = 0
    hvt_captured = False

    trajectory = {
        "dists": [[] for _ in range(num_p)],      # dist to HVT per pursuer
        "rewards": [[] for _ in range(num_p)],
    }

    while True:
        with torch.no_grad():
            normalized_obs = state_preprocessor(obs_t)
            action, _, _ = policy.act({"states": normalized_obs}, role="policy")

        obs_t, rew_t, term_t, trunc_t, info = env.step(action)
        obs_t = obs_t.to(device)

        rew_np  = rew_t.cpu().numpy().squeeze(-1)   # (P,)
        term_np = term_t.cpu().numpy().squeeze(-1)  # (P,)
        trunc_np = trunc_t.cpu().numpy().squeeze(-1) # (P,)

        # Accumulate rewards for still-alive pursuers this step
        for p in range(num_p):
            if alive_mask[p]:
                total_rewards[p] += rew_np[p]
                trajectory["rewards"][p].append(float(rew_np[p]))

                # Distance from obs: obs[6] = dist/20km
                obs_np = obs_t[p].cpu().numpy()
                trajectory["dists"][p].append(float(obs_np[6] * 20.0))

                # Interception detection: reward = -150 → intercepted
                if term_np[p] and rew_np[p] <= -149.0:
                    was_intercepted[p] = True

                if term_np[p] or trunc_np[p]:
                    alive_mask[p] = False

        steps += 1
        hvt_captured = hvt_captured or bool(info.get("hvt_captured", False))

        # Episode ends when all pursuers done or global truncation
        if not alive_mask.any() or trunc_np.any():
            break

    # Pursuers that were never intercepted AND not dead from fuel = "survived"
    pursuers_alive = int(env.num_envs - was_intercepted.sum())

    return {
        "hvt_captured":     hvt_captured,
        "pursuers_alive":   pursuers_alive,
        "was_intercepted":  was_intercepted.copy(),
        "avg_reward":       float(total_rewards.mean()),
        "total_rewards":    total_rewards.copy(),
        "steps":            steps,
        "trajectory":       trajectory,
    }


def evaluate(checkpoint_path, num_episodes=20,
             num_evaders=2, num_pursuers=2,
             intercept_distance=30.0, threat_obs=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Scenario: {num_pursuers} pursuers vs {num_evaders} evaders "
          f"(1 HVT + {num_evaders - 1} interceptors)")

    env = PursuerSharedPolicyWrapper(
        env_cfg=env_cfg,
        num_evaders=num_evaders,
        num_pursuers=num_pursuers,
        intercept_distance=intercept_distance,
        threat_obs=threat_obs,
    )

    obs_dim = env.observation_space.shape[0]
    policy = MAPursuerPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["policy"])
    policy.to(device)
    policy.eval()

    state_preprocessor = RunningStandardScaler(size=obs_dim, device=device)
    if "state_preprocessor" in checkpoint:
        state_preprocessor.load_state_dict(checkpoint["state_preprocessor"])
    state_preprocessor.eval()

    results = []

    print(f"\n=== Random Init Episodes ({num_episodes}) ===")
    print(f"  {'Ep':>4}  {'HVT':>7}  {'Alive':>5}  {'AvgRew':>8}  {'Steps':>6}  {'PerPursuerRewards'}")
    print("  " + "-" * 80)

    for ep in range(num_episodes):
        r = run_episode(env, policy, state_preprocessor, device)
        results.append(r)

        cap_str  = "CAPT" if r["hvt_captured"] else "MISS"
        rew_str  = "  ".join(f"{v:6.1f}" for v in r["total_rewards"])
        print(f"  {ep+1:4d}  {cap_str:>7}  {r['pursuers_alive']:5d}  "
              f"{r['avg_reward']:8.2f}  {r['steps']:6d}  [{rew_str}]")

    # ── Summary ──────────────────────────────────────────────────────────────
    capture_rate    = sum(r["hvt_captured"]    for r in results) / num_episodes * 100.0
    intercept_rate  = sum(r["pursuers_alive"] < num_pursuers for r in results) / num_episodes * 100.0
    avg_alive       = np.mean([r["pursuers_alive"]  for r in results])
    avg_reward      = np.mean([r["avg_reward"]      for r in results])
    avg_steps       = np.mean([r["steps"]           for r in results])

    print(f"\n=== Summary ({num_episodes} episodes) ===")
    print(f"  Capture Rate        : {capture_rate:.1f}%")
    print(f"  Any Intercept Rate  : {intercept_rate:.1f}%")
    print(f"  Avg Alive Pursuers  : {avg_alive:.2f} / {num_pursuers}")
    print(f"  Avg Reward/Pursuer  : {avg_reward:.2f}")
    print(f"  Avg Steps           : {avg_steps:.1f}")

    # ── Trajectory detail for last episode ───────────────────────────────────
    print(f"\n=== Last Episode Trajectory (every 20 steps) ===")
    r    = results[-1]
    traj = r["trajectory"]
    max_steps = max(len(traj["dists"][p]) for p in range(num_pursuers))

    header = f"  {'Step':>6}"
    for p in range(num_pursuers):
        header += f"  {'P'+str(p)+' Dist':>9}  {'Reward':>7}"
    print(header)

    for i in range(0, max_steps, 20):
        row = f"  {i:6d}"
        for p in range(num_pursuers):
            if i < len(traj["dists"][p]):
                d = traj["dists"][p][i]
                rv = traj["rewards"][p][i]
                row += f"  {d:9.2f}  {rv:7.3f}"
            else:
                row += f"  {'---':>9}  {'---':>7}"
        print(row)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate multi-agent pursuit-evasion policy")
    parser.add_argument("--checkpoint",  type=str,   required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--episodes",   type=int,   default=20,
                        help="Number of evaluation episodes  (default 20)")
    parser.add_argument("--evaders",    type=int,   default=2,
                        help="Blue team size: 1 HVT + (evaders-1) interceptors  (default 2)")
    parser.add_argument("--pursuers",   type=int,   default=2,
                        help="Red team size  (default 2)")
    parser.add_argument("--intercept",  type=float, default=30.0,
                        help="Intercept distance km  (default 30)")
    parser.add_argument("--no-threat",  action="store_true",
                        help="Disable threat dims in obs (match training flag)")
    args = parser.parse_args()

    evaluate(
        checkpoint_path    = args.checkpoint,
        num_episodes       = args.episodes,
        num_evaders        = args.evaders,
        num_pursuers       = args.pursuers,
        intercept_distance = args.intercept,
        threat_obs         = not args.no_threat,
    )
