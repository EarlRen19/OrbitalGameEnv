"""Multi-agent pursuit-evasion wrapper — parameter-sharing IPPO.
File: modules/env_wrapper_ma.py  (companion to modules/env_wrapper.py)

Design
------
All N pursuers share ONE policy (parameter sharing).
skrl PPO sees them as num_envs = N_pursuers parallel environments.

Agent roles
  - red_sat_0 .. red_sat_{P-1}  : pursuers   (RL, shared policy)
  - blue_sat_0                  : HVT         (passive, zero thrust)
  - blue_sat_1 .. blue_sat_{E-1}: interceptors (scripted, PN guidance)

Observation per pursuer (obs_dim = 13 or 17)
  13-dim baseline (same layout as OGESingleEnvWrapper):
    [0:3]   rel_pos to HVT in pursuer LVLH / 200 km
    [3:6]   rel_vel to HVT in pursuer LVLH * 10   (m/s → ~1)
    [6]     dist(pursuer, HVT) / 20 km
    [7]     solar_angle / pi
    [8:11]  sun_dir in HVT LVLH
    [11]    dv_ratio  (pursuer fuel ratio)
    [12]    time_progress

  +4 threat dims (when num_evaders > 1, i.e. interceptors exist):
    [13]    dist to nearest interceptor / 20 km
    [14:17] rel_pos to nearest interceptor in pursuer LVLH / 200 km

Episode termination (per slot)
  - HVT captured by THIS pursuer   → terminated=True,  reward = +200
  - HVT captured by OTHER pursuer  → truncated=True,   reward = +50  (team bonus)
  - THIS pursuer intercepted        → terminated=True,  reward = -150
  - THIS pursuer out of fuel        → terminated=True,  reward = -40
  - time limit                      → truncated=True

Supports scenarios
  4v1 : num_evaders=1, num_pursuers=4  (no interceptors, pure swarm)
  4v4 : num_evaders=4, num_pursuers=4  (1 HVT + 3 interceptors vs 4 pursuers)
  1v2 : num_evaders=2, num_pursuers=1
  ...
"""

from __future__ import annotations

import numpy as np
import torch
import gymnasium

from dataclasses import asdict
from skrl.envs.wrappers.torch.base import Wrapper

try:
    from oge_py._oge_py_ma import MultiAgentOGEEnv as _CppEnv
except ImportError:
    from _oge_py_ma import MultiAgentOGEEnv as _CppEnv


# ── Raw obs index helpers ─────────────────────────────────────────────────────

def _raw_base(num_agents: int) -> int:
    """Index of the 7-dim tail in the raw obs for any agent."""
    return 6 + 7 * (num_agents - 1)


def _blk(k: int) -> int:
    """Start index of the k-th 'other agent' block (k=0 is first listed other)."""
    return 6 + 7 * k


def _other_k(agent_i: int, agent_j: int, num_agents: int) -> int:
    """Which block index k does agent j occupy in agent i's obs?
    Agents are listed in order 0..N-1 skipping i itself.
    """
    k = 0
    for j in range(num_agents):
        if j == agent_i:
            continue
        if j == agent_j:
            return k
        k += 1
    raise ValueError(f"agent {agent_j} not found in obs of agent {agent_i}")


# ── Wrapper ───────────────────────────────────────────────────────────────────

class PursuerSharedPolicyWrapper(Wrapper):
    """Parameter-sharing IPPO wrapper for N pursuers.

    skrl sees this as `num_envs = num_pursuers` parallel envs.
    All pursuers share a single Policy/Value network.

    Parameters
    ----------
    env_cfg      : OGEEnvCfg (or any object with the right fields + asdict support)
    num_evaders  : blue team size  (index 0 = HVT, 1+ = interceptors)
    num_pursuers : red team size
    intercept_distance : km, distance at which interceptor neutralises a pursuer
    threat_obs   : if True and num_evaders > 1, append 4 nearest-interceptor dims
    """

    def __init__(self,
                 env_cfg,
                 num_evaders:  int   = 1,
                 num_pursuers: int   = 1,
                 intercept_distance: float = 30.0,
                 threat_obs:   bool  = True) -> None:

        self._num_evaders  = num_evaders
        self._num_pursuers = num_pursuers
        self._N            = num_evaders + num_pursuers
        self._threat_obs   = threat_obs and (num_evaders > 1)

        # C++ env
        cfg_dict = asdict(env_cfg)
        self._oge = _CppEnv(cfg_dict, num_evaders, num_pursuers, intercept_distance)

        # Raw obs size from C++
        self._raw_obs_dim = self._oge.get_obs_size()  # 13 + 7*(N-1)

        # Compact obs dim
        self._obs_dim = 13 + (4 if self._threat_obs else 0)

        # Action / fuel params
        self._dv_max_pursuer     = float(env_cfg.dv_max_per_step_red)
        self._dv_max_interceptor = float(env_cfg.dv_max_per_step_blue)
        self._dv_init_pursuer    = float(env_cfg.dv_init_red)
        self._init_fuel          = float(env_cfg.dv_init_red)
        self._capture_distance   = float(env_cfg.capture_distance)

        # Per-pursuer episode state
        self._last_dist  = np.full(num_pursuers, 200.0, dtype=np.float64)  # km
        self._alive      = np.ones(num_pursuers, dtype=bool)   # True = still active
        self._last_raw   = None   # (N, raw_obs_dim) last observations

        # Dummy inner env for skrl Wrapper base class
        _obs_dim  = self._obs_dim
        _act_dim  = 3
        _n_envs   = num_pursuers

        class _Dummy:
            observation_space = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(_obs_dim,), dtype=np.float32)
            action_space = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(_act_dim,), dtype=np.float32)
            num_envs = _n_envs

        super().__init__(_Dummy())

        self._observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
        self._action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def num_envs(self) -> int:
        return self._num_pursuers

    # ── Obs extraction ────────────────────────────────────────────────────────

    def _pursuer_global_idx(self, p: int) -> int:
        """Global agent index for pursuer p (0-indexed within red team)."""
        return self._num_evaders + p

    def _extract_obs_for_pursuer(self, raw: np.ndarray, p: int) -> np.ndarray:
        """Extract compact obs for pursuer p from raw obs array.

        raw : shape (num_agents, raw_obs_dim)
        """
        gi  = self._pursuer_global_idx(p)   # global index
        row = raw[gi]                         # pursuer's row in raw obs
        N   = self._N
        base = _raw_base(N)

        # HVT (agent 0) is always the first "other" in pursuer's obs
        # because pursuer index > 0, so agent 0 maps to k=0
        k_hvt = _other_k(gi, 0, N)
        blk_hvt = _blk(k_hvt)

        rel_pos  = row[blk_hvt     : blk_hvt + 3]   # km
        rel_vel  = row[blk_hvt + 3 : blk_hvt + 6]   # m/s
        dist_n   = row[blk_hvt + 6]                  # dist/20km

        solar_angle = row[base + 0]
        dv_ratio    = row[base + 3]
        time_prog   = row[base + 2]
        sun_dir     = row[base + 4 : base + 7]

        obs = np.zeros(self._obs_dim, dtype=np.float32)
        obs[0:3]  = rel_pos / 200.0
        obs[3:6]  = rel_vel * 10.0
        obs[6]    = dist_n
        obs[7]    = solar_angle / np.pi
        obs[8:11] = sun_dir
        obs[11]   = dv_ratio
        obs[12]   = time_prog

        # Optional: nearest interceptor threat
        if self._threat_obs:
            best_dist = np.inf
            best_rp   = np.zeros(3)
            for e in range(1, self._num_evaders):   # interceptors: evader[1+]
                k_e = _other_k(gi, e, N)
                b   = _blk(k_e)
                d   = row[b + 6] * 20.0             # km
                if d < best_dist:
                    best_dist = d
                    best_rp   = row[b : b + 3]
            obs[13]    = best_dist / 20.0
            obs[14:17] = best_rp / 200.0

        return obs

    def _all_pursuer_obs(self, raw: np.ndarray) -> np.ndarray:
        """Returns (num_pursuers, obs_dim) array."""
        return np.stack(
            [self._extract_obs_for_pursuer(raw, p) for p in range(self._num_pursuers)],
            axis=0
        )

    # ── Scripted interceptor policy ───────────────────────────────────────────

    def _interceptor_action(self, raw: np.ndarray, evader_idx: int) -> np.ndarray:
        """Proportional navigation toward nearest alive pursuer."""
        row = raw[evader_idx]
        N   = self._N

        best_dir  = np.zeros(3)
        best_dist = np.inf
        for j in range(self._num_evaders, self._N):   # pursuer global indices
            if not self._alive[j - self._num_evaders]:
                continue
            k   = _other_k(evader_idx, j, N)
            blk = _blk(k)
            rp  = row[blk : blk + 3]           # km in own LVLH
            d   = row[blk + 6] * 20.0           # km
            if d < best_dist:
                best_dist = d
                norm = np.linalg.norm(rp)
                best_dir = rp / (norm + 1e-9)

        return (best_dir * self._dv_max_interceptor).astype(np.float64)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if options and "states" in options:
            self._oge.reset_with_states(options["states"])
        else:
            self._oge.reset()

        raw = np.asarray(self._oge.get_observations(), dtype=np.float64)
        self._last_raw  = raw
        self._alive[:]  = True

        # Init last_dist per pursuer from raw obs
        for p in range(self._num_pursuers):
            gi  = self._pursuer_global_idx(p)
            row = raw[gi]
            k0  = _other_k(gi, 0, self._N)
            self._last_dist[p] = row[_blk(k0) + 6] * 20.0

        obs_batch = self._all_pursuer_obs(raw)   # (P, obs_dim)
        obs_t = torch.tensor(obs_batch, dtype=torch.float32).to(self.device)
        return obs_t, {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, actions):
        """
        actions : tensor (num_pursuers, 3), normalised in [-1, 1]
        Returns obs (P, obs_dim), reward (P, 1), terminated (P, 1), truncated (P, 1)
        """
        N = self._N
        combined = np.zeros((N, 3), dtype=np.float64)

        acts_np = actions.cpu().numpy()  # (P, 3)

        # Assign pursuer actions (scale by dv_max)
        for p in range(self._num_pursuers):
            if self._alive[p]:
                gi = self._pursuer_global_idx(p)
                combined[gi] = acts_np[p] * self._dv_max_pursuer

        # HVT: zero thrust (stays passive)
        # Interceptors: proportional navigation
        for e in range(1, self._num_evaders):
            combined[e] = self._interceptor_action(self._last_raw, e)

        self._oge.act(combined)

        raw = np.asarray(self._oge.get_observations(), dtype=np.float64)
        self._last_raw = raw

        truncated_global = bool(self._oge.is_truncated())
        hvt_captured     = bool(self._oge.is_hvt_captured())

        # Per-pursuer termination signals
        rewards    = np.zeros((self._num_pursuers, 1), dtype=np.float32)
        terminated = np.zeros((self._num_pursuers, 1), dtype=bool)
        truncated  = np.zeros((self._num_pursuers, 1), dtype=bool)

        # Check per-pursuer interception
        sat_states = self._oge.get_sat_states()

        for p in range(self._num_pursuers):
            if not self._alive[p]:
                # Already dead: pass through a zero reward, terminated=True
                terminated[p, 0] = True
                continue

            gi     = self._pursuer_global_idx(p)
            p_id   = f"red_sat_{p}"
            is_intercepted = (
                p_id in sat_states and not sat_states[p_id].is_alive
            )

            rew, done = self._compute_reward_for_pursuer(
                raw, acts_np[p] * self._dv_max_pursuer, p,
                hvt_captured, is_intercepted
            )
            rewards[p, 0]    = rew
            terminated[p, 0] = done
            truncated[p, 0]  = truncated_global and not done

            if done or is_intercepted:
                self._alive[p] = False

        obs_batch = self._all_pursuer_obs(raw)

        obs_t  = torch.tensor(obs_batch, dtype=torch.float32).to(self.device)
        rew_t  = torch.tensor(rewards,   dtype=torch.float32).to(self.device)
        term_t = torch.tensor(terminated, dtype=torch.bool).to(self.device)
        trunc_t = torch.tensor(truncated, dtype=torch.bool).to(self.device)

        return obs_t, rew_t, term_t, trunc_t, {
            "current_time":    self._oge.get_current_time(),
            "hvt_captured":    hvt_captured,
            "alive_pursuers":  int(self._alive.sum()),
        }

    # ── Per-pursuer reward ────────────────────────────────────────────────────

    def _compute_reward_for_pursuer(
        self,
        raw: np.ndarray,
        action: np.ndarray,
        p: int,
        hvt_captured: bool,
        is_intercepted: bool,
    ) -> tuple[float, bool]:
        gi   = self._pursuer_global_idx(p)
        row  = raw[gi]
        N    = self._N
        base = _raw_base(N)

        k_hvt   = _other_k(gi, 0, N)
        rel_pos = row[_blk(k_hvt) : _blk(k_hvt) + 3]
        dist    = np.linalg.norm(rel_pos)              # km
        dv_rem  = row[base + 1]                        # km/s
        action_ms = np.linalg.norm(action) * 1000.0   # m/s

        reward = 0.0
        done   = False

        if hvt_captured:
            # Did THIS pursuer make the capture?
            this_capture = dist < self._capture_distance
            reward = 200.0 + (dv_rem / self._init_fuel) * 50.0 if this_capture \
                     else 50.0   # team bonus for other pursuers
            done   = True
        elif is_intercepted:
            reward = -150.0
            done   = True
        elif dv_rem <= 0.0:
            reward = -40.0
            done   = True
        else:
            # Dense approach reward
            dist_change  = self._last_dist[p] - dist
            dist_reward  = 2.0 * dist_change
            fuel_penalty = -0.01 * action_ms
            reward = dist_reward + fuel_penalty

        self._last_dist[p] = dist
        return reward, done

    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass
