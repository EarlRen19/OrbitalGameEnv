"""Multi-agent config for pursuit-evasion scenarios."""

from dataclasses import dataclass


@dataclass
class MultiAgentCfg:
    """Configuration for multi-agent pursuit-evasion environment.

    Orbital physics parameters are inherited from OGEEnvCfg (same OGESettings).
    Only the agent-count and role-specific parameters are added here.
    """
    # ── Agent counts ──────────────────────────────────────────────────────────
    num_evaders:  int   = 2   # blue team: index 0 = HVT, 1+ = interceptors
    num_pursuers: int   = 1   # red  team: all are pursuers

    # ── Intercept distance (km) ───────────────────────────────────────────────
    # A pursuer is neutralised when any interceptor comes within this range.
    intercept_distance: float = 30.0

    # ── Per-role dv budgets (km/s) ────────────────────────────────────────────
    # These override env_cfg values if provided (set to None to use env_cfg).
    dv_init_pursuer:          float = 0.02   # 20 m/s
    dv_max_per_step_pursuer:  float = 0.002  # 2 m/s
    dv_init_interceptor:      float = 0.015  # 15 m/s
    dv_max_per_step_interceptor: float = 0.002

    # ── Reward weights (pursuer's perspective) ────────────────────────────────
    reward_hvt_capture:      float = 200.0   # bonus for HVT capture
    reward_intercept_penalty: float = -150.0  # penalty if pursuer intercepted
    reward_approach_scale:   float = 2.0     # per km approaching HVT
    reward_fuel_penalty:     float = -0.01   # per m/s action


# Default multi-agent config for 2v1 (1 HVT + 1 interceptor vs 1 pursuer)
multi_agent_cfg = MultiAgentCfg(
    num_evaders=2,
    num_pursuers=1,
    intercept_distance=30.0,
)

# 4v1: 1 HVT + 3 interceptors vs 1 pursuer
multi_agent_cfg_4v1 = MultiAgentCfg(
    num_evaders=4,
    num_pursuers=1,
    intercept_distance=30.0,
)
