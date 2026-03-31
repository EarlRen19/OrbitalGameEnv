from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from typing import TypedDict, Any, Literal

import oge_py
import gymnasium as gym
import numpy as np
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from scipy.signal import step


class OGEEnvStepMetadata(TypedDict):
    """Step info options."""
    current_time: float


@dataclass
class OGEEnvCfg:
    # int settings
    random_seed: int = 42

    # float settings - orbital elements
    sma_base: float = 42164.0  # semi-major axis, km
    ecc_base: float = 0.0   # eccentricity
    incl_base: float = 0.0  # inclination, rad
    RA_base: float = 0.0  # right ascension of the ascending node, rad
    w_base: float = 0.0  # argument of perigee, rad
    TA_base: float = 0.0  # true anomaly, rad

    # float settings - agent params
    dv_init_red: float = 0.2  # km/s
    dv_init_blue: float = 0.2  # km/s
    dv_max_per_step_red: float = 0.01  # km/s
    dv_max_per_step_blue: float = 0.01  # km/s
    capture_distance: float = 30.0  # km
    timestep: float = 60.0  # s
    terminal_time: float = 3600.0  # s

    # float settings - initialization
    sma_perturb_max: float = 10.0  # km
    dist_init_offset_min: float = 90.0  # km
    dist_init_offset_max: float = 100.0  # km

    # float settings - reward weights
    reward_time_weight: float = -0.05
    reward_formation_weight: float = 0.04
    reward_fuel_weight: float = -0.2
    reward_capture_weight: float = 10.0
    reward_timeout_weight: float = -2.0
    reward_fuelout_weight: float = -1.0
    reward_phase_dist_weight: float = 1.0

    # distance reward sub-parameters
    reward_far_sma_penalty_scale: float = 10.0  # far field: coefficient for sma_diff_ratio penalty when drifting in wrong direction
    reward_far_drift_scale: float = 5.0  # far field: scale sma_diff_ratio to [0, reward_far_drift_max] drift reward
    reward_far_drift_max: float = 2.0  # far field: upper clamp for drift reward component
    reward_far_angle_weight: float = 0.5  # far field: weight of angle component in positive far-field reward
    reward_near_energy_scale: float = 10.0  # near field: scale sma_diff_ratio for energy penalty
    reward_near_energy_weight: float = 0.05  # near field: weight of energy penalty in near-field blend
    reward_dist_capture_bonus: float = 0.1  # extra bonus per unit inside capture zone (dist < capture_distance)
    reward_dist_min: float = -1.0  # lower clamp for distance reward when pursuer is very far
    reward_alpha_scale: float = 1.0  # blending factor scale between near and far field rewards


class OGEEnv(gym.Env, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
            self,
            cfg: OGEEnvCfg
    ):
        # TODO: check cfg

        utils.EzPickle.__init__(
            self,
            cfg
        )

        # Initialize OGE
        self.oge = oge_py.OGEInterface()
        self.cfg = cfg
        self.set_params()
        self.oge.init()

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.oge.get_obs_size(),),
            dtype=np.float64,
        )

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, OGEEnvStepMetadata]:
        super().reset(seed=seed, options=options)
        states = options.get("states") if options else None
        if states is not None:
            self.oge.reset_with_states(states)
        else:
            self.oge.reset()
        observations = self.oge.get_observations()
        return observations, self._get_info()

    def step(
            self,
            actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, OGEEnvStepMetadata]:
        """Perform one step

        :param actions:
        :return:
        """
        self.oge.act(actions)
        observations = self.oge.get_observations()
        rewards = self.oge.get_rewards(actions)
        termination = self.oge.is_terminal()
        truncation = self.oge.is_truncated()

        return observations, rewards, termination, truncation, self._get_info()

    def render(self):
        # TODO: need to render here or using c++ backend?
        pass

    def set_params(self) -> None:
        cfg_dict = asdict(self.cfg)
        for key, value in cfg_dict.items():
            if isinstance(value, bool):
                self.oge.setBool(key, value)
            elif isinstance(value, int):
                self.oge.setInt(key, value)
            elif isinstance(value, float):
                self.oge.setFloat(key, value)
            elif isinstance(value, str):
                self.oge.setString(key, value)
            else:
                raise TypeError(f"Unsupported type of key={key}: {type(value)}")

    def get_sat_states(self) -> dict:
        """Returns all satellite states as a dict: {agent_id: {field: value}}."""
        raw = self.oge.get_sat_states()
        return {
            name: {
                "r_j2000": state.r_j2000,
                "v_j2000": state.v_j2000,
                "dv_remain": state.dv_remain,
                "is_alive": state.is_alive,
            }
            for name, state in raw.items()
        }

    def is_captured(self) -> bool:
        """Returns True if any pursuer has captured the evader."""
        return self.oge.is_captured()

    def _get_info(self) -> OGEEnvStepMetadata:
        return {
            "current_time": self.oge.get_current_time()
        }
