from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, overload

import gymnasium as gym
import numpy as np
import numpy.typing as npt


# ==========================================
# C++ 绑定的全局函数
# ==========================================
def ma2ta(ma: float, ecc: float, tol: float = 1e-10, max_iter: int = 100) -> float:
    """Convert mean anomaly to true anomaly (rad)"""
    ...

def coe2rv(sma: float, ecc: float, incl: float, raan: float, argp: float, ta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert orbital elements (km, rad) to position (km) and velocity (km/s) vectors"""
    ...

def solar_illumination_angle(pos_sun_j2000: np.ndarray, pos_evader_j2000: np.ndarray, pos_chaser_j2000: np.ndarray) -> float:
    """Calculate solar illumination angle between Sun->Evader and Evader->Chaser vectors"""
    ...
# ==========================================


class SatState:
    r_j2000: np.ndarray  # Eigen::Vector3d, shape (3,)
    v_j2000: np.ndarray  # Eigen::Vector3d, shape (3,)
    dv_remain: float
    is_alive: bool

    def __init__(self) -> None: ...

    def __repr__(self) -> str: ...


class OGESettings:
    def __init__(self) -> None: ...

    def validate(self) -> None: ...

    def set_int(self, key: str, value: int) -> None: ...

    def set_float(self, key: str, value: float) -> None: ...

    def set_bool(self, key: str, value: bool) -> None: ...

    def set_string(self, key: str, value: str) -> None: ...

    def get_int(self, key: str, strict: bool = False) -> int: ...

    def get_float(self, key: str, strict: bool = False) -> float: ...

    def get_bool(self, key: str, strict: bool = False) -> bool: ...

    def get_string(self, key: str, strict: bool = False) -> str: ...


class OGEInterface:
    def __init__(self) -> None: ...

    def init(self) -> None: ...

    def reset(self) -> None: ...

    def get_rewards(self, actions: np.ndarray) -> np.ndarray: ...

    def get_observations(self) -> np.ndarray: ...

    def get_current_time(self) -> float: ...

    def get_settings(self) -> OGESettings: ...

    def get_sat_states(self) -> Dict[str, SatState]: ...

    def reset_with_states(self, states: Dict[str, SatState]) -> None: ...

    def is_terminal(self) -> bool: ...

    def is_truncated(self) -> bool: ...

    def is_captured(self) -> bool: ...

    def get_obs_size(self) -> int: ...

    def act(self, actions: np.ndarray) -> None: ...

    def setInt(self, key: str, value: int) -> None: ...

    def setFloat(self, key: str, value: float) -> None: ...

    def setBool(self, key: str, value: bool) -> None: ...

    def setString(self, key: str, value: str) -> None: ...

    def getInt(self, key: str, strict: bool = False) -> int: ...

    def getFloat(self, key: str, strict: bool = False) -> float: ...

    def getBool(self, key: str, strict: bool = False) -> bool: ...

    def getString(self, key: str, strict: bool = False) -> str: ...


class OGEVectorInterface:
    @overload
    def __init__(
            self,
            num_envs: int,
            batch_size: int = 0,
            num_threads: int = 0,
            thread_affinity_offset: int = -1,
            autoreset_mode: str = "NextStep",
    ) -> None: ...

    @overload
    def __init__(
            self,
            num_envs: int,
            batch_size: int = 0,
            num_threads: int = 0,
            thread_affinity_offset: int = -1,
            autoreset_mode: str = "NextStep",
            settings: OGESettings = ...,
    ) -> None: ...

    def reset(
            self,
            reset_indices: List[int],
            reset_seeds: List[int],
    ) -> Tuple[np.ndarray, Dict[str, Any]]: ...

    def send(
            self,
            actions: np.ndarray,  # shape: (batch_size, num_agents, 3)
    ) -> None: ...

    def recv(self) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]
    ]: ...

    def get_num_envs(self) -> int: ...

    def get_single_observation_size(self) -> int:
        """
        Get the size of one agent's observation
        """
        ...

    def handle(self) -> np.ndarray: ...


try:
    from oge_py.env import OGEEnv, OGEEnvCfg
    from oge_py.vector_env import OGEVectorEnv, OGEVectorEnvCfg

    OGEEnv: TypeAlias = OGEEnv
    OGEEnvCfg: TypeAlias = OGEEnvCfg
    OGEVectorEnv: TypeAlias = OGEVectorEnv
    OGEVectorEnvCfg: TypeAlias = OGEVectorEnvCfg

except ImportError:
    pass
