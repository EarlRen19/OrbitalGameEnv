from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import gymnasium as gym

try:
    from oge_py._oge_py import OGEVectorInterface
    _VECTOR_AVAILABLE = True
except ImportError:
    _VECTOR_AVAILABLE = False


@dataclass
class OGEVectorEnvCfg:
    num_envs: int = 1
    batch_size: int = 0
    num_threads: int = 0
    thread_affinity_offset: int = -1
    autoreset_mode: str = "NextStep"


class OGEVectorEnv:
    def __init__(self, cfg: OGEVectorEnvCfg):
        if not _VECTOR_AVAILABLE:
            raise ImportError(
                "OGEVectorEnv requires BUILD_VECTOR_LIB=ON. "
                "Recompile with -DBUILD_VECTOR_LIB=ON."
            )
        self._env = OGEVectorInterface(
            num_envs=cfg.num_envs,
            batch_size=cfg.batch_size,
            num_threads=cfg.num_threads,
            thread_affinity_offset=cfg.thread_affinity_offset,
            autoreset_mode=cfg.autoreset_mode,
        )

    def reset(self, reset_indices=None, reset_seeds=None):
        return self._env.reset(reset_indices or [], reset_seeds or [])

    def send(self, actions: np.ndarray):
        self._env.send(actions)

    def recv(self):
        return self._env.recv()

    def get_num_envs(self) -> int:
        return self._env.get_num_envs()
