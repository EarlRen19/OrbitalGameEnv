"""
集群蓝方训练 Wrapper
====================
场景：蓝方 6 颗星各自对红色高价值星(Red HV)执行不同任务。
红 HV 和红护卫星均无机动，固定在初始轨道根数对应的 ECI 位置。
第一阶段只训练蓝方，每种任务独立一个 Wrapper，独立策略。

初始时间：北京时间 2023-11-16 22:30 = UTC 2023-11-16 14:30:00
JD_EPOCH = 2460264.770833

C++ 角色映射（num_evaders=7, num_pursuers=6）：
  evader[0]    = blue_sat_0  → 红色高价值星 (Red HV)    — 无机动
  evader[1..6] = blue_sat_1~6 → 红色护卫星 1~6          — 第一阶段无机动
  pursuer[0]   = red_sat_0   → 蓝方打击星 1  (AGENT_IDX=7)
  pursuer[1]   = red_sat_1   → 蓝方打击星 2  (AGENT_IDX=8)
  pursuer[2]   = red_sat_2   → 蓝方干扰星 3  (AGENT_IDX=9)
  pursuer[3]   = red_sat_3   → 蓝方侦照星 4  (AGENT_IDX=10)
  pursuer[4]   = red_sat_4   → 蓝方侦照星 5  (AGENT_IDX=11)
  pursuer[5]   = red_sat_5   → 蓝方操控星 6  (AGENT_IDX=12)

观测维度（17 维，由 C++ get_task_observations 直接输出）：
  [0:3]   rel_pos to Red HV / 200km  (LVLH)
  [3:6]   rel_vel to Red HV * 10     (m/s 量级)
  [6]     dist / 20km
  [7]     task_angle / pi
            STRIKE/RECON → solar_illumination_angle(sun, HV, self) / pi
            JAM          → jamming_angle(HV, self) / pi
            OPERATE      → rel_speed_ms * 10
  [8:11]  auxiliary direction in HV LVLH
            STRIKE/RECON → sun_dir
            JAM          → HV-to-earth unit vector
            OPERATE      → zeros
  [11]    dv_ratio (dv_remain / dv_init)
  [12]    time_progress
  [13]    0  (第一阶段无威胁，第二阶段填护卫星距离)
  [14:17] 0  (第一阶段无威胁)
"""

from __future__ import annotations
import numpy as np
import torch
import gymnasium
from copy import deepcopy
from dataclasses import asdict
from skrl.envs.wrappers.torch.base import Wrapper

try:
    from oge_py._oge_py_ma import MultiAgentOGEEnv as _CppEnv
    from oge_py._oge_py import SatState
except ImportError:
    from _oge_py_ma import MultiAgentOGEEnv as _CppEnv
    from _oge_py import SatState

# ── 常量 ──────────────────────────────────────────────────────────────────────
MU = 398600.4418   # km^3/s^2

# C++ TaskType 枚举值（与 multi_agent_oge.h 对应）
TASK_STRIKE  = 0
TASK_RECON   = 1
TASK_JAM     = 2
TASK_OPERATE = 3

# Red HV 的全局 agent index（evader[0]）
HV_IDX = 0

# 6 颗红护卫星的全局 index（evader[1..6]）
RED_ESC_IDXS = [1, 2, 3, 4, 5, 6]

# 6 颗蓝星的全局 index（pursuer[0..5] → 全局 7..12）
BLUE_IDXS = [7, 8, 9, 10, 11, 12]

# 蓝星任务类型（与 BLUE_IDXS 顺序对应）
BLUE_TASK_TYPES = [
    TASK_STRIKE,   # 蓝1 (global=7)
    TASK_STRIKE,   # 蓝2 (global=8)
    TASK_JAM,      # 蓝3 (global=9)
    TASK_RECON,    # 蓝4 (global=10)
    TASK_RECON,    # 蓝5 (global=11)
    TASK_OPERATE,  # 蓝6 (global=12)
]

# ── 轨道根数 → ECI ─────────────────────────────────────────────────────────────

def _ma2ta(M, e, tol=1e-10):
    E = M if e < 0.8 else np.pi
    for _ in range(100):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return (2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                              np.sqrt(1 - e) * np.cos(E / 2))) % (2 * np.pi)


def _coe2rv(a, e, i, raan, w, M):
    ta = _ma2ta(M, e)
    h = np.sqrt(a * MU * (1 - e * e))
    r_mag = (h * h / MU) / (1 + e * np.cos(ta))
    rp = r_mag * np.array([np.cos(ta), np.sin(ta), 0.0])
    vp = (MU / h) * np.array([-np.sin(ta), e + np.cos(ta), 0.0])
    def Rz(a_): return np.array([[np.cos(a_), -np.sin(a_), 0],
                                  [np.sin(a_),  np.cos(a_), 0], [0, 0, 1]])
    def Rx(a_): return np.array([[1, 0, 0],
                                  [0,  np.cos(a_), -np.sin(a_)],
                                  [0,  np.sin(a_),  np.cos(a_)]])
    Q = Rz(raan) @ Rx(i) @ Rz(w)
    return Q @ rp, Q @ vp


def _make_state(r, v, dv):
    s = SatState()
    s.r_j2000   = np.asarray(r, dtype=np.float64)
    s.v_j2000   = np.asarray(v, dtype=np.float64)
    s.dv_remain = float(dv)
    s.is_alive  = True
    return s


# ── 基类 ──────────────────────────────────────────────────────────────────────

class _BlueBaseWrapper(Wrapper):
    """
    蓝方单星训练基类。每个实例只训练一颗蓝星，其余星固定不动。

    子类必须定义：
      TASK_TYPE : int        C++ TaskType 枚举值
      DV_INIT   : float      本星燃料 (km/s)
      AGENT_IDX : int        本星全局 agent index（7~12）
    """

    TASK_TYPE : int   = TASK_RECON
    DV_INIT   : float = 0.020
    AGENT_IDX : int   = 7
    OBS_DIM   : int   = 17

    def __init__(self, env_cfg, red_hv_oe: dict,
                 red_esc_oe_list: list, blue_oe_list: list):
        """
        Parameters
        ----------
        env_cfg          : OGEEnvCfg dataclass（来自 cluster_escort_cfg）
        red_hv_oe        : 红 HV 轨道根数
        red_esc_oe_list  : 6 颗红护卫星轨道根数列表（固定，无机动）
        blue_oe_list     : 6 颗蓝星轨道根数列表（本星有燃料，其余零燃料）
        """
        self._red_hv_oe       = red_hv_oe
        self._red_esc_oe_list = red_esc_oe_list
        self._blue_oe_list    = blue_oe_list
        self._dv_max          = float(env_cfg.dv_max_per_step_red)
        self._timestep        = float(env_cfg.timestep)

        # 预计算所有固定星的 ECI 状态
        self._r_hv, self._v_hv = _coe2rv(**red_hv_oe)
        self._esc_states = [_coe2rv(**oe) for oe in red_esc_oe_list]
        self._blue_rv    = [_coe2rv(**oe) for oe in blue_oe_list]

        # C++ 后端：7 evaders (HV + 6护卫) + 6 pursuers (6蓝星)
        cfg_dict = asdict(env_cfg)
        cfg_dict["jd_epoch"] = 2460264.770833   # BJT 2023-11-16 22:30
        self._oge = _CppEnv(cfg_dict,
                            num_evaders=7,
                            num_pursuers=6,
                            intercept_distance=0.1)  # 蓝星间不互相拦截

        # 设置任务分配（13 个 agent）
        # evader[0]=HV, evader[1..6]=红护卫（无任务），pursuer[0..5]=蓝星
        assignments = [
            {"task_type": TASK_RECON, "target_idx": 0, "threat_idx": -1},  # HV
        ]
        for k in range(6):  # 红护卫 1~6，第一阶段无任务
            assignments.append({"task_type": TASK_RECON,
                                 "target_idx": 0, "threat_idx": -1})
        for k, t in enumerate(BLUE_TASK_TYPES):  # 蓝星 1~6
            assignments.append({"task_type":  t,
                                 "target_idx": HV_IDX,
                                 "threat_idx": -1})
        self._oge.set_task_assignment(assignments)

        # episode 状态
        self._last_dist = 200.0
        self._acc_time  = 0.0
        self._in_zone   = False

        class _Dummy:
            observation_space = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
            action_space = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
            num_envs = 1

        super().__init__(_Dummy())
        self._observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32)
        self._action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    @property
    def observation_space(self): return self._observation_space
    @property
    def action_space(self):      return self._action_space
    @property
    def num_envs(self):          return 1

    def _build_states(self) -> dict:
        """所有 13 个 agent 的初始状态：HV + 6护卫（无燃料）+ 6蓝星（本星有燃料）。"""
        states = {"blue_sat_0": _make_state(self._r_hv, self._v_hv, 0.0)}
        for k, (r, v) in enumerate(self._esc_states):
            states[f"blue_sat_{k+1}"] = _make_state(r, v, 0.0)
        for k, (r, v) in enumerate(self._blue_rv):
            # 本星有燃料，其余蓝星零燃料（固定不动）
            dv = self.DV_INIT if (k + 7) == self.AGENT_IDX else 0.0
            states[f"red_sat_{k}"] = _make_state(r, v, dv)
        return states

    def reset(self, seed=None, options=None):
        self._oge.reset_with_states(self._build_states())
        raw = np.asarray(self._oge.get_task_observations(), dtype=np.float32)
        obs = raw[self.AGENT_IDX]
        self._last_dist = float(obs[6]) * 20.0
        self._acc_time  = 0.0
        self._in_zone   = False
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device), {}

    def step(self, actions):
        act_np = actions.squeeze().cpu().numpy() * self._dv_max
        combined = np.zeros((13, 3), dtype=np.float64)
        combined[self.AGENT_IDX] = act_np

        self._oge.act(combined)
        raw = np.asarray(self._oge.get_task_observations(), dtype=np.float32)
        obs = raw[self.AGENT_IDX]

        truncated = bool(self._oge.is_truncated())
        rew, done = self._compute_reward(obs, act_np)

        obs_t   = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        rew_t   = torch.tensor([[rew]], dtype=torch.float32).to(self.device)
        term_t  = torch.tensor([[done]], dtype=torch.bool).to(self.device)
        trunc_t = torch.tensor([[truncated and not done]], dtype=torch.bool).to(self.device)
        return obs_t, rew_t, term_t, trunc_t, {"current_time": self._oge.get_current_time()}

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray):
        raise NotImplementedError

    def render(self, *args, **kwargs): pass
    def close(self): pass


# ── 打击任务 ──────────────────────────────────────────────────────────────────

class BlueStrikeWrapper(_BlueBaseWrapper):
    """
    打击任务：dist ≤ 20km, solar_angle ≤ 90°, 持续 ≥ 40s
    燃料预算：30 m/s

    obs[7] = solar_illumination_angle(sun, HV, self) / pi  （顶点=HV）
    obs[8:11] = sun_dir in HV LVLH

    奖励设计参考侦照任务，阈值放宽（90° vs 60°，40s vs 120s）。
    """

    TASK_TYPE = TASK_STRIKE
    DV_INIT   = 0.030

    DIST_KM    = 20.0
    ANGLE_DEG  = 90.0
    DURATION_S = 40.0

    def _compute_reward(self, obs, action):
        dist_km   = float(obs[6]) * 20.0
        angle_rad = float(obs[7]) * np.pi
        dv_ratio  = float(obs[11])
        action_ms = np.linalg.norm(action) * 1000.0

        reward = 0.0
        done   = False

        in_zone = (dist_km <= self.DIST_KM and
                   np.rad2deg(angle_rad) <= self.ANGLE_DEG)

        if in_zone:
            if not self._in_zone:
                reward += 100.0 + dv_ratio * 30.0
                self._acc_time = self._timestep
                self._in_zone  = True
            else:
                self._acc_time += self._timestep
                reward += 5.0
                if self._acc_time >= self.DURATION_S:
                    reward += 200.0 + dv_ratio * 50.0
                    done = True
            self._last_dist = dist_km
            return reward, done

        self._acc_time = 0.0
        self._in_zone  = False

        if dv_ratio <= 0.0:
            return -40.0, True

        # 距离引导
        reward += 2.0 * (self._last_dist - dist_km)

        # 角度引导（30km 内，阈值 90°，系数 1.0）
        if dist_km <= 30.0:
            angle_deg = np.rad2deg(angle_rad)
            if angle_deg <= self.ANGLE_DEG:
                reward += 1.0 * (1.0 - angle_deg / self.ANGLE_DEG)

        reward -= 0.01 * action_ms
        self._last_dist = dist_km
        return reward, done


# ── 侦照任务 ──────────────────────────────────────────────────────────────────

class BlueReconWrapper(_BlueBaseWrapper):
    """
    侦照任务：dist ≤ 20km, solar_angle ≤ 60°, 持续 ≥ 120s
    燃料预算：20 m/s

    obs[7] = solar_illumination_angle(sun, HV, self) / pi  （顶点=HV）
    obs[8:11] = sun_dir in HV LVLH

    直接对齐 env_wrapper.py 的 _compute_recon_reward。
    """

    TASK_TYPE = TASK_RECON
    DV_INIT   = 0.020

    DIST_KM    = 20.0
    ANGLE_DEG  = 60.0
    DURATION_S = 120.0

    def _compute_reward(self, obs, action):
        dist_km   = float(obs[6]) * 20.0
        angle_rad = float(obs[7]) * np.pi
        dv_ratio  = float(obs[11])
        # obs[3:6] = rel_vel_lvlh * 10，norm 直接是 m/s 量级
        rel_vel_ms = float(np.linalg.norm(obs[3:6]))
        action_ms  = np.linalg.norm(action) * 1000.0

        reward = 0.0
        done   = False

        in_zone = (dist_km <= self.DIST_KM and
                   np.rad2deg(angle_rad) <= self.ANGLE_DEG)

        if in_zone:
            if not self._in_zone:
                reward += 100.0 + dv_ratio * 30.0
                self._acc_time = self._timestep
                self._in_zone  = True
            else:
                self._acc_time += self._timestep
                reward += 5.0
                if self._acc_time >= self.DURATION_S:
                    reward += 200.0 + dv_ratio * 50.0
                    done = True
            self._last_dist = dist_km
            return reward, done

        self._acc_time = 0.0
        self._in_zone  = False

        if dv_ratio <= 0.0:
            return -40.0, True

        # 距离引导
        reward += 2.0 * (self._last_dist - dist_km)

        # 角度引导 + 速度漏斗（25km 内）
        if dist_km <= 25.0:
            angle_deg = np.rad2deg(angle_rad)
            if angle_deg <= self.ANGLE_DEG:
                reward += 2.0 * (1.0 - angle_deg / self.ANGLE_DEG)
            target_vel_ms = 1.0 + (dist_km / 25.0) * 4.0
            if rel_vel_ms > target_vel_ms:
                reward -= 0.2 * (rel_vel_ms - target_vel_ms)

        reward -= 0.01 * action_ms
        self._last_dist = dist_km
        return reward, done


# ── 干扰任务 ──────────────────────────────────────────────────────────────────

class BlueJamWrapper(_BlueBaseWrapper):
    """
    干扰任务：dist ≤ 20km, jamming_angle ≤ 5°, 持续 ≥ 600s
    燃料预算：20 m/s

    obs[7] = jamming_angle(HV, self) / pi  （顶点=HV，角度越小越好）
    obs[8:11] = HV-to-earth 方向在 HV LVLH 下的单位向量

    干扰角定义：HV 与地心连线 和 HV 到干扰星连线 之间的夹角。
    5° 极小，需要两段式引导：先靠近，再压角度。
    """

    TASK_TYPE = TASK_JAM
    DV_INIT   = 0.020

    DIST_KM    = 20.0
    ANGLE_DEG  = 5.0
    DURATION_S = 600.0

    def _compute_reward(self, obs, action):
        dist_km   = float(obs[6]) * 20.0
        angle_rad = float(obs[7]) * np.pi
        angle_deg = np.rad2deg(angle_rad)
        dv_ratio  = float(obs[11])
        action_ms = np.linalg.norm(action) * 1000.0

        reward = 0.0
        done   = False

        in_zone = (dist_km <= self.DIST_KM and angle_deg <= self.ANGLE_DEG)

        if in_zone:
            if not self._in_zone:
                reward += 100.0 + dv_ratio * 30.0
                self._acc_time = self._timestep
                self._in_zone  = True
            else:
                self._acc_time += self._timestep
                reward += 5.0
                if self._acc_time >= self.DURATION_S:
                    reward += 200.0 + dv_ratio * 50.0
                    done = True
            self._last_dist = dist_km
            return reward, done

        self._acc_time = 0.0
        self._in_zone  = False

        if dv_ratio <= 0.0:
            return -40.0, True

        # 距离引导（主导阶段：dist > 20km）
        reward += 2.0 * (self._last_dist - dist_km)

        # 角度引导（进入 20km 后分两段）
        if dist_km <= self.DIST_KM:
            if dist_km > 5.0:
                # 20~5km：轻度压角度，参考上限 90°
                reward += 2.0 * (1.0 - angle_deg / 90.0)
            else:
                # 5km 内：强力压角度到 5° 以内
                reward += 5.0 * max(0.0, 1.0 - angle_deg / self.ANGLE_DEG)

        reward -= 0.01 * action_ms
        self._last_dist = dist_km
        return reward, done


# ── 操控任务 ──────────────────────────────────────────────────────────────────

class BlueOperateWrapper(_BlueBaseWrapper):
    """
    操控任务：dist ≤ 2km, rel_vel ≤ 1m/s
    燃料预算：20 m/s

    obs[7] = rel_speed_ms * 10（C++ 里直接存的，不是角度）
    obs[8:11] = zeros

    直接对齐 env_wrapper.py 的 _compute_operate_reward。
    """

    TASK_TYPE = TASK_OPERATE
    DV_INIT   = 0.020

    DIST_KM  = 2.0
    VEL_MS   = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_vel_ms = 0.0

    def _compute_reward(self, obs, action):
        dist_km    = float(obs[6]) * 20.0
        # obs[7] = rel_speed_ms * 10，还原为 m/s
        rel_vel_ms = float(obs[7]) / 10.0
        dv_ratio   = float(obs[11])
        action_ms  = np.linalg.norm(action) * 1000.0

        reward = 0.0
        done   = False

        if dist_km <= self.DIST_KM:
            reward = 200.0 + dv_ratio * 50.0
            done   = True
            return reward, done

        if dv_ratio <= 0.0:
            return -20.0, True

        # 距离引导
        reward += 2.0 * (self._last_dist - dist_km)

        # 速度匹配（分段目标速度，对齐 env_wrapper.py）
        if dist_km > 20.0:
            target_vel_ms = 10.0
        elif dist_km > 5.0:
            target_vel_ms = 5.0
        else:
            target_vel_ms = 1.0

        if rel_vel_ms > target_vel_ms:
            reward -= 0.1 * (rel_vel_ms - target_vel_ms)

        reward -= 0.01 * action_ms
        self._last_dist   = dist_km
        self._last_vel_ms = rel_vel_ms
        return reward, done


# ── 工厂函数 ──────────────────────────────────────────────────────────────────

# 任务索引 → (Wrapper 类, AGENT_IDX, DV_INIT)
_BLUE_SPEC = [
    (BlueStrikeWrapper,  7,  0.030),   # 蓝1 打击
    (BlueStrikeWrapper,  8,  0.030),   # 蓝2 打击
    (BlueJamWrapper,     9,  0.020),   # 蓝3 干扰
    (BlueReconWrapper,  10,  0.020),   # 蓝4 侦照
    (BlueReconWrapper,  11,  0.020),   # 蓝5 侦照
    (BlueOperateWrapper,12,  0.020),   # 蓝6 操控
]


def make_blue_wrapper(task_idx: int, env_cfg,
                      red_hv_oe: dict,
                      red_esc_oe_list: list,
                      blue_oe_list: list) -> _BlueBaseWrapper:
    """
    创建指定蓝星的训练 Wrapper。

    Parameters
    ----------
    task_idx       : 0~5，对应蓝星 1~6
    env_cfg        : OGEEnvCfg（来自 cluster_escort_cfg）
    red_hv_oe      : 红 HV 轨道根数
    red_esc_oe_list: 6 颗红护卫星轨道根数列表
    blue_oe_list   : 6 颗蓝星轨道根数列表
    """
    cls, agent_idx, dv_init = _BLUE_SPEC[task_idx]
    w = cls(env_cfg, red_hv_oe, red_esc_oe_list, blue_oe_list)
    w.AGENT_IDX = agent_idx
    w.DV_INIT   = dv_init
    return w
