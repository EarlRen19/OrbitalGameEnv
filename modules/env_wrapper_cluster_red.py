"""
集群红方护卫训练 Wrapper
========================
场景：红方 6 颗护卫星各自对对应的蓝星执行侦照/拦截任务。
第二阶段：蓝星加载已训练的 ckpt 运行，红护卫为 RL 训练目标。

C++ 角色映射（num_evaders=7, num_pursuers=6）：
  evader[0]    = blue_sat_0  → 红色高价值星 (Red HV)    — 无机动
  evader[1..6] = blue_sat_1~6 → 红色护卫星 1~6          — 第二阶段 RL agent
  pursuer[0]   = red_sat_0   → 蓝方打击星 1  (AGENT_IDX=7)
  pursuer[1]   = red_sat_1   → 蓝方打击星 2  (AGENT_IDX=8)
  pursuer[2]   = red_sat_2   → 蓝方干扰星 3  (AGENT_IDX=9)
  pursuer[3]   = red_sat_3   → 蓝方侦照星 4  (AGENT_IDX=10)
  pursuer[4]   = red_sat_4   → 蓝方侦照星 5  (AGENT_IDX=11)
  pursuer[5]   = red_sat_5   → 蓝方操控星 6  (AGENT_IDX=12)

红护卫任务：对对应蓝星执行侦照（RECON），侦照成功=拦截/驱离。
  evader[1] → 侦照 pursuer[0] (蓝1，打击)   target_idx=7
  evader[2] → 侦照 pursuer[1] (蓝2，打击)   target_idx=8
  evader[3] → 侦照 pursuer[2] (蓝3，干扰)   target_idx=9
  evader[4] → 侦照 pursuer[3] (蓝4，侦照)   target_idx=10
  evader[5] → 侦照 pursuer[4] (蓝5，侦照)   target_idx=11
  evader[6] → 侦照 pursuer[5] (蓝6，操控)   target_idx=12

观测维度（17 维，与蓝方相同结构）：
  [0:3]   rel_pos to target_blue / 200km  (LVLH)
  [3:6]   rel_vel to target_blue * 10     (m/s)
  [6]     dist / 20km
  [7]     solar_angle(sun, target_blue, self) / pi  (顶点=目标蓝星)
  [8:11]  sun_dir in target_blue LVLH
  [11]    dv_ratio
  [12]    time_progress
  [13]    0  (第二阶段可填红HV到蓝星的距离作为全局威胁)
  [14:17] 0

成功条件（红护卫侦照蓝星）：
  dist ≤ 20km, solar_angle ≤ 60°, 持续 ≥ 200s (1步)
"""

from __future__ import annotations
import numpy as np
import torch
import gymnasium
from copy import deepcopy
from dataclasses import asdict
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.resources.preprocessors.torch import RunningStandardScaler

try:
    from oge_py._oge_py_ma import MultiAgentOGEEnv as _CppEnv
    from oge_py._oge_py import SatState
except ImportError:
    from _oge_py_ma import MultiAgentOGEEnv as _CppEnv
    from _oge_py import SatState

from modules.networks import Policy

# ── 常量 ──────────────────────────────────────────────────────────────────────
MU = 398600.4418

TASK_STRIKE  = 0
TASK_RECON   = 1
TASK_JAM     = 2
TASK_OPERATE = 3

HV_IDX       = 0
RED_ESC_IDXS = [1, 2, 3, 4, 5, 6]   # global index of red escorts
BLUE_IDXS    = [7, 8, 9, 10, 11, 12] # global index of blue stars

# 蓝星任务类型（用于 C++ set_task_assignment，红护卫的 obs 计算）
BLUE_TASK_TYPES = [
    TASK_STRIKE,   # 蓝1 (global=7)
    TASK_STRIKE,   # 蓝2 (global=8)
    TASK_JAM,      # 蓝3 (global=9)
    TASK_RECON,    # 蓝4 (global=10)
    TASK_RECON,    # 蓝5 (global=11)
    TASK_OPERATE,  # 蓝6 (global=12)
]

# 红护卫燃料（与对应蓝星燃料对等）
# 红1/2 追 蓝1/2（打击，30m/s），给 30m/s；其余追 20m/s 的蓝星，给 20m/s
RED_ESC_DV_INIT = 0.020   # 默认值，实际由 RED_ESC_DV_LIST 按编号分配
RED_ESC_DV_LIST = [0.030, 0.030, 0.020, 0.020, 0.020, 0.020]

# 各任务蓝星燃料（用于 reset 时给蓝星分配燃料）
BLUE_DV_INIT = [0.030, 0.030, 0.020, 0.020, 0.020, 0.020]


# ── 轨道工具 ──────────────────────────────────────────────────────────────────

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


# ── 蓝星策略加载工具 ──────────────────────────────────────────────────────────

def load_blue_policy(ckpt_path: str, obs_dim: int, action_dim: int, device):
    """加载蓝星策略网络和 state_preprocessor。"""
    import gymnasium as gym
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                shape=(action_dim,), dtype=np.float32)
    policy = Policy(observation_space=obs_space, action_space=act_space,
                    device=device, dv_max=0.002, clip_actions=False)
    prep = RunningStandardScaler(size=obs_dim, device=device)

    ckpt = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    prep.load_state_dict(ckpt["state_preprocessor"])
    policy.to(device).eval()
    prep.eval()
    return policy, prep


# ── 基类 ──────────────────────────────────────────────────────────────────────

class _RedEscBaseWrapper(Wrapper):
    """
    红方护卫单星训练基类。每个实例只训练一颗红护卫，对应一颗蓝星。

    子类必须定义：
      AGENT_IDX   : int   红护卫全局 index（1~6）
      BLUE_IDX    : int   对应蓝星全局 index（7~12）
      BLUE_TASK   : int   对应蓝星的 C++ TaskType（用于 obs 计算的角度类型）
    """

    AGENT_IDX : int = 1   # red escort global index
    BLUE_IDX  : int = 7   # target blue star global index
    BLUE_TASK : int = TASK_RECON

    # 任务类型（镜像对应蓝星，决定 obs[7] 的语义和成功条件）
    TASK_TYPE        : int   = TASK_RECON
    SUCCESS_DIST_KM  : float = 20.0
    SUCCESS_ANGLE_DEG: float = 60.0   # RECON:60° / STRIKE:90° / JAM:5° / OPERATE:不用
    SUCCESS_DURATION : float = 200.0  # s

    OBS_DIM = 17

    def __init__(self, env_cfg, red_hv_oe: dict,
                 red_esc_oe_list: list, blue_oe_list: list,
                 blue_ckpt_paths: list,   # 6个蓝星的ckpt路径，None=零推力
                 ):
        self._red_hv_oe       = red_hv_oe
        self._red_esc_oe_list = red_esc_oe_list
        self._blue_oe_list    = blue_oe_list
        self._dv_max          = float(env_cfg.dv_max_per_step_blue)  # 红护卫用blue参数
        self._timestep        = float(env_cfg.timestep)

        # 预计算 ECI 初始状态
        self._r_hv, self._v_hv = _coe2rv(**red_hv_oe)
        self._esc_states = [_coe2rv(**oe) for oe in red_esc_oe_list]
        self._blue_rv    = [_coe2rv(**oe) for oe in blue_oe_list]

        # C++ 后端
        cfg_dict = asdict(env_cfg)
        cfg_dict["jd_epoch"] = 2460264.770833   # BJT 2023-11-16 22:30
        self._oge = _CppEnv(cfg_dict,
                            num_evaders=7,
                            num_pursuers=6,
                            intercept_distance=0.1)

        # 任务分配：
        # HV(0): 无任务; 红护卫(1~6): 各自侦照对应蓝星; 蓝星(7~12): 各自任务
        assignments = [
            {"task_type": TASK_RECON, "target_idx": 0, "threat_idx": -1},  # HV
        ]
        for k in range(6):  # 红护卫 1~6，目标=对应蓝星
            blue_target = BLUE_IDXS[k]   # 7,8,9,10,11,12
            assignments.append({
                "task_type":  BLUE_TASK_TYPES[k],  # 镜像蓝星任务类型，决定 obs[7] 语义
                "target_idx": blue_target,          # 红护卫k+1 → 蓝星k (global index 7~12)
                "threat_idx": -1,
            })
        for k, t in enumerate(BLUE_TASK_TYPES):  # 蓝星 1~6，目标=红HV
            assignments.append({
                "task_type":  t,
                "target_idx": HV_IDX,
                "threat_idx": -1,
            })
        self._oge.set_task_assignment(assignments)

        # 加载蓝星策略（6颗）
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._blue_device = torch.device(device_str)
        self._blue_policies = []
        self._blue_preps    = []
        for i, ckpt_path in enumerate(blue_ckpt_paths):
            if ckpt_path is not None and os.path.exists(ckpt_path):
                pol, prep = load_blue_policy(ckpt_path, 17, 3, self._blue_device)
                self._blue_policies.append(pol)
                self._blue_preps.append(prep)
            else:
                self._blue_policies.append(None)
                self._blue_preps.append(None)

        # 预分配 GPU tensor 缓冲区，避免每步重复申请
        self._blue_obs_buf = torch.zeros(1, 17, dtype=torch.float32,
                                         device=self._blue_device)
        # 预分配 combined actions numpy 数组
        self._combined = np.zeros((13, 3), dtype=np.float64)

        # episode 状态
        self._last_dist  = 200.0
        self._acc_time   = 0.0
        self._in_zone    = False
        self._last_raw   = None   # 上一步的 task_obs，用于蓝星推理

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
        """13 个 agent 初始状态：HV(无燃料) + 6红护卫(本星有燃料) + 6蓝星(有燃料)。"""
        states = {"blue_sat_0": _make_state(self._r_hv, self._v_hv, 0.0)}
        for k, (r, v) in enumerate(self._esc_states):
            # 按编号分配燃料：红护卫1/2追30m/s蓝星，给30m/s；其余给20m/s
            dv = RED_ESC_DV_LIST[k] if (k + 1) == self.AGENT_IDX else 0.0
            states[f"blue_sat_{k+1}"] = _make_state(r, v, dv)
        for k, (r, v) in enumerate(self._blue_rv):
            states[f"red_sat_{k}"] = _make_state(r, v, BLUE_DV_INIT[k])
        return states

    def _blue_action(self, blue_k: int, raw_obs: np.ndarray) -> np.ndarray:
        """获取第 blue_k 颗蓝星（0~5）的动作，复用预分配 buffer。"""
        pol  = self._blue_policies[blue_k]
        prep = self._blue_preps[blue_k]
        if pol is None:
            return np.zeros(3, dtype=np.float64)
        gi = BLUE_IDXS[blue_k]
        # 直接写入预分配 buffer，避免重新申请 tensor
        self._blue_obs_buf[0].copy_(torch.from_numpy(raw_obs[gi]))
        with torch.no_grad():
            obs_norm = prep(self._blue_obs_buf)
            act, _, _ = pol.act({"states": obs_norm}, role="policy")
        return (act.squeeze().cpu().numpy() * 0.002).astype(np.float64)

    def reset(self, seed=None, options=None):
        self._oge.reset_with_states(self._build_states())
        raw = np.asarray(self._oge.get_task_observations(), dtype=np.float32)
        self._last_raw  = raw
        obs = raw[self.AGENT_IDX]
        self._last_dist = float(obs[6]) * 20.0
        self._acc_time  = 0.0
        self._in_zone   = False
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device), {}

    def step(self, actions):
        # 复用预分配 combined 数组
        self._combined[:] = 0.0

        # 红护卫 RL 动作
        act_np = actions.squeeze().cpu().numpy() * self._dv_max
        self._combined[self.AGENT_IDX] = act_np

        # 蓝星策略动作（复用 buffer，避免重复申请 tensor）
        if self._last_raw is not None:
            for k in range(6):
                self._combined[BLUE_IDXS[k]] = self._blue_action(k, self._last_raw)

        self._oge.act(self._combined)
        raw = np.asarray(self._oge.get_task_observations(), dtype=np.float32)
        self._last_raw = raw
        obs = raw[self.AGENT_IDX]

        truncated = bool(self._oge.is_truncated())
        rew, done = self._compute_reward(obs, act_np)

        obs_t   = torch.tensor(obs,  dtype=torch.float32).unsqueeze(0).to(self.device)
        rew_t   = torch.tensor([[rew]], dtype=torch.float32).to(self.device)
        term_t  = torch.tensor([[done]], dtype=torch.bool).to(self.device)
        trunc_t = torch.tensor([[truncated and not done]], dtype=torch.bool).to(self.device)
        return obs_t, rew_t, term_t, trunc_t, {"current_time": self._oge.get_current_time()}

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray):
        """
        红护卫执行任务的奖励，根据 TASK_TYPE 区分：
          STRIKE/RECON : obs[7] = solar_angle / pi，成功条件 dist≤20km & angle≤60°/90° & 持续≥200s
          JAM          : obs[7] = jamming_angle / pi，成功条件 dist≤20km & angle≤5° & 持续≥600s
          OPERATE      : obs[7] = rel_speed*10，成功条件 dist≤2km
        """
        dist_km   = float(obs[6]) * 20.0
        dv_ratio  = float(obs[11])
        action_ms = np.linalg.norm(action) * 1000.0

        reward = 0.0
        done   = False

        # OPERATE 任务只看距离
        if self.TASK_TYPE == TASK_OPERATE:
            in_zone = (dist_km <= self.SUCCESS_DIST_KM)
            if in_zone:
                if not self._in_zone:
                    reward += 100.0 + dv_ratio * 30.0
                    self._in_zone = True
                else:
                    reward += 5.0
                    self._acc_time += self._timestep
                    if self._acc_time >= self.SUCCESS_DURATION:
                        reward += 200.0 + dv_ratio * 50.0
                        done = True
                self._last_dist = dist_km
                return reward, done

            self._acc_time = 0.0
            self._in_zone  = False
            if dv_ratio <= 0.0:
                return -40.0, True
            reward += 2.0 * (self._last_dist - dist_km)
            reward -= 0.01 * action_ms
            self._last_dist = dist_km
            return reward, done

        # STRIKE / RECON / JAM：obs[7] 都是角度 / pi
        angle_rad = float(obs[7]) * np.pi
        angle_deg = np.rad2deg(angle_rad)

        in_zone = (dist_km <= self.SUCCESS_DIST_KM and angle_deg <= self.SUCCESS_ANGLE_DEG)

        if in_zone:
            if not self._in_zone:
                reward += 100.0 + dv_ratio * 30.0
                self._acc_time = self._timestep
                self._in_zone  = True
            else:
                self._acc_time += self._timestep
                reward += 5.0
                if self._acc_time >= self.SUCCESS_DURATION:
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

        # 角度引导（根据任务类型区分）
        if self.TASK_TYPE == TASK_JAM:
            # JAM：两段式引导（照搬蓝方 JAM 成功经验）
            if dist_km <= self.SUCCESS_DIST_KM:  # 20km
                if dist_km > 5.0:
                    # 20~5km：轻度压角度，参考上限 90°
                    reward += 2.0 * (1.0 - angle_deg / 90.0)
                else:
                    # 5km 内：强力压角度到 5° 以内
                    reward += 5.0 * max(0.0, 1.0 - angle_deg / self.SUCCESS_ANGLE_DEG)
        else:
            # STRIKE/RECON：原逻辑，25km 内角度+速度引导
            if dist_km <= 25.0:
                if angle_deg <= self.SUCCESS_ANGLE_DEG:
                    reward += 2.0 * (1.0 - angle_deg / self.SUCCESS_ANGLE_DEG)
                rel_vel_ms = float(np.linalg.norm(obs[3:6]))
                target_vel_ms = 1.0 + (dist_km / 25.0) * 4.0
                if rel_vel_ms > target_vel_ms:
                    reward -= 0.2 * (rel_vel_ms - target_vel_ms)

        reward -= 0.01 * action_ms
        self._last_dist = dist_km
        return reward, done

    def render(self, *args, **kwargs): pass
    def close(self): pass


# ── 6 颗红护卫具体子类 ────────────────────────────────────────────────────────

class RedEsc1Wrapper(_RedEscBaseWrapper):
    """红护卫1：侦照蓝1（打击星），太阳角≤90°，持续≥200s。"""
    AGENT_IDX = 1; BLUE_IDX = 7;  BLUE_TASK = TASK_STRIKE
    TASK_TYPE = TASK_STRIKE; SUCCESS_DIST_KM = 20.0; SUCCESS_ANGLE_DEG = 90.0; SUCCESS_DURATION = 200.0

class RedEsc2Wrapper(_RedEscBaseWrapper):
    """红护卫2：侦照蓝2（打击星），太阳角≤90°，持续≥200s。"""
    AGENT_IDX = 2; BLUE_IDX = 8;  BLUE_TASK = TASK_STRIKE
    TASK_TYPE = TASK_STRIKE; SUCCESS_DIST_KM = 20.0; SUCCESS_ANGLE_DEG = 90.0; SUCCESS_DURATION = 200.0

class RedEsc3Wrapper(_RedEscBaseWrapper):
    """红护卫3：干扰蓝3（干扰星），干扰角≤5°，持续≥600s。"""
    AGENT_IDX = 3; BLUE_IDX = 9;  BLUE_TASK = TASK_JAM
    TASK_TYPE = TASK_JAM; SUCCESS_DIST_KM = 20.0; SUCCESS_ANGLE_DEG = 5.0; SUCCESS_DURATION = 600.0

class RedEsc4Wrapper(_RedEscBaseWrapper):
    """红护卫4：侦照蓝4（侦照星），太阳角≤60°，持续≥200s。"""
    AGENT_IDX = 4; BLUE_IDX = 10; BLUE_TASK = TASK_RECON
    TASK_TYPE = TASK_RECON; SUCCESS_DIST_KM = 20.0; SUCCESS_ANGLE_DEG = 60.0; SUCCESS_DURATION = 200.0

class RedEsc5Wrapper(_RedEscBaseWrapper):
    """红护卫5：侦照蓝5（侦照星），太阳角≤60°，持续≥200s。"""
    AGENT_IDX = 5; BLUE_IDX = 11; BLUE_TASK = TASK_RECON
    TASK_TYPE = TASK_RECON; SUCCESS_DIST_KM = 20.0; SUCCESS_ANGLE_DEG = 60.0; SUCCESS_DURATION = 200.0

class RedEsc6Wrapper(_RedEscBaseWrapper):
    """红护卫6：操控蓝6（操控星），dist≤2km。"""
    AGENT_IDX = 6; BLUE_IDX = 12; BLUE_TASK = TASK_OPERATE
    TASK_TYPE = TASK_OPERATE; SUCCESS_DIST_KM = 2.0; SUCCESS_ANGLE_DEG = 180.0; SUCCESS_DURATION = 200.0


# ── 工厂函数 ──────────────────────────────────────────────────────────────────

import os

_RED_ESC_SPEC = [
    RedEsc1Wrapper,   # esc1 → 蓝1
    RedEsc2Wrapper,   # esc2 → 蓝2
    RedEsc3Wrapper,   # esc3 → 蓝3
    RedEsc4Wrapper,   # esc4 → 蓝4
    RedEsc5Wrapper,   # esc5 → 蓝5
    RedEsc6Wrapper,   # esc6 → 蓝6
]


def make_red_wrapper(esc_idx: int, env_cfg,
                     red_hv_oe: dict,
                     red_esc_oe_list: list,
                     blue_oe_list: list,
                     blue_ckpt_paths: list) -> _RedEscBaseWrapper:
    """
    创建指定红护卫的训练 Wrapper。

    Parameters
    ----------
    esc_idx        : 0~5，对应红护卫 1~6
    env_cfg        : OGEEnvCfg
    red_hv_oe      : 红 HV 轨道根数
    red_esc_oe_list: 6 颗红护卫轨道根数列表
    blue_oe_list   : 6 颗蓝星轨道根数列表
    blue_ckpt_paths: 6 个路径（str or None），None = 零推力
    """
    cls = _RED_ESC_SPEC[esc_idx]
    return cls(env_cfg, red_hv_oe, red_esc_oe_list, blue_oe_list, blue_ckpt_paths)
