"""
护卫侦照场景环境包装器
======================
三星场景：
  evader[0] = blue_sat_0  → 红色高价值星 (Red HV)   — 始终无机动
  evader[1] = blue_sat_1  → 红色护卫星   (Red Esc)  — 训练红方时为 RL agent
  pursuer[0] = red_sat_0  → 蓝色侦照星   (Blue Recon) — 训练蓝方时为 RL agent

train_blue=True  : 蓝色侦照星为 RL，红护卫固定（无机动或加载 checkpoint）
train_blue=False : 红色护卫星为 RL，蓝色侦照星加载 checkpoint 运行

观测（17 维，两种模式共用）：
  [0:3]   rel_pos to target / 200 km  (LVLH)
  [3:6]   rel_vel to target * 10      (m/s)
  [6]     dist_to_target / 20 km
  [7]     solar_angle / pi
  [8:11]  sun_dir in HV LVLH
  [11]    dv_ratio
  [12]    time_progress
  [13]    dist_to_threat / 20 km
  [14:17] rel_pos_to_threat / 200 km  (LVLH)
"""

from __future__ import annotations
import numpy as np
import torch
import gymnasium
from dataclasses import asdict
from skrl.envs.wrappers.torch.base import Wrapper

try:
    from oge_py._oge_py_ma import MultiAgentOGEEnv as _CppEnv
    from oge_py._oge_py import SatState
except ImportError:
    from _oge_py_ma import MultiAgentOGEEnv as _CppEnv
    from _oge_py import SatState

MU = 398600.4418
AU = 149597870.691


# ── 轨道力学工具 ──────────────────────────────────────────────────────────────

def _ma2ta(M, e, tol=1e-10):
    E = M if e < 0.8 else np.pi
    for _ in range(100):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    ta = 2.0 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    return ta % (2*np.pi)


def _coe2rv(a, e, i, raan, w, M):
    ta = _ma2ta(M, e)
    h = np.sqrt(a * MU * (1 - e*e))
    r_mag = (h*h/MU) / (1 + e*np.cos(ta))
    rp = r_mag * np.array([np.cos(ta), np.sin(ta), 0.0])
    vp = (MU/h) * np.array([-np.sin(ta), e+np.cos(ta), 0.0])
    def Rz(a_): return np.array([[np.cos(a_),-np.sin(a_),0],[np.sin(a_),np.cos(a_),0],[0,0,1]])
    def Rx(a_): return np.array([[1,0,0],[0,np.cos(a_),-np.sin(a_)],[0,np.sin(a_),np.cos(a_)]])
    Q = Rz(raan) @ Rx(i) @ Rz(w)
    return Q @ rp, Q @ vp


def _solar_pos_j2000(jd):
    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + 36000.76983*T + 0.0003032*T*T) % 360.0
    M_deg = (357.52911 + 35999.05029*T - 0.0001537*T*T) % 360.0
    M_rad = np.deg2rad(M_deg)
    e_orb = 0.016708634 - 0.000042037*T
    C = (1.914602 - 0.004817*T)*np.sin(M_rad) + 0.019993*np.sin(2*M_rad) + 0.000289*np.sin(3*M_rad)
    lam = np.deg2rad(L0 + C)
    v_rad = np.deg2rad(M_deg + C)
    R_km = AU * 1.000001018 * (1-e_orb**2) / (1+e_orb*np.cos(v_rad))
    eps = np.deg2rad(23.439291 - 0.0130042*T)
    return np.array([R_km*np.cos(lam), R_km*np.cos(eps)*np.sin(lam), R_km*np.sin(eps)*np.sin(lam)])


# ── obs 索引工具（与 env_wrapper_ma.py 一致） ─────────────────────────────────

def _blk(k):      return 6 + 7*k
def _base(N):     return 6 + 7*(N-1)

def _other_k(agent_i, agent_j, N):
    k = 0
    for j in range(N):
        if j == agent_i: continue
        if j == agent_j: return k
        k += 1
    raise ValueError


# ── 主 Wrapper ────────────────────────────────────────────────────────────────

class EscortReconWrapper(Wrapper):
    """
    Parameters
    ----------
    env_cfg         : OGEEnvCfg dataclass
    jd_epoch        : Julian Day epoch（BJT 2027-09-01 20:00 → 2461650.0）
    red_hv_oe       : dict(a,e,i,raan,peri,M) 红HV轨道根数
    red_esc_oe      : dict(a,e,i,raan,peri,M) 红护卫轨道根数
    blue_dist_range : (min_km, max_km) 蓝星初始距离范围
    blue_sun_range  : (min_deg, max_deg) 蓝星初始太阳角范围
    train_blue      : True=训练蓝色侦照星，False=训练红色护卫星
    blue_policy     : 训练红方时传入蓝方策略网络（torch.nn.Module）
    blue_preprocessor: 训练红方时传入蓝方状态预处理器
    escort_intercept_dist : 护卫星拦截距离 km
    seed            : 随机种子
    """

    NUM_EVADERS  = 2   # blue_sat_0=RedHV, blue_sat_1=RedEsc
    NUM_PURSUERS = 1   # red_sat_0=BlueRecon
    N            = 3

    OBS_DIM = 17   # 13 基础 + 4 威胁

    RECON_DIST_KM  = 20.0
    RECON_ANGLE_DEG = 60.0
    RECON_DURATION  = 200.0   # s，等于一个时间步长

    def __init__(
        self,
        env_cfg,
        jd_epoch: float,
        red_hv_oe:  dict,
        red_esc_oe: dict,
        blue_dist_range: tuple = (190.0, 195.0),
        blue_sun_range:  tuple = (43.0, 47.0),
        train_blue: bool = True,
        blue_policy=None,
        blue_preprocessor=None,
        escort_intercept_dist: float = 20.0,
        esc_dv_init: float = 0.020,
        blue_oe: dict = None,          # Phase 2：固定蓝星初始六根数，None=随机采样
        seed: int = 42,
    ):
        self._train_blue   = train_blue
        self._jd_epoch     = jd_epoch
        self._red_hv_oe    = red_hv_oe
        self._red_esc_oe   = red_esc_oe
        self._dist_range   = blue_dist_range
        self._sun_range    = blue_sun_range
        self._blue_policy  = blue_policy
        self._blue_prep    = blue_preprocessor
        self._rng          = np.random.default_rng(seed)
        self._esc_dv_init  = float(esc_dv_init)
        self._blue_oe      = blue_oe   # None → 随机采样；dict → 固定初始化

        # 预计算红方固定 ECI 状态
        self._r_hv,  self._v_hv  = _coe2rv(**red_hv_oe)
        self._r_esc, self._v_esc = _coe2rv(**red_esc_oe)

        # 任务参数
        self._dv_max   = float(env_cfg.dv_max_per_step_red)   # 蓝方 or 红护卫
        self._dv_init  = float(env_cfg.dv_init_red)
        self._timestep = float(env_cfg.timestep)

        # C++ 后端
        cfg_dict = asdict(env_cfg)
        cfg_dict["jd_epoch"] = float(jd_epoch)
        self._oge = _CppEnv(cfg_dict, self.NUM_EVADERS, self.NUM_PURSUERS,
                            escort_intercept_dist)

        # 每 episode 状态
        self._last_dist  = 200.0
        self._recon_acc  = 0.0
        self._in_zone    = False
        self._last_raw   = None

        # skrl Wrapper 需要一个 dummy inner env
        _obs_dim = self.OBS_DIM
        class _Dummy:
            observation_space = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(_obs_dim,), dtype=np.float32)
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

    # ── 初始化：采样蓝星位置 ──────────────────────────────────────────────────

    def _sample_blue_state(self) -> tuple[np.ndarray, np.ndarray]:
        """在太阳角约束下采样蓝星初始 ECI 位置和速度。"""
        jd = self._jd_epoch
        pos_sun = _solar_pos_j2000(jd)
        hv_to_sun = pos_sun - self._r_hv
        hv_to_sun /= np.linalg.norm(hv_to_sun)

        # 构建以 hv_to_sun 为 z 轴的正交基
        tmp = np.array([1, 0, 0]) if abs(hv_to_sun[0]) < 0.9 else np.array([0, 1, 0])
        e1 = np.cross(hv_to_sun, tmp); e1 /= np.linalg.norm(e1)
        e2 = np.cross(hv_to_sun, e1)

        # 在 [sun_min, sun_max] 范围内采样太阳角 theta
        theta_min, theta_max = np.deg2rad(self._sun_range[0]), np.deg2rad(self._sun_range[1])
        theta = self._rng.uniform(theta_min, theta_max)
        phi   = self._rng.uniform(0, 2*np.pi)
        dist  = self._rng.uniform(self._dist_range[0], self._dist_range[1])

        # 方向向量（与 hv_to_sun 夹角 = theta）
        direction = (np.cos(theta)*hv_to_sun
                     + np.sin(theta)*np.cos(phi)*e1
                     + np.sin(theta)*np.sin(phi)*e2)

        r_blue = self._r_hv + dist * direction

        # 速度：近圆轨道，沿 HV 轨道面内切向
        h_hv = np.cross(self._r_hv, self._v_hv)
        h_hv /= np.linalg.norm(h_hv)
        r_norm = np.linalg.norm(r_blue)
        v_circ = np.sqrt(MU / r_norm)
        v_dir  = np.cross(h_hv, r_blue / r_norm)
        v_dir /= np.linalg.norm(v_dir)
        v_blue = v_circ * v_dir

        return r_blue, v_blue

    # ── 观测提取 ──────────────────────────────────────────────────────────────

    def _extract_obs(self, raw: np.ndarray, agent_gi: int,
                     target_gi: int, threat_gi: int) -> np.ndarray:
        """从 raw (N, raw_obs_dim) 提取 17 维观测。"""
        row  = raw[agent_gi]
        N    = self.N
        base = _base(N)

        k_tgt  = _other_k(agent_gi, target_gi, N)
        blk_t  = _blk(k_tgt)
        k_thr  = _other_k(agent_gi, threat_gi, N)
        blk_th = _blk(k_thr)

        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        obs[0:3]  = row[blk_t     : blk_t+3]   / 200.0   # rel_pos km
        obs[3:6]  = row[blk_t+3   : blk_t+6]   * 10.0    # rel_vel m/s
        obs[6]    = row[blk_t+6]                           # dist/20km
        obs[7]    = row[base+0]   / np.pi                  # solar_angle/pi
        obs[8:11] = row[base+4    : base+7]                # sun_dir
        obs[11]   = row[base+3]                            # dv_ratio
        obs[12]   = row[base+2]                            # time_progress
        obs[13]   = row[blk_th+6]                          # threat dist/20km
        obs[14:17]= row[blk_th    : blk_th+3]  / 200.0   # threat rel_pos
        return obs

    def _blue_obs(self, raw):
        # pursuer=2 → target=RedHV(0), threat=RedEsc(1)
        return self._extract_obs(raw, agent_gi=2, target_gi=0, threat_gi=1)

    def _esc_obs(self, raw):
        # evader[1]=1 → target=BlueRecon(2), threat=RedHV(0)
        # obs[7] = raw[1][base+0]：C++ 已提供 Blue 为顶点的太阳角
        return self._extract_obs(raw, agent_gi=1, target_gi=2, threat_gi=0)

    def _esc_solar_angle(self, raw) -> float:
        """从 C++ 观测直接读取 RedEsc 的太阳角（Blue 为顶点）。"""
        base = _base(self.N)
        return float(raw[1][base + 0])

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if self._blue_oe is not None:
            # Phase 2：固定六根数初始化蓝星
            r_bl, v_bl = _coe2rv(**self._blue_oe)
        else:
            # Phase 1：随机采样
            r_bl, v_bl = self._sample_blue_state()

        def _make_state(r, v, dv):
            s = SatState()
            s.r_j2000   = r.astype(np.float64)
            s.v_j2000   = v.astype(np.float64)
            s.dv_remain = float(dv)
            s.is_alive  = True
            return s

        states = {
            "blue_sat_0": _make_state(self._r_hv,  self._v_hv,  0.0),
            "blue_sat_1": _make_state(self._r_esc, self._v_esc,
                                      0.0 if self._train_blue else self._esc_dv_init),
            "red_sat_0":  _make_state(r_bl,        v_bl,        self._dv_init),
        }
        self._oge.reset_with_states(states)

        raw = np.asarray(self._oge.get_observations(), dtype=np.float64)
        self._last_raw = raw
        self._recon_acc      = 0.0
        self._in_zone        = False
        self._blue_recon_acc = 0.0   # Phase 2：追踪蓝星侦照红高的累计时间
        self._blue_in_zone   = False

        if self._train_blue:
            obs = self._blue_obs(raw)
            self._last_dist = raw[2][_blk(_other_k(2, 0, self.N)) + 6] * 20.0
        else:
            obs = self._esc_obs(raw)
            self._last_dist = raw[1][_blk(_other_k(1, 2, self.N)) + 6] * 20.0

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        return obs_t, {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, actions):
        N = self.N
        combined = np.zeros((N, 3), dtype=np.float64)
        act_np = actions.squeeze().cpu().numpy() * self._dv_max

        if self._train_blue:
            combined[2] = act_np          # 蓝侦照星
            combined[0] = 0.0             # 红HV：无机动
            combined[1] = 0.0             # 红护卫：第一阶段无机动
        else:
            combined[1] = act_np          # 红护卫
            combined[0] = 0.0             # 红HV：无机动
            combined[2] = self._blue_scripted_action()  # 蓝侦照星：加载策略

        self._oge.act(combined)
        raw = np.asarray(self._oge.get_observations(), dtype=np.float64)
        self._last_raw = raw

        truncated = bool(self._oge.is_truncated())

        if self._train_blue:
            obs  = self._blue_obs(raw)
            rew, done = self._reward_blue(raw, act_np)
        else:
            obs  = self._esc_obs(raw)
            rew, done = self._reward_esc(raw, act_np)

        obs_t  = torch.tensor(obs,  dtype=torch.float32).unsqueeze(0).to(self.device)
        rew_t  = torch.tensor([[rew]], dtype=torch.float32).to(self.device)
        term_t = torch.tensor([[done]], dtype=torch.bool).to(self.device)
        trunc_t= torch.tensor([[truncated and not done]], dtype=torch.bool).to(self.device)
        return obs_t, rew_t, term_t, trunc_t, {
            "current_time": self._oge.get_current_time(),
        }

    # ── 蓝方脚本动作（训练红方时使用） ───────────────────────────────────────

    def _blue_scripted_action(self) -> np.ndarray:
        """若有蓝方策略则推理，否则零推力。"""
        if self._blue_policy is None or self._last_raw is None:
            return np.zeros(3, dtype=np.float64)
        obs = self._blue_obs(self._last_raw)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self._blue_prep is not None:
                obs_t = self._blue_prep(obs_t)
            act = self._blue_policy.act({"states": obs_t}, role="policy")[0]
        return (act.squeeze().cpu().numpy() * self._dv_max).astype(np.float64)

    # ── 奖励：蓝色侦照星 ──────────────────────────────────────────────────────

    def _reward_blue(self, raw, action) -> tuple[float, bool]:
        row  = raw[2]
        N    = self.N
        base = _base(N)

        dist_km      = row[_blk(_other_k(2, 0, N)) + 6] * 20.0
        solar_angle  = row[base + 0]
        dv_rem       = row[base + 1]
        dist_esc_km  = row[_blk(_other_k(2, 1, N)) + 6] * 20.0
        action_ms    = np.linalg.norm(action) * 1000.0

        reward = 0.0
        done   = False

        # 侦照区判定
        in_zone = (dist_km <= self.RECON_DIST_KM
                   and np.rad2deg(solar_angle) <= self.RECON_ANGLE_DEG)

        if in_zone:
            if not self._in_zone:
                reward += 100.0 + (dv_rem / self._dv_init) * 30.0
                self._recon_acc = self._timestep
                self._in_zone   = True
            else:
                self._recon_acc += self._timestep
                reward += 5.0
                if self._recon_acc >= self.RECON_DURATION:
                    reward += 200.0 + (dv_rem / self._dv_init) * 50.0
                    done = True
            self._last_dist = dist_km
            return reward, done

        # 离开侦照区
        self._recon_acc = 0.0
        self._in_zone   = False

        # 终端：燃料耗尽（对齐 1v1：在 in_zone 之后判定）
        if dv_rem <= 0.0:
            return -40.0, True

        # 距离引导
        reward += 2.0 * (self._last_dist - dist_km)

        # 角度引导 + 速度漏斗（25km 内，对齐 1v1）
        if dist_km <= 25.0:
            angle_deg = np.rad2deg(solar_angle)
            if angle_deg <= self.RECON_ANGLE_DEG:
                reward += 2.0 * (1.0 - angle_deg / self.RECON_ANGLE_DEG)

            # 速度漏斗：25km允许5m/s，0km要求1m/s
            rel_vel_ms = np.linalg.norm(
                raw[2][_blk(_other_k(2, 0, self.N))+3 : _blk(_other_k(2, 0, self.N))+6]
            )  # C++ 已乘 1000，单位 m/s
            target_vel_ms = 1.0 + (dist_km / 25.0) * 4.0
            if rel_vel_ms > target_vel_ms:
                reward -= 0.2 * (rel_vel_ms - target_vel_ms)

        # 护卫威胁惩罚（距护卫 < 50km 时）
        if dist_esc_km < 50.0:
            reward -= 1.0 * (50.0 - dist_esc_km) / 50.0

        # 护卫进入 22km 内：每步额外惩罚
        if dist_esc_km < 22.0:
            reward -= 5.0

        # 落入红护卫侦察区惩罚：BlueRecon→Sun ∧ BlueRecon→RedEsc ≤ 60° 且 dist ≤ 20km
        if dist_esc_km <= self.RECON_DIST_KM:
            r_esc  = raw[1][0:3]   # RedEsc ECI km
            r_blue = raw[2][0:3]   # BlueRecon ECI km
            jd_now = self._jd_epoch + self._oge.get_current_time() / 86400.0
            pos_sun = _solar_pos_j2000(jd_now)
            blue_to_sun = pos_sun - r_blue;  blue_to_sun /= np.linalg.norm(blue_to_sun)
            blue_to_esc = r_esc   - r_blue;  blue_to_esc /= np.linalg.norm(blue_to_esc)
            esc_recon_angle_deg = np.rad2deg(
                np.arccos(np.clip(np.dot(blue_to_sun, blue_to_esc), -1.0, 1.0))
            )
            if esc_recon_angle_deg <= self.RECON_ANGLE_DEG:
                reward -= 50.0

        # 燃料惩罚
        reward -= 0.01 * action_ms

        self._last_dist = dist_km
        return reward, done

    # ── 奖励：红色护卫星 ──────────────────────────────────────────────────────

    def _reward_esc(self, raw, action) -> tuple[float, bool]:
        """红护卫目标：在蓝星侦照红高成功之前，侦照蓝星（Blue基准60°，20km，200s）。"""
        row_esc  = raw[1]
        row_blue = raw[2]
        N        = self.N
        base     = _base(N)

        # ── 红护卫到蓝星的距离、自身燃料 ──────────────────────────────────────
        dist_km   = row_esc[_blk(_other_k(1, 2, N)) + 6] * 20.0
        solar_rad = self._esc_solar_angle(raw)   # Blue 为顶点
        dv_rem    = row_esc[base + 1]
        action_ms = np.linalg.norm(action) * 1000.0

        # ── 蓝星对红高的侦照状态（HVT 基准角，C++ 直接提供） ──────────────────
        blue_dist_hv_km   = row_blue[_blk(_other_k(2, 0, N)) + 6] * 20.0
        blue_solar_rad    = row_blue[base + 0]           # HVT→Sun ∧ HVT→Blue
        blue_in_zone_now  = (blue_dist_hv_km <= self.RECON_DIST_KM
                             and np.rad2deg(blue_solar_rad) <= self.RECON_ANGLE_DEG)

        # 追踪蓝星侦照进度
        if blue_in_zone_now:
            self._blue_recon_acc += self._timestep
            self._blue_in_zone    = True
        else:
            self._blue_recon_acc  = 0.0
            self._blue_in_zone    = False

        reward = 0.0
        done   = False

        # ── 终止条件 1：蓝星先完成侦照��高 → 护卫失败 ────────────────────────
        if self._blue_recon_acc >= self.RECON_DURATION:
            return -200.0, True

        # ── 终止条件 2：燃料耗尽 ──────────────────────────────────────────────
        if dv_rem <= 0.0:
            return -40.0, True

        # ── 护卫侦照区判定（Blue 为顶点） ─────────────────────────────────────
        in_zone = (dist_km <= self.RECON_DIST_KM
                   and np.rad2deg(solar_rad) <= self.RECON_ANGLE_DEG)

        if in_zone:
            if not self._in_zone:
                reward += 100.0 + (dv_rem / self._esc_dv_init) * 30.0
                self._recon_acc = self._timestep
                self._in_zone   = True
            else:
                self._recon_acc += self._timestep
                reward += 5.0
                # ── 终止条件 3：护卫成功侦照蓝星 → 成功 ──────────────────────
                if self._recon_acc >= self.RECON_DURATION:
                    reward += 200.0 + (dv_rem / self._esc_dv_init) * 50.0
                    done = True
            self._last_dist = dist_km
            return reward, done

        # 离开侦照区
        self._recon_acc = 0.0
        self._in_zone   = False

        # 距离引导
        reward += 2.0 * (self._last_dist - dist_km)

        # 角度引导 + 速度漏斗（25km 内，对齐 1v1）
        if dist_km <= 25.0:
            angle_deg = np.rad2deg(solar_rad)
            if angle_deg <= self.RECON_ANGLE_DEG:
                reward += 2.0 * (1.0 - angle_deg / self.RECON_ANGLE_DEG)

            rel_vel_ms = np.linalg.norm(
                raw[1][_blk(_other_k(1, 2, N))+3 : _blk(_other_k(1, 2, N))+6]
            )
            target_vel_ms = 1.0 + (dist_km / 25.0) * 4.0
            if rel_vel_ms > target_vel_ms:
                reward -= 0.2 * (rel_vel_ms - target_vel_ms)

        # 燃料惩罚
        reward -= 0.01 * action_ms

        self._last_dist = dist_km
        return reward, done

    def render(self, *args, **kwargs): pass
    def close(self): pass
