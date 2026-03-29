"""
侦照任务环境 (Reconnaissance Environment)
任务目标: 追击者(红)接近逃逸者(蓝)至 <=20km 且光照角 <=60°
逃逸者: 不机动，自然飘飞
追击者: 累积 ΔV <=20 m/s，单次 <=2 m/s，机动间隔 >=200s
初始时间: UTC+8 2027.09.02 00:00:00 = UTC 2027.09.01 16:00:00

参考轨道根数 (用户给定, 单位 km 和 rad):
  红(追): sma=42060.338261 km, e=0.003001, i=0.002287 rad,
          raan=1.592759 rad, argp=3.303797 rad, MA=1.033423 rad
  蓝(逃): sma=42169.502913 km, e=0,       i=0.002287 rad,
          raan=1.592829 rad, argp=0,          MA=4.345423 rad

两星初始距离计算:
  红 u = argp+TA ≈ 3.303797+1.0426 = 4.3464 rad, r ≈ 41994.9 km
  蓝 u = 0+4.345423 = 4.3454 rad,                r ≈ 42169.5 km
  Δu ≈ 0.001 rad (相位差极小), ΔR ≈ 174.6 km (径向差)
  总距离 ≈ √(174.6² + 42²) ≈ 179 km

OrbitLib.coe2rv 接口: [sma(m), ecc, inc(rad), raan(rad), argp(rad), ta(rad)]
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces
from copy import copy

from .mpe_pomdp_env import MPE_POMDP_Env, MPE_POMDP_EnvCfg
from .orbitx import solar_position, time2jd, Keplers_Eqn


@dataclass
class ReconEnvCfg(MPE_POMDP_EnvCfg):
    """侦照任务环境配置"""
    # check_params 需要 dim_mode
    dim_mode: int = 2

    # 1v1
    num_p: int = 1
    num_e: int = 1

    # 初始时间: UTC+8 2027.09.02 00:00:00 = UTC 2027.09.01 16:00:00
    init_utc = datetime.datetime(2027, 9, 1, 16, 0, 0)

    # 成功条件
    dist_cap: float = 20000.0          # 20 km 成功判定阈值 (单位 m，与父类一致)
    target_depth_m: float = 10000.0    # 目标点深度: 逃逸者沿太阳方向后退 10 km
    lighting_angle_cap: float = 60.0   # degrees

    # 追击者约束
    p_init_dv: float = 20.0            # 20 m/s 总量
    p_dv_step: float = 2.0             # 单次最大 2 m/s
    min_maneuver_interval: float = 200.0  # 机动间隔 >=200s
    maneuver_deadzone: float = 0.05    # 动作死区: 低于此比例(×p_dv_step)视为不机动

    # 逃逸者: 不机动 (e_dv_step/e_init_dv 需 >0 以通过 check_params)
    e_init_dv: float = 0.1
    e_dv_step: float = 0.001
    evader_policy_level: int = 0

    # 奖励权重
    # r_approach: 基于距逃逸者本体距离的势函数shaping，主要引导
    reward_approach_weight: float = 0.5    # 靠近逃逸者1km → +0.5
    # r_lighting: 光照角奖励，分远近两段
    reward_lighting_far_weight: float = 0.05   # 20km外：轻微惩罚大角度
    reward_lighting_near_weight: float = 0.5   # 20km内：强力引导顺光
    # r_fuel_save: 滑行激励（不机动时给小正奖励）
    reward_fuel_save: float = 0.02
    # r_time: 极小时间惩罚，防磨洋工
    reward_time_weight: float = 0.005
    # 以下保留兼容父类接口，但实际不用
    reward_phase_dist_weight: float = 0.0  # 关闭：远场引导力度不足
    reward_fuel_weight: float = 0.0
    reward_formation_weight: float = 0.0
    reward_advantage_weight: float = 0.0
    reward_lighting_weight: float = 0.5    # 兼容旧接口
    reward_target_weight: float = 0.0      # 关闭虚拟星shaping，改用逃逸者本体
    capture_reward: float = 20.0
    reward_timeout_penalty: float = -20.0
    reward_fuelout_penalty: float = -20.0

    # ---------------------------------------------------------------
    # 参考轨道根数 (用户给定, sma 单位 km, 角度单位 rad)
    # OrbitLib.coe2rv 需要 sma 单位 m，所以存储时保持 km，使用时 *1000
    # ---------------------------------------------------------------
    # 蓝(逃逸者)
    ref_evader_sma_km: float = 42169.502913   # km
    ref_evader_ecc: float = 0.0
    ref_evader_incl: float = 0.002287         # rad
    ref_evader_raan: float = 1.592829         # rad
    ref_evader_argp: float = 0.0              # rad
    ref_evader_ma: float = 4.345423           # rad (MA, e=0 所以 TA=MA)

    # 红(追击者)
    ref_pursuer_sma_km: float = 42060.338261  # km
    ref_pursuer_ecc: float = 0.003001
    ref_pursuer_incl: float = 0.002287        # rad
    ref_pursuer_raan: float = 1.592759        # rad
    ref_pursuer_argp: float = 3.303797        # rad
    ref_pursuer_ma: float = 1.033423          # rad (MA)

    # ---------------------------------------------------------------
    # 课程学习距离参数
    # 含义: 追击者距逃逸者的初始距离 (圆环半径)
    #   初始 30km (刚好在成功区域外) → 目标 206km (参考轨道距逃逸者距离)
    #   成功条件: 距逃逸者 <=20km 且光照角 <=60°
    # ---------------------------------------------------------------
    init_distance_m: float = 30000.0
    ring_width_delta: float = 5000.0
    # 同步到 reset 直接读取的字段
    e_init_dist_min_offset: float = 30000.0
    e_init_dist_max_offset: float = 35000.0

    # episode 时长
    episode_length: float = 3600.0 * 5   # 5 小时

    # 全量观测 (非 POMDP)
    use_partial_obs: bool = False
    obs_interval: int = 1
    history_len: int = 1

    # SMA 扰动课程
    sma_perturb_start_update: int = 500
    sma_perturb_end_update: int = 1500
    sma_perturb_km_max: float = 5.0

    # GEO 参考半径 (m)
    GEO_ORBIT_RADIUS: float = 42164000.0


class ReconEnv(MPE_POMDP_Env):
    """
    侦照任务环境。继承 MPE_POMDP_Env，复用 POMDP 训练基础设施。

    主要改动:
      1. reset(): 初始位置基于参考轨道根数生成，训练时加扰动，评估时固定
      2. step(): 机动间隔约束 (>=200s)
      3. _compute_lighting_angle(): 用 orbitx.solar_position 计算光照角
      4. _get_observations(): 追加 cos(光照角) 特征
      5. _get_rewards(): 加入光照角奖励
      6. _get_terminations(): 距离 <=20km 且光照角 <=60° 才算成功

    两星参考距离约 216.33 km:
      径向差 ΔR ≈ 173.5 km，切向弧长 ≈ 129.2 km
      目标点(最优侦照位置)距逃逸者 20 km，故参考轨道距目标点 ≈ 196 km
    """

    def __init__(self, config: ReconEnvCfg = ReconEnvCfg()):
        super().__init__(config)
        self._config: ReconEnvCfg = config

        # 机动间隔计数器 (步数)
        self.steps_since_last_maneuver: dict[str, int] = {}

        # 覆盖观测空间: base(9) + lighting(1) + virtual_rel(3) + fuel_ratio(1) + cooldown_ready(1) = 15D
        for a in self.possible_agents:
            if a.startswith('p_'):
                base_dim = 6 + 3 * self._config.num_e + 3 * (self._config.num_p - 1)
                self.observation_spaces[a] = spaces.Box(
                    -np.inf, np.inf, shape=(base_dim + 6,)
                )

    # ------------------------------------------------------------------
    # 辅助: MA -> TA (Kepler 方程)
    # ------------------------------------------------------------------
    @staticmethod
    def _ma_to_ta(MA: float, e: float) -> float:
        EA = Keplers_Eqn(MA, e)
        y = np.sqrt(max(0.0, 1.0 - e * e)) * np.sin(EA)
        x = np.cos(EA) - e
        return float(np.arctan2(y, x))

    # ------------------------------------------------------------------
    # 辅助: 光照角计算
    # 太阳方向与逃逸者->追击者方向的夹角 (度)
    # 追击者在向阳面时夹角=0°，<=60° 表示适合侦照
    # ------------------------------------------------------------------
    def _compute_lighting_angle(self) -> float:
        if 'p_0' not in self.states or 'e_0' not in self.states:
            return 90.0

        jd = time2jd(self._time)
        r_sun_km = solar_position(jd)          # km, ECI J2000

        p_pos_km = self.states['p_0'][:3] / 1000.0
        e_pos_km = self.states['e_0'][:3] / 1000.0

        u_sun = r_sun_km / (np.linalg.norm(r_sun_km) + 1e-12)

        # 逃逸者→追击者方向：追击者在向阳面时与 u_sun 同向，夹角=0°
        obs_vec  = p_pos_km - e_pos_km
        obs_norm = np.linalg.norm(obs_vec)
        if obs_norm < 1e-6:
            return 180.0
        u_obs = obs_vec / obs_norm

        cos_a = float(np.clip(np.dot(u_sun, u_obs), -1.0, 1.0))
        return float(np.rad2deg(np.arccos(cos_a)))

    # ------------------------------------------------------------------
    # 辅助: 虚拟星位置 = 逃逸者沿太阳方向后退 target_depth_m (10 km)
    # 追击者追这个点，光照角=0°时恰好在此处
    # ------------------------------------------------------------------
    def _get_virtual_target_pos(self) -> np.ndarray:
        if 'e_0' not in self.states:
            return np.zeros(3)
        jd = time2jd(self._time)
        r_sun_km = solar_position(jd)
        u_sun = r_sun_km / (np.linalg.norm(r_sun_km) + 1e-12)
        e_pos = self.states['e_0'][:3]          # m
        return e_pos - self._config.target_depth_m * u_sun

    # 兼容旧调用
    def _compute_target_point(self) -> np.ndarray:
        return self._get_virtual_target_pos()

    # ------------------------------------------------------------------
    # reset: 基于参考轨道根数生成初始位置
    #
    # 训练模式 (seed=None): 在参考位置附近加随机扰动，有课程学习
    # 评估模式 (use_fixed_seed_for_reset=True 或 seed 固定):
    #   直接使用参考轨道根数，不加扰动
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        cfg = self._config

        # 评估模式: 固定种子 -> 直接用参考轨道根数
        is_eval = cfg.use_fixed_seed_for_reset or (seed is not None)
        if is_eval:
            np.random.seed(seed if seed is not None else 42)

        self.agents = copy(self.possible_agents)
        self.states = {}

        # ---------------------------------------------------------------
        # 1. 逃逸者 (蓝星)
        # OrbitLib.coe2rv: [sma(m), ecc, inc(rad), raan(rad), argp(rad), ta(rad)]
        # ---------------------------------------------------------------
        eva_sma_m = cfg.ref_evader_sma_km * 1000.0
        eva_ecc   = cfg.ref_evader_ecc
        eva_incl  = cfg.ref_evader_incl
        eva_raan  = cfg.ref_evader_raan
        eva_argp  = cfg.ref_evader_argp
        # e=0 时 TA=MA
        eva_ta_ref = cfg.ref_evader_ma

        # e=0 时 TA=MA，训练和评估都用固定参考位置
        eva_ta = eva_ta_ref

        # SMA 不扰动，逃逸者始终从参考轨道出发自然飘飞

        self.states['e_0'] = self._orbit_lib.coe2rv(
            np.array([eva_sma_m, eva_ecc, eva_incl, eva_raan, eva_argp, eva_ta])
        )

        # ---------------------------------------------------------------
        # 2. 追击者 (红星)
        # ---------------------------------------------------------------
        if is_eval:
            # 评估: 直接用参考轨道根数
            pur_sma_m  = cfg.ref_pursuer_sma_km * 1000.0
            pur_ecc    = cfg.ref_pursuer_ecc
            pur_incl   = cfg.ref_pursuer_incl
            pur_raan   = cfg.ref_pursuer_raan
            pur_argp   = cfg.ref_pursuer_argp
            pur_ta     = self._ma_to_ta(cfg.ref_pursuer_ma, pur_ecc)

            self.states['p_0'] = self._orbit_lib.coe2rv(
                np.array([pur_sma_m, pur_ecc, pur_incl, pur_raan, pur_argp, pur_ta])
            )
        else:
            # 训练: 在逃逸者周围圆环上生成，课程控制圆环半径 (30km → 206km)
            # 圆环半径 = [e_init_dist_min_offset, e_init_dist_max_offset]
            # 用 2D 轨道平面近似: 径向分量→sma_offset，切向分量→ta_offset
            perturb_dist = np.random.uniform(
                cfg.e_init_dist_min_offset, cfg.e_init_dist_max_offset
            )
            angle = np.random.uniform(0, 2 * np.pi)
            sma_offset = perturb_dist * np.sin(angle)
            ta_offset  = (perturb_dist / (eva_sma_m + 1e-6)) * np.cos(angle)

            pur_sma_m = eva_sma_m + sma_offset
            pur_ta    = (eva_ta + ta_offset) % (2 * np.pi)

            pur_ecc  = 0.0
            pur_incl = cfg.ref_pursuer_incl
            pur_raan = cfg.ref_pursuer_raan
            pur_argp = 0.0

            if self.current_sma_perturb_km > 0:
                pur_sma_m += np.random.uniform(
                    -self.current_sma_perturb_km * 1000,
                    self.current_sma_perturb_km * 1000
                )

            self.states['p_0'] = self._orbit_lib.coe2rv(
                np.array([pur_sma_m, pur_ecc, pur_incl, pur_raan, pur_argp, pur_ta])
            )

        # ---------------------------------------------------------------
        # 3. 重置时间、燃料、计数器
        # ---------------------------------------------------------------
        self._time = cfg.init_utc
        self.remain_Dvs = {
            'p_0': cfg.p_init_dv,
            'e_0': cfg.e_init_dv,
        }
        self.steps_since_last_maneuver = {'p_0': 9999}
        # shaping: 记录上一步到目标点的距离，用于计算靠近/远离奖励
        self._prev_dist_to_target: dict[str, float] = {}

        self.terminations = {a: False for a in self.agents}
        self.truncations  = {a: False for a in self.agents}
        self.current_episode_seed = seed if seed is not None else np.random.randint(0, 1_000_000_000)
        self.current_formation_name = 'recon_1v1'

        if self.viewer is not None:
            self.viewer.reset()

        # ---------------------------------------------------------------
        # 4. 初始化观测和 infos
        # ---------------------------------------------------------------
        observations = self._get_observations()
        privileged_state = self._get_privileged_state()
        self.infos = {a: {} for a in self.agents}
        self.infos['p_0']['privileged_state'] = privileged_state
        self.infos['p_0']['episode_statistics'] = self.episode_statistics.copy()

        return observations, self.infos

    # ------------------------------------------------------------------
    # step: 加入机动间隔约束 (>=200s)
    # ------------------------------------------------------------------
    def step(self, actions: dict):
        for pid in self.pursuer_ids:
            if pid not in actions:
                continue
            steps_elapsed = self.steps_since_last_maneuver.get(pid, 9999)
            if steps_elapsed * self._config.dt < self._config.min_maneuver_interval:
                # 冷却中，清零动作
                actions[pid] = np.zeros(3)
            else:
                # 动作死区: 幅度低于 deadzone × p_dv_step 视为不机动
                deadzone = self._config.maneuver_deadzone * self._config.p_dv_step
                if np.linalg.norm(actions[pid]) > deadzone:
                    self.steps_since_last_maneuver[pid] = 0
                else:
                    actions[pid] = np.zeros(3)

        # 递增计数器
        for pid in self.pursuer_ids:
            self.steps_since_last_maneuver[pid] = (
                self.steps_since_last_maneuver.get(pid, 0) + 1
            )

        return super().step(actions)

    # ------------------------------------------------------------------
    # 观测: 全量观测 + cos(光照角) + 目标点相对方向
    # ------------------------------------------------------------------
    def _get_observations(self):
        observations = super()._get_observations()

        lighting_angle = self._compute_lighting_angle()
        lighting_feat  = np.array([np.cos(np.deg2rad(lighting_angle))], dtype=np.float32)

        for pid in self.pursuer_ids:
            if pid not in observations or pid not in self.states:
                continue

            p_pos = self.states[pid][:3]
            v_pos = self._get_virtual_target_pos()
            rel_to_virtual = self._symlog(v_pos - p_pos).astype(np.float32)

            # 剩余燃料比例 [0,1]
            fuel_ratio = np.array(
                [self.remain_Dvs.get(pid, 0.0) / max(self._config.p_init_dv, 1e-6)],
                dtype=np.float32
            )

            # 冷却是否就绪 (1=可机动, 0=冷却中)
            steps_elapsed = self.steps_since_last_maneuver.get(pid, 9999)
            cooldown_ready = np.array(
                [1.0 if steps_elapsed * self._config.dt >= self._config.min_maneuver_interval else 0.0],
                dtype=np.float32
            )

            observations[pid] = np.concatenate(
                [observations[pid], lighting_feat, rel_to_virtual, fuel_ratio, cooldown_ready]
            ).astype(np.float32)

        return observations

    # ------------------------------------------------------------------
    # 相位距离奖励: 基于追击者与虚拟星的距离/轨道关系
    # 虚拟星位置实时更新，追击者追虚拟星
    # ------------------------------------------------------------------
    def _get_phase_distance_reward(self, agent_id, target_id):
        # target_id 参数保留兼容性，但实际用虚拟星位置
        DIST_CAP = self._config.target_depth_m    # 10000 m，虚拟星就是目标
        TRANSITION_DIST = DIST_CAP * 6.0          # 60000 m

        p_state = self.states[agent_id]
        v_pos   = self._get_virtual_target_pos()  # 虚拟星位置 (m)
        dist    = np.linalg.norm(p_state[:3] - v_pos)

        # 用追击者和逃逸者的轨道参数判断漂移方向（远场引导）
        e_state = self.states[target_id]
        sma_p, theta_p = self._calculate_orbital_metrics(p_state)
        sma_e, theta_e = self._calculate_orbital_metrics(e_state)

        delta_theta   = (theta_p - theta_e + np.pi) % (2 * np.pi) - np.pi
        sma_diff_ratio = (sma_p - sma_e) / (sma_e + 1e-6)

        drift_product = -delta_theta * sma_diff_ratio
        if drift_product > 0:
            R_Far = -1.0 - np.abs(sma_diff_ratio) * 2000.0
        else:
            r_drift = np.clip(np.abs(sma_diff_ratio) * 1000.0, 0.0, 2.0)
            r_angle = (np.pi - np.abs(delta_theta)) / np.pi
            R_Far = 1.0 * r_drift + 0.5 * r_angle

        norm_dist = dist / DIST_CAP
        if norm_dist <= 1.0:
            R_dist = 1.0 + 0.1 * (1.0 - norm_dist)
        elif norm_dist <= 2.0:
            R_dist = 2.0 - norm_dist
        else:
            R_dist = np.clip(2.0 - norm_dist, -1.0, 0.0)

        R_energy = -np.abs(sma_diff_ratio) * 2000.0
        R_Near   = 1.0 * R_dist + 0.05 * R_energy

        alpha = np.tanh(dist / TRANSITION_DIST)
        return (alpha * R_Far + (1.0 - alpha) * R_Near) * 0.1

    # ------------------------------------------------------------------
    # 奖励: 分阶段设计，优先进入20km，再满足光照角
    #
    # 过程奖励:
    #   r_approach  : 距逃逸者本体的势函数shaping，主要引导，量级0~0.5/步
    #   r_lighting  : 光照角，20km外轻微惩罚，20km内强力引导，量级0~0.5/步
    #   r_fuel_save : 滑行激励（不机动时+0.02），引导脉冲+滑行策略
    #   r_time      : 极小时间惩罚(-0.005)，防磨洋工
    #
    # 漏洞修复: 父类只要距离<20km就发 capture_reward/10，不管光照角
    #   → 无条件扣除，再按真实侦照条件重新发
    #
    # 终端奖励 (直接发):
    #   侦照成功: +capture_reward * (1 + 0.5*quality)  ≈ +20~+30
    #   fuel_out: reward_fuelout_penalty               = -20
    #   timeout : reward_timeout_penalty               = -20
    # ------------------------------------------------------------------
    def _get_rewards(self, actions: dict):
        rewards, debug_info = super()._get_rewards(actions)

        lighting_angle = self._compute_lighting_angle()
        cap = self._config.lighting_angle_cap
        e_pos = self.states['e_0'][:3] if 'e_0' in self.states else np.zeros(3)

        for pid in self.pursuer_ids:
            if pid not in rewards or pid not in self.states:
                continue

            p_pos = self.states[pid][:3]
            dist_to_evader = float(np.linalg.norm(p_pos - e_pos))

            # --- 1. r_approach: 距逃逸者本体的势函数shaping ---
            # 靠近1km → +reward_approach_weight，远离1km → -reward_approach_weight
            dist_prev = self._prev_dist_to_target.get(pid, dist_to_evader)
            r_approach = self._config.reward_approach_weight * (dist_prev - dist_to_evader) / 1000.0
            self._prev_dist_to_target[pid] = dist_to_evader

            # --- 2. r_lighting: 分阶段光照角奖励 ---
            in_range = dist_to_evader <= self._config.dist_cap  # 20km内
            if in_range:
                # 20km内：强力引导顺光，angle=0→0，angle=cap→-w_near
                r_lighting = -self._config.reward_lighting_near_weight * (lighting_angle / cap)
            else:
                # 20km外：轻微惩罚大角度，防止完全忽略光照
                r_lighting = -self._config.reward_lighting_far_weight * (lighting_angle / 180.0)

            # --- 3. r_fuel_save: 滑行激励 ---
            # 判断本步是否机动（actions已经过deadzone处理，零向量=不机动）
            action_norm = np.linalg.norm(actions.get(pid, np.zeros(3)))
            r_fuel_save = self._config.reward_fuel_save if action_norm < 1e-6 else 0.0

            # --- 4. r_time: 极小时间惩罚 ---
            r_time = -self._config.reward_time_weight

            rewards[pid] += r_approach + r_lighting + r_fuel_save + r_time

        # 5. 无条件扣除父类因纯距离判定错发的 capture_bonus
        for pid in self.pursuer_ids:
            if pid in self.states:
                dist_to_evader = np.linalg.norm(self.states[pid][:3] - e_pos)
                if dist_to_evader < self._config.dist_cap:
                    rewards[pid] -= self._config.capture_reward / 10.0

        # 6. 终端奖励/惩罚，基于真实侦照条件（含光照角）
        terminations, reasons = self._get_terminations()

        for pid in self.pursuer_ids:
            if pid not in rewards:
                continue
            reason = reasons.get(pid)

            if reason == 'capture_success':
                quality = 1.0 - lighting_angle / cap
                rewards[pid] += self._config.capture_reward * (1.0 + 0.5 * quality)

            elif reason == 'fuel_out':
                rewards[pid] += self._config.reward_fuelout_penalty

            elif reason == 'timeout':
                rewards[pid] += self._config.reward_timeout_penalty

        return rewards, debug_info

    # ------------------------------------------------------------------
    # 终止条件: 独立实现，不依赖父类
    # 成功: 距真实逃逸者 <=20km 且光照角 <=60°
    # ------------------------------------------------------------------
    def _get_terminations(self):
        terminations = {a: False for a in self.agents}
        reasons      = {a: None  for a in self.agents}

        if 'e_0' not in self.states:
            return terminations, reasons

        lighting_angle = self._compute_lighting_angle()
        e_pos = self.states['e_0'][:3]

        for pid in self.pursuer_ids:
            if pid not in self.states:
                continue

            dist_to_evader = np.linalg.norm(self.states[pid][:3] - e_pos)

            # 1. 侦照成功: 距真实逃逸者 <=20km 且光照角 <=60°
            if (dist_to_evader <= self._config.dist_cap and
                    lighting_angle <= self._config.lighting_angle_cap):
                terminations[pid] = True
                reasons[pid] = 'capture_success'

            # 2. 燃料耗尽
            elif self.remain_Dvs.get(pid, 0) <= 0:
                terminations[pid] = True
                reasons[pid] = 'fuel_out'

            # 3. 超时
            elif self._time >= self._config.init_utc + datetime.timedelta(
                    seconds=self._config.episode_length):
                terminations[pid] = True
                reasons[pid] = 'timeout'

        # 任一智能体终止则全部终止
        if any(terminations.values()):
            primary_reason = reasons.get(self.pursuer_ids[0])
            for a in self.agents:
                terminations[a] = True
                if reasons[a] is None:
                    reasons[a] = primary_reason

        return terminations, reasons

    # ------------------------------------------------------------------
    # 课程学习接口 (兼容 train_pomdp.py 的 set_difficulty_parameters)
    # ------------------------------------------------------------------
    def set_difficulty_parameters(self, m_distance=None, ring_width_delta=None,
                                   p_init_dv=None, dist_cap=None):
        if m_distance is not None:
            self._config.init_distance_m = m_distance
        if ring_width_delta is not None:
            self._config.ring_width_delta = ring_width_delta
        # 同步到 reset 直接读取的字段
        self._config.e_init_dist_min_offset = self._config.init_distance_m
        self._config.e_init_dist_max_offset = (
            self._config.init_distance_m + self._config.ring_width_delta
        )
        if p_init_dv is not None:
            self._config.p_init_dv = p_init_dv
        if dist_cap is not None:
            self._config.dist_cap = dist_cap
