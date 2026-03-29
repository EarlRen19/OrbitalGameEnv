# MPE-Env: 使用继承实现的多智能体追逃环境
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import datetime
import numpy as np
from gymnasium import spaces
import ctypes

# 导入基础环境及其配置
from .pe_env import PEEnv, PEEnvCfg

@dataclass
class MPEEnvCfg(PEEnvCfg):
    """多智能体追逃环境的配置"""
    num_p: int = 4
    num_e: int = 1
    
    # 多智能体场景下的奖励权重
    reward_time_weight: float = 0.01
    reward_formation_weight: float = 0.04
    reward_fuel_weight: float = 0.05
    capture_reward: float = 10.0
    reward_timeout_penalty: float = -2.0
    reward_fuelout_penalty: float = -1.0
    fuel_penalty_weight: float = 0.1

    # 过程优势奖励
    reward_advantage_weight: float = 0.05
    advantage_reward_horizon: float = 3600*2
    
    # 新的相位距离奖励权重
    reward_phase_dist_weight: float = 1.0

    # === 失败重练机制 ===
    replay_failed_prob: float = 0.6
    max_failed_pool_size: int = 1000
    
    # === 初始生成安全参数 ===
    init_min_separation: float = 5000.0  # 米

    # === [新] 课程学习最终版 ===
    start_ring_ratio: float = 1.0
    final_ring_ratio: float = 0.5
    formation_curriculum_end_step: int = 800_000
    
    # [新] 早停条件
    early_stop_start_step: int = 800_000
    early_stop_success_rate: float = 0.85


import random
from collections import deque

class GlobalTrainManager:
    """
    用于跨多个环境实例共享状态的管理器。
    所有 MPEEnv 实例将引用这个类的数据。
    """
    failed_seeds = []
    current_ring_ratio = 1.0
    
    @classmethod
    def add_failed_seed(cls, seed, max_size):
        if seed not in cls.failed_seeds:
            cls.failed_seeds.append(seed)
            if len(cls.failed_seeds) > max_size:
                cls.failed_seeds.pop(0)
    
    @classmethod
    def get_failed_seed(cls):
        if not cls.failed_seeds:
            return None
        idx = random.randint(0, len(cls.failed_seeds) - 1)
        return cls.failed_seeds.pop(idx)

    @classmethod
    def update_ratios(cls, current_step: int, cfg: MPEEnvCfg):
        """外部训练循环调用此函数更新课程进度"""
        progress = np.clip(current_step / cfg.formation_curriculum_end_step, 0.0, 1.0)
        cls.current_ring_ratio = cfg.start_ring_ratio - progress * (cfg.start_ring_ratio - cfg.final_ring_ratio)


class MPEEnv(PEEnv):
    """
    继承自PEEnv的多智能体追逃环境。
    为多追击者场景修改了观测空间、奖励函数和终止条件。
    """
    def __init__(self, config: MPEEnvCfg = MPEEnvCfg()):
        # 初始化基类
        super().__init__(config)

        self.pursuer_ids = [f'p_{i}' for i in range(self._config.num_p)]
        self.evader_ids = [f'e_{i}' for i in range(self._config.num_e)]
        
        self.metadata["is_parallelizable"] = True
        
        # 为多智能体场景覆盖观测空间
        self.observation_spaces = {}
        for a in self.possible_agents:
            if a.startswith('p_'):
                # 自身状态(6) + 所有逃逸者绝对位置(3*num_e) + 其他追击者相对位置(3*(num_p-1))
                obs_shape = (6 + 3 * self._config.num_e + 3 * (self._config.num_p - 1),)
                self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=obs_shape)
            else:
                self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=(6,))
        
        self.infos = {a: {} for a in self.possible_agents}

        # 用于课程学习的统计信息
        self.episode_statistics = {
            'success_count': 0,
            'timeout_count': 0,
            'fuelout_count': 0,
            'total_episodes': 0,
            'success_rate': 0.0
        }

        # 用于控制打印频率的步数计数器
        self.step_count = 0
        self.current_episode_seed = None
        self.current_formation_name = "unknown" # 用于Debug

    def observe(self, agent):
        return self._get_observations().get(agent)

    @staticmethod
    def _symlog(x):
        """对称对数函数，用于归一化，处理大范围数值。"""
        return np.sign(x) * np.log(np.abs(x) + 1.0)

    def _calculate_orbital_metrics(self, state):
        """Helper to get SMA and True Anomaly from a state vector."""
        rv = np.ascontiguousarray(state, dtype=np.float64)
        coe = np.zeros(6, dtype=np.float64)
        if self._orbit_lib and hasattr(self._orbit_lib, 'orbit_lib_c') and self._orbit_lib.orbit_lib_c is not None:
            # Ensure argtypes are set, as a safeguard
            self._orbit_lib.orbit_lib_c.RV2COE.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), 
                                                           np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')]
            self._orbit_lib.orbit_lib_c.RV2COE.restype = None
            self._orbit_lib.orbit_lib_c.RV2COE(rv, coe)
            # coe = [sma, ecc, inc, raan, argp, ta]
            return coe[0], coe[5]
        return 0, 0 # Return default values if library fails

    def _get_phase_distance_reward(self, agent_id, target_id):
        """
        针对 30km 捕获半径 + 课程学习的优化版奖励
        """
        # --- 1. 关键参数定义 ---
        DIST_CAP = 30000.0
        TRANSITION_DIST = 60000.0
        
        # --- 2. 获取物理量 ---
        p_state = self.states[agent_id]
        e_state = self.states[target_id]
        dist = np.linalg.norm(p_state[:3] - e_state[:3])
        
        sma_p, theta_p = self._calculate_orbital_metrics(p_state)
        sma_e, theta_e = self._calculate_orbital_metrics(e_state)
        
        delta_theta = (theta_p - theta_e + np.pi) % (2 * np.pi) - np.pi
        sma_diff_ratio = (sma_p - sma_e) / (sma_e + 1e-6)
        
        # --- 3. 远场逻辑 (Far-field) ---
        drift_product = -delta_theta * sma_diff_ratio
        if drift_product > 0:
            R_Far = -1.0 - np.abs(sma_diff_ratio) * 2000.0
        else:
            r_drift = np.clip(np.abs(sma_diff_ratio) * 1000.0, 0.0, 2.0)
            r_angle = (np.pi - np.abs(delta_theta)) / np.pi
            R_Far = 1.0 * r_drift + 0.5 * r_angle

        # --- 4. 近场逻辑 (Near-field) ---
        norm_dist = dist / DIST_CAP
        
        if norm_dist <= 1.0:
            R_dist = 1.0 + 0.1 * (1.0 - norm_dist)
        elif norm_dist <= 2.0:
            R_dist = 2.0 - norm_dist
        else:
            R_dist = np.clip(2.0 - norm_dist, -1.0, 0.0)
            
        R_energy = -np.abs(sma_diff_ratio) * 2000.0
        R_Near = 1.0 * R_dist + 0.05 * R_energy
        
        # --- 5. 混合 ---
        alpha = np.tanh(dist / TRANSITION_DIST)
        total_reward = alpha * R_Far + (1.0 - alpha) * R_Near
        
        return total_reward * 0.1

    def _get_observations(self):
        observations = {}
        all_states_dict = {aid: self.states[aid] for aid in self.possible_agents if aid in self.states}
        for agent_id in self.possible_agents:
            if agent_id not in all_states_dict:
                continue
            if agent_id.startswith('p_'):
                obs_components = []
                my_state = all_states_dict[agent_id]
                my_pos = my_state[:3]
                obs_components.append(self._symlog(my_state))
                evader_rel_positions = []
                for i in range(self._config.num_e):
                    evader_id = f'e_{i}'
                    if evader_id in all_states_dict:
                        evader_pos = all_states_dict[evader_id][:3]
                        rel_pos = evader_pos - my_pos
                        evader_rel_positions.append(self._symlog(rel_pos))
                    else:
                        evader_rel_positions.append(np.zeros(3))
                obs_components.append(np.concatenate(evader_rel_positions))
                other_pursuer_rel_positions = []
                for i in range(self._config.num_p):
                    pursuer_id = f'p_{i}'
                    if pursuer_id != agent_id:
                        if pursuer_id in all_states_dict:
                            pursuer_pos = all_states_dict[pursuer_id][:3]
                            rel_pos = pursuer_pos - my_pos
                            other_pursuer_rel_positions.append(self._symlog(rel_pos))
                        else:
                            other_pursuer_rel_positions.append(np.zeros(3))
                if self._config.num_p > 1:
                    obs_components.append(np.concatenate(other_pursuer_rel_positions))
                observations[agent_id] = np.concatenate(obs_components)
            else:
                observations[agent_id] = self._symlog(all_states_dict[agent_id])
        return observations

    def _get_apf_action(self, evader_id):
        if evader_id not in self.states:
            return np.zeros(3)
        e_pos = self.states[evader_id][:3]
        total_force = np.zeros(3)
        active_pursuers = [pid for pid in self.pursuer_ids if pid in self.states]
        if not active_pursuers:
            return np.zeros(3)
        for p_id in active_pursuers:
            p_pos = self.states[p_id][:3]
            diff_vec = e_pos - p_pos
            dist = np.linalg.norm(diff_vec)
            if dist < 1.0: dist = 1.0
            force_vec = (diff_vec / dist) / (dist ** 2)
            total_force += force_vec
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude < 1e-9:
            action_dir = np.random.randn(3)
            action_dir /= np.linalg.norm(action_dir)
        else:
            action_dir = total_force / force_magnitude
        action = action_dir
        return action

    def get_evader_actions(self):
        actions = {}
        active_evaders = [eid for eid in self.evader_ids if eid in self.agents]
        for e_id in active_evaders:
            if self._config.evader_policy_level == 0:
                actions[e_id] = np.zeros(3)
            elif self._config.evader_policy_level == 1:
                actions[e_id] = self.action_spaces[e_id].sample()
            elif self._config.evader_policy_level == 2:
                actions[e_id] = self._get_apf_action(e_id)
            else:
                actions[e_id] = self.action_spaces[e_id].sample()
        return actions

    def _get_rewards(self, actions: Dict[str, np.ndarray]):
        rewards = {a: 0.0 for a in self.agents}
        debug_reward_info = {agent_id: {} for agent_id in self.pursuer_ids if agent_id in self.agents}
        
        pursuer_positions = [self.states[pid][:3] for pid in self.pursuer_ids if pid in self.states]
        main_evader_id = self.evader_ids[0]
        evader_positions = [self.states[main_evader_id][:3]] if main_evader_id in self.states else []
        
        if not evader_positions or not pursuer_positions:
            return rewards, debug_reward_info

        dists_to_evader = [np.linalg.norm(p_pos - evader_positions[0]) for p_pos in pursuer_positions]
        
        formation_reward = self._config.reward_formation_weight * (1.0 / (1.0 + self._calculate_formation_score(pursuer_positions, evader_positions[0])))
        capture_occurred = min(dists_to_evader) < self._config.dist_cap

        # === [新增] 防碰撞惩罚逻辑 ===
        collision_penalties = {pid: 0.0 for pid in self.pursuer_ids}
        active_pursuers = [pid for pid in self.pursuer_ids if pid in self.agents]
        p_positions = {pid: self.states[pid][:3] for pid in active_pursuers}
        
        if len(active_pursuers) > 1:
            for i in range(len(active_pursuers)):
                for j in range(i + 1, len(active_pursuers)):
                    id_a = active_pursuers[i]
                    id_b = active_pursuers[j]
                    
                    pos_a = p_positions[id_a]
                    pos_b = p_positions[id_b]
                    
                    dist = np.linalg.norm(pos_a - pos_b)
                    
                    if dist < self._config.dist_collision:
                        penalty_factor = (self._config.dist_collision - dist) / self._config.dist_collision
                        raw_penalty = self._config.max_collision_penalty * (penalty_factor ** 2)
                        final_penalty = -self._config.reward_collision_weight * raw_penalty
                        
                        collision_penalties[id_a] += final_penalty
                        collision_penalties[id_b] += final_penalty

        # --- 保留旧的 r_adv 计算逻辑 ---
        num_future_steps = int(self._config.advantage_reward_horizon / self._config.dt)
        advantage_rewards = {a: 0.0 for a in self.pursuer_ids}
        if num_future_steps > 0 and main_evader_id in self.states:
            temp_e_state = np.copy(self.states[main_evader_id])
            future_e_traj = []
            for step in range(num_future_steps):
                current_sim_time = self._time + datetime.timedelta(seconds=step * self._config.dt)
                _, temp_e_state = self._orbit_lib.orbit_hpop(current_sim_time, temp_e_state, self._config.dt, self._config.hpop_in)
                future_e_traj.append(temp_e_state[:3])
            
            for agent_id in self.pursuer_ids:
                if agent_id in self.agents:
                    temp_p_state = np.copy(self.states[agent_id])
                    min_future_dist = float('inf')
                    min_step_idx = num_future_steps
                    for step in range(num_future_steps):
                        current_sim_time = self._time + datetime.timedelta(seconds=step * self._config.dt)
                        _, temp_p_state = self._orbit_lib.orbit_hpop(current_sim_time, temp_p_state, self._config.dt, self._config.hpop_in)
                        dist = np.linalg.norm(temp_p_state[:3] - future_e_traj[step])
                        if dist < min_future_dist:
                            min_future_dist = dist
                            min_step_idx = step
                    
                    dist_threshold = self._config.dist_cap
                    norm_time = min_step_idx / num_future_steps 
                    time_discount = np.exp(-2.0 * norm_time)
                    if min_future_dist < dist_threshold:
                        dist_advantage = (dist_threshold - min_future_dist) / dist_threshold
                        radv = 1.0 + dist_advantage + time_discount
                    else:
                        miss_ratio = min_future_dist / dist_threshold
                        penalty = np.log(miss_ratio) 
                        radv = -1.0 * penalty * 0.5
                        radv = max(radv, -2.0)
                    advantage_rewards[agent_id] = self._config.reward_advantage_weight * radv

        # --- 分配总奖励 ---
        for agent_id in self.pursuer_ids:
            if agent_id in self.agents:
                r_dist_phase = self._get_phase_distance_reward(agent_id, main_evader_id)
                r_adv = advantage_rewards.get(agent_id, 0.0)
                r_coll = collision_penalties.get(agent_id, 0.0)
                
                rewards[agent_id] = (self._config.reward_phase_dist_weight * r_dist_phase +
                                     formation_reward +
                                     -self._config.reward_time_weight +
                                     -self._config.reward_fuel_weight * np.linalg.norm(actions.get(agent_id, np.zeros(3))) +
                                     r_adv +
                                     r_coll)
                
                if self._config.debug_rewards:
                    debug_reward_info[agent_id].update({
                        "r_dist_phase": self._config.reward_phase_dist_weight * r_dist_phase,
                        "r_formation": formation_reward,
                        "r_time": -self._config.reward_time_weight,
                        "r_fuel": -self._config.reward_fuel_weight * np.linalg.norm(actions.get(agent_id, np.zeros(3))),
                        "r_adv_original": r_adv,
                        "r_collision": r_coll,
                        "total_pre_terminal": rewards[agent_id]
                    })

        # 终端奖励
        if capture_occurred:
            for i, agent_id in enumerate(self.pursuer_ids):
                if agent_id in self.agents:
                    capture_bonus = self._config.capture_reward * (0.5 if dists_to_evader[i] >= self._config.dist_cap else 1.0)
                    rewards[agent_id] += capture_bonus
                    if self._config.debug_rewards:
                        debug_reward_info[agent_id]['r_capture'] = capture_bonus
                        debug_reward_info[agent_id]['final_total'] = rewards[agent_id]

        # 全局缩放
        for agent_id in rewards:
            rewards[agent_id] /= 10.0

        return rewards, debug_reward_info

    def _calculate_formation_score(self, pursuer_positions, evader_pos):
        if len(pursuer_positions) < 2: return 0.0
        unit_vectors = [(p_pos - evader_pos) / (np.linalg.norm(p_pos - evader_pos) + 1e-6) for p_pos in pursuer_positions]
        return np.linalg.norm(np.sum(unit_vectors, axis=0))

    def _get_terminations(self):
        terminations = {a: False for a in self.agents}
        reasons = {a: None for a in self.agents}
        evader_pos = self.states.get(self.evader_ids[0], [0,0,0])[:3]
        
        if any(np.linalg.norm(self.states[pid][:3] - evader_pos) < self._config.dist_cap for pid in self.pursuer_ids if pid in self.states):
            return {a: True for a in self.agents}, {a: 'capture_success' for a in self.agents}
        if any(self.remain_Dvs.get(pid, 0) <= 0 for pid in self.pursuer_ids):
            return {a: True for a in self.agents}, {a: 'fuel_out' for a in self.agents}
        if self._time >= self._config.init_utc + datetime.timedelta(seconds=self._config.episode_length):
            return {a: True for a in self.agents}, {a: 'timeout' for a in self.agents}
        return terminations, reasons

    def _get_truncations(self):
        return {a: False for a in self.agents}

    def step(self, actions: Dict[str, np.ndarray]):
        observations, rewards, terminations, truncations, infos = super().step(actions)
        
        # 检查是否因为失败结束
        any_done = any(terminations.values()) or any(truncations.values())
        if any_done:
            # 判断原因
            reason = None
            for info in infos.values():
                if 'termination_reason' in info:
                    reason = info['termination_reason']
                    break
            
            # 如果不是成功，加入失败池
            if reason != 'capture_success':
                if self.current_episode_seed is not None:
                    GlobalTrainManager.add_failed_seed(
                        self.current_episode_seed, 
                        self._config.max_failed_pool_size
                    )
            # 注意：如果成功了，不用做任何事。reset里如果它是pop出来的，它已经不在池子里了。
            
        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        force_formation = options.get('force_formation', None) if options else None

        if self._config.use_fixed_seed_for_reset: np.random.seed(42)
        
        # === 1. 决定种子与阵型策略 ===
        is_replay = False
        
        # 外部强制指定 Seed (评估模式)
        if seed is not None:
            self.current_episode_seed = seed
        else:
            # 训练模式
            if GlobalTrainManager.failed_seeds and np.random.random() < self._config.replay_failed_prob:
                popped_seed = GlobalTrainManager.get_failed_seed()
                if popped_seed is not None:
                    self.current_episode_seed = popped_seed
                    is_replay = True
                else:
                    self.current_episode_seed = np.random.randint(0, 1_000_000_000)
            else:
                self.current_episode_seed = np.random.randint(0, 1_000_000_000)
        
        np.random.seed(self.current_episode_seed)
        
        # === 2. 决定阵型 ===
        if force_formation is not None:
            chosen_formation = force_formation
        else:
            if is_replay:
                pass
            
            r_ring = GlobalTrainManager.current_ring_ratio
            r_others = (1.0 - r_ring) / 3.0
            
            formations = ['ring', 'cluster', 'string', 'pincer']
            probs = [r_ring, r_others, r_others, r_others]
            probs = np.array(probs) / np.sum(probs)
            
            chosen_formation = np.random.choice(formations, p=probs)

        self.current_formation_name = chosen_formation

        # === 3. 初始化对象 ===
        self.agents = self.possible_agents[:]
        self.states = {}
        
        # --- Evader (Anchor) ---
        base_sma = 42166300.0
        ecc, inc, raan, argp = 0.0, 0.0, 0.0, 0.0
        
        ta_eva = np.random.uniform(0.0, 2 * np.pi)
        eva_sma = base_sma
        if self.current_sma_perturb_km > 0:
            eva_sma += np.random.uniform(-self.current_sma_perturb_km * 1000, self.current_sma_perturb_km * 1000)
        
        self.states['e_0'] = self._orbit_lib.coe2rv(np.array([eva_sma, ecc, inc, raan, argp, ta_eva]))
        
        # --- Pursuers (围绕圆环生成) ---
        pursuer_ids_shuffled = [f'p_{i}' for i in range(self._config.num_p)]
        np.random.shuffle(pursuer_ids_shuffled)
        
        min_r = self._config.dist_cap + self._config.e_init_dist_min_offset
        max_r = self._config.dist_cap + self._config.e_init_dist_max_offset
        
        ring_base_angle = np.random.uniform(0, 2 * np.pi)
        ring_shared_dist = np.random.uniform(min_r, max_r)
        string_base_angle = np.random.uniform(0, 2 * np.pi)
        pincer_axis_angle = np.random.uniform(0, 2 * np.pi)

        generated_positions_2d = []

        for i, agent_id in enumerate(pursuer_ids_shuffled):
            valid_pos = False
            attempt = 0
            
            while not valid_pos:
                attempt += 1
                if attempt > 100: 
                    pass
                
                if chosen_formation == 'ring':
                    angle = ring_base_angle + i * (2 * np.pi / self._config.num_p)
                    dist = ring_shared_dist + np.random.uniform(-500, 500)
                    angle += np.random.uniform(-0.05, 0.05)
                    
                elif chosen_formation == 'cluster':
                    angle = np.random.uniform(0, 2 * np.pi)
                    dist = np.random.uniform(min_r, max_r)
                    
                elif chosen_formation == 'string':
                    lane_dist = np.random.uniform(min_r, max_r)
                    separation_dist = self._config.init_min_separation * np.random.uniform(1.2, 2.0)
                    angle_step = separation_dist / lane_dist
                    center_offset_idx = i - (self._config.num_p - 1) / 2.0
                    angle = string_base_angle + center_offset_idx * angle_step
                    dist_noise = np.random.uniform(-1000, 1000)
                    dist = np.clip(lane_dist + dist_noise, min_r, max_r)
                    angle += np.random.uniform(-0.01, 0.01)
                    
                elif chosen_formation == 'pincer':
                    side_offset = 0 if i % 2 == 0 else np.pi
                    angle = pincer_axis_angle + side_offset + np.random.uniform(-np.pi/6, np.pi/6)
                    dist = np.random.uniform(min_r, max_r)

                pos_2d = np.array([dist * np.cos(angle), dist * np.sin(angle)])
                
                is_colliding = False
                if attempt <= 100 and len(generated_positions_2d) > 0:
                    for exist_p in generated_positions_2d:
                        if np.linalg.norm(pos_2d - exist_p) < self._config.init_min_separation:
                            is_colliding = True
                            break
                
                if not is_colliding or attempt > 100:
                    valid_pos = True
                    generated_positions_2d.append(pos_2d)
                    
                    ta_offset = (dist / base_sma) * np.cos(angle)
                    sma_offset = dist * np.sin(angle)
                    
                    pur_sma = eva_sma + sma_offset
                    pur_ta = (ta_eva + ta_offset) % (2 * np.pi)
                    
                    if self.current_sma_perturb_km > 0:
                        pur_sma += np.random.uniform(-self.current_sma_perturb_km * 1000, self.current_sma_perturb_km * 1000)
                    
                    self.states[agent_id] = self._orbit_lib.coe2rv(np.array([pur_sma, ecc, inc, raan, argp, pur_ta]))

        self._time = self._config.init_utc
        self.remain_Dvs = {a: (self._config.p_init_dv if a.startswith('p_') else self._config.e_init_dv) for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        if self.viewer is not None: self.viewer.reset()
        observations = self._get_observations()
        self.infos = {a: {'episode_statistics': self.episode_statistics.copy()} for a in self.agents}
        return observations, self.infos

    def get_success_rate(self):
        return self.episode_statistics['success_rate']

    def get_episode_statistics(self):
        return self.episode_statistics.copy()

    def set_difficulty_parameters(self, episode_length=None, dist_cap=None, reward_weights=None, p_init_dv=None):
        if episode_length is not None: self._config.episode_length = episode_length
        if dist_cap is not None: self._config.dist_cap = dist_cap
        if p_init_dv is not None: self._config.p_init_dv = p_init_dv
        if reward_weights is not None:
            if 'reward_capture' in reward_weights: self._config.reward_capture = reward_weights['reward_capture']
            if 'reward_timeout_penalty' in reward_weights: self._config.reward_timeout_penalty = reward_weights['reward_timeout_penalty']
