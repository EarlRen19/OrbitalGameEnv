# MPE-Env: 使用继承实现的多智能体追逃环境
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import datetime
import numpy as np
from gymnasium import spaces

# 导入基础环境及其配置
from .pe_env import PEEnv, PEEnvCfg

@dataclass
class MPEEnvCfg(PEEnvCfg):
    """多智能体追逃环境的配置"""
    num_p: int = 4
    num_e: int = 1
    
    # 多智能体场景下的奖励权重
    reward_dist_weight: float = 0.002
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
    arena_radius: float = 10e+3  # 参考距离（米），用于奖励归一化

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

        # 用于渐进式奖励的状态变量
        self.previous_dists: Dict[str, float] = {}
        self.min_dists: Dict[str, float] = {}

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

    def observe(self, agent):
        return self._get_observations().get(agent)

    @staticmethod
    def _symlog(x):
        """对称对数函数，用于归一化，处理大范围数值。"""
        return np.sign(x) * np.log(np.abs(x) + 1.0)

    def _get_observations(self):
        observations = {}
        
        # 收集所有智能体的完整状态
        all_states_dict = {aid: self.states[aid] for aid in self.possible_agents if aid in self.states}
        
        # 为每个智能体构建观测
        for agent_id in self.possible_agents:
            if agent_id not in all_states_dict:
                continue
                
            if agent_id.startswith('p_'):  # 追击方
                obs_components = []
                my_state = all_states_dict[agent_id]
                my_pos = my_state[:3]
                
                # 1. 自身状态（6维）- 应用symlog
                obs_components.append(self._symlog(my_state))
                
                # 2. 所有逃逸方相对位置（3*num_e维）- 先计算相对位置，再symlog
                evader_rel_positions = []
                for i in range(self._config.num_e):
                    evader_id = f'e_{i}'
                    if evader_id in all_states_dict:
                        evader_pos = all_states_dict[evader_id][:3]
                        rel_pos = evader_pos - my_pos
                        evader_rel_positions.append(self._symlog(rel_pos))
                    else:
                        # 如果逃逸者不存在，用零填充
                        evader_rel_positions.append(np.zeros(3))
                
                obs_components.append(np.concatenate(evader_rel_positions))
                
                # 3. 其他追击方相对位置（3*(num_p-1)维）- 先计算相对位置，再symlog
                other_pursuer_rel_positions = []
                for i in range(self._config.num_p):
                    pursuer_id = f'p_{i}'
                    if pursuer_id != agent_id:
                        if pursuer_id in all_states_dict:
                            pursuer_pos = all_states_dict[pursuer_id][:3]
                            rel_pos = pursuer_pos - my_pos
                            other_pursuer_rel_positions.append(self._symlog(rel_pos))
                        else:
                            # 如果队友不存在，用零填充
                            other_pursuer_rel_positions.append(np.zeros(3))

                if self._config.num_p > 1:
                    obs_components.append(np.concatenate(other_pursuer_rel_positions))

                observations[agent_id] = np.concatenate(obs_components)
            else:  # 逃方 - 保持绝对坐标的symlog
                observations[agent_id] = self._symlog(all_states_dict[agent_id])
        
        return observations

    def _get_apf_action(self, evader_id):
        """
        计算势场法动作: 逃逸者受到所有追捕者的反向斥力
        """
        if evader_id not in self.states:
            return np.zeros(3)

        e_pos = self.states[evader_id][:3]
        total_force = np.zeros(3)
        
        # 遍历所有存活的追捕者
        active_pursuers = [pid for pid in self.pursuer_ids if pid in self.states]
        
        # 如果没有追捕者了，就随机动或者不动
        if not active_pursuers:
            return np.zeros(3)

        for p_id in active_pursuers:
            p_pos = self.states[p_id][:3]
            diff_vec = e_pos - p_pos
            dist = np.linalg.norm(diff_vec)
            
            # 防止除以零
            if dist < 1.0: 
                dist = 1.0
            
            # 斥力公式: 方向远离追捕者，大小与距离平方成反比
            # Force = k * (1/r^2) * unit_vec
            force_vec = (diff_vec / dist) / (dist ** 2)
            total_force += force_vec
            
        # 计算合力方向
        force_magnitude = np.linalg.norm(total_force)
        
        if force_magnitude < 1e-9:
            # 极小力情况下（例如极其完美的对称包围），随机选择一个方向突围
            action_dir = np.random.randn(3)
            action_dir /= np.linalg.norm(action_dir)
        else:
            action_dir = total_force / force_magnitude
            
        # 输出动作：最大机动能力 * 方向
        action = action_dir * self._config.e_dv_step
        
        return action

    def get_evader_actions(self):
        """
        根据配置的 evader_policy_level 生成逃逸者的动作字典
        Level 0: 无机动 (Drift)
        Level 1: 随机机动 (Random)
        Level 2: 势场法 (APF)
        """
        actions = {}
        # 遍历所有存活的逃逸者
        active_evaders = [eid for eid in self.evader_ids if eid in self.agents]
        
        for e_id in active_evaders:
            if self._config.evader_policy_level == 0:
                # 模式 0: 自由漂浮
                actions[e_id] = np.zeros(3)
                
            elif self._config.evader_policy_level == 1:
                # 模式 1: 随机机动
                actions[e_id] = self.action_spaces[e_id].sample()
                
            elif self._config.evader_policy_level == 2:
                # 模式 2: 势场法逃逸
                actions[e_id] = self._get_apf_action(e_id)
            
            else:
                # 默认随机
                actions[e_id] = self.action_spaces[e_id].sample()
                
        return actions

    def _get_rewards(self, actions: Dict[str, np.ndarray]):
        rewards = {a: 0.0 for a in self.agents}
        debug_reward_info = {agent_id: {} for agent_id in self.pursuer_ids if agent_id in self.agents}
        
        # 收集位置信息
        pursuer_positions = []
        for pid in self.pursuer_ids:
            if pid in self.states:
                pursuer_positions.append(self.states[pid][:3])
        
        evader_positions = []
        for eid in self.evader_ids:
            if eid in self.states:
                evader_positions.append(self.states[eid][:3])
        
        if not evader_positions or not pursuer_positions:
            return rewards, debug_reward_info

        dists_to_evader = [np.linalg.norm(p_pos - evader_positions[0]) for p_pos in pursuer_positions]
        
        # 渐进式接近奖励
        dist_rewards = {}
        for i, agent_id in enumerate(self.pursuer_ids):
            if agent_id in self.agents:
                current_dist = dists_to_evader[i]
                prev_dist = self.previous_dists.get(agent_id, float('inf'))
                min_dist = self.min_dists.get(agent_id, float('inf'))

                # 奖励接近，惩罚远离
                dist_change_reward = self._config.reward_dist_weight * (prev_dist - current_dist) / self._config.arena_radius

                # 达到更小距离时给予额外奖励
                min_dist_bonus = 0.0
                if current_dist < min_dist:
                    min_dist_bonus = self._config.reward_dist_weight * (min_dist - current_dist) / self._config.arena_radius
                    self.min_dists[agent_id] = current_dist
                
                dist_rewards[agent_id] = dist_change_reward + min_dist_bonus
                self.previous_dists[agent_id] = current_dist

        # 队形奖励
        formation_score = self._calculate_formation_score(pursuer_positions, evader_positions[0])
        formation_reward = self._config.reward_formation_weight * (1.0 / (1.0 + formation_score))
        
        capture_occurred = min(dists_to_evader) < self._config.dist_cap

        # --- 3. 过程优势诱导奖励 (r_adv) 优化版 ---
        # 基于论文 4.4.1 节逻辑进行平滑处理
        num_future_steps = int(self._config.advantage_reward_horizon / self._config.dt)
        future_rewards = {a: 0.0 for a in self.pursuer_ids}
        
        if num_future_steps > 0 and self.evader_ids[0] in self.states:
            temp_e_state = np.copy(self.states[self.evader_ids[0]])
            future_e_traj = []
            for step in range(num_future_steps):
                current_sim_time = self._time + datetime.timedelta(seconds=step * self._config.dt)
                _, temp_e_state = self._orbit_lib.orbit_hpop(current_sim_time, temp_e_state, self._config.dt, self._config.hpop_in)
                future_e_traj.append(temp_e_state[:3])
            
            for i, agent_id in enumerate(self.pursuer_ids):
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

                    future_rewards[agent_id] = self._config.reward_advantage_weight * radv

        # 分配总奖励
        for i, agent_id in enumerate(self.pursuer_ids):
            if agent_id in self.agents:
                r_dist = dist_rewards.get(agent_id, 0.0)
                r_formation = formation_reward
                r_time = -self._config.reward_time_weight
                r_fuel = -self._config.reward_fuel_weight * np.linalg.norm(actions.get(agent_id, np.zeros(3)))
                r_adv = future_rewards.get(agent_id, 0.0)

                rewards[agent_id] = r_dist + r_formation + r_time + r_fuel + r_adv
                
                if self._config.debug_rewards:
                    debug_reward_info[agent_id].update({
                        "r_dist": r_dist,
                        "r_formation": r_formation,
                        "r_time": r_time,
                        "r_fuel": r_fuel,
                        "r_adv": r_adv,
                        "total_pre_terminal": rewards[agent_id]
                    })

        # 抓捕成功/失败的终端奖励
        if capture_occurred:
            for i, agent_id in enumerate(self.pursuer_ids):
                if agent_id in self.agents:
                    capture_bonus = 0.0
                    if dists_to_evader[i] < self._config.dist_cap:
                        capture_bonus = self._config.capture_reward
                    else:
                        capture_bonus = self._config.capture_reward * 0.5
                    
                    rewards[agent_id] += capture_bonus
                    if self._config.debug_rewards:
                        debug_reward_info[agent_id]['r_capture'] = capture_bonus
                        debug_reward_info[agent_id]['final_total'] = rewards[agent_id]

        if self._config.debug_rewards and any(debug_reward_info.values()) and self.step_count % 20 == 0:
            print(f"\n--- Step @ {self._time} Reward Debug (Step: {self.step_count}) ---")
            for agent_id, reward_data in debug_reward_info.items():
                if reward_data:
                    reward_str = ", ".join([f"{k}: {v:.4f}" for k, v in reward_data.items()])
                    print(f"  Agent {agent_id}: {reward_str}")
            print("------------------------------------")

        return rewards, debug_reward_info

    def _calculate_formation_score(self, pursuer_positions, evader_pos):
        """计算形成得分，鼓励良好的包围阵型。返回值越小越好。"""
        if len(pursuer_positions) < 2:
            return 0.0
        
        # 计算每个追击方到逃逸方的单位向量
        unit_vectors = []
        for pursuer_pos in pursuer_positions:
            vec = pursuer_pos - evader_pos
            dist = np.linalg.norm(vec)
            if dist < 1e-6:
                unit_vectors.append(np.zeros(3))
            else:
                unit_vectors.append(vec / dist)
        
        # 计算所有单位向量之和的模长，理想包围是各向量互相抵消，模长为0
        sum_of_vectors = np.sum(unit_vectors, axis=0)
        formation_score = np.linalg.norm(sum_of_vectors)
        
        return formation_score

    def _get_terminations(self):
        """判断终止条件，分为成功和失败两种情况"""
        terminations = {a: False for a in self.agents}
        termination_reasons = {a: None for a in self.agents}  # 记录终止原因
        
        # 检查抓捕成功
        evader_pos = None
        if 'e_0' in self.states:
            evader_pos = self.states['e_0'][:3]
        
        if evader_pos is not None:
            capture_occurred = False
            for pursuer_id in self.pursuer_ids:
                if pursuer_id in self.states:
                    dist = np.linalg.norm(self.states[pursuer_id][:3] - evader_pos)
                    if dist < self._config.dist_cap:
                        capture_occurred = True
                        break
            if capture_occurred:
                terminations = {a: True for a in self.agents}
                termination_reasons = {a: 'capture_success' for a in self.agents}
                return terminations, termination_reasons

        # 检查燃料耗尽
        for a in self.agents:
          if ('p' in a) and self.remain_Dvs[a] <=0: 
            terminations = {a: True for a in self.agents}
            termination_reasons = {a: 'fuel_out' for a in self.agents}
            return terminations, termination_reasons


        # 检查超时
        if self._time >= self._config.init_utc + datetime.timedelta(seconds=self._config.episode_length):
            terminations = {a: True for a in self.agents}
            termination_reasons = {a: 'timeout' for a in self.agents}
            return terminations, termination_reasons

        return terminations, termination_reasons

    def _get_truncations(self):
        """截断条件（当前版本为空，所有终止都通过终止条件处理）"""
        return {a: False for a in self.agents}

    def step(self, actions: Dict[str, np.ndarray]):
        """重写step方法,支持终止条件分类和课程学习统计"""
        self.step_count += 1
        for a in self.agents:
            if a.startswith('p_'):
                dv_step = self._config.p_dv_step
            else:
                dv_step = self._config.e_dv_step

            # 确保动作在合理范围内
            action = actions.get(a, np.zeros(3))
            if np.linalg.norm(action) > dv_step:
                action = action / np.linalg.norm(action) * dv_step

            if np.linalg.norm(action) > self.remain_Dvs[a]:
                action = action / np.linalg.norm(action) * self.remain_Dvs[a]

            self.states[a][3:] += action
            self.remain_Dvs[a] -= np.linalg.norm(action)

            # 防止除法误差使燃料变为负数
            self.remain_Dvs[a]=max(0,self.remain_Dvs[a])

        for a in self.agents:
            _, new_state = self._orbit_lib.orbit_hpop(
                self._time,
                self.states[a],
                self._config.dt,
                self._config.hpop_in
            )
            self.states[a] = new_state
        self._time = self._time + datetime.timedelta(seconds=self._config.dt)

        observations = self._get_observations()
        rewards, debug_reward_info = self._get_rewards(actions)
        
        terminations, termination_reasons = self._get_terminations()
        truncations = self._get_truncations()

        self.terminations = terminations
        self.truncations = truncations

        if any(terminations.values()):
            self.episode_statistics['total_episodes'] += 1
            reason = list(termination_reasons.values())[0] if termination_reasons else 'unknown'
            
            if reason == 'capture_success':
                self.episode_statistics['success_count'] += 1
            elif reason == 'timeout':
                self.episode_statistics['timeout_count'] += 1
                for a in self.agents:
                    if a.startswith('p_'):
                        rewards[a] += self._config.reward_timeout_penalty
            elif reason == 'fuel_out':
                self.episode_statistics['fuelout_count'] += 1
                for a in self.agents:
                    if a.startswith('p_'):
                        rewards[a] += self._config.reward_fuelout_penalty
            
            if self.episode_statistics['total_episodes'] > 0:
                self.episode_statistics['success_rate'] = (
                    self.episode_statistics['success_count'] / self.episode_statistics['total_episodes']
                )

        self.infos = {a: {} for a in self.agents}
        for agent_id in self.agents:
            self.infos[agent_id]['termination_reason'] = termination_reasons.get(agent_id, None)
            self.infos[agent_id]['episode_statistics'] = self.episode_statistics.copy()
            if self._config.debug_rewards and agent_id in debug_reward_info:
                self.infos[agent_id]['reward_components'] = debug_reward_info[agent_id]

        for agent in list(self.agents):
            if terminations.get(agent, False) or truncations.get(agent, False):
                if agent in observations:
                    self.infos[agent]['final_observation'] = observations[agent]
                self.agents.remove(agent)

        return observations, rewards, self.terminations, self.truncations, self.infos

    def reset(self, seed=None, options=None):
        # If debug flag is set, always use the same seed for reset to get a fixed scenario
        if self._config.use_fixed_seed_for_reset:
            np.random.seed(42) # Use a fixed seed, e.g., 42

        self.agents = self.possible_agents[:]
        
        # --- 1. 基础轨道参数 ---
        base_sma = 42166300.0  # 地球同步轨道
        ecc = 0.0
        inc = 0.0 
        raan, argp = 0.0, 0.0
        
        # --- 2. 生成逃逸者 (Evader) ---
        self.states = {} # Reset states dict
        # 逃逸者在圆周上随机位置
        ta_eva = np.random.uniform(0.0, 2 * np.pi)
        
        # 给逃逸者一点点高度随机性
        eva_sma = base_sma
        if self.current_sma_perturb_km > 0:
            perturb_m = np.random.uniform(-self.current_sma_perturb_km * 1000, self.current_sma_perturb_km * 1000)
            eva_sma += perturb_m
        
        self.states['e_0'] = self._orbit_lib.coe2rv(np.array([
            eva_sma, ecc, inc, raan, argp, ta_eva
        ]))
        
        # --- 3. 生成追击者 (Pursuers) - 动态扇环分布 ---
        current_cap = self._config.dist_cap 
        inner_dist = current_cap + self._config.e_init_dist_min_offset
        outer_dist = current_cap + self._config.e_init_dist_max_offset
        
        # 创建一个平衡的前后方向列表并随机打乱
        num_forward = self._config.num_p // 2 + self._config.num_p % 2
        num_backward = self._config.num_p // 2
        directions = ([1.0] * num_forward) + ([-1.0] * num_backward)
        np.random.shuffle(directions)

        for i in range(self._config.num_p):
            agent_id = f'p_{i}'
            
            target_dist = np.random.uniform(inner_dist, outer_dist)
            
            # 使用打乱后的方向
            direction = directions[i]
            
            # 将距离转换为角度偏移
            angle_offset = (target_dist / base_sma) * direction
            ta_pur = (ta_eva + angle_offset) % (2 * np.pi)
            
            # 轨道高度 (SMA) 随机化
            pur_sma = base_sma
            if self.current_sma_perturb_km > 0:
                perturb_m = np.random.uniform(-self.current_sma_perturb_km * 1000, self.current_sma_perturb_km * 1000)
                pur_sma += perturb_m

            self.states[agent_id] = self._orbit_lib.coe2rv(np.array([
                pur_sma, ecc, inc, raan, argp, ta_pur
            ]))

        self._time = self._config.init_utc
        
        # 初始化燃料
        self.remain_Dvs = {}
        for a in self.agents:
            if a.startswith('p_'):
                self.remain_Dvs[a] = self._config.p_init_dv
            else:
                self.remain_Dvs[a] = self._config.e_init_dv
        
        # 初始化奖励计算所需的状态
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        evader_pos = self.states[self.evader_ids[0]][:3]
        for p_id in self.pursuer_ids:
            p_pos = self.states[p_id][:3]
            initial_dist = np.linalg.norm(p_pos - evader_pos)
            self.previous_dists[p_id] = initial_dist
            self.min_dists[p_id] = initial_dist

        # Reset Viewer
        if self.viewer is not None:
            self.viewer.reset()

        # 生成观测
        observations = self._get_observations()
        
        # 初始化 Info 统计
        self.infos = {a: {} for a in self.agents}
        for agent in self.agents:
            self.infos[agent]['episode_statistics'] = self.episode_statistics.copy()

        return observations, self.infos

    def get_success_rate(self):
        """获取当前成功率，用于课程学习"""
        return self.episode_statistics['success_rate']

    def get_episode_statistics(self):
        """获取完整的课程学习统计信息"""
        return self.episode_statistics.copy()

    def set_difficulty_parameters(self, 
                                 episode_length=None, 
                                 dist_cap=None, 
                                 reward_weights=None,
                                 p_init_dv=None):
        """动态调整难度参数，支持课程学习"""
        if episode_length is not None:
            self._config.episode_length = episode_length
        if dist_cap is not None:
            self._config.dist_cap = dist_cap
        if p_init_dv is not None:
            self._config.p_init_dv = p_init_dv
        if reward_weights is not None:
            # 可以动态调整奖励权重
            if 'reward_capture' in reward_weights:
                self._config.reward_capture = reward_weights['reward_capture']
            if 'reward_timeout_penalty' in reward_weights:
                self._config.reward_timeout_penalty = reward_weights['reward_timeout_penalty']
