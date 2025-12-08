#此处是部分可观的环境，对应的训练脚本是train_pomdp.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces
from collections import deque
import torch
import datetime

# 导入父类环境和配置
from .mpe_env import MPEEnv, MPEEnvCfg

# 导入新的lambert求解器
try:
    from lambert_solver import solve_lambert
except ImportError:
    print("\033[93mWarning: C++ Lambert solver not found. Lambert-based rewards will be disabled.\033[0m")
    solve_lambert = None

@dataclass
class MPE_POMDP_EnvCfg(MPEEnvCfg):
    """加入POMDP和Transformer的特定参数"""
    use_partial_obs: bool = True
    obs_interval: int = 2
    history_len: int = 20 # History length for Transformer

    # Lambert奖励配置 
    use_lambert_reward: bool = True
    lambert_reward_weight: float = 5
    lambert_transfer_time: float = 7200 #Lambert转移时间（秒）
    mu: float = 3.986004418e14

class MPE_POMDP_Env(MPEEnv):
    """
    部分可观MPE环境，为基于Transformer的智能体准备数据。
    - 历史记录(History): 存储原始物理坐标。
    - 观测(Observation): 包含自身状态和相对于自身的队友状态。
    - Info字典: 为每个智能体提供一个定制的、相对的、经过symlog处理的逃逸者历史轨迹，以及一个用于Transformer的掩码。
    """
    @staticmethod
    def _symlog(x):
        """对称对数函数，用于归一化。"""
        return np.sign(x) * np.log(np.abs(x) + 1.0)

    @staticmethod
    def _inv_symlog(y):
        """对称对数函数的逆函数。"""
        return np.sign(y) * (np.exp(np.abs(y)) - 1.0)

    def __init__(self, config: MPE_POMDP_EnvCfg = MPE_POMDP_EnvCfg()):
        super().__init__(config)
        self._config: MPE_POMDP_EnvCfg = config

        if self._config.use_partial_obs:
            self.evader_history_buffers = {f'e_{i}': deque(maxlen=self._config.history_len) for i in range(self._config.num_e)}
            self.obs_counters = {f'e_{i}': 0 for i in range(self._config.num_e)}
            # 存储每个时间步是否是真实观测
            self.evader_history_mask_buffers = {f'e_{i}': deque(maxlen=self._config.history_len) for i in range(self._config.num_e)}

            # 重新定义观测空间
            self.observation_spaces = {}
            for a in self.possible_agents:
                if a.startswith('p_'):
                    # 自身维度: 6(状态) + 1(燃料)
                    self_obs_dim = 7
                    # 队友维度: (N-1) * (6状态 + 1燃料)
                    other_pursuers_obs_dim = 7 * (self._config.num_p - 1)
                    # 最终观测不包含历史轨迹，历史轨迹将作为单独的输入进入网络
                    obs_shape = (self_obs_dim + other_pursuers_obs_dim,)
                    self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=obs_shape)
                else: # Evader's observation
                    self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=(6,))

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed, options)

        if self._config.use_partial_obs:
            self.obs_counters = {f'e_{i}': 0 for i in range(self._config.num_e)}
            
            for evader_id in [f'e_{i}' for i in range(self._config.num_e)]:
                self.evader_history_buffers[evader_id].clear()
                self.evader_history_mask_buffers[evader_id].clear()
                if evader_id in self.states:
                    # [修改]：直接存原始物理状态
                    initial_state = self.states[evader_id]
                    for _ in range(self._config.history_len):
                        self.evader_history_buffers[evader_id].append(initial_state)
                        # 初始时都视为“真实”观测的填充
                        self.evader_history_mask_buffers[evader_id].append(1.0)

            self._update_history_and_prepare_data()
            observations = self._get_observations()

            # 为所有智能体准备特权信息 (新的混合坐标系：锚点+相对)
            privileged_components = []
            anchor_id = self.evader_ids[0]
            if anchor_id in self.states:
                anchor_state = self.states[anchor_id]
                privileged_components.append(self._symlog(anchor_state))
            else:
                anchor_state = np.zeros(6)
                privileged_components.append(anchor_state)

            # 其他智能体使用相对于锚点的状态
            agent_order = self.pursuer_ids + [eid for eid in self.evader_ids if eid != anchor_id]
            for agent_id in agent_order:
                if agent_id in self.states:
                    relative_state = self.states[agent_id] - anchor_state
                    privileged_components.append(self._symlog(relative_state))
                else:
                    privileged_components.append(np.zeros(6))
            
            privileged_state = np.concatenate(privileged_components)

            for agent in self.agents:
                if agent.startswith('p_'):
                    if agent not in infos: infos[agent] = {}
                    infos[agent]['privileged_state'] = privileged_state

        return observations, infos

    def step(self, actions):
        if self._config.use_partial_obs:
            # 在step执行前，基于上一步的状态准备好给agent的输入
            self._update_history_and_prepare_data()
        
        # (与父类MPEEnv相同的动力学和奖励计算)
        clipped_actions = {}
        for a in self.agents:
            if a.startswith('p_'): dv_step = self._config.p_dv_step
            else: dv_step = self._config.e_dv_step

            action = actions.get(a, np.zeros(3))
            if np.linalg.norm(action) > dv_step:
                action = action / np.linalg.norm(action) * dv_step
            if np.linalg.norm(action) > self.remain_Dvs[a]: 
                action = action / np.linalg.norm(action) * self.remain_Dvs[a]
            
            clipped_actions[a] = action
            self.states[a][3:] += action
            self.remain_Dvs[a] = max(0, self.remain_Dvs[a] - np.linalg.norm(action))

        for a in self.agents:
            _, new_state = self._orbit_lib.orbit_hpop(self._time, self.states[a], self._config.dt, self._config.hpop_in)
            self.states[a] = new_state
        self._time = self._time + datetime.timedelta(seconds=self._config.dt)

        observations = self._get_observations()
        rewards, debug_reward_info = self._get_rewards(clipped_actions)
        terminations, termination_reasons = self._get_terminations()
        truncations = self._get_truncations()
        self.terminations, self.truncations = terminations, truncations

        current_infos = {a: self.infos.get(a, {}) for a in self.possible_agents if a in self.agents}
        if any(terminations.values()):
            self.episode_statistics['total_episodes'] += 1
            reason = list(termination_reasons.values())[0] if termination_reasons else 'unknown'
            
            if reason == 'capture_success': self.episode_statistics['success_count'] += 1
            elif reason == 'timeout':
                self.episode_statistics['timeout_count'] += 1
                for a in self.agents:
                    if a.startswith('p_'): rewards[a] += self._config.reward_timeout_penalty
            elif reason == 'fuel_out':
                self.episode_statistics['fuelout_count'] += 1
                for a in self.agents:
                    if a.startswith('p_'): rewards[a] += self._config.reward_fuelout_penalty
            
            if self.episode_statistics['total_episodes'] > 0:
                self.episode_statistics['success_rate'] = self.episode_statistics['success_count'] / self.episode_statistics['total_episodes']

        # 为Critic准备特权信息 (新的混合坐标系：锚点+相对)
        privileged_components = []
        anchor_id = self.evader_ids[0]
        if anchor_id in self.states:
            anchor_state = self.states[anchor_id]
            privileged_components.append(self._symlog(anchor_state))
        else:
            anchor_state = np.zeros(6)
            privileged_components.append(anchor_state)

        # 其他智能体使用相对于锚点的状态
        agent_order = self.pursuer_ids + [eid for eid in self.evader_ids if eid != anchor_id]
        for agent_id in agent_order:
            if agent_id in self.states:
                relative_state = self.states[agent_id] - anchor_state
                privileged_components.append(self._symlog(relative_state))
            else:
                privileged_components.append(np.zeros(6))
        
        privileged_state = np.concatenate(privileged_components)

        for agent in self.agents:
            current_infos[agent]['termination_reason'] = termination_reasons.get(agent, None)
            current_infos[agent]['episode_statistics'] = self.episode_statistics.copy()
            if self._config.debug_rewards and agent in debug_reward_info:
                current_infos[agent]['reward_components'] = debug_reward_info[agent]
            if agent.startswith('p_'):
                current_infos[agent]['privileged_state'] = privileged_state

        for agent in list(self.agents):
            if terminations.get(agent, False) or truncations.get(agent, False):
                if agent in observations: current_infos[agent]['final_observation'] = observations[agent]
                self.agents.remove(agent)

        return observations, rewards, self.terminations, self.truncations, current_infos

    def _get_rewards(self, actions):
        # 继承父类奖励逻辑
        return super()._get_rewards(actions)

    def _get_observations(self):
        if not self._config.use_partial_obs:
            return super()._get_observations()

        observations = {}
        all_states = {agent_id: self.states[agent_id] for agent_id in self.possible_agents if agent_id in self.states}
        all_fuels = self.remain_Dvs
        
        for agent_id in self.possible_agents:
            if agent_id not in self.states: continue
            
            if agent_id.startswith('p_'):
                obs_components = []
                my_state = all_states[agent_id]
                
                # 1. 自身信息 (7维) - 绝对坐标，但经过symlog
                obs_components.append(self._symlog(my_state))
                obs_components.append(np.array([all_fuels[agent_id] / self._config.p_init_dv]))

                # 2. 队友信息 (每个队友 7维) - 相对坐标
                other_pursuer_info = []
                for i in range(self._config.num_p):
                    pursuer_id = f'p_{i}'
                    if pursuer_id != agent_id:
                        if pursuer_id in all_states:
                            other_state = all_states[pursuer_id]
                            # 相对状态 = 对方状态 - 自身状态
                            rel_state = other_state - my_state
                            other_fuel = all_fuels[pursuer_id] / self._config.p_init_dv
                            # 对相对状态进行symlog
                            other_pursuer_info.append(np.concatenate([
                                self._symlog(rel_state),
                                [other_fuel]
                            ]))
                        else:
                            # 如果队友不存在，用0填充
                            other_pursuer_info.append(np.zeros(7))
                
                if other_pursuer_info:
                    obs_components.append(np.concatenate(other_pursuer_info))
                elif self._config.num_p > 1:
                    # 确保在没有其他队友时维度仍然正确
                    obs_components.append(np.zeros(7 * (self._config.num_p - 1)))

                observations[agent_id] = np.concatenate(obs_components)
            else: # Evader
                observations[agent_id] = self._symlog(self.states[agent_id])
        
        return observations

    def _update_history_and_prepare_data(self):
        """
        核心函数：更新历史缓冲区，并为每个追踪者准备相对的、symlog处理过的轨迹数据。
        """
        for evader_id in [f'e_{i}' for i in range(self._config.num_e)]:
            if evader_id not in self.states: continue

            self.obs_counters[evader_id] += 1
            history_buffer = self.evader_history_buffers[evader_id]
            mask_buffer = self.evader_history_mask_buffers[evader_id]
            
            # 如果是观测步，存入真实状态和掩码1；否则，重复上一帧状态并存入掩码0
            if self.obs_counters[evader_id] % self._config.obs_interval == 0:
                history_buffer.append(self.states[evader_id])
                mask_buffer.append(1.0)
            else:
                if len(history_buffer) > 0:
                    history_buffer.append(history_buffer[-1]) # 重复最后一个已知状态
                    mask_buffer.append(0.0) # 标记为非真实观测
                else: # 缓冲区为空的罕见情况
                    history_buffer.append(self.states[evader_id])
                    mask_buffer.append(1.0)

            # 将历史轨迹（绝对物理坐标）转换为numpy数组
            history_abs_real = np.array(list(history_buffer)) # Shape: [history_len, 6]
            history_mask = np.array(list(mask_buffer)) # Shape: [history_len]

            # 为每个追踪者生成其“相对视野”下的历史轨迹
            for p_agent_id in [f'p_{i}' for i in range(self._config.num_p)]:
                if p_agent_id in self.states:
                    my_current_pos = self.states[p_agent_id][:3]
                    my_current_vel = self.states[p_agent_id][3:]
                    my_current_state_vec = np.concatenate([my_current_pos, my_current_vel])

                    # [核心逻辑]：输入 = Symlog(Evader历史 - 我当前状态)
                    rel_history_real = history_abs_real - my_current_state_vec
                    rel_history_symlog = self._symlog(rel_history_real)

                    # 将处理好的数据放入info字典
                    sl_data = {
                        f'history_input_{evader_id}': rel_history_symlog,
                        f'history_mask_{evader_id}': history_mask,
                    }
                    
                    if p_agent_id in self.infos:
                        self.infos[p_agent_id].update(sl_data)
                    else:
                        self.infos[p_agent_id] = sl_data

    def get_evader_actions(self):
        """
        获取逃逸者的动作。
        """
        evader_actions = {}
        for i in range(self._config.num_e):
            evader_id = f'e_{i}'
            if evader_id in self.agents:
                # 逃逸者随机机动
                action = np.random.randn(3)
                action_norm = np.linalg.norm(action)
                if action_norm > self._config.e_dv_step:
                    action = action / action_norm * self._config.e_dv_step
                
                if np.linalg.norm(action) > self.remain_Dvs[evader_id]:
                    action = action / np.linalg.norm(action) * self.remain_Dvs[evader_id]
                
                evader_actions[evader_id] = action
        return evader_actions
