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
# 导入LSTM模型
from .lstm import TrajectoryPredictor

# 导入新的lambert求解器
try:
    from lambert_solver import solve_lambert
except ImportError:
    print("\033[93mWarning: C++ Lambert solver not found. Lambert-based rewards will be disabled.\033[0m")
    solve_lambert = None

@dataclass
class MPE_POMDP_EnvCfg(MPEEnvCfg):
    """加入POMDP和LSTM的特定参数"""
    use_partial_obs: bool = True
    obs_interval: int = 2
    lstm_history_len: int = 20
    lstm_future_len: int = 10 # Note: This is now unused for distillation
    lstm_scheme: int = 2  

    # Lambert奖励配置 
    use_lambert_reward: bool = True
    lambert_reward_weight: float = 5
    lambert_transfer_time: float = 7200#Lambert转移时间（秒）
    mu: float = 3.986004418e14

class MPE_POMDP_Env(MPEEnv):
    """
    继承自MPEEnv
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

        self.lstm_model = None
        self.device = None 
        if self._config.use_partial_obs:
            self.evader_history_buffers = {f'e_{i}': deque(maxlen=self._config.lstm_history_len) for i in range(self._config.num_e)}
            self.obs_counters = {f'e_{i}': 0 for i in range(self._config.num_e)}

            if self._config.lstm_scheme in [2, 3]:
                self.virtual_star_states = {f'e_{i}': np.zeros(6) for i in range(self._config.num_e)}
                self.virtual_star_history_buffers = {f'e_{i}': deque(maxlen=self._config.lstm_history_len) for i in range(self._config.num_e)}

            # 重新定义观测空间
            self.observation_spaces = {}
            for a in self.possible_agents:
                if a.startswith('p_'):
                    # 自身维度: 6(状态) + 1(燃料)
                    self_obs_dim = 7
                    # 队友维度: (N-1) * (3位置 + 3速度 + 1燃料)
                    other_pursuers_obs_dim = 7 * (self._config.num_p - 1)
                    # 学生网络的观测只包含自身和队友，LSTM的输出在网络内部处理
                    obs_shape = (self_obs_dim + other_pursuers_obs_dim,)
                    self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=obs_shape)
                else:
                    self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=(6,))

    def set_policy_lstm(self, lstm_model: TrajectoryPredictor):
        self.lstm_model = lstm_model
        try:
            self.device = next(self.lstm_model.parameters()).device
            print(f"LSTM model has been set in the POMDP environment on device: {self.device}")
        except StopIteration:
            print("Warning: LSTM model has no parameters. Cannot determine device.")

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed, options)

        if self._config.use_partial_obs:
            self.obs_counters = {f'e_{i}': 0 for i in range(self._config.num_e)}
            
            for evader_id in [f'e_{i}' for i in range(self._config.num_e)]:
                self.evader_history_buffers[evader_id].clear()
                if evader_id in self.states:
                    initial_state_normalized = self._symlog(self.states[evader_id])
                    for _ in range(self._config.lstm_history_len):
                        self.evader_history_buffers[evader_id].append(initial_state_normalized)
                    
                    if self._config.lstm_scheme in [2, 3]:
                        self.virtual_star_states[evader_id] = np.copy(self.states[evader_id])
                        self.virtual_star_history_buffers[evader_id].clear()
                        vs_initial_state_normalized = self._symlog(self.virtual_star_states[evader_id])
                        for _ in range(self._config.lstm_history_len):
                            self.virtual_star_history_buffers[evader_id].append(vs_initial_state_normalized)

            self._update_history_and_prepare_data()
            observations = self._get_observations()

            all_true_states = []
            for agent_id in self.possible_agents:
                if agent_id in self.states:
                    norm_state = self._symlog(self.states[agent_id])
                    all_true_states.append(norm_state)
                else:
                    all_true_states.append(np.zeros(6))
            privileged_state = np.concatenate(all_true_states)

            for agent in self.agents:
                if agent.startswith('p_'):
                    infos[agent]['privileged_state'] = privileged_state

        return observations, infos

    def step(self, actions):
        if self._config.use_partial_obs:
            self._update_history_and_prepare_data()
        
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

        if self._config.use_partial_obs and self._config.lstm_scheme in [2, 3]:
            for evader_id in [f'e_{i}' for i in range(self._config.num_e)]:
                if evader_id in self.virtual_star_states:
                    vs_time = self._time - datetime.timedelta(seconds=self._config.dt)
                    _, new_vs_state = self._orbit_lib.orbit_hpop(vs_time, self.virtual_star_states[evader_id], self._config.dt, self._config.hpop_in)
                    self.virtual_star_states[evader_id] = new_vs_state

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

        all_true_states = []
        for agent_id in self.possible_agents:
            if agent_id in self.states:
                norm_state = self._symlog(self.states[agent_id])
                all_true_states.append(norm_state)
            else:
                all_true_states.append(np.zeros(6))
        privileged_state = np.concatenate(all_true_states)

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
                my_pos = my_state[:3]
                
                # 1. 自身信息 (7维)
                obs_components.append(self._symlog(my_state))
                obs_components.append(np.array([all_fuels[agent_id] / self._config.p_init_dv]))

                # 2. 队友信息 (每个队友 7维)
                other_pursuer_info = []
                for i in range(self._config.num_p):
                    pursuer_id = f'p_{i}'
                    if pursuer_id != agent_id:
                        if pursuer_id in all_states:
                            other_state = all_states[pursuer_id]
                            rel_pos = self._symlog(other_state[:3] - my_pos)
                            rel_vel = self._symlog(other_state[3:] - my_state[3:])
                            other_fuel = all_fuels[pursuer_id] / self._config.p_init_dv
                            other_pursuer_info.append(np.concatenate([rel_pos, rel_vel, [other_fuel]]))
                        else:
                            other_pursuer_info.append(np.zeros(7))
                
                if other_pursuer_info:
                    obs_components.append(np.concatenate(other_pursuer_info))
                elif self._config.num_p > 1:
                    obs_components.append(np.zeros(7 * (self._config.num_p - 1)))

                # 最终观测不包含LSTM预测，它将在网络内部与LSTM特征融合
                observations[agent_id] = np.concatenate(obs_components)
            else:
                observations[agent_id] = self._symlog(self.states[agent_id])
        
        return observations

    def _update_history_and_prepare_data(self):
        """仅维护历史观测buffer，并将其放入info字典"""
        for evader_id in [f'e_{i}' for i in range(self._config.num_e)]:
            if evader_id not in self.states: continue

            self.obs_counters[evader_id] += 1
            history_buffer = self.evader_history_buffers[evader_id]
            
            # 如果是观测步，存入真实状态；否则，重复上一帧状态
            if self.obs_counters[evader_id] % self._config.obs_interval == 0:
                true_state_normalized = self._symlog(self.states[evader_id])
                history_buffer.append(true_state_normalized)
                if self._config.lstm_scheme in [2, 3]:
                    vs_state_normalized = self._symlog(self.virtual_star_states[evader_id])
                    self.virtual_star_history_buffers[evader_id].append(vs_state_normalized)
            else:
                if len(history_buffer) > 0:
                    history_buffer.append(history_buffer[-1])
                if self._config.lstm_scheme in [2, 3] and len(self.virtual_star_history_buffers[evader_id]) > 0:
                    self.virtual_star_history_buffers[evader_id].append(self.virtual_star_history_buffers[evader_id][-1])

            # 准备LSTM输入数据
            history_abs_normalized = np.array(list(history_buffer))
            input_data_for_lstm = history_abs_normalized
            if self._config.lstm_scheme in [2, 3]:
                vs_history_normalized = np.array(list(self.virtual_star_history_buffers[evader_id]))
                if len(vs_history_normalized) == len(history_abs_normalized):
                    input_data_for_lstm = history_abs_normalized - vs_history_normalized
            
            sl_data = {f'lstm_history_input_{evader_id}': input_data_for_lstm}
            
            for p_agent_id in [f'p_{i}' for i in range(self._config.num_p)]:
                 if p_agent_id in self.infos:
                    self.infos[p_agent_id].update(sl_data)