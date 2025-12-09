from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces
from collections import deque
import torch
import datetime
import ctypes
import os

# 导入父类
from .mpe_env import MPEEnv, MPEEnvCfg

# ================= Ctypes Interface Start =================
# 尝试加载库，失败则回退到 ECI
try:
    # 请根据实际路径修改
    so_path = "/home/star/Downloads/gemini-cli-main/OrbitalGameEnv/demos/OrbitLib/so/X86/libOrbit.so"
    if not os.path.exists(so_path):
        # 尝试相对路径
        so_path = os.path.join(os.path.dirname(__file__), "..", "OrbitLib", "so", "X86", "libOrbit.so")
    
    if os.path.exists(so_path):
        orbit_lib_c = ctypes.CDLL(so_path)
    else:
        raise FileNotFoundError("libOrbit.so not found")
        
except Exception as e:
    print(f"[93mWarning: Failed to load libOrbit.so ({e}). LVLH transformation disabled.")
    orbit_lib_c = None

if orbit_lib_c:
    orbit_lib_c.DCM_J2000_to_LVLH.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER((ctypes.c_double * 3) * 3)]
    orbit_lib_c.DCM_J2000_to_LVLH.restype = None

def get_lvlh_dcm(state_j2000: np.ndarray) -> np.ndarray | None:
    if not orbit_lib_c: return None
    rv_in = state_j2000.astype(np.float64)
    dcm_out = ((ctypes.c_double * 3) * 3)()
    orbit_lib_c.DCM_J2000_to_LVLH(rv_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), dcm_out)
    return np.array([[dcm_out[i][j] for j in range(3)] for i in range(3)])

def eci_to_lvlh_relative(observer_state, target_pos, target_vel=None):
    """ 计算目标相对于观察者的 LVLH 坐标 """
    dcm = get_lvlh_dcm(observer_state)
    observer_pos = observer_state[:3]
    rel_pos_eci = target_pos - observer_pos
    
    if dcm is None: # 回退模式
        rel_vel_eci = (target_vel - observer_state[3:]) if target_vel is not None else None
        return rel_pos_eci, rel_vel_eci

    rel_pos_lvlh = dcm @ rel_pos_eci
    rel_vel_lvlh = None
    if target_vel is not None:
        rel_vel_eci = target_vel - observer_state[3:]
        rel_vel_lvlh = dcm @ rel_vel_eci
        
    return rel_pos_lvlh, rel_vel_lvlh
# ================= Ctypes Interface End =================


@dataclass
class MPE_POMDP_EnvCfg(MPEEnvCfg):
    use_partial_obs: bool = True
    obs_interval: int = 2
    history_len: int = 20
    # Lambert 相关参数保留...
    use_lambert_reward: bool = True
    lambert_reward_weight: float = 5
    GEO_ORBIT_RADIUS: float = 42164000.0

class MPE_POMDP_Env(MPEEnv):
    @staticmethod
    def _symlog(x):
        return np.sign(x) * np.log(np.abs(x) + 1.0)

    def __init__(self, config: MPE_POMDP_EnvCfg = MPE_POMDP_EnvCfg()):
        super().__init__(config)
        self._config: MPE_POMDP_EnvCfg = config

        if self._config.use_partial_obs:
            self.evader_history_buffers = {f'e_{i}': deque(maxlen=self._config.history_len) for i in range(self._config.num_e)}
            self.obs_counters = {f'e_{i}': 0 for i in range(self._config.num_e)}
            self.evader_history_mask_buffers = {f'e_{i}': deque(maxlen=self._config.history_len) for i in range(self._config.num_e)}

            self.observation_spaces = {}
            for a in self.possible_agents:
                if a.startswith('p_'):
                    self_obs_dim = 8
                    teammates_obs_dim = 7 * (self._config.num_p - 1)
                    obs_shape = (self_obs_dim + teammates_obs_dim,)
                    self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=obs_shape)
                else:
                    self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=(6,))

    def _get_privileged_state(self):
        """辅助函数：计算基于 Anchor LVLH 的特权观测"""
        privileged_components = []
        anchor_id = self.evader_ids[0]
        
        anchor_state = np.zeros(6)
        anchor_dcm = None

        if anchor_id in self.states:
            anchor_state = self.states[anchor_id]
            anchor_dcm = get_lvlh_dcm(anchor_state)
            # Anchor 自身依然保留 symlog 的绝对状态 (或者改为相对于GEO理想轨道的偏差)
            privileged_components.append(self._symlog(anchor_state))
        else:
            privileged_components.append(np.zeros(6))

        # 顺序：所有追击者 -> 剩余逃逸者
        agent_order = self.pursuer_ids + [eid for eid in self.evader_ids if eid != anchor_id]
        
        for agent_id in agent_order:
            if agent_id in self.states:
                target_state = self.states[agent_id]
                
                # 如果有DCM，则转为LVLH相对；否则使用ECI相对
                if anchor_dcm is not None:
                    diff_pos = target_state[:3] - anchor_state[:3]
                    diff_vel = target_state[3:] - anchor_state[3:]
                    rel_pos = anchor_dcm @ diff_pos
                    rel_vel = anchor_dcm @ diff_vel
                    rel_state = np.concatenate([rel_pos, rel_vel])
                else:
                    rel_state = target_state - anchor_state
                
                privileged_components.append(self._symlog(rel_state))
            else:
                privileged_components.append(np.zeros(6))
        
        return np.concatenate(privileged_components)

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed, options)

        if self._config.use_partial_obs:
            self.obs_counters = {f'e_{i}': 0 for i in range(self._config.num_e)}
            for evader_id in self.evader_ids:
                self.evader_history_buffers[evader_id].clear()
                self.evader_history_mask_buffers[evader_id].clear()
                if evader_id in self.states:
                    initial_state = self.states[evader_id]
                    for _ in range(self._config.history_len):
                        self.evader_history_buffers[evader_id].append(initial_state)
                        self.evader_history_mask_buffers[evader_id].append(1.0)

            self._update_history_and_prepare_data()
            observations = self._get_observations()

            # 使用新的 LVLH 特权观测
            privileged_state = self._get_privileged_state()
            for agent in self.pursuer_ids:
                if agent not in infos: infos[agent] = {}
                infos[agent]['privileged_state'] = privileged_state

        return observations, infos

    def step(self, actions):
        self.step_count += 1
        # 在step执行前，基于上一步的状态准备好给agent的输入
        if self._config.use_partial_obs:
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

        # 观测和奖励计算
        observations = self._get_observations()
        rewards, debug_reward_info = self._get_rewards(clipped_actions)
        terminations, termination_reasons = self._get_terminations()
        truncations = self._get_truncations()
        self.terminations, self.truncations = terminations, truncations

        # 准备Infos
        current_infos = {a: self.infos.get(a, {}) for a in self.possible_agents if a in self.agents}
        
        if any(terminations.values()) or any(truncations.values()):
            self.episode_statistics['total_episodes'] += 1
            reason = list(termination_reasons.values())[0] if termination_reasons else 'unknown'
            
            if reason == 'capture_success': self.episode_statistics['success_count'] += 1
            elif reason == 'timeout': self.episode_statistics['timeout_count'] += 1
            elif reason == 'fuel_out': self.episode_statistics['fuelout_count'] += 1
            
            if self.episode_statistics['total_episodes'] > 0:
                self.episode_statistics['success_rate'] = self.episode_statistics['success_count'] / self.episode_statistics['total_episodes']

        # 更新特权信息和最终观测
        privileged_state = self._get_privileged_state()
        for agent in self.possible_agents:
            if agent not in current_infos: current_infos[agent] = {}
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


    def _get_observations(self):
        if not self._config.use_partial_obs:
            return super()._get_observations()

        observations = {}
        all_states = {aid: self.states[aid] for aid in self.possible_agents if aid in self.states}
        
        for agent_id in self.pursuer_ids:
            if agent_id not in all_states: continue
            
            my_state = all_states[agent_id]
            
            # 1. 自身信息 (Symlog处理)
            my_pos = my_state[:3]
            my_vel = my_state[3:]
            my_dist = np.linalg.norm(my_pos)
            alt_dev = self._symlog(np.array([my_dist - self._config.GEO_ORBIT_RADIUS]))
            pos_dir = my_pos / (my_dist + 1e-6)
            vel_sym = self._symlog(my_vel)
            fuel = np.array([self.remain_Dvs.get(agent_id, 0.0) / self._config.p_init_dv])
            
            obs_list = [alt_dev, pos_dir, vel_sym, fuel]

            # 2. 队友信息 (LVLH 转换)
            teammate_data = []
            for i in range(self._config.num_p):
                tid = f'p_{i}'
                if tid != agent_id:
                    if tid in all_states:
                        t_state = all_states[tid]
                        # 转换！
                        rel_pos, rel_vel = eci_to_lvlh_relative(my_state, t_state[:3], t_state[3:])
                        t_fuel = np.array([self.remain_Dvs.get(tid, 0.0) / self._config.p_init_dv])
                        
                        teammate_data.append(np.concatenate([
                            self._symlog(rel_pos), 
                            self._symlog(rel_vel), 
                            t_fuel
                        ]))
                    else:
                        teammate_data.append(np.zeros(7))
            
            if teammate_data:
                obs_list.append(np.concatenate(teammate_data))
                
            observations[agent_id] = np.concatenate(obs_list)

        # 逃逸者观测保持原样
        for eid in self.evader_ids:
            if eid in all_states:
                observations[eid] = all_states[eid]
                
        return observations

    def _update_history_and_prepare_data(self):
        # 这里的逻辑与你之前的代码一致，通过 eci_to_lvlh_relative 处理历史数据
        # 确保 history buffer 更新逻辑正确
        for evader_id in self.evader_ids:
            if evader_id not in self.states: continue

            self.obs_counters[evader_id] += 1
            h_buf = self.evader_history_buffers[evader_id]
            m_buf = self.evader_history_mask_buffers[evader_id]
            
            # 更新 Buffer (存绝对 ECI)
            if self.obs_counters[evader_id] % self._config.obs_interval == 0:
                h_buf.append(self.states[evader_id])
                m_buf.append(1.0)
            else:
                if h_buf:
                    h_buf.append(h_buf[-1])
                    m_buf.append(0.0)
                else:
                    h_buf.append(self.states[evader_id])
                    m_buf.append(1.0)

            # 准备 Transformer 输入 (转为相对 LVLH)
            hist_eci = np.array(list(h_buf))
            hist_mask = np.array(list(m_buf))

            for pid in self.pursuer_ids:
                if pid in self.states:
                    my_state = self.states[pid]
                    
                    processed_hist = []
                    for h_state in hist_eci:
                        rp, rv = eci_to_lvlh_relative(my_state, h_state[:3], h_state[3:])
                        processed_hist.append(np.concatenate([rp, rv]))
                    
                    rel_hist_symlog = self._symlog(np.array(processed_hist))
                    
                    # 存入 self.infos 供 step 合并
                    if pid not in self.infos: self.infos[pid] = {}
                    self.infos[pid].update({
                        f'history_input_{evader_id}': rel_hist_symlog,
                        f'history_mask_{evader_id}': hist_mask
                    })
    
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