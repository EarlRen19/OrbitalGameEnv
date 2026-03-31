"""Single-agent wrapper for OGE + skrl PPO."""

from __future__ import annotations
from typing import Any
import numpy as np
import torch
import gymnasium
from skrl.envs.wrappers.torch.base import Wrapper


class OGESingleEnvWrapper(Wrapper):
    """Wraps OGEEnv for single-agent PPO (controls red_sat)."""

    def __init__(self, env, cfg) -> None:
        super().__init__(env)
        self._oge = env.oge
        self._dv_max_blue = cfg.dv_max_per_step_blue

        # Blue sat discrete action set
        dv = self._dv_max_blue
        self._blue_actions = np.array([
            [dv, 0, 0], [-dv, 0, 0], [0, dv, 0],
            [0, -dv, 0], [0, 0, dv], [0, 0, -dv], [0, 0, 0]
        ], dtype=np.float32)

        self._obs_size = self._oge.get_obs_size()
        # 精炼观测维度：
        # [0:3] rel_pos_lvlh / 200km  (3)
        # [3:6] rel_vel_lvlh * 10     (3)  (m/s, *10 使量级~1)
        # [6]   dist / 20km           (1)
        # [7]   solar_angle / pi      (1)
        # [8:11] sun_dir_target_lvlh  (3)
        # [11]  dv_remain / dv_init   (1)
        # [12]  time_progress         (1)
        self._single_obs_size = 13
        self._last_obs = None

        # 侦照任务参数
        self._recon_dist_threshold = 20.0  # km
        self._recon_angle_threshold = np.deg2rad(60)  # 60度
        self._recon_time_target = 120.0  # 秒
        self._timestep = cfg.timestep
        self._recon_time_accumulated = 0.0
        self._init_fuel = cfg.dv_init_red  # km/s
        self._dv_max_red = cfg.dv_max_per_step_red  # km/s
        self._last_dist = 200.0
        self._last_in_recon_zone = False  # 追踪上一步是否在侦照区

        # Override spaces for single red_sat agent
        self._observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._single_obs_size,), dtype=np.float32
        )
        self._action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def num_envs(self) -> int:
        return 1

    def _get_refined_obs(self, obs_np):
        """
        构造 13 维精炼观测（red_sat视角）：
        [0:3]  rel_pos_lvlh / 200      (km, 200km→1.0)
        [3:6]  rel_vel_lvlh * 10       (m/s, 0.1m/s→1.0)
        [6]    dist / 20               (km, 20km→1.0)
        [7]    solar_angle / pi        (rad, pi→1.0)
        [8:11] sun_dir in target LVLH  (单位向量)
        [11]   dv_remain / dv_init     (0~1)
        [12]   time_progress           (0~1)
        """
        red_obs = obs_np[1]
        rel_pos_lvlh = red_obs[6:9]          # km
        rel_vel_lvlh = red_obs[11:14]        # m/s (已由C++乘1000)
        dist_norm    = red_obs[14]           # dist/20km
        solar_angle  = red_obs[9]            # rad
        sun_dir      = red_obs[15:18]        # 单位向量
        dv_ratio     = red_obs[19]           # dv_remain/dv_init
        time_prog    = red_obs[18]           # 0~1

        obs = np.zeros(13, dtype=np.float32)
        obs[0:3]  = rel_pos_lvlh / 200.0
        obs[3:6]  = rel_vel_lvlh * 10.0
        obs[6]    = dist_norm
        obs[7]    = solar_angle / np.pi
        obs[8:11] = sun_dir
        obs[11]   = dv_ratio
        obs[12]   = time_prog
        return obs

    def reset(self, seed=None, options=None):
        obs_np, info = self._env.reset(seed=seed, options=options)
        self._last_obs = np.asarray(obs_np, dtype=np.float32)
        self._recon_time_accumulated = 0.0
        self._last_in_recon_zone = False
        refined_obs = self._get_refined_obs(self._last_obs)
        red_obs = self._last_obs[1]
        self._last_dist = np.linalg.norm(red_obs[6:9])
        return torch.tensor(refined_obs, dtype=torch.float32).unsqueeze(0).to(self.device), info

    def step(self, actions):
        red_action_scaled = actions.squeeze().cpu().numpy() * self._dv_max_red
        blue_action = self._blue_actions[np.random.randint(len(self._blue_actions))]
        combined_actions = np.vstack([blue_action, red_action_scaled])
        self._oge.act(combined_actions)
        obs_np = self._oge.get_observations()
        truncated = bool(self._oge.is_truncated())
        info = {"current_time": self._oge.get_current_time()}

        self._last_obs = np.asarray(obs_np, dtype=np.float32)
        red_obs = self._last_obs[1]

        reward_red, custom_done = self._compute_recon_reward(red_obs, red_action_scaled)
        terminated = custom_done

        refined_obs = self._get_refined_obs(self._last_obs)
        obs_t = torch.tensor(refined_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        rew_t = torch.tensor([[reward_red]], dtype=torch.float32).to(self.device)
        term_t = torch.tensor([[terminated]], dtype=torch.bool).to(self.device)
        trunc_t = torch.tensor([[truncated]], dtype=torch.bool).to(self.device)

        return obs_t, rew_t, term_t, trunc_t, info

    def _compute_recon_reward(self, obs, action):
        """侦照任务奖励：第一次进入侦照区给奖励，持续120s再给大奖励。"""
        target_lvlh = obs[6:9]
        solar_angle = obs[9]
        dv_remain = obs[10]
        rel_vel_lvlh = obs[11:14]  # m/s (C++已乘1000)

        distance = np.linalg.norm(target_lvlh)          # km
        relative_vel_ms = np.linalg.norm(rel_vel_lvlh)  # m/s
        action_ms = np.linalg.norm(action) * 1000.0     # m/s
        dv_remain_ms = dv_remain * 1000.0               # m/s

        reward = 0.0
        done = False

        # --- 1. 终端判定 ---
        in_recon_zone = (distance <= self._recon_dist_threshold and
                         solar_angle <= self._recon_angle_threshold)
        is_out_of_fuel = (dv_remain <= 0.0)

        if in_recon_zone:
            if not self._last_in_recon_zone:
                # 第一次进入侦照区：给初始奖励
                reward = 100.0 + (dv_remain / self._init_fuel) * 30.0
                self._recon_time_accumulated = self._timestep
                self._last_in_recon_zone = True
            else:
                # 持续在侦照区：累计时间
                self._recon_time_accumulated += self._timestep
                reward = 5.0  # 每步小奖励鼓励保持

                # 累计满120s：给大奖励并终止
                if self._recon_time_accumulated >= self._recon_time_target:
                    reward = 200.0 + (dv_remain / self._init_fuel) * 50.0
                    done = True

            self._last_dist = distance
            return reward, done
        else:
            # 离开侦照区：重置状态
            self._recon_time_accumulated = 0.0
            self._last_in_recon_zone = False

        if is_out_of_fuel:
            reward = -40.0
            done = True
            return reward, done

        # --- 2. 密集奖励 ---

        # A. 距离引导
        dist_change = self._last_dist - distance
        dist_reward = 2.0 * dist_change

        # B. 25km内加入角度引导和速度漏斗约束
        angle_guide = 0.0
        vel_penalty = 0.0
        if distance <= 25.0:
            # 角度引导
            angle_guide = 2.0 * (1.0 - solar_angle / self._recon_angle_threshold)

            # 速度漏斗：25km允许5m/s，0km要求1m/s，线性插值
            target_vel_ms = 1.0 + (distance / 25.0) * 4.0  # 1 + (d/25)*4 = [1, 5]
            if relative_vel_ms > target_vel_ms:
                vel_penalty = -0.2 * (relative_vel_ms - target_vel_ms)

        # C. 燃料惩罚
        fuel_penalty = -0.01 * action_ms

        reward = dist_reward + angle_guide + vel_penalty + fuel_penalty

        # --- 3. 更新历史状态 ---
        self._last_dist = distance

        return reward, done

    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass


class OGESingleEnvWrapper_operate(Wrapper):
    """Wraps OGEEnv for single-agent PPO (controls red_sat) - Operate Task."""

    def __init__(self, env, cfg) -> None:
        super().__init__(env)
        self._oge = env.oge
        self._dv_max_blue = cfg.dv_max_per_step_blue
        
        # Blue sat discrete action set
        dv = self._dv_max_blue
        self._blue_actions = np.array([
            [dv, 0, 0], [-dv, 0, 0], [0, dv, 0],
            [0, -dv, 0], [0, 0, dv], [0, 0, -dv], [0, 0, 0]
        ], dtype=np.float32)

        # 【精简观测】：仅保留相对量和燃料，共 7 维
        # [0-2: Rel_R_LVLH, 3-5: Rel_V_LVLH, 6: Fuel_Scaled]
        self._single_obs_size = 7
        self._last_obs = None

        # 操控任务参数
        self._operate_dist_threshold = 2.0   # km
        self._operate_vel_threshold = 1.0    # m/s
        self._init_fuel = cfg.dv_init_red    # km/s
        self._dv_max_red = cfg.dv_max_per_step_red  # km/s
        self._timestep = cfg.timestep
        
        self._last_dist = 200.0
        self._last_vel_ms = 0.0

        self._curriculum_step = 0

        self._observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._single_obs_size,), dtype=np.float32
        )
        
        self._action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def num_envs(self) -> int:
        return 1

    def _get_refined_obs(self, obs_np):
        """
        构造 7 维精炼观测：
        [0-2: Rel_R_LVLH/100, 3-5: Rel_V_LVLH*10, 6: Fuel/init_fuel]
        """
        blue_obs = obs_np[0]
        red_obs = obs_np[1]
        
        r_red = red_obs[0:3]
        v_red = red_obs[3:6]
        r_blue = blue_obs[0:3]
        v_blue = blue_obs[3:6]
        r_rel_lvlh = red_obs[6:9]
        fuel = red_obs[10]
        
        # 计算 LVLH 旋转矩阵 
        z = -r_red / (np.linalg.norm(r_red) + 1e-9)
        h = np.cross(r_red, v_red)
        y = -h / (np.linalg.norm(h) + 1e-9)
        x = np.cross(y, z)
        rotation_matrix = np.vstack([x, y, z])
        
        v_rel_j2000 = v_blue - v_red
        v_rel_lvlh = rotation_matrix @ v_rel_j2000
        
        # 归一化处理
        obs_7 = np.zeros(7, dtype=np.float32)
        obs_7[0:3] = r_rel_lvlh / 100.0  # 200km -> 2.0
        obs_7[3:6] = v_rel_lvlh * 10.0   # 0.01km/s -> 0.1
        obs_7[6] = fuel / self._init_fuel # 0-1 之间
        
        return obs_7

    def reset(self, seed=None, options=None):
        obs_np, info = self._env.reset(seed=seed, options=options)
        self._last_obs = np.asarray(obs_np, dtype=np.float32)
        
        refined_obs = self._get_refined_obs(self._last_obs)
        
        # 初始化状态
        red_obs = self._last_obs[1]
        blue_obs = self._last_obs[0]
        self._last_dist = np.linalg.norm(red_obs[6:9])
        self._last_vel_ms = np.linalg.norm(red_obs[3:6] - blue_obs[3:6]) * 1000.0
        
        return torch.tensor(refined_obs, dtype=torch.float32).unsqueeze(0).to(self.device), info

    def step(self, actions):
        self._curriculum_step += 1
        #加入课程学习:
        # if self._curriculum_step <=self._timestep*0.25:
        #     self._operate_dist_threshold = 80.0
        # elif self._curriculum_step <=self._timestep*0.5:
        #     self._operate_dist_threshold = 30.0
        # elif self._curriculum_step <= self._timestep*0.75:
        #     self._operate_dist_threshold = 10.0
        # else:
        #     self._operate_dist_threshold = 2.0


        red_action_scaled = actions.squeeze().cpu().numpy() * self._dv_max_red
        blue_action = self._blue_actions[np.random.randint(len(self._blue_actions))]
        combined_actions = np.vstack([blue_action, red_action_scaled])
        
        self._oge.act(combined_actions)
        obs_np = self._oge.get_observations()
        
        truncated = bool(self._oge.is_truncated())
        info = {"current_time": self._oge.get_current_time()}

        self._last_obs = np.asarray(obs_np, dtype=np.float32)
        blue_obs = self._last_obs[0]
        red_obs = self._last_obs[1]

        refined_obs = self._get_refined_obs(self._last_obs)
        reward_red, custom_done = self._compute_operate_reward(red_obs, blue_obs, red_action_scaled)#传入的是正常的obs
        
        terminated = custom_done

        obs_t = torch.tensor(refined_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        rew_t = torch.tensor([[reward_red]], dtype=torch.float32).to(self.device)
        term_t = torch.tensor([[terminated]], dtype=torch.bool).to(self.device)
        trunc_t = torch.tensor([[truncated]], dtype=torch.bool).to(self.device)

        return obs_t, rew_t, term_t, trunc_t, info


    def _compute_operate_reward(self, red_obs, blue_obs, action):
        red_vel = red_obs[3:6]
        blue_vel = blue_obs[3:6]
        target_lvlh = red_obs[6:9]
        dv_remain = red_obs[10] 

        distance = np.linalg.norm(target_lvlh) # km
        relative_vel_ms = np.linalg.norm(red_vel - blue_vel) * 1000.0 # m/s
        dv_remain_ms = dv_remain * 1000.0 # m/s
        action_ms = np.linalg.norm(action) * 1000.0 # m/s

        reward = 0.0
        done = False

        # --- 1. 终端判定 (恢复速度约束) ---
        is_close = (distance <= self._operate_dist_threshold)
        #is_slow = (relative_vel_ms <= self._operate_vel_threshold)
        is_out_of_fuel = (dv_remain <= 0.0)

        # if is_close:
        #     if is_slow:
        #         # 真正的成功：高奖励
        #         reward = 100.0 + (dv_remain / self._init_fuel) * 50.0
        #         done = True
        #         return reward, done
        #     else:
        #         # 虽近但快：给予警告性惩罚，不一定结束，让它学会减速
        #         reward = -10.0
        #         # 如果实在太快（比如 > 10m/s）判定为碰撞失败
        #         if relative_vel_ms > 10.0:
        #             reward = -50.0
        #             done = True
        #         return reward, done

        if is_close:
        
            # 真正的成功：高奖励
            reward = 200.0 + (dv_remain / self._init_fuel) * 50.0
            done = True
            return reward, done

        if is_out_of_fuel:
            reward = -20.0
            done = True
            return reward, done

        # --- 2. 密集奖励 (Dense Rewards) ---
        
        # A. 距离引导：加大系数 (1.0 per km)，并引入非线性靠近奖励
        dist_change = self._last_dist - distance
        dist_reward = 2.0 * dist_change #* np.exp(-(distance-200) / 200.0) 
        
        # B. 速度匹配引导 (Velocity Matching)
        # 越近越要求速度小。
        target_vel_ms = 0.0
        if distance > 20.0:
            target_vel_ms = 10.0 # 远距离允许较快
        elif distance > 5.0:
            target_vel_ms = 5.0
        else:
            target_vel_ms = 1.0
            
        vel_penalty = 0.0
        if relative_vel_ms > target_vel_ms:
            vel_penalty = -0.1 * (relative_vel_ms - target_vel_ms)

        # C. 燃料惩罚：轻微惩罚以保持效率
        fuel_penalty = -0.01 * action_ms

        reward = dist_reward  + fuel_penalty

        # --- 3. 更新历史状态 ---
        self._last_dist = distance
        self._last_vel_ms = relative_vel_ms

        return reward, done


    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass