"""Fine-tune a trained PPO agent on a fixed initial state scenario.

针对固定初始化场景（Fixed Init）的微调脚本。
- 不修改 env_wrapper.py，不影响正常训练流程
- 奖励函数与 OGESingleEnvWrapper 完全对齐（120s 累计侦照）
- 角度引导提前到 50km，解决接近时太阳角恶化问题
- 支持小范围扰动初始化，防止过拟合单一状态
"""

import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import torch
import numpy as np
from copy import deepcopy
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch.base import Wrapper
from oge_py import OGEEnv
import oge_py
import gymnasium
from modules.networks import Policy, Value
from configs.env_cfg import env_cfg
from configs.ppo_cfg import ppo_cfg as base_ppo_cfg
from skrl.resources.preprocessors.torch import RunningStandardScaler


# ── 固定初始轨道根数 ──────────────────────────────────────────────────────────
FIXED_INIT = {
    "red": {
        "sma": 42060.338261, "ecc": 0.003001, "incl": 0.002287,
        "raan": 1.592759, "argp": 3.303797, "ma": 1.033423,
    },
    "blue": {
        "sma": 42169.502913, "ecc": 0.0, "incl": 0.002287,
        "raan": 1.592829, "argp": 0.0, "ma": 4.345423,
    },
}


def ma2ta(ma, ecc, tol=1e-10, max_iter=100):
    E = ma if ecc < 0.8 else np.pi
    for _ in range(max_iter):
        dE = (ma - E + ecc * np.sin(E)) / (1.0 - ecc * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    ta = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(E / 2.0),
    )
    return ta % (2 * np.pi)


def coe2rv_py(sma, ecc, incl, raan, argp, ta):
    MU = 398600.4418
    h = np.sqrt(sma * MU * (1.0 - ecc ** 2))
    r_pf = (h ** 2 / MU) / (1.0 + ecc * np.cos(ta)) * np.array([np.cos(ta), np.sin(ta), 0.0])
    v_pf = (MU / h) * np.array([-np.sin(ta), ecc + np.cos(ta), 0.0])

    def Rz(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    def Rx(a): return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

    Q = Rz(raan) @ Rx(incl) @ Rz(argp)
    return Q @ r_pf, Q @ v_pf


def build_fixed_states(dv_init_red, dv_init_blue, perturb_ma_std=0.0):
    """
    将 FIXED_INIT 轨道根数转换为 SatState。
    perturb_ma_std > 0 时对红星 MA 加高斯扰动（rad），防止过拟合单一状态。
    """
    states = {}
    for name, oe in FIXED_INIT.items():
        oe_use = dict(oe)
        if name == "red" and perturb_ma_std > 0.0:
            oe_use["ma"] = oe["ma"] + np.random.normal(0.0, perturb_ma_std)
        ta = ma2ta(oe_use["ma"], oe_use["ecc"])
        r, v = coe2rv_py(oe_use["sma"], oe_use["ecc"], oe_use["incl"],
                         oe_use["raan"], oe_use["argp"], ta)
        s = oge_py.SatState()
        s.r_j2000 = r
        s.v_j2000 = v
        s.dv_remain = dv_init_red if name == "red" else dv_init_blue
        s.is_alive = True
        agent_key = "red_sat" if name == "red" else "blue_sat"
        states[agent_key] = s
    return states


# ── 微调专用 Wrapper ──────────────────────────────────────────────────────────
class OGEFixedInitWrapper(Wrapper):
    """
    固定初始化场景的 Wrapper。
    - reset 时使用固定初始状态（可选小扰动）
    - 奖励函数与 OGESingleEnvWrapper 完全对齐
    - 角度引导从 50km 开始（原版 25km），帮助模型更早规避太阳角恶化
    """

    def __init__(self, env, cfg, perturb_ma_std=0.02) -> None:
        super().__init__(env)
        self._oge = env.oge
        self._dv_max_blue = cfg.dv_max_per_step_blue
        self._perturb_ma_std = perturb_ma_std  # MA 扰动标准差（rad），0 表示纯固定

        dv = self._dv_max_blue
        self._blue_actions = np.array([
            [dv, 0, 0], [-dv, 0, 0], [0, dv, 0],
            [0, -dv, 0], [0, 0, dv], [0, 0, -dv], [0, 0, 0]
        ], dtype=np.float32)

        self._single_obs_size = 13
        self._last_obs = None

        self._recon_dist_threshold = 20.0
        self._recon_angle_threshold = np.deg2rad(60)
        self._recon_time_target = 120.0
        self._timestep = cfg.timestep
        self._recon_time_accumulated = 0.0
        self._init_fuel = cfg.dv_init_red
        self._dv_max_red = cfg.dv_max_per_step_red
        self._last_dist = 200.0
        self._last_in_recon_zone = False

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
    def num_envs(self):
        return 1

    def _get_refined_obs(self, obs_np):
        red_obs = obs_np[1]
        rel_pos_lvlh = red_obs[6:9]
        rel_vel_lvlh = red_obs[11:14]
        dist_norm    = red_obs[14]
        solar_angle  = red_obs[9]
        sun_dir      = red_obs[15:18]
        dv_ratio     = red_obs[19]
        time_prog    = red_obs[18]

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
        fixed_states = build_fixed_states(
            dv_init_red=self._init_fuel,
            dv_init_blue=self._dv_max_blue,
            perturb_ma_std=self._perturb_ma_std,
        )
        obs, info = self._env.reset(options={"states": fixed_states})
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._recon_time_accumulated = 0.0
        self._last_in_recon_zone = False
        red_obs = self._last_obs[1]
        self._last_dist = np.linalg.norm(red_obs[6:9])
        refined_obs = self._get_refined_obs(self._last_obs)
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
        """
        针对 Fixed Init 场景优化的奖励函数（仅用于微调）：
        核心思路：在太阳角好时（<30°）强烈鼓励快速接近，在太阳角差时（>50°）惩罚接近

        Fixed Init 失败分析：
        - Step 40-60: 距离 40-75km，太阳角 19-25°（最佳窗口）
        - 模型没有意识到这是机会，继续慢慢接近
        - Step 80+: 距离 <40km，太阳角恶化到 58-82°（失败）
        """
        target_lvlh = obs[6:9]
        solar_angle = obs[9]
        dv_remain = obs[10]
        rel_vel_lvlh = obs[11:14]

        distance = np.linalg.norm(target_lvlh)
        relative_vel_ms = np.linalg.norm(rel_vel_lvlh)
        action_ms = np.linalg.norm(action) * 1000.0
        solar_angle_deg = np.rad2deg(solar_angle)

        reward = 0.0
        done = False

        # --- 1. 终端判定 ---
        in_recon_zone = (distance <= self._recon_dist_threshold and
                         solar_angle <= self._recon_angle_threshold)
        is_out_of_fuel = (dv_remain <= 0.0)

        if in_recon_zone:
            if not self._last_in_recon_zone:
                reward = 100.0 + (dv_remain / self._init_fuel) * 30.0
                self._recon_time_accumulated = self._timestep
                self._last_in_recon_zone = True
            else:
                self._recon_time_accumulated += self._timestep
                reward = 5.0
                if self._recon_time_accumulated >= self._recon_time_target:
                    reward = 200.0 + (dv_remain / self._init_fuel) * 50.0
                    done = True
            self._last_dist = distance
            return reward, done
        else:
            self._recon_time_accumulated = 0.0
            self._last_in_recon_zone = False

        if is_out_of_fuel:
            reward = -40.0
            done = True
            return reward, done

        # --- 2. 分阶段密集奖励 ---
        dist_change = self._last_dist - distance

        if distance > 50.0:
            # 阶段一（>50km）：快速接近为主
            dist_reward = 2.0 * dist_change
            fuel_penalty = -0.005 * action_ms
            reward = dist_reward + fuel_penalty

        elif distance > 25.0:
            # 阶段二（50-25km）：太阳角敏感区（Fixed Init 关键区域）
            if solar_angle_deg < 30.0:
                # 太阳角好（<30°）：强烈鼓励快速接近
                dist_reward = 4.0 * dist_change  # 加倍距离奖励
                angle_bonus = 3.0 * (1.0 - solar_angle / self._recon_angle_threshold)
                reward = dist_reward + angle_bonus - 0.005 * action_ms
            elif solar_angle_deg > 50.0:
                # 太阳角差（>50°）：惩罚继续接近
                if dist_change > 0:
                    approach_penalty = -3.0 * dist_change
                else:
                    approach_penalty = 0.0
                reward = approach_penalty - 0.01 * action_ms
            else:
                # 太阳角中等（30-50°）：正常接近
                dist_reward = 2.0 * dist_change
                angle_guide = 1.0 * (1.0 - solar_angle / self._recon_angle_threshold)
                reward = dist_reward + angle_guide - 0.01 * action_ms

        else:
            # 阶段三（<25km）：精确进入
            dist_reward = 2.0 * dist_change
            angle_guide = 2.0 * (1.0 - solar_angle / self._recon_angle_threshold)

            # 速度漏斗
            target_vel_ms = 1.0 + (distance / 25.0) * 4.0
            vel_penalty = 0.0
            if relative_vel_ms > target_vel_ms:
                vel_penalty = -0.2 * (relative_vel_ms - target_vel_ms)

            fuel_penalty = -0.01 * action_ms
            reward = dist_reward + angle_guide + vel_penalty + fuel_penalty

        self._last_dist = distance
        return reward, done

    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="预训练 checkpoint 路径 (.pt)")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="微调步数（默认 1M）")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="微调学习率（默认 5e-5，远小于原始 3e-4）")
    parser.add_argument("--perturb", type=float, default=0.02,
                        help="红星 MA 扰动标准差 rad（默认 0.02，约 1.1°；设 0 为纯固定）")
    parser.add_argument("--name", type=str, default="ppo_recon_finetune",
                        help="实验名称")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_env = OGEEnv(env_cfg)
    env = OGEFixedInitWrapper(raw_env, env_cfg, perturb_ma_std=args.perturb)

    dv_max = 0.002 / (3 ** 0.5)

    ppo_cfg = deepcopy(base_ppo_cfg)
    ppo_cfg["learning_rate"] = args.lr
    ppo_cfg["rollouts"] = 2048
    ppo_cfg["learning_epochs"] = 4          # 减少 epoch，防止过拟合
    ppo_cfg["entropy_loss_scale"] = 0.005   # 降低熵，减少无效探索
    ppo_cfg["experiment"]["directory"] = f"runs/{args.name}"
    ppo_cfg["experiment"]["experiment_name"] = args.name
    ppo_cfg["experiment"]["wandb_kwargs"] = {
        "project": "OGE-Recon",
        "name": args.name,
        "tags": ["ppo", "finetune", "fixed-init"],
    }
    ppo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space.shape[0], "device": device}
    ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    policy = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        dv_max=dv_max,
        clip_actions=False,
    )
    value = Value(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    models = {"policy": policy, "value": value}

    memory = RandomMemory(
        memory_size=ppo_cfg["rollouts"],
        num_envs=env.num_envs,
        device=device,
    )

    agent = PPO(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    print(f"加载 checkpoint: {args.checkpoint}")
    agent.load(args.checkpoint)

    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={"timesteps": args.timesteps, "headless": False},
    )

    print(f"开始微调：{args.timesteps} steps，lr={args.lr}，MA扰动std={args.perturb} rad")
    print(f"  Red : sma=42060.338261 e=0.003001 incl=0.002287 raan=1.592759 argp=3.303797 MA=1.033423")
    print(f"  Blue: sma=42169.502913 e=0.0      incl=0.002287 raan=1.592829 argp=0.0      MA=4.345423")
    trainer.train()


if __name__ == "__main__":
    main()
