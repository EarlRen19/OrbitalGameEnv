"""Fine-tune a trained PPO agent on a fixed initial state scenario."""

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


# ── 固定初始状态 ──────────────────────────────────────────────────────────────
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


def build_fixed_states(dv_init_red, dv_init_blue):
    states = {}
    for name, oe in FIXED_INIT.items():
        ta = ma2ta(oe["ma"], oe["ecc"])
        r, v = coe2rv_py(oe["sma"], oe["ecc"], oe["incl"], oe["raan"], oe["argp"], ta)
        s = oge_py.SatState()
        s.r_j2000 = r
        s.v_j2000 = v
        s.dv_remain = dv_init_red if name == "red" else dv_init_blue
        s.is_alive = True
        agent_key = "red_sat" if name == "red" else "blue_sat"
        states[agent_key] = s
    return states


# ── 固定初始状态的 Wrapper ────────────────────────────────────────────────────
class OGEFixedInitWrapper(Wrapper):
    """与 OGESingleEnvWrapper 完全相同，但 reset 时强制使用固定初始状态。"""

    def __init__(self, env, cfg) -> None:
        super().__init__(env)
        self._oge = env.oge
        self._dv_max_blue = cfg.dv_max_per_step_blue

        dv = self._dv_max_blue
        self._blue_actions = np.array([
            [dv, 0, 0], [-dv, 0, 0], [0, dv, 0],
            [0, -dv, 0], [0, 0, dv], [0, 0, -dv], [0, 0, 0]
        ], dtype=np.float32)

        self._obs_size = self._oge.get_obs_size()
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

        self._fixed_states = build_fixed_states(cfg.dv_init_red, cfg.dv_init_blue)

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
        # 始终使用固定初始状态
        obs, info = self._env.reset(options={"states": self._fixed_states})
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._recon_time_accumulated = 0.0
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
        target_lvlh = obs[6:9]
        solar_angle = obs[9]
        dv_remain = obs[10]

        distance = np.linalg.norm(target_lvlh)
        action_ms = np.linalg.norm(action) * 1000.0

        reward = 0.0
        done = False

        in_recon_zone = (distance <= self._recon_dist_threshold and
                         solar_angle <= self._recon_angle_threshold)
        is_out_of_fuel = (dv_remain <= 0.0)

        if in_recon_zone:
            reward = 200.0 + (dv_remain / self._init_fuel) * 50.0
            self._recon_time_accumulated += self._timestep
            if self._recon_time_accumulated >= self._recon_time_target:
                reward += 100.0
            done = True
            return reward, done
        else:
            self._recon_time_accumulated = 0.0

        if is_out_of_fuel:
            reward = -20.0
            done = True
            return reward, done

        dist_change = self._last_dist - distance
        dist_reward = 2.0 * dist_change

        angle_guide = 0.0
        if distance <= 25.0:
            angle_guide = 2.0 * (1.0 - solar_angle / self._recon_angle_threshold)

        fuel_penalty = -0.01 * action_ms

        reward = dist_reward + angle_guide + fuel_penalty
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
                        help="Path to pretrained checkpoint (.pt)")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Fine-tuning timesteps (default: 1M)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for fine-tuning (default: 5e-5, smaller than original 3e-4)")
    parser.add_argument("--name", type=str, default="ppo_recon_finetune",
                        help="Experiment name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_env = OGEEnv(env_cfg)
    env = OGEFixedInitWrapper(raw_env, env_cfg)

    dv_max = 0.002 / (3 ** 0.5)

    ppo_cfg = deepcopy(base_ppo_cfg)
    ppo_cfg["learning_rate"] = args.lr          # 小学习率，避免破坏已有策略
    ppo_cfg["rollouts"] = 2048
    ppo_cfg["learning_epochs"] = 4              # 减少 epoch，防止过拟合固定场景
    ppo_cfg["entropy_loss_scale"] = 0.005       # 降低熵系数，减少探索，专注利用
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

    print(f"Loading checkpoint: {args.checkpoint}")
    agent.load(args.checkpoint)

    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={"timesteps": args.timesteps, "headless": True},
    )

    print(f"Fine-tuning on fixed init scenario for {args.timesteps} steps (lr={args.lr})")
    print(f"  Red : sma=42060.338261 e=0.003001 incl=0.002287 raan=1.592759 argp=3.303797 MA=1.033423")
    print(f"  Blue: sma=42169.502913 e=0.0      incl=0.002287 raan=1.592829 argp=0.0      MA=4.345423")
    trainer.train()


if __name__ == "__main__":
    main()
