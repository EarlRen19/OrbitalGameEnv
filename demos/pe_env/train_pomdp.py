import sys
import os
import math
import signal
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import time
import random
from gymnasium import spaces

# 将项目根目录添加到Python路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from env.mpe_pomdp_env import MPE_POMDP_Env, MPE_POMDP_EnvCfg
from env.hrg_models import HRG_Student_Encoder, Aligned_Teacher

class TrainConfig:
    """训练超参数配置"""
    # 模型与架构
    distil_coef: float = 1.0
    hls_weight_decay: float = 1e-4

    # PPO 核心参数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    
    # 学习率与优化器
    lr: float = 3e-4
    anneal_ent: bool = True
    ent_anneal_start_frac: float = 0.3
    final_ent_coef: float = 0.0001
    
    # 训练流程
    total_timesteps: int = 5_000_000
    num_steps: int = 2048
    num_envs: int = 1
    num_mini_batches: int = 4
    update_epochs: int = 5
    
    # 课程学习
    initial_episode_length: int = 3600 * 10
    curriculum_check_episodes: int = 50
    success_rate_threshold: float = 0.7
    initial_dist_cap: float = 60e3
    initial_p_init_dv: float = 500.0
    dist_cap_decrement: float = 1e3
    p_init_dv_decrement: float = 100.0
    min_dist_cap: float = 30e3
    min_p_init_dv: float = 300.0

    # 调试与杂项
    debug_critic: bool = False 
    debug_observation: bool = False
    resume_from_checkpoint: str = None
    checkpoint_interval: int = 50
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_name: str = f"hrg_maddpg_{int(time.time())}"

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class HRG_ActorCritic(nn.Module):
    def __init__(self, env_cfg: MPE_POMDP_EnvCfg, act_dim: int, priv_obs_dim: int):
        super().__init__()
        
        d_model = 128
        history_input_dim = 6
        
        # --- 新增: 输入归一化层 ---
        self.history_norm = nn.LayerNorm(history_input_dim)
        
        self.history_embedding = layer_init(nn.Linear(history_input_dim, d_model))
        self.pos_encoder = PositionalEncoding(d_model, max_len=env_cfg.history_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.student_enc = HRG_Student_Encoder(env_cfg, hidden_dim=d_model)
        
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(self.student_enc.output_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)
        
        self.teacher_enc = Aligned_Teacher(
            priv_obs_dim, 
            student_out_dim=self.student_enc.output_dim
        )
        
        self.critic = nn.Sequential(
            nn.LayerNorm(priv_obs_dim), # 在输入端对特权信息进行归一化
            layer_init(nn.Linear(priv_obs_dim, 512)), nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512, 256)), nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

        action_space = spaces.Box(-env_cfg.p_dv_step, env_cfg.p_dv_step, shape=(3,))
        self.register_buffer("action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))

    def get_history_feats(self, history, history_mask=None):
        normed_history = self.history_norm(history)
        embedded_history = self.history_embedding(normed_history)
        pos_encoded_history = self.pos_encoder(embedded_history.permute(1, 0, 2)).permute(1, 0, 2)
        src_key_padding_mask = (history_mask == 0) if history_mask is not None else None
        transformer_output = self.transformer_encoder(pos_encoded_history, src_key_padding_mask=src_key_padding_mask)
        
        if src_key_padding_mask is not None:
            mask_expanded = ~src_key_padding_mask.unsqueeze(-1).expand_as(transformer_output)
            sum_features = (transformer_output * mask_expanded).sum(dim=1)
            num_unmasked = mask_expanded.sum(dim=1)
            history_features = sum_features / torch.clamp(num_unmasked, min=1e-9)
        else:
            history_features = transformer_output.mean(dim=1)
        return history_features

    def get_value(self, privileged_obs):
        return self.critic(privileged_obs)

    def get_action_and_value(self, obs, privileged_obs, history, history_mask=None, action=None, deterministic=False):
        hist_feats = self.get_history_feats(history, history_mask)
        student_features, attn_weights = self.student_enc(obs, hist_feats)
        action_mean = self.actor_head(student_features)
        
        clipped_logstd = torch.clamp(self.actor_logstd, -2, 1)
        action_std = torch.exp(clipped_logstd).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            pre_tanh_action = probs.rsample() if not deterministic else action_mean
            tanh_action = torch.tanh(pre_tanh_action)
            final_action = self.action_bias + self.action_scale * tanh_action
        else:
            unscaled_action = (action - self.action_bias) / self.action_scale
            pre_tanh_action = torch.atanh(torch.clamp(unscaled_action, -1.0 + 1e-6, 1.0 - 1e-6))
            final_action = action
            tanh_action = torch.tanh(pre_tanh_action)

        log_prob_pre_tanh = probs.log_prob(pre_tanh_action).sum(1)
        log_prob_correction = torch.log(self.action_scale * (1.0 - tanh_action.pow(2)) + 1e-6).sum(1)
        log_prob = log_prob_pre_tanh - log_prob_correction
        entropy = probs.entropy().sum(1)
        value = self.get_value(privileged_obs)
        
        return final_action, log_prob, entropy, value, student_features, attn_weights

class CentralizedRolloutBuffer:
    def __init__(self, num_steps, num_agents, student_obs_dim, privileged_obs_dim, act_dim, device, history_cfg):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.device = device
        self.obs = torch.zeros((num_steps, num_agents, student_obs_dim)).to(device)
        self.privileged_obs = torch.zeros((num_steps, num_agents, privileged_obs_dim)).to(device)
        self.actions = torch.zeros((num_steps, num_agents, act_dim)).to(device)
        self.logprobs = torch.zeros((num_steps, num_agents)).to(device)
        self.rewards = torch.zeros((num_steps, num_agents)).to(device)
        self.dones = torch.zeros((num_steps, num_agents)).to(device)
        self.values = torch.zeros((num_steps, num_agents)).to(device)
        self.history = torch.zeros((num_steps, num_agents, history_cfg.history_len, 6)).to(device)
        self.history_masks = torch.zeros((num_steps, num_agents, history_cfg.history_len)).to(device)
        self.step = 0

    def add(self, obs, privileged_obs, actions, logprobs, rewards, dones, values, infos, pursuer_ids, evader_ids):
        self.obs[self.step] = obs
        self.privileged_obs[self.step] = privileged_obs
        self.actions[self.step] = actions
        self.logprobs[self.step] = logprobs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        for i, agent_id in enumerate(pursuer_ids):
            evader_id = evader_ids[0]
            if agent_id in infos and f'history_input_{evader_id}' in infos[agent_id]:
                self.history[self.step, i] = torch.from_numpy(infos[agent_id][f'history_input_{evader_id}']).to(self.device)
                self.history_masks[self.step, i] = torch.from_numpy(infos[agent_id][f'history_mask_{evader_id}']).to(self.device)
        self.step = (self.step + 1) % self.num_steps

    def compute_returns(self, next_value, next_done, gamma, gae_lambda):
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done.float()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1].float()
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        self.returns = self.advantages + self.values

    def get(self, batch_size, mini_batch_size):
        num_samples = batch_size
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start in range(0, num_samples, mini_batch_size):
            end = start + mini_batch_size
            batch_indices = indices[start:end]
            step_indices = batch_indices // self.num_agents
            agent_indices = batch_indices % self.num_agents
            yield (
                self.obs[step_indices, agent_indices],
                self.privileged_obs[step_indices, agent_indices],
                self.actions[step_indices, agent_indices],
                self.logprobs[step_indices, agent_indices],
                self.advantages[step_indices, agent_indices],
                self.returns[step_indices, agent_indices],
                self.history[step_indices, agent_indices],
                self.history_masks[step_indices, agent_indices],
            )

def train(cfg: TrainConfig, env_cfg: MPE_POMDP_EnvCfg, all_params: dict):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    run_dir = Path("runs") / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 将所有参数写入文件
    params_path = run_dir / "all_params.txt"
    with open(params_path, "w") as f:
        f.write("--- TrainConfig ---\n")
        for key, value in vars(cfg).items():
            f.write(f"{key}: {value}\n")
        f.write("\n--- MPE_POMDP_EnvCfg ---\n")
        for key, value in vars(env_cfg).items():
            f.write(f"{key}: {value}\n")
        f.write("\n--- All CLI Params ---\n")
        for key, value in sorted(all_params.items()):
            f.write(f"{key}: {value}\n")

    writer = SummaryWriter(str(run_dir))
    progress_path = run_dir / "curriculum_progress_log.txt"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    shutdown_requested = False
    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            print("\nCtrl+C received! Finishing current update and saving checkpoint...")
            shutdown_requested = True
        else:
            print("\nSecond Ctrl+C received! Forcing exit.")
            sys.exit(1)
    signal.signal(signal.SIGINT, signal_handler)

    writer.add_text("hyperparameters", f"<pre>{vars(cfg)}</pre>")

    env = MPE_POMDP_Env(env_cfg)
    
    current_dist_cap = cfg.initial_dist_cap
    current_p_init_dv = cfg.initial_p_init_dv
    env.set_difficulty_parameters(dist_cap=current_dist_cap, p_init_dv=current_p_init_dv)
    print(f"Initial Difficulty: Dist_Cap={current_dist_cap}m, Fuel={current_p_init_dv}m/s")

    pursuer_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
    evader_ids = [f'e_{i}' for i in range(env_cfg.num_e)]
    
    student_obs_dim = env.observation_spaces[pursuer_ids[0]].shape[0]
    priv_obs_dim = (env_cfg.num_p + env_cfg.num_e) * 6
    act_dim = env.action_spaces[pursuer_ids[0]].shape[0]

    agent = HRG_ActorCritic(env_cfg, act_dim, priv_obs_dim).to(cfg.device)
    
    hls_params, other_params = [], []
    for name, param in agent.named_parameters():
        if not param.requires_grad: continue
        if "hls_" in name:
            print(f"Applying L2 Regularization to HLS param: {name}")
            hls_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': hls_params, 'weight_decay': cfg.hls_weight_decay},
        {'params': other_params, 'weight_decay': 0.0}
    ], lr=cfg.lr, eps=1e-5)

    buffer = CentralizedRolloutBuffer(cfg.num_steps, env_cfg.num_p, student_obs_dim, priv_obs_dim, act_dim, cfg.device, env_cfg)

    start_update = 1
    global_step = 0
    # (Checkpoint loading logic would go here)

    num_updates = cfg.total_timesteps // (cfg.num_steps * cfg.num_envs)
    recent_episode_stats = deque(maxlen=cfg.curriculum_check_episodes)
    current_episode_return = 0.0
    ep_len_counter = 0
    
    obs, infos = env.reset()
    
    for update in range(start_update, num_updates + 1):
        if cfg.anneal_ent:
            anneal_start_update = cfg.ent_anneal_start_frac * num_updates
            progress = max(0.0, (update - anneal_start_update) / (num_updates - anneal_start_update))
            current_ent_coef = cfg.ent_coef - progress * (cfg.ent_coef - cfg.final_ent_coef)
        else:
            current_ent_coef = cfg.ent_coef

        agent.eval()
        for step in range(cfg.num_steps):
            global_step += 1
            ep_len_counter += 1
            
            pursuer_obs_tensor = torch.stack([torch.Tensor(obs[name]).to(cfg.device) for name in pursuer_ids])
            pursuer_priv_obs_tensor = torch.stack([torch.Tensor(infos[name]['privileged_state']).to(cfg.device) for name in pursuer_ids])
            history_tensor = torch.stack([torch.from_numpy(infos[name][f'history_input_{evader_ids[0]}']).float().to(cfg.device) for name in pursuer_ids])
            mask_tensor = torch.stack([torch.from_numpy(infos[name][f'history_mask_{evader_ids[0]}']).float().to(cfg.device) for name in pursuer_ids])

            with torch.no_grad():
                actions_tensor, log_prob, _, values, _, _ = agent.get_action_and_value(pursuer_obs_tensor, pursuer_priv_obs_tensor, history_tensor, mask_tensor)
                values = values.flatten()

            evader_actions = env.get_evader_actions()
            actions_to_step = {name: actions_tensor[i].cpu().numpy() for i, name in enumerate(pursuer_ids)}
            actions_to_step.update(evader_actions)

            next_obs, rewards, terminations, truncations, infos = env.step(actions_to_step)
            
            avg_pursuer_reward = np.mean([rewards.get(pid, 0) for pid in pursuer_ids])
            current_episode_return += avg_pursuer_reward

            pursuer_rewards_tensor = torch.tensor([rewards[name] for name in pursuer_ids]).to(cfg.device)
            pursuer_dones = torch.tensor([terminations.get(name, False) or truncations.get(name, False) for name in pursuer_ids]).to(cfg.device)
            
            buffer.add(pursuer_obs_tensor, pursuer_priv_obs_tensor, actions_tensor, log_prob, pursuer_rewards_tensor, pursuer_dones, values, infos, pursuer_ids, evader_ids)
            obs = next_obs
            
            if any(terminations.values()) or any(truncations.values()):
                final_info = next(iter(infos.values()), None)
                if final_info:
                    stats = final_info.get('episode_statistics', {})
                    if stats: recent_episode_stats.append(stats)
                    
                    print(f"Update {update}, G_Step {global_step}: Ep Ret: {current_episode_return:.2f}, Len: {ep_len_counter}, Reason: {final_info.get('termination_reason', 'Unknown')}")
                    writer.add_scalar("charts/episodic_return", current_episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_len_counter, global_step)

                obs, infos = env.reset()
                current_episode_return = 0.0
                ep_len_counter = 0

        with torch.no_grad():
            next_priv_obs = torch.stack([torch.Tensor(infos[name]['privileged_state']).to(cfg.device) for name in pursuer_ids])
            next_value = agent.get_value(next_priv_obs).mean()
            next_done = torch.tensor([False for _ in pursuer_ids]).to(cfg.device)
            buffer.compute_returns(next_value, next_done, cfg.gamma, cfg.gae_lambda)

        agent.train()
        for epoch in range(cfg.update_epochs):
            for b_obs, b_priv_obs, b_actions, b_logprobs, b_advantages, b_returns, b_hist, b_hist_mask in buffer.get(cfg.num_steps * env_cfg.num_p, cfg.num_steps * env_cfg.num_p // cfg.num_mini_batches):
                
                _, new_logprob, entropy, new_value, student_features, _ = agent.get_action_and_value(b_obs, b_priv_obs, b_hist, b_hist_mask, b_actions)
                new_value = new_value.view(-1)
                
                logratio = new_logprob - b_logprobs
                ratio = logratio.exp()
                adv_norm = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
                pg_loss = torch.max(-adv_norm * ratio, -adv_norm * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                v_loss = 0.5 * ((new_value - b_returns) ** 2).mean()
                entropy_loss = entropy.mean()
                
                with torch.no_grad():
                    teacher_targets = agent.teacher_enc(b_priv_obs)
                distil_loss = F.mse_loss(student_features, teacher_targets)
                
                loss = pg_loss + cfg.vf_coef * v_loss - current_ent_coef * entropy_loss + cfg.distil_coef * distil_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        if len(recent_episode_stats) >= 10:
            if recent_episode_stats[-1]['total_episodes'] > recent_episode_stats[0]['total_episodes']:
                total_eps = recent_episode_stats[-1]['total_episodes'] - recent_episode_stats[0]['total_episodes']
                successes = recent_episode_stats[-1]['success_count'] - recent_episode_stats[0]['success_count']
                current_success_rate = successes / total_eps
                writer.add_scalar("charts/success_rate", current_success_rate, global_step)

                if current_success_rate >= cfg.success_rate_threshold:
                    changed = False
                    if current_dist_cap > cfg.min_dist_cap:
                        current_dist_cap -= cfg.dist_cap_decrement
                        changed = True
                    if current_p_init_dv > cfg.min_p_init_dv:
                        current_p_init_dv -= cfg.p_init_dv_decrement
                        changed = True
                    if changed:
                        env.set_difficulty_parameters(dist_cap=current_dist_cap, p_init_dv=current_p_init_dv)
                        print(f"*** CURRICULUM UPDATE: SR={current_success_rate:.2f} -> New DistCap={current_dist_cap}m, New Fuel={current_p_init_dv}m/s ***")
                        recent_episode_stats.clear()

    env.close()
    writer.close()

if __name__ == "__main__":
    train(TrainConfig(), MPE_POMDP_EnvCfg(), {})
