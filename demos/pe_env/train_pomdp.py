import sys
import os
import math

# 将项目根目录添加到Python路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import signal
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import time
import random

from gymnasium import spaces
from env.mpe_pomdp_env import MPE_POMDP_Env, MPE_POMDP_EnvCfg
from env.encoder import AttentionBasedEncoder

class TrainConfig:
    """训练超参数配置"""
    use_encoder: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    anneal_ent: bool = True
    ent_anneal_start_frac: float = 0.3
    final_ent_coef: float = 0.0001
    distil_coef: float = 1.0  # 蒸馏损失的权重
    lr: float = 3e-4
    num_mini_batches: int = 4
    update_epochs: int = 5
    total_timesteps: int = 5_000_000
    num_steps: int = 2048
    num_envs: int = 1
    initial_episode_length: int = 3600 * 10
    curriculum_check_episodes: int = 50
    success_rate_threshold: float = 0.7
    initial_dist_cap: float = 60e3
    initial_p_init_dv: float = 200.0
    dist_cap_decrement: float = 1e3
    p_init_dv_decrement: float = 100.0
    min_dist_cap: float = 30e3
    min_p_init_dv: float = 200.0
    debug_critic: bool = False
    debug_observation: bool = False
    resume_from_checkpoint: str = None
    checkpoint_interval: int = 50
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_name: str = f"mpe_transformer_distil_{int(time.time())}"

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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ActorCritic(nn.Module):
    """非对称Actor-Critic，学生网络使用Transformer处理历史轨迹"""
    def __init__(self, student_obs_dim, privileged_obs_dim, act_dim, env_cfg: MPE_POMDP_EnvCfg, train_cfg: TrainConfig):
        super().__init__()
        self.use_encoder = train_cfg.use_encoder
        
        # Transformer配置
        d_model = 128  # Transformer的特征维度
        history_input_dim = 6 # 历史轨迹中每个时间步的维度 (rel_pos, rel_vel)
        
        # Transformer Encoder
        self.history_embedding = layer_init(nn.Linear(history_input_dim, d_model))
        self.pos_encoder = PositionalEncoding(d_model, max_len=env_cfg.history_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 学生网络的主体
        if self.use_encoder:
            self_dim = 7
            other_dim = 7 * (env_cfg.num_p - 1)
            # 注意：这里的`lstm_pred_dim`现在是`transformer_feature_dim`
            self.student_encoder = AttentionBasedEncoder(
                self_dim=self_dim,
                other_dim=other_dim,
                ob_dim=0, # ob_dim is part of self_dim and other_dim now
                lstm_pred_dim=d_model * env_cfg.num_e,
                embed_dim=128,
                nhead=4
            )
            student_feature_dim = 256 # Output of AttentionBasedEncoder
        else:
            # 简化的学生网络
            simple_student_obs_dim = 7 + 7 * (env_cfg.num_p - 1)
            self.student_encoder = nn.Sequential(
                layer_init(nn.Linear(simple_student_obs_dim + d_model, 256)),
                nn.Tanh()
            )
            student_feature_dim = 256

        # 教师网络
        teacher_feature_dim = 256
        self.teacher_encoder = nn.Sequential(
            layer_init(nn.Linear(privileged_obs_dim, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, teacher_feature_dim)),
            nn.LayerNorm(teacher_feature_dim)
        )

        # Actor-Critic的头部
        self.actor_head = layer_init(nn.Linear(student_feature_dim, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)
        self.critic_head = layer_init(nn.Linear(teacher_feature_dim, 1), std=1.0)

        # 动作空间的缩放和偏置
        action_space = spaces.Box(-env_cfg.p_dv_step, env_cfg.p_dv_step, shape=(3,))
        self.register_buffer("action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))

    def get_value(self, privileged_obs):
        teacher_features = self.teacher_encoder(privileged_obs)
        return self.critic_head(teacher_features)

    def get_student_features(self, obs, history, history_mask=None):
        # history shape: [batch, seq_len, features]
        # history_mask shape: [batch, seq_len]
        
        # 1. 通过Transformer处理历史轨迹
        embedded_history = self.history_embedding(history)
        pos_encoded_history = self.pos_encoder(embedded_history.permute(1, 0, 2)).permute(1, 0, 2)
        
        # 创建Transformer需要的padding mask
        # Transformer mask: True表示被mask掉（忽略）
        if history_mask is not None:
            src_key_padding_mask = (history_mask == 0)
        else:
            src_key_padding_mask = None

        transformer_output = self.transformer_encoder(pos_encoded_history, src_key_padding_mask=src_key_padding_mask)
        
        # 从Transformer输出中提取固定大小的特征向量
        # 方法：对所有未被mask的时间步的输出取平均
        if src_key_padding_mask is not None:
            # 扩展mask以便于对特征进行mask
            mask_expanded = ~src_key_padding_mask.unsqueeze(-1).expand_as(transformer_output)
            # 对未被mask的部分求和
            sum_features = (transformer_output * mask_expanded).sum(dim=1)
            # 计算未被mask的元素数量
            num_unmasked = mask_expanded.sum(dim=1)
            # 计算平均值，避免除以零
            history_features = sum_features / torch.clamp(num_unmasked, min=1e-9)
        else:
            history_features = transformer_output.mean(dim=1)

        # 2. 将历史特征与当前观测融合
        # 注意：这里的融合方式取决于student_encoder的设计
        # 对于AttentionBasedEncoder，它期望一个扁平化的输入
        # 假设只有一个逃逸者，所以直接用history_features
        # 如果有多个逃逸者，需要将它们的特征拼接起来
        encoder_input_tensor = torch.cat([obs, history_features], dim=-1)
        student_features = self.student_encoder(encoder_input_tensor)
            
        return student_features

    def get_action_and_value(self, obs, privileged_obs, history, history_mask=None, action=None, deterministic=False):
        student_features = self.get_student_features(obs, history, history_mask)
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
        
        return final_action, log_prob, entropy, value, student_features

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
        # 修改：存储历史轨迹和掩码
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
            # 假设只有一个逃逸者
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

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    run_dir = base_dir / "runs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))

    params_path = run_dir / "all_params.txt"
    progress_path = run_dir / "curriculum_progress_log.txt"
    with open(params_path, "w", encoding="utf-8") as f:
        f.write("--- All Run Parameters ---\n")
        for key, value in sorted(all_params.items()):
            f.write(f"{key}: {value}\n")
    print(f"All run parameters saved to {params_path}")

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

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer.add_text("hyperparameters", f"<pre>{vars(cfg)}</pre>")

    env = MPE_POMDP_Env(env_cfg)
    
    curriculum_max_episode_length = cfg.initial_episode_length
    current_dist_cap = cfg.initial_dist_cap
    current_p_init_dv = cfg.initial_p_init_dv
    env.set_difficulty_parameters(episode_length=curriculum_max_episode_length, dist_cap=current_dist_cap, p_init_dv=current_p_init_dv)
    print(f"POMDP Mode: {env_cfg.use_partial_obs}, Obs Interval: {env_cfg.obs_interval}, History Length: {env_cfg.history_len}")
    print(f"任务时长固定: {curriculum_max_episode_length}s, 初始捕获距离: {current_dist_cap}m, 初始燃料: {current_p_init_dv}m/s")

    pursuer_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
    evader_ids = [f'e_{i}' for i in range(env_cfg.num_e)]
    
    student_obs_dim = env.observation_spaces[pursuer_ids[0]].shape[0]
    privileged_obs_dim = (env_cfg.num_p + env_cfg.num_e) * 6
    act_dim = env.action_spaces[pursuer_ids[0]].shape[0]

    agent = ActorCritic(student_obs_dim, privileged_obs_dim, act_dim, env_cfg, cfg).to(cfg.device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    buffer = CentralizedRolloutBuffer(cfg.num_steps, env_cfg.num_p, student_obs_dim, privileged_obs_dim, act_dim, cfg.device, env_cfg)

    start_update = 1
    global_step = 0
    if cfg.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {cfg.resume_from_checkpoint}")
        checkpoint = torch.load(cfg.resume_from_checkpoint, map_location=cfg.device)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_update = checkpoint['update'] + 1
        global_step = checkpoint['global_step']
        # curriculum resume logic...

    num_updates = cfg.total_timesteps // (cfg.num_steps * cfg.num_envs)
    anneal_lr_start_update = 500
    final_lr_fraction = 0.1
    
    recent_episode_stats = deque(maxlen=cfg.curriculum_check_episodes) 
    current_episode_return = 0.0
    ep_len_counter = 0
    
    obs, infos = env.reset() 
    last_sma_perturb_km = env.current_sma_perturb_km
    
    for update in range(start_update, num_updates + 1):
        # --- Curriculum & Annealing Logic ---
        start = env_cfg.sma_perturb_start_update
        end = env_cfg.sma_perturb_end_update
        max_perturb = env_cfg.sma_perturb_km_max
        if update < start: new_perturb = 0.0
        elif update >= end: new_perturb = max_perturb
        else: new_perturb = ((update - start) / (end - start)) * max_perturb
        env.set_sma_perturb(new_perturb)

        if update < anneal_lr_start_update: current_lr = cfg.lr
        else:
            progress = min(1.0, max(0.0, (update - anneal_lr_start_update) / (num_updates - anneal_lr_start_update)))
            lr_decay_factor = 1.0 - (1.0 - final_lr_fraction) * progress
            current_lr = cfg.lr * lr_decay_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        if cfg.anneal_ent:
            anneal_start_update_ent = cfg.ent_anneal_start_frac * num_updates
            if update < anneal_start_update_ent: current_ent_coef = cfg.ent_coef
            else:
                progress = min(1.0, (update - anneal_start_update_ent) / (num_updates - anneal_start_update_ent))
                current_ent_coef = cfg.ent_coef - progress * (cfg.ent_coef - cfg.final_ent_coef)
        else:
            current_ent_coef = cfg.ent_coef

        # --- Logging ---
        writer.add_scalar("charts/learning_rate", current_lr, global_step)
        writer.add_scalar("info/ent_coef", current_ent_coef, global_step)
        writer.add_scalar("curriculum/capture_distance_cap", current_dist_cap, global_step)
        writer.add_scalar("curriculum/pursuer_initial_fuel", current_p_init_dv, global_step)
        writer.add_scalar("curriculum/sma_perturbation_km", env.current_sma_perturb_km, global_step)

        if env.current_sma_perturb_km > last_sma_perturb_km:
            print(f"*** Curriculum: SMA Perturbation -> {env.current_sma_perturb_km:.2f} km ***")
            file_exists = os.path.exists(progress_path)
            with open(progress_path, "a", encoding="utf-8") as f:
                if not file_exists:
                    f.write("--- Curriculum Progress Log ---\nAbbreviations:\n  U: Update\n  GS: Global Step\n  Trig: Trigger Type (SR=Success Rate, SMA=SMA Perturbation)\n  SR_trig: Trigger Success Rate\n  DistCap: New Capture Distance Cap (m)\n  Fuel: New Pursuer Initial Fuel (m/s)\n  SMA_km: New SMA Perturbation (km)\n\n")
                f.write("--------------------------------------------------\n")
                f.write(f"U: {update} | GS: {global_step}\n")
                f.write(f"Trig: SMA\n")
                f.write(f"SMA_km: {env.current_sma_perturb_km:.2f}\n")
            last_sma_perturb_km = env.current_sma_perturb_km

        # --- Rollout Collection ---
        start_time = time.time()
        avg_reward_components = {}
        
        agent.eval()
        for step in range(cfg.num_steps):
            global_step += 1
            ep_len_counter += 1
            
            pursuer_obs_list = [torch.Tensor(obs[name]).to(cfg.device) for name in pursuer_ids]
            pursuer_obs_tensor = torch.stack(pursuer_obs_list)
            
            pursuer_priv_obs_list = [torch.Tensor(infos[name]['privileged_state']).to(cfg.device) for name in pursuer_ids]
            pursuer_priv_obs_tensor = torch.stack(pursuer_priv_obs_list)

            # 从infos中获取历史和掩码
            history_list = [torch.from_numpy(infos[name][f'history_input_{evader_ids[0]}']).float().to(cfg.device) for name in pursuer_ids]
            history_tensor = torch.stack(history_list)
            mask_list = [torch.from_numpy(infos[name][f'history_mask_{evader_ids[0]}']).float().to(cfg.device) for name in pursuer_ids]
            mask_tensor = torch.stack(mask_list)

            # --- [新增] 调试观测值 ---
            if cfg.debug_observation and global_step > 0 and global_step % 200 == 0:
                p0_obs = pursuer_obs_tensor[0]
                p0_history = history_tensor[0]
                p0_mask = mask_tensor[0]
                priv_obs = pursuer_priv_obs_tensor[0] # 特权观测对所有智能体都是一样的

                print("\n" + "="*40 + f" DEBUG OBSERVATION @ G_Step:{global_step} " + "="*40)
                
                print(f"--- Input to Student Actor (p_0) ---")
                print(f"  - Regular Obs (self+teammates): shape={p0_obs.shape}\n{p0_obs.cpu().numpy()}")
                print(f"  - History Obs (for Transformer): shape={p0_history.shape}\n{p0_history.cpu().numpy()}")
                print(f"  - History Mask: shape={p0_mask.shape}\n{p0_mask.cpu().numpy()}")
                
                print(f"\n--- Input to Critic (Privileged) ---")
                print(f"  - Privileged State (Anchor+Relative): shape={priv_obs.shape}\n{priv_obs.cpu().numpy()}")
                print("="*105 + "\n")
            # --- 调试结束 ---

            with torch.no_grad():
                actions_tensor, log_prob, _, values, _ = agent.get_action_and_value(pursuer_obs_tensor, pursuer_priv_obs_tensor, history_tensor, history_mask=mask_tensor)
                values = values.flatten()

            evader_actions = env.get_evader_actions()
            actions_to_step = {name: actions_tensor[i].cpu().numpy() for i, name in enumerate(pursuer_ids)}
            actions_to_step.update(evader_actions)

            next_obs, rewards, terminations, truncations, infos = env.step(actions_to_step)
            
            avg_pursuer_reward = np.mean([rewards.get(pid, 0) for pid in pursuer_ids])
            current_episode_return += avg_pursuer_reward

            if env_cfg.debug_rewards:
                for name in pursuer_ids:
                    if name in infos and 'reward_components' in infos[name]:
                        for key, value in infos[name]['reward_components'].items():
                            avg_reward_components[key] = avg_reward_components.get(key, 0.0) + value

            pursuer_rewards_tensor = torch.tensor([rewards[name] for name in pursuer_ids]).to(cfg.device)
            pursuer_dones = torch.tensor([terminations.get(name, False) or truncations.get(name, False) for name in pursuer_ids]).to(cfg.device)
            
            buffer.add(pursuer_obs_tensor, pursuer_priv_obs_tensor, actions_tensor, log_prob, pursuer_rewards_tensor, pursuer_dones, values, infos, pursuer_ids, evader_ids)
            
            obs = next_obs
            if any(terminations.values()) or any(truncations.values()):
                final_info = next(iter(infos.values()), None)
                if final_info:
                    stats = final_info.get('episode_statistics', {})
                    if stats: recent_episode_stats.append(stats)
                    
                    print(f"Update {update}, Step {global_step}: Ep Ret: {current_episode_return:.2f}, Len: {ep_len_counter}, Reason: {final_info.get('termination_reason', 'Unknown')}")
                    writer.add_scalar("charts/episodic_return", current_episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_len_counter, global_step)

                obs, infos = env.reset() 
                current_episode_return = 0.0
                ep_len_counter = 0

        with torch.no_grad():
            next_priv_obs = torch.Tensor(infos[pursuer_ids[0]]['privileged_state']).unsqueeze(0).to(cfg.device)
            next_value = agent.get_value(next_priv_obs.repeat(env_cfg.num_p, 1)).mean()
            next_done = torch.tensor([terminations.get(name, False) for name in pursuer_ids]).to(cfg.device)
            buffer.compute_returns(next_value, next_done, cfg.gamma, cfg.gae_lambda)

        # --- PPO & Distillation Update ---
        agent.train()
        avg_v_loss, avg_pg_loss, avg_entropy_loss, avg_distil_loss = 0, 0, 0, 0
        num_minibatches_processed = 0

        for epoch in range(cfg.update_epochs):
            for b_obs, b_priv_obs, b_actions, b_logprobs, b_advantages, b_returns, b_hist, b_hist_mask in buffer.get(cfg.num_steps * env_cfg.num_p, cfg.num_steps * env_cfg.num_p // cfg.num_mini_batches):
                _, new_logprob, entropy, new_value, student_features = agent.get_action_and_value(b_obs, b_priv_obs, b_hist, b_hist_mask, b_actions)
                new_value = new_value.view(-1)
                
                logratio = new_logprob - b_logprobs
                ratio = logratio.exp()
                adv_norm = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
                pg_loss1 = -adv_norm * ratio
                pg_loss2 = -adv_norm * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss = 0.5 * ((new_value - b_returns) ** 2).mean()
                entropy_loss = entropy.mean()
                
                with torch.no_grad():
                    teacher_features = agent.teacher_encoder(b_priv_obs)
                distillation_loss = F.mse_loss(student_features, teacher_features)
                
                loss = pg_loss - current_ent_coef * entropy_loss + v_loss * cfg.vf_coef + cfg.distil_coef * distillation_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

                avg_v_loss += v_loss.item()
                avg_pg_loss += pg_loss.item()
                avg_entropy_loss += entropy_loss.item()
                avg_distil_loss += distillation_loss.item()
                num_minibatches_processed += 1
        
        writer.add_scalar("losses/value_loss", avg_v_loss / num_minibatches_processed, global_step)
        writer.add_scalar("losses/policy_loss", avg_pg_loss / num_minibatches_processed, global_step)
        writer.add_scalar("losses/entropy_loss", avg_entropy_loss / num_minibatches_processed, global_step)
        writer.add_scalar("losses/distillation_loss", avg_distil_loss / num_minibatches_processed, global_step)
        
        if env_cfg.debug_rewards:
            total_reward_steps = cfg.num_steps * cfg.num_p
            for key, value in avg_reward_components.items():
                writer.add_scalar(f"rewards/{key}", value / total_reward_steps, global_step)

        if len(recent_episode_stats) >= 2:
            if update > 10 and len(recent_episode_stats) > 10:
                total_eps = recent_episode_stats[-1]['total_episodes'] - recent_episode_stats[0]['total_episodes']
                if total_eps > 0:
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
                            print(f"*** Curriculum: SR -> DistCap={current_dist_cap}m, Fuel={current_p_init_dv}m/s ***")
                            
                            file_exists = os.path.exists(progress_path)
                            with open(progress_path, "a", encoding="utf-8") as f:
                                if not file_exists:
                                    f.write("--- Curriculum Progress Log ---\nAbbreviations:\n  U: Update\n  GS: Global Step\n  Trig: Trigger Type (SR=Success Rate, SMA=SMA Perturbation)\n  SR_trig: Trigger Success Rate\n  DistCap: New Capture Distance Cap (m)\n  Fuel: New Pursuer Initial Fuel (m/s)\n  SMA_km: New SMA Perturbation (km)\n\n")
                                f.write("--------------------------------------------------\n")
                                f.write(f"U: {update} | GS: {global_step}\n")
                                f.write(f"Trig: SR\n")
                                f.write(f"SR_trig: {current_success_rate:.2f}\n")
                                f.write(f"DistCap: {current_dist_cap}\n")
                                f.write(f"Fuel: {current_p_init_dv}\n")
                            recent_episode_stats.clear()

    env.close()
    writer.close()
    print("训练完成!")

if __name__ == "__main__":
    # 在这里传递更新后的env_cfg
    train(TrainConfig(), MPE_POMDP_EnvCfg(), {})