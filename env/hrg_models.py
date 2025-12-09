import torch
import torch.nn as nn
import torch.nn.functional as F

def symlog(x):
    """
    对称对数函数，用于处理具有大范围的输入值。
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

class HRG_Student_Encoder(nn.Module):
    """
    分层博弈学生编码器 (V3 - 最终版)
    - 严格遵守POMDP假设，obs中只包含自身和队友信息。
    - 对手信息完全来自于Transformer处理后的history_feats。
    - HLS通过注意力机制，权衡“团队态势”和“对手态势”的重要性。
    """
    def __init__(self, env_cfg, self_input_dim=8, teammate_input_dim=7, hidden_dim=128):
        super().__init__()
        self.num_p = env_cfg.num_p
        self.num_teammates = env_cfg.num_p - 1
        self.hidden_dim = hidden_dim
        
        # --- 0. 输入归一化层 ---
        total_obs_dim = self_input_dim + teammate_input_dim * self.num_teammates
        self.input_norm = nn.LayerNorm(total_obs_dim)

        # --- A. 感知层 (Perception) ---
        self.self_enc = nn.Sequential(nn.Linear(self_input_dim, hidden_dim), nn.Tanh()) 
        self.teammate_enc = nn.Sequential(nn.Linear(teammate_input_dim, hidden_dim), nn.Tanh())

        # --- B. HLS (High-Level Strategy) - 博弈态势权衡 ---
        # Query: "我是谁?"
        self.hls_query = nn.Linear(hidden_dim, hidden_dim)
        # Keys: "团队情况" 和 "敌人情况"
        self.hls_key_team = nn.Linear(hidden_dim, hidden_dim)
        self.hls_key_evader = nn.Linear(hidden_dim, hidden_dim)
        
        # --- C. LLS 接口 (Output Interface) ---
        # 输出维度 = 自身嵌入 + 最终的博弈上下文
        self.output_dim = hidden_dim + hidden_dim

    def forward(self, obs, history_feats):
        batch_size = obs.shape[0]
        
        # --- 1. 输入归一化 ---
        obs_normalized = self.input_norm(obs)

        # --- 2. 数据解析 ---
        self_in = obs_normalized[:, :8]
        teammates_in = obs_normalized[:, 8:].view(batch_size, self.num_teammates, 7)
        
        # --- 3. 编码 (Perception) ---
        self_emb = self.self_enc(self_in) # [B, H]
        teammate_embs = self.teammate_enc(teammates_in) # [B, Np-1, H]
        
        # --- 4. HLS: 权衡团队与对手 ---
        # 生成团队态势的统一表示 (通过平均池化)
        team_context = teammate_embs.mean(dim=1) # [B, H]
        
        # Query: "基于我的状态，我应该如何分配注意力？"
        Q = self.hls_query(self_emb).unsqueeze(1) # [B, 1, H]
        
        # Keys: "团队态势" vs "对手态势"
        K_team = self.hls_key_team(team_context).unsqueeze(1)
        K_evader = self.hls_key_evader(history_feats).unsqueeze(1)
        
        # 将两个Key拼接，形成注意力评估的范围
        K = torch.cat([K_team, K_evader], dim=1) # [B, 2, H]
        
        # 计算注意力分数
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1) # [B, 1, 2] -> 对"团队"和"对手"的注意力权重
        
        # Values: 两个态势的原始特征向量
        V = torch.stack([team_context, history_feats], dim=1) # [B, 2, H]
        
        # 根据权重，生成加权的最终博弈上下文
        final_context = torch.bmm(attn_weights, V).squeeze(1) # [B, H]
        
        # --- 5. 整合输出 ---
        # 这个 student_features 就是我们要和 Teacher 对齐的向量
        student_features = torch.cat([self_emb, final_context], dim=-1)
        
        # 返回结构化特征，以及用于分析的注意力权重
        return student_features, attn_weights

class Aligned_Teacher(nn.Module):
    """
    维度对齐的教师网络
    接收上帝视角的特权信息，输出一个与学生网络输出维度完全相同的“理想态势感知向量”。
    """
    def __init__(self, priv_obs_dim, student_out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(priv_obs_dim), # 在输入端进行归一化
            nn.Linear(priv_obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            # 关键: 输出维度严格对齐 Student Encoder 的输出
            nn.Linear(256, student_out_dim) 
        )
    
    def forward(self, priv_obs):
        # 直接将特权信息传入网络
        return self.net(priv_obs)
