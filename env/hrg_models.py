import torch
import torch.nn as nn
import torch.nn.functional as F

def symlog(x):
    """
    对称对数函数，用于处理具有大范围的输入值。
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

class HRG_Student_Encoder(nn.Module):
    def __init__(self, env_cfg, self_input_dim=8, target_input_dim=3, teammate_input_dim=7, hidden_dim=128):
        super().__init__()
        self.num_p = env_cfg.num_p
        self.num_teammates = env_cfg.num_p - 1
        self.hidden_dim = hidden_dim
        
        # Obs = Self(8) + Target(3*Ne) + Teammates(...)
        self.target_obs_dim = target_input_dim * env_cfg.num_e
        total_obs_dim = self_input_dim + self.target_obs_dim + teammate_input_dim * self.num_teammates
        self.input_norm = nn.LayerNorm(total_obs_dim)

        # A. 感知层
        # 1. 自身编码
        self.self_enc = nn.Sequential(nn.Linear(self_input_dim, hidden_dim), nn.Tanh())
        
        # 2. 目标即时观测编码 (新增)
        # 这代表了“不完美但实时的视觉信息”
        self.target_enc = nn.Sequential(nn.Linear(self.target_obs_dim, hidden_dim), nn.Tanh())
        
        # 3. 队友编码
        self.teammate_enc = nn.Sequential(nn.Linear(teammate_input_dim, hidden_dim), nn.Tanh())

        # B. HLS 
        # 我们需要融合 Self 和 TargetObs 作为 Query
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim) # 融合 Self + TargetObs
        
        self.hls_query = nn.Linear(hidden_dim, hidden_dim)
        self.hls_key_team = nn.Linear(hidden_dim, hidden_dim)
        self.hls_key_evader = nn.Linear(hidden_dim, hidden_dim) # 来自 History Transformer
        
        self.output_dim = hidden_dim + hidden_dim

    def forward(self, obs, history_feats):
        batch_size = obs.shape[0]
        obs_normalized = self.input_norm(obs)

        # 切分数据
        # [Self(8) | Target(3*Ne) | Teammates...]
        idx_self = 8
        idx_target = idx_self + self.target_obs_dim
        
        self_in = obs_normalized[:, :idx_self]
        target_in = obs_normalized[:, idx_self:idx_target]
        teammates_in = obs_normalized[:, idx_target:].view(batch_size, self.num_teammates, 7)
        
        # 编码
        self_emb = self.self_enc(self_in)       # [B, H]
        target_emb = self.target_enc(target_in) # [B, H]
        teammate_embs = self.teammate_enc(teammates_in) # [B, Np-1, H]
        
        # HLS 逻辑升级
        # 现在的“我”不仅包含燃料状态，还包含我看到的那个残缺的目标位置
        # 这有助于网络判断：如果我看不到目标（Obs是旧的），我是不是该多信一点 History？
        self_context = torch.cat([self_emb, target_emb], dim=-1)
        self_context = F.relu(self.fusion_layer(self_context)) # [B, H]
        
        team_context = teammate_embs.mean(dim=1)
        
        Q = self.hls_query(self_context).unsqueeze(1)
        K_team = self.hls_key_team(team_context).unsqueeze(1)
        K_evader = self.hls_key_evader(history_feats).unsqueeze(1)
        
        K = torch.cat([K_team, K_evader], dim=1)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        V = torch.stack([team_context, history_feats], dim=1)
        final_context = torch.bmm(attn_weights, V).squeeze(1)
        
        # 输出：融合后的自我感知 + 博弈上下文
        student_features = torch.cat([self_context, final_context], dim=-1)
        
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
