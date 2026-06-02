"""
集群护卫侦照场景配置
====================
13 星场景：
  红方高价值星 (Red HV)   × 1  — 无机动，固定轨道，被蓝方目标
  红方护卫星   (Red Esc)  × 6  — RL 训练目标，侦照蓝方、保护红 HV
  蓝方星       (Blue)     × 6  — 规则/对手策略，攻击红 HV

初始时间：北京时间 2023-11-16 22:30 = UTC 2023-11-16 14:30:00
JD_EPOCH = 2460264.770833

C++ 多智能体框架角色映射（num_evaders=7, num_pursuers=6）：
  evader[0]      = blue_sat_0  → 红色高价值星 (Red HV)   — 无机动
  evader[1..6]   = blue_sat_1~6 → 红色护卫星 1~6         — 第二阶段 RL 智能体
  pursuer[0..5]  = red_sat_0~5  → 蓝色星 1~6             — 第一阶段 RL 智能体

蓝方任务分配：
  pursuer[0] = 蓝1：打击红HV，燃料 30m/s
  pursuer[1] = 蓝2：打击红HV，燃料 30m/s
  pursuer[2] = 蓝3：干扰红HV，燃料 20m/s
  pursuer[3] = 蓝4：侦照红HV，燃料 20m/s
  pursuer[4] = 蓝5：侦照红HV，燃料 20m/s
  pursuer[5] = 蓝6：操控红HV，燃料 20m/s

红方护卫分配：
  evader[1] = 红护卫1：侦照蓝1（打击），燃料 30m/s
  evader[2] = 红护卫2：侦照蓝2（打击），燃料 30m/s
  evader[3] = 红护卫3：侦照蓝3（干扰），燃料 20m/s
  evader[4] = 红护卫4：侦照蓝4（侦照），燃料 20m/s
  evader[5] = 红护卫5：侦照蓝5（侦照），燃料 20m/s
  evader[6] = 红护卫6：侦照蓝6（操控），燃料 20m/s
"""

from dataclasses import dataclass, asdict
from oge_py import OGEEnvCfg

# ── 时间 ──────────────────────────────────────────────────────────────────────
# 北京时间 2023-11-16 22:30 = UTC 2023-11-16 14:30:00
# Unix = 1700116200, JD = Unix/86400 + 2440587.5
JD_EPOCH_CLUSTER = 2460264.770833

# ── 精确初始轨道根数（六根数，角度 rad，距离 km） ─────────────────────────────

# 红方高价值星
RED_HV_OE = dict(
    a    = 42165.548506,
    e    = 0.000111,
    i    = 0.002346,
    raan = 1.584594,
    w    = 6.054891,
    M    = 5.495957,
)

# 红方护卫星 1~6
RED_ESC_OE_LIST = [
    dict(a=42128.800784, e=0.001738, i=0.002346, raan=1.584495, w=1.031159, M=4.239059),  # Esc-1
    dict(a=42130.292713, e=0.001795, i=0.002346, raan=1.584495, w=0.993165, M=4.277215),  # Esc-2
    dict(a=42096.664277, e=0.002625, i=0.002346, raan=1.584495, w=1.192620, M=4.078730),  # Esc-3
    dict(a=42114.484464, e=0.002498, i=0.002346, raan=1.584493, w=1.028589, M=4.242990),  # Esc-4
    dict(a=42123.164909, e=0.002428, i=0.002346, raan=1.584493, w=0.949349, M=4.322267),  # Esc-5
    dict(a=42140.072751, e=0.002267, i=0.002346, raan=1.584492, w=0.791518, M=4.480022),  # Esc-6
]

# 蓝方星 1~6
BLUE_REC_OE_LIST = [
    dict(a=42145.439305, e=0.001452, i=0.002345, raan=1.584605, w=4.406570, M=0.857419),  # Blue-1 打击
    dict(a=42146.346177, e=0.001428, i=0.002345, raan=1.584604, w=4.452799, M=0.811319),  # Blue-2 打击
    dict(a=42149.944834, e=0.001442, i=0.002345, raan=1.584603, w=4.546262, M=0.718043),  # Blue-3 干扰
    dict(a=42161.152343, e=0.001635, i=0.002345, raan=1.584602, w=4.681890, M=0.582541),  # Blue-4 侦照
    dict(a=42166.977779, e=0.001754, i=0.002345, raan=1.584602, w=4.725849, M=0.538595),  # Blue-5 侦照
    dict(a=42179.072192, e=0.002024, i=0.002345, raan=1.584601, w=4.776067, M=0.488303),  # Blue-6 操控
]

# ── 燃料预算 (km/s) ────────────────────────────────────────────────────────────
DV_INIT_STRIKE  = 0.030   # 打击：30 m/s
DV_INIT_JAM     = 0.020   # 干扰：20 m/s
DV_INIT_RECON   = 0.020   # 侦照：20 m/s
DV_INIT_OPERATE = 0.020   # 操控：20 m/s
DV_MAX_PER_STEP = 0.002   # 所有任务单步最大 2 m/s

# 蓝方各任务燃料（与 BLUE_REC_OE_LIST 顺序对应）
BLUE_DV_INIT_LIST = [
    DV_INIT_STRIKE,   # Blue-1 打击
    DV_INIT_STRIKE,   # Blue-2 打击
    DV_INIT_JAM,      # Blue-3 干扰
    DV_INIT_RECON,    # Blue-4 侦照
    DV_INIT_RECON,    # Blue-5 侦照
    DV_INIT_OPERATE,  # Blue-6 操控
]

# ── OGE 基础环境配置 ───────────────────────────────────────────────────────────
# dv_init_red  → pursuer (蓝方) 的燃料参考值（C++ 计算 dv_ratio 用，取最大值 30m/s）
# dv_init_blue → evader  (红方护卫) 的燃料参考值（取最大值 30m/s，护卫1/2=30m/s从1.0开始）
#   护卫3~6 实际燃料 20m/s，dv_ratio 初始=0.667，网络会学到"低燃料"信号
env_cfg = OGEEnvCfg(
    random_seed               = 42,
    sma_base                  = RED_HV_OE["a"],
    ecc_base                  = RED_HV_OE["e"],
    incl_base                 = RED_HV_OE["i"],
    RA_base                   = RED_HV_OE["raan"],
    w_base                    = RED_HV_OE["w"],
    TA_base                   = RED_HV_OE["M"],
    dv_init_red               = DV_INIT_STRIKE,   # pursuer 参考燃料（取最大值）
    dv_init_blue              = DV_INIT_STRIKE,   # evader  参考燃料（取最大值，避免 dv_ratio>1）
    dv_max_per_step_red       = DV_MAX_PER_STEP,
    dv_max_per_step_blue      = DV_MAX_PER_STEP,
    capture_distance          = 0.1,
    timestep                  = 200.0,
    terminal_time             = 72000.0,          # 20 小时
    sma_perturb_max           = 0.0,
    dist_init_offset_min      = 30.0,
    dist_init_offset_max      = 200.0,
    reward_time_weight        = 0.0,
    reward_formation_weight   = 0.0,
    reward_fuel_weight        = 0.0,
    reward_capture_weight     = 0.0,
    reward_timeout_weight     = 0.0,
    reward_fuelout_weight     = 0.0,
    reward_phase_dist_weight  = 0.0,
    reward_far_sma_penalty_scale  = 0.0,
    reward_far_drift_scale        = 0.0,
    reward_far_drift_max          = 0.0,
    reward_far_angle_weight       = 0.0,
    reward_near_energy_scale      = 0.0,
    reward_near_energy_weight     = 0.0,
    reward_dist_capture_bonus     = 0.0,
    reward_dist_min               = 0.0,
    reward_alpha_scale            = 0.0,
)
