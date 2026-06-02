"""
护卫侦照场景配置
===============
三星场景：红色高价值星(Red HV) + 红色护卫星(Red Escort) + 蓝色侦照星(Blue Recon)

初始时间：BJT 2027-09-01 20:00 = UTC 2027-09-01 12:00:00
JD_EPOCH = 2461650.0  (UTC 2027-09-01 12:00:00)
  └─ C++ 默认值 2461650.166667 = UTC 2027-09-01 16:00:00，差 4 小时
  └─ 通过 cfg_dict["jd_epoch"] 传入 C++ settings，不覆盖默认常量

C++ 多智能体框架中的角色映射（名称与用户场景相反）：
  evader[0] = blue_sat_0  → 红色高价值星 (Red HV)  — 无机动，固定轨道
  evader[1] = blue_sat_1  → 红色护卫星   (Red Esc) — 第一阶段无机动，第二阶段加载策略
  pursuer[0] = red_sat_0  → 蓝色侦照星   (Blue Recon) — RL 训练目标

侦照成功条件：
  dist(Blue, RedHV) ≤ 20 km  AND  solar_illum_angle ≤ 60°
  持续累计 ≥ 200 s (一个时间步长)
"""

from dataclasses import dataclass, asdict
from oge_py import OGEEnvCfg

# ── 时间 (BJT 2027-09-01 20:00 = UTC 2027-09-01 12:00:00) ────────────────────
# Julian Day 计算：Unix timestamp / 86400 + 2440587.5
#   2027-09-01 12:00:00 UTC → Unix = 1819915200 s → JD = 2461650.0
JD_EPOCH_ESCORT_RECON = 2461650.0   # 本场景正确 JD epoch

# ── 固定轨道根数（六根数，角度单位 rad，距离单位 km） ────────────────────────
# 注意：由于 C++ reset() 会随机化 TA，固定初始位置通过 reset_with_states 传入
# 这里的根数用于 Python 层计算初始 ECI 位置

# 红色高价值星 (Red HV)
RED_HV_OE = dict(
    a    = 42169.502913,   # km
    e    = 0.0,
    i    = 0.002287,       # rad
    raan = 1.592853,       # rad
    w    = 0.0,            # rad
    M    = 0.419833,       # rad  (平近点角)
)

# 红色护卫星 (Red Escort)
RED_ESC_OE = dict(
    a    = 42169.502913,
    e    = 0.0,
    i    = 0.002287,
    raan = 1.592853,
    w    = 0.0,
    M    = 0.421133,
)

# 蓝色侦照星初始化分布（在 Python wrapper 中通过采样得到）
BLUE_DIST_MIN_KM   = 190.0   # 与红HV的初始距离下限 km
BLUE_DIST_MAX_KM   = 195.0   # 与红HV的初始距离上限 km
BLUE_SUN_ANGLE_MIN = 43.0    # 太阳角下限 deg
BLUE_SUN_ANGLE_MAX = 47.0    # 太阳角上限 deg

# ── 任务参数 ─────────────────────────────────────────────────────────────────
RECON_DIST_THRESHOLD_KM    = 20.0         # 侦照距离阈值 km
RECON_ANGLE_THRESHOLD_DEG  = 60.0        # 侦照太阳角阈值 deg
RECON_DURATION_SEC         = 200.0       # 侦照持续时间 s (= 1 个时间步长)
ESCORT_THREAT_DIST_KM      = 20.0        # 护卫星威胁距离（被"拦截"距离）km

# ── OGE 环境配置（传给 C++ 后端） ────────────────────────────────────────────
# 角色映射说明：
#   "red"  in OGEEnvCfg = pursuer  = 蓝色侦照星
#   "blue" in OGEEnvCfg = evader   = 红色高价值星 / 护卫星
env_cfg = OGEEnvCfg(
    random_seed               = 42,
    # 基准轨道（C++ reset() 用，实际通过 reset_with_states 覆盖）
    sma_base                  = 42169.502913,
    ecc_base                  = 0.0,
    incl_base                 = 0.002287,
    RA_base                   = 1.592853,
    w_base                    = 0.0,
    TA_base                   = 0.419833,
    # 燃料预算
    dv_init_red               = 0.020,      # 蓝侦照星：20 m/s 总燃料
    dv_init_blue              = 0.0,        # 红方：无机动
    dv_max_per_step_red       = 0.002,      # 蓝侦照星：最大单步 2 m/s
    dv_max_per_step_blue      = 0.0,        # 红方：无机动
    # 碰撞距离（极小值，由 Python 层处理任务终止）
    capture_distance          = 0.1,
    # 时间参数
    timestep                  = 200.0,      # 每步 200 s
    terminal_time             = 72000.0,    # 20 小时任务时间
    # 初始化分布（被 reset_with_states 覆盖，保留合理值避免 C++ 报错）
    sma_perturb_max           = 0.0,
    dist_init_offset_min      = 190.0,
    dist_init_offset_max      = 195.0,
    # 奖励权重（由 Python wrapper 控制，C++ 层全部置零）
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
