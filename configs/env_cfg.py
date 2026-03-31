from oge_py import OGEEnvCfg

env_cfg = OGEEnvCfg(
    random_seed=10,
    sma_base=42164.0,
    ecc_base=0.0,
    incl_base=0.0,
    RA_base=0.0,
    w_base=0.0,
    TA_base=0.0,
    dv_init_red=0.02,  # 20m/s
    dv_init_blue=0.001,  # 极小值，实际无机动
    dv_max_per_step_red=0.002,  # 2m/s
    dv_max_per_step_blue=0.0001,
    capture_distance=1.0,  # 碰撞距离1km，不是侦照距离
    timestep=200.0,  # 机动间隔200秒
    terminal_time=18000.0*3,  # 5小时任务时间 (5*3600)
    sma_perturb_max=10.0,
    dist_init_offset_min=190.0,  # 扩大下限，覆盖更多初始构型
    dist_init_offset_max=210.0,  # 扩大上限，覆盖固定场景213km
    reward_time_weight=0.0,
    reward_formation_weight=0.0,
    reward_fuel_weight=-0.1,
    reward_capture_weight=0.0,
    reward_timeout_weight=0.0,
    reward_fuelout_weight=-5.0,
    reward_phase_dist_weight=0.0,
    reward_far_sma_penalty_scale=0.0,
    reward_far_drift_scale=0.0,
    reward_far_drift_max=0.0,
    reward_far_angle_weight=0.0,
    reward_near_energy_scale=0.0,
    reward_near_energy_weight=0.0,
    reward_dist_capture_bonus=0.0,
    reward_dist_min=0.0,
    reward_alpha_scale=0.0,
)