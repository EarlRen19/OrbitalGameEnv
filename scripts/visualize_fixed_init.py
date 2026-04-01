"""可视化 Fixed Init 场景的轨迹（论文级别图表 + GIF 动画）"""

import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import oge_py
from modules.networks import Policy
from configs.env_cfg import env_cfg
from skrl.resources.preprocessors.torch import RunningStandardScaler

# 设置中文字体（可选，如果需要中文标签）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

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


def run_fixed_episode(checkpoint_path):
    """运行 Fixed Init episode 并记录完整轨迹"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from oge_py import OGEEnv
    from modules.env_wrapper import OGESingleEnvWrapper

    raw_env = OGEEnv(env_cfg)
    env = OGESingleEnvWrapper(raw_env, env_cfg)

    dv_max = 0.002 / (3 ** 0.5)
    policy = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        dv_max=dv_max,
        clip_actions=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["policy"])
    policy.to(device)
    policy.eval()

    state_preprocessor = RunningStandardScaler(size=env.observation_space.shape[0], device=device)
    if "state_preprocessor" in checkpoint:
        state_preprocessor.load_state_dict(checkpoint["state_preprocessor"])
    state_preprocessor.eval()

    # Reset with fixed states
    fixed_states = build_fixed_states(env_cfg.dv_init_red, env_cfg.dv_init_blue)
    obs, info = env._env.reset(options={"states": fixed_states})
    env._last_obs = np.asarray(obs, dtype=np.float32)
    env._recon_time_accumulated = 0.0
    env._last_in_recon_zone = False
    red_obs = env._last_obs[1]
    env._last_dist = np.linalg.norm(red_obs[6:9])
    refined_obs = env._get_refined_obs(env._last_obs)
    obs_t = torch.tensor(refined_obs, dtype=torch.float32).unsqueeze(0).to(device)

    trajectory = {
        "time": [0.0],
        "distance": [],
        "solar_angle": [],
        "in_zone": [],
        "red_pos": [],
        "blue_pos": [],
        "fuel": [],
        "sun_dir": [],  # 添加太阳方向
    }

    done = False
    step = 0

    while not done:
        # 记录当前状态
        obs_np = obs_t.squeeze().cpu().numpy()
        distance = obs_np[6] * 20.0
        solar_angle = obs_np[7] * np.pi
        in_zone = (distance <= 20.0 and solar_angle <= np.deg2rad(60.0))
        fuel = obs_np[11]
        sun_dir = obs_np[8:11]  # 太阳方向单位向量

        red_obs = env._last_obs[1]
        blue_obs = env._last_obs[0]

        trajectory["distance"].append(distance)
        trajectory["solar_angle"].append(np.rad2deg(solar_angle))
        trajectory["in_zone"].append(in_zone)
        trajectory["red_pos"].append(red_obs[0:3].copy())
        trajectory["blue_pos"].append(blue_obs[0:3].copy())
        trajectory["fuel"].append(fuel)
        trajectory["sun_dir"].append(sun_dir.copy())

        # Step
        with torch.no_grad():
            normalized_obs = state_preprocessor(obs_t)
            action = policy.act({"states": normalized_obs}, role="policy")[0]
        obs_t, reward, terminated, truncated, info = env.step(action)
        obs_t = obs_t.to(device)

        done = terminated.item() or truncated.item()
        step += 1
        trajectory["time"].append(step * env_cfg.timestep)

    return trajectory


def plot_static_figure(trajectory, output_path):
    """生成论文级别的静态图（4 子图）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fixed Init Scenario Trajectory Analysis', fontsize=16, fontweight='bold')

    time = trajectory["time"][:-1]
    timesteps = np.arange(len(time))  # 转换为 timestep
    distance = trajectory["distance"]
    solar_angle = trajectory["solar_angle"]
    in_zone = trajectory["in_zone"]
    fuel = trajectory["fuel"]

    # 子图 1: 距离随时间变化
    ax1 = axes[0, 0]
    ax1.plot(timesteps, distance, 'b-', linewidth=2, label='Distance')
    ax1.axhline(y=20, color='r', linestyle='--', linewidth=1.5, label='Recon Threshold (20km)')
    ax1.fill_between(timesteps, 0, 20, alpha=0.2, color='green', label='Recon Zone')
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Distance (km)', fontsize=12)
    ax1.set_title('Distance vs Timestep', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 子图 2: 太阳角随时间变化
    ax2 = axes[0, 1]
    ax2.plot(timesteps, solar_angle, 'orange', linewidth=2, label='Solar Angle')
    ax2.axhline(y=60, color='r', linestyle='--', linewidth=1.5, label='Angle Threshold (60°)')
    ax2.fill_between(timesteps, 0, 60, alpha=0.2, color='green', label='Valid Zone')
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Solar Angle (°)', fontsize=12)
    ax2.set_title('Solar Angle vs Timestep', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 子图 3: 距离 vs 太阳角（相空间图）
    ax3 = axes[1, 0]
    colors = ['green' if iz else 'red' for iz in in_zone]
    scatter = ax3.scatter(distance, solar_angle, c=colors, s=30, alpha=0.6)
    ax3.axvline(x=20, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax3.axhline(y=60, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax3.add_patch(plt.Rectangle((0, 0), 20, 60, fill=True, alpha=0.1, color='green'))
    ax3.set_xlabel('Distance (km)', fontsize=12)
    ax3.set_ylabel('Solar Angle (°)', fontsize=12)
    ax3.set_title('Phase Space (Distance vs Solar Angle)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(['In Zone', 'Out of Zone'], loc='upper right')

    # 子图 4: 燃料消耗
    ax4 = axes[1, 1]
    fuel_percent = np.array(fuel) * 100
    ax4.plot(timesteps, fuel_percent, 'purple', linewidth=2)
    ax4.set_xlabel('Timestep', fontsize=12)
    ax4.set_ylabel('Remaining Fuel (%)', fontsize=12)
    ax4.set_title('Fuel Consumption', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 105])

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # 在底部添加任务说明（仅英文）
    fig.text(0.5, 0.01, 'Reconnaissance and Illumination Task',
             ha='center', fontsize=13, fontweight='bold', style='italic')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"静态图已保存: {output_path}")
    plt.close()


def create_local_gif(trajectory, output_path):
    """生成局部放大 GIF（以蓝星为中心的相对运动）"""
    red_pos = np.array(trajectory["red_pos"])
    blue_pos = np.array(trajectory["blue_pos"])
    in_zone = trajectory["in_zone"]
    distance = trajectory["distance"]
    solar_angle = trajectory["solar_angle"]

    fig, ax = plt.subplots(figsize=(10, 10))

    def update(frame):
        ax.clear()

        # 计算相对位置（红星相对于蓝星）
        rel_pos = red_pos[:frame+1] - blue_pos[:frame+1]

        # 蓝星固定在原点（蓝色方块）
        ax.scatter(0, 0, c='blue', s=250, marker='s', edgecolors='black',
                   linewidths=3, label='Blue Sat (Center)', zorder=5)

        # 绘制红星相对轨迹
        ax.plot(rel_pos[:, 0], rel_pos[:, 1], 'r-', linewidth=2, alpha=0.6,
                label='Red Sat Trajectory')

        # 当前红星位置
        color = 'green' if in_zone[frame] else 'red'
        ax.scatter(rel_pos[-1, 0], rel_pos[-1, 1], c=color, s=250, marker='o',
                   edgecolors='black', linewidths=3, label='Red Sat', zorder=5)

        # 连线显示距离
        ax.plot([0, rel_pos[-1, 0]], [0, rel_pos[-1, 1]],
                'k--', linewidth=2, alpha=0.5)

        # 绘制侦照区域（20km 圆圈，以蓝星为中心）
        recon_zone = plt.Circle((0, 0), 20, color='green', alpha=0.15,
                                linestyle='--', linewidth=2, fill=True, label='Recon Zone (20km)')
        ax.add_patch(recon_zone)

        # 动态坐标轴范围
        max_dist = max(300, distance[frame] * 1.5)
        ax.set_xlim([-max_dist, max_dist])
        ax.set_ylim([-max_dist, max_dist])
        ax.set_xlabel('Relative X (km)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Relative Y (km)', fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')

        # 标题显示详细信息
        status = "IN ZONE" if in_zone[frame] else "OUT OF ZONE"
        color_text = "green" if in_zone[frame] else "red"
        ax.set_title(f'Timestep: {frame} | Distance: {distance[frame]:.1f} km | '
                     f'Solar Angle: {solar_angle[frame]:.1f}° | Status: {status}',
                     fontsize=13, fontweight='bold', color=color_text, pad=15)

        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        # 底部任务说明
        fig.text(0.5, 0.02, 'Reconnaissance and Illumination Task (Local View)',
                 ha='center', fontsize=12, fontweight='bold', style='italic')

    # 每 5 帧取一帧
    frames = range(0, len(red_pos), 5)
    anim = FuncAnimation(fig, update, frames=frames, interval=200)

    writer = PillowWriter(fps=5)
    anim.save(output_path, writer=writer)
    print(f"GIF 动画（局部视图）已保存: {output_path}")
    plt.close()


def create_gif(trajectory, output_path):
    """生成 GIF 动画（2D 俯视图，从 Z 轴向下看）"""
    red_pos = np.array(trajectory["red_pos"])
    blue_pos = np.array(trajectory["blue_pos"])
    in_zone = trajectory["in_zone"]
    distance = trajectory["distance"]

    # 计算坐标范围
    all_pos = np.vstack([red_pos, blue_pos])
    max_range = np.max(np.abs(all_pos[:, :2])) * 1.1

    fig, ax = plt.subplots(figsize=(10, 10))

    def update(frame):
        ax.clear()

        # 绘制地球（圆形）
        earth = plt.Circle((0, 0), 6371, color='cyan', alpha=0.3, label='Earth')
        ax.add_patch(earth)

        # 绘制轨迹
        ax.plot(red_pos[:frame+1, 0], red_pos[:frame+1, 1],
                'r-', linewidth=2, alpha=0.6, label='Red Sat Trajectory')
        ax.plot(blue_pos[:frame+1, 0], blue_pos[:frame+1, 1],
                'b-', linewidth=2, alpha=0.6, label='Blue Sat Trajectory')

        # 当前位置（先画蓝色，再画红色，红色在上层）
        ax.scatter(blue_pos[frame, 0], blue_pos[frame, 1],
                   c='blue', s=200, marker='s', edgecolors='black', linewidths=2.5,
                   label='Blue Sat', zorder=4)

        color = 'green' if in_zone[frame] else 'red'
        ax.scatter(red_pos[frame, 0], red_pos[frame, 1],
                   c=color, s=200, marker='o', edgecolors='black', linewidths=2.5,
                   label=f'Red Sat ({"IN ZONE" if in_zone[frame] else "OUT"})', zorder=5)

        # 绘制连线显示距离
        ax.plot([red_pos[frame, 0], blue_pos[frame, 0]],
                [red_pos[frame, 1], blue_pos[frame, 1]],
                'k--', linewidth=1.5, alpha=0.5)

        # 固定坐标轴
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_xlabel('X (km)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y (km)', fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')

        # 标题显示进度和状态
        status = "IN ZONE" if in_zone[frame] else "OUT OF ZONE"
        color_text = "green" if in_zone[frame] else "red"
        ax.set_title(f'Timestep: {frame}/{len(red_pos)-1} | Distance: {distance[frame]:.1f} km | Status: {status}',
                     fontsize=14, fontweight='bold', color=color_text, pad=15)

        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        # 底部任务说明
        fig.text(0.5, 0.02, 'Reconnaissance and Illumination Task',
                 ha='center', fontsize=12, fontweight='bold', style='italic')

    # 每 5 帧取一帧，放慢速度
    frames = range(0, len(red_pos), 5)
    anim = FuncAnimation(fig, update, frames=frames, interval=200)

    writer = PillowWriter(fps=5)
    anim.save(output_path, writer=writer)
    print(f"GIF 动画（全局视图）已保存: {output_path}")
    plt.close()

    blue_pos = np.array(trajectory["blue_pos"])
    in_zone = trajectory["in_zone"]
    distance = trajectory["distance"]

    # 计算坐标范围
    all_pos = np.vstack([red_pos, blue_pos])
    max_range = np.max(np.abs(all_pos[:, :2])) * 1.1

    fig, ax = plt.subplots(figsize=(10, 10))

    def update(frame):
        ax.clear()

        # 绘制地球（圆形）
        earth = plt.Circle((0, 0), 6371, color='cyan', alpha=0.3, label='Earth')
        ax.add_patch(earth)

        # 绘制轨迹
        ax.plot(red_pos[:frame+1, 0], red_pos[:frame+1, 1],
                'r-', linewidth=2, alpha=0.6, label='Red Sat Trajectory')
        ax.plot(blue_pos[:frame+1, 0], blue_pos[:frame+1, 1],
                'b-', linewidth=2, alpha=0.6, label='Blue Sat Trajectory')

        # 当前位置（先画蓝色，再画红色，红色在上层）
        ax.scatter(blue_pos[frame, 0], blue_pos[frame, 1],
                   c='blue', s=200, marker='s', edgecolors='black', linewidths=2.5,
                   label='Blue Sat', zorder=4)

        color = 'green' if in_zone[frame] else 'red'
        ax.scatter(red_pos[frame, 0], red_pos[frame, 1],
                   c=color, s=200, marker='o', edgecolors='black', linewidths=2.5,
                   label=f'Red Sat ({"IN ZONE" if in_zone[frame] else "OUT"})', zorder=5)

        # 绘制连线显示距离
        ax.plot([red_pos[frame, 0], blue_pos[frame, 0]],
                [red_pos[frame, 1], blue_pos[frame, 1]],
                'k--', linewidth=1.5, alpha=0.5)

        # 固定坐标轴
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_xlabel('X (km)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y (km)', fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')

        # 标题显示进度和状态
        status = "IN ZONE" if in_zone[frame] else "OUT OF ZONE"
        color_text = "green" if in_zone[frame] else "red"
        ax.set_title(f'Timestep: {frame}/{len(red_pos)-1} | Distance: {distance[frame]:.1f} km | Status: {status}',
                     fontsize=14, fontweight='bold', color=color_text, pad=15)

        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        # 底部任务说明
        fig.text(0.5, 0.02, 'Reconnaissance and Illumination Task',
                 ha='center', fontsize=12, fontweight='bold', style='italic')

    # 每 5 帧取一帧，放慢速度
    frames = range(0, len(red_pos), 5)
    anim = FuncAnimation(fig, update, frames=frames, interval=200)  # 200ms 更慢

    writer = PillowWriter(fps=5)  # 降低 fps
    anim.save(output_path, writer=writer)
    print(f"GIF 动画（全局视图）已保存: {output_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint 路径")
    parser.add_argument("--output-dir", type=str, default="visualizations",
                        help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("运行 Fixed Init episode...")
    trajectory = run_fixed_episode(args.checkpoint)

    print(f"轨迹长度: {len(trajectory['distance'])} 步")
    print(f"最终距离: {trajectory['distance'][-1]:.2f} km")
    print(f"最终太阳角: {trajectory['solar_angle'][-1]:.1f}°")
    print(f"是否成功: {'是' if trajectory['in_zone'][-1] else '否'}")

    # 生成静态图
    static_path = os.path.join(args.output_dir, "fixed_init_analysis.png")
    plot_static_figure(trajectory, static_path)

    # 生成全局 GIF
    gif_global_path = os.path.join(args.output_dir, "fixed_init_global.gif")
    print("生成全局 GIF 动画...")
    create_gif(trajectory, gif_global_path)

    # 生成局部 GIF
    gif_local_path = os.path.join(args.output_dir, "fixed_init_local.gif")
    print("生成局部 GIF 动画...")
    create_local_gif(trajectory, gif_local_path)

    print("\n可视化完成！")


if __name__ == "__main__":
    main()
