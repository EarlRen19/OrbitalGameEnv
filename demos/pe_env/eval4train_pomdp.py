import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from collections import defaultdict
from tqdm import tqdm
import io
from pathlib import Path

# 将项目根目录添加到Python路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from env.mpe_pomdp_env import MPE_POMDP_Env, MPE_POMDP_EnvCfg
from demos.pe_env.train_pomdp import ActorCritic, TrainConfig

def load_checkpoint(path, device):
    """加载模型检查点"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    print(f"Loading checkpoint from {path}...")
    return torch.load(path, map_location=device)

def set_axes_equal(ax):
    """让3D Matplotlib的坐标轴比例尺相等"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def save_static_plot(traj_data, pursuer_ids, evader_id, filename):
    """生成并保存在绝对坐标系下的静态轨迹图"""
    print(f"Generating static trajectory plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 地球
    R_earth = 6378.137 # km
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x = R_earth * np.cos(u) * np.sin(v)
    y = R_earth * np.sin(u) * np.sin(v)
    z = R_earth * np.cos(v)
    ax.plot_surface(x, y, z, color='blue', alpha=0.1, edgecolor='none')

    scale = 1000.0 # m to km
    
    # 追击者轨迹
    colors = plt.cm.jet(np.linspace(0, 1, len(pursuer_ids)))
    for i, pid in enumerate(pursuer_ids):
        if pid in traj_data:
            pos = traj_data[pid] / scale
            ax.plot(pos[:,0], pos[:,1], pos[:,2], label=f'Pursuer {i}', color=colors[i], linewidth=1.5, alpha=0.8)
            ax.scatter(pos[0,0], pos[0,1], pos[0,2], marker='o', color=colors[i], s=30, alpha=0.8, label=f'P{i} Start')
            ax.scatter(pos[-1,0], pos[-1,1], pos[-1,2], marker='x', color=colors[i], s=60, linewidth=2, label=f'P{i} End')

    # 逃逸者轨迹
    if evader_id in traj_data:
        pos = traj_data[evader_id] / scale
        ax.plot(pos[:,0], pos[:,1], pos[:,2], label='Evader', color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.scatter(pos[0,0], pos[0,1], pos[0,2], marker='o', color='red', s=30, alpha=0.8, label='Evader Start')
        ax.scatter(pos[-1,0], pos[-1,1], pos[-1,2], marker='*', color='red', s=100, label='Evader End')

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Absolute Trajectory')
    ax.legend()
    set_axes_equal(ax)

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Static plot saved to {filename}")

def create_gif(traj_data, evader_id, pursuer_ids, filename="trajectory.gif", duration=0.1):
    """创建以逃逸者为中心的GIF动画 (高清版)"""
    print(f"Generating high-quality GIF, this may take a moment...")
    images = []
    
    evader_traj = traj_data[evader_id]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    max_range = 0
    for i in range(len(evader_traj)):
        for pid in pursuer_ids:
            if pid in traj_data:
                p_traj = traj_data[pid]
                if i < len(p_traj):
                    rel_pos = p_traj[i] - evader_traj[i]
                    max_range = max(max_range, np.max(np.abs(rel_pos)))

    plot_radius = max_range / 1000 * 1.2

    for i in tqdm(range(len(evader_traj)), desc="Generating GIF frames"):
        ax.clear()
        ax.scatter(0, 0, 0, marker='*', color='red', s=200, label='Evader (Reference)')
        colors = plt.cm.jet(np.linspace(0, 1, len(pursuer_ids)))
        for idx, pid in enumerate(pursuer_ids):
            if pid in traj_data:
                p_traj = traj_data[pid]
                if i < len(p_traj):
                    rel_pos = (p_traj[i] - evader_traj[i]) / 1000.0
                    ax.scatter(rel_pos[0], rel_pos[1], rel_pos[2], marker='o', color=colors[idx], s=80, label=f'Pursuer {idx}' if i==0 else "")

        ax.set_xlim([-plot_radius, plot_radius])
        ax.set_ylim([-plot_radius, plot_radius])
        ax.set_zlim([-plot_radius, plot_radius])
        ax.set_xlabel('Relative X (km)')
        ax.set_ylabel('Relative Y (km)')
        ax.set_zlabel('Relative Z (km)')
        ax.set_title(f'Evader-Centric View (Step {i})')
        if i == 0:
            ax.legend()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        
    plt.close(fig)
    imageio.mimsave(filename, images, duration=duration)
    print(f"High-quality GIF saved to {filename}")

def run_eval(args):
    """主评估函数"""
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    checkpoint = load_checkpoint(args.checkpoint, device)
    
    # --- 1. 创建结果文件夹 ---
    run_name = Path(args.checkpoint).parent.parent.name
    checkpoint_name = Path(args.checkpoint).stem
    output_dir_name = f"{run_name}_{checkpoint_name}"
    results_base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    output_dir = os.path.join(results_base_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # --- 2. 配置和加载模型 ---
    env_cfg = MPE_POMDP_EnvCfg()
    train_cfg = TrainConfig()

    if 'env_cfg' in checkpoint:
        env_cfg_ckpt = checkpoint['env_cfg']
        for key, value in env_cfg_ckpt.items():
            if hasattr(env_cfg, key):
                setattr(env_cfg, key, value)
    
    env_cfg.num_p = args.num_p if args.num_p is not None else env_cfg.num_p
    env_cfg.num_e = args.num_e
    train_cfg.use_encoder = args.use_encoder if args.use_encoder is not None else train_cfg.use_encoder
    env_cfg.history_len = args.history_len
    env_cfg.evader_policy_level = args.evader_policy_level

    print(f"Evaluating with: Pursuers={env_cfg.num_p}, Evaders={env_cfg.num_e}, Use Encoder={train_cfg.use_encoder}")

    env = MPE_POMDP_Env(env_cfg)
    
    pursuer_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
    evader_ids = [f'e_{i}' for i in range(env_cfg.num_e)]
    
    student_obs_dim = env.observation_spaces[pursuer_ids[0]].shape[0]
    privileged_obs_dim = (env_cfg.num_p + env_cfg.num_e) * 6
    act_dim = env.action_spaces[pursuer_ids[0]].shape[0]

    model = ActorCritic(student_obs_dim, privileged_obs_dim, act_dim, env_cfg, train_cfg).to(device)
    model.load_state_dict(checkpoint['agent_state_dict'])
    model.eval()
    print("Model loaded successfully.")

    # --- 3. 开始评估循环 ---
    success_count = 0
    media_saved_traj = None

    print(f"Running evaluation for {args.test_episodes} episodes...")
    for ep in tqdm(range(args.test_episodes), desc="Evaluating Episodes"):
        obs, infos = env.reset(seed=args.seed + ep)
        traj_data = defaultdict(list)
        done = False
        
        while not done:
            pursuer_obs_list = [torch.Tensor(obs[name]).to(device) for name in pursuer_ids]
            pursuer_obs_tensor = torch.stack(pursuer_obs_list)
            pursuer_priv_obs_list = [torch.Tensor(infos[name]['privileged_state']).to(device) for name in pursuer_ids]
            pursuer_priv_obs_tensor = torch.stack(pursuer_priv_obs_list)
            history_list = [torch.from_numpy(infos[name][f'history_input_{evader_ids[0]}']).float().to(device) for name in pursuer_ids]
            history_tensor = torch.stack(history_list)
            mask_list = [torch.from_numpy(infos[name][f'history_mask_{evader_ids[0]}']).float().to(device) for name in pursuer_ids]
            mask_tensor = torch.stack(mask_list)

            with torch.no_grad():
                actions_tensor, _, _, _, _ = model.get_action_and_value(
                    pursuer_obs_tensor, pursuer_priv_obs_tensor, history_tensor, mask_tensor, deterministic=True
                )
            
            evader_actions = env.get_evader_actions()
            actions_to_step = {name: actions_tensor[i].cpu().numpy() for i, name in enumerate(pursuer_ids)}
            actions_to_step.update(evader_actions)

            next_obs, _, terminations, truncations, infos = env.step(actions_to_step)
            
            for agent_id in env.states:
                traj_data[agent_id].append(env.states[agent_id][:3].copy())

            obs = next_obs
            if any(terminations.values()) or any(truncations.values()):
                done = True
                reason = infos.get(pursuer_ids[0], {}).get('termination_reason', 'unknown')
                if reason == "capture_success":
                    success_count += 1
                    if media_saved_traj is None and args.save_media:
                        media_saved_traj = traj_data
                        print(f"\nEpisode {ep+1} was successful. Trajectory saved for media generation.")

    env.close()

    # --- 4. 最终结果 ---
    success_rate = (success_count / args.test_episodes) * 100
    print("\n" + "="*40)
    print("       EVALUATION REPORT       ")
    print("="*40)
    print(f"Success Rate over {args.test_episodes} episodes: {success_rate:.2f}% ({success_count}/{args.test_episodes})")
    print("="*40 + "\n")

    # --- 5. 生成媒体文件 ---
    if media_saved_traj and args.save_media:
        gif_path = os.path.join(output_dir, "trajectory_animation.gif")
        static_plot_path = os.path.join(output_dir, "trajectory_static.png")
        
        print("\n--- Generating Media ---")
        create_gif(media_saved_traj, evader_ids[0], pursuer_ids, filename=gif_path)
        save_static_plot(media_saved_traj, pursuer_ids, evader_ids[0], filename=static_plot_path)
    elif args.save_media:
        print("No successful episodes were recorded, so no media will be generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MPE POMDP Agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file")
    parser.add_argument("--num_p", type=int, default=None, help="Number of pursuers (loads from checkpoint if not set)")
    parser.add_argument("--num_e", type=int, default=1, help="Number of evaders")
    parser.add_argument("--use_encoder", type=lambda x: (str(x).lower() == 'true'), default=None, help="Use Attention Encoder (loads from checkpoint if not set)")
    parser.add_argument("--history_len", type=int, default=20, help="History length for Transformer")
    parser.add_argument("--evader_policy_level", type=int, default=0, choices=[0, 1, 2], help="Evader policy: 0=Drift, 1=Random, 2=APF")
    
    parser.add_argument("--test_episodes", type=int, default=20, help="Number of episodes to test for success rate")
    parser.add_argument("--save_media", action="store_true", help="Save GIF and static plot for the first successful run")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for inference")
    
    args = parser.parse_args()
    
    run_eval(args)
