# OGE — Orbital Game Environment

## 1. 开发环境配置

### 1.1 更新 gcc

安装 GCC 13

```
sudo apt update
sudo apt install build-essential gcc-13 g++-13
```

设置为默认版本

```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100
```

如果系统里有多个版本，可以用以下命令切换：

```
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

安装完后验证：

```
gcc --version
g++ --version
```

输出类似如下内容：

```text
(oge) ➜  OGE git:(main) ✗ gcc --version
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

(oge) ➜  OGE git:(main) ✗ g++ --version
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### 1.2 安装 vcpkg

克隆 `vcpkg` 并且执行安装

```bash
git clone https://www.github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
```

将以下内容添加到 `~/.bashrc` 或者 `~/.zshrc`

```bash
# >>> vcpkg
export VCPKG_ROOT=<path-to-vcpkg>
export PATH=$VCPKG_ROOT:$PATH
# <<< vcpkg
```

### 1.3 Python 环境配置

创建 `conda` 环境并且安装基础依赖

```bash
conda create -n oge python=3.13
conda activate oge
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```

安装 `skrl` 库

```bash
conda activate oge
cd third_party/skrl-1.4.3
pip install -e ".["torch"]" -i https://mirrors.ustc.edu.cn/pypi/simple
```

---

## 2. 编译与安装

### 2.1 完整编译（首次 / 修改了 C++ 代码后）

```bash
cd /home/star/Downloads/oge_2.0/OGE
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..
pip install .
```

编译成功后会生成两个 Python 扩展模块：
- `oge_py/_oge_py.so`   — 1v1 单智能体环境（`OGEInterface`）
- `oge_py/_oge_py_ma.so` — 多智能体环境（`MultiAgentOGEEnv`）

### 2.2 仅重新安装 Python 包（未修改 C++ 时）

```bash
pip install .
```

---

## 3. 项目依赖结构

```
C++ 物理计算层
  src/oge/simcore/         ← rv_from_r0v0, LVLH变换, 太阳位置等
  src/oge/environment/
    orbital_game_environment.cpp  ← 1v1 物理环境
    multi_agent_oge.cpp           ← NvM 多智能体物理环境 (新增, 不影响1v1)
  src/oge/python/
    oge_python_interface.cpp      → 编译为 _oge_py.so
    multi_agent_python_interface.cpp → 编译为 _oge_py_ma.so (新增)

Python 配置层
  configs/env_cfg.py       ← 轨道参数、燃料、时间步等
  configs/ppo_cfg.py       ← PPO 超参数
  configs/multi_agent_cfg.py ← 多智能体场景参数 (新增)

Python 环境层
  modules/env_wrapper.py     ← 1v1 单智能体 skrl Wrapper
  modules/env_wrapper_ma.py  ← NvM 多智能体 skrl Wrapper (新增)

神经网络层
  modules/networks.py        ← 1v1 Policy/Value (256-256 MLP)
  modules/ma_networks.py     ← 多智能体 MAPursuerPolicy/Value (新增)

训练脚本
  scripts/train.py           ← 1v1 训练 (侦察任务 / 操控任务)
  scripts/train_multi_agent.py ← NvM 多智能体训练 (新增)
  scripts/fine_tune.py       ← 1v1 固定初始化微调

评估 / 可视化
  scripts/evaluate.py        ← 1v1 评估
  scripts/visualize_fixed_init.py ← 固定初始化轨迹可视化
```

---

## 4. 1v1 单智能体训练

### 4.1 侦察照射任务（recon）

```bash
cd /home/star/Downloads/oge_2.0/OGE

# 新训练
python scripts/train.py --task recon --name my_recon_v1

# 续训
python scripts/train.py --task recon --name my_recon_v1 \
    --checkpoint runs/my_recon_v1/my_recon_v1/checkpoints/best_agent.pt
```

### 4.2 操控接近任务（operate）

```bash
python scripts/train.py --task operate --name my_operate_v1
```

### 4.3 固定初始化微调（fine-tune）

```bash
python scripts/fine_tune.py \
    --checkpoint runs/March_31th_recon_v3_210_215_newreward_20h/\
March_31th_recon_v3_210_215_newreward_20h/checkpoints/best_agent.pt \
    --timesteps 1000000 \
    --lr 5e-5 \
    --perturb 0.02 \
    --name finetune_fixed_v1
```

### 4.4 评估

```bash
conda run -n oge python scripts/evaluate.py \
    --checkpoint runs/<name>/<name>/checkpoints/best_agent.pt \
    --task recon \
    --episodes 100
```

### 4.5 可视化（固定初始化轨迹）

```bash
conda run -n oge python scripts/visualize_fixed_init.py \
    --checkpoint runs/<name>/<name>/checkpoints/best_agent.pt \
    --output-dir visualizations
```

---

## 5. 多智能体追逃博弈训练

### 5.1 场景说明

| 参数 | 含义 |
|------|------|
| `--evaders E` | 蓝方总数：`blue_sat_0` = HVT（被动），`blue_sat_1`~`blue_sat_{E-1}` = 拦截星（脚本比例导引） |
| `--pursuers P` | 红方追击星数量，所有追击星**共享同一个策略网络**（参数共享 IPPO） |
| `--intercept D` | 拦截距离 km，拦截星进入此范围内即中和对应追击星（默认 30 km） |
| `--threat` | obs 中附加最近拦截星方向信息（evaders > 1 时有效，obs_dim 从 13 变为 17） |

常用场景组合：

| 指令 | 场景 | obs_dim |
|------|------|---------|
| `--evaders 1 --pursuers 4` | 4 追击星 vs 1 HVT（无拦截，纯群体追捕） | 13 |
| `--evaders 2 --pursuers 1` | 1 追击星 vs 1HVT+1拦截（躲避单拦截） | 17 |
| `--evaders 2 --pursuers 2` | 2 追击星 vs 1HVT+1拦截 | 17 |
| `--evaders 4 --pursuers 4` | 4 追击星 vs 1HVT+3拦截（4v4） | 17 |

### 5.2 训练指令

```bash
cd /home/star/Downloads/oge_2.0/OGE

# 4v1：4 追击星 vs 1 HVT（无拦截，群体追捕入门）
python scripts/train_multi_agent.py \
    --evaders 1 --pursuers 4 \
    --timesteps 5000000 \
    --name ma_4v1

# 2v2：2 追击星 vs 1HVT+1拦截
python scripts/train_multi_agent.py \
    --evaders 2 --pursuers 2 \
    --threat \
    --timesteps 8000000 \
    --name ma_2v2

# 4v4：完整追逃博弈
python scripts/train_multi_agent.py \
    --evaders 4 --pursuers 4 \
    --threat \
    --timesteps 10000000 \
    --name ma_4v4

# 续训
python scripts/train_multi_agent.py \
    --evaders 4 --pursuers 4 --threat \
    --checkpoint runs/ma_4v4/ma_4v4/checkpoints/best_agent.pt \
    --name ma_4v4_v2
```

### 5.3 参数共享原理

```
num_pursuers = 4  →  skrl 看到 num_envs = 4 个平行环境
每步：
  obs   : (4, obs_dim)  ← 每个追击星独立观测
  action: (4, 3)        ← 同一网络为 4 个追击星生成动作
  reward: (4, 1)        ← 每个追击星独立奖励
  done  : (4, 1)        ← 每个追击星独立终止

Policy / Value 网络只有一个实例，参数被所有追击星共享。
rollout buffer 大小 = rollouts × num_pursuers（相当于数据增强）。
```

### 5.4 观测向量说明

#### C++ 原始观测（每个智能体，41 维，以 4v1 / N=5 为例）

C++ 环境为每个智能体生成大小相同的原始观测，尺寸 = `13 + 7*(N-1)`（N = 总智能体数）。

```
4v1 场景：N=5，原始 obs_dim = 13 + 7×4 = 41
---------------------------------------------------
[0:3]     自身 r_j2000 (km)
[3:6]     自身 v_j2000 (km/s)

对其余 N-1=4 个智能体，每个占 7 维（按全局编号跳过自身排列）：
  [6:9]   rel_pos(其他_0 → 自身 LVLH) (km)
  [9:12]  rel_vel(其他_0 → 自身 LVLH) (m/s)
  [12]    dist(自身, 其他_0) / 20 km
  [13:20] 其他_1 的 7 维块
  [20:27] 其他_2 的 7 维块
  [27:34] 其他_3 的 7 维块

尾部 7 维（base = 34）：
  [34]    solar_angle(HVT→太阳 ∧ HVT→本追击星) (rad)  ← 每个追击星各自独立
  [35]    dv_remain (km/s)
  [36]    time_progress [0,1]
  [37]    dv_ratio = dv_remain / dv_init
  [38:41] sun_dir in HVT LVLH（单位向量，所有智能体相同）
```

> **注**：`solar_angle` 由 `solar_illumination_angle(太阳位置, HVT位置, 本追击星位置)` 计算，
> 即 HVT→太阳 与 HVT→本追击星 之间的夹角，**每个追击星因自身位置不同而各不相同**。
> 太阳位置由 `JD_EPOCH + current_time/86400` 计算（JD_EPOCH 对应 UTC+8 2027-09-02 00:00:00）。

---

#### Python 侧压缩观测（策略网络实际输入）

Python Wrapper 从原始观测中只保留对 RL 有用的部分，压缩为 **13 维**（4v1）或 **17 维**（有拦截星且开启 threat）。

```
基础 13 维（4v1 场景，或 --evaders 1）：
  [0:3]   rel_pos(追击星→HVT) in 追击星 LVLH / 200 km
  [3:6]   rel_vel(追击星→HVT) in 追击星 LVLH × 10  (m/s→约1)
  [6]     dist(追击星, HVT) / 20 km
  [7]     solar_angle / π                ← 本追击星独立计算，范围 [0,1]
  [8:11]  sun_dir in HVT LVLH（单位向量）← 所有追击星共享同一值
  [11]    dv_ratio（本追击星剩余燃料比）
  [12]    time_progress [0,1]

威胁感知 +4 维（--evaders > 1 且 --threat，自动启用于 2v2 / 4v4）：
  [13]    dist(追击星, 最近拦截星) / 20 km
  [14:17] rel_pos(追击星→最近拦截星) in 追击星 LVLH / 200 km
```

#### 4v1 关键设计说明

| 项目 | 说明 |
|------|------|
| **追击星能看到** | HVT 相对位置/速度、距离、太阳角、太阳方向、自身燃料、时间 |
| **追击星看不到** | 其他 3 个追击星的位置（原始obs中存在但被 Wrapper 丢弃） |
| **solar_angle** | 每个追击星独立：取决于自身位置与 HVT、太阳的几何关系 |
| **合作方式** | 隐式参数共享（IPPO）：共用同一网络权重，但无显式队友通信 |
| **obs_dim** | 13（threat 维在 evaders=1 时自动关闭） |

### 5.5 奖励设计

| 事件 | 奖励 |
|------|------|
| **本追击星**捕获 HVT | `+200 + 剩余燃料比×50` |
| **其他追击星**捕获 HVT（团队奖励） | `+50` |
| 本追击星被拦截 | `-150` |
| 燃料耗尽 | `-40` |
| 每步接近 HVT | `+2 × Δdist(km)` |
| 每步燃料消耗 | `-0.01 × action_ms` |

### 5.6 训练后评估

使用专用评估脚本 `scripts/evaluate_multi_agent.py`：

```bash
# 2v2 场景评估（20 episodes）
python scripts/evaluate_multi_agent.py \
    --checkpoint runs/ma_2v2/ma_2v2/checkpoints/best_agent.pt \
    --evaders 2 --pursuers 2 --episodes 20

# 4v4 场景评估（50 episodes）
python scripts/evaluate_multi_agent.py \
    --checkpoint runs/ma_4v4/ma_4v4/checkpoints/best_agent.pt \
    --evaders 4 --pursuers 4 --episodes 50

# 4v1 纯追捕评估（无拦截，关闭 threat obs）
python scripts/evaluate_multi_agent.py \
    --checkpoint runs/ma_4v1/ma_4v1/checkpoints/best_agent.pt \
    --evaders 1 --pursuers 4 --no-threat --episodes 20
```

**评估指标说明**

| 指标 | 含义 |
|------|------|
| Capture Rate | 至少有一个追击星捕获 HVT 的 episode 比例 |
| Any Intercept Rate | 至少有一个追击星被拦截的 episode 比例 |
| Avg Alive Pursuers | episode 结束时平均存活追击星数 |
| Avg Reward/Pursuer | 所有追击星的平均累积奖励 |
| Avg Steps | 平均 episode 长度 |

**示例输出**

```
Device: cuda
Scenario: 2 pursuers vs 2 evaders (1 HVT + 1 interceptors)

=== Random Init Episodes (20) ===
    Ep     HVT  Alive    AvgRew   Steps  PerPursuerRewards
  --------------------------------------------------------------------------------
     1     CAPT      2    185.43     312  [ 245.2   125.6]
     2     MISS      1     -62.10    480  [-150.0   25.8]
     ...

=== Summary (20 episodes) ===
  Capture Rate        : 75.0%
  Any Intercept Rate  : 30.0%
  Avg Alive Pursuers  : 1.70 / 2
  Avg Reward/Pursuer  : 112.34
  Avg Steps           : 356.2
```

---

## 6. 护卫侦照博弈训练（三星场景）

### 6.1 场景说明

| 角色 | C++ 内部名 | 行为 |
|------|-----------|------|
| 红色高价值星 (Red HV) | `blue_sat_0` | 固定轨道，无机动 |
| 红色护卫星 (Red Escort) | `blue_sat_1` | Phase 1 被动；Phase 2 为 RL |
| 蓝色侦照星 (Blue Recon) | `red_sat_0` | Phase 1 为 RL；Phase 2 加载策略 |

侦照成功条件：`dist(Blue, RedHV) ≤ 20 km` 且 `solar_angle ≤ 60°`，持续累计 ≥ 200 s。

### 6.2 相关文件

| 文件 | 用途 |
|------|------|
| `configs/escort_recon_cfg.py` | 三星场景轨道根数、燃料、时间步等配置 |
| `modules/env_wrapper_escort_recon.py` | 护卫侦照 skrl Wrapper（支持 `train_blue` 切换） |
| `scripts/visualize_escort_recon.py` | 三星初始位置可视化（输入六根数） |
| `scripts/train_blue_recon_phase1.py` | Phase 1：蓝色侦照星 RL，红护卫被动 |

### 6.3 训练流程

**Phase 1：训练蓝色侦照星**

```bash
# 首次训练
python scripts/train_blue_recon_phase1.py --timesteps 10000000 --name blue_recon_phase1

# 续训
python scripts/train_blue_recon_phase1.py \
    --checkpoint runs/blue_recon_phase1/blue_recon_phase1/checkpoints/best_agent.pt \
    --name blue_recon_phase1_v2
```

**Phase 2：训练红色护卫星**（待 Phase 1 完成后）

```bash
# 使用 Phase 1 最优 checkpoint 作为蓝方对手
python scripts/train_red_escort_phase2.py \
    --blue-checkpoint runs/blue_recon_phase1/blue_recon_phase1/checkpoints/best_agent.pt \
    --timesteps 10000000 --name red_escort_phase2
```

### 6.4 可视化

```bash
# 查看三星初始相对位置（默认参数）
python scripts/visualize_escort_recon.py

# 自定义轨道根数
python scripts/visualize_escort_recon.py \
    --hv  42169.502913 0 0.002287 1.592853 0 0.419833 \
    --esc 42169.502913 0 0.002287 1.592853 0 0.421133 \
    --blue 42169.502913 0 0.002287 1.592829 0 0.424435
```

### 6.5 观测向量（17 维，蓝侦照星与红护卫星共用）

```
[0:3]   rel_pos → 目标 / 200 km  (LVLH)
[3:6]   rel_vel → 目标 × 10      (m/s)
[6]     dist_to_target / 20 km
[7]     solar_angle / π
[8:11]  sun_dir in HV LVLH
[11]    dv_ratio
[12]    time_progress
[13]    dist_to_threat / 20 km
[14:17] rel_pos → 威胁 / 200 km  (LVLH)
```

> 蓝侦照星：目标=RedHV，威胁=RedEsc；红护卫星：目标=BlueRecon，威胁=RedHV。

### 6.6 重要提示

C++ 修改（`multi_agent_oge.h/.cpp` 新增运行时 `jd_epoch_`）需重新编译后才能使用：

```bash
cd /home/star/Downloads/oge_2.0/OGE
rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc) && cd .. && pip install .
```

---

## 7. 文件说明

| 文件 | 用途 |
|------|------|
| `configs/env_cfg.py` | 轨道参数、燃料预算、时间步等（1v1 和多智能体共用） |
| `configs/ppo_cfg.py` | PPO 超参数 |
| `configs/multi_agent_cfg.py` | 多智能体场景专用参数（agent 数量、拦截距离等） |
| `modules/env_wrapper.py` | 1v1 skrl Wrapper（侦察/操控任务） |
| `modules/env_wrapper_ma.py` | 多智能体 skrl Wrapper（参数共享 IPPO） |
| `modules/networks.py` | 1v1 Policy/Value 网络 |
| `modules/ma_networks.py` | 多智能体 MAPursuerPolicy/Value 网络 |
| `configs/escort_recon_cfg.py` | 护卫侦照场景配置（轨道根数、燃料、JD epoch） |
| `modules/env_wrapper_escort_recon.py` | 护卫侦照 skrl Wrapper |
| `scripts/visualize_escort_recon.py` | 三星初始位置可视化 |
| `scripts/train_blue_recon_phase1.py` | 护卫侦照 Phase 1 训练（蓝侦照 RL） |
| `scripts/train.py` | 1v1 训练入口 |
| `scripts/train_multi_agent.py` | 多智能体训练入口 |
| `scripts/fine_tune.py` | 固定初始化微调 |
| `scripts/evaluate.py` | 1v1 评估 |
| `scripts/evaluate_multi_agent.py` | 多智能体评估 |
| `scripts/visualize_fixed_init.py` | 固定初始化可视化（PNG + GIF） |
