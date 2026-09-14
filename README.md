# 多机器人协同避障 DRL 仿真验证平台

> **Multi-Robot Collision-Avoidance DRL Verification Platform (ROS2 / Gazebo)**
>
> 基于 SAC / TD3 的 N 机器人协同导航训练与评估框架，配套参数化仿真验证环境、
> 固定题目评测体系与纯 Python 测试关卡。

---

## 项目背景

多机器人共享狭窄通道时，「各自走最优路径」会导致对向冲突、排队僵局与死锁。
本项目在 ROS2 + Gazebo 中搭建参数化、可复现、可对比的仿真验证环境，
用 DRL（SAC / TD3）训练 N 台 TurtleBot3 在瓶颈、十字、中心交汇等场景下协同通过，
并围绕「全局成功率、碰撞率、机间安全距离、死锁率、任务耗时」建立评测指标体系。

平台的目标不只是训练出能到达目标的策略，更在于把实验过程做成
可复现、可对比、可定位问题的验证流程。

---

## 来源与许可

本项目基于开源项目 [reiniscimurs/DRL-Robot-Navigation-ROS2](https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2)
（MIT License）改造，将原单机器人 SAC 导航扩展为 N 机器人协同训练与评估框架。
上游的完整提交历史保留在本仓库中。

本仓库新增的部分包括：参数化环境 `multi_robot_env.py`、实验注册表
`experiment_registry.py`、共享评估库 `eval_lib.py`、参数化训练与评估入口
（`multi_robot_train.py` / `multi_robot_eval_3round.py`）、pytest 测试套件，
以及多机器人 Gazebo 仿真资产（launch / worlds / models）。

---

## 开发方式

本仓库的代码实现由 AI 编程助手（Claude Code）完成。研究方向的确定、技术方案决策
（场景几何、多层次奖励函数、评测指标体系）、实验矩阵设计、结果验收与缺陷定位
由本人完成。

---

## 系统架构

### 单进程同步调度器

与「每台机器人一个 ROS 节点」的常见做法不同，本项目由**单个 Python 进程**统一控制
所有机器人，通过显式步进 Gazebo 物理引擎运行：

```
env.reset()  →  /gazebo/reset_world → SetModelStateClient 设置各车姿态 → 轮询稳定性 → 返回 [obs × N]
env.step()   →  发布 N 个 cmd_vel → 取消暂停物理 ~150ms → 轮询传感器 3 次 → 暂停物理 → 返回 [next_obs × N]
```

这样做的收益是**确定性**：物理推进的时长、传感器采样次数、动作生效时机全部显式可控，
避免了多节点异步带来的时序抖动 —— 这是「可复现的配对 A/B 对比」的前提。

### 三层分工

```
┌──────────────────────────────────────────────────────────────┐
│  experiment_registry.py    配置的唯一来源                     │
│  SEED=42 · BASELINES · EXPERIMENTS(12) · resolve_checkpoints │
└───────────────────────┬──────────────────────────────────────┘
                        │ 被所有脚本读取，避免各自硬编码
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌───────────────┐ ┌──────────────┐ ┌──────────────────────────┐
│ multi_robot_  │ │ eval_lib.py  │ │ multi_robot_train.py     │
│ env.py        │ │ 布局生成     │ │ 参数化训练入口            │
│ 环境 + 奖励   │ │ 配对多轮评测 │ │ --num-robots / --seq-len │
│ 布局采样      │ │ 指标聚合     │ │ --warmstart / --tag      │
│ 校验链        │ │ 结果归档     │ └──────────────────────────┘
└───────────────┘ └──────────────┘
```

环境由单个 `MultiRobotEnv` 承担，不使用类继承表达场景差异：几何由 `SCENE_CONFIGS`
配置驱动（`scene0` / `scene1b`），机器人数量由 `num_robots` 参数驱动，
支持 N ≥ 2。

### 状态空间与动作

状态 25 维，动作 2 维连续 `[线速度, 角速度] ∈ [-1, 1]`（发布前线速度映射到 `[0, 1]`）。

| 索引 | 内容 |
|---|---|
| 0–19 | 20 个分箱 LiDAR 最小值（每箱 9 个样本取最小） |
| 20 | 到目标距离（L2 范数） |
| 21–22 | cos / sin（到目标角度） |
| 23–24 | 上一步线速度 / 角速度指令 |
| 25+ | 可选的邻居相对位姿扩展（`--neighbors`，可扩展至 29 / 31 维） |

### 奖励结构

四分量叠加：

1. **主奖励** — 到达 `+100`，碰撞 `-100`，否则 `线速度 - |角速度|/2 - 障碍物接近惩罚`
2. **Anti-stall** — 连续 12 步无进展后每步 `-0.02`，单 episode 上限 `-0.25`
3. **社会规范** — 1.20 m 内按接近度平方加罚，0.75 m 内额外加罚，单 episode 上限 `-0.55`
4. **连续偏向冲突** — 小方向性惩罚，鼓励右侧通行惯例

### 固定题目与配对 A/B

`eval_lib.generate_or_load_layouts()` 用 headless 环境（无需 Gazebo，秒级）为每个评估
模式预采样布局并缓存到 `eval_layouts/`，文件名编码全部参数。评估时 `run_rounds()`
让两个待比较模型**看到同一批题目的同一顺序**，消除「随机题目不同导致的伪差异」。

---

## 快速开始

> ⚠️ **以下所有命令都必须在仓库根目录执行。**
> `experiment_registry.MODELS_ROOT` 是相对路径（`src/drl_navigation_ros2/models`），
> 换到别的目录会找不到权重与布局文件。

### 1. 环境要求

- Ubuntu 20.04 · ROS2 Foxy（**desktop 版** —— 需要 `gazebo_ros`、`urdf`、`xacro`）
- Python 3.8 · PyTorch

```bash
source /opt/ros/foxy/setup.bash
source <你的 venv>/bin/activate

# Python 依赖：torch / numpy / squaternion / tensorboard / matplotlib / pytest
pip install -r requirements.txt

colcon build
source install/setup.bash
```

> `colcon build` 只编译仓库内 vendor 的两个 turtlebot3 包
> （`turtlebot3_description`、`turtlebot3_gazebo`），因此**不需要**额外
> `apt install ros-foxy-turtlebot3*`。
>
> `src/drl_navigation_ros2/` **不是 ROS 包**（没有 `package.xml`），`colcon` 会静默跳过它；
> 它通过各脚本内部的 `sys.path` 追加以及 `pytest.ini` 的 `pythonpath` 生效，
> 这也是「必须在仓库根目录执行」的原因之一。

### 2. 测试（无需 Gazebo，也无需权重）

```bash
pytest tests/ -v                                        # 单元测试套件（24 个用例）
python src/drl_navigation_ros2/test_dryrun_training.py  # 训练循环接口预检（10 项 Check）
```

> 修改任何代码后应**先跑这两道关卡**，全部通过再启动 Gazebo 训练。
>
> 两点说明：
> - `tests/test_layouts.py` 的 8 个用例需要 `rclpy`，无 ROS 环境下该模块整体 skip，
>   输出为 `16 passed, 8 skipped`；有 ROS 时才是 24 个全跑。
> - `test_dryrun_training.py` 不依赖 Gazebo、也不依赖任何权重文件。
>   若训练用的 social-v1 权重不在本地，Check 1 会打印警告并**回退冷启动**继续。

### 3. 训练

```bash
# 终端 1 —— 启动仿真（gui:=false 关闭渲染，可显著提速）
ros2 launch turtlebot3_gazebo ros2_drl_multi_robot_sync_3robot.launch.py \
    world_name:=multi_robot_drl_scene1b_bottleneck.world gui:=false

# 终端 2 —— 训练（--tag 决定模型目录名）
python src/drl_navigation_ros2/multi_robot_train.py \
    --num-robots 3 --tag shared_buffer_v1 --epochs 30
```

支持 `--num-robots 2..5+`、`--scene scene0|scene1b`、`--neighbors`（邻居位姿扩展状态）、
`--seq-len N`（帧堆叠）。

可选参数 `--warmstart <registry-exp-id>` 会从 `models/` 读取该实验的权重；
**若权重不在本地会直接报错**，此时去掉该参数从零训练即可。

> ⚠️ 训练脚本会构造**非 headless** 环境连接 Gazebo。
> 请确认终端 1 已经起来，否则脚本会卡在等待 `/gazebo/set_entity_state` 服务的循环里
> （表现为反复打印 `Service not available, waiting again...`），且**不会自行退出**。
>
> ⚠️ 另一处不容易定位的坑：Gazebo 的 `spawn_entity.py` 以 `#!/usr/bin/env python3`
> 运行，用的是 **PATH 中第一个 `python3`**。若你激活的 venv 里没有 `numpy`，
> 三台车会**全部** spawn 失败，日志里只有 `ModuleNotFoundError: No module named 'numpy'`
> 和 `spawn_entity.py: process has died`，看不出和 numpy 的关系。
> 按第 1 步执行 `pip install -r requirements.txt` 即可避免。

### 4. 评估

> ⚠️ **评估同样需要 Gazebo 已运行**，沿用上面的终端 1。

```bash
python src/drl_navigation_ros2/multi_robot_eval_3round.py \
    --exp 25dim_shared_buffer_v1_extended_3r \
    --rounds 3 --episodes 20
```

`--exp` 取 `experiment_registry.EXPERIMENTS` 里的 id，
可先执行 `python src/drl_navigation_ros2/experiment_registry.py` 列出全部条目。

> **注意**：`models/` 权重目录因体积原因未纳入版本控制。
> 请从 [**Releases**](../../releases) 下载对应的权重包并解压到
> `src/drl_navigation_ros2/models/`：
>
> ```bash
> # Part 1 最优模型（Scene1b 3-robot, Shared Buffer 25-dim E30）
> tar -xzf SAC_3r_part1_shared_buffer_e30.tar.gz -C src/drl_navigation_ros2/models/
> ```
>
> 解压后的目录结构必须为 `<模型名>/<机器人序号>/<模型名>_<序号>_actor.pth`，
> 与 `experiment_registry.py` 中该实验的 `checkpoints` 字段一致。
> 未提供权重的机器人按 registry 约定回退到 robot0 的策略。
>
> **目前只发布了 `25dim_shared_buffer_v1_extended_3r`（Part 1 最优）与
> Scene2 Stage4（Part 2 最优）两组权重。**
> 评估其余 registry 条目（`29dim_*`、`4robot`、`5robot` 等）需先用
> `multi_robot_train.py` 自行训练。

---

## 已知限制

- **`--scene scene2` 尚未接通** —— `SCENE_CONFIGS` 目前只定义了 `scene0` / `scene1b`，
  传入 `scene2` 会抛 `ValueError`；Scene2 的实验在早期环境版本上完成，未迁移到统一环境
- **N ≥ 4 时第 3 台起沿用 robot0 权重** —— 评估的是「同策略多实例」而非异构分工策略，
  结果应按零样本同策略解读
- **传感器轮询按索引固定配额** —— N ≥ 5 时极端情况下首帧数据可能偏旧，N ≤ 4 实测稳定
- **训练分布与评估分布不完全一致** —— 训练用课程采样、评估用固定题目，
  因此评估指标反映的是分布外泛化
- **URDF 引用了两个不存在的轮胎网格（上游遗留）** —— 仅影响视觉，
  轮胎碰撞体是独立 cylinder，物理完好

---

## 评估结果

**最优模型**：`25dim_shared_buffer_v1_extended_3r`（Shared Buffer 25 维 E30）
3 轮 × 20 episodes，Scene1b 3 机器人，seed=42，评价指标为 `all_goal_rate`（全部机器人到达目标的 episode 比例）。

| 评估模式 | 最优模型 | 零样本基线 | 提升 |
|---|---|---|---|
| `queue`（排队通过） | **0.92** ± 0.02 | 0.13 | +0.79 |
| `cross`（交叉通行） | **0.95** ± 0.04 | 0.48 | +0.47 |
| `center`（中心交汇） | **0.75** ± 0.07 | 0.58 | +0.17 |
| `random`（随机起止点） | **0.80** ± 0.15 | 0.40 | +0.40 |

> 基线 = 新环境下 **social-v1 2 机器人模型**的零样本表现（未针对 3 机器人微调），
> 用以量化「多机器人协同能力」本身的增益。

**Scene0 泛化**（同一模型，3 轮 × 60 episodes）：

| 模式 | all_goal_rate |
|---|---|
| `parallel` | 0.67 ± 0.10 |
| `cross` | **0.93** ± 0.05 |
| `mixed` | **0.88** ± 0.10 |

**Part 2（Scene2 双瓶颈，早期环境版本）**：`headon` both_goal 0.95（Stage4），
`random_bottleneck` 0.92。模型见 Releases。

结果文件位于 `eval_results/`，每条含完整配置快照。

---

## 目录结构

```
├── src/drl_navigation_ros2/
│   ├── multi_robot_env.py         # 参数化环境（几何 / 奖励 / 布局采样 / 校验链）
│   ├── experiment_registry.py     # 配置唯一来源：SEED / BASELINES / EXPERIMENTS
│   ├── eval_lib.py                # 共享评估库：布局、配对多轮、指标、归档
│   ├── multi_robot_train.py       # 参数化训练入口
│   ├── multi_robot_eval_3round.py # 固定题目评估入口
│   ├── multi_robot_ros_nodes.py   # ROS2 节点：传感器订阅 / 物理暂停 / cmd_vel
│   ├── test_dryrun_training.py    # 无 Gazebo 的训练接口预检（10 项）
│   ├── validate_env_training.py   # 启动前环境校验
│   ├── test_layouts_multi_n.py    # 纯 Python 布局采样关卡
│   ├── SAC/                       # SAC 算法（Actor / Critic / utils / frame_stack）
│   ├── TD3/                       # TD3 算法
│   ├── replay_buffer.py           # 经验回放（支持帧堆叠）
│   ├── eval_layouts/              # 固定题目缓存（确定性 JSON）
│   └── eval_results/              # 评估结果归档（JSON + Markdown）
├── src/turtlebot3_simulations/    # Gazebo 世界 / 障碍物模型 / 参数化 launch
├── tests/                         # pytest 单元测试套件
└── pytest.ini
```

**说明**：`models/`（权重，1.4 GB）、`runs/`（TensorBoard 日志，5.2 GB）
以及开发期的归档脚本与内部文档未纳入版本控制。

---

## English Summary

A parameterized **multi-robot DRL navigation and verification platform** built on
ROS2 Foxy + Gazebo, extended from the single-robot
[reiniscimurs/DRL-Robot-Navigation-ROS2](https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2) (MIT)
to N-robot cooperative training and evaluation.

**Key design points:**

- **Single-process synchronous scheduler** — all robots are driven by one Python process
  that explicitly steps the physics engine, giving deterministic sampling for paired A/B evaluation
- **Single `MultiRobotEnv`** — scenario geometry is config-driven (`SCENE_CONFIGS`),
  robot count is a parameter; no class-inheritance hierarchy
- **Centralized experiment registry** — model paths, baselines, eval modes and hyperparameters
  read from one place; adding an experiment is one registry entry, not a copied script
- **Fixed-question paired evaluation** — layouts are pre-sampled headlessly (no Gazebo, sub-second)
  and cached with deterministic seeds, so two models are compared on identical question sets
- **Two-gate testing** — a pytest unit suite (24 cases) and a 10-check dry-run harness,
  both requiring no Gazebo

**Verified results** (Scene1b, 3 robots, 3 rounds × 20 episodes): queue **0.92**,
cross **0.95**, center **0.75**, random **0.80** all-goal rate, against a 2-robot
zero-shot baseline of 0.13 / 0.48 / 0.58 / 0.40.

**Development note:** the codebase was implemented with an AI coding assistant (Claude Code).
Research direction, technical decisions, experiment design, result validation and
defect localization were carried out by the author.

---

## 许可证与致谢

本项目基于 MIT License 发布，保留上游版权声明：

- 原始项目 **DRL-Robot-Navigation-ROS2** — Copyright © 2022 Lari Liuhamo
- 本仓库的改造与扩展 — Copyright © 2026 ky12326

详见 [LICENSE](LICENSE)。仿真环境基于 [TurtleBot3](https://github.com/ROBOTIS-GIT/turtlebot3)
与 [turtlebot3_simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations)（Apache 2.0）。
