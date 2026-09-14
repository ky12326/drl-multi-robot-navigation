# 多机器人协同避障 DRL 仿真验证平台

> **Multi-Robot Collision-Avoidance DRL Verification Platform (ROS2 / Gazebo)**
>
> 基于 SAC / TD3 的 N 机器人协同导航训练与评估框架，配套参数化仿真验证环境、
> 固定题目评测体系与纯 Python 测试关卡。

---

## 项目背景

多机器人共享狭窄通道时，「各自走最优路径」会导致对向冲突、排队僵局与死锁。
本项目在 ROS2 + Gazebo 中搭建一套**参数化、可复现、可对比**的仿真验证环境，
用 DRL（SAC / TD3）训练 N 台 TurtleBot3 在瓶颈、十字、中心交汇等场景下协同通过，
并围绕「全局成功率、碰撞率、机间安全距离、死锁率、任务耗时」建立评测指标体系。

平台的核心目标不是「训练一个会走的机器人」，而是**把跑实验变成可复现、可对比、
可定位问题的验证流程**——固定题目、配对 A/B、结果自动归档、缺陷可追溯到具体楼层。

---

## 来源声明与本人工作范围

### 来源

本仓库基于开源项目 **[reiniscimurs/DRL-Robot-Navigation-ROS2](https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2)**
（MIT License, Copyright © 2022 Lari Liuhamo）重构改造。原项目为**单机器人** SAC 导航，
在随机目标点场景下训练避障。本仓库保留了上游的完整提交历史，便于追溯来源。

上游贡献的部分（保留原样或仅做适配）：

- `SAC/SAC_actor.py`、`SAC/SAC_critic.py` — 网络结构与 TanhSquashedNormal 动作分布
- `SAC/SAC.py` — SAC 算法主体（`update()` 的数学逻辑）
- `SAC/SAC_utils.py` — MLP 构建器与权重初始化
- `replay_buffer.py`、`utils.py` — 回放缓冲与位置工具
- `multi_robot_ros_nodes.py` — 由上游 `ros_nodes.py` + `ros_python.py` 改写为多机器人版本

### 本人工作范围

**新增**（本仓库的主体）：

| 模块 | 职责 |
|---|---|
| `multi_robot_env.py` | 统一参数化环境，替代上游的单机器人 env 与早期 6 层继承链 |
| `experiment_registry.py` | 模型路径 / 基线 / 评估模式 / 超参的**唯一来源**（12 个实验条目） |
| `eval_lib.py` | 共享评估库：布局生成、配对多轮评测、指标聚合、结果归档 |
| `multi_robot_train.py` | 参数化训练入口，支持任意 N、帧堆叠、跨实验热启动 |
| `multi_robot_eval_3round.py` | 固定题目 3 轮评估入口 |
| `test_dryrun_training.py` | 无 Gazebo 的训练循环接口预检（10 项 Check，10 秒） |
| `tests/` | pytest 单元测试套件（24 个用例） |
| `launch/`、`worlds/`、`models/` | N 台车参数化 launch、4 个场景世界文件、障碍物 SDF |

**改造**（在上游文件上修改）：

- `SAC/SAC.py` — 增加网络可注入协议、`seq_len` 时序接口、辅助状态扩展点
- `SAC/SAC_utils.py` — 增加跨架构热启动的权重按层分派填充（`pad_input_weights`）
- `replay_buffer.py` — 支持时序帧堆叠

**移除**（上游内容，保留在本地 `源代码文档/` 备份中，不入库）：

- `train.py`（单机器人训练）、`ros_nodes.py` / `ros_python.py`（单机器人节点）
- `pretrain_utils.py`、`lib.py`
- `hardcoded_model.py` + `assets/data.yml`（39 MB 离线数据）— 上游早期硬编码行为基线，与当前 DRL 框架无关

---

## 开发方式说明

**本仓库的代码实现与文档由 AI 编程助手（Claude Code）完成。**
本人负责：研究方向的确定、技术方案的讨论与决策（场景几何、多层次奖励函数、
评测指标体系）、实验矩阵设计与结果验收、缺陷的发现与定位、修复优先级的判断。

仓库中的**缺陷清单、评估数据与决策记录均来自真实运行的调试与实验过程** ——
每一条缺陷都有复现路径与修复前后的对照数据。

之所以显式标注：这些缺陷的定位过程与实验设计是本项目真正的产出，
而它们是可以被追问、也必须被追问的部分。

---

## 系统架构

### 单进程同步调度器

与常见的「每台机器人一个 ROS 节点」不同，本项目由**单个 Python 进程**统一控制所有机器人，
通过显式步进 Gazebo 物理引擎运行：

```
env.reset()  →  /gazebo/reset_world → SetModelStateClient 设置各车姿态 → 轮询稳定性 → 返回 [obs × N]
env.step()   →  发布 N 个 cmd_vel → 取消暂停物理 ~150ms → 轮询传感器 3 次 → 暂停物理 → 返回 [next_obs × N]
```

这样做的收益是**确定性**：物理推进的时长、传感器采样次数、动作生效时机全部显式可控，
避免了多节点异步带来的时序抖动——这对「可复现的配对 A/B 对比」是必需的。

### 三层分工

```
┌──────────────────────────────────────────────────────────────┐
│  experiment_registry.py    配置的唯一来源                     │
│  SEED=42 · BASELINES · EXPERIMENTS(12) · resolve_checkpoints │
└───────────────────────┬──────────────────────────────────────┘
                        │ 被所有脚本读取，不再各自硬编码
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

**关键设计：环境不再继承。** 早期版本用 6 层类继承（`_StandaloneScene1BaseEnv` →
`MultiRobotSyncResetEnv` → Scene2 Stage1 → Stage2 → Stage3b）表达场景差异，导致
「改一处 bug 要在多个子类同步」。现在统一为单个 `MultiRobotEnv` + `SCENE_CONFIGS` 字典，
几何差异由配置驱动，N 台车由参数驱动。

### 状态空间（25 维，可扩展至 29/31 维）

| 索引 | 内容 |
|---|---|
| 0–19 | 20 个分箱 LiDAR 最小值（每箱 9 个样本取最小） |
| 20 | 到目标距离（L2 范数） |
| 21–22 | cos / sin（到目标角度） |
| 23–24 | 上一步线速度 / 角速度指令 |
| 25+ | 可选的邻居相对位姿扩展（`--neighbors`） |

动作空间为 2 维连续 `[线速度, 角速度] ∈ [-1, 1]`，发布前线速度映射到 `[0, 1]`。

### 奖励结构（四分量叠加）

1. **主奖励** — 到达 `+100`，碰撞 `-100`，否则 `线速度 - |角速度|/2 - 障碍物接近惩罚`
2. **Anti-stall** — 连续 12 步无进展后每步 `-0.02`，单 episode 上限 `-0.25`
3. **社会规范** — 机间接近惩罚（1.20 m 内按接近度平方，0.75 m 内额外加罚），前方逼近惩罚，单 episode 上限 `-0.55`
4. **连续偏向冲突** — 小方向性惩罚，鼓励右侧通行惯例

### 固定题目 + 配对 A/B

`eval_lib.generate_or_load_layouts()` 用 **headless 环境**（无需 Gazebo，秒级）
为每个评估模式预采样布局并缓存到 `eval_layouts/`，文件名编码了全部参数
（`{scene}_{N}r_{rounds}x{episodes}_seed{seed}.json`）。

评估时 `run_rounds()` 让两个待比较模型**看到同一批题目的同一顺序**（每轮用布局列表的连续切片），
消除了「随机题目不同导致的伪差异」。布局文件种子固定为 42，可重复生成。

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

结果自动归档到 `eval_results/<exp_id>_<时间戳>.json` + 同名 `.md`，
含配置快照（git commit、seed、布局文件、场景/N/维度）。

---

## 框架缺陷清单

按「现象 → 根因 → 修复」组织。这些是开发过程中真实遇到并修复的问题。

### 1. 动作空间缩放不一致（最隐蔽）

- **现象**：训练 reward 持续下降，critic loss 发散，策略不收敛；跨 2/3 机器人复现
- **根因**：训练循环把**发布用的 cmd_vel**（线速度已映射到 `[0,1]`）存入 replay buffer，
  但 SAC critic 的输入约定是**原始动作**（`[-1,1]`）。即 buffer 里的动作语义与 critic 期望不一致。
  症状表现为「训练发散」，实际是数据契约错误——从 loss 曲线完全看不出问题所在。
- **修复**：在存入 buffer 前使用未缩放的原始动作。修复后 2-robot 与 3-robot 训练均不再发散。

### 2. 三机器人传感器同步（5 个连锁缺陷）

从 2 台扩展到 3 台时集中暴露，**在 2 机器人配置下完全不复现**：

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | robot2 的 odometry 在 `step()` 内不更新，动作恒为 -1.0，全部超时 | `step()` 的总 spin 窗口（0.05s × 3 = 0.15s）不足以触发 robot2 的 odom callback | 对后 N-1 个订阅器补足 spin 配额 |
| 2 | `get_model_pose()` 全返回 None | 订阅端用 `RELIABLE`，Gazebo `libgazebo_ros_state.so` 发布 `/gazebo/model_states` 用 `BEST_EFFORT`，QoS 不兼容 | 订阅端改 `BEST_EFFORT` |
| 3 | 目标到达检测失效 | odom（66 Hz）被 model_state（1 Hz、延迟 ~1s）的旧值覆盖 | odom 优先，model_state 仅作 fallback |
| 4 | 每次 reset 都 "sensor wait timed out" | `_probe_stable_state` 先清空所有订阅缓存再等 callback，而 warmup step 已填充过数据 | 去掉清缓存，probe 复用已有数据（spin 从 2s 降到 0.5s） |
| 5 | r1 对其它机器人「隐身」，撞上也不停 | `_compute_robot_contacts` 只读 model_state，而 robot2 不在其中 | 用 odom 的 `_last_world_position` 补齐 |

**核心教训**：这 5 个问题都是「2 台车时 spin 窗口恰好够用」掩盖的时序假设，
扩到 3 台才暴露。**N 扩展不是加参数，是对既有同步假设的压力测试。**

### 3. 评估基线值漂移（导致过一次错误结论）

- **现象**：`multi_robot_eval_3round.py` 与其余脚本使用了**两组不同的基线值**
  （`0.21/0.58/0.62/0.58` vs `0.13/0.48/0.58/0.40`）
- **根因**：基线值在多个评估脚本里各自硬编码，重构后只有部分脚本更新
- **影响**：曾据此判定「cross/center 模式显著退化（p<0.05）」，
  换用正确基线后复核，实际是 **4/4 模式改善或持平，零退化** —— 一次由工具缺陷造成的假阳性结论
- **修复**：基线上收到 `experiment_registry.BASELINES`，
  并由 `tests/test_registry.py::test_baselines_are_the_new_values` 断言锁死，防止旧值回流

### 4. 布局采样器在 N≥5 时必然崩溃

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | center 采样器 N≥5 必 `IndexError` | 方向池 `["left","right","bottom","top"]` 只有 4 项 | 循环取边 + 分边错位 + 环形目标 |
| 2 | cross 采样器 N≥6 必越界 | 两对 head-on 占满 4 方向后取 `dirs[4]` | 重构为「先配对后 leftover」，配对用镜像车道 |
| 3 | 目标离对向车仅 0.5–0.7 m 被大量拒绝 | cross 采样器目标落点几何错误 | 镜像车道；N=2 场景单次采样从 **10 s → 瞬时、命中率 100%** |
| 4 | 固定布局兜底在 N≠3 时反而崩溃 | 兜底模板硬编码 3 组，`len != n` 抛 `ValueError` | 按 N 过滤 + 新增通用 N 兜底 |

**教训**：「随机采样失败 → 固定布局兜底」这条保命路径，在 N≠3 时自己变成了崩溃路径。
**兜底逻辑必须和主路径接受同样的参数化测试。**

### 5. 维度扩展热启动导致训练崩溃

- **现象**：状态从 25 维 padded 扩展到 29 维后，8 个 epoch 内 `all_goal` 掉到 0；
  critic loss 从 `13` 涨到 `586,888`，alpha 从 `0.1` 涨到 `7.2`，batch reward 持续为负
- **根因**：padded 热启动后首次训练即进入发散螺旋 ——
  新增维度是零填充的 OOD 状态 → 巨大 TD error → 大梯度 → Q 值爆炸 → actor 追逐虚假 Q → α 飙升补偿
- **加重因素**：训练循环丢弃环境返回的渐进 reward、硬编码 ±100 终点奖励（梯度极稀疏）；无梯度裁剪
- **修复**（四项同时）：使用环境返回的渐进 reward；critic 梯度裁剪 `max_norm=1.0`；
  `critic_lr` 从 `1e-4` 降到 `3e-5`；前 5 个 epoch 冻结 actor 只训 critic

### 6. launch 写死机器人数量（静默失败）

- **现象**：N=4/5 时 4/5 号车不被 spawn
- **根因**：launch 文件只声明到 robot3 的位姿参数
- **危害**：**环境不崩溃** —— 等待传感器超时后以空观测继续，
  表现为 4/5 号车用 7.0 的空 scan 值乱撞。没有任何报错，极难排查
- **修复**：`num_robots` 参数化 + 位姿参数扩展到 robot6

### 7. 模型资源重命名后引用未同步（静默失败）

- **现象**：world 文件通过 `<include><uri>model://10by10</uri></include>` 引用场地模型，
  但 Gazebo 启动时**不报错**，只是静默跳过该模型
- **根因**：开发过程中把 `10by10/model.sdf` 重命名为 `wall_model.sdf`（`obstacle1..4` 同理），
  但各目录的 `model.config` 仍声明 `<sdf version="1.6">model.sdf</sdf>`，指向已不存在的文件。
  Gazebo 依据 `model.config` 解析 `model://` URI，找不到目标文件时只记录一条错误后跳过，
  **场景仍然"启动成功"** —— 但场地围栏缺失
- **危害**：`10by10` 是全部 4 个在用场景的公共依赖，缺了它机器人会跑出场地边界；
  而终端没有醒目的失败信息，排查成本极高
- **修复**：把 5 个 `model.config` 的 `<sdf>` 引用改为实际文件名。
  重命名前后内容逐字节一致，因此这是**零行为变化**的修复

**教训**：重命名资源文件时，**引用它的元数据文件是最容易被漏掉的一环**。
这类缺陷的共同特征是「失败是静默的」—— 与本清单第 6 条同源。

### 8. 其它工程问题

- **脚本爆炸**：早期为每个消融实验复制训练脚本（`_baseline` / `_curriculum` / `_shared_buffer` × `25dim` / `29dim`），
  导致 10+ 个近乎相同的文件。后收敛为 `multi_robot_train.py` + registry 模式
- **无预检**：改动训练代码后直接启动 Gazebo，接口错误要等仿真跑起来才暴露。
  后增加 `test_dryrun_training.py`（10 项接口 Check，10 秒，无需 Gazebo）
- **结果不可追溯**：评估结果只打印到终端。后增加自动归档 `eval_results/*.json`，
  含 git commit / seed / 布局文件 / 场景参数快照

---

## 已知限制

以下问题已知但未修复，属于评估后的取舍（如实列出）：

1. **传感器轮询按索引固定配额**（`multi_robot_env.py`）
   现象：对后 N-1 个订阅器补足 spin，robot0 反而没有额外配额
   影响：N≥5 时极端情况下首帧数据可能偏旧；N≤4 实测未触发误判
   不修原因：改为「按数据新旧动态补 spin」需重构轮询逻辑，收益边际，
   而当前固定配额在已验证的 N 范围内稳定

2. **robot2 及之后沿用 robot0 权重**
   `--num-robots 4/5` 时，第 3 台起的机器人加载 robot0 的权重（`resolve_checkpoints` 的约定）。
   即评估的是「同策略多实例」而非异构分工策略，结果应按**零样本同策略**解读。
   `multi_robot_train.py` 支持逐台独立训练，但当前实验矩阵未覆盖。

3. **`--scene scene2` 参数尚未接通**
   `multi_robot_train.py` 的 `--scene` 接受 `scene2`，但 `MultiRobotEnv.SCENE_CONFIGS`
   目前只定义了 `scene0` / `scene1b`，传入 `scene2` 会抛 `ValueError`。
   Scene2（双瓶颈）的实验是在早期环境版本上完成的，尚未迁移到统一环境。

4. **训练分布与评估分布不完全一致**
   训练用课程采样（25% open + 25% center + 20% cross + 20% queue + 10% random），
   评估用固定题目。两者的布局生成器相同，但采样比例不同，
   因此评估指标反映的是**分布外泛化**，不是训练分布内的表现。

5. **URDF 引用了两个不存在的轮胎网格（上游遗留）**
   `turtlebot3_waffle.urdf` 引用了
   `meshes/wheels/{left,right}_tire.stl`，但该目录**从上游版本起就不存在**
   （已核对：`git ls-tree HEAD` 中无此路径，上游仓库亦然）。
   影响**仅限视觉** —— 轮胎的 `<collision>` 是独立的 `<cylinder>`，物理完好，
   Gazebo 会打印一条 `Unable to find file ...` 后正常 spawn。
   启动日志里看到这条警告属正常现象，不代表环境没装好。

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
│   ├── multi_robot_env.py         # 统一参数化环境（几何 / 奖励 / 布局采样 / 校验链）
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
ROS2 Foxy + Gazebo, refactored from the single-robot
[reiniscimurs/DRL-Robot-Navigation-ROS2](https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2) (MIT).

**Key contributions:**

- **Single-process synchronous scheduler** — all robots are driven by one Python process
  that explicitly steps the physics engine, giving deterministic sampling for paired A/B evaluation
- **Unified `MultiRobotEnv`** replacing a 6-level class inheritance chain; geometry differences
  are config-driven, robot count is a parameter
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
defect localization were carried out by the author. See the Chinese section
「框架缺陷清单」 for defects found and fixed during development.

---

## 许可证与致谢

本项目基于 MIT License 发布，保留上游版权声明：

- 原始项目 **DRL-Robot-Navigation-ROS2** — Copyright © 2022 Lari Liuhamo
- 本仓库的改造与扩展 — Copyright © 2026 ky12326

详见 [LICENSE](LICENSE)。仿真环境基于 [TurtleBot3](https://github.com/ROBOTIS-GIT/turtlebot3)
与 [turtlebot3_simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations)（Apache 2.0）。
