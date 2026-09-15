# orca_rl — PPO baseline for OrcaHand in-hand reorientation

放到 `orca_sim/` 仓库根目录下（和 `src/`、`random_policy.py` 平级），就能直接跑。

```
orca_sim/
├── src/orca_sim/        # 原仓库
├── random_policy.py
└── orca_rl/             # 这个包
    ├── task.py          # 修好的任务：奖励 + 成功判据 + 动作空间
    ├── checks.py        # 证明奖励漏洞已修复
    ├── train.py         # PPO 训练
    ├── evaluate.py      # 诚实的评估
    └── record.py        # 录视频
```

---

## 装依赖

```bash
source ../orca/bin/activate        # 你的虚拟环境
uv pip install stable-baselines3 "imageio[ffmpeg]" tensorboard   # zsh 需要给 [] 加引号
```

`tensorboard` 是可选的 —— 没装的话训练照跑，只是没有曲线。

---

## 第一件事：先跑 checks

**在训练任何东西之前跑这个。**它用四个脚本化的策略对比原任务和修好的任务，把奖励漏洞直接摆出来：

```bash
python -m orca_rl.checks
```

实测输出：

```
STOCK TASK  (orca_sim.OrcaHandRightCubeOrientation)
  hover just outside tolerance for 200 steps :    196.1
  actually solve in 20 steps, then terminate :      9.8
  --> stalling is 20x better. PPO will stall.

FIXED TASK  (orca_rl.task.CubeReorientContinuous)
  policy                          return   solves   steps
  -------------------------------------------------------
  do nothing                        0.06        0     400
  random                           -1.75        0     400
  oracle (teleport to goal)       251.99       40     400

  all checks passed
```

「do nothing」那一行从 **196 掉到 0.06**，这就是修复的全部意义。

---

## 训练

```bash
python -m orca_rl.train --name run1 --timesteps 20_000_000 --n-envs 8
```

**接着已有模型练下去**（不要浪费已经跑出来的策略）：

```bash
python -m orca_rl.train --name run3 --resume runs/run2/final_model.zip \
       --timesteps 20_000_000 --n-envs 8
```

`--resume` 会一并载入同目录的 `vecnormalize.pkl`。**这一点很重要** —— 观测归一化的统计量如果从零重新累积，等于给策略喂了尺度不同的观测，表现会像是「突然失忆」。步数计数也接着走，TensorBoard 上是连续的一条线。

超参数按命令行重新读取，所以 `--resume` 的同时改 `--lr` 之类是生效的。注意课程难度存在环境里、不随模型保存，续训会从 `--curriculum-start-deg`（默认 30°）重新开始。

`--n-envs` 设成你 Mac 的性能核心数。M1/M2 Pro 一般是 8，M3 Max 可以到 12。开太多反而会因为调度开销变慢，先跑几分钟看 `fps` 再定。

产物全在 `runs/run1/`：`final_model.zip`、`vecnormalize.pkl`（评估时必须要）、`checkpoints/`、`tb/`。

```bash
tensorboard --logdir runs/
```

**要看的不是 `ep_rew_mean`，是 `task/` 那一组：**

| 指标 | 含义 | 期望 |
|---|---|---|
| `task/drop_rate` | 掉落率 | 最先下降。几十万步内应该从 ~90% 掉到 20% 以下 |
| `task/solves_per_episode` | 每 episode 完成几次翻转 | **真正的进度指标。**长时间是 0 就是没学会 |
| `task/solved_any_frac` | 至少完成一次的 episode 占比 | 从 0 开始爬升 |
| `train/clip_fraction` | 更新幅度 | 健康值 0.05–0.2 |
| `train/approx_kl` | 策略移动距离 | 应该被 `target_kl=0.02` 压住 |
| `train/std` | 动作标准差 | 应在 0.6 附近缓慢下降。**涨上去说明熵失控**，掉到接近 0 说明探索死了 |
| `task/goal_angle_deg` | 当前课程难度 | 从 30° 起步，应该缓慢上升 |

学习通常分两个阶段：**先学会不掉方块**（drop_rate 快速下降，但 solves 还是 0），**然后才学会转**（solves 开始爬）。第一阶段一般几十万步，第二阶段慢得多。如果 500 万步后 solves 还是 0，别干等，去调参或者看视频。

---

## 评估

```bash
# 基线，不需要模型
python -m orca_rl.evaluate --policy zero
python -m orca_rl.evaluate --policy random

# 你的模型（会自动找同目录的 vecnormalize.pkl）
python -m orca_rl.evaluate --model runs/run1/final_model.zip --episodes 50
```

**每一次 success 都要求方块与手有接触、高于 0.15 米、并且连续保持 10 步对齐。**原仓库的判据下随机策略有 21% 成功率（方块被弹飞、半空翻滚时蒙中一帧），那些在这里一个都不算。

永远和两个基线一起看。**打不过 `zero` 的策略，学到的只是「别把方块弄掉」。**

---

## 录视频

```bash
python -m orca_rl.record --model runs/run1/final_model.zip --out run1.mp4
python -m orca_rl.record --policy random --out random.mp4
```

用普通 `python` 就行，离屏渲染不需要 `mjpython`。

**曲线动了就录一段看。**这个任务的数字非常容易骗人 —— 策略可以一边刷出漂亮的 return 一边做着明显荒唐的事。十秒视频省几小时。

---

## 改了什么

### 1. 奖励不再奖励「不完成任务」

原来：每步 `0.5 * (alignment + 1)`，成功即终止。悬停在阈值外 200 步 ≈ 196 分，20 步真解 ≈ 10 分 —— PPO 必然学会悬停。

现在：

```
r = shaping_coef * Δalignment          # 势函数式，只有「进步」得分，等待得 0
  + success_bonus     (完成时 +10)
  - drop_penalty      (掉落时 -5)
  - action_rate_penalty * ||Δtarget||²  # 动作平滑，对真机很重要
```

**成功不再终止 episode** —— 完成后奖励 +10 并换一个新目标，继续赚。做任务严格优于不做任务。

`Δalignment` 只在方块被握住时才计入，否则半空翻滚也能白拿 shaping 分。

### 2. 成功判据要求真的握住

必须同时满足：与手有接触（查 MuJoCo contact）、高于 `in_hand_height=0.15`、连续 `hold_steps=10` 步对齐。

`drop_height` 从 0.05 提到 0.10 —— 原来的 0.05 太低，方块早就离开手掌了才算掉。

### 3. 目标从简单的开始，按能力自动加难度

默认 `goal_mode="curriculum"`：目标是从当前朝向绕随机垂直轴转 θ 度，θ 从 30° 起步，上限 180°。

课程的升降级判据有两个关键设计（都是踩坑踩出来的，见文末复盘二）：

- **按速率不按次数** —— `successes × (400 / 实际步数)`。用绝对次数的话，一个 90 步就掉落的 episode 物理上装不下两次成功，会被误判成「太难」，于是掉落率一高课程就被系统性往下拽。
- **20 个 episode 的滑动窗口** —— 这个任务的 solves 方差极大（均值 0.84、最高 4、近一半是 0）。逐 episode 反应会让难度做随机游走，来回抵消。窗口平均 ≥1.0 升 5°、≤0.25 降 5°，调整后清空窗口在新难度上重新测量。

`task/goal_angle_deg` 和 `task/curriculum_solve_rate` 都会记进 TensorBoard。用 `--goal-mode axis` 可以切回固定的 6 个轴向目标，此时新目标必须与当前朝向相差至少 60°。不加这一条的话，6 个轴向目标里约 1/6 在 reset 时就已经满足，而且刚完成一次后重采样很容易抽到刚解完的那个 —— 白送成功。**这个 bug 我在实测中抓到过**：修之前随机策略每 40 个 episode 有 5 个拿到成功，修之后降到 1 个。

### 4. 动作空间统一成 `[-1, 1]^17`

原来是 17 个不同范围的关节角（有的跨度 1.05，有的 2.18，而且不对称），不归一化基本训不动。现在环境内部负责映射，**不需要 `RescaleAction` 包装器**。

默认 `action_mode="relative"`：动作是在上一个目标上的增量（`action_scale=0.15` 表示每步最多走半程的 15%）。比绝对位置好训，而且轨迹平滑得多 —— 如果这些策略以后要上真机，平滑性直接关系到腱和电机的寿命。想用绝对模式传 `--action-mode absolute`。

### 5. 域随机化（默认关闭）

```bash
python -m orca_rl.train --randomize-physics
```

每次 reset 随机方块质量（±30%）、摩擦（±30%）、关节阻尼（±30%）、执行器增益（±20%），并给关节角观测加约 0.3° 的噪声。

**先不要开。**先在确定性环境里证明任务能学会，再打开做 sim-to-real。一上来就开会让本来就难的探索问题雪上加霜。

---

## 超参数

默认值都写在 `train.py` 的 `parse_args()` 里。两个偏离 SB3 默认、但对这个任务很重要的：

- **`--ent-coef 0.0`**。直觉上 17 维连续动作需要熵来探索，但在这个任务上**熵项会失控** —— 详见文末的事故复盘。不要调高。`--max-log-std 0.0` 会额外硬 clamp σ ≤ 1 兜底。
- **`--log-std-init -0.5`**（SB3 默认 0，即 std=1）。在 `[-1,1]` 的动作空间里 std=1 意味着动作几乎总是撞到边界，每步都是最大增量，轨迹极度抖动。实测把它改成 -0.5 后 `clip_fraction` 从 0.57 降到 0.16，`approx_kl` 从 0.16 降到 0.03，全部回到健康区间。

还加了 `--target-kl 0.02`，让 SB3 在策略移动过远时提前结束 epoch 循环。

---

## 已知的坑

- **`SubprocVecEnv` 在 macOS 上必须有 `if __name__ == "__main__"` 保护** —— `train.py` 里已经有了。如果你自己写脚本调用，别忘了。调试时可以加 `--no-subproc` 退回单进程，报错栈会清楚很多。
- **评估时忘了加载 `vecnormalize.pkl`** 是最常见的「训练好好的、评估一塌糊涂」的原因。`evaluate.py` 和 `record.py` 会自动找同目录下的这个文件。
- **`checks.py` 里的 oracle 会瞬移方块**，那不是一个真实策略，只是奖励上限的参照。
- `record.py` 和 `evaluate.py` 用了 `env._cube_body_id` 这类私有属性 —— 上游没提供公开接口，先这么用。

---

## 接下来

1. **跑通、看到 solves 开始上涨**，确认任务可学
2. 打开 `--randomize-physics`，看性能掉多少 —— 这是 sim-to-real 的第一个真实信号
3. **非对称 actor-critic** —— 现在的 54 维观测里有方块位姿，真机上拿不到。critic 保留它，actor 只用手的本体感知（前 17 + 17 维）加历史。这一步做完，策略才有可能上真机
4. 如果 wall-clock 成为瓶颈，考虑移植到 MJX 用 GPU 跑几千个并行环境


---

## 事故复盘：v1 跑了 2000 万步，零进展

第一版（`success_bonus=5 / drop_penalty=10 / ent_coef=0.005 / goal_mode="axis"`）在 M4 上跑满 2000 万步，`task/solves_per_episode` 全程在 0.01–0.055 之间抖动，**没有任何趋势**。三个独立的错误叠在一起，每一个单独都足以让训练失败。

### 错误 1：熵项失控，σ 从 0.6 涨到 11

`train/std` 单调指数上升到 **11**。动作空间是 `[-1,1]`，std=11 意味着每个采样动作都饱和在边界 —— 训练时收集的全是 bang-bang 噪声，策略在自己均值附近的探索完全消失。

原因：高斯熵对 σ **没有上界**。正常情况下任务奖励会提供反向梯度把 σ 拉回来，但这个任务从没成功过，奖励梯度近乎为零，于是损失里唯一稳定的梯度就是熵项，PPO 老老实实把 σ 推向无穷。

`evaluate.py` 用 `deterministic=True`（只取均值 μ）所以看起来正常 —— **均值动作和实际训练行为已经完全脱节**。这是这个 bug 特别阴险的地方。

修复：`ent_coef` 默认改成 **0**，另加 `ClampLogStdCallback` 每轮硬 clamp `log_std <= 0`（σ ≤ 1）。

### 错误 2：掉落惩罚大于成功奖励，策略学会「夹住不动」

看视频才发现的：训练出来的策略把方块**卡死在手指和掌心之间，然后完全静止**，从第 4 秒到第 8 秒画面几乎不变。

原因很简单 —— `drop_penalty=10` 比 `success_bonus=5` 大一倍。手内操作必然伴随掉落风险，当「掉一次」的代价是「成功一次」的两倍、而成功还不确定时，**冻结是理性选择**。更糟的是夹死之后方块根本转不动，策略把自己锁死在了任务之外。

修复：`success_bonus` 5 → **10**，`drop_penalty` 10 → **5**。

### 错误 3：任务从第一步就是满难度

6 个轴向目标 + 至少偏离 60° 的约束 = **每次都是 90° 或 180° 的翻转**。这是 Dactyl 级难度，指望随机策略偶然完成一次来提供第一个学习信号，概率太低 —— 2000 万步实测 0 次。

修复：`goal_mode="curriculum"`（新默认）。目标是从当前朝向绕随机垂直轴转 θ 度，θ 从 **30° 起步**，按最近成功率自动调整（每 episode 解 ≥2 次就 +5°，≤0.5 次就 −5°），上限 180°。`task/goal_angle_deg` 会记进 TensorBoard，可以直接看课程进展。

### 修完之后

同一台容器（2 核，700 fps），**25 万步**：

| 指标 | v1 @ 2000 万步 | v2 @ 25 万步 |
|---|---|---|
| `solves_per_episode` | 0.01–0.055（无趋势） | **0.59** |
| `ep_rew_mean` | ~0.2 | **+4.9** |
| `train/std` | 0.6 → 11（失控） | 0.588（稳定，缓慢下降） |
| `drop_rate` | 0.04 | 0.14 |

**80 倍的步数差距，结果反过来。**顺带一提：修复后随机策略在 30° 课程下也能拿到 0.28 solves/episode，这正是「任务现在会发出学习信号」的直接证据 —— 之前是 0。

`drop_rate` 从 0.04 涨到 0.14 不是退步，是好事：策略终于愿意冒险动方块了，而不是夹死不动。

### 教训

三个 bug 里，**只有第一个能从曲线上看出来**（std 那条）。第二个只有看视频才能发现，第三个要靠算「这个任务到底有多难」。

`ep_rew_mean`、`explained_variance`、`approx_kl` 在 v1 里全部是健康的 —— explained_variance 甚至到了 0.95。**曲线全绿不代表在学正确的东西。**


---

## 复盘二：课程卡在下限，2000 万步没升过级

修完前面三个 bug 之后，run2 的结果是**真的学会了**：

| | run1 | run2 |
|---|---|---|
| solves / episode | 0.02 | **0.84**（最高 4） |
| ≥1 次的 episode | 2% | **52%** |
| mean return | −0.08 | **+6.23** |
| 首次成功耗时 | 从没有过 | **68 步**（1.4 秒） |
| 握住比例 | 90% | 97% |
| drop rate | 4% | 34% |

视频里的逐帧运动量也印证了行为的转变：

| | 运动量中位数 |
|---|---|
| run1（夹住不动） | 0.071 |
| run2（主动操作） | **2.051** |
| random | 4.928 |

**但 `task/goal_angle_deg` 在 2000 万步里一直在 26–27° 震荡，一次都没升上去。**下限是 25°，也就是说它被压在地板上。

原因是课程判据本身有两个错：

1. **用绝对成功次数** —— 一个 90 步就掉落的 episode 装不下 2 次成功，必然被判「太难」。掉落率一高，课程就被系统性地往下拽，**难度和 episode 长度被错误耦合**。
2. **每个 episode 都调整** —— solves 方差极大（近一半 episode 是 0），逐次反应等于随机游走：一次 0 降 5°，一次 2 升 5°，来回抵消。

修完之后用模拟数据验证判据的响应（200 个 episode）：

| 输入水平 | 课程终点 |
|---|---|
| 每 episode 解 2.0 次 | 30° → **80°** |
| run2 的水平（0.84 次） | 30° → **75°** |
| 从来解不出 | 30° → **25°**（正确地退到下限） |

而且它是自平衡的：难度上去之后成功率会掉，最终稳定在「速率≈1.0」的那个难度上。

### 教训

**自动课程本身也是一个需要调试的系统。**它有自己的失败模式，而且和策略的失败模式长得不一样 —— 策略明明在进步（0.02 → 0.84），课程却一动不动。如果只盯着 `solves_per_episode`，会以为一切正常；只有把课程难度也画出来，才能看到「策略在原地打转」这件事。
