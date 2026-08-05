# Agent Prompt — Phase 1 Task 001（Pattern Teacher Research）

> 直接复制以下全部内容给执行 Agent。

---

你现在负责 `darkforest-x/yolo-xx`。

必须先阅读（按顺序）：

1. `reports/YOLO_XX_V5_ROADMAP_AND_EXECUTION_BLUEPRINT.md`
2. YOLO-XX V4 Architecture Adjustment
3. `reports/PHASE0_EXPERIMENT_POSTMORTEM.md`

## 当前状态

Phase 0 已完成。以下六条是**不可推翻的约束**，你的任何产出不得与之冲突：

- **C1** owner 的判别力真实存在，模型能学会人工视觉形态。
- **C2** owner_v10_chain 不是实时交易触发器（tip 复现率 10%，完整上下文 62%，n=100/34 币）。
- **C3** FULL context 检测能力显著高于 TIP，且 confidence 不随之下降——
  失效模式是「要么认出、要么完全看不见」。
- **C4** TIP 裁剪 / H-TIP 重训不能解决监督目标错位（v12/v13/v14/v15/v16 五次失败）。
- **C5** 检测、质量、交易必须拆分。
- **C6** 净 edge ≈ 成本（检测器 +9.0bp vs 往返 10bp）。

因此：**不要重新设计架构。不要训练模型。不要进入交易优化。**

## 当前执行：Phase 1 — Pattern Teacher Research

**唯一目标**：明确 owner_v10_chain 到底学到了什么，并把能力边界冻结成文档。

---

### Task 1 — 资产清单

输出：`reports/pattern_teacher_asset_inventory.md`

必含：权重路径 + **SHA256**、训练配置全量（model / data / epochs / imgsz / lr0 /
全部 augment 项）、训练数据集来源与规模、标签来源与标注协议、render 配置、
window 配置、frozen-eval 定义、**已知污染**（泄漏、lr bug 的影响范围）。

已知起点：`fable-trading/models/owner_v10_chain.pt`（18.3MB，2026-07-17 02:30，取自 3060）；
`model: base_v9.pt` / `data: dense_owner_v9` / imgsz 960 / lr0 1e-4；
同源 61 个历史权重在 `fable-trading/models/archive_3060/`。

---

### Task 2 — Pattern Library v1（候选版）

输出：`reports/pattern_library_candidate.json`

字段规格见 V5 蓝图 §3.2。**以下两条是硬要求**：

1. **不修改任何原始数据。** golden_pool / owner 标注 / v10 预测结果均只读。
2. **未经人工审的 `human_label` 必须为 `null`。**
   **禁止用规则自动预标充数**——2026-07-23 教训：自动标签被当成金标，
   导致 v16 被误判为「51.5% 误火」，owner 逐图核实后发现是标签错不是模型错。

种子池：`fable-trading/data/golden_pool.json`（12567 stem / 6229 框 / 6630 纯背景）、
`datasets/owner_eval_frozen`（47 币 464 图）、`data/benchmark_exemplars.json`（176 张）、
yolo-xx 的 1313 个 owner 手标框。

---

### Task 3 — Teacher 解剖（决定 Phase 3/4 标签定义）

**假设**：owner_v10_chain 依赖信号右侧信息（C2/C3 已证），
但依赖的是**形态后半段**（pattern structure）还是**启动确认**（launch confirmation）？

按信号后 20 bar 的实际走势分三层：

| 类 | 定义 |
|---|---|
| A | 形态完整 + **未启动**（后 20 bar 未突破） |
| B | 形态完整 + **已启动** |
| C | 无经典密集，仅后续走势漂亮 |

**分层规则与裁决阈值必须在跑之前预注册落盘（带时间戳），不得看到结果再定。**

预注册裁决规则：

- A 类检出率 ≥ B 类的 70% → pattern structure 主导
- A 类检出率 < B 类的 40% → launch confirmation 主导
- 中间 → 混合，需扩样本或引入人工审

**「启动」的定义直接沿用 `fable-trading/analysis/p_launch_entry_base_rate.md` 与
`p_launch_entry_long_short.md` 中已用过的口径，不要新造一个。**

输出：`reports/pattern_teacher_analysis.md`

---

## 仓库边界

- 产出全部写入 **yolo-xx**。
- **`fable-trading` 与 `darkforest-one` 在本阶段为只读**：
  可以读取其 K 线数据、`render_chart` 渲染代码、历史报告、权重，
  **不得修改其任何代码或产物**。
- 复用 `fable-trading/src/detection/render.py::render_chart`，
  不要另写渲染——渲染语义一旦漂移，与历史结果就不可比。

## 禁止

修改模型 / 训练模型 / 调整阈值 / 添加 LR / 添加交易标签 / 做收益回测 /
做执行系统 / 修改 fable-trading / 修改 darkforest-one / 消耗 holdout。

## 报告口径

给数字，不下判决。数字不许粉饰。样本内外分开报，小样本标注笔数。
发现测量方法有问题必须直接指出。

## 失败退出条件

若 Task 3 判定为 **C 类主导**（teacher 主要识别后续走势而非形态），
则 C1 被削弱、Pattern Library 种子有效性存疑 —— **立即停止并报告，
不得进入 Phase 2**。

## 完成后

停止。不要进入 Formation Model，不要进入 Pattern Quality，等待下一阶段指令。
