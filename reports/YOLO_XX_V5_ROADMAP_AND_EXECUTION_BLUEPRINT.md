# YOLO-XX V5 Roadmap & Execution Blueprint

**版本**：V5 · **日期**：2026-08-05 · **上游**：V4 架构调整文档 + `PHASE0_EXPERIMENT_POSTMORTEM.md`
**适用仓库**：`darkforest-x/yolo-xx`（视觉）· `darkforest-x/fable-trading`（判断）· `darkforest-x/darkforest-one`（执行）

---

## 0. 这份文档与 V4 的关系

V4 定了**方向**（三层拆分、v10 重定位为 Pattern Teacher、停止 tip 裁剪）。
V5 把方向变成**可执行、可验收、可失败**的蓝图：每个 Phase 有输入、输出、
数据格式、验收门、**以及失败退出条件**。

最后一项是这个项目最缺的东西。历史上多次出现「某阶段结论是否定的，但工作继续沿着
原方向推进」——v12→v16 五次修复即是（详见 Postmortem 阶段 G）。V5 的每个 Phase
都必须能判死。

**命名**：系统整体称 **Pattern Intelligence System (PIS)**，不再称「YOLO 模型」。
YOLO 只是 Layer 1 的当前实现。

---

## 1. Phase 0 结论：不可推翻的约束

以下六条来自 Phase 0，**作为 V5 的公理，任何 Phase 不得与之冲突**。
如需推翻，必须提交新证据并由 owner 批准。

| # | 约束 | 证据 |
|---|---|---|
| **C1** | **owner 的判别力是真实的。** 模型能学会人工视觉形态。 | 1313 手标框 val 期做空 PF 8.98 / 胜率 79%，同语境随机只赢 32%；v10 在 ETH 3m 上 owner 认可 93/200 |
| **C2** | **owner_v10_chain 不是实时交易触发器。** | tip 复现率 10%，完整上下文 62%（2026-08-05 实验，n=100/34 币） |
| **C3** | **FULL context 检测能力显著高于 TIP context，且 confidence 不随之下降。** 失效模式是「要么认出、要么完全看不见」。 | conf 0.449→0.497 全程平稳；框/图 0.22→0.73 |
| **C4** | **TIP 裁剪 / H-TIP 重训不能解决监督目标错位。** | v12_htip 专为此训，tip 9% vs 未修复 v10 的 10%；归一化后反而更低（12.5% vs 16.1%）；v13/v14/v15/v16 四次失败 |
| **C5** | **检测、质量、交易必须拆分。** 混训导致标签污染与反预测。 | 07-23 判断层在盘口 top5% PF 0.48（反预测） |
| **C6** | **净 edge ≈ 成本。** 检测器自身 +9.0bp，往返成本 10bp；池内 +16.9bp 中 +7.2bp 是做空 beta。 | 07-28 匹配对照组终判 |

**第一原则（建议写入三仓 README）**：

> 被证伪的不是模型能不能学会 owner 的眼睛，而是把这只眼睛直接接到盘口上下单。

---

## 2. Pattern Intelligence System 总架构

```
                          人工交易经验
                                |
                                v
                    ┌───────────────────────┐
          Layer 1   │  owner_v10_chain      │
                    │  Pattern Teacher      │
                    └───────────┬───────────┘
                                |
                                v
                    ┌───────────────────────┐
          Layer 1.5 │   Pattern Library     │  ← 资产沉淀层（V5 新增）
                    └───────────┬───────────┘
                                |
                +---------------+---------------+
                |                               |
                v                               v
     ┌──────────────────┐            ┌────────────────────┐
 L2  │ Pattern Quality  │        L3  │  Formation Model   │
     │ 这个形态有多经典  │            │  未来会不会形成    │
     └────────┬─────────┘            └─────────┬──────────┘
              |                                |
              |                      ┌─────────v──────────┐
              |                  L4  │   Launch Model     │
              |                      │  形成后会不会启动  │
              |                      └─────────┬──────────┘
              +---------------+----------------+
                              |
                              v
                   ┌──────────────────────┐
               L5  │  Trade Decision      │   fable-trading
                   │  quality × formation │
                   │  × launch → score    │
                   └──────────┬───────────┘
                              |
                              v
               ┌──────────────────────────┐
           L6  │       Execution          │   darkforest-one
               └──────────────────────────┘
```

**为什么 Formation 与 Launch 必须分开**（采纳 GPT-5.6 建议并给出证据支撑）：

- 合成一个「未来是否启动」的标签，等于把交易结果重新注入视觉任务 → 触犯 C5。
- 合成一个「未来是否形成形态」的标签，则永远不知道形成之后值不值得做 → 触犯 C6。
- 分开后两者可独立验收：Formation 用形态学裁判（人工/teacher），Launch 用价格裁判。
  **只有 Launch 允许接触收益类信息，且它的输出不得回流到 Layer 1/1.5/2/3。**

---

## 3. Pattern Library 规格（Layer 1.5）

### 3.1 为什么必须有

当前最大的资产不是权重，而是「模型认为哪些东西像 owner 的眼睛」。
权重会过期、会被删（v11 即为教训），Library 不会。

### 3.2 数据格式

```json
{
  "pattern_id": "dense_00001",
  "source": "golden_pool | v10_prediction | owner_review",
  "symbol": "ARM_USDT_SWAP",
  "timeframe": "15m",
  "signal_time": "2026-08-02T03:00:00+00:00",
  "image_path": "datasets/.../ARM_USDT_SWAP_009107.png",
  "window": {"bars": 200, "start_i": 8908, "end_i": 9107},
  "bbox_xywhn": [0.956, 0.62, 0.07, 0.18],
  "confidence": 0.439,

  "human_label": "A|B|C|null",       // 人工质量分级，未审为 null
  "human_reviewed_at": null,

  "ma_structure": {"spread_atr": 0.31, "slope_20": -0.002, "duration_bars": 9},
  "embedding": null,                 // Phase 1 暂不填，Phase 2 需要时补
  "teacher_version": "owner_v10_chain",
  "teacher_sha256": "<权重指纹>"
}
```

### 3.3 种子池（已在手）

| 来源 | 数量 | 位置 |
|---|---|---|
| golden_pool | 12567 stem / 6229 框 / 6630 纯背景 | `data/golden_pool.json` |
| frozen-eval 尺子 | 47 币 / 464 图 | `datasets/owner_eval_frozen` |
| ⭐ exemplar 基准 | 176 张 | `data/benchmark_exemplars.json` |
| v10 在 ETH 3m 的 owner 审 | 200 张（93 为「是」） | `analysis/p_eth_3m_v10_owner_labels_timing.md` |
| yolo-xx owner 手标 | 1313 框 | yolo-xx 仓 |
| 2026-08-05 ablation 样本 | 100 样本 × 6 arm | `reports/*_future_dependency_curve.json` |

---

## 4. 各 Phase 规格

### Phase 0 — 实验复盘 ✅ 已完成

产出：`reports/PHASE0_EXPERIMENT_POSTMORTEM.md` / `.html`

---

### Phase 1 — Pattern Teacher Research

**唯一目标**：搞清楚 owner_v10_chain 到底学到了什么，并把它的能力边界冻结成文档。
**不训练任何模型。**

#### Task 1.1 — 资产清单

输出：`reports/pattern_teacher_asset_inventory.md`

必含：权重路径与 **SHA256**、训练配置（`model/data/epochs/imgsz/lr0/augment` 全量）、
训练数据集来源与规模、标签来源与标注协议、render 配置、window 配置、
frozen-eval 定义、已知污染（如泄漏、lr bug 影响范围）。

**已知起点**：`models/owner_v10_chain.pt`（18.3MB，2026-07-17 02:30，取自 3060）；
`model: base_v9.pt` / `data: dense_owner_v9` / imgsz 960 / lr0 1e-4；
同源备份 61 个权重在 `models/archive_3060/`。

#### Task 1.2 — Pattern Library v1（候选版）

输出：`reports/pattern_library_candidate.json`（格式见 §3.2）

规则：**不修改任何原始数据**；
未经人工审的 `human_label` 必须为 `null`，**不得用规则自动预标充数**
（2026-07-23 教训：自动标签被当成金标，导致 v16 被误判为 51.5% 误火）。

#### Task 1.3 — Teacher 解剖（B/C 分离实验）

**这是 Phase 1 的核心，它决定 Phase 3/4 的标签定义。**

假设：owner_v10_chain 的检出依赖信号右侧信息（C2/C3 已证），
但依赖的究竟是 **形态的后半段**（pattern structure）还是 **启动确认**（launch confirmation）？

三类样本（按信号后 20 bar 的实际走势分层，**分层规则必须在跑之前预注册**）：

| 类 | 定义 | 若 teacher 能检出，说明 |
|---|---|---|
| A | 形态完整 + **未启动**（后 20 bar 未突破） | 它识别 pattern structure |
| B | 形态完整 + **已启动** | — |
| C | 无经典密集，仅后续走势漂亮 | 它识别的是走势不是形态（最坏情况） |

裁决规则（预注册）：

- A 类检出率 ≥ B 类检出率的 70% → **pattern structure 主导**
- A 类检出率 < B 类的 40% → **launch confirmation 主导**
- 中间区间 → 混合，需扩样本或引入人工审

输出：`reports/pattern_teacher_analysis.md`

**复用而非重造**：「启动」的定义已在 `p_launch_entry_base_rate.md` /
`p_launch_entry_long_short.md` 中使用过，直接沿用，不要新造一个。

#### Phase 1 验收门

- [ ] 三份报告齐备，`pattern_library_candidate.json` 可被程序读取且 schema 校验通过
- [ ] Task 1.3 的分层规则在跑之前落盘（预注册文件带时间戳）
- [ ] 未训练、未改权重/标签/数据集/阈值、未消耗 holdout

#### Phase 1 失败退出条件

若 Task 1.3 判定为 **C 类主导**（teacher 主要识别的是后续走势而非形态），
则 C1 被削弱，**Pattern Library 的种子有效性存疑**，必须停下并重新审视标注协议，
**不得进入 Phase 2**。

---

### Phase 2 — Pattern Quality Model

**目标**：给形态打质量分（「这个形态有多接近经典人工形态」），**不是打赚钱分**。

- 输入：YOLO embedding、bbox 几何、confidence、MA spread / slope、duration、
  volatility、trend
- 输出：`quality_score ∈ [0,1]`
- 标签来源：**人工分级 A/B/C**（Pattern Library 的 `human_label`），**禁止使用盈亏**
- 依据：Phase 0 阶段 A 的 P0 结论——人工标签在**风险端**有真实显著 alpha、
  收益端没有。这是项目第一天就拿到、但 30 天未被采纳的信号。

**验收门**：与人工分级的一致性（Cohen's κ 或分级准确率），在**从未参与训练的
币种**上评估。**不看 PF、不看收益。**

**失败退出**：若 quality_score 与人工分级一致性不显著优于随机，
说明「经典程度」不可从当前特征学习，须回 Phase 1 补标注协议。

---

### Phase 3 — Formation Model

**目标**：提前发现——当前状态是否将在未来形成经典形态。

- 输入：**前置窗口**（完整形态 T0–T100 之前的 T0-T30 / T0-T50 / T0-T70）
- 输出：`formation_probability`
- 标签：**未来 N 根内是否形成经典形态**（形态学裁判，非价格裁判）

**已证伪的做法（禁止重试）**：把完整形态图直接裁到右侧（v13/v14/v15/v16，见 C4）。

**基线（已知）**：2026-08-05 ablation 的 `future=0` 复现率
**9–10% 就是不做任何改造直接上盘口的天花板**。Formation Model 必须显著高于它才有意义。

**验收门（必含，历史教训）**：

- [ ] 严格时间切分，禁止随机切分
- [ ] 正负样本**必须来自同一条渲染管线**
      → `pos-neg-must-share-one-render-pipeline`（v15 死因：风格捷径）

---

### Phase 4 — Launch Model

**目标**：形态形成之后，是否会启动。

- 输出：`launch_probability`
- **这是唯一允许接触价格/收益类信息的视觉侧模型。**
- **硬约束**：其输出**不得回流**到 Layer 1 / 1.5 / 2 / 3 的任何训练或标注环节。
  违反即触犯 C5。

**验收门**：必须带**匹配随机对照组**（同币 × 同时间块 × 同波动桶），
只报置换检验不够——它验排序，抓不到整池踩在 beta 上
（07-28 教训：+16.9bp 中 +7.2bp 是做空 beta）。

---

### Phase 5 — Trade Decision（fable-trading）

```
trade_score = f(quality_score, formation_probability, launch_probability)
```

**验收门（沿用现有铁律，不放宽）**：

- top-decile 扣 0.2% 往返成本后净收益为正，且置换检验 p < 0.01
- 匹配随机对照组
- 确认级只认**前向新鲜样本**（不是 val、不是 accept 回测）
- holdout 消耗需 owner 逐次批准并记账（**当前已消耗 7 次**）

---

### Phase 6 — Execution（darkforest-one）

沿用现有实盘纪律 7–11：新鲜度三门同值、脉冲预算 <15min、VPS 唯一写者、
不自动 promote、真金操作只有 owner 亲手做。

---

## 5. 跨 Phase 铁律（从 240 条 learnings 提炼，违反即返工）

1. **不用 val mAP / 自家 frozen-F1 作裁决**（纪律 12）。它们量的是有后文的分布。
2. **每张方向性结果表必须带匹配随机对照组**（07-28）。
3. **正负样本同一条渲染管线**（v15）。
4. **人工标注不得用规则自动预标充数**（v16 误判）。
5. **holdout 每次消耗需 owner 批准 + 报告记账**。
6. **一次实验只改一个变量**；多变量打包需 owner 批准。
7. **训练产物立刻异地备份**（v11 永久丢失的教训：
   `weights-live-only-where-they-were-trained`）。
8. **清除/迁移类操作的记录不等于事实，须逐机核查**
   （`purge-records-are-claims-not-facts`）。

---

## 6. Agent 工作规则

1. 不改变任务定义；不自行升级 Phase。
2. 不让检测器承担交易任务；不用收益污染视觉标签。
3. 实验必须**先写假设与预注册**，再跑。
4. 修改必须说明原因与验证方式。
5. **仓库边界**：Phase 1–4 在 `yolo-xx` 产出；
   **`fable-trading` 与 `darkforest-one` 在 Phase 1–4 期间为只读**
   —— 可以读取其数据、渲染代码、历史报告，**不得修改其代码或产物**。
6. 报告口径（yolo-xx CLAUDE.md）：**给数字，不下判决；数字不许粉饰；
   样本内外分开报；小样本标注笔数。**
7. 每个非平凡问题解决后写 `docs/learnings/` 笔记。

---

## 7. 当前资产登记（2026-08-05 实测）

| 类别 | 数量 | 位置 |
|---|---|---|
| 自训权重 | 61 | `fable-trading/models/archive_3060/`（1.1G，v7→v16 全系） |
| 训练配置/曲线 | 25 套 | 同上（args.yaml + results.csv） |
| Pattern Teacher | 1 | `models/owner_v10_chain.pt`（18.3MB） |
| golden_pool | 12567 stem / 6229 框 | `data/golden_pool.json` |
| frozen-eval | 47 币 / 464 图 | `datasets/owner_eval_frozen` |
| ⭐ 基准 | 176 张 | `data/benchmark_exemplars.json` |
| 历史报告 | 187 | `analysis/` |
| learnings | 242 | `docs/learnings/` |
| **永久丢失** | **owner_v11_chain** | 唯一不可恢复的模型 |

---

## 8. 与 GPT-5.6 建议的差异（供 owner 对照）

本蓝图采纳了 GPT-5.6 的：三层拆分、Pattern Library 中间层、Formation/Launch 分离、
Phase 1 先解剖 teacher、PIS 命名。以下四处基于本仓实测证据做了调整：

| # | GPT 建议 | V5 调整 | 依据 |
|---|---|---|---|
| 1 | Phase 编号两处不一致（Launch 分别列为 Phase 4 / 无） | 统一为 0–6，Launch = Phase 4 | — |
| 2 | 「不要修改 fable-trading」 | 细化为**只读**：可读数据/渲染/报告，不可写 | Pattern Library 必须复用 `render_chart` 与 K 线，否则渲染语义漂移 |

另补充 GPT 未涉及的：每个 Phase 的**失败退出条件**、跨 Phase 铁律 8 条、
holdout 预算现状（已用 7 次）。

**Owner 决定（2026-08-05）**：曾提议为 Library 增加 `visible_future_bars` 必填字段、
为 Phase 3 增加 fire rate 强制验收项；owner 决定两项均按 GPT 原案执行，不加入。
本文档已按该决定移除相关条目。
