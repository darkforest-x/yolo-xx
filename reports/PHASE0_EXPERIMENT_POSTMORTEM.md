# YOLO-XX / FABLE / DARKFOREST 实验后验报告（Phase 0）

**编制日期**：2026-08-05 · **对应**：V4 架构调整文档 Phase 0
**性质**：只读复盘。未训练、未改模型/标签/数据集/阈值、未写新代码、未做收益回测。
**证据基座**：614 commits（fable-trading）+ 18（yolo-xx）+ 11（darkforest-one）、
187 份 `analysis/` 报告、240 条 `docs/learnings/`、2026-08-05 未来依赖实验。

---

## 摘要（先看这段）

30 天里这条链路被反复验证了三层：**检测能不能认出形态**、**认出的形态有没有 edge**、
**edge 扣成本后还剩多少**。三层的答案分别是：**能（但只在完整上下文里）**、
**有（但边际）**、**不够（约等于成本）**。

V4 把 owner_v10_chain 重新定位为 Pattern Teacher 而非实时触发器，这个判断
**与全部历史证据一致**，且今天的未来依赖实验给了它第一个量化数字：
**tip 复现率 9–10%，完整上下文 62–72%**。

真正被证伪的从来不是「模型能不能学会 owner 的眼睛」——那件事在多个独立口径上都成立。
被证伪的是「**把这只眼睛直接接到盘口上下单**」。V4 的三层拆分正是对这一条的回应。

---

## 1. Commit 历史

| 仓库 | 起止 | commits | 说明 |
|---|---|---|---|
| fable-trading | 2026-07-07 → 08-05 | 614 | 主仓，30 天 |
| darkforest-one | 2026-07-31 → 08-03 | 11 | ETH 15m short-only 因果管线 |
| yolo-xx | 2026-08-03 → 08-05 | 18 | 视觉研究独立出仓 |
| yolo-ma-cluster-trader | (private, 至 07-05) | — | 前身项目，180 版失败，教训写进 README |

fable-trading 提交密度峰值：07-10（97）、08-03（54）、07-15（53）、07-23（41）。
这四个峰值恰好对应四次范式切换：MA206 统一 / L2 重构 / YOLO 主线切换 / 盘口教义落地。

---

## 2. 按阶段：目标 · 假设 · 结果

### 阶段 A（07-07 ~ 07-08）：人工标签里有没有 alpha

- **假设**：owner 手标的「双均线密集启动」含可学习的超额收益。
- **结果**：`p0_alpha_report.md` — **收益端没有 alpha，风险端有真实且显著的 alpha**。
- **判定**：这是整个项目最早、也最被忽视的一个信号。它已经指出「标签擅长的是识别风险
  结构，不是预测收益」，但后续 20 天仍按「预测收益」组织了大量实验。

### 阶段 B（07-07 ~ 07-11）：两层架构立起来

- **2a 检测**：YOLO 冒烟 val mAP50 = 0.835，过 0.8 线。
- **2b 判断**：triple-barrier + LightGBM，holdout AUC 0.59、置换 p=0.002，
  稳定优于单特征基线 → **2b 验收通过**（07-08 holdout 第 1 次消耗，owner 批准）。
- **阶段 3 回测**：157 笔 @0.3% 成本净 +0.06% → **未通过**（PF 1.01）。
- **判定**：**AUC 过线但经济性不过线**，这是全项目第一次出现，后来反复出现。
  沉淀为 CLAUDE.md「把 AUC 当成功标准」的头号警告与
  `high-auc-can-still-lose-economics`。

### 阶段 C（07-08 ~ 07-15）：出场结构与假设扫描

- **目标**：既然入场信号边际，从出场找空间。H1–H19 共 19 个假设。
- **结果**：TP5/SL2 是唯一在 0.3% 成本下净收益明显为正的结构（`p2b_v3_barrier_sweep`）；
  H9（1h EMA144 对齐）+0.051%/trade 过发现级；maker 入场是「大杠杆」；
  时间假设被证伪；成交量三因子（H14/17/18）全部不进入。
- **资产**：TP5/SL2 沿用至今；maker 成本路线。
- **失败沉淀**：`fixed-tp-cuts-short-trend-edge`、`structure-exit-can-beat-fixed-barriers`
  ——固定障碍与趋势出场的张力此时已出现，直到 07-23 holdout#7 才彻底证伪趋势出场。

### 阶段 D（07-10）：MA206 统一

- owner 07-10 推翻 07-09 裁决，要求检测层/判断层/运行路径统一 SMA+EMA 20/60/120。
- **失败沉淀**：`ma-profile-migrations-rotate-the-whole-evidence-chain`
  ——均线口径一改，全部既往证据链失效，这是一次高成本迁移。

### 阶段 E（07-14 ~ 07-18）：检测器 v1 → v11

- **重大 bug**：`ultralytics-auto-lr-destroys-finetune` — `optimizer='auto'` 把所有 chain 模型
  在 epoch 3 打飞，产出的「模型」实为底座 + 1 个 warmup epoch，而其**貌似合理的
  frozen-F1 掩盖了这个问题数月**。
- **修复后**：v8_chain frozen-F1 0.650 → v10_chain 0.645 → v11_chain 0.658。
- **同期发现泄漏**：v5 的 0.663 是泄漏（43/47 eval 币种在其训练集内），
  引出 `owner_eval_frozen`（47 币种从未参训）作为固定尺子。
- **资产**：frozen-eval 制度、⭐ exemplar gate（176 张标杆，`benchmark_check.py`）。
- **失败沉淀**：`checkpoint-selection-must-optimize-the-real-acceptance-gate`。

### 阶段 F（07-18 ~ 07-19）：上实盘，然后撞墙

- 07-18 v11 池切 ACTIVE，accept 回测 703 笔 / 净 +245.8% / PF 6.61 / 胜率 77.1%
  （holdout 第 5 次消耗，owner 批准）。**这是全项目账面最好的数字。**
- 07-19 前向实测：10 笔全部 `closed`、多数 `tp`、净为正，但
  **新鲜检出（lag≤55m）= 0 笔**，延迟 78–768 分钟。
  EDEN 在 VPS 上复现：tip 附近 40 根内根本扫不出，需 tip 前移约 40 根（~10h）才首次命中。
- **判定**：**PF 6.61 与 0 笔可交易信号并存**。这是全项目最关键的一次断裂——
  离线口径与盘口口径第一次被证明测的不是同一件事。
- **沉淀**：`p_forward_hindsight_20260719.md`、
  `freshness-gates-must-be-derived-from-pipeline-arithmetic`。

### 阶段 G（07-20 ~ 07-23）：五次修复尝试，全部失败

| 版本 | 假设 | 结果 | 失败原因（learnings） |
|---|---|---|---|
| v12 H-TIP | 换 H-TIP 训练分布可让 tip 开火 | owner 强制切主线，tip 未解决 | `tip-birth-needs-train-distribution-not-schedule` |
| v13 pad200 | 重渲染窗口对齐 | 错窗 bug，val mAP 不代表 tip | `v13-val-map-is-not-tip-verdict`、`stem-index-is-window-end-not-start` |
| v14 pad200 MAD-on | 修掉错窗 | 错窗≈0 但 tip 仍失败 | `mad-on-pad200-still-fails-tip-smoke`、`pad200-train-fire-not-live-tip` |
| v15 tipval | tip 验证集 | **正负样本来自两条渲染管线 = 风格捷径** | `pos-neg-must-share-one-render-pipeline` |
| v16 tipuni 冷启动 | 统一渲染管线冷启动 | 空背景误火 17/33 = 51.5%，不上线 | 后经 owner 目视核实：**标签比模型错**，那 33 张是规则自动预标 |

- **07-23 v16 holdout 终审（第 6 次消耗）**：纯检测亏损，且
  **判断层反预测——判断分越高越亏（top5% PF 0.48）**。
  根因：v11 判断在「事后」候选上训练，拿到盘口就反向。
  → `hindsight-trained-judgment-is-anti-predictive-at-the-tip`。
- **07-23 owner 立纪律 12**：检测只认盘口；pre-v16 权重清除；detector=none 诚实空转。
- **同日归因**：`p_chain_failure_attribution.md`、
  `p_samesource_judgment_verdict.md` —— walk-forward 证伪「稳健 edge」：
  「双均线密集启动」在实时盘口、扣成本、TP5/SL2 结构下**没有稳健可交易 edge**。
- **holdout #7（空边趋势出场）**：train 过线的两档在 holdout 全塌到 ~1.0 → 证伪。

### 阶段 H（07-23 ~ 07-28）：为什么没有 edge

- `p_base_rate_dense_verdict`：密集几何**信号真实但边际，成本才是杀手**。
- `p_launch_entry_base_rate` / `_long_short`：机械启动入场抬 PF 但**过不了 1.3**。
- **07-28 对照组终判**：100×6m 池 +16.9bp 中 **+7.2bp 是做空 beta**，
  检测器自身只值 **+9.0bp**，而往返成本 **10bp**。
  → `pool-internal-metrics-cannot-see-beta`（写进 CLAUDE.md 弱模型警告）。
- **金标可交易性审计**：499 个 ⭐ 标杆里**只有 2 个画在盘口**，中位可见未来 **97 根**。
  → `zero-live-edge-labels-means-the-target-is-unverified`。
- **判定**：这是 Phase 0 中最硬的一组结论。它说明问题不在模型容量、不在调度、
  不在阈值，而在**监督目标本身不是一个盘口可执行的对象**。

### 阶段 I（07-29 ~ 07-30）：小周期支线

- ETH 3m pilot v1：严格 OOS 774 根 eligible bars 中开火 **772 根（99.74%）**，
  没有形成稀疏事件 → 不通过。→ `recall-without-fire-rate-rewards-a-detector-that-fires-everywhere`。
- ETH 3m v2 图像分类诊断：静态 val 第一门 FAIL。
- v10 对 ETH 3m 的 owner 标注：93/200 为「是」（46.5%）——
  **v10 确实能找到目标形态**，但框的横向中位跨度 36 分钟，
  到开火时已从框内最高收盘下跌中位 **4.47 个 3m ATR**，93/93 在信号端位于六条均线下方。
  → 这是「Pattern Teacher 成立、实时触发器不成立」的最早直接证据，早于今天的实验。

### 阶段 J（08-03）：L2 重构与三次只读审计

- P0 Runtime Parity **REJECTED**：`models/ACTIVE` 不是 07-30 研究优胜配置，
  研究结论不得转移 → `a-research-result-cannot-be-assigned-to-a-different-runtime-artifact`。
- P1-DATA accepted：从冻结的 pre-holdout L1 proposal ledger 重建 immutable dataset（18,103 行）。
- P2-L2 **REJECTED**：主模型 `best_iteration=1`、1 棵树、15 个不同分数，
  calibration q90 实际放行 85.51%，模型/selector health 全失败。
- P2-R 根因审计：fold-local exact-top 加权 **−15.91bp**，同期整池 **−15.33bp**，
  **相对整池仅 −0.59bp** → ranking 没有增量，调阈值救不回来。
- P2-M 机制审计：raw return IC 大部分含 ATR/barrier 尺度成分
  → `atr-scaled-barriers-entangle-outcome-and-return-magnitude`。
- **资产**：层间契约（`yoyo/layers/` 四层禁止互相 import，AST 强制）。

### 阶段 K（08-05，今天）：未来依赖量化

- 100 样本 × 34 币种 × 6 档 future，窗口/渲染/均线/conf/iou 全冻结。

| future bars | v10_chain | v12_htip |
|---|---|---|
| 0（盘口） | **10%** | **9%** |
| 20（5h） | 39% | 41% |
| 40（10h） | 48% | 51% |
| 99（完整） | 62% | 72% |

- confidence 几乎不随 future 变化（0.45–0.55）→ 失效模式是**要么认出要么完全看不见**。
- **H-TIP 重训没有减轻未来依赖**：v12_htip 归一化后 12.5%，v10_chain 16.1%。
- 证实 V4 文档判断：**tip 裁剪不是可事后施加的变换**，判据本身长在被裁掉的部分上。

---

## 3. 成功资产（可继承，不要重造）

| # | 资产 | 证据 | V4 中的位置 |
|---|---|---|---|
| 1 | **owner 的判别力是真实的** | yolo-xx：1313 个手标框 val 期做空 PF 8.98、胜率 79%，同语境随机进场只赢 32% | 整个体系的地基 |
| 2 | **人工视觉可被模型学习** | v10 在 ETH 3m 上 93/200 owner 认可；今天完整视角 62–72% | Layer 1 Pattern Teacher |
| 3 | **owner_v10_chain 权重本体** | 今天从 3060 取回，18.3MB | Pattern Teacher 起点 |
| 4 | **61 个历史权重 + 25 套 args/results** | `models/archive_3060/`（1.1G），v7→v16 全系 | 可做版本演进对照 |
| 5 | **golden_pool 12567 框** | `data/golden_pool.json`，含 round9 | Pattern Dataset 种子 |
| 6 | **frozen-eval 尺子 + ⭐ 基准** | `datasets/owner_eval_frozen`（47 币 464 图）、176 张标杆 | Layer 1 评价 |
| 7 | **实验纪律体系** | holdout 记账（已 7 次）、匹配随机对照组、置换检验、预注册 | 全阶段 |
| 8 | **240 条 learnings** | 每条一个已付学费的坑 | 全阶段 |
| 9 | **数据/渲染/回测基础设施** | 456 币 15m、`render_chart`、tip-replay 回测器、层间契约 | 全阶段 |
| 10 | **P0 风险端 alpha 结论** | 人工标签在风险端有真实显著 alpha | Layer 2 Pattern Quality 的正确目标方向 |

---

## 4. 失败原因（按根因归类，不按时间）

### 4.1 根因一：监督目标不是盘口可执行对象（**主因**）

- 499 个 ⭐ 标杆只有 2 个画在盘口，中位可见未来 97 根。
- 训练图 = 过去 + 形成 + **后续启动**；实盘 = 过去 + 当前。
- 今天的量化：去掉右侧，复现率从 62–72% 掉到 9–10%，且 confidence 不降。
- 这一条独立解释了 v12→v16 五次修复的全部失败：**它们都在改模型和渲染，
  没有改监督目标的时间语义**。

### 4.2 根因二：用错裁判

- val mAP / frozen-F1 / 自家 AUC 全都在「有后文」的分布上量，量不到盘口。
- v11 三道门（曲线健康 / F1 0.658 / ⭐ 0.974–0.917）全过，实盘 0 笔新鲜。
- 沉淀为纪律 12：**自家 val/mAP/旧 frozen-F1 永不作裁决**。
- 相关：`v13-val-map-is-not-tip-verdict`、`detector-eval-rulers-are-not-live-universe-authorities`。

### 4.3 根因三：成本吃掉全部边际

- 检测器自身 +9.0bp，往返成本 10bp。
- 机械启动入场抬 PF 但过不了 1.3；趋势出场 train 过线、holdout 塌回 1.0。
- 相关：`edge-vs-cost-tolerance`、`gross-edge-must-be-separated-from-cost-and-generalization`。

### 4.4 根因四：beta 混入池内指标

- 100×6m 池 +16.9bp 中 +7.2bp 是做空 beta。池内口径与置换检验都看不见它。
- 沉淀：每张方向性结果表必须带**匹配随机对照组**。

### 4.5 根因五：收益标签污染视觉任务（V4 明确要停止的）

- L2 用盈亏标签训练，学到的是「赚钱形态」不是「高质量形态」。
- 极端表现：07-23 判断层在盘口上**反预测**（top5% PF 0.48）。
- 相关：`the-edge-is-in-magnitude-so-a-classifier-learns-nothing`、
  `owner-label-oracle-alpha-is-not-causal-tip-alpha`。

### 4.6 工程类失败（不影响结论方向，但消耗了大量时间）

`ultralytics-auto-lr-destroys-finetune`（数月的假成绩）、
`pos-neg-must-share-one-render-pipeline`（v15 风格捷径）、
`stem-index-is-window-end-not-start`（v13 错窗）、
泄漏（v5 的 0.663）、
`weights-live-only-where-they-were-trained`（v11 永久丢失）、
`purge-records-are-claims-not-facts`（清除记录与事实分叉）。

---

## 5. 对 V4 架构的复盘意见（仅供 owner 判断，不改任务定义）

1. **V4 的三层拆分与证据一致。** 检测（能）、质量（未验）、交易（不成立）三者
   在历史中被混在一个目标里训练，五次修复失败可完全由此解释。
2. **Layer 3 Formation Model 的前置窗口方案（T0-T30/50/70），历史上没有被试过。**
   已试并失败的是「完整形态直接裁到右侧」（v13/v14/v15/v16）。
   今天的 ablation 曲线可直接作为它的基线：**future=0 的 9–10% 就是不做任何改造的天花板**。
3. **Layer 2 换成 pattern_quality 而非盈亏标签**，与阶段 A 的 P0 结论
   （风险端有 alpha、收益端没有）方向一致——这是 30 天前就拿到、但一直没被采纳的信号。
4. **建议 Phase 1 补一个对照**：本次实验无法分离「依赖后续启动」（C）与
   「依赖形态后半段」（B）。按信号后 20 bar 走势分层重跑即可分离，成本低。
   这个区分直接决定 Formation Model 的标签定义应该是「未来是否形成经典形态」
   还是「未来是否启动」。

---

## 6. 诚实声明

- 本报告是文献综合，不是重新实验。除 08-05 未来依赖实验外，全部数字引自既有报告原文。
- 187 份报告中约 40 份未被逐字读取（读了 INDEX 的标题与结论摘录）。
- v11 的实测数据永久缺失（权重不可恢复），阶段 F 的结论依赖当时的 forward_log 与 VPS 复现记录。
- darkforest-one（11 commits）与 yolo-ma-cluster-trader（private 前身）未做代码级复盘，
  只做了范围登记。若需要，应单独一轮。
- 本轮未消耗 holdout。累计 holdout 消耗记录仍为 7 次（①07-08 2b ②07-15 回归 ③07-16 v8
  ④07-17 v10 ⑤07-18 v11 ⑥07-23 v16 ⑦07-23 A 空边趋势）。
