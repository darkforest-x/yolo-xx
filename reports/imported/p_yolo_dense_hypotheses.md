# YOLO 均线密集检测层假设簇（H-DET）— 发现级汇总

日期：2026-07-22（外源 EXT 簇同日补录）  
授权：owner「参考 RESEARCH_AGENDA，针对 YOLO 检测均线密集提出假设并研究测试」  
纪律：发现级 · **未**耗 holdout · **未** promote · **未**打断 `owner_v13_pad200` 训练

议程正文：[`docs/RESEARCH_AGENDA_DETECT.md`](../docs/RESEARCH_AGENDA_DETECT.md)  
外源调研：[`analysis/p_yolo_external_sources.md`](p_yolo_external_sources.md)  
主议程指针：[`docs/RESEARCH_AGENDA.md`](../docs/RESEARCH_AGENDA.md)

## 结论先行

**调度/阈值不是解药；几何训练分布才是。** tip-only 与 TIP_CONF 已证伪抬 tip_fire。  
外源调研（2026-07-22）**独立印证**同一方向：右缘锚定、禁事后烛、流式口径、截断紧框——见 H-DET-EXT-\*；**没有**换公开权重的捷径。

| 状态 | 假设 |
|---|---|
| 🔴 已证伪 | H-DET-5 tip 窗降 conf；H-DET-6 tip-only 调度；H-DET-8 当 tip 解药的 A′ |
| 🟢 已证实（发现级） | H-DET-3 右缘 N 验收口径；H-DET-7 离线 tip_hit≠实盘 tip_fire；H-DET-8 A′ 止血事后账；H-DET-EXT-3 流式口径登记 |
| 🔵/🟡 等 v13 | **H-DET-1 pad200**（训练中；终局评测脚本已备） |
| 🟡 外源已离线审计 | H-DET-EXT-1/2/4/7（框宽≈11–12 bar、v13 train 右缘≥0.95 占 96%；见 `analysis/output/tip_box_geometry_vs_lit.json`） |
| 🟡 清单已备、未训 | **H-DET-2** 硬负（`analysis/output/hardneg_mid_cluster/`，v13 后再训） |
| 🟡/⚪ 可排期 | H-DET-4 / EXT-5 渲染；EXT-6/8（tip 后） |

## 下一步唯一推荐（与 v13 不冲突）

**仍是等 v13 终局 → `bash scripts/eval_v13_vs_v12_tip.sh`（H-DET-1）。**

- 外源便宜小测（标签几何对照文献）**已完成**，不另插 GPU 实验。  
- 勿用 mid-run `best.pt` 抢 MPS；勿用 v13 val mAP 冒充 tip（val 标签与 v11 相同，未 pad）。  
- 若 tip-smoke 仍≈0：再开 H-DET-4；H-DET-2 用已备清单立项开训（需批准）。

## 1. 假设表（人话）

| # | 人话 | 设计 | 判定 | 状态 |
|---|---|---|---|---|
| H-DET-1 | pad200（框后无后文）训出的 v13 比 v12 更能 tip 贴边开火 | `dense_owner_v13_pad200` finetune from v12 | tip-smoke 贴边开火 ≫ v12 的 0；true_tip 不崩 F1 | 🔵 训中；🟡 等终局对照 |
| H-DET-2 | 有后文的中段簇作硬负 → 抑制事后框 | 只加 hard-neg，其它不变 | 中段框↓ 且 tip 召回不塌 | 🟡 清单 2892 已备；未训 |
| H-DET-3 | 验收只评右缘 N 根有框，不只 mAP | tip_hit / tip_edge / tip_subset strict | 每份检测报告必报 | 🟢 |
| H-DET-4 | MA 线宽/颜色/留白影响 tip | 固定权重极小消融 | 开火率相对基线可测变化 | 🟡 协议已写 |
| H-DET-5 | tip 窗单独 conf 阈值抬 tip_fire | TIP_CONF=0.22 vs 0.30 | tip-smoke / tip_fire↑ | 🔴 |
| H-DET-6 | tip-only 调度抬 tip_fire | MODE=tip vs live | tip_fire↑ | 🔴 |
| H-DET-7 | true_tip tip_hit ≠ 盘口 tip_fire | v12 0.925 vs smoke 0/27 | 鸿沟存在 → 训分布问题 | 🟢 |
| H-DET-8 | A′ 贴边门止血事后账，不造 tip | TIP_EDGE_BARS=2 | 拒 KORU 类；tip_fire 仍可为 0 | 🟢止血 / 🔴解药 |
| H-DET-EXT-1…8 | 外源：右缘锚定 / 禁事后烛 / 流式口径 / 截断框 / MA+安全增广 / 框→2b / 窗长 / 单向时序 | 见 `RESEARCH_AGENDA_DETECT.md` | 各条分列发现/确认门槛 | 见专表；几何审计已做 |

## 2. 已测结论（登记入库）

### H-DET-5 / H-DET-6 — tip-smoke（v12 主线）

来源：`analysis/p_tip_only_smoke.md` + `analysis/output/diag_tip_smoke.json`

| 条件 | 强制 tip 扫描（27 币） | 账本 tip_fire（32 行） |
|---|---|---|
| live@0.30 | **0/27** | 1/32 |
| tip@0.30 + TIP_CONF=0.22 | **0/27** | 1/32 |

→ 降 conf、换 tip-only **都不抬** 贴边开火率。

### H-DET-7 / H-DET-3 — 离线高 tip_hit vs 实盘

| 指标 | 数字 | 来源 |
|---|---|---|
| v12 true_tip tip_hit | **0.925** (111/120) | `p_v12_htip_eval` |
| tip-smoke 开火 | **0/27** | diag_tip_smoke |
| tip_subset strict 折扣 | **0.0465** | `p_tip_subset_val` |

### H-DET-8 — box→bar + A′

映射往返正确；语义错位。A′ 挡事后账，不产生 tip。

### H-DET-EXT — 外源离线几何（2026-07-22，无 GPU）

| 集 | 右缘≥0.95 | 宽 p50 |
|---|---|---|
| v11 train | 2.8% | ~11 bar |
| v13_pad200 train | **96%** | ~12 bar |
| v13 val（=v11 val） | 2.1% | ~11 bar |

→ 外源「5–16 bar / 右缘锚定」与 pad200 训分布一致；**确认仍靠 tip-smoke**。详 `p_yolo_external_sources.md`。

### H-DET-1 — v13 现状

| 项 | 值 |
|---|---|
| 进程 | `python -m src.detection.train ... --name owner_v13_pad200`（勿杀） |
| 稳定权重 | **尚无** `models/owner_v13_pad200.pt` |
| 本轮动作 | 外源文档 + 标签几何；H-DET-2 硬负清单 + tip-smoke 评测包加固（`p_yolo_while_v13_trains.md`）；**不**抢 MPS 对照 |

## 3. 复现 / 训完命令

```bash
bash scripts/eval_v13_vs_v12_tip.sh   # 等稳定权重
# 几何审计复现（CPU）：
PYTHONPATH=. .venv/bin/python -c "print('see analysis/p_yolo_external_sources.md §4')"
```

渲染消融（H-DET-4）：见 `docs/RESEARCH_AGENDA_DETECT.md`。

## 4. 解读

1. 内源死路已关：降 conf、tip-only。  
2. 外源没有捷径权重；协议层与 pad200 **同向**。  
3. pad200 仍是当前唯一在训的分布对齐实验。  
4. val 未 pad → 禁止用 val mAP 讲 tip 故事。

## 5. 风险与诚实声明

- 外源调研不替代 tip-smoke；公开金融 YOLO 任务错配。  
- mid-run 权重不作 H-DET-1 终审。  
- 发现级赢 ≠ promote；确认级 = 前向新鲜 100。

## 6. 下一步（唯一推荐）

**H-DET-1 终局对照** → 写 `analysis/p_v13_pad200_train.md`。

| 若 tip-smoke 开火明显 >0 | owner 目视小样 → 再议影子（需批准） |
|---|---|
| 若仍≈0 | H-DET-4；H-DET-2 按已备清单开训（须批准）；EXT-7 改窗长须批准 |
| 不做 | 自动 promote、ChartScan 权重、StreamYOLO 整栈、再降 TIP_CONF |
