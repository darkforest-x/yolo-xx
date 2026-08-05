# v13 pad200 终局 + H-DET-1 tip 对照 — 2026-07-22

**纪律**：未 promote `owner_best` / ACTIVE / frozen；未评 holdout；未清 forward_log。

## 复现命令

```bash
bash scripts/v13_train_status.sh
# train DEAD；stable 由 best.pt 拷贝（pipeline 未自动落盘时手工）：
cp -f runs/detect/runs/detect/owner_v13_pad200/weights/best.pt models/owner_v13_pad200.pt
bash scripts/eval_v13_vs_v12_tip.sh
```

## 训练终局

| 项 | 值 |
|---|---|
| 数据 | `dense_owner_v13_pad200`（train=pad200 tip 几何；**val=v11 中段金标未 pad**） |
| 底座 | `models/owner_v12_htip.pt` |
| 计划 | 40 ep，patience=10，MPS |
| 实际 | **ep32 early stop**（~20.0h）；best=**ep22** → `best.pt` |
| 稳定权重 | `models/owner_v13_pad200.pt`（= best.pt，sha 一致） |

Owner 贴的 `P≈0.110 R≈0.051 mAP50≈0.027 mAP50-95≈0.010 fitness≈0.010 nt=1587`
= **best ep22 的官方 val 表**（与 `results.csv` / 终局 `results_dict` 对齐），**不是** tip-smoke。

## 同口径 val mAP（辅表；不可当 tip 裁决）

| 权重 | best ep | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| v12_htip | 22 | 0.513 | 0.638 | **0.534** | **0.284** |
| v13_pad200 | 22 | 0.109 | 0.051 | **0.027** | **0.010** |

**差多少**：mAP50 约 **20×** 崩（0.534→0.027）。  
**是否可比**：**数字同 Ultralytics val 协议，语义不可比 tip**——v13 val 标签与 v11 val 逐文件相同（中段金标），train 却是右缘贴窗末的 pad200；分布错位会系统性压垮 val mAP（H-DET-3 / EXT-3）。v12 train 含 tip 克隆但仍保留大量中段正样本，故同 val 上 mAP 仍可读。**禁止**用这张表单独判 tip 失败或成功。

## H-DET-1 发现级（主表）

| 指标 | v12 (`owner_best`) | v13 pad200 | 判定 |
|---|---:|---:|---|
| true_tip tip_hit（val 重渲 tip 几何，conf=0.3，n=120） | **0.925** (111/120) | **0.0083** (1/120) | 崩 |
| tip-smoke 贴边开火 | **0/27** | **0/27** | **未过线**（须 ≫ v12） |

产物：`analysis/output/tip_rate_v13_pad200.json`、`analysis/output/diag_tip_smoke_v13.json`。

## 解读

1. **val mAP 烂 ≠ 已证明 tip 废**——预期内的 train/val 几何错位；单独引用属指标作弊。  
2. **但 tip 口径也烂**：true_tip 从 0.925→0.008，tip-smoke 仍 0/27。H-DET-1 发现级 **未通过**（诚实：不是「只有 val 吓人」）。  
3. 假设「纯 pad200 无后文正样本」未能在 tip 重渲或强制 tip 窗上超过 v12；v12 的 tip_hit 优势未迁移到 v13，反而 catastrophic 于 tip 协议。  
4. **未**跑 frozen-F1 / holdout；主线权重仍是 v12。

## 风险与诚实声明

- tip-smoke 本机用 `forward_log_vps_20260721.csv` 快照；与 VPS 当日币池一致口径，但非新前向 100。  
- 未做 owner 目视叠框；若怀疑 conf/渲染，下一步是 H-DET-4 极小消融，不是再盯 val mAP。  
- 不自动 promote；不杀已结束进程。

## 下一步（需 owner 点头的标出）

1. **默认**：主线保持 v12；H-DET-1 记 🔴。  
2. **排队**：H-DET-4 / EXT-5 渲染消融（GPU 空闲、单变量）。  
3. **可选（owner）**：H-DET-2 硬负开训时机与样本量。  
4. **禁止**：用 val mAP「再训一轮凑数」、自动切 ACTIVE、耗 holdout。
