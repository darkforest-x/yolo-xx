# Pattern Library v1（候选版）

**Phase 1 / Task 2** · 2026-08-06
**产物**：`reports/pattern_library_candidate.json`（2366 条，2.5 MB）
**性质**：只读构建。未修改 `datasets/`、`fable-trading/` 下任何原始数据；未训练；未消耗 holdout。

---

## 1. 内容

| source | 条数 | `human_confirmed_pattern` | 含义 |
|---|---|---|---|
| `golden_pool` | **2207** | `true` | owner 亲手画的框（dense_owner_v9 train+val） |
| `v10_scan_20260805` | **159** | `false` | teacher 在最近 5 天全市场扫描中的检出 |

| 维度 | 分布 |
|---|---|
| split | train 1692 · val 515 |
| 币种类型 | 全部 `_USDT_SWAP`（非永续 0 条，见 §2） |
| stem 约定 | start 1646 · end 561 |
| `render_mad` | **全部 0.00000** |
| `ma_structure` 填充 | 2366 / 2366 |
| `human_label` | **全部 null**（见 §4） |

## 2. 入库门与损失

| 阶段 | 数量 |
|---|---|
| 有框 stem（train 3031 + val 1128） | 4159 |
| 本机无该币 K 线 | −444 |
| MAD ≥ 1.0（判定索引与当前 K 线不对齐） | **−1587** |
| **通过** | **2125 stem → 2207 框** |
| 空 label stem（背景负样本，本版不入库） | 5005 |

**被 MAD 拒绝的 1587 个基本都是现货**：入库的 2207 条全部是永续，非永续 0 条。
现货 K 线自标注以来已与当时不一致（实测 MAD 8–11），若不设闸直接入库，
这批样本的 signal 位置会整体错位。

入库样本的 MAD **全部为 0.00000** —— 不是「接近」，是像素级完全一致。

stem 数字的两种约定（窗起点 / 窗末根）逐样本用像素 MAD 消歧，不按前缀猜。
依据：`docs/learnings/stem-conventions-must-be-disambiguated-per-sample-not-per-prefix.md`。

## 3. 字段

```json
{
  "pattern_id": "dense_00001",
  "source": "golden_pool | v10_scan_20260805",
  "split": "train | val | null",
  "symbol": "ARM_USDT_SWAP", "timeframe": "15m",
  "signal_time": "...", "signal_i": 9107,
  "image_path": "...", "stem": "...", "stem_convention": "start|end",
  "render_mad": 0.0,
  "window": {"bars": 200, "start_i": 8908, "end_i": 9107},
  "bbox_xywhn": [cx, cy, w, h],
  "confidence": null,                 // owner 框无 conf；teacher 检出有
  "human_confirmed_pattern": true,
  "human_label": null,                // 质量分级 A/B/C，未审
  "human_reviewed_at": null,
  "ma_structure": {
    "fast_spread", "full_spread", "ma_spread_pct", "cluster_width_atr",
    "atr_pct", "slow_slope_12", "order_score", "down_order_score",
    "trend_order_score", "dense_run_bars"
  },
  "embedding": null,                  // Phase 2 需要时再补
  "teacher_version": "owner_v10_chain",
  "teacher_sha256": "b9a84b5f5ebf0032…"
}
```

## 4. human_label 政策

**全部为 `null`，且禁止用规则自动预标填充。**

2026-07-23：v16 被判「空背景误火 17/33 = 51.5%」而否决上线；owner 逐图核实后发现，
那 33 张是**规则自动预标**，v16 的框画在真实密集启动上——**标签比模型错**。
自动标签一旦进入 `human_label` 字段，就会以「人工确认」的身份参与后续所有裁决。

## 5. 建库过程中出现的一个数字

对每条记录在 signal 位置计算 `ma_structure` 后，可以直接问：
**owner 画框的地方，按机械密集规则算不算「密集」？**

| 样本 | 满足 `fast_spread ≤ 0.0028` 且 `full_spread ≤ 0.0055` |
|---|---|
| **owner 手标框**（n=2207） | **282 = 12.8%** |
| **v10 自己检出**（n=159） | **84 = 52.8%** |

owner 框位置的 `fast_spread` 中位数为 **0.00529**，接近阈值 0.0028 的两倍；
按机械规则的信号条件（连续密集 ≥5 根）衡量，owner 框只有 **12.1%** 达标。

两条并列：owner 认定的形态里约 **87%** 不满足机械密集定义，
而 teacher 检出的点有 **过半** 满足。

按本仓口径不下判决。仅指出这组数字与既有记录一致
（`docs/learnings/owner-eye-is-anticorrelated-with-the-mechanical-dense-definition`），
并且它同时给 Task 3 的低命中率提供了一个可检验的方向：
Task 3 的样本全部来自机械规则，而 owner 与 teacher 在该定义上的重合度分别是 12.8% 与 52.8%。

**注意方向**：本节测的是 `P(满足机械规则 | 被标注/被检出)`，
Task 3 测的是 `P(被检出 | 满足机械规则)`。两者不是同一个量，不可互推。

## 6. 已知限制

1. **空 label（5005 个背景负样本）未入库。** 本版只收形态，负样本另议。
2. **非永续样本全部落选**，Library 当前只覆盖 SWAP 宇宙。
3. `embedding` 全为 null——Phase 2 若需要，再用 teacher 统一提取。
4. `signal_i` 由框右缘映射得到，沿用 L1 现有口径；
   `box-right-edge-maps-launch-bar-not-tip` 指出该口径对应 launch bar 而非 tip。
5. owner 标注时可见未来（499 个 ⭐ 中仅 2 个画在盘口，中位可见 97 根）。
6. 同一 stem 的多个框各自成条，故条数（2207）多于 stem 数（2125）。

## 7. 产物

| 文件 | 内容 |
|---|---|
| `reports/pattern_library_candidate.json` | 2366 条记录 |
| `scripts/build_pattern_library.py` | 构建脚本 |
| `reports/pattern_library_v1.md` | 本报告 |
