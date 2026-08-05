# 历史资产复用裁决（PR-00）

日期：2026-08-05
基线 commit：`4e8b3e0fff0bca8b49986fe5e81aab75c3efef06`
范围：`small_timeframe_perfect_pattern_detector`
机器可读版本：[`asset_registry_v2.json`](asset_registry_v2.json)、[`../reports/pr00_asset_audit.json`](../reports/pr00_asset_audit.json)
生成脚本（只读）：[`../scripts/pr00_asset_registry.py`](../scripts/pr00_asset_registry.py)

本轮**没有训练、没有修改任何标签、没有移动或删除任何历史资产、没有联网**。
全部数字来自对本机文件的实际读取与 SHA-256 计算。

## 1. 登记总量

| 项 | 值 |
|---|---:|
| 登记资产条目 | 157 |
| 实际读取并计算 SHA 的字节数 | 20,526,722,053 |
| SHA 核对次数 | 65 |
| 核对一致 | 65 |
| 核对不一致 | 0 |

按类型：snapshot 17、dataset 15、scan_set 27、weight 26、run 9、prediction_set 41、
review_gallery 3、annotation_queue 2、report 17。

按裁决：`DIRECT_REUSE` 9、`REVIEW_AND_REUSE` 44、`LEGACY_BASELINE_ONLY` 87、`REJECT` 17。

核对内容包括：四个迁入数据集的迁移口径内容哈希、5 个迁入权重、17 个 source snapshot
清单、2 个 dataset manifest、5 个 pair manifest、owner 标注 CSV，以及 31 处
predictions.json 里记录的 run 权重 SHA 与本地 `.pt` 实际 SHA 的比对。全部一致。

内容哈希口径：`sha256(concat(排序后的相对路径 + 0x00 + 文件字节))`，排除 `data.yaml`
（内含按机器改写的绝对路径）、`*.cache`（ultralytics 可再生）与 `.DS_Store`。
2026-08-03 迁移时的旧口径只排除 `data.yaml`，作为 `migration_convention_sha256` 单独保留，
用来核对 `docs/migrated_assets.json` 中已记录的四个哈希。

## 2. 四档含义

| 状态 | 含义 |
|---|---|
| `DIRECT_REUSE` | 可直接作为新流水线的可信输入或官方基座 |
| `REVIEW_AND_REUSE` | 可作为候选池，但必须按新“完美形态”规范逐条重新审核 |
| `LEGACY_BASELINE_ONLY` | 仅用于历史对照、预标注、误报挖掘或 warm-start A/B |
| `REJECT` | 不进入新项目主线（仍然保留，不删除） |

硬规则：旧资产冻结不动；新训练集另起 `dataset_id`，不在旧目录原地改标签；旧标签只有
符合新任务定义才能进入新数据集；模型预测不是真值；空标签不是负样本；outcome 派生的
hard negative 不等于形态负类；2026-05-04 之后已被反复查看的区间不得作为新 final test。

## 3. 原始 OHLCV 快照

全部 17 个快照的 `source_snapshot.json` SHA 与 `snapshot_summary.json` 中记录值一致，
且 `post_cutoff_ohlcv_rows_materialized` 均为 0。

### 3.1 pre-holdout（`DIRECT_REUSE`，7 条）

| 路径 | 周期 | 规模 | 说明 |
|---|---|---|---|
| `data/manual_short_preholdout_15m` | 15m | 215 symbols / 5,900,085 行 | owner 框恢复的原始数据；15m |
| `data/preholdout_15m_scan` | 15m | 215 文件 / 5,900,085 行 | 连续扫描源；15m |
| `data/preholdout_30m` | 30m | 54 文件 / 854,498 行 | 非当前目标周期 |
| `data/micro_preholdout/5m` | 5m | 14 symbols / 542,628 行 | **建议主周期的唯一可信 pre-holdout 输入** |
| `data/micro_preholdout/3m` | 3m | 2 symbols / 49,430 行 | 只有 BTC、ETH |
| `data/micro_preholdout/2m` | 2m | 18 symbols / 659,802 行 | 非首轮周期 |
| `data/micro_preholdout/1m` | 1m | 1 symbol / 23,584 行 | 只有 ETH；排最后 |

小周期 pre-holdout 覆盖（这是 PR-02 的硬约束，先写清楚）：

```text
5m  14 symbols  2025-12-20T10:00Z → 2026-05-03T23:55Z
    ADA ARB AVAX BNB BTC DOGE DOT ETH LINK LTC OP SOL TRX XRP
3m   2 symbols  2026-03-13T12:15Z → 2026-05-03T23:57Z   BTC ETH
2m  18 symbols  2026-03-13T13:26Z → 2026-05-03T23:58Z
1m   1 symbol   2026-04-17T14:56Z → 2026-05-03T23:59Z   ETH
```

15m 是 215 个币种、5,900,085 行；小周期只有上面这些。首轮若冻结 5m，跨币种能力上限由
这 14 个币种决定；若冻结 3m，pre-holdout 只有 BTC 和 ETH。

**15m 框不能缩放成小周期标签。** 15m 数据的作用是形态定义、参考画廊和候选时间定位。

### 3.2 holdout 期（`LEGACY_BASELINE_ONLY`，10 条）

`data/holdout_scan_20260804/{1m,2m,3m,5m,15m,30m}`、`data/wide_holdout_5m`、
`data/review_{1m,3m,5m}`：cutoff 在 2026-08-04，覆盖 2026-05-04 之后区间且已被反复查看。
只能用于 development、回归与历史复现，**不得作为新 final test**。

## 4. 标注资产

| 资产 | 裁决 | 说明 |
|---|---|---|
| `data/manual_short_preholdout_15m/owner_short_annotations.csv` | `REVIEW_AND_REUSE` | 1,361 条 owner 手标做空框，`label_origin=owner`，SHA `23fd378a…`。是形态定义的最高价值来源，但语义是 15m，进新数据集前必须在小周期图上逐框重新确认 |
| `datasets/eth_short_tip_label2000` | `REVIEW_AND_REUSE` | 2,000 张 ETH 图（3m 667 / 5m 667 / 10m 666）配 2,000 个**空** label。空标签 = 未标注，绝不能解释成 2,000 个负样本；只能作人工标注队列 |

## 5. 训练数据集（15 条）

| 数据集 | 周期 | 裁决 | 关键理由 |
|---|---|---|---|
| `dense_owner_short_star_tip_v10` | unknown | `REVIEW_AND_REUSE` + `LEGACY_BASELINE_ONLY` | train 2,207 图 / 1,006 框 / 1,201 背景，val 1,098 / 316 / 782。旧 dense 语义，可筛正样本与 near-miss、固定历史 benchmark；不能直接作最终数据集 |
| `dense_owner_side_short_tip_v3` | unknown | `REVIEW_AND_REUSE` + `LEGACY_BASELINE_ONLY` | train 1,320 / 571 / 749，val 592 / 194 / 398。right-edge/tip 偏差案例、位置捷径测试、困难负样本候选 |
| `eth_3m_short_pilot_v1` | 3m | `REVIEW_AND_REUSE`（前提：首轮周期冻结为 3m） | train 135 / 60 / 75，val 48 / 16 / 32。唯一的小周期 owner 复核数据集；其他周期降为 `LEGACY_BASELINE_ONLY` |
| `owner_short_original_w200` | 15m | `REVIEW_AND_REUSE` | train 1,035 图 / 1,076 框，val 277 / 280。框几何与 200 根位置分布；正图为主、无可信真实背景 |
| `owner_short_staggered_w96` | 15m | `REVIEW_AND_REUSE` | train 1,076 / 1,076，val 280 / 280，pixels_per_bar 13.221053。0/8/16/24 右侧上下文与位置去偏设计 |
| `owner_short_paired_ab_v2` | 15m | `LEGACY_BASELINE_ONLY` | 1,331 对 1:1 正负。pair ledger 与单变量 A/B 合同可复用；1:1 不是真实基率，“无 owner 框”不等于形态负类 |
| `owner_short_paired_ab_v1` | 15m | `LEGACY_BASELINE_ONLY` | 被 v2 取代 |
| `owner_short_paired_ab_fixture8` | 15m | `LEGACY_BASELINE_ONLY` | 8 对构建 fixture，只作 smoke |
| `owner_short_hardneg_v1` | 15m | `REVIEW_AND_REUSE` | w96 臂 1,331 正 + 5,304 背景（train 5,254 图 / 1,172 框，val 1,381 / 305）。构造条件含“做空亏损”的 outcome 过滤，**不继承负标签** |
| `owner_short_hardneg_v2` | 15m | `REVIEW_AND_REUSE` | w96 臂 1,331 正 + 5,097 背景（train 5,165 图 / 1,172 框，val 1,263 / 305）。语境更接近正样本，是高价值 near-miss 池，仍需逐图人工确认 positive/negative/uncertain |

`dense_owner_short_star_tip_v10` 与 `dense_owner_side_short_tip_v3` 的周期标为 `unknown`：
两个目录里的 `build_meta.json`、`data.yaml` 和文件名都没有声明 timeframe。旁证
（`datasets/eth_short_tip_label2000/README.md` 称 v10 权重是「15m OOD 提议源」）指向 15m，
但仓库内没有可核验的声明，按 PR-00 规则不猜，标 `unknown`。两者都是 `REVIEW_AND_REUSE`，
周期未确认这件事本身就是重新审核时必须解决的问题。

图像规格：两者均为 1280×742，与当前渲染尺寸一致。

## 6. 连续扫描集（27 条）

pre-holdout（`REVIEW_AND_REUSE`，8 条）——可用于连续背景、误报挖掘、位置审计与人工复核：

| 扫描集 | 周期 | 每臂样本 | 币种 |
|---|---|---:|---:|
| `preholdout_15m_train_scan` | 15m | 5,824 | 208 |
| `preholdout_15m_val` | 15m | 4,000 | 215 |
| `preholdout_dense_v2/5m` | 5m | 6,000 | 14 |
| `preholdout_dense_v2/2m` | 2m | 4,000 | 18 |
| `micro_scan_preholdout_v1/{5m,3m,2m,1m}` | 5m/3m/2m/1m | 各 512 | 14/2/18/1 |

holdout 期（`LEGACY_BASELINE_ONLY`，19 条）：`holdout_scan_20260804{,_v2}` 各 6 个周期、
`holdout_dense_20260804/{2m,3m,5m}`、`wide_holdout_5m`、`review_{1m,3m,5m}`。
全部 `holdout_read: true`，只作 development 与历史复现。

扫描集只有图、没有标签：**任何“无框”都不是已确认负样本。**

## 7. 模型权重（26 条）与 run（9 条）

### 7.1 官方基座 `DIRECT_REUSE`

```text
weights/bases/yolo11n.pt  0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
weights/bases/yolo11s.pt  85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5
```

### 7.2 历史基线

| 权重 | 裁决 | 说明 |
|---|---|---|
| `weights/baselines/owner_short_star_v10.pt` | `LEGACY_BASELINE_ONLY` | 旧 dense 基线，可预标注 / FP mining |
| `weights/baselines/owner_side_short_tip_v3.pt` | `LEGACY_BASELINE_ONLY` | 旧 tip 基线 |
| `weights/baselines/eth3m_short_pilot_v1_mac_cold.pt` | `REVIEW_AND_REUSE`（目标 3m 时） | ETH 3m pilot；历史结论是严格 OOS 774 根中开火 772 根，已被拒绝为成功先验 |

### 7.3 本仓库训练的 run

| run | 数据集 | 窗口 | 底座 | epoch | 裁决 |
|---|---|---:|---|---:|---|
| `hardneg_w96_v2_s` | hardneg_v2/w96 | 96 | yolo11s | 18 | `LEGACY_BASELINE_ONLY` + **`LEGACY_CHAMPION`** |
| `hardneg_w96_v2_s2` | hardneg_v2/w96 | 96 | yolo11s | 40 | `LEGACY_BASELINE_ONLY` |
| `hardneg_w96_v2` | hardneg_v2/w96 | 96 | yolo11n | 34 | `LEGACY_BASELINE_ONLY` |
| `hardneg_w96_v1` | hardneg_v1/w96 | 96 | yolo11n | 21 | `LEGACY_BASELINE_ONLY` |
| `hardneg_w200_v2` | hardneg_v2/w200 | 200 | yolo11n | 35 | `LEGACY_BASELINE_ONLY` |
| `hardneg_w200_v1` | hardneg_v1/w200 | 200 | yolo11n | 28 | `LEGACY_BASELINE_ONLY` |
| `owner_short_ab_w96_v2` | paired_ab_v2/w96 | 96 | yolo11n | 20 | `LEGACY_BASELINE_ONLY` + `WARM_START_CANDIDATE` |
| `owner_short_ab_w200_v2` | paired_ab_v2/w200 | 200 | yolo11n | 19 | `LEGACY_BASELINE_ONLY` |
| `imported/eth3m_short_pilot_v1_mac_cold` | eth_3m_short_pilot_v1 | 200 | yolo11n | 45 | `LEGACY_BASELINE_ONLY` |

历史数字（不作为新验收依据，只作 benchmark）：

```text
owner_short_ab_w96_v2   连续扫描 overall precision 5.5%；top-50/100/200 均 6.0%
                        配对集 mAP50 0.548 / mAP50-95 0.195（w200 为 0.339 / 0.091）
hardneg_w96_v2_s        连续扫描 overall precision 3.4%；top-50 10.0%、top-100 10.0%、top-200 7.5%
                        best checkpoint epoch 15
```

因此新项目首轮 baseline 用 96 根窗口，不默认重跑 w200。

`best.pt` 用于复现与 benchmark；`last.pt` 只用于恢复中断或历史审计，新实验不得默认从旧
`last.pt` 续训；warm-start 必须被声明为唯一实验变量；最终冻结只能用通过新 test 的 `best.pt`。

旧阈值 `conf=0.30 / iou=0.70` 是 `LEGACY_BASELINE_ONLY`：可用于横向 benchmark，
新模型的最终阈值只能在 val 上冻结，不能在 final test 上调。

## 8. 预测产物（41 条）

`reports/` 下每个 `predictions.json` 都记录了权重路径、权重 SHA、scan 合同 SHA、conf、
图片数与检出数。31 处引用本仓库权重的记录，其 SHA 与本地 `.pt` 实测值全部一致。

- pre-holdout 扫描的预测：`REVIEW_AND_REUSE`，用于 FP mining 与预标注候选；
- holdout 期扫描的预测：`LEGACY_BASELINE_ONLY`；
- 一处引用了本仓库没有的外部权重（`owner_short_star_v8/weights/best.pt`，记录 SHA
  `9178ecde…`），已登记为不在仓库、无法核对。

**模型预测不是真值。** 任何框进入新数据集前都必须 owner 人工确认。

## 9. `REJECT`（17 条，保留不删）

| 资产 | 原因 |
|---|---|
| `reports/*/outcome/`（5 处 `signal_outcome.json`） | 收益/outcome 派生产物，当前明确禁止范围 |
| `preholdout_dense_v2/{backtest_report,judgment_and_control,root_cause}.html`、`wide_holdout_5m/final_diagnosis.html` | 回测与判断层报告，当前范围外 |
| `dense_owner_v16_tipuni` 及 `owner_v16_tipuni_cold.pt` | 含 2026-07 样本（不在本仓库） |
| `label_live_tip_1000` | 无法证明全部 pre-cutoff（不在本仓库） |
| side-short-tip v1 / v2 | 被 v3 替代且重复（不在本仓库） |
| 分类数据集与 cls 权重 | 当前只做 detection（不在本仓库） |
| judgment / backtest 数据 | 当前范围外 |
| 未审核空 label 作为 negative | 空标签不等于负样本 |

仓库里的 `src/yolo_xx/outcome.py`、收益报告与判断层尝试保留为历史资产：不删除、不扩展、
不被新 core import、不作为默认 CLI、不作为标签依据，也不作为模型是否完成的验收依据。

## 10. PR-00 之后的已知约束

1. 首轮周期尚未冻结。`configs/PERFECT_PATTERN_SPEC_V1.yaml` 由 owner 填写，其他模型不得
   代填“双均线/均线组”具体指哪些线。周期选择直接决定可用数据：5m 有 14 个币种，
   3m 只有 BTC 和 ETH。
2. 小周期的 owner 真值目前只有 `eth_3m_short_pilot_v1`（76 个唯一正锚点，单币种）。
   5m 目前**没有**任何 owner 确认的小周期框——1,361 个 owner 框全部是 15m 语境。
3. 新 final test 必须避开 2026-05-04 之后已被反复查看的区间；现有 holdout 扫描集与
   holdout 快照都已标记 `development_consumed`。
