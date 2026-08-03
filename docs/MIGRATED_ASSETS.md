# Migrated YOLO assets

迁移日期：2026-08-03。来源：`/Users/zhangzc/fable-trading`。

本清单只覆盖独立 YOLO 检测项目需要的本地资产。迁移是复制，不是移动；父仓库源文件未删除。
本轮未训练、未读取 holdout、未修改 ACTIVE、未部署、未下单。

## Dataset copies

| Dataset | Role | Audit | Content SHA-256 |
|---|---|---:|---|
| `datasets/dense_owner_short_star_tip_v10` | balanced dense-cluster baseline | valid; train 2,207 images / 1,006 boxes; val 1,098 / 316 | `624dd884fd76b7c5a2bab5a36e1e48f3d1ca5a4d9da2ae35a07f43fad2c292a8` |
| `datasets/dense_owner_side_short_tip_v3` | right-edge tip baseline with hard/easy negatives | valid; train 1,320 / 571; val 592 / 194 | `1c0286cb2ff43d0991c0c57b1a27f352393537d2c4f5da8249712fed30f52d68` |
| `datasets/eth_3m_short_pilot_v1` | causal-tip detector pilot | valid; train 135 / 60; val 48 / 16 | `c97c88d474b9141a14b0d6aaa7fdf61f5f1f825ed98e0c6fb23c6cf7bff8e969` |
| `datasets/eth_short_tip_label2000` | annotation/review pack, not train-ready | 2,000 images paired with 2,000 empty labels | `9473039b8fb964dde0f8a128b5770a1168b5bcb40b8ad57950e494d258edad4b` |

The first three hashes cover relative file names and bytes except `data.yaml`; their copied YAML roots were
intentionally changed from the parent repository to `/Users/zhangzc/yolo-xx/datasets/...`. The annotation-pack
hash covers every file. Each source/copy hash pair matched after the intentional YAML rewrite.

All four datasets are documented as strictly earlier than `2026-05-04T00:00:00Z`. The 2,000-image pack is
an unlabeled work queue; empty labels must not be interpreted as negative examples.

## Weight copies

| Weight | Role | SHA-256 |
|---|---|---|
| `weights/bases/yolo11n.pt` | Ultralytics YOLO11n base | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| `weights/bases/yolo11s.pt` | Ultralytics YOLO11s base | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |
| `weights/baselines/owner_short_star_v10.pt` | dense-cluster historical baseline | `86d969c830189b2d1048dca24e10bacc27341e75643cf4f7a912e5a8d5542ad9` |
| `weights/baselines/owner_side_short_tip_v3.pt` | right-edge tip historical baseline | `f1c65e70c94930b9e7c052c43ca61353528747f740687a7f821a42b74e275f61` |
| `weights/baselines/eth3m_short_pilot_v1_mac_cold.pt` | causal-tip pilot baseline | `7f1f3b2d6300952ac5c4adb3937c0b3b2e9d7879b73ed2855bcf08b55cbe202e` |

The full `eth3m_short_pilot_v1_mac_cold` run record was copied to `runs/imported/`, including `args.yaml`,
`results.csv`, `best.pt`, and `last.pt`. Relevant YOLO-only historical reports were copied to
`reports/imported/`.

## Deliberately excluded

- `dense_owner_v16_tipuni` and `owner_v16_tipuni_cold.pt`: the dataset contains samples dated 2026-07 and
  is not a clean pre-holdout baseline.
- `label_live_tip_1000`: its metadata does not prove every sample is strictly pre-holdout.
- `dense_owner_side_short_tip` v1 and v2: superseded by the balanced v3 copy and unnecessarily duplicate data.
- classification datasets and `yolo11n-cls.pt`: this repository currently implements object detection only.
- `owner_best.json`, ACTIVE pointers, frozen LightGBM files, judgment/backtest outputs, deployment and trading
  files: outside this repository's YOLO-only scope.

## 2026-08-04 owner short-box recovery

The missing-image `dense_owner_side_short` wrapper was rebuilt from the surviving owner review coordinates and
a newly materialized immutable pre-holdout OHLCV prefix. The recovery produced:

- `data/manual_short_preholdout_15m`: 215 symbols / 5,900,085 rows / 1,361 short annotations;
- `datasets/owner_short_original_w200`: original 200-bar positions;
- `datasets/owner_short_staggered_w96`: 96-bar position-balanced short-window experiment.

These assets remain gitignored. Their hashes, split counts, audits and safety declaration are recorded in
[`MANUAL_SHORT_RECOVERY.md`](MANUAL_SHORT_RECOVERY.md) and
[`manual_short_recovery_20260804.json`](manual_short_recovery_20260804.json).
