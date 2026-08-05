# yolo-xx

`yolo-xx` 是从 `fable-trading` 检测层抽出的独立离线项目，只做一件事：准备、训练和验证
YOLO 图像检测模型。

## 当前范围（2026-08-05 起，收敛）

当前唯一目标：

> 在一个冻结的小周期上，训练出能够高精度识别「非常标准、非常完美的均线密集形态」的
> 单类别 YOLO 检测模型。

链路只有这一条：小周期 OHLCV → 固定均线 → 固定渲染 → 完美形态人工标签 → 高质量正负
数据集 → 可复现训练 → 标准/困难/连续背景验证 → 错误分析 → 数据集修正 → 重训 → 冻结。

模型冻结之前（Model Card / Dataset Card / Final Acceptance 全部产出之前），不得实现、
恢复或扩展：判断层与 LightGBM、收益标签、TP/SL、回测、交易成本、仓位、多周期交易共振、
实盘扫描、交易所连接、下单、ACTIVE、模型晋升、通知、Agent/MCP 编排、强化学习和产品层。

仓库里已有的 `src/yolo_xx/outcome.py`、收益报告和判断层尝试保留为历史资产：不删除、
不扩展、不被新 core import、不作为默认 CLI，也不决定正负标签或验收结论。

历史资产（数据集、权重、run、扫描集、预测产物）已全部登记并分四档（`DIRECT_REUSE` /
`REVIEW_AND_REUSE` / `LEGACY_BASELINE_ONLY` / `REJECT`）：

- [docs/ASSET_REUSE_DECISIONS.md](docs/ASSET_REUSE_DECISIONS.md)：裁决与理由；
- [docs/asset_registry_v2.json](docs/asset_registry_v2.json)：157 条机器可读登记，含 SHA-256；
- [reports/pr00_asset_audit.json](reports/pr00_asset_audit.json)：PR-00 机器验收。

旧资产一律冻结：不删除、不移动、不覆盖、不原地改标签。新训练集另起 `dataset_id`。

## 边界

包含：

- 从本地 OHLCV CSV 按真实分钟时长计算对应的 SMA/EMA；
- 渲染 K 线和六条均线；
- 生成/校验单类别 `dense_cluster` YOLO 数据集；
- 使用固定的语义安全增强配置训练 YOLO；
- 离线验证并输出 JSON 指标；
- 对本地图片离线预测，导出 YOLO 标签、检查图和 JSON manifest。

不包含：数据抓取、方向或收益判断、LightGBM、标签收益、回测、成本、交易所连接、实盘扫描、
订单、通知、看板、ACTIVE、模型晋升和部署。项目不导入父仓库的 `src` 或其他业务包。

项目已从父仓库复制一组可核验的 pre-holdout YOLO 数据、底座权重、检测基线和历史训练记录；
这些本地资产由 `.gitignore` 排除，不会进入 Git。源目录未移动或删除，也不会自动开始训练。
完整清单、哈希与排除理由见 [docs/MIGRATED_ASSETS.md](docs/MIGRATED_ASSETS.md)。

## 安装

```bash
cd yolo-xx
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

## 从本地 CSV 构建数据集

输入 CSV 至少要有 `ts,open,high,low,close,volume`，`ts` 为毫秒时间戳；文件名建议为
`okx_BTC_USDT_SWAP_15m_10000.csv`。非 dry-run 构建除了目录以外，必须显式提供一个由可信快照
流程生成的 `--source-manifest`。构建器不会自行扫描目录凑数据，也不会联网取数。

source manifest 必须使用 `yolo_xx_source_snapshot` schema v1，声明
`immutable=true`、绝对 `source_dir`、`timeframe`、`cutoff_exclusive`，以及每个文件的
`path/size_bytes/mtime_ns/sha256/row_count/first_open_time/last_open_time/last_closed_at`。
`cutoff_exclusive`、`--end-before` 和可选 `--split-at` 均不得晚于
`2026-05-04T00:00:00Z`。示意结构如下（摘要值不能用占位符）：

```json
{
  "schema_version": 1,
  "manifest_type": "yolo_xx_source_snapshot",
  "immutable": true,
  "source_dir": "/absolute/path/to/pre_holdout_csvs",
  "timeframe": "15m",
  "cutoff_exclusive": "2026-05-04T00:00:00Z",
  "files": [{
    "path": "okx_BTC_USDT_SWAP_15m_10000.csv",
    "size_bytes": 1234567,
    "mtime_ns": 1700000000000000000,
    "sha256": "<真实的64位小写SHA-256>",
    "row_count": 10000,
    "first_open_time": "2025-01-19T20:00:00Z",
    "last_open_time": "2025-05-03T23:45:00Z",
    "last_closed_at": "2025-05-04T00:00:00Z"
  }]
}
```

```bash
yolo-xx-build \
  --cache-dir /absolute/path/to/pre_holdout_csvs \
  --source-manifest /absolute/path/to/source_snapshot.json \
  --out datasets/dense_15m \
  --end-before 2026-05-04T00:00:00Z \
  --max-images 3200

yolo-xx-audit --dataset datasets/dense_15m --out reports/dataset_audit.json
```

不带周期参数时保持原来的 15m 语义：MA 物理时长为 300/900/1800 分钟，对应 15m 图上的
20/60/120 根；dense 区间为 75–180 分钟，对应 5–12 根。先用 dry-run 检查 5m 配置；该
命令只解析参数：不会读取 source manifest、不会列举或读取任何 CSV，也不会创建输出目录：

```bash
yolo-xx-build \
  --cache-dir /absolute/path/to/pre_holdout_5m_csvs \
  --out datasets/dense_5m_fixture \
  --timeframe 5m \
  --ma-minutes 300,900,1800 \
  --dense-min-minutes 75 --dense-max-minutes 180 \
  --merge-gap-minutes 30 \
  --max-images 200 \
  --end-before 2026-05-04T00:00:00Z \
  --dry-run
```

确认 manifest 后，补上 `--source-manifest` 并删除 `--dry-run` 才会实际构建。未显式传
`--window` 时，窗口固定为 3000 物理分钟：15m=200 根、5m=600 根、3m=1000 根、
2m=1500 根、1m=3000 根；未传 `--stride` 时等于窗口。计划与 manifest 会记录
`physical_window_minutes/window_bars/pixels_per_bar`。1m/2m 当前只记录亚像素横向分辨率风险，
不会伪装成训练门；实验顺序是首轮只做 5m，3m 后续，1m 最后。

MA 分钟数必须能被周期整除；dense 最短时长向上取整为完整 bar，最长时长与合并间隔向下
取整，实际换算值写入 manifest。默认开启严格 cadence，重复、乱序、缺失 K 线或任何必需
字段（包括 volume）解析失败都会报错。

`--end-before` 是严格的输入可用时间上界，默认 `2026-05-04T00:00:00Z`。构建开始时先凭
manifest、stat 和 SHA-256 对所有文件做 fail-closed 身份校验，校验完成之前不调用
`pd.read_csv`；实际解析后再核对行数、首尾开盘时间和最后闭合时间，扫描前后继续检查
stat/hash 漂移。混有 holdout、截止时间过晚、文件缺失或被修改都会在训练数据生成前失败。
输出目录如已有图片或标签，构建器会拒绝覆盖，防止混合两轮数据。

train/val 使用一个全局 UTC `split_at`，不再按每个 symbol 的行数切分。train 窗口要求
`available_at < split_at`，val 窗口要求 `window_start_time >= split_at`，跨界窗口直接丢弃。
构建后强制验证 `max(train.available_at) < min(val.window_start_time)`，并把 `split_at` 与
`dropped_cross_split` 写入 summary/manifest。

每轮构建生成 schema v2 `dataset_manifest.json`。每个 source、image 和 label 都记录
SHA-256；快照 manifest 原文也复制进数据集并锁定哈希。每个窗口包含绝对
`window_start_time/window_end_close_time`，每个框包含真实 `segment`、`box_start_time`、
`box_end_time`、`box_end_close_time` 和归一化 `xywhn`。特别注意：

> `available_at` 始终等于该输入窗口的 `window_end_close_time`。

框可以位于图的左、中、右任意真实位置，但中部历史框只能在整个输入窗口闭合后被模型看到；
不得把检测可用时间回填成 `box_end_time`。位置变化只来自真实窗口与自动规则标签，不使用平移、
缩放、翻转或 mosaic 制造位置变化。

## 训练与验证

先用 dry-run 核对所有参数；dry-run 不导入模型、不读取数据、不训练：

```bash
yolo-xx-train --data datasets/dense_15m/data.yaml --model yolo11n.pt --dry-run
```

确认后才运行训练：

```bash
yolo-xx-train \
  --data datasets/dense_15m/data.yaml \
  --model yolo11n.pt \
  --epochs 100 --patience 20 --name dense_15m_cold

yolo-xx-eval \
  --weights runs/detect/dense_15m_cold/weights/best.pt \
  --data datasets/dense_15m/data.yaml \
  --out reports/dense_15m_cold_val.json
```

## 离线预测检查

先 dry-run 核对路径和参数；不会读取权重或图片：

```bash
yolo-xx-predict \
  --weights runs/detect/dense_15m_cold/weights/best.pt \
  --source /absolute/path/to/local/images \
  --out reports/dense_15m_cold_predictions \
  --dry-run
```

真实 train/eval 在导入 Ultralytics 和加载权重之前都会执行 schema-v2 强审计：样本/图片/标签
唯一性与对应关系、UTC 因果顺序、availability、manifest 与 YOLO label 逐行一致性、全局时间
切分不变量，以及 source snapshot/image/label 哈希任一失败都会拒绝运行。Ultralytics 固定为
`8.4.89`，避免安全增强参数集合随依赖漂移。

确认后执行离线预测：

```bash
yolo-xx-predict \
  --weights runs/detect/dense_15m_cold/weights/best.pt \
  --source /absolute/path/to/local/images \
  --out reports/dense_15m_cold_predictions \
  --save-overlays
```

对 schema v2 数据集做预测时，可把数据集根目录作为递归 source，并附带 manifest：

```bash
yolo-xx-predict \
  --weights runs/detect/dense_5m_cold/weights/best.pt \
  --source datasets/dense_5m_fixture \
  --out reports/dense_5m_fixture_predictions \
  --recursive \
  --dataset-manifest datasets/dense_5m_fixture/dataset_manifest.json
```

带 manifest 的预测要求 `--source` 正好是 manifest 所在的数据集根目录，并在推理前
核验整个数据集及每张输入图 SHA-256。结果使用 Ultralytics 的 `result.path` 回配输入，而不
假设模型输出顺序。预测器按 `relative_image` 精确匹配 manifest sample，将 `sample_id`、`symbol`、
`detector_timeframe`、窗口绝对时间和 `available_at` 写入 prediction item；每个 detection 也只
继承该 sample 的 `available_at`。预测框的右边界只表示图中位置，绝不会被反推成
`signal_time`。

输出包含 `labels/`、可选的 `overlays/` 和 `predictions.json`。源目录的相对层级和图片扩展名
会被保留（例如 `sample.jpg.txt`），同 stem 不同格式的图片不会互相覆盖；非空输出目录会被
拒绝。manifest 同时记录权重文件的 SHA-256。

续训权重默认启用保守 AdamW 配方；`yolo11*.pt` 冷启动默认保留 Ultralytics 优化器选择。
平移、缩放、翻转、mosaic、mixup、cutmix、HSV、BGR 交换、multi-scale 和其他会改变时间、
位置或颜色语义的增强始终关闭。
训练输出目录如果已经存在会直接拒绝，避免两次实验混到同一 run。

### 200 / 96 根单变量 A/B

当前 owner-short A/B 不再直接拿正样本集开训。它先用同一个 anchor ledger 给每个正图匹配一张
相同 symbol、split、右侧上下文桶和邻近时间块的背景图；背景在 200/96 两个视图中都必须不与
任何已知 owner 框重叠，并且不触发冻结 dense 规则。两臂 sample id、正负身份和窗口终点一致，
只有窗口长度、真实渲染内容和相应框坐标不同：

```bash
yolo-xx-paired-ab build \
  --snapshot-dir data/manual_short_preholdout_15m \
  --out datasets/owner_short_paired_ab_fixture8 \
  --split-at 2026-02-15T00:00:00Z \
  --right-contexts 0,8,16,24 \
  --seed 20260804 --max-positive-anchors 8

# fixture 通过后执行冻结的 full build
yolo-xx-paired-ab build \
  --snapshot-dir data/manual_short_preholdout_15m \
  --out datasets/owner_short_paired_ab_v2 \
  --split-at 2026-02-15T00:00:00Z \
  --right-contexts 0,8,16,24 \
  --seed 20260804

yolo-xx-paired-ab audit --pair-root datasets/owner_short_paired_ab_v2
yolo-xx-paired-gallery \
  --pair-root datasets/owner_short_paired_ab_v2 \
  --out reports/owner_short_paired_ab_v2_sample24
```

训练默认显式锁定 `deterministic=true`、`amp=true`、`workers=4`、`cache=false`，适配 12GB
RTX 3060 + 16GB 主机内存。`train.py` 会生成一个忽略 dataset/name/project 的 comparison contract；
两臂 contract hash 不相同就不是单变量实验。

Windows 只作为可清空的拟合 worker。Mac 先完成完整 source snapshot 审计，再生成逐图/逐标签
SHA-256 的 portable receipt；远端必须拿命令行单独传入的 receipt hash 重验 payload，因而无需
复制 287MB OHLCV 快照：

```bash
yolo-xx-portable create \
  --data datasets/owner_short_paired_ab_v2/w200/data.yaml \
  --out datasets/owner_short_paired_ab_v2/w200/portable_receipt.json

YOLO_XX_3060_HOST=zzc@CURRENT_IP \
  bash scripts/train_paired_ab_on_3060.sh --check
YOLO_XX_3060_HOST=zzc@CURRENT_IP \
  bash scripts/train_paired_ab_on_3060.sh
```

启动器使用 detached WMI 顺序训练 w200、w96，SSH 断开不会杀进程；它没有 holdout、ACTIVE、
部署或交易入口。完整冻结协议见
[docs/WINDOW_AB_PREREG_20260804.md](docs/WINDOW_AB_PREREG_20260804.md)。

训练完成后，可把本地缓存裁成严格 pre-holdout 的 1m/2m/3m/5m immutable snapshot，生成
两臂共用终点的离线扫描集，再以预注册的固定阈值批量推理：

```bash
yolo-xx-micro-snapshot \
  --cache-dir /path/to/local/kline_cache \
  --out data/micro_preholdout/5m --timeframe 5m
yolo-xx-scan-set build \
  --snapshot-dir data/micro_preholdout/5m \
  --out datasets/micro_scan_preholdout_v1/5m --max-images 512

YOLO_XX_3060_HOST=zzc@CURRENT_IP \
  bash scripts/scan_micro_on_3060.sh

# 拉回完成结果后生成机器审计、技术报告和原始预测框画廊
yolo-xx-scan-report \
  --scan-results reports/micro_scan_preholdout_v1 \
  --scan-sets datasets/micro_scan_preholdout_v1 \
  --runs-root runs/detect \
  --training-contracts reports/training \
  --out reports/micro_scan_preholdout_v1
```

扫描固定使用 `conf=0.30`、`iou=0.70`，只输出检测数量、置信度、原始框位置和抽样叠框图；
它没有收益标签、判断层、holdout、ACTIVE、部署或交易入口。

## 测试

```bash
.venv/bin/pytest
```

测试包含端到端小样本数据集构建，以及 AST 作用域守卫；后者会拒绝父项目、判断层、交易、
部署和网络依赖进入本项目。

迁移来源与改动见 [docs/EXTRACTION_MAP.md](docs/EXTRACTION_MAP.md)。

## 恢复 owner 只做空原始框

`manual-short` 流程用于恢复旧 `dense_owner_side_short`：只保留 owner 明确标为 short 的框，
不使用盘口重裁框，也不读取 holdout。第一步把父项目当前 CSV 中截止线以前的连续 OHLCV 前缀
复制成独立、带哈希的快照；循环只检查第一条边界时间，绝不解析或写出边界后的 OHLCV：

```bash
yolo-xx-manual-short snapshot \
  --review-sheet /Users/zhangzc/fable-trading/analysis/output/owner_side_review/review_sheet.csv \
  --source-dir /Users/zhangzc/fable-trading/data/kline_fetched \
  --out data/manual_short_preholdout_15m \
  --dry-run

# 核对计划后去掉 --dry-run
```

随后分别恢复 200 根原图语义基线，以及 96 根短窗实验版：

```bash
yolo-xx-manual-short build \
  --snapshot-dir data/manual_short_preholdout_15m \
  --out datasets/owner_short_original_w200 \
  --layout original --window 200 --dry-run

yolo-xx-manual-short build \
  --snapshot-dir data/manual_short_preholdout_15m \
  --out datasets/owner_short_staggered_w96 \
  --layout staggered_causal --window 96 \
  --right-contexts 0,8,16,24 --dry-run
```

96 根不是把完整形态的结束时间凭空提前，而是把每根 K 线的横向像素提高约 2.1 倍。框右侧
上下文按 `box_id` 确定性散列到 0/8/16/24 根，避免所有正框固定贴最右侧。manifest 仍把
`available_at` 记为整张输入图最后一根 K 线的闭合时间，不能拿 `box_end_time` 冒充实时可用时间。
恢复集只有正图；在加入经审计、位置匹配的负样本前，不得把它称为 precision 可用的训练集。
