# yolo-xx

`yolo-xx` 是从 `fable-trading` 检测层抽出的独立离线项目，只做一件事：准备、训练和验证
YOLO 图像检测模型。

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

## 测试

```bash
.venv/bin/pytest
```

测试包含端到端小样本数据集构建，以及 AST 作用域守卫；后者会拒绝父项目、判断层、交易、
部署和网络依赖进入本项目。

迁移来源与改动见 [docs/EXTRACTION_MAP.md](docs/EXTRACTION_MAP.md)。
