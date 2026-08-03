# yolo-xx

`yolo-xx` 是从 `fable-trading` 检测层抽出的独立离线项目，只做一件事：准备、训练和验证
YOLO 图像检测模型。

## 边界

包含：

- 从本地 OHLCV CSV 计算 SMA/EMA 20、60、120；
- 渲染 K 线和六条均线；
- 生成/校验单类别 `dense_cluster` YOLO 数据集；
- 使用固定的语义安全增强配置训练 YOLO；
- 离线验证并输出 JSON 指标。

不包含：数据抓取、方向或收益判断、LightGBM、标签收益、回测、成本、交易所连接、实盘扫描、
订单、通知、看板、ACTIVE、模型晋升和部署。项目不导入父仓库的 `src` 或其他业务包。

本次抽离只复制代码，不复制或移动父项目的数据、权重和运行结果，也不会自动开始训练。

## 安装

```bash
cd yolo-xx
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

## 从本地 CSV 构建数据集

输入 CSV 至少要有 `ts,open,high,low,close,volume`，`ts` 为毫秒时间戳；文件名建议为
`okx_BTC_USDT_SWAP_10000.csv`，最后一段是声明行数。构建器只读取显式指定的本地目录，
不会联网取数。

```bash
yolo-xx-build \
  --cache-dir /absolute/path/to/pre_holdout_csvs \
  --out datasets/dense_15m \
  --end-before 2026-05-04T00:00:00Z \
  --window 200 --stride 200 --max-images 3200

yolo-xx-audit --dataset datasets/dense_15m --out reports/dataset_audit.json
```

`--end-before` 是严格的时间上界，默认 `2026-05-04T00:00:00Z`。输出目录如已有图片或
标签，构建器会拒绝覆盖，防止混合两轮数据。

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

续训权重默认启用保守 AdamW 配方；`yolo11*.pt` 冷启动默认保留 Ultralytics 优化器选择。
翻转、mosaic、mixup、HSV 和其他会改变时间/颜色语义的增强始终关闭。

## 测试

```bash
.venv/bin/pytest
```

测试包含端到端小样本数据集构建，以及 AST 作用域守卫；后者会拒绝父项目、判断层、交易、
部署和网络依赖进入本项目。

迁移来源与改动见 [docs/EXTRACTION_MAP.md](docs/EXTRACTION_MAP.md)。
