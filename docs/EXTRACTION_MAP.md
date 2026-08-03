# Extraction map

`yolo-xx` 在 `fable-trading` 的 2026-08-03 `main` 上建立。它是可独立安装的副本，不是
父项目目录的转发壳，也不依赖父项目运行时。

| 新模块 | 原始职责 | 抽离时的处理 |
|---|---|---|
| `yolo_xx.data` | `src/detection/data.py` | 移除旧机器绝对路径；输入目录必须显式传入 |
| `yolo_xx.render` | `src/detection/render.py` | 保留画布、颜色和坐标语义，改为包内依赖 |
| `yolo_xx.labels` | `src/detection/auto_label.py` | 保留密集段和 YOLO box 规则，改为包内依赖 |
| `yolo_xx.dataset` | `src/detection/build_dataset.py` | 增加时间上界、空输出目录保护和可核验摘要 |
| `yolo_xx.train` | `src/detection/train.py` | 保留安全增强和续训学习率，增加纯 dry-run |
| `yolo_xx.evaluate` | `src/detection/eval_visualize.py` | 只保留离线 val 指标，不带业务裁决或晋升 |
| `yolo_xx.consistency` | `src/detection/consistency_check.py` | 保留 GT/pred 一对一 IoU 一致性检查 |
| `yolo_xx.audit` | 新增 | 独立检查 YOLO 目录、标签格式和配对完整性 |

明确没有抽取 `src/judgment`、`src/backtest`、`src/execution`、`src/webapp`、数据抓取、
实时扫描、模型晋升或部署脚本。
