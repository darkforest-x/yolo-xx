# ETH 3m 做空检测器 pilot v1 — 数据质量与训练启动记录

日期：2026-07-29

## 一句话结论

数据链路通过了结构性检查，但 **pilot 最终验收失败**：连续严格 OOS 在 774 根 eligible bars 中开火 772 根（99.74%），没有形成稀疏事件。最终训练池为 183 张（76 正 / 107 负），已按事件做严格时间切分。3060 队列在 owner 确认后取消，改由 Mac M4 MPS 训练；未读取 holdout、未 promote、未写 ACTIVE。

## 当前状态

| 项目 | 状态 |
|---|---|
| 数据集 | `datasets/eth_3m_short_pilot_v1` |
| 远端数据 | `C:/fable/datasets/eth_3m_short_pilot_v1` |
| 训练名 | `eth3m_short_pilot_v1_mac_cold` |
| Mac 训练进程 | PID 70152 已正常结束，MPS / Apple M4 |
| 3060 队列 | PID 96316 已精确取消；v10 扫描 PID 93656 保留 |
| 本地产物 | `runs/detect/runs/detect/eth3m_short_pilot_v1_mac_cold` |
| 当前结果 | 45/100 触发 patience=20 早停；总耗时 13.76 分钟 |

## 数据定义

正例来自 project 53 中 owner 判断“是”的 93 张：

1. 在原 owner 认可的 v10 框内，从原始 3m OHLC 逐 bar 重算“第一根完整收盘跌到六条 MA 下方”；
2. 只保留比原 v10 至少提前 2 根的 90 张；
3. 相同提前锚点去重 14 张，保留原框因果跨度更完整者，得到 76 个正例；
4. 每张只渲染到该锚点为止的 200 根，框右缘固定为当前 tip；
5. 框宽保留原框在锚点前的因果部分，并限制为 5–12 根；框高只由当时可见的六条 MA 计算。

负例仅使用 owner 亲自判断“不是”的 107 张，沿用原 v10 因果时点并写空 YOLO 标签。没有用规则补未经人工确认的 easy negative。

未来收益、未来最大跌幅、3h 结果等列没有参与样本选择、图片、框或 split。30 张提前时机校准中 owner 判断 30/30 来得及，但这只是分层校准样本，不等于 93 张最终框逐张金标。

## 数据统计

| split | 事件 | 正例 | 负例 | 合计 |
|---|---:|---:|---:|---:|
| train | 77 | 60 | 75 | 135 |
| val | 26 | 16 | 32 | 48 |
| 总计 | 103 | 76 | 107 | 183 |

- 时间范围：2026-03-14 06:39 UTC ～ 2026-05-01 20:18 UTC。
- 时间切点：2026-04-19 11:45 UTC；train 最大时点 07:18，val 最小时点 16:12。
- 事件定义：相邻锚点间隔大于 60 分钟才开启新事件；同一事件不得跨 split。
- 框跨度：5 根 62 张、6 根 2 张、7 根 4 张、8 根 2 张、9 根 2 张、10 根 3 张、11 根 1 张。
- 图片/标签：183 / 183；正标签 76，空标签 107；全部 1280×742。
- 图片 SHA-256 重复组：0；正负精确锚点冲突：0。
- holdout 起点：2026-05-04 00:00 UTC；本轮未消费 holdout。

## 训练配置

这是单一 cold-start pilot，不混入 v10 权重，避免把 v10 的“确认过晚”偏差直接继承：

- base：`models/yolo11n.pt`
- epochs：100
- patience：20
- imgsz：960
- batch：8
- device：CUDA 0
- workers：4
- cache：false
- finetune：false（COCO cold start）
- fliplr / flipud / mosaic / mixup / copy_paste：0
- hsv_h / hsv_s / hsv_v：0
- 不执行 frozen-eval、holdout、promote 或 ACTIVE 切换

## 复现命令

```bash
env PYTHONPATH=. .venv/bin/python scripts/build_eth3m_short_pilot_dataset.py
env PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_build_eth3m_short_pilot_dataset.py \
  tests/test_detection_train_speed_knobs.py \
  tests/test_auto_label_padding.py
bash scripts/train_eth3m_short_pilot_on_3060.sh --check
bash scripts/train_eth3m_short_pilot_on_3060.sh --wait-pid 93656
bash scripts/train_eth3m_short_pilot_on_3060.sh --status \
  --name eth3m_short_pilot_v1_cold

# owner 确认改用 Mac 后：
env PYTHONPATH=. .venv/bin/python -u -m src.detection.train \
  --data datasets/eth_3m_short_pilot_v1/data.yaml \
  --model models/yolo11n.pt --epochs 100 --patience 20 \
  --imgsz 960 --batch 8 --device mps --workers 6 --cache disk \
  --no-finetune --name eth3m_short_pilot_v1_mac_cold
```

## 结果表

| 模型 | 状态 | val P/R/mAP50 | 开火密度 | 经济性 |
|---|---|---|---|---|
| eth3m_short_pilot_v1_mac_cold | 45 轮早停 | best.pt：P 0.729 / R 0.675 / mAP50 0.735 / mAP50-95 0.443 | **FAIL：strict OOS 99.74%，去重 26.67 笔/有效日** | **FAIL：模型 -31.97bp；匹配随机 -31.17bp；超额 -0.80bp** |

曲线的单项峰值分别为：epoch 27 mAP50 0.880（P 0.922 / R 0.740），epoch 25 mAP50-95 0.455。最终重新加载 `best.pt` 的验证结果为 P 0.729、R 0.675、mAP50 0.735、mAP50-95 0.443。val 只有 16 个正例，因此两组数字都只能说明模型已学到目标，不能作为泛化或晋升证据。

检测器内部 val 指标只用于确认是否学到目标，不作晋升裁决。逐 bar causal 扫描已经证明静态 mAP 没有转化为连续盘口选择力，因此 v1 停止，不进入判断层或 holdout。完整协议与结果见 `analysis/p_eth_3m_short_pilot_v1_backtest.md`。

## 风险与诚实声明

1. 76 个唯一正例确实偏少，val 只有 16 个正例，单次 mAP 方差会很大。
2. 200 张都先由 v10 候选池筛出，仍有选择偏差；当前模型学不到 v10 从未提出过的形态。
3. owner 确认了 93 张的“形态是/不是”，并确认分层 30 张提前线都来得及；最终 5–12 根紧框是因果重建，不是 76 张逐框人工金标。
4. 因此这轮的目标只是回答“这个口径能否学到、是否明显降低晚开火”，不能直接进入实盘。

## 训练完成后的验收顺序

1. 看训练曲线是否健康，best 是否落在预热前两轮；若是，按失败处理。
2. 报 val P、R、mAP50 与正负混淆，但明确 16 个 val 正例的置信区间风险。
3. 在完全未参与训练的 pre-holdout 时间块逐 bar 扫描，统计每日电火数、重复框、tip/tip-1/tip-2 命中和迟到程度。
4. 抽取模型有框的图交给 owner 看“是不是、是否来得及”；不再让人工判断不可见的盘口密度。
5. 只有检测层通过，才构造 3h 未来结果标签与判断层；holdout 仍保持封存。
