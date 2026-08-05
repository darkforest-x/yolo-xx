# 多周期 YOLO 检测准备审计 — 2026-08-04

## 结论

多周期检测的工程入口已经按“相同物理时长、每周期独立模型、候选只在完整窗口闭合后可用”设计，但本轮**没有训练，也没有产生任何收益结论**。当前唯一硬阻塞是：尚无带不可变清单、物理上截止于 `2026-05-04T00:00:00Z` 之前的 1m/2m/3m/5m OHLCV 快照。现存混合 CSV 不得通过读取后截断来伪装成 pre-holdout 数据。

因此，下一次真实实验应从 5m 小样本开始；在安全快照到位前，训练门保持关闭。

## 已核验的历史基线

| 项目 | 可核验事实 | 本轮解释 |
|---|---|---|
| `dense_owner_short_star_tip_v10` | 3,305 张：1,322 正样本 / 1,983 负样本；正框右边界集中在 `x≈0.9969` | 数据可作 dense-cluster 语义参考，但存在明显右缘捷径风险 |
| v16 tipuni | 真 tip 命中 3/9；空背景误火 17/33 | 已拒绝，不作为新实验成功先验 |
| ETH 3m pilot v1 | 严格 OOS 774 根中开火 772 根（99.74%）；模型相对匹配随机超额 -0.80bp | 已拒绝，只保留机械链路与失败经验 |
| `owner_v11_chain.pt` | 当前项目、迁移目录及历史 Git 中未找到原权重 | 不能声称复用原 PF6.61 检测器 |

上述失败不推出“小周期检测必然无效”，但足以禁止继承旧 PF、mAP 或单张画廊作为成功结论。

## 同物理时长检测规格

默认完整窗口固定为 3,000 分钟，MA 与 dense 区间也按物理分钟换算。一个周期一个模型，不混合训练。

| detector 周期 | 窗口 bars | MA bars | dense bars | warmup bars | 顺序 |
|---|---:|---|---|---:|---:|
| 5m | 600 | 60 / 180 / 360 | 15–36 | 1,080 | 1 |
| 3m | 1,000 | 100 / 300 / 600 | 25–60 | 1,800 | 2 |
| 2m | 1,500 | 150 / 450 / 900 | 38–90 | 2,700 | 3 |
| 1m | 3,000 | 300 / 900 / 1,800 | 75–180 | 5,400 | 4 |

1m/2m 在固定 1,280px 画布上会出现多根 bar 共用横向像素的风险，必须晚于 5m/3m 才测试。

## 安全门

- 非 dry-run 构建必须提供不可变 source snapshot manifest；解析 CSV 前先核对清单声明、文件 stat 与 SHA-256，解析后再核对行数及首尾时间，任一不一致都拒绝继续。
- snapshot 的 `cutoff_exclusive`、每个文件最后一根 bar 的闭合时间及构建 `end_before` 均不得越过 holdout 起点。
- train/val 使用一个全局 UTC 切点；跨切点完整窗口丢弃，并要求 `max(train.available_at) < min(val.window_start_time)`。
- dataset manifest 保存 source/image/label SHA-256、窗口绝对时间及 `available_at=window_end_close_time`；强审计是训练和评估的前置门。
- 预测结果按模型返回的真实 `result.path` 绑定输入，不依赖返回顺序；候选携带权重、dataset manifest、source 和 image 身份。
- flip、mosaic、mixup、HSV、平移、缩放等会破坏时间或 K 线语义的增强全部关闭；Ultralytics 固定为 `8.4.89`。

## 真实权重机械烟测

- 图：`datasets/eth_3m_short_pilot_v1/images/val/pos_eth3m_20260427T145400Z_t032.png`
- 权重：`weights/baselines/eth3m_short_pilot_v1_mac_cold.pt`
- 权重 SHA-256：`7f1f3b2d6300952ac5c4adb3937c0b3b2e9d7879b73ed2855bcf08b55cbe202e`
- 输出：1 个 detection，confidence `0.90257198`，原始 `xywhn=[0.98154277,0.68235797,0.03260221,0.04087608]`

该框位于窗口右缘，与已知失败 pilot 一致。它只证明真实 Ultralytics 权重→原始归一化框→manifest 的机械路径可运行，不证明定位正确或有正期望。

该旧 pilot 输出的 `predictions.json` 是 schema v1，不含新 schema-v2 所要求的 dataset/source/`available_at` 身份链；它没有进入 L2 bridge，因此不是“真实权重 → schema-v2 → L2”端到端验收。

## 测试与复现

最终真实仓验证：

```bash
cd /Users/zhangzc/yolo-xx
PYTHONDONTWRITEBYTECODE=1 /Users/zhangzc/fable-trading/.venv/bin/python \
  -m pytest -q -p no:cacheprovider
# 38 passed

PYTHONPATH=src /Users/zhangzc/fable-trading/.venv/bin/python -m yolo_xx.dataset \
  --cache-dir /private/tmp/nonexistent-safe-5m \
  --out /private/tmp/nonexistent-yoloxx-5m --timeframe 5m --dry-run

PYTHONPATH=src /Users/zhangzc/fable-trading/.venv/bin/python -m yolo_xx.predict \
  --weights weights/baselines/eth3m_short_pilot_v1_mac_cold.pt \
  --source datasets/eth_3m_short_pilot_v1/images/val/pos_eth3m_20260427T145400Z_t032.png \
  --out /private/tmp/yoloxx-final-smoke.rVo0BO/predictions --device cpu --conf 0.25 --save-overlays
```

1m/2m/3m/5m dry-run 全部通过且未读取 manifest/CSV、未创建 dataset；Ultralytics `8.4.89` 配置解析通过；模块 `py_compile` 通过。真实权重首次烟测还发现：把路径列表传给 Ultralytics 会令 `result.path` 变成合成的 `image0.jpg`。预测器已改为每张图独立调用，强制每次恰好一个结果且返回路径等于该输入，因而不依赖批量顺序。修复后真实烟测通过，`predictions.json` SHA-256 为 `881c047269c22bd7a8cc71c293429303838df0784c616d096c2285d8275c3902`。

没有安全 source snapshot 时不得执行 full build 或 train。

实现已提交到 `yolo-xx` 的 `main`：`97ad14ee46000cf0cba03785e1316b3ef91d3865`；未 push。

## 下一次实验的唯一入口

1. 在 YOLO 项目外产生物理 pre-holdout、不可变的 5m OHLCV 快照与清单；不得读取现有混合文件后截断。
2. 先跑 fixture，再用极小币种/小样本 5m dry build，机器审计通过后才允许 full build。
3. 只用自动 dense-rule labels，不要求 owner 人工盘口打标。
4. 先判断检测是否稀疏、是否跨位置工作、相对匹配随机是否有增量；未过门不进入 3m。
5. 5m 过门后依次 3m、2m、1m，每次只改 detector timeframe。

## 风险与诚实声明

- 旧 PF6.61 属于历史 YOLO + 判断层整链，且 full-window box-time 回填已被证伪；不能作为本项目先验。
- 旧 3m 模型和 v16 均失败；本轮未生成新模型。
- 重叠完整窗口可能反复检测同一旧框；正式滚动实验前还需要 first-seen/dedupe 协议。
- 当前没有可安全读取的小周期源数据，因此没有候选数、正类率、val mAP、收益、PF 或显著性可报告；任何此类数字均为 N/A。
- 按本轮执行记录：未读 holdout、未训练、未调阈值、未改 ACTIVE、未部署、未下单、未 push。可见工作区没有新增权重/run/ACTIVE 变更作为旁证，但这不是对整台主机历史行为的完备审计。
