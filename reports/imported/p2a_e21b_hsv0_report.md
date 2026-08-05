# P2a E2.1b 全 HSV 关闭正式验收

## 结论

E2.1b 于 `2026-07-11 01:42:07 CST` 自然结束，exit 0，完成 40/40 epochs。训练配置
满足铁律：`hsv_h/s/v=0`、flips/mosaic/mixup/copy_paste=0；没有 NaN、traceback、恢复
训练或 holdout 读取。

正式 `best.pt` 复验 mAP50 为 `0.8505`，未达到 `0.90`；一致率为 `51.27%`，未达到
`95%`。因此 E2.1b **验收失败，不晋升、不继续相同配方重跑**。

## 配置与数据

| 项目 | 值 |
|---|---|
| model | yolo11s |
| dataset | dense_15m_full，train 5,805 / val 1,255 images |
| val instances | 1,297 |
| epochs / batch / imgsz | 40 / 8 / 960 |
| patience / device | 12 / MPS |
| label contract | MAX_DENSE_BARS=12，X_PAD_PX=6 |
| augment | HSV、flip、mosaic、mixup、copy_paste 全 0；translate=0.02，scale=0.1 |
| duration | 47,147.9 秒，约 13.10 小时 |

## 指标

| 指标 | E2.1 | E2.1b HSV0 | 变化 |
|---|---:|---:|---:|
| mAP50 | 0.8503 | 0.8505 | +0.0002 |
| mAP50-95 | 0.6655 | 0.6622 | -0.0033 |
| precision | 0.8106 | 0.7614 | -0.0492 |
| recall | 0.7047 | 0.7405 | +0.0358 |

`results.csv` 的最高 mAP50 是 epoch 30 的 `0.85080`；epoch 40 为 `0.85078`，但其
mAP50-95 更高。Ultralytics 保存的 `best.pt` 独立复验结果为上表正式值。全 HSV 关闭只
改变了 precision/recall 平衡，没有改善 mAP50，也没有填补到 0.90 的约 4.95 个百分点。

## 一致率

按固定 `conf=0.30`、IoU≥0.5、one-to-one greedy 口径导出 val 预测：

| 项目 | 数量/值 |
|---|---:|
| val images | 1,255 |
| GT boxes | 1,297 |
| predicted boxes | 1,629 |
| IoU50 matched | 665 |
| match rate vs GT | 0.5127 |
| precision-like | 0.4082 |
| gate ≥0.95 | false |

mAP 与该一致率不是同一指标：mAP 对置信度排序积分，一致率固定在 conf 0.30 且要求每个
GT 一对一命中。即便如此，51.27% 明确说明当前模型不能承担 95% 规则替身目标。

## 训练曲线风险

前 24 个 epoch 多次出现 mAP50 接近 0 的剧烈波动，25 轮以后才稳定在约 0.83-0.85。
曲线没有非有限值，最终模型可加载、可完成全量 val 推理；但 MPS 训练稳定性和小目标/框
边界不一致仍是风险。继续同配方追加 epoch 没有证据价值。

## 风险与诚实声明

- E2.1b 是检测层，不预测 long/short/no_trade，也不能证明交易盈利。
- 一致率只对当前 auto-label GT；GT 本身的人工抽样仍有不准项。
- 本轮没有读取 judgment holdout，没有改 ACTIVE、LightGBM 阈值、TP/SL 或成本。
- 结果失败仍完整记录；不做 result-dependent 第二轮检测训练。

## 下一步

1. E2.1b 保留为失败但合规的检测证据，不晋升 95% 替身。
2. 等因果方形 long/short/no_trade 数据集完成，按预注册配置只训练一次方向分类器。
3. SAHI 可作为独立推理实验，但不能和模型大小、标签或阈值同时改；在盈利主线中优先级
   低于方向与经济性验证。

## 复现

```bash
python3 scripts/export_yolo_preds_for_audit.py \
  --dataset datasets/dense_15m_full --split val --conf 0.30 \
  --weights runs/detect/runs/detect/dense_15m_full_s_e21b_hsv0/weights/best.pt \
  --out datasets/dense_15m_full/preds_val_e21b_hsv0_best

python3 -m src.detection.consistency_check \
  --dataset datasets/dense_15m_full --split val \
  --preds datasets/dense_15m_full/preds_val_e21b_hsv0_best \
  --out analysis/output/consistency_e21b_hsv0_vs_gt.json
```
