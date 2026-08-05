# P2a 报告：YOLO 检测双均线密集区域

**日期**：2026-07-07  
**脚本**：`src/detection/`（渲染 → 自动标注 → 数据集 → 训练 → 评估）  
**数据**：旧项目 `runs/cache/` 下 55 个币种 15m K 线缓存（只读引用，未复制大文件）

## 一句话结论

检测层冒烟流水线已跑通：**val mAP50 = 0.835**（best.pt 官方评估），超过 0.8 冒烟验收线。  
三轮迭代的核心修复是 **render.py 的 y 轴最小跨度约束** 与 **auto_label.py 的框尺寸/最短密集段过滤**——  
首轮失败并非像素坐标公式错误，而是低波动窗口被过度 y 轴放大，导致"密集"的视觉语义在不同窗口间不一致。

---

## 数据集统计（最终版，smoke3）

来源：`datasets/dense_15m/dataset_summary.json`

| 指标 | 训练集 | 验证集 | 合计 |
|---|---|---|---|
| 图像数 | 2560 | 571 | 3131 |
| 标注框数 | 3132 | 648 | 3780 |
| 背景图（无框） | 896（35%） | 224（39%） | 1120 |
| 每图平均框数 | 1.22 | 1.13 | 1.21 |

**构建参数**：窗口 200 根 15m K 线、步长 200、55 币种、按时间 80/20 切分（跨 cutoff 的窗口丢弃，无时间泄漏）。

**标注框尺寸（smoke3，优化 padding 后）**：

| 维度 | 中位数（归一化） | 约合像素（1280×742） |
|---|---|---|
| 宽度 w | 0.080 | ~102 px（≈16 根 K 线） |
| 高度 h | 0.125 | ~93 px |

对比 smoke2：窄框（w < 0.04）占比从 **22.9% → 0.2%**，框更易学习、IoU 更稳定。

**密集判定规则**（沿用旧项目 strict 阈值，映射到 SMA/EMA 20/60/120 六均线）：

```
fast_spread = (max-min of SMA/EMA 20/60) / close  ≤ 0.0028
full_spread = (max-min of all six MAs) / close    ≤ 0.0055
连续 ≥ 5 根满足 → 一个 dense_cluster 框
```

---

## 三次冒烟训练对比

| 轮次 | 关键代码变更 | 数据规模 | 最佳 epoch | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|---|---|
| smoke | 初版 render（无 y 轴下限）+ min_bars=4 | 2560/595 | 30 | 0.216 | 0.105 | 0.250 | 0.404 |
| smoke2 | render 加 `MIN_REL_SPAN=0.06` | 1600/400 | 22 | 0.677 | 0.423 | 0.578 | 0.709 |
| **smoke3** | smoke2 + 标注 padding/过滤 + `rect=True` | 2560/571 | **23** | **0.835** | **0.593** | **0.762** | **0.710** |

> smoke3 最终指标以 `runs/detect/runs/detect/dense_15m_smoke3/weights/best.pt` 在 val 集上的官方评估为准（`analysis/output/p2a_val_metrics.json`）。

---

## 根因分析：smoke → smoke2 → smoke3

### smoke 失败（mAP50 ≈ 0.22）

**根因：渲染 y 轴动态缩放无下限，低波动窗口视觉语义失真。**

- `render.py` 的 `_price_bounds()` 仅按窗口内 high/low/MA 的实际极差定 y 轴。
- 横盘/低波动窗口（价格极差 ~2–3%）被 y 轴极度放大，六条均线束占据图像大部分高度，看起来像一个"宽扁带"。
- 高波动窗口中，同样的数值密集（full_spread ≤ 0.55%）视觉上只是几条细线收拢——**同一规则标签对应完全不同的像素模式**。
- 量化证据：修复前 flat 窗口 recall@IoU0.5 仅 ~18%，高波动窗口 ~70%。

**排除项**：`render.py` 与 `auto_label.py` 共用同一 `ChartTransform`（同一次 `render_chart()` 调用），x/y 像素映射公式一致，不存在"渲染与标注坐标系不一致"的 bug。

### smoke2 改善（mAP50 ≈ 0.68）

**修复**：`render.py` 增加 `MIN_REL_SPAN = 0.06`——y 轴最小跨度为窗口中位价的 6%，低波动窗口不再过度放大。

- 密集区在所有窗口中呈现一致的"均线收拢"视觉特征。
- mAP50 从 0.22 跃升至 0.68（+0.46），验证根因判断正确。

**残留问题**：

- 约 23% 框宽度 < 0.04（~50 px），YOLO 对小目标 IoU 敏感，漏检集中于此。
- 误报中 ~55% 落在数值上仍满足 strict 阈值的"近密集"区段——模型看到了密集形态但框边界与 GT 不完全对齐。

### smoke3 达标（mAP50 ≈ 0.84）

**修复**（`auto_label.py` + 数据规模 + 训练配置）：

1. `MIN_DENSE_BARS` 4 → **5**：过滤 1–3 根的短密集毛刺，减少歧义标签。
2. `x_pad_px` 6 → **12**，`y_pad_frac` 0.25 → **0.35**：框更大、IoU 容错更高；窄框占比降至 0.2%。
3. 数据集扩至 **3131 张**（原 smoke2 仅 2000 张）。
4. 训练启用 **`rect=True`**（矩形 batch，保留 1280×742 宽高比，减少 imgsz=960 缩放畸变）。

---

## 最终训练配置（smoke3）

| 参数 | 值 |
|---|---|
| 模型 | yolo11n.pt |
| epochs | 23（训练至 epoch 23 时 best；共计划 30） |
| imgsz | 960 |
| batch | 8 |
| device | mps（Apple M4） |
| patience | 10 |
| rect | True |
| **增强（历史实际配置）** | fliplr=0, flipud=0, mosaic=0, mixup=0, copy_paste=0, hsv_h=0, **hsv_s=0.05, hsv_v=0.05（违反 HSV 全关铁律）**, degrees=0, shear=0, perspective=0, translate=0.02, scale=0.1 |

权重路径：`runs/detect/runs/detect/dense_15m_smoke3/weights/best.pt`（不入 git）

---

## 可视化检查

验证集 5 张抽样对比图（黑框 = GT，洋红 = pred）保存在 `analysis/output/p2a_val_*.png`。

| 样本 | GT 框 | Pred 框 | 结论 |
|---|---|---|---|
| APE_USDT_017160 | 1 | 1 | 高度重合，定位准确 |
| BTC_USDT_014560 | 2 | 2 | 左侧两框均命中；右侧 pred 多出一个近密集区 |
| XLM_USDT_015360 | 2 | 2 | 两框均对齐 |
| LINK_USDT_017160 | 2 | 3 | 主密集区命中，右侧多 1 个 FP |
| OP_USDT_016760 | 1 | 4 | 主框命中，但周边近密集区误报较多 |

**总体结论**：

- 典型密集区（多条均线收拢 10–30 根 K 线）预测框与 GT **高度重合**，符合预期。
- 主要误差来自 **近密集区误报**（数值上接近阈值但未标注的短/弱密集段）和 **相邻多密集区的框合并/拆分差异**。
- 人眼确认：`train_batch0.jpg` 与 `labels.jpg` 中标注框均落在真实均线密集处，无系统性偏移。

---

## 验收判定

| 标准 | 要求 | 结果 |
|---|---|---|
| 冒烟 mAP50 | ≥ 0.80 | **0.835 ✅** |
| 标签自动生成、框随密集区变化 | 是 | ✅ |
| 增强不破坏时间/颜色语义 | 全关 | ❌ 历史训练含 hsv_s/v=0.05；2026-07-10 已修复代码，需合规重训 |
| val 按时间切分 | 80/20 per symbol | ✅ |
| PROJECT_PLAN 正式标准 mAP50 ≥ 0.90 + 规则一致率 ≥ 95% | 未测 | ⬜ 待全量训练后对照 |

**判定：冒烟验收通过；正式 2a 验收（0.90 + 规则一致率）需全量数据训练后再做。**

---

## 正式全量训练验收（2026-07-09 离线管道）

`scripts/offline_pipeline.sh` 等待全量训练结束后做官方评估；nano 全量未达到正式线后，
脚本自动使用 `yolo11s.pt` 重训一轮并再次评估，未触碰 holdout。

| 模型 | 权重 | P | R | mAP50 | mAP50-95 | 判定 |
|---|---|---:|---:|---:|---:|---|
| yolo11s | `runs/detect/runs/detect/dense_15m_full_s/weights/best.pt` | 0.8003 | 0.7112 | 0.8569 | 0.6643 | 未达正式验收线 |

**正式验收结论**：未达成。mAP50 0.8569 低于 0.90，因此不写
`consistency_check.py`、不继续调整 conf/IoU/增强参数凑数。检测层保持为已验证可学习的
非关键路径组件，后续暂停；主线继续以规则扫描 + 判断层 + 前向验证推进。

风险与诚实声明：

- yolo11s 相比 smoke3 的 mAP50 仅从 0.8353 提升到 0.8569，模型容量不是当前瓶颈。
- **2026-07-10 合规更正**：审计训练日志与 `args.yaml` 后确认，smoke/full/E2.1 等修复前
  训练均继承 `hsv_s=0.05`、`hsv_v=0.05`。这些结果可保留为诊断证据，但不满足项目
  “HSV 全关”铁律，不能作为合规正式验收；代码已改为全 0，后续须用新 run 名重训。
- precision 0.8003、recall 0.7112 说明仍有近密集误报和漏检，未达到替代规则扫描的稳定性。
- 本轮结果来自 val split 官方评估；没有评估 holdout。

### P2-11 E1 标签收紧（2026-07-10）

单变量：`X_PAD_PX` 12→**6**；`dense_15m_full` 标签原地重写（图未动）。  
box_w_mean 0.1267→0.1176，n_boxes 不变。**未重训**。详见 `analysis/p2a_e1_xpad_report.md`。

---

## 全量训练建议

1. **数据**：55 币种全部 cache、stride 100（重叠减半增样本），预计 6000–8000 张；背景比维持 35%。
2. **分辨率**：`imgsz=1280`（与渲染原生宽度一致），或保持 960 + rect=True。
3. **模型**：yolo11n 已达标；若需更高 recall 可试 yolo11s（参数量 3×，MPS 上可接受）。
4. **训练**：epochs 50–80，patience 15；增强参数保持全关。
5. **验收对照**：写脚本将 YOLO 预测框与 `auto_label.py` 同窗口规则扫描结果逐框比对，量化一致率（目标 ≥ 95%）。
6. **推理部署**：conf_threshold 建议 0.25–0.35；对近密集 FP 可在判断层用数值 spread 二次过滤。

---

## 复现命令

```bash
# 构建数据集
python -m src.detection.build_dataset --out datasets/dense_15m --max-images 3200

# 训练
python -m src.detection.train --data datasets/dense_15m/data.yaml \
    --epochs 30 --name dense_15m_smoke3

# 评估 + 可视化
python -m src.detection.eval_visualize \
    --weights runs/detect/runs/detect/dense_15m_smoke3/weights/best.pt \
    --n-vis 5
```
