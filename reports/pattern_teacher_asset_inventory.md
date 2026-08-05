# Pattern Teacher 资产清单 — owner_v10_chain

**Phase 1 / Task 1** · 2026-08-05 · 只读盘点，未训练、未改任何权重/标签/数据集/阈值

---

## 1. 标的与指纹

| 项 | 值 |
|---|---|
| 权重 | `fable-trading/models/owner_v10_chain.pt` |
| **SHA256** | `b9a84b5f5ebf0032dfa8ddf1ed1f12c19b7cc2d410a57480bd196d76cbc7d953` |
| 大小 | 18,185,754 B（18.3 MB） |
| 训练完成 | 2026-07-17 02:30（3060 本地时间） |
| 来源 | Windows 3060 `C:\fable\runs\detect\runs\detect\owner_v10_chain\weights\best.pt`，2026-08-05 取回 |
| 同指纹副本 | `C:\fable\base_hts.pt`（同一文件，后被用作 H-TS 实验起点） |
| 备份 | `fable-trading/models/archive_3060/owner_v10_chain/best.pt` |

指纹已双向核对：取回的文件与 3060 上的 `best.pt`、`base_hts.pt` 三者 SHA256 完全一致。

---

## 2. 血统链（由 args.yaml 的 `model:` 字段 + SHA256 逐级验证）

```
yolo11s.pt (COCO)
      ↓
owner_v7_chain.pt                    ← 本清单可追溯的最早祖先
      ↓  data = dense_owner_v7
owner_v8_chain      → base_v8.pt     SHA 86c4cce0…  frozen-F1 0.650
      ↓  data = dense_owner_v8
owner_v9_chain      → base_v9.pt     SHA 3a725696…  frozen-F1 0.627
      ↓  data = dense_owner_v9
owner_v10_chain     → base_hts.pt    SHA b9a84b5f…  frozen-F1 0.645   ★ Pattern Teacher
      ↓  data = dense_owner_v11（Mac 上训练）
owner_v11_chain                                     frozen-F1 0.658   ✗ 权重永久丢失
      ↓  data = dense_owner_v12_htip
owner_v12_htip                                      frozen-F1 0.650
      ↓
owner_v13 / v14_pad200 / v15_tipval / v16_tipuni_cold
```

**命名陷阱（记录以免后人误读）**：版本 N 训练在数据集 N−1 上。
`owner_v10_chain` 的训练集是 `dense_owner_v9`，不是 `dense_owner_v10`。

---

## 3. 训练配置（`args.yaml` 全量关键项）

| 类别 | 值 |
|---|---|
| 起点 | `model: C:\fable\base_v9.pt`（= owner_v9_chain/best.pt） |
| 数据 | `data: C:\fable\datasets\dense_owner_v9\data.yaml` |
| 规模 | imgsz 960 · batch 8 · epochs 40（patience 10）· rect true · device 0(3060) |
| 优化 | AdamW · lr0 1e-4 · lrf 0.01 · momentum 0.937 · weight_decay 5e-4 · warmup 0.5 |
| 其他 | amp true · seed 0 · deterministic true · single_cls false · cache false |
| 推理侧 | iou 0.7 · max_det 300 · conf null（运行时给） |

**增强（符合铁律 5，逐项核实全关）**：

```
fliplr 0.0   flipud 0.0   mosaic 0.0   mixup 0.0   cutmix 0.0
copy_paste 0.0   erasing 0.0   degrees 0.0   shear 0.0   perspective 0.0
hsv_h 0.0    bgr 0.0      auto_augment null    augment false
（仅存：hsv_s 0.05 · hsv_v 0.05 · scale 0.1 · translate 0.02）
```

**`lr0` 是显式指定的，不是 `optimizer='auto'`** —— 这一点重要，见 §6 已知污染。

---

## 4. 训练曲线

`results.csv`（16 epochs，40 上限 + patience 10 触发早停）：

| 指标 | 值 |
|---|---|
| best epoch | 16 / 16 |
| best mAP50 | 0.5178 |
| 前 5 轮 | 0.514 · 0.506 · 0.513 · 0.510 · 0.513 |
| 末 5 轮 | 0.514 · 0.516 · 0.510 · 0.513 · 0.518 |
| **全程波动范围** | **0.0123** |

第 1 轮 0.514，第 16 轮 0.518。

### 4.1 这个形状不是 v10 独有

对 3060 取回的 25 份 `results.csv` 做同一统计：

| 类型 | run | best mAP50 | 全程波动 |
|---|---|---|---|
| chain（续训） | owner_v8_chain | 0.3886 | **0.0204** |
| chain | owner_v9_chain | 0.5017 | **0.0116** |
| chain | **owner_v10_chain** | **0.5178** | **0.0123** |
| chain | owner_hts_chain | 0.5193 | 0.0181 |
| chain | owner_exemplar_chain | 0.3936 | 0.0458 |
| cold（COCO 起） | owner_v8_coco | 0.4075 | 0.4035 |
| cold | owner_v9_coco | 0.4313 | 0.4310 |
| cold | owner_v10_coco | 0.4852 | 0.4852 |

**全部 chain 系列波动 ≤0.046，全部 cold 系列波动 ≥0.21。**

---

## 5. 固定尺子（frozen-eval，47 币种从未参训 / 464 图）

从 git 历史中每次 promote 的 `models/owner_best.json` 提取：

| 日期 | 版本 | frozen-eval F1 |
|---|---|---|
| 07-14 | owner_v4 | 0.563 |
| 07-14 | owner_v5_from_v4 | ~~0.663~~（已判定泄漏，见 §6） |
| 07-15 | owner_v6_chain | 0.595 |
| 07-16 | owner_v8_chain | 0.650 |
| 07-16 | owner_v9_chain | 0.627 |
| 07-17 | **owner_v10_chain** | **0.645** |
| 07-18 | owner_v11_chain | 0.658 |
| 07-20 | owner_v12_htip | 0.650 |

v8 → v12 五个版本：**0.650 / 0.627 / 0.645 / 0.658 / 0.650，跨度 0.031，非单调。**

这五个版本之间发生过：round6/7/8/9 四批新标注（golden_pool 从约 4.5k 增长到 12,567 stem）、
四次训练、一次数据集重建。

### 5.1 两把尺子给出的结论方向不同

- 各版**自家 val mAP50**：0.389 → 0.502 → 0.518（看起来在涨）
- **固定尺子 frozen-F1**：0.650 → 0.627 → 0.645（没有涨）

原因：自家 val 随数据集换代而更换，**不同版本量的不是同一张考卷**；
frozen-eval 是唯一跨版本可比的口径。这与 `p2a_lr_bug_audit.md` 记录的
「两把尺子打架」（v8_coco 的 val mAP 高于 v8_chain，但 frozen-F1 相反）是同一现象。

**按纪律 12，frozen-F1 同样不作晋升裁决**（它也在有后文的分布上量）。
此处只用于版本间横向对照，不构成任何验收。

---

## 6. 已知污染与影响范围

| # | 问题 | 影响范围 | 对 v10 的影响 |
|---|---|---|---|
| 1 | **`optimizer='auto'` lr bug** | 修复前的全部 chain 模型在 epoch 3 被打飞，产出实为「底座 + 1 warmup epoch」，而其貌似合理的 frozen-F1 掩盖了数月 | **不影响 v10**：其 args.yaml 中 `lr0: 0.0001` 为显式值，属修复后 |
| 2 | **v5 泄漏** | `owner_v5_from_v4` 的 0.663 虚高——训练集含 43/47 个 eval 币种 | 不影响 v10；但表 §5 中该行不可用作对照 |
| 3 | **考卷不一致** | 各版自家 val mAP 不可跨版本比较 | 影响对 v10「进步幅度」的任何解读 |
| 4 | **训练分布含未来** | 训练图中信号右侧带启动后文 | **直接影响**：2026-08-05 ablation 实测 v10 在 tip 复现率 10%、完整上下文 62% |
| 5 | frozen-eval 本身的分布 | 与训练同分布（有后文） | frozen-F1 不能预测盘口表现 |

---

## 7. 数据资产状态（逐机核查）

| 资产 | 3060 | 本机 Mac | 说明 |
|---|---|---|---|
| **`dense_owner_v9`（v10 的训练集）** | **✅ train 6671 / val 2493，标签齐全** | ❌ | Phase 1 Task 2 的原图来源 |
| `dense_owner_v7` / `v8` | ✅ | ❌ | 祖先版本训练集 |
| `dense_owner_hts` / `star` | ✅ | ❌ | — |
| `dense_owner_v14_pad200` / `v15_tipval` / `v16_tipuni` | ✅ | ✅ | 两边都有 |
| `dense_owner_v11`（v11 训练集） | ❌ | ❌ | 与 v11 权重一起消失 |
| `golden_pool.json` | — | ✅ 12,567 stem / 6,229 框 / 6,630 纯背景 | 标注本体（框坐标）完好 |
| `owner_eval_frozen` | — | ✅ 47 币 / 464 图 | 固定尺子 |
| `benchmark_exemplars.json` | — | ✅ 176 张 ⭐ | — |
| 历史权重 | ✅ 59 个 | ✅ 61 个已备份至 `models/archive_3060/` | 含 25 套 args.yaml + results.csv |

**标注本体（golden_pool 的框坐标）与 v10 训练集的原图，两者都在。**
Pattern Library 的种子数据完整可用。

---

## 8. render / window 配置

| 项 | 值 | 来源 |
|---|---|---|
| 渲染函数 | `fable-trading/src/detection/render.py::render_chart` | 与训练同一函数 |
| 窗口长度 | 200 bar | `yoyo.layers.l1_detection.candidates.WINDOW` |
| 图像尺寸 | imgsz 960 | args.yaml |
| 均线 | `src/detection/data.py::add_mas`，SMA/EMA 20/60/120（MA206 口径，07-10 起） | — |
| 推理默认 | conf 0.30 / iou 0.70 | `DEFAULT_CONF`，项目冻结值 |

**Task 2 必须复用同一 `render_chart`**，不得另写——渲染语义一旦漂移，
新产出与全部历史结果不可比。

---

## 9. 交付清单

- 权重：`fable-trading/models/owner_v10_chain.pt`（SHA256 见 §1）
- 训练元数据：`fable-trading/models/archive_3060/owner_v10_chain/{args.yaml,results.csv}`
- 25 个历史 run 的同类元数据：`fable-trading/models/archive_3060/*/`
- 本报告：`yolo-xx/reports/pattern_teacher_asset_inventory.md`

## 10. 未做的事

- 未训练、未改权重/标签/数据集/阈值、未消耗 holdout。
- 未从 3060 拉取 `dense_owner_v9` 图像本体（约数 GB），Task 2 需要时再取。
- 未对 v7 之前的祖先做追溯（`owner_v7_chain.pt` 的 args.yaml 不在 3060 的 runs 目录下）。
- §5 的两把尺子差异只做了记录，未做统计显著性检验（464 图规模下 0.031 的差异是否属噪声，
  需要 bootstrap 才能回答；本轮不做）。
