# Owner-short YOLO 200/96 根单变量 A/B 预注册

状态：**已冻结，full build 与训练前登记**
登记时间：2026-08-04（Asia/Shanghai）

## 目的

只验证一个问题：在相同 owner-short 人工框、相同背景负样本、相同时间切分、相同底座和
相同训练参数下，把可见窗口从 200 根缩短到 96 根，是否改善 YOLO 对均线密集形态的检测与
定位。实验不判断方向收益，也不声称短窗能把历史人工框的完成时间提前。

## 数据合同

- 唯一 OHLCV 输入：`data/manual_short_preholdout_15m` immutable snapshot；截止线严格早于
  `2026-05-04T00:00:00Z`。
- 正样本：snapshot 内 1,361 个 `owner_side=short` 原始人工框。先按每个框生成窗口终点，完全
  相同的终点去重；窗口内所有完整可见的 owner-short 框都必须写入标签，任何被窗口边缘截断的
  人工框会使该 anchor 共同丢弃，避免把另一处已知正框漏标成背景。
  若不同 annotation ID 重映射成完全相同的 YOLO 坐标，只写一行训练标签并在 manifest 的
  `annotation_box_ids` 保留全部来源；强审计拒绝任何重复 YOLO 行，防止训练库静默改写输入。
- 全局时间切分：`2026-02-15T00:00:00Z`。train 要求最宽的 200 根窗口
  `available_at < split_at`；val 要求最宽的 200 根窗口 `window_start >= split_at`；跨界样本
  对两臂共同丢弃。
- 正框右侧上下文：按 anchor `box_id` 确定性映射到 `0/8/16/24` 根；两臂使用相同窗口终点与
  `available_at`。因此两臂只改变左侧历史长度和相应的真实渲染/框坐标。
- 负样本：每个保留正样本匹配一个相同 symbol、相同 split、相同右侧上下文桶、邻近时间块的
  空背景窗口。候选必须同时容纳 200/96 根窗口，不与任一 owner-short 框时段重叠，且 200/96
  两个视图都不触发冻结的 chart-only dense 规则。找不到安全候选时共同丢弃该正负 pair，绝不
  把未标注的疑似 dense 形态硬当负类。
- 负样本不含框；“位置匹配”指匹配 symbol、时间块、split、窗口终点和右侧上下文桶，不伪造
  一个负框位置。
- 目标比例：正图:负图 = 1:1；两臂 sample id、正负身份、split、symbol、窗口终点完全相同。
  200 根左侧额外可见的完整人工框会如实标注，因此两臂框总数可以不同；共享 anchor ledger
  不因结果改变。
- 先跑合成 fixture 与真实小样本 build/audit，再执行 full build。

## 唯一实验变量

| 项目 | A 臂 | B 臂 |
|---|---:|---:|
| window bars | 200 | 96 |
| 其余数据 ledger | 相同 | 相同 |
| 图像尺寸 | 1280×742 | 1280×742 |
| MA | 20/60/120 SMA+EMA | 20/60/120 SMA+EMA |

## 冻结训练配方

- GPU：Windows RTX 3060 12GB；CUDA device `0`。
- 底座：`weights/bases/yolo11n.pt`，SHA-256
  `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`。
- Ultralytics：`8.4.89`。
- 冷启动；`epochs=30`，`patience=10`，`imgsz=960`，`batch=8`，`workers=4`，
  `cache=false`，`seed=42`，`deterministic=true`，`amp=true`，`rect=true`。
- 优化器：两臂都使用 Ultralytics 冷启动默认选择；不以任一臂结果反向修改配方。
- 所有改变时间、位置或红绿语义的增强固定关闭：flip、mosaic、mixup、cutmix、HSV、BGR、
  translate、scale、rotate、shear、perspective、multi-scale、erasing、auto-augment。
- A 臂训练完成后再训 B 臂；机器可读 contract hash 必须证明除 dataset/name 外参数一致。

## 验证与跨周期扫描

- 只使用各自 pre-holdout val；报告 precision、recall、mAP50、mAP50-95、训练曲线和定位分布。
- 扫描阈值冻结为 `conf=0.30`、`iou=0.70`，不得按结果调阈值。
- 后续只对本地 pre-holdout 1m/2m/3m/5m OHLCV 渲染图做离线扫描；使用同一组
  20/60/120 bar 均线，目的是测试时间压缩路线，不保持 15m 的物理分钟长度。
- 扫描只比较信号数量、置信度、框位置、跨窗口一致性和抽样画廊；本阶段没有收益/正期望结论。

## 禁止项与停止条件

- 不读取或评分 holdout；不训练判断层；不调 threshold；不创建/修改 ACTIVE；不部署；不下单。
- 任一输入、pair contract、完整数据审计或训练 contract 失败即停止，不以“先跑起来”为理由
  绕过。
- 结果好得反常时第一假设是泄漏、重复图或位置捷径，先做最小复现审计。
