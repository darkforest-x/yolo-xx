# 完美均线密集形态 标注指南 V1（5m）

对应规格：[`configs/PERFECT_PATTERN_SPEC_V1.yaml`](../configs/PERFECT_PATTERN_SPEC_V1.yaml)
（`task_id: perfect_5m_six_ma_dense_v1`，当前 `status: draft`）
画廊：[`reports/pr01a_owner_gallery/index.html`](../reports/pr01a_owner_gallery/index.html)

## 0. 这一轮在问什么

只有一个问题：

> 这张 5m 图里，是否存在**非常标准、非常完美**的六线均线密集形态？

六线固定为：

```text
SMA20 + EMA20
SMA60 + EMA60
SMA120 + EMA120
```

周期数是 bar 数（bar_equivalent），不是物理分钟：在 5m 图上就是 20/60/120 根 5m K 线。

这不是在问「这里能不能做空」，也不是在问「后面赚不赚钱」。收益、方向、盈亏在当前阶段
一律不参与判断。

## 1. 判断的八个维度

逐张图按下面八条过一遍。**八条都成立**才是 positive。

```text
A. 六条线是否全部进入密集，而不是只有快线靠在一起；
B. 密集是否持续得足够完整，而不是瞬时交叉一下就散；
C. 线条是否明显互相靠拢、交织或压缩；
D. 整体斜率是否符合「标准形态」，而不是六条线平行地单边跑；
E. 价格结构是否同步收缩，而不是大幅波动来回穿越均线；
F. 是否尚未发生明显突破；
G. 框的开始与结束是否覆盖真实的密集段；
H. 是否足够标准，能让你无歧义地确认。
```

只要有一条明显不成立，就不是 positive。**宁可少，不可混。** 500 个极高质量的正框
好过 5,000 个混杂正框。

## 2. 四种状态

| 状态 | 什么时候用 |
|---|---|
| `positive` | 非常标准，可以进入 Gold Positive |
| `negative` | 明确不是目标形态，可进入 Gold negative 或 near-miss negative |
| `uncertain` | 不稳定、不够标准、说不清楚。**不会进入任何训练或验证** |
| `rejected` | 图片、渲染、数据或来源本身有问题，这张图不该被用 |

`uncertain` 不是失败选项，是保护结论质量的选项。犹豫超过几秒就选 `uncertain`。

## 3. 原因代码

一张图可以选多个。`negative` 至少给一个原因代码，`positive` 建议给
`PERFECT_SIX_LINE_DENSE`。

| 代码 | 含义 |
|---|---|
| `PERFECT_SIX_LINE_DENSE` | 六线标准密集，正样本 |
| `FAST_ONLY` | 只有快线密集 |
| `SLOW_LINES_SEPARATED` | 慢线明显分开 |
| `SLOPE_TOO_LARGE` | 线靠得近但整体斜率太大 |
| `DURATION_TOO_SHORT` | 密集持续太短 |
| `DURATION_TOO_LONG` | 密集拖得太长，形态散掉 |
| `PRICE_NOT_COMPRESSED` | 均线近但价格没有同步收缩 |
| `ALREADY_BROKEN_OUT` | 已经明显突破 |
| `INCOMPLETE_PATTERN` | 形态被窗口截断或没走完 |
| `SCALE_ILLUSION` | 纵轴缩放造成的假密集 |
| `BOX_START_WRONG` | 框的起点不对 |
| `BOX_END_WRONG` | 框的终点不对 |
| `AMBIGUOUS` | 说不清 |
| `BAD_RENDER` | 渲染或数据有问题 |
| `OTHER` | 其他，写在 notes 里 |

## 4. 框

首轮画廊不预先画框，避免锚定。

- `box_action = none`：这张图不涉及框（negative / uncertain / rejected 常用）；
- `box_action = accept`：形态位置就按你看到的密集段，不需要额外说明；
- `box_action = adjust`：需要明确框的位置，在输入框里填归一化坐标
  `xc,yc,w,h`（相对整张 1280×742 图，取值 0–1，框必须完全在图内）。

框的原则：**覆盖真实密集段**，起点在六线开始收拢处，终点在密集结束或突破发生前。

## 5. 绝对不能做的推断

这几条是硬规则，写进了 spec 与 ledger，任何工具都会拒绝违反它们的输入：

```text
空 label            ≠ negative
没有旧 owner 框      ≠ negative
旧模型没开火         ≠ negative
做空亏损            ≠ 形态 negative
outcome / 收益标签   ≠ 任何标签
规则筛不过           ≠ negative
旧模型高置信         ≠ positive
规则候选            ≠ positive
```

规则里的 `fast_spread<=0.0028 / full_spread<=0.0055 / 5–12 根 / 合并间隔 2` 只用来
**减少你要看的图**，它证明不了「标准、完美」。画廊里六个桶的分层也只是抽样策略，
不是答案——页面上不会显示桶名、模型名、置信度、币种和时间，就是为了避免锚定。

## 6. 操作流程

1. 打开 `reports/pr01a_owner_gallery/index.html`（本地文件，不联网）；
2. 逐张选 `positive / negative / uncertain / rejected`，需要时勾原因代码、填框和备注；
3. 进度会存在浏览器本地，可以分几次做完；
4. 点「导出 JSONL」，得到 `owner_reviews_pr01a.jsonl`；
5. 审计这份结果：

```bash
yolo-xx-pattern audit-reviews --manifest reports/pr01a_owner_gallery/review_manifest.json --reviews owner_reviews_pr01a.jsonl --out reports/pr01a_owner_gallery/review_audit.json
```

审计会检查：ID 一一对应、无重复、无未知样本、状态合法、decision 非空、框在图内。
未审核的图记为 `missing`，**不会**被算成 negative。

## 7. 这一轮之后

首批裁决完成后，进入 PR-01B：

1. 按正负边界把「完美形态」的文字定义写实；
2. 决定是否调整候选挖掘规则（只影响召回，不影响真值）；
3. 把 spec 置为 `status: owner_frozen`，写入 `frozen_by / frozen_at / spec_sha256`；
4. 之后才允许构建 `perfect_small_tf_v1` 和开始训练。

在 spec 冻结之前，`require_owner_frozen_spec()` 会让任何正式数据集构建和训练直接失败。
