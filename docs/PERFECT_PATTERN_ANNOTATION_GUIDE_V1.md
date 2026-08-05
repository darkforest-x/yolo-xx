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

图上的蓝框是**规则挖出来的候选框**，不是标签，也不是模型预测。240 张里有 200 张带候选框，
另外 40 张（规则完全没命中的连续背景）没有框。

你要做的就是判断：**这个框圈的东西，是不是完美形态**。

- 框准 → 直接判 `positive`，导出时记为 `box_action=accept`，用候选框坐标；
- 框不准 → 用鼠标拖动/拉角改，改过的框变橙色，导出时记为 `box_action=adjust`；
- 框错地方 → 在图上空白处直接拖出一个新框；
- 图上没框但你认为有形态 → 按 `1` 会自动给一个默认框，拖到位置上；
- `0` 复原成候选框，`\` 删掉框。

**判 `positive` 必须有框**——没有框的正样本训不了检测器，ledger 会直接报错。
`negative` / `uncertain` / `rejected` 不需要框。

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

一张图一个键，240 张不用滚动。

| 键 | 作用 |
|---|---|
| `1` `2` `3` `4` | positive / negative / uncertain / rejected（判完自动跳下一张） |
| `←` `→` | 上一张 / 下一张（也可用 `K` / `J`，空格 = 下一张） |
| `0` | 把框复原成候选框 |
| `\` | 删掉框 |
| `U` | 清除这张的判定 |
| `G` | 跳到第一张未判 |
| `X` | 原尺寸放大 / 还原 |
| `N` | 写备注，`Esc` 退出输入 |
| `E` | 导出 JSONL |
| `?` | 快捷键与判据说明 |

原因代码也有单键：`P` PERFECT_SIX_LINE_DENSE、`F` FAST_ONLY、`S` SLOW_LINES_SEPARATED、
`L` SLOPE_TOO_LARGE、`D` DURATION_TOO_SHORT、`C` PRICE_NOT_COMPRESSED、
`B` ALREADY_BROKEN_OUT、`I` INCOMPLETE_PATTERN、`Z` SCALE_ILLUSION、`M` AMBIGUOUS、
`R` BAD_RENDER；其余用鼠标点。

1. 打开 `reports/pr01a_owner_gallery/index.html`（本地文件，不联网）；
2. 逐张判，框不准就拖；
3. 进度存在浏览器本地，可以分几次做完；
4. 按 `E` 导出，得到 `owner_reviews_pr01a.jsonl`；
5. 审计这份结果：

```bash
yolo-xx-pattern audit-reviews --manifest reports/pr01a_owner_gallery/review_manifest.json --reviews owner_reviews_pr01a.jsonl --out reports/pr01a_owner_gallery/review_audit.json
```

审计会检查：ID 一一对应、无重复、无未知样本、状态合法、decision 非空、框在图内、
positive 必须带框。未审核的图记为 `missing`，**不会**被算成 negative。

## 7. 这份 JSONL 之后会被拿去做什么

```text
positive + 框  →  gold positive 标签（框来自 accept 的候选框或你调整后的框）
negative       →  背景图 / near-miss negative
uncertain      →  一律不进 train / val / test
rejected       →  连图带样本一起剔除
```

240 张大概只能给出几十个正框，不足以直接训练。它的作用是**把「完美」的定义钉死**：
定义确定后，按同一标准扩量挖候选，才构成正式训练集。

## 8. 这一轮之后

首批裁决完成后，进入 PR-01B：

1. 按正负边界把「完美形态」的文字定义写实；
2. 决定是否调整候选挖掘规则（只影响召回，不影响真值）；
3. 把 spec 置为 `status: owner_frozen`，写入 `frozen_by / frozen_at / spec_sha256`；
4. 之后才允许构建 `perfect_small_tf_v1` 和开始训练。

在 spec 冻结之前，`require_owner_frozen_spec()` 会让任何正式数据集构建和训练直接失败。
