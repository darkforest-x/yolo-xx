# v16 tipuni(统一管线冷启动)训练与金标验收 — 2026-07-23

**纪律**:未 promote;未评 holdout;未清 forward_log;主线维持 detector=none 空转。

## 复现命令

```bash
# 数据集(已建):scripts/build_v16_tipuni_dataset.py → datasets/dense_owner_v16_tipuni
# 训练(3060):bash scripts/sync_v16_to_windows.sh && bash scripts/v16_train_start.sh
scp zzc@192.168.1.3:C:/fable/runs/detect/runs/detect/owner_v16_tipuni_cold/weights/best.pt \
  models/owner_v16_tipuni_cold.pt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FABLE_YOLO_DEVICE=cpu PYTHONPATH=. .venv/bin/python \
  scripts/eval_v15_fair_tip.py --v12 models/owner_v16_tipuni_cold.pt \
  --v15 models/owner_v16_tipuni_cold.pt --skip-v14 --skip-true-tip \
  --out analysis/output/v16_realtip_gate.json
# 注:评测脚本槽位名写死 v12/v15,本轮两槽同指 v16(输出 JSON 内 v12/v15 键均为 v16)
```

## 训练快照

| 项 | 值 |
|---|---|
| 数据 | dense_owner_v16_tipuni(train 6963:2635 pad200 正 + 4295 同代重渲负 + 33 真实盘口空背景;val 803 tip 对齐正 + 1478 重渲负) |
| 底座 | yolo11n.pt(COCO 冷启动;owner 裁定 v12 系永不作底座) |
| 结果 | 60/60 跑满(未早停);终局 P 0.704 / R 0.727 / mAP50 0.725(**训练期参考,不作裁决**) |

## 金标验收(真 tip 47 张,conf 0.30,A′ 贴边门)

| 指标 | 过线标准 | v16 | 历史对照(记录值) | 判定 |
|---|---|---|---|---|
| 应开火命中(hit∪miss-dense,n=9) | ≫ 1/9(v12) | **3/9 (0.33)** | v14 3/9 · v15 2/9 | 未达"≫" |
| 空背景误火(empty-ok,n=33) | **≈0**(v12 为 0/33) | **17/33 (51.5%)** | v14 18/33 · v15 19/33 | **不合格** |

**结论:不上线。** 主线维持 detector=none。

## 解读(最重要的部分)

统一渲染管线(消除新旧像素代差)**没有**治好空背景误火——v16 的 51.5% 与
v14/v15 的 55~58% 同量级。这排除了"像素代差捷径"是主因的假设,剩下的解释按
可能性排序:

1. **窗末几何捷径仍在**:正样本窗口永远终止于"形态完成 bar",负样本(原窗重渲)
   终止于任意 bar——模型可学"右缘看起来像个结构结尾 → 画框",而真实盘口的
   右缘一半时间"像个结尾"。33 张真实空背景是唯一同分布负样本,数量太少(0.7%)
   压不住这个捷径。
2. **标注语义本身在 tip 视角不可分**:owner 原始框定义的"密集"在裁掉后文之后,
   与大量普通盘口在视觉上可能本就难分——这正是主因 C 的完整形态。

两个解释指向同一个药方:**训练分布必须以真实盘口 tip 窗为主体**(正例=owner
确认的真 tip 成功形态,负例=大量真实盘口空背景),而不是从旧标注几何改造。

## 风险与诚实声明

- 金标分母小(应开火仅 9)且为规则∩v12 门预标,owner 尚未改判 review_sheet;
  应开火 3/9 的差异在小样噪声内,但 17/33 误火的量级结论稳健。
- 未跑 tip-smoke(当前盘口),误火结论已足够否决;未跑 frozen-F1(旧尺,不裁决)。
- 评测脚本槽位名与实际模型不一致(见复现命令注),JSON 键名需按注解读。

## 下一步(需 Owner 决策)

1. **真实 tip 数据集路线转正**(本报告主建议):
   a) Owner 填 review_sheet(48 张)→ 金标分母硬化;
   b) VPS 持续采集扩容(每脉冲旁路,数周积累千张级真实 tip 窗);
   c) `datasets/label_live_tip_1000`(1000 张盘口视角待标图)给 Owner 打标——
      正例负例都出自真实盘口分布,从根上消灭窗末几何捷径。
2. v16 权重保留作对照(不 promote);v17 = 真实盘口分布首训,等数据。
3. 实盘管道维持诚实空转,无需改动。
