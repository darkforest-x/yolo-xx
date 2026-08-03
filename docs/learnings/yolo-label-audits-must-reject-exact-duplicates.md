# YOLO 标签审计必须拒绝完全重复的框行

- **问题**：同一图中的多个历史 annotation ID 可能重映射成完全相同的 YOLO 坐标。Ultralytics
  会在训练缓存阶段静默删除这些重复标签，导致输入 manifest 的框数与实际训练框数不一致。
- **死胡同**：仅检查每行格式、边界和类别合法，仍会让逐行都合法的重复框通过；依赖训练日志
  warning 才发现问题太晚，而且两臂可能被不同程度地改写。
- **有效路径**：构建时按精确 `(class_id, xywhn)` 合并训练行，同时在 manifest 保留所有来源
  annotation ID；强审计对任何残留重复行直接失败，旧 v1 因而被明确判为无效并重建 v2。
- **通用规则**：训练前审计必须覆盖框集合语义，不只覆盖单行语法；任何框去重都应在可追溯的
  dataset build 阶段完成，不能交给训练框架隐式处理。
- **牵连**：`src/yolo_xx/audit.py`、`src/yolo_xx/paired_ab.py`、dataset manifest、A/B 框数统计。
