# 可清空 GPU worker 必须用内容回执衔接完整源审计

- **问题**：schema-v2 完整审计会追溯 immutable OHLCV snapshot。只把训练数据复制到 Windows 后，
  manifest 中的 Mac source path 不存在；若直接跳过审计，远端又无法证明图片和标签就是 Mac
  验收过的那一版。
- **死胡同**：把 287MB OHLCV 一并复制会扩大敏感输入和同步成本；只检查样本数量或 manifest
  文件名无法发现传输损坏、旧文件混入或标签被改写。
- **有效路径**：Mac 先完成 full source audit，再生成 portable receipt，锁定 data YAML、dataset
  manifest 和每张 image/label 的 SHA-256。启动命令把 receipt 自身的 SHA-256 作为独立参数传给
  Windows，worker 对全部 payload 重新散列后才允许训练。
- **通用规则**：source-of-truth 与可清空计算 worker 分离时，先在源端验来源，再用逐文件内容回执
  验运输；不要把“源审计不可达”降级为“跳过审计”。
- **牵连**：`src/yolo_xx/portable.py`、`src/yolo_xx/train.py`、训练数据 manifest、Windows 启动脚本；
  worker 不接收 OHLCV、holdout、ACTIVE 或交易运行时。
