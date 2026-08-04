# 可清空 worker 的代码包应先验收再替换运行目录

- **问题**：扫描数据和 launcher 上传成功，但远端仍保留上一轮训练时的旧 `src/`，新入口
  `yolo_xx.scan_predict` 不存在，任务在处理任何图片前失败。
- **死胡同**：直接把 tar 解到已有运行目录并只检查 `tar` 退出码，不能证明关键新文件已经落位；
  `tar exit=0` 只证明解包过程没有报告错误。
- **有效路径**：代码包先解到独立 staging，逐项验证 `__init__.py`、`scan_set.py`、
  `scan_predict.py` 和 `pyproject.toml`，全部存在后再替换 disposable worker 的 `src/`。
- **通用规则**：跨机发布必须验证“运行入口存在”，不能只验证“压缩包解开”；数据 staging 与代码
  staging 分开，任何一侧失败都不得启动计算。
- **牵连**：`scripts/scan_micro_on_3060.sh`、Windows `C:/yolo-xx/src`、离线扫描 launcher。
