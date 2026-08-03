# macOS 跨机启动器必须以 Bash 3.2 为兼容基线

- **问题**：Mac 启动器本地逻辑正确，却在上传前因 `declare -A` 直接退出，Windows 训练没有启动。
- **死胡同**：脚本使用 `#!/bin/bash` 不代表具备 Bash 4；在 Linux 上通过语法检查也不能证明默认
  macOS `/bin/bash` 能执行关联数组。
- **有效路径**：去掉关联数组，改用两个明确命名的 receipt 变量，并把 `bash -n` 纳入提交前检查。
- **通用规则**：面向系统自带 macOS Bash 的启动器默认只使用 3.2 语法；确需新语法时必须显式
  固定解释器及版本，而不是依赖开发机 PATH。
- **牵连**：`scripts/train_paired_ab_on_3060.sh`、`scripts/scan_micro_on_3060.sh`、Mac→Windows
  离线训练与扫描入口。
