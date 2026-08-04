# Bash 3.2 中不要把函数 heredoc 管道嵌进 command substitution

- **问题**：Windows 日志已经写入 `all complete`，独立进程审计也是 `COUNT=0`，但 Mac 的
  `--fetch` 一直报告 launcher 仍在运行；相同传输形态还导致扫描 prepare 只执行片段。
- **死胡同**：重复等待和重复执行不会改变结果；安全门实际读到的不是非零计数，而是空字符串。
  原写法把函数调用、heredoc、管道和 command substitution 叠在一起，在系统 Bash 3.2 下丢失
  远端 `Write-Output 0`。
- **有效路径**：先把 PowerShell 探针存入普通 shell 字符串，再用已验证的 here-string 喂给
  `remote_ps`；同时把错误改成回显实际值，让 `missing` 与真实非零计数可区分。
- **通用规则**：跨机启动器传递多行远端脚本时，统一使用简单变量 + here-string；安全门不得把
  “无输出”和“检测到运行进程”折叠成同一错误，也不能仅凭后续 launcher 启动推断 prepare 已执行。
- **牵连**：`scripts/train_paired_ab_on_3060.sh`、`scripts/scan_micro_on_3060.sh`、macOS Bash
  3.2、Windows PowerShell stdin。
