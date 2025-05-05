#!/bin/bash

# 股票分析系统 - 直接启动器

# 清理屏幕并显示标题
clear
echo "=============================================="
echo "        股票分析系统 - 正在启动              "
echo "=============================================="

# 获取当前脚本所在目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 强制使用系统Python，完全隔离环境
# exec替换当前进程，-c清空所有环境变量
exec -c env -i \
  HOME="$HOME" \
  USER="$USER" \
  DISPLAY="$DISPLAY" \
  PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin" \
  /usr/bin/python3 "$DIR/stock_analysis_gui.py"

# 注意：这个脚本使用exec，所以下面的代码永远不会执行 