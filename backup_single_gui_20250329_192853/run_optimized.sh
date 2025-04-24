#!/bin/bash

# 优化版运行脚本 - 绕过pyenv干扰
echo "=============================================="
echo "股票分析系统 - 优化版启动器"
echo "=============================================="

# 添加SYSTEM_VERSION_COMPAT=1环境变量来处理macOS版本兼容性问题
exec env -i \
  HOME="$HOME" \
  USER="$USER" \
  DISPLAY="$DISPLAY" \
  SYSTEM_VERSION_COMPAT=1 \
  PATH="/usr/bin:/usr/local/bin:/bin:/sbin" \
  /usr/bin/python3 "$(dirname "$0")/stock_analysis_gui.py"
