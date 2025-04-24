#!/bin/bash

# 完全禁用pyenv的干扰
unset PYENV_VERSION
unset PYENV_ROOT
unset PYENV_DIR
unset PYENV_HOOK_PATH

# 确保PATH中不包含pyenv路径
PATH=$(echo "$PATH" | sed -E 's|:/Users/mac/.pyenv/shims||g')
PATH=$(echo "$PATH" | sed -E 's|/Users/mac/.pyenv/shims:||g')

# 直接使用系统Python
echo "正在启动股票分析系统..."
/usr/bin/python3 "$(dirname "$0")/start.py"