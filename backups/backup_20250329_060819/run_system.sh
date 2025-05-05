#!/bin/bash

# 股票分析系统启动脚本 - 完全隔离环境，绕过pyenv
echo "=============================================="
echo "股票分析系统 - 隔离环境启动器"
echo "=============================================="
echo "正在启动..."

# 使用完全隔离的环境启动系统
bash -c "PATH=/usr/bin:/usr/local/bin:/bin:/sbin /usr/bin/python3 main_gui.py" 