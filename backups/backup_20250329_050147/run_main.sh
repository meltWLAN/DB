#!/bin/bash

# 尝试找到系统Python路径
if [ -x "/usr/bin/python3" ]; then
    PYTHON_PATH="/usr/bin/python3"
elif [ -x "/usr/local/bin/python3" ]; then
    PYTHON_PATH="/usr/local/bin/python3"
else
    PYTHON_PATH=$(which python)
fi

echo "使用Python路径: $PYTHON_PATH"

# 创建必要的目录
mkdir -p logs results/charts results/ma_charts data

# 设置Tushare Token (您的token可能会不同)
export TUSHARE_TOKEN="0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"

# 显示启动信息
echo "=========================================="
echo "   股票分析系统 - 主系统启动脚本"
echo "=========================================="
echo ""
echo "正在启动主系统..."
echo "这将执行完整的分析流程"
echo ""

# 使用env命令创建一个干净的环境执行Python脚本
env -i PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin" HOME="$HOME" $PYTHON_PATH stock_analysis_gui.py 