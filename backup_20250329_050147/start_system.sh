#!/bin/bash

# 使用完整路径的Python解释器
PYTHON_PATH="/Users/mac/.pyenv/versions/3.10.12/bin/python"

# 检查是否存在
if [ ! -f "$PYTHON_PATH" ]; then
    echo "指定的Python解释器不存在，尝试使用系统默认的Python"
    PYTHON_PATH=$(which python3)
fi

# 创建必要的目录
mkdir -p logs results/charts results/ma_charts data

# 打印环境信息
echo "使用的Python解释器: $PYTHON_PATH"
echo "当前工作目录: $(pwd)"
echo "环境变量PATH: $PATH"

# 启动程序
echo "正在启动股票分析系统..."
"$PYTHON_PATH" stock_analysis_gui.py

# 如果程序启动失败，输出错误消息
if [ $? -ne 0 ]; then
    echo "程序启动失败，可能是Python环境问题"
    echo "请尝试以下方法解决:"
    echo "1. 确保已安装所有依赖: pip install -r requirements.txt"
    echo "2. 检查Python版本: $PYTHON_PATH --version"
    echo "3. 检查日志文件获取详细错误信息"
fi 