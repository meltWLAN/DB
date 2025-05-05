#!/bin/bash
# 股票分析系统主启动脚本
# 解决macOS环境下的兼容性问题并启动主界面

# 设置环境变量解决macOS兼容性问题
export SYSTEM_VERSION_COMPAT=1

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 检查Python可执行文件
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "错误: 未找到Python可执行文件"
    exit 1
fi

# 启动主界面
# 应用动量分析模块修复
echo "正在应用动量分析模块修复..."
if [ -f patches/momentum_fix_complete.py ]; then
    export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR/patches"
    echo "动量分析模块修复已应用"
else
    echo "警告: 未找到动量分析模块修复文件"
fi

$PYTHON stock_analysis_gui.py
