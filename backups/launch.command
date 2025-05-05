#!/bin/bash
# macOS兼容的股票分析系统启动脚本
# 双击可直接在Finder中启动

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

# 启动主启动器
$PYTHON main_launcher.py

# 如果有错误，等待用户按键退出
if [ $? -ne 0 ]; then
    echo "启动器执行出错，请按任意键退出..."
    read -n 1
fi 