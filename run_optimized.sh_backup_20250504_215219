#!/bin/bash
# 优化版本股票分析系统启动脚本
# 使用多进程和缓存优化

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

# 创建必要的目录
mkdir -p logs results cache data

# 显示优化信息
echo "启动优化版股票分析系统"
echo "使用多进程和缓存加速分析"

# 计算可用CPU核心数，设置为合理的进程数
CPU_COUNT=$($PYTHON -c "import multiprocessing; print(max(1, multiprocessing.cpu_count()-1))")
echo "检测到 $(($CPU_COUNT+1)) 个CPU核心，将使用 $CPU_COUNT 个工作进程"

# 启动优化版本GUI
echo "正在启动优化版本界面..."
$PYTHON stock_analysis_gui.py --use-optimized --workers $CPU_COUNT
