#!/bin/bash
# 超级优化版本股票分析系统启动脚本
# 使用动态参数优化、多进程计算和智能缓存

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
mkdir -p logs results cache optimized_cache

# 显示优化信息
echo "启动超级优化版股票分析系统"
echo "使用动态参数优化、多进程和智能缓存加速分析"

# 计算可用CPU核心数，设置为合理的进程数
CPU_COUNT=$($PYTHON -c "import multiprocessing; print(max(1, multiprocessing.cpu_count()-1))")
echo "检测到 $(($CPU_COUNT+1)) 个CPU核心，将使用 $CPU_COUNT 个工作进程"

# 检查可用内存，动态调整参数
MEMORY_GB=$($PYTHON -c "import psutil; print(int(psutil.virtual_memory().total / (1024**3)))")
echo "检测到系统内存 ${MEMORY_GB}GB"

# 配置环境变量以优化性能
export NUMBA_NUM_THREADS=$(($CPU_COUNT+1))  # Numba线程数
export MKL_NUM_THREADS=$(($CPU_COUNT+1))    # Intel MKL线程数
export OPENBLAS_NUM_THREADS=$(($CPU_COUNT+1))  # OpenBLAS线程数
export PYTHONHASHSEED=0  # 确保哈希结果一致性

# 优化Python虚拟机
export PYTHONFAULTHANDLER=1  # 启用错误处理器
export PYTHONUNBUFFERED=1    # 无缓冲输出

# 添加优化模块路径
export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR"

# 应用动量分析模块修复
echo "正在应用动量分析模块修复..."
if [ -f patches/momentum_fix_complete.py ]; then
    export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR/patches"
    echo "动量分析模块修复已应用"
else
    echo "警告: 未找到动量分析模块修复文件"
fi

# 应用性能优化模块
echo "正在加载性能优化模块..."
if [ -d "optimizations" ]; then
    export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR/optimizations"
    echo "性能优化模块已加载"
else
    echo "警告: 未找到性能优化模块"
fi

# 启动超优化版本GUI
echo "正在启动超级优化版本界面..."
$PYTHON stock_analysis_gui.py --use-optimized --use-hyper-optimized --workers $CPU_COUNT 