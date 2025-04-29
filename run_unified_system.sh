#!/bin/bash
# 股票分析系统统一启动器脚本

# 设置环境变量解决macOS兼容性问题
export SYSTEM_VERSION_COMPAT=1

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 确保logs目录存在
mkdir -p logs

# 当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/unified_launcher_${TIMESTAMP}.log"

echo "[$(date)] 正在启动股票分析系统..." | tee -a "$LOG_FILE"

# 检查Python可执行文件
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "[$(date)] 错误: 未找到Python可执行文件" | tee -a "$LOG_FILE"
    exit 1
fi

echo "[$(date)] 使用Python: $PYTHON" | tee -a "$LOG_FILE"

# 检查统一启动器脚本
if [ ! -f "unified_launcher.py" ]; then
    echo "[$(date)] 错误: 未找到统一启动器文件 unified_launcher.py" | tee -a "$LOG_FILE"
    exit 1
fi

# 启动统一启动器
echo "[$(date)] 正在启动统一启动器..." | tee -a "$LOG_FILE"
$PYTHON unified_launcher.py

# 退出状态
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date)] 启动器以错误代码退出: $EXIT_CODE" | tee -a "$LOG_FILE"
else
    echo "[$(date)] 启动器正常退出" | tee -a "$LOG_FILE"
fi

exit $EXIT_CODE 