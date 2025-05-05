#!/bin/bash
# 股票分析系统 - Shell启动器

echo "=============================================="
echo "      股票分析系统 - Shell启动器"
echo "=============================================="

# 查找系统Python
SYSTEM_PYTHON="/usr/bin/python3"
if [ ! -x "$SYSTEM_PYTHON" ]; then
    echo "错误: 系统Python不可执行: $SYSTEM_PYTHON"
    exit 1
fi

# 显示系统信息
echo "系统Python: $SYSTEM_PYTHON"
echo "Python版本: $($SYSTEM_PYTHON --version)"
echo "当前目录: $(pwd)"

# 确保必要目录存在
for DIR in "data" "logs" "results" "cache"; do
    mkdir -p "$DIR"
done

# 可用的程序
echo ""
echo "可用的程序:"
echo "1. stock_analysis_gui.py"
echo "2. simple_gui.py"
echo "3. integrated_system.py"
echo "4. direct_start.py"
echo "5. cli_analyst.py"

# 选择程序
echo ""
read -p "请选择要运行的程序 [1-5]: " choice

# 根据选择设置脚本
SCRIPT=""
case $choice in
        1) SCRIPT="stock_analysis_gui.py" ;;
        2) SCRIPT="simple_gui.py" ;;
        3) SCRIPT="integrated_system.py" ;;
        4) SCRIPT="direct_start.py" ;;
        5) SCRIPT="cli_analyst.py" ;;
        *) echo "无效的选择"; exit 1 ;;
esac

if [ -z "$SCRIPT" ] || [ ! -f "$SCRIPT" ]; then
    echo "错误: 无效的脚本选择"
    exit 1
fi

echo ""
echo "正在启动 $SCRIPT..."

# 设置环境变量
export PYTHONPATH="$(pwd)"
export PYENV_VERSION=system
export PYENV_DISABLE_PROMPT=1

# 取消可能干扰的环境变量
unset PYTHONHOME
unset PYTHONNOUSERSITE

# 运行程序
"$SYSTEM_PYTHON" "$SCRIPT"
