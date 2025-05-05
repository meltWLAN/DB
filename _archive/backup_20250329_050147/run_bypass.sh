#!/bin/bash

# 绕过pyenv拦截，启动整合系统的shell脚本
echo "===== 股票分析系统启动器 ====="

# 查找系统Python路径
system_python=""
for path in /usr/bin/python3 /usr/local/bin/python3 /opt/homebrew/bin/python3; do
    if [ -x "$path" ]; then
        system_python="$path"
        break
    fi
done

if [ -z "$system_python" ]; then
    echo "未找到系统Python，尝试使用当前Python"
    system_python=$(which python3)
fi

echo "使用Python: $system_python"

# 运行整合系统
"$system_python" integrated_system.py 