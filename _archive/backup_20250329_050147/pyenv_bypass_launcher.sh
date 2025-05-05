#!/bin/bash
# =========================================
# pyenv绕过启动器 - 完全绕过pyenv拦截
# =========================================

echo "==== 股票分析系统 - 绕过pyenv启动器 ===="

# 强制使用系统Python，完全绕过pyenv拦截
export PYENV_VERSION=system
export PYENV_DISABLE_PROMPT=1

# 检查系统Python路径
SYSTEM_PYTHON=""
for python_path in "/usr/bin/python3" "/usr/local/bin/python3" "/opt/homebrew/bin/python3"; do
    if [ -x "$python_path" ]; then
        SYSTEM_PYTHON="$python_path"
        break
    fi
done

if [ -z "$SYSTEM_PYTHON" ]; then
    echo "错误: 找不到系统Python路径"
    exit 1
fi

echo "使用系统Python: $SYSTEM_PYTHON"

# 保存当前目录
CURRENT_DIR=$(pwd)

# 创建一个临时启动Python脚本
TEMP_SCRIPT="$CURRENT_DIR/temp_bypass_launcher.py"

cat > "$TEMP_SCRIPT" << 'EOF'
import os
import sys
import importlib.util
import subprocess

def run_main_system():
    """运行主系统"""
    # 直接启动direct_start.py
    script_path = os.path.join(os.getcwd(), "direct_start.py")
    
    if not os.path.exists(script_path):
        print(f"错误: 找不到启动文件 {script_path}")
        return False
        
    try:
        # 读取文件内容
        with open(script_path, 'r') as f:
            script_content = f.read()
            
        # 通过Python执行该文件
        exec(script_content, globals())
        return True
    except Exception as e:
        print(f"启动失败: {e}")
        return False

if __name__ == "__main__":
    print("系统Python启动器 - 绕过pyenv")
    print(f"Python路径: {sys.executable}")
    print(f"Python版本: {sys.version}")
    
    # 启动系统
    if not run_main_system():
        print("启动失败! 按任意键退出...")
        input()
EOF

# 删除pyenv拦截，直接使用系统Python运行
"$SYSTEM_PYTHON" "$TEMP_SCRIPT"

# 清理临时文件
rm -f "$TEMP_SCRIPT"

echo "执行完毕!" 