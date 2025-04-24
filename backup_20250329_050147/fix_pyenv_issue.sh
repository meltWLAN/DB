#!/bin/bash
# =========================================
# pyenv问题修复脚本
# =========================================

echo "==== 股票分析系统 - pyenv问题修复工具 ===="
echo "正在诊断并修复pyenv拦截问题..."

# 检查pyenv配置
PYENV_CONFIG_FILES=(
  "$HOME/.bash_profile"
  "$HOME/.bashrc"
  "$HOME/.zshrc"
  "$HOME/.profile"
)

echo "1. 检查pyenv拦截状态..."
PYENV_INTERCEPT_FOUND=0

for config_file in "${PYENV_CONFIG_FILES[@]}"; do
  if [ -f "$config_file" ]; then
    if grep -q "pyenv init" "$config_file"; then
      echo "在 $config_file 中发现pyenv配置"
      PYENV_INTERCEPT_FOUND=1
    fi
  fi
done

# 创建直接执行脚本
echo "2. 创建直接执行脚本..."

cat > direct_run.sh << 'EOF'
#!/bin/bash
# 直接执行系统Python，绕过pyenv

# 查找系统Python
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

# 直接运行目标文件，无需考虑pyenv
if [ -f "stock_analysis_gui.py" ]; then
    $SYSTEM_PYTHON stock_analysis_gui.py
elif [ -f "direct_start.py" ]; then
    $SYSTEM_PYTHON direct_start.py
else
    echo "错误: 找不到主程序文件"
    exit 1
fi
EOF

chmod +x direct_run.sh

# 创建临时环境
echo "3. 创建临时隔离环境..."

cat > run_isolated.py << 'EOF'
#!/usr/bin/env python3
"""
隔离环境运行器 - 完全绕过pyenv和环境干扰
"""
import os
import sys
import subprocess

# 查找系统Python
def find_system_python():
    """查找系统Python路径"""
    potential_paths = [
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3"
    ]
    
    for path in potential_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    return None

def main():
    """主函数"""
    print("==== 隔离环境运行器 ====")
    
    # 找到系统Python
    system_python = find_system_python()
    if not system_python:
        print("错误: 找不到系统Python")
        return False
    
    print(f"使用系统Python: {system_python}")
    
    # 检查目标文件
    target_files = ["stock_analysis_gui.py", "direct_start.py"]
    target_file = None
    
    for file in target_files:
        if os.path.exists(file):
            target_file = file
            break
    
    if not target_file:
        print("错误: 找不到主程序文件")
        return False
    
    # 创建一个临时运行脚本
    temp_script = "temp_run_script.py"
    with open(temp_script, "w") as f:
        f.write(f"""
import os
import sys
import importlib.util
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IsolatedRunner')

# 设置环境变量
os.environ["PYTHONPATH"] = os.getcwd()
os.environ["TUSHARE_TOKEN"] = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"

try:
    # 直接运行目标文件
    logger.info(f"正在运行: {target_file}")
    
    # 使用spec导入模块
    spec = importlib.util.spec_from_file_location("main_module", "{target_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    logger.info("程序运行完成")
except Exception as e:
    logger.error(f"运行失败: {{str(e)}}")
""")
    
    # 使用系统Python运行临时脚本
    try:
        subprocess.call([system_python, temp_script])
        # 清理临时文件
        if os.path.exists(temp_script):
            os.remove(temp_script)
        return True
    except Exception as e:
        print(f"运行失败: {str(e)}")
        return False

if __name__ == "__main__":
    main()
EOF

chmod +x run_isolated.py

# 创建一键运行脚本
echo "4. 创建一键运行脚本..."

cat > launch_program.sh << 'EOF'
#!/bin/bash
# 一键启动脚本

echo "==== 股票分析系统 - 启动器 ===="
echo "尝试三种方法启动系统:"

# 方法1: 直接运行
echo "方法1: 使用direct_run.sh..."
if [ -f "direct_run.sh" ]; then
    ./direct_run.sh
    if [ $? -eq 0 ]; then
        echo "启动成功!"
        exit 0
    else
        echo "方法1失败，尝试方法2..."
    fi
else
    echo "direct_run.sh不存在，尝试方法2..."
fi

# 方法2: 隔离环境
echo "方法2: 使用run_isolated.py..."
if [ -f "run_isolated.py" ]; then
    # 查找系统Python
    SYSTEM_PYTHON=""
    for python_path in "/usr/bin/python3" "/usr/local/bin/python3" "/opt/homebrew/bin/python3"; do
        if [ -x "$python_path" ]; then
            SYSTEM_PYTHON="$python_path"
            break
        fi
    done
    
    if [ -n "$SYSTEM_PYTHON" ]; then
        $SYSTEM_PYTHON run_isolated.py
        if [ $? -eq 0 ]; then
            echo "启动成功!"
            exit 0
        else
            echo "方法2失败，尝试方法3..."
        fi
    else
        echo "找不到系统Python，尝试方法3..."
    fi
else
    echo "run_isolated.py不存在，尝试方法3..."
fi

# 方法3: 环境变量修改
echo "方法3: 使用环境变量修改..."
PYTHON_COMMAND=""

# 查找系统Python
for python_path in "/usr/bin/python3" "/usr/local/bin/python3" "/opt/homebrew/bin/python3"; do
    if [ -x "$python_path" ]; then
        PYTHON_COMMAND="$python_path"
        break
    fi
done

if [ -z "$PYTHON_COMMAND" ]; then
    echo "错误: 找不到系统Python"
    exit 1
fi

echo "使用Python: $PYTHON_COMMAND"

# 设置环境变量绕过pyenv
export PYENV_VERSION=system
export PYENV_DISABLE_PROMPT=1
export PYTHONNOUSERSITE=1

# 尝试运行主程序
if [ -f "stock_analysis_gui.py" ]; then
    $PYTHON_COMMAND stock_analysis_gui.py
elif [ -f "direct_start.py" ]; then
    $PYTHON_COMMAND direct_start.py
else
    echo "错误: 找不到主程序文件"
    exit 1
fi
EOF

chmod +x launch_program.sh

echo ""
echo "========== 解决方案 =========="
if [ $PYENV_INTERCEPT_FOUND -eq 1 ]; then
    echo "检测到pyenv拦截配置。请使用下面的方法启动:"
    echo "方法1: ./direct_run.sh         (直接使用系统Python)"
    echo "方法2: ./run_isolated.py       (隔离环境运行)"
    echo "方法3: ./launch_program.sh     (自动尝试三种方法)"
    echo ""
    echo "推荐使用方法3: ./launch_program.sh"
else
    echo "未检测到pyenv拦截配置。请尝试:"
    echo "1. ./launch_program.sh"
    echo "2. /usr/bin/python3 direct_start.py"
fi

echo ""
echo "修复完成! 如有问题请联系开发人员" 