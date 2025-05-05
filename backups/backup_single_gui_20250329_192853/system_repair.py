"""
股票分析系统 - 系统修复工具
解决环境问题并提供多种启动方式
"""
import os
import sys
import platform
import subprocess
import logging
import shutil
import tempfile
import time
from pathlib import Path

# 配置日志
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/system_repair_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SystemRepair")

def check_system():
    """检查系统状态"""
    print("\n系统信息:")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"Python路径: {sys.executable}")
    print(f"当前目录: {os.getcwd()}")
    
    # 检查pyenv
    pyenv_active = False
    if "PYENV_ROOT" in os.environ:
        print(f"检测到pyenv: {os.environ['PYENV_ROOT']}")
        pyenv_active = True
    
    # 检查虚拟环境
    venv_active = False
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"检测到虚拟环境: {sys.prefix}")
        venv_active = True
    
    return {
        "os": platform.system(),
        "python_version": sys.version.split()[0],
        "python_path": sys.executable,
        "pyenv_active": pyenv_active,
        "venv_active": venv_active
    }

def find_system_python():
    """查找系统Python路径"""
    system_paths = [
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/Library/Developer/CommandLineTools/usr/bin/python3",
        "/opt/homebrew/bin/python3"
    ]
    
    if platform.system() == "Windows":
        system_paths = [
            r"C:\Python39\python.exe",
            r"C:\Python310\python.exe",
            r"C:\Program Files\Python39\python.exe",
            r"C:\Program Files\Python310\python.exe"
        ]
    
    for path in system_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            logger.info(f"找到系统Python: {path}")
            return path
    
    logger.warning("未找到系统Python")
    return None

def find_main_scripts():
    """查找主程序脚本"""
    main_scripts = [
        "stock_analysis_gui.py",
        "simple_gui.py",
        "integrated_system.py",
        "direct_start.py",
        "cli_analyst.py"
    ]
    
    found_scripts = []
    for script in main_scripts:
        if os.path.exists(script):
            found_scripts.append(script)
    
    return found_scripts

def check_dependencies():
    """检查依赖包"""
    print("\n检查依赖包:")
    
    dependencies = {
        "基础包": ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn"],
        "数据处理": ["pytz", "python-dateutil", "statsmodels"],
        "数据获取": ["yfinance", "requests", "beautifulsoup4", "lxml"],
        "数据存储": ["sqlalchemy", "sqlite3"],
        "GUI": ["tkinter", "pillow"]
    }
    
    missing_packages = []
    
    for category, packages in dependencies.items():
        print(f"\n{category}:")
        for package in packages:
            try:
                if package == "tkinter":
                    import tkinter
                    print(f"  ✓ {package}")
                elif package == "sqlite3":
                    import sqlite3
                    print(f"  ✓ {package}")
                else:
                    __import__(package)
                    print(f"  ✓ {package}")
            except ImportError:
                print(f"  ✗ {package}")
                missing_packages.append(package)
            except Exception as e:
                print(f"  ? {package} - 出错: {str(e)}")
                missing_packages.append(package)
    
    return missing_packages

def install_packages(packages, use_user=True):
    """安装缺失的Python包"""
    print("\n安装缺失的包:")
    
    # 对包进行分组，避开tkinter (需要系统安装)
    pip_packages = [pkg for pkg in packages if pkg not in ["tkinter", "sqlite3"]]
    
    if not pip_packages:
        print("没有需要通过pip安装的包")
        return
    
    for package in pip_packages:
        try:
            print(f"安装 {package}...")
            cmd = [sys.executable, "-m", "pip", "install"]
            if use_user:
                cmd.append("--user")
            cmd.append(package)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  ✓ {package} 安装成功")
            else:
                print(f"  ✗ {package} 安装失败: {result.stderr}")
        except Exception as e:
            print(f"  ✗ {package} 安装出错: {str(e)}")

def create_launcher_script():
    """创建启动器脚本"""
    launcher_file = "start_stock_analyzer.py"
    print(f"\n创建启动器脚本: {launcher_file}")
    
    # 获取可用的主程序脚本
    main_scripts = find_main_scripts()
    if not main_scripts:
        print("错误: 未找到主程序脚本，无法创建启动器")
        return None
    
    with open(launcher_file, "w") as f:
        f.write(f"""#!/usr/bin/env python3
\"\"\"
股票分析系统启动器 - 自动修复版本
\"\"\"
import os
import sys
import subprocess
import platform
import logging
import time
from pathlib import Path

# 配置日志
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/launcher_{{time.strftime("%Y%m%d_%H%M%S")}}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Launcher")

def find_system_python():
    \"\"\"查找系统Python路径\"\"\"
    system_paths = [
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/Library/Developer/CommandLineTools/usr/bin/python3",
        "/opt/homebrew/bin/python3"
    ]
    
    if platform.system() == "Windows":
        system_paths = [
            r"C:\\Python39\\python.exe",
            r"C:\\Python310\\python.exe",
            r"C:\\Program Files\\Python39\\python.exe",
            r"C:\\Program Files\\Python310\\python.exe"
        ]
    
    for path in system_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    return None

def main():
    \"\"\"主函数\"\"\"
    print("=" * 60)
    print("股票分析系统启动器")
    print("=" * 60)
    
    # 检查是否使用系统Python
    is_system_python = False
    python_path = sys.executable
    
    if platform.system() == "Darwin":  # macOS
        if python_path.startswith(("/usr/bin/", "/usr/local/bin/", "/Library/Developer/")):
            is_system_python = True
    elif platform.system() == "Windows":
        if "Program Files" in python_path or not "Users" in python_path:
            is_system_python = True
    
    if not is_system_python:
        system_python = find_system_python()
        if system_python:
            print(f"当前不是使用系统Python，将切换到: {{system_python}}")
            
            # 使用系统Python重新运行此脚本
            try:
                subprocess.run([system_python, __file__])
                return
            except Exception as e:
                print(f"切换到系统Python失败: {{str(e)}}")
                logger.error(f"切换到系统Python失败: {{str(e)}}")
        else:
            print("警告: 未找到系统Python")
    
    # 可用的主程序
    main_scripts = {main_scripts}
    
    if not main_scripts:
        print("错误: 未找到主程序脚本")
        return
    
    # 显示可用程序
    print("\\n可用的程序:")
    for i, script in enumerate(main_scripts):
        print(f"{{i+1}}. {{script}}")
    
    # 选择程序
    choice = input("\\n请选择要运行的程序 [1-{{len(main_scripts)}}]: ")
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(main_scripts):
            print("无效的选择")
            return
        
        selected_script = main_scripts[idx]
        
        # 设置环境变量
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        
        # 移除可能干扰的环境变量
        for var in ["PYTHONHOME", "PYTHONNOUSERSITE"]:
            if var in env:
                env.pop(var)
        
        print(f"\\n正在启动 {{selected_script}}...")
        
        # 运行所选程序
        subprocess.run([sys.executable, selected_script], env=env)
        
    except ValueError:
        print("请输入有效的数字")
    except Exception as e:
        print(f"启动失败: {{str(e)}}")
        logger.error(f"启动失败: {{str(e)}}")

if __name__ == "__main__":
    main()
""")
    
    # 添加执行权限
    os.chmod(launcher_file, 0o755)
    print(f"启动器脚本已创建: {launcher_file}")
    
    return launcher_file

def create_shell_launcher():
    """创建shell启动脚本"""
    launcher_file = "start_analyzer.sh"
    print(f"\n创建Shell启动脚本: {launcher_file}")
    
    # 获取系统Python路径
    system_python = find_system_python()
    if not system_python:
        print("错误: 未找到系统Python，无法创建Shell启动脚本")
        return None
    
    # 获取可用的主程序脚本
    main_scripts = find_main_scripts()
    if not main_scripts:
        print("错误: 未找到主程序脚本，无法创建Shell启动脚本")
        return None
    
    main_script_lines = "\n".join([f'echo "{i+1}. {script}"' for i, script in enumerate(main_scripts)])
    case_lines = "\n".join([f'        {i+1}) SCRIPT="{script}" ;;' for i, script in enumerate(main_scripts)])
    
    with open(launcher_file, "w") as f:
        f.write(f"""#!/bin/bash
# 股票分析系统 - Shell启动器

echo "=============================================="
echo "      股票分析系统 - Shell启动器"
echo "=============================================="

# 查找系统Python
SYSTEM_PYTHON="{system_python}"
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
{main_script_lines}

# 选择程序
echo ""
read -p "请选择要运行的程序 [1-{len(main_scripts)}]: " choice

# 根据选择设置脚本
SCRIPT=""
case $choice in
{case_lines}
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
""")
    
    # 添加执行权限
    os.chmod(launcher_file, 0o755)
    print(f"Shell启动脚本已创建: {launcher_file}")
    
    return launcher_file

def backup_folder():
    """备份当前文件夹"""
    backup_dir = f"backup_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"\n创建备份: {backup_dir}")
    
    try:
        os.makedirs(backup_dir, exist_ok=True)
        
        # 复制Python文件
        py_files = list(Path(".").glob("*.py"))
        for file in py_files:
            shutil.copy2(file, backup_dir)
            print(f"  备份: {file.name}")
        
        # 复制配置文件
        config_files = list(Path(".").glob("*.json")) + list(Path(".").glob("*.yaml")) + list(Path(".").glob("*.txt"))
        for file in config_files:
            shutil.copy2(file, backup_dir)
            print(f"  备份: {file.name}")
        
        print(f"备份完成: {len(py_files) + len(config_files)} 个文件")
        return backup_dir
    except Exception as e:
        print(f"备份失败: {str(e)}")
        logger.error(f"备份失败: {str(e)}")
        return None

def repair_system():
    """执行系统修复"""
    # 创建备份
    backup_folder()
    
    # 安装缺失的包
    missing_packages = check_dependencies()
    if missing_packages:
        print(f"\n检测到 {len(missing_packages)} 个缺失的包")
        install = input("是否安装缺失的包? [y/n]: ").lower()
        if install == 'y':
            install_packages(missing_packages)
    
    # 创建启动脚本
    launcher_py = create_launcher_script()
    launcher_sh = create_shell_launcher()
    
    print("\n系统修复完成!")
    print("请使用以下方式启动系统:")
    
    if launcher_py:
        print(f"1. Python启动器: /usr/bin/python3 {launcher_py}")
    
    if launcher_sh:
        print(f"2. Shell启动器: ./{launcher_sh}")
    
    print(f"3. 命令行模式: /usr/bin/python3 cli_analyst.py")

def main():
    """主函数"""
    print("=" * 60)
    print("股票分析系统 - 系统修复工具")
    print("=" * 60)
    
    system_info = check_system()
    
    if system_info["pyenv_active"]:
        # 检查是否使用系统Python
        if not system_info["python_path"].startswith(("/usr/bin/", "/usr/local/bin/", "/Library/Developer/")):
            system_python = find_system_python()
            if system_python:
                print(f"\n检测到使用pyenv Python，尝试切换到系统Python: {system_python}")
                
                # 为了确保使用系统Python，我们重新启动此脚本
                try:
                    result = subprocess.run([system_python, __file__], check=True)
                    return result.returncode
                except subprocess.CalledProcessError as e:
                    print(f"使用系统Python重新启动失败: {e}")
                    return e.returncode
                except Exception as e:
                    print(f"切换到系统Python时出错: {e}")
                    return 1
    
    # 继续修复流程
    repair_system()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 