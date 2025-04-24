#!/usr/bin/env python3
"""
股票分析系统启动器 - 自动修复版本
"""
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
        logging.FileHandler(f'logs/launcher_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Launcher")

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
            return path
    
    return None

def main():
    """主函数"""
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
            print(f"当前不是使用系统Python，将切换到: {system_python}")
            
            # 使用系统Python重新运行此脚本
            try:
                subprocess.run([system_python, __file__])
                return
            except Exception as e:
                print(f"切换到系统Python失败: {str(e)}")
                logger.error(f"切换到系统Python失败: {str(e)}")
        else:
            print("警告: 未找到系统Python")
    
    # 可用的主程序
    main_scripts = ['stock_analysis_gui.py', 'simple_gui.py', 'integrated_system.py', 'direct_start.py', 'cli_analyst.py']
    
    if not main_scripts:
        print("错误: 未找到主程序脚本")
        return
    
    # 显示可用程序
    print("\n可用的程序:")
    for i, script in enumerate(main_scripts):
        print(f"{i+1}. {script}")
    
    # 选择程序
    choice = input("\n请选择要运行的程序 [1-{len(main_scripts)}]: ")
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
        
        print(f"\n正在启动 {selected_script}...")
        
        # 运行所选程序
        subprocess.run([sys.executable, selected_script], env=env)
        
    except ValueError:
        print("请输入有效的数字")
    except Exception as e:
        print(f"启动失败: {str(e)}")
        logger.error(f"启动失败: {str(e)}")

if __name__ == "__main__":
    main()
