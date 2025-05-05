#!/usr/bin/env python3
"""
启动无界面股票分析系统
"""
import os
import sys
import subprocess
import platform

def main():
    """主函数"""
    print("=" * 60)
    print("股票分析系统 - 启动器")
    print("=" * 60)
    
    # 检查系统Python
    python_path = "/usr/bin/python3"
    if not os.path.exists(python_path):
        print(f"错误: 找不到系统Python: {python_path}")
        return 1
    
    print(f"系统Python: {python_path}")
    
    # 检查headless_gui.py
    gui_script = "headless_gui.py"
    if not os.path.exists(gui_script):
        print(f"错误: 找不到GUI脚本: {gui_script}")
        return 1
    
    print(f"GUI脚本: {gui_script}")
    
    # 确保文件有执行权限
    try:
        os.chmod(gui_script, 0o755)
    except:
        pass
    
    # 准备环境变量
    env = os.environ.copy()
    
    # 设置PYTHONPATH
    env["PYTHONPATH"] = os.getcwd()
    
    # 禁用pyenv
    env["PYENV_VERSION"] = "system"
    env["PYENV_DISABLE_PROMPT"] = "1"
    
    # 移除可能干扰的环境变量
    if "PYTHONHOME" in env:
        del env["PYTHONHOME"]
    
    print("\n正在启动无界面分析系统...")
    
    # 运行程序
    try:
        subprocess.run([python_path, gui_script], env=env)
        return 0
    except Exception as e:
        print(f"启动失败: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 