#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用启动脚本 - 在任何环境下启动股票分析系统
"""

import os
import sys
import subprocess
import platform

def get_system_python():
    """获取系统Python路径"""
    if platform.system() == "Darwin" or platform.system() == "Linux":
        # macOS 或 Linux
        if os.path.exists("/usr/bin/python3"):
            return "/usr/bin/python3"
        elif os.path.exists("/usr/local/bin/python3"):
            return "/usr/local/bin/python3"
    elif platform.system() == "Windows":
        # Windows
        if os.path.exists(r"C:\Python39\python.exe"):
            return r"C:\Python39\python.exe"
        elif os.path.exists(r"C:\Python310\python.exe"):
            return r"C:\Python310\python.exe"
        elif os.path.exists(r"C:\Python311\python.exe"):
            return r"C:\Python311\python.exe"
    
    # 如果无法确定系统Python，使用当前解释器
    return sys.executable

def main():
    """主函数"""
    print("=" * 60)
    print("股票分析系统 - 通用启动器")
    print("=" * 60)
    
    # 获取系统Python
    python_exe = get_system_python()
    print(f"使用Python解释器: {python_exe}")
    
    # 检查主程序文件
    target_script = "main_gui.py"
    if not os.path.exists(target_script):
        print(f"错误: 主程序文件不存在: {target_script}")
        return 1
    
    # 确保文件有执行权限
    try:
        if platform.system() != "Windows":
            os.chmod(target_script, 0o755)
    except:
        pass
    
    # 准备环境变量
    env = os.environ.copy()
    
    # 设置PYTHONPATH
    env["PYTHONPATH"] = os.getcwd()
    
    # 移除可能导致问题的环境变量
    if "PYTHONHOME" in env:
        del env["PYTHONHOME"]
    
    # 禁用pyenv
    env["PYENV_VERSION"] = "system"
    env["PYENV_DISABLE_PROMPT"] = "1"
    
    # 启动主程序
    print(f"\n正在启动股票分析系统...")
    try:
        subprocess.run([python_exe, target_script], env=env)
        return 0
    except Exception as e:
        print(f"启动失败: {str(e)}")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
    print("\n程序已退出") 