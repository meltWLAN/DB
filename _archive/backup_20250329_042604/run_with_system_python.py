#!/usr/bin/env python3
"""
特殊的Python启动器
用于绕过pyenv直接使用系统Python启动整合系统
"""

import os
import sys
import subprocess
import platform

def find_system_python():
    """尝试找到系统Python路径"""
    potential_paths = [
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3"
    ]
    
    for path in potential_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    # 如果找不到系统Python，使用当前Python
    return sys.executable

def main():
    """主函数"""
    print("========================================")
    print("   股票分析系统 - 特殊启动程序")
    print("========================================")
    
    # 系统信息
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"当前Python路径: {sys.executable}")
    
    # 找到系统Python
    python_path = find_system_python()
    print(f"使用Python: {python_path}")
    
    # 直接使用系统Python启动整合系统
    try:
        subprocess.run([python_path, "integrated_system.py"])
    except Exception as e:
        print(f"启动失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 