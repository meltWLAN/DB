#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接启动器 - 绕过pyenv直接使用系统Python
"""

import os
import sys
import subprocess
import platform
import tempfile

def find_system_python():
    """查找系统Python路径"""
    if platform.system() == "Darwin":  # macOS
        if os.path.exists("/usr/bin/python3"):
            return "/usr/bin/python3"
    elif platform.system() == "Linux":
        if os.path.exists("/usr/bin/python3"):
            return "/usr/bin/python3"
    elif platform.system() == "Windows":
        return "python"  # 在Windows上通常直接使用python命令
    
    # 尝试使用which命令查找
    try:
        python_path = subprocess.check_output(["which", "python3"]).decode().strip()
        if python_path and os.path.exists(python_path):
            return python_path
    except:
        pass
    
    # 默认返回python3
    return "python3"

def create_clean_env():
    """创建干净的环境变量"""
    env = os.environ.copy()
    
    # 删除所有pyenv相关环境变量
    for var in list(env.keys()):
        if "PYENV" in var:
            del env[var]
    
    # 清理PATH中的pyenv路径
    if "PATH" in env:
        path_parts = env["PATH"].split(os.pathsep)
        clean_path_parts = [p for p in path_parts if "pyenv" not in p.lower()]
        env["PATH"] = os.pathsep.join(clean_path_parts)
    
    return env

def run_with_system_python():
    """使用系统Python运行start.py"""
    # 获取系统Python路径
    system_python = find_system_python()
    
    # 创建干净的环境变量
    env = create_clean_env()
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_script = os.path.join(current_dir, "start.py")
    
    # 确保目标脚本存在
    if not os.path.exists(target_script):
        print(f"错误: 找不到目标脚本 {target_script}")
        return 1
    
    # 打印启动信息
    print("=" * 60)
    print("直接启动器")
    print("=" * 60)
    print(f"系统Python路径: {system_python}")
    print(f"目标脚本: {target_script}")
    print("-" * 60)
    print("正在启动股票分析系统...")
    
    # 启动进程
    try:
        subprocess.run([system_python, target_script], env=env)
        return 0
    except Exception as e:
        print(f"启动失败: {str(e)}")
        return 1

def restart_with_system_python():
    """检查是否需要并重启为系统Python"""
    # 如果检测到当前运行的不是系统Python，则重新启动
    current_python = sys.executable
    system_python = find_system_python()
    
    # 检查是否与系统Python相同
    if os.path.normpath(current_python) != os.path.normpath(system_python):
        print(f"当前Python: {current_python}")
        print(f"系统Python: {system_python}")
        print("正在切换到系统Python...")
        
        # 创建一个临时脚本
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp:
            temp_script = temp.name
            current_script = os.path.abspath(__file__)
            
            # 写入临时启动脚本
            temp.write(f"""
import os
import subprocess

# 使用系统Python重新启动此脚本
subprocess.run(["{system_python}", "{current_script}"])

# 删除临时脚本
try:
    os.unlink("{temp_script}")
except:
    pass
""")
        
        # 使用系统Python执行临时脚本
        env = create_clean_env()
        subprocess.Popen([system_python, temp_script], env=env)
        return True
    
    return False

def main():
    """主函数"""
    # 检查是否需要重启为系统Python
    if restart_with_system_python():
        return 0
    
    # 使用系统Python运行start.py
    return run_with_system_python()

if __name__ == "__main__":
    sys.exit(main()) 