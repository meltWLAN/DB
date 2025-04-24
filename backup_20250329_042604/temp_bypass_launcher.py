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
