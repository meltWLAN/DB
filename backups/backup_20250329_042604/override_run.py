#!/usr/bin/env python3
"""
特殊的启动脚本，用于绕过pyenv拦截，启动主系统
"""

import os
import sys
import subprocess
import platform

def main():
    """主函数"""
    print("========================================")
    print("   股票分析系统 - 特殊启动脚本")
    print("========================================")
    print()
    
    # 创建必要的目录
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results/charts", exist_ok=True)
    os.makedirs("results/ma_charts", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 系统信息
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    print()
    
    # 设置环境变量
    os.environ["TUSHARE_TOKEN"] = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    
    print("正在启动主系统...")
    
    # 直接使用当前Python解释器启动主系统
    try:
        # 使用子进程直接运行主系统
        subprocess.run([sys.executable, "stock_analysis_gui.py"])
    except Exception as e:
        print(f"启动失败: {e}")
        # 尝试备选方案
        try:
            print("尝试使用备选启动方式...")
            # 直接导入和运行模块
            sys.path.insert(0, os.getcwd())
            from stock_analysis_gui import main
            main()
        except Exception as e2:
            print(f"备选启动也失败: {e2}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 