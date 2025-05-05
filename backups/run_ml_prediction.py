#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
机器学习预测界面启动脚本
启动股票机器学习预测系统的图形界面
"""

import os
import sys
from pathlib import Path

def main():
    """主函数，启动机器学习预测GUI"""
    # 确保模块路径正确
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # 创建必要的目录
    directories = [
        "data/stock_data",
        "logs",
        "src/models",
    ]
    
    for directory in directories:
        os.makedirs(current_dir / directory, exist_ok=True)
    
    # 导入GUI模块
    try:
        from src.visualization.ml_prediction_gui import main as run_gui
        
        # 启动GUI
        run_gui()
    except ImportError as e:
        print(f"错误：无法导入必要的模块: {e}")
        print("请确保所有依赖已正确安装")
        print("可以运行以下命令安装必要的依赖:")
        print("pip install pandas numpy matplotlib scikit-learn tkinter pillow")
        sys.exit(1)
    except Exception as e:
        print(f"启动GUI时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 