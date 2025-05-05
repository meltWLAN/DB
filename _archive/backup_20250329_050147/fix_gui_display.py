#!/usr/bin/env python3
"""
优化GUI界面展示的辅助脚本
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from pathlib import Path
import subprocess
import glob
import shutil

# 设置根目录和结果目录
ROOT_DIR = Path(__file__).parent
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
MA_CHARTS_DIR = os.path.join(RESULTS_DIR, "ma_charts")

def find_latest_results():
    """查找最新的分析结果文件"""
    result_files = []
    
    # 搜索动量分析结果
    momentum_patterns = [
        os.path.join(RESULTS_DIR, "momentum_*.csv"),
        os.path.join(RESULTS_DIR, "momentum", "*.csv"),
        os.path.join(RESULTS_DIR, "*momentum*.csv")
    ]
    
    for pattern in momentum_patterns:
        files = glob.glob(pattern)
        for file in files:
            result_files.append(("momentum", file))
    
    # 搜索均线交叉结果
    ma_patterns = [
        os.path.join(RESULTS_DIR, "ma_*.csv"),
        os.path.join(RESULTS_DIR, "ma_cross", "*.csv"),
        os.path.join(RESULTS_DIR, "*ma_strategy*.csv"),
        os.path.join(RESULTS_DIR, "*crossover*.csv")
    ]
    
    for pattern in ma_patterns:
        files = glob.glob(pattern)
        for file in files:
            result_files.append(("ma", file))
    
    # 按修改时间排序
    result_files.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
    
    return result_files

def organize_results():
    """整理分析结果文件到指定目录"""
    # 创建必要的目录
    os.makedirs(os.path.join(RESULTS_DIR, "momentum"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "ma_cross"), exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(MA_CHARTS_DIR, exist_ok=True)
    
    result_files = find_latest_results()
    
    for result_type, file_path in result_files:
        if not os.path.exists(file_path):
            continue
            
        # 确定目标目录
        if result_type == "momentum":
            target_dir = os.path.join(RESULTS_DIR, "momentum")
        else:
            target_dir = os.path.join(RESULTS_DIR, "ma_cross")
            
        # 如果文件不在目标目录中，复制过去
        if not file_path.startswith(target_dir):
            target_path = os.path.join(target_dir, os.path.basename(file_path))
            try:
                shutil.copy2(file_path, target_path)
                print(f"已复制 {file_path} 到 {target_path}")
            except Exception as e:
                print(f"复制 {file_path} 时出错: {e}")

def fix_charts_directory():
    """修复图表目录结构"""
    # 查找所有PNG文件
    chart_files = glob.glob(os.path.join(RESULTS_DIR, "*.png"))
    
    for chart_file in chart_files:
        if "_momentum" in chart_file.lower():
            target_dir = CHARTS_DIR
        elif "_ma_" in chart_file.lower() or "_cross" in chart_file.lower():
            target_dir = MA_CHARTS_DIR
        else:
            # 猜测图表类型
            if "_rsi" in chart_file.lower() or "_macd" in chart_file.lower():
                target_dir = CHARTS_DIR
            else:
                continue  # 无法确定类型，跳过
                
        target_path = os.path.join(target_dir, os.path.basename(chart_file))
        try:
            if not os.path.exists(target_path):
                shutil.copy2(chart_file, target_path)
                print(f"已复制图表 {chart_file} 到 {target_path}")
        except Exception as e:
            print(f"复制图表 {chart_file} 时出错: {e}")

def check_display_issues():
    """检查可能的界面显示问题"""
    try:
        import matplotlib
        print(f"Matplotlib后端: {matplotlib.get_backend()}")
        
        # 检查是否使用了合适的GUI后端
        if matplotlib.get_backend() not in ['TkAgg', 'Qt5Agg', 'WXAgg']:
            print("警告: Matplotlib可能没有使用GUI后端，这可能导致图表显示问题")
            print("尝试设置为TkAgg后端...")
            matplotlib.use('TkAgg')
            print(f"设置后的后端: {matplotlib.get_backend()}")
    except Exception as e:
        print(f"检查Matplotlib后端时出错: {e}")
        
    # 检查是否安装了Pillow库
    try:
        from PIL import Image, ImageTk
        print("Pillow库已安装，可以正常显示图片")
    except ImportError:
        print("警告: 未安装Pillow库，这可能导致图片显示问题")
        print("建议执行: pip install pillow")

def start_enhanced_viewer():
    """启动增强的结果查看器"""
    # 尝试启动结果查看器
    try:
        # 优先使用view_results.py
        if os.path.exists("view_results.py"):
            print("启动结果查看器 view_results.py...")
            subprocess.Popen([sys.executable, "view_results.py"])
        else:
            print("未找到view_results.py，尝试其他方法...")
            # 检查是否有其他查看器
            if os.path.exists("result_viewer.py"):
                subprocess.Popen([sys.executable, "result_viewer.py"])
            else:
                print("未找到合适的结果查看器，请使用手动方法查看结果")
    except Exception as e:
        print(f"启动结果查看器时出错: {e}")

def show_help_dialog():
    """显示帮助对话框"""
    # 创建一个临时的Tk窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    help_text = """
优化界面展示的方法：

1. 结果展示问题解决方案：
   - 所有分析结果已整理到对应目录
   - 动量分析结果保存在 results/momentum/
   - 均线交叉结果保存在 results/ma_cross/
   - 图表文件保存在 results/charts/ 和 results/ma_charts/

2. 如何查看结果：
   - 双击 view_results.py 或运行 python view_results.py
   - 在结果查看器中可以浏览所有分析结果
   - 双击任何股票行可查看详细图表

3. 常见问题：
   - 如果图表显示问题，请确保安装了Pillow库
   - 如果界面响应缓慢，请关闭其他应用程序
   - 如果数据显示不完整，请尝试调整窗口大小

4. 提示：
   - 使用 python override_run.py 启动主系统
   - 使用 python view_results.py 查看分析结果
   - 所有图表和数据都保存在 results 目录
    """
    
    messagebox.showinfo("优化界面展示帮助", help_text)
    root.destroy()

def main():
    """主函数"""
    print("========================================")
    print("   优化界面展示助手")
    print("========================================")
    
    # 整理结果文件
    print("\n整理分析结果文件...")
    organize_results()
    
    # 修复图表目录
    print("\n修复图表目录结构...")
    fix_charts_directory()
    
    # 检查显示问题
    print("\n检查可能的界面显示问题...")
    check_display_issues()
    
    # 显示帮助信息
    print("\n显示帮助对话框...")
    show_help_dialog()
    
    # 启动增强的结果查看器
    print("\n启动增强的结果查看器...")
    start_enhanced_viewer()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 