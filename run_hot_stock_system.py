#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import tkinter as tk
from tkinter import messagebox
from datetime import datetime

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入热门股票GUI
try:
    from hot_stock_gui import HotStockGUI
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

# 设置日志
def setup_logging():
    log_dir = os.path.join(current_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'hot_stock_system_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('暴涨股系统')

def main():
    logger = setup_logging()
    logger.info("暴涨股捕捉系统启动中...")
    
    # 确保必要的目录存在
    required_dirs = ['logs', 'data', 'results']
    for directory in required_dirs:
        dir_path = os.path.join(current_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"创建目录: {dir_path}")
    
    try:
        # 创建主窗口
        root = tk.Tk()
        root.title("暴涨股捕捉系统")
        root.geometry("1200x800")
        root.minsize(1000, 700)
        
        # 设置图标(如果有)
        # icon_path = os.path.join(current_dir, 'resources', 'icon.ico')
        # if os.path.exists(icon_path):
        #     root.iconbitmap(icon_path)
        
        # 创建GUI实例
        app = HotStockGUI(root)
        
        # 添加关闭窗口事件处理
        def on_closing():
            if messagebox.askokcancel("退出", "确定要退出暴涨股捕捉系统吗?"):
                logger.info("用户关闭暴涨股捕捉系统")
                root.destroy()
                
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # 启动主循环
        logger.info("暴涨股捕捉系统界面加载完成")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"启动暴涨股捕捉系统时出错: {e}", exc_info=True)
        messagebox.showerror("错误", f"启动系统时发生错误:\n{str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 