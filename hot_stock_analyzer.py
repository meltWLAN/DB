#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
暴涨股捕捉系统启动脚本
"""

import os
import sys
import logging
import tkinter as tk
from hot_stock_gui import HotStockGUI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hot_stock.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def ensure_directories():
    """确保必要的目录存在"""
    directories = [
        'logs',
        'data',
        'data/cache',
        'results',
        'results/hot_stocks'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"创建目录: {directory}")

def main():
    """主函数"""
    try:
        # 确保必要的目录存在
        ensure_directories()
        
        # 创建应用
        root = tk.Tk()
        
        # 设置图标
        try:
            if os.path.exists("assets/icon.ico"):
                root.iconbitmap("assets/icon.ico")
        except:
            pass
        
        # 初始化应用
        app = HotStockGUI(root)
        
        # 启动主循环
        root.mainloop()
        
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}", exc_info=True)
        
        # 如果有GUI，显示错误消息
        if 'root' in locals() and root:
            import tkinter.messagebox as messagebox
            messagebox.showerror("启动失败", f"应用启动失败: {str(e)}")
        
        sys.exit(1)

if __name__ == "__main__":
    main() 