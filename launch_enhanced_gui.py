#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版股票分析系统启动脚本
集成增强API可靠性模块的股票分析GUI启动入口
"""

import os
import sys
import logging
import tkinter as tk
from datetime import datetime
from pathlib import Path

# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 配置日志
log_file = f"enhanced_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def show_splash_screen():
    """显示启动画面"""
    splash = tk.Tk()
    splash.title("启动中...")
    splash.geometry("400x200")
    
    # 设置窗口位置居中
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = (screen_width - 400) // 2
    y = (screen_height - 200) // 2
    splash.geometry(f"400x200+{x}+{y}")
    
    # 设置窗口样式
    splash.configure(bg="#f0f0f0")
    splash.overrideredirect(True)  # 无边框
    
    # 添加标题
    title_label = tk.Label(
        splash, 
        text="增强版股票分析系统", 
        font=("Arial", 18, "bold"),
        bg="#f0f0f0"
    )
    title_label.pack(pady=(20, 10))
    
    # 添加状态信息
    status_label = tk.Label(
        splash, 
        text="正在加载模块...", 
        font=("Arial", 10),
        bg="#f0f0f0"
    )
    status_label.pack(pady=5)
    
    # 添加进度条
    progress = tk.Canvas(splash, width=300, height=20, bg="white", highlightthickness=1)
    progress.pack(pady=10)
    
    # 更新状态函数
    def update_status(message, percent):
        status_label.config(text=message)
        progress.delete("progress")
        progress.create_rectangle(2, 2, 2 + percent * 3, 18, fill="#007bff", tags="progress")
        splash.update()
    
    # 显示窗口并开始加载
    splash.update()
    
    update_status("检查API可靠性模块...", 10)
    
    # 检查增强API模块
    try:
        from enhance_api_reliability import (
            enhance_get_stock_name, 
            enhance_get_stock_names_batch,
            enhance_get_stock_industry
        )
        has_enhanced_api = True
        update_status("增强API可靠性模块已加载", 30)
    except ImportError:
        has_enhanced_api = False
        update_status("增强API可靠性模块未找到，将使用基本功能", 30)
    
    update_status("加载GUI控制器...", 50)
    
    # 加载控制器
    try:
        if has_enhanced_api:
            from enhanced_gui_controller import EnhancedGuiController
            update_status("增强版GUI控制器已加载", 70)
        else:
            from gui_controller import GuiController
            update_status("基本GUI控制器已加载", 70)
    except ImportError as e:
        update_status(f"加载GUI控制器失败: {str(e)}", 70)
        logger.error(f"加载GUI控制器失败: {str(e)}")
    
    update_status("初始化界面...", 90)
    
    # 加载主界面模块
    try:
        from stock_analysis_gui import StockAnalysisGUI
        update_status("准备就绪，正在启动界面...", 100)
    except ImportError as e:
        update_status(f"加载界面模块失败: {str(e)}", 90)
        logger.error(f"加载界面模块失败: {str(e)}")
    
    # 延时关闭启动画面
    splash.after(1000, splash.destroy)
    
    return splash

def main():
    """主函数"""
    logger.info("启动增强版股票分析系统")
    
    # 显示启动画面
    splash = show_splash_screen()
    
    # 启动主程序
    def start_main_app():
        try:
            from stock_analysis_gui import StockAnalysisGUI
            
            root = tk.Tk()
            app = StockAnalysisGUI(root)
            root.mainloop()
            
        except Exception as e:
            logger.error(f"启动程序失败: {str(e)}", exc_info=True)
            # 如果出现错误，显示错误信息
            error_root = tk.Tk()
            error_root.title("启动错误")
            error_root.geometry("500x300")
            tk.Label(
                error_root, 
                text="启动程序失败", 
                font=("Arial", 16, "bold")
            ).pack(pady=(20, 10))
            
            tk.Label(
                error_root, 
                text=str(e), 
                font=("Arial", 10),
                wraplength=450
            ).pack(pady=10)
            
            tk.Button(
                error_root, 
                text="确定", 
                command=error_root.destroy
            ).pack(pady=20)
            
            error_root.mainloop()
    
    # 在启动画面关闭后启动主程序
    splash.after(1000, start_main_app)
    splash.mainloop()
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序发生未处理的异常: {str(e)}", exc_info=True)
        sys.exit(1) 