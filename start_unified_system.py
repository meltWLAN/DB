#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一系统启动器
整合所有子系统并提供统一的启动界面
"""

import os
import sys
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import json
import threading

# 设置日志
def setup_logging():
    """配置日志系统"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f'unified_system_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('UnifiedSystem')

class UnifiedSystemLauncher:
    """统一系统启动器"""
    
    def __init__(self):
        """初始化启动器"""
        self.logger = setup_logging()
        self.root = tk.Tk()
        self.root.title("统一系统启动器")
        self.root.geometry("800x600")
        
        # 设置窗口图标
        try:
            if os.path.exists('resources/icon.ico'):
                self.root.iconbitmap('resources/icon.ico')
        except Exception as e:
            self.logger.warning(f"加载图标失败: {str(e)}")
        
        self.setup_gui()
        self.check_components()
        
    def setup_gui(self):
        """设置GUI界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(
            main_frame,
            text="股票分析系统",
            font=('Helvetica', 24)
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=20)
        
        # 描述
        desc_label = ttk.Label(
            main_frame,
            text="集成热门股票分析、技术指标分析、资金流向分析等功能",
            wraplength=600
        )
        desc_label.grid(row=1, column=0, columnspan=2, pady=10)
        
        # 组件状态框架
        status_frame = ttk.LabelFrame(main_frame, text="组件状态", padding="10")
        status_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # 状态指示器
        self.status_labels = {}
        components = [
            'hot_stock_scanner',
            'stock_analysis_gui',
            'momentum_analysis',
            'ma_cross_strategy',
            'financial_analysis'
        ]
        
        for i, component in enumerate(components):
            label = ttk.Label(status_frame, text=f"{component}: ")
            label.grid(row=i, column=0, sticky=tk.W, pady=2)
            
            status = ttk.Label(status_frame, text="检查中...", foreground="orange")
            status.grid(row=i, column=1, sticky=tk.W, pady=2)
            self.status_labels[component] = status
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        # 启动按钮
        self.start_button = ttk.Button(
            button_frame,
            text="启动系统",
            command=self.start_system
        )
        self.start_button.grid(row=0, column=0, padx=10)
        
        # 退出按钮
        exit_button = ttk.Button(
            button_frame,
            text="退出",
            command=self.root.quit
        )
        exit_button.grid(row=0, column=1, padx=10)
        
        # 日志框架
        log_frame = ttk.LabelFrame(main_frame, text="系统日志", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # 日志文本框
        self.log_text = tk.Text(log_frame, height=10, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # 滚动条
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text['yscrollcommand'] = scrollbar.set
        
    def check_components(self):
        """检查各组件状态"""
        components = {
            'hot_stock_scanner': 'hot_stock_scanner.py',
            'stock_analysis_gui': 'stock_analysis_gui.py',
            'momentum_analysis': 'momentum_analysis.py',
            'ma_cross_strategy': 'ma_cross_strategy.py',
            'financial_analysis': 'financial_analysis.py'
        }
        
        all_ok = True
        for component, file in components.items():
            try:
                if os.path.exists(file):
                    self.status_labels[component].config(
                        text="正常",
                        foreground="green"
                    )
                    self.log(f"组件 {component} 检查通过")
                else:
                    self.status_labels[component].config(
                        text="未找到",
                        foreground="red"
                    )
                    self.log(f"警告: 组件 {component} 未找到")
                    all_ok = False
            except Exception as e:
                self.status_labels[component].config(
                    text="错误",
                    foreground="red"
                )
                self.log(f"错误: 检查组件 {component} 时出错: {str(e)}")
                all_ok = False
        
        # 更新启动按钮状态
        self.start_button['state'] = 'normal' if all_ok else 'disabled'
        
    def start_system(self):
        """启动系统"""
        try:
            self.log("正在启动系统...")
            self.start_button['state'] = 'disabled'
            
            # 创建启动线程
            thread = threading.Thread(target=self._start_system_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.log(f"错误: 启动系统时出错: {str(e)}")
            messagebox.showerror("错误", f"启动系统时出错:\n{str(e)}")
            self.start_button['state'] = 'normal'
    
    def _start_system_thread(self):
        """在单独的线程中启动系统"""
        try:
            # 启动股票分析GUI
            self.log("正在启动股票分析GUI...")
            os.system('python stock_analysis_gui.py')
            
        except Exception as e:
            self.log(f"错误: 启动系统线程时出错: {str(e)}")
            messagebox.showerror("错误", f"启动系统线程时出错:\n{str(e)}")
        finally:
            self.root.after(0, lambda: self.start_button.config(state='normal'))
    
    def log(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.logger.info(message)

def main():
    """主函数"""
    try:
        # 记录系统启动
        logger = logging.getLogger('SystemStart')
        logger.info("系统启动中...")
        
        # 创建启动器实例
        launcher = UnifiedSystemLauncher()
        
        # 运行主循环
        launcher.root.mainloop()
        
    except Exception as e:
        logger.error(f"系统启动失败: {str(e)}")
        messagebox.showerror("错误", f"系统启动失败:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()