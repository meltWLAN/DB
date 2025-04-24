#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from datetime import datetime
import logging
from pathlib import Path
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analysis.log'),
        logging.StreamHandler()
    ]
)

class StockAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("股票分析系统")
        self.root.geometry("800x600")
        
        # 加载配置
        self.config = self.load_config()
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建配置区域
        self.create_config_frame()
        
        # 创建日志显示区域
        self.log_frame = ttk.LabelFrame(self.main_frame, text="运行日志", padding="5")
        self.log_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=20)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建控制按钮区域
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板", padding="5")
        self.control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # 创建按钮
        self.create_buttons()
        
        # 创建状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # 配置grid权重
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # 重定向日志到GUI
        self.setup_logging()
        
    def create_config_frame(self):
        # 创建配置框架
        config_frame = ttk.LabelFrame(self.main_frame, text="JoinQuant配置", padding="5")
        config_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # 用户名输入
        ttk.Label(config_frame, text="用户名:").grid(row=0, column=0, padx=5)
        self.username_var = tk.StringVar(value=self.config.get('username', ''))
        self.username_entry = ttk.Entry(config_frame, textvariable=self.username_var)
        self.username_entry.grid(row=0, column=1, padx=5)
        
        # 密码输入
        ttk.Label(config_frame, text="密码:").grid(row=0, column=2, padx=5)
        self.password_var = tk.StringVar(value=self.config.get('password', ''))
        self.password_entry = ttk.Entry(config_frame, textvariable=self.password_var, show="*")
        self.password_entry.grid(row=0, column=3, padx=5)
        
        # 保存按钮
        ttk.Button(config_frame, text="保存配置", 
                  command=self.save_config).grid(row=0, column=4, padx=5)
        
    def load_config(self):
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"加载配置文件失败: {str(e)}")
        return {}
        
    def save_config(self):
        try:
            config = {
                'username': self.username_var.get(),
                'password': self.password_var.get()
            }
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("成功", "配置已保存")
        except Exception as e:
            logging.error(f"保存配置文件失败: {str(e)}")
            messagebox.showerror("错误", f"保存配置失败: {str(e)}")
        
    def create_buttons(self):
        # 创建按钮框架
        button_frame = ttk.Frame(self.control_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # 创建按钮
        ttk.Button(button_frame, text="运行增强分析", 
                  command=self.run_enhanced_analysis).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="运行策略回测", 
                  command=self.run_backtest).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="生成分析报告", 
                  command=self.generate_report).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="运行行业轮动分析", 
                  command=self.run_sector_rotation).grid(row=0, column=3, padx=5)
        
    def setup_logging(self):
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                logging.Handler.__init__(self)
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                def append():
                    self.text_widget.configure(state='normal')
                    self.text_widget.insert(tk.END, msg + '\n')
                    self.text_widget.configure(state='disabled')
                    self.text_widget.see(tk.END)
                self.text_widget.after(0, append)
        
        # 添加GUI日志处理器
        text_handler = TextHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(text_handler)
        
    def run_enhanced_analysis(self):
        def run():
            try:
                self.status_var.set("正在运行增强分析...")
                # 设置环境变量
                os.environ['JQ_USERNAME'] = self.username_var.get()
                os.environ['JQ_PASSWORD'] = self.password_var.get()
                
                from enhanced_analysis import main as enhanced_main
                enhanced_main()
                self.status_var.set("增强分析完成")
                messagebox.showinfo("完成", "增强分析已完成")
            except Exception as e:
                logging.error(f"增强分析出错: {str(e)}")
                messagebox.showerror("错误", f"增强分析出错: {str(e)}")
            finally:
                self.status_var.set("就绪")
        
        threading.Thread(target=run, daemon=True).start()
        
    def run_backtest(self):
        def run():
            try:
                self.status_var.set("正在运行策略回测...")
                # 设置环境变量
                os.environ['JQ_USERNAME'] = self.username_var.get()
                os.environ['JQ_PASSWORD'] = self.password_var.get()
                
                from strategy_backtest import main as backtest_main
                backtest_main()
                self.status_var.set("策略回测完成")
                messagebox.showinfo("完成", "策略回测已完成")
            except Exception as e:
                logging.error(f"策略回测出错: {str(e)}")
                messagebox.showerror("错误", f"策略回测出错: {str(e)}")
            finally:
                self.status_var.set("就绪")
        
        threading.Thread(target=run, daemon=True).start()
        
    def generate_report(self):
        def run():
            try:
                self.status_var.set("正在生成分析报告...")
                from generate_report import main as report_main
                report_main()
                self.status_var.set("报告生成完成")
                messagebox.showinfo("完成", "分析报告已生成")
            except Exception as e:
                logging.error(f"报告生成出错: {str(e)}")
                messagebox.showerror("错误", f"报告生成出错: {str(e)}")
            finally:
                self.status_var.set("就绪")
        
        threading.Thread(target=run, daemon=True).start()
        
    def run_sector_rotation(self):
        def run():
            try:
                self.status_var.set("正在运行行业轮动分析...")
                # 设置环境变量
                os.environ['JQ_USERNAME'] = self.username_var.get()
                os.environ['JQ_PASSWORD'] = self.password_var.get()
                
                from sector_rotation_analysis import main as sector_main
                sector_main()
                self.status_var.set("行业轮动分析完成")
                messagebox.showinfo("完成", "行业轮动分析已完成")
            except Exception as e:
                logging.error(f"行业轮动分析出错: {str(e)}")
                messagebox.showerror("错误", f"行业轮动分析出错: {str(e)}")
            finally:
                self.status_var.set("就绪")
        
        threading.Thread(target=run, daemon=True).start()

def main():
    root = tk.Tk()
    app = StockAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 