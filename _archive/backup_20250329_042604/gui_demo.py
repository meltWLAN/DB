#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票推荐系统GUI界面
提供交互式界面来获取和展示股票推荐
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from datetime import datetime
import os
from pathlib import Path

from src.enhanced.config.settings import LOG_DIR
from src.enhanced.strategies.stock_picker import StockPicker

# 配置日志
log_file = os.path.join(LOG_DIR, f"gui_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class StockRecommenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("股票推荐系统")
        self.root.geometry("800x600")
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建标题
        title_label = ttk.Label(
            self.main_frame,
            text="智能股票推荐系统",
            font=("Helvetica", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # 创建参数设置区域
        self.create_parameter_frame()
        
        # 创建推荐按钮
        self.recommend_button = ttk.Button(
            self.main_frame,
            text="获取推荐",
            command=self.get_recommendations
        )
        self.recommend_button.grid(row=2, column=0, columnspan=2, pady=10)
        
        # 创建结果显示区域
        self.create_result_frame()
        
        # 初始化选股器
        self.stock_picker = StockPicker()
        
    def create_parameter_frame(self):
        """创建参数设置区域"""
        param_frame = ttk.LabelFrame(self.main_frame, text="参数设置", padding="5")
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 推荐数量设置
        ttk.Label(param_frame, text="推荐数量:").grid(row=0, column=0, padx=5)
        self.top_n = tk.StringVar(value="5")
        ttk.Entry(param_frame, textvariable=self.top_n, width=10).grid(row=0, column=1, padx=5)
        
    def create_result_frame(self):
        """创建结果显示区域"""
        result_frame = ttk.LabelFrame(self.main_frame, text="推荐结果", padding="5")
        result_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # 创建表格
        columns = ("股票代码", "当前价格", "涨跌幅", "风险分数", "推荐理由")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings")
        
        # 设置列标题
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
    def get_recommendations(self):
        """获取股票推荐"""
        try:
            # 清空现有结果
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # 获取推荐数量
            try:
                top_n = int(self.top_n.get())
            except ValueError:
                messagebox.showerror("错误", "请输入有效的推荐数量")
                return
            
            # 获取推荐
            logger.info("开始获取股票推荐...")
            recommendations = self.stock_picker.get_stock_recommendations(top_n=top_n)
            
            if not recommendations:
                messagebox.showinfo("提示", "未找到符合条件的股票")
                return
            
            # 显示结果
            for stock in recommendations:
                self.tree.insert("", tk.END, values=(
                    stock['stock_code'],
                    f"{stock['current_price']:.2f}",
                    f"{stock['change_pct']*100:.2f}%",
                    f"{stock['risk_score']:.2f}",
                    stock['recommendation_reason']
                ))
            
            logger.info(f"成功获取 {len(recommendations)} 只股票推荐")
            
        except Exception as e:
            logger.error(f"获取推荐时出错: {str(e)}", exc_info=True)
            messagebox.showerror("错误", f"获取推荐时出错: {str(e)}")

def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = StockRecommenderGUI(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"GUI程序运行出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 