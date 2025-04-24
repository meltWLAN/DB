#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票分析系统GUI界面
整合动量分析与均线交叉策略的图形界面
"""

import os
import sys
import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from PIL import Image, ImageTk

# 导入GUI控制器
from gui_controller import GuiController

# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 导入项目配置
from src.enhanced.config.settings import LOG_DIR, DATA_DIR, RESULTS_DIR

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志
log_file = os.path.join(LOG_DIR, f"stock_analysis_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class StockAnalysisGUI:
    """股票分析系统GUI界面类"""
    
    def __init__(self, root):
        """初始化GUI界面"""
        self.root = root
        self.root.title("股票分析系统")
        self.root.geometry("1200x800")
        
        # 初始化控制器
        self.controller = GuiController(use_tushare=True)
        
        # 初始化数据
        self.status_message = tk.StringVar()
        self.status_message.set("就绪")
        self.industries = self.controller.get_stock_industries()
        
        # 创建UI组件
        self.create_menu()
        self.create_main_frame()
        self.create_status_bar()
        
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="导入数据", command=self.import_data)
        file_menu.add_command(label="导出结果", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 分析菜单
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="分析", menu=analysis_menu)
        analysis_menu.add_command(label="动量分析", command=self.run_momentum_analysis)
        analysis_menu.add_command(label="均线交叉策略", command=self.run_ma_cross_strategy)
        analysis_menu.add_command(label="组合策略", command=self.run_combined_strategy)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="刷新数据", command=self.refresh_data)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用帮助", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
        
    def create_main_frame(self):
        """创建主框架"""
        # 创建选项卡控件
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建动量分析选项卡
        self.momentum_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.momentum_tab, text="动量分析")
        self.setup_momentum_tab()
        
        # 创建均线交叉选项卡
        self.ma_cross_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ma_cross_tab, text="均线交叉策略")
        self.setup_ma_cross_tab()
        
        # 创建组合策略选项卡
        self.combined_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.combined_tab, text="组合策略")
        self.setup_combined_tab()
        
        # 创建市场概览选项卡
        self.overview_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_tab, text="市场概览")
        self.setup_overview_tab()
        
        # 创建日志选项卡
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="日志")
        self.setup_log_tab()
        
    def setup_momentum_tab(self):
        """设置动量分析选项卡"""
        # 创建左右分割窗口
        paned = ttk.PanedWindow(self.momentum_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧参数设置区域
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # 参数设置框架
        param_frame = ttk.LabelFrame(left_frame, text="分析参数")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 样本大小设置
        ttk.Label(param_frame, text="样本数量:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.sample_size_var = tk.StringVar(value="100")
        ttk.Entry(param_frame, textvariable=self.sample_size_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # 行业选择
        ttk.Label(param_frame, text="行业选择:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.industry_var = tk.StringVar(value="全部")
        industry_combo = ttk.Combobox(param_frame, textvariable=self.industry_var, values=self.industries)
        industry_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # 最低得分
        ttk.Label(param_frame, text="最低得分:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.min_score_var = tk.StringVar(value="60")
        ttk.Entry(param_frame, textvariable=self.min_score_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # 分析按钮
        ttk.Button(param_frame, text="运行分析", command=self.run_momentum_analysis).grid(
            row=3, column=0, columnspan=2, pady=10)
        
        # 右侧结果区域
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)
        
        # 结果表格
        self.momentum_result_frame = ttk.LabelFrame(right_frame, text="分析结果")
        self.momentum_result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建表格
        columns = ("股票代码", "股票名称", "行业", "当前价格", "20日动量", "RSI", "MACD", "成交量比", "得分")
        self.momentum_tree = ttk.Treeview(self.momentum_result_frame, columns=columns, show="headings", height=20)
        
        # 设置列标题
        for col in columns:
            self.momentum_tree.heading(col, text=col)
        
        # 设置列宽
        self.momentum_tree.column("股票代码", width=100)
        self.momentum_tree.column("股票名称", width=100)
        self.momentum_tree.column("行业", width=100)
        self.momentum_tree.column("当前价格", width=80)
        self.momentum_tree.column("20日动量", width=80)
        self.momentum_tree.column("RSI", width=60)
        self.momentum_tree.column("MACD", width=80)
        self.momentum_tree.column("成交量比", width=80)
        self.momentum_tree.column("得分", width=60)
        
        # 添加滚动条
        momentum_scrollbar = ttk.Scrollbar(self.momentum_result_frame, orient=tk.VERTICAL, command=self.momentum_tree.yview)
        self.momentum_tree.configure(yscrollcommand=momentum_scrollbar.set)
        
        # 布局
        self.momentum_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        momentum_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 表格双击事件
        self.momentum_tree.bind("<Double-1>", self.on_momentum_tree_double_click)
    
    def setup_ma_cross_tab(self):
        """设置均线交叉策略选项卡"""
        # 创建左右分割窗口
        paned = ttk.PanedWindow(self.ma_cross_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧参数设置区域
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # 参数设置框架
        param_frame = ttk.LabelFrame(left_frame, text="策略参数")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 短期均线参数
        ttk.Label(param_frame, text="短期均线:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.short_ma_var = tk.StringVar(value="5")
        ttk.Entry(param_frame, textvariable=self.short_ma_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # 长期均线参数
        ttk.Label(param_frame, text="长期均线:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.long_ma_var = tk.StringVar(value="20")
        ttk.Entry(param_frame, textvariable=self.long_ma_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # 止损比例
        ttk.Label(param_frame, text="止损比例(%):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.stop_loss_var = tk.StringVar(value="5")
        ttk.Entry(param_frame, textvariable=self.stop_loss_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # 初始资金
        ttk.Label(param_frame, text="初始资金:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.initial_capital_var = tk.StringVar(value="100000")
        ttk.Entry(param_frame, textvariable=self.initial_capital_var, width=10).grid(row=3, column=1, padx=5, pady=5)
        
        # 行业选择
        ttk.Label(param_frame, text="行业选择:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.ma_industry_var = tk.StringVar(value="全部")
        ma_industry_combo = ttk.Combobox(param_frame, textvariable=self.ma_industry_var, values=self.industries)
        ma_industry_combo.grid(row=4, column=1, padx=5, pady=5)
        
        # 样本大小
        ttk.Label(param_frame, text="样本数量:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.ma_sample_size_var = tk.StringVar(value="100")
        ttk.Entry(param_frame, textvariable=self.ma_sample_size_var, width=10).grid(row=5, column=1, padx=5, pady=5)
        
        # 分析按钮
        ttk.Button(param_frame, text="运行策略", command=self.run_ma_cross_strategy).grid(
            row=6, column=0, columnspan=2, pady=10)
        
        # 右侧结果区域
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)
        
        # 结果表格
        self.ma_result_frame = ttk.LabelFrame(right_frame, text="策略结果")
        self.ma_result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建表格
        columns = ("股票代码", "股票名称", "行业", "收盘价", "当前信号", "短期均线", "长期均线", "回测收益", "年化收益", "最大回撤", "胜率")
        self.ma_tree = ttk.Treeview(self.ma_result_frame, columns=columns, show="headings", height=20)
        
        # 设置列标题
        for col in columns:
            self.ma_tree.heading(col, text=col)
        
        # 设置列宽
        self.ma_tree.column("股票代码", width=100)
        self.ma_tree.column("股票名称", width=100)
        self.ma_tree.column("行业", width=100)
        self.ma_tree.column("收盘价", width=80)
        self.ma_tree.column("当前信号", width=80)
        self.ma_tree.column("短期均线", width=80)
        self.ma_tree.column("长期均线", width=80)
        self.ma_tree.column("回测收益", width=80)
        self.ma_tree.column("年化收益", width=80)
        self.ma_tree.column("最大回撤", width=80)
        self.ma_tree.column("胜率", width=60)
        
        # 添加滚动条
        ma_scrollbar = ttk.Scrollbar(self.ma_result_frame, orient=tk.VERTICAL, command=self.ma_tree.yview)
        self.ma_tree.configure(yscrollcommand=ma_scrollbar.set)
        
        # 布局
        self.ma_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ma_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 表格双击事件
        self.ma_tree.bind("<Double-1>", self.on_ma_tree_double_click)
    
    def setup_combined_tab(self):
        """设置组合策略选项卡"""
        # 创建左右分割窗口
        paned = ttk.PanedWindow(self.combined_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧参数设置区域
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # 参数设置框架
        param_frame = ttk.LabelFrame(left_frame, text="组合策略参数")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 动量分析参数区域
        momentum_frame = ttk.LabelFrame(param_frame, text="动量分析参数")
        momentum_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(momentum_frame, text="最低得分:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.combined_min_score_var = tk.StringVar(value="60")
        ttk.Entry(momentum_frame, textvariable=self.combined_min_score_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # 均线交叉参数区域
        ma_frame = ttk.LabelFrame(param_frame, text="均线交叉参数")
        ma_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(ma_frame, text="短期/长期均线:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.combined_short_ma_var = tk.StringVar(value="5")
        ttk.Entry(ma_frame, textvariable=self.combined_short_ma_var, width=5).grid(row=0, column=1, padx=2, pady=5)
        ttk.Label(ma_frame, text="/").grid(row=0, column=2, padx=1, pady=5)
        self.combined_long_ma_var = tk.StringVar(value="20")
        ttk.Entry(ma_frame, textvariable=self.combined_long_ma_var, width=5).grid(row=0, column=3, padx=2, pady=5)
        
        # 通用参数区域
        common_frame = ttk.LabelFrame(param_frame, text="通用参数")
        common_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(common_frame, text="行业选择:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.combined_industry_var = tk.StringVar(value="全部")
        combined_industry_combo = ttk.Combobox(common_frame, textvariable=self.combined_industry_var, values=self.industries)
        combined_industry_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(common_frame, text="样本数量:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.combined_sample_size_var = tk.StringVar(value="50")
        ttk.Entry(common_frame, textvariable=self.combined_sample_size_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # 策略权重设置
        weight_frame = ttk.LabelFrame(param_frame, text="策略权重")
        weight_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(weight_frame, text="动量权重:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.momentum_weight_var = tk.StringVar(value="50")
        ttk.Scale(weight_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                 variable=self.momentum_weight_var).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(weight_frame, textvariable=self.momentum_weight_var).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(weight_frame, text="均线权重:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.ma_weight_var = tk.StringVar(value="50")
        ttk.Scale(weight_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                 variable=self.ma_weight_var).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(weight_frame, textvariable=self.ma_weight_var).grid(row=1, column=2, padx=5, pady=5)
        
        # 分析按钮
        ttk.Button(param_frame, text="运行组合策略", command=self.run_combined_strategy).pack(pady=10)
        
        # 右侧结果区域
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)
        
        # 结果表格
        self.combined_result_frame = ttk.LabelFrame(right_frame, text="组合策略结果")
        self.combined_result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建表格
        columns = ("股票代码", "股票名称", "行业", "收盘价", "动量得分", "均线信号", "综合评分", "排名")
        self.combined_tree = ttk.Treeview(self.combined_result_frame, columns=columns, show="headings")
        
        # 设置列标题
        for col in columns:
            self.combined_tree.heading(col, text=col)
            self.combined_tree.column(col, width=100)
        
        # 添加滚动条
        combined_scrollbar = ttk.Scrollbar(self.combined_result_frame, orient=tk.VERTICAL, command=self.combined_tree.yview)
        self.combined_tree.configure(yscrollcommand=combined_scrollbar.set)
        
        # 布局
        self.combined_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        combined_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 表格双击事件
        self.combined_tree.bind("<Double-1>", self.on_combined_tree_double_click)
    
    def setup_overview_tab(self):
        """设置市场概览选项卡"""
        # 创建市场概览框架
        overview_frame = ttk.Frame(self.overview_tab)
        overview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建刷新按钮
        ttk.Button(overview_frame, text="刷新市场数据", command=self.refresh_market_overview).pack(pady=10)
        
        # 创建概览信息区域
        info_frame = ttk.LabelFrame(overview_frame, text="市场概况")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 市场信息标签
        self.market_info_text = ScrolledText(info_frame, height=10, width=80)
        self.market_info_text.pack(fill=tk.X, padx=5, pady=5)
        self.market_info_text.insert(tk.END, "点击「刷新市场数据」按钮获取最新市场概况...")
        self.market_info_text.config(state=tk.DISABLED)
        
        # 创建行业表现区域
        industry_frame = ttk.LabelFrame(overview_frame, text="行业表现")
        industry_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 行业表现表格
        columns = ("行业名称", "涨跌幅", "上涨家数", "下跌家数", "总家数", "领涨股", "领跌股")
        self.industry_tree = ttk.Treeview(industry_frame, columns=columns, show="headings")
        
        # 设置列标题
        for col in columns:
            self.industry_tree.heading(col, text=col)
            self.industry_tree.column(col, width=120)
        
        # 添加滚动条
        industry_scrollbar = ttk.Scrollbar(industry_frame, orient=tk.VERTICAL, command=self.industry_tree.yview)
        self.industry_tree.configure(yscrollcommand=industry_scrollbar.set)
        
        # 布局
        self.industry_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        industry_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def setup_log_tab(self):
        """设置日志选项卡"""
        # 创建日志文本区域
        self.log_text = ScrolledText(self.log_tab, height=20, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加日志处理器
        self.log_handler = LogHandler(self.log_text)
        logger.addHandler(self.log_handler)
        
        # 清空日志按钮
        ttk.Button(self.log_tab, text="清空日志", command=self.clear_log).pack(pady=5)
        
    def create_status_bar(self):
        """创建状态栏"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 状态标签
        ttk.Label(status_frame, textvariable=self.status_message, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        
        # 版本信息
        ttk.Label(status_frame, text="v1.0.0", relief=tk.SUNKEN).pack(side=tk.RIGHT)
        
    # 功能方法
    def import_data(self):
        """导入数据"""
        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                self.status_message.set(f"正在导入数据: {file_path}...")
                # 这里实现数据导入逻辑
                self.status_message.set(f"数据导入成功: {file_path}")
                messagebox.showinfo("导入成功", f"成功导入数据文件: {file_path}")
            except Exception as e:
                logger.error(f"导入数据失败: {str(e)}", exc_info=True)
                self.status_message.set("导入数据失败")
                messagebox.showerror("导入失败", f"导入数据时出错: {str(e)}")
                
    def export_results(self):
        """导出结果"""
        # 检查当前选中的选项卡
        current_tab = self.notebook.index(self.notebook.select())
        result_type = ""
        
        if current_tab == 0:  # 动量分析选项卡
            result_type = "动量分析"
        elif current_tab == 1:  # 均线交叉选项卡
            result_type = "均线策略"
        elif current_tab == 2:  # 组合策略选项卡
            result_type = "组合策略"
        else:
            messagebox.showinfo("导出提示", "请先切换到动量分析、均线交叉策略或组合策略选项卡")
            return
            
        file_path = filedialog.asksaveasfilename(
            title=f"保存{result_type}结果",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx")]
        )
        
        if file_path:
            self.controller.export_results(result_type, file_path, self.gui_callback)
                
    def run_momentum_analysis(self):
        """运行动量分析"""
        try:
            # 获取参数
            sample_size = int(self.sample_size_var.get())
            industry = self.industry_var.get()
            min_score = int(self.min_score_var.get())
            
            # 清空当前结果
            for item in self.momentum_tree.get_children():
                self.momentum_tree.delete(item)
                
            # 运行分析
            self.status_message.set("正在运行动量分析...")
            self.controller.run_momentum_analysis(
                sample_size=sample_size, 
                industry=industry, 
                min_score=min_score, 
                gui_callback=self.gui_callback
            )
            
        except Exception as e:
            logger.error(f"动量分析失败: {str(e)}", exc_info=True)
            self.status_message.set("动量分析失败")
            messagebox.showerror("分析失败", f"动量分析时出错: {str(e)}")
            
    def run_ma_cross_strategy(self):
        """运行均线交叉策略"""
        try:
            # 获取参数
            short_ma = int(self.short_ma_var.get())
            long_ma = int(self.long_ma_var.get())
            stop_loss_pct = float(self.stop_loss_var.get())
            initial_capital = float(self.initial_capital_var.get())
            industry = self.ma_industry_var.get()
            sample_size = int(self.ma_sample_size_var.get())
            
            # 清空当前结果
            for item in self.ma_tree.get_children():
                self.ma_tree.delete(item)
                
            # 运行策略
            self.status_message.set("正在运行均线交叉策略...")
            self.controller.run_ma_cross_strategy(
                short_ma=short_ma,
                long_ma=long_ma,
                initial_capital=initial_capital,
                stop_loss_pct=stop_loss_pct,
                sample_size=sample_size,
                industry=industry,
                gui_callback=self.gui_callback
            )
            
        except Exception as e:
            logger.error(f"均线交叉策略失败: {str(e)}", exc_info=True)
            self.status_message.set("均线交叉策略失败")
            messagebox.showerror("策略失败", f"均线交叉策略时出错: {str(e)}")
            
    def run_combined_strategy(self):
        """运行组合策略分析"""
        try:
            # 获取参数
            sample_size = int(self.combined_sample_size_var.get())
            industry = self.combined_industry_var.get()
            min_score = int(self.combined_min_score_var.get())
            short_ma = int(self.combined_short_ma_var.get())
            long_ma = int(self.combined_long_ma_var.get())
            momentum_weight = float(self.momentum_weight_var.get()) / 100
            ma_weight = float(self.ma_weight_var.get()) / 100
            
            # 清空当前结果
            for item in self.combined_tree.get_children():
                self.combined_tree.delete(item)
                
            # 设置状态
            self.status_message.set("正在运行组合策略分析...")
            
            # 使用控制器的组合策略功能
            self.controller.run_combined_strategy(
                momentum_weight=momentum_weight,
                ma_weight=ma_weight,
                sample_size=sample_size,
                industry=industry,
                min_score=min_score,
                short_ma=short_ma,
                long_ma=long_ma,
                initial_capital=100000,
                stop_loss_pct=5,
                gui_callback=self.gui_callback
            )
            
        except Exception as e:
            logger.error(f"启动组合策略分析失败: {str(e)}", exc_info=True)
            self.status_message.set("组合策略分析失败")
            messagebox.showerror("分析失败", f"组合策略分析时出错: {str(e)}")
    
    def _combine_strategy_results(self, momentum_results, ma_results, momentum_weight, ma_weight):
        """合并动量分析和均线交叉策略的结果"""
        try:
            # 创建股票代码到结果的映射
            momentum_map = {r['ts_code']: r for r in momentum_results}
            ma_map = {r['ts_code']: r for r in ma_results}
            
            # 获取所有股票代码
            all_stocks = set(momentum_map.keys()) | set(ma_map.keys())
            
            # 合并结果
            combined_results = []
            for stock in all_stocks:
                # 获取动量分析结果
                m_result = momentum_map.get(stock, {})
                # 获取均线交叉结果
                ma_result = ma_map.get(stock, {})
                
                # 只有当两个策略都有结果时才合并
                if m_result and ma_result:
                    # 标准化动量分数 (0-100)
                    momentum_score = m_result.get('score', 0)
                    
                    # 标准化均线交叉信号 (-100 to 100)
                    ma_signal = ma_result.get('current_signal', '')
                    ma_score = 0
                    if ma_signal == '买入信号':
                        ma_score = 100
                    elif ma_signal == '卖出信号':
                        ma_score = -100
                    elif ma_signal == '持有多头':
                        ma_score = 50
                    elif ma_signal == '持有空头':
                        ma_score = -50
                    
                    # 计算加权综合分数
                    combined_score = momentum_score * momentum_weight + (ma_score + 100) / 2 * ma_weight
                    
                    # 创建组合结果
                    combined_results.append({
                        'ts_code': stock,
                        'name': m_result.get('name') or ma_result.get('name', ''),
                        'industry': m_result.get('industry') or ma_result.get('industry', ''),
                        'close': m_result.get('close') or ma_result.get('close', 0),
                        'momentum_score': momentum_score,
                        'ma_signal': ma_signal,
                        'combined_score': combined_score
                    })
            
            # 按综合得分排序
            combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # 添加排名
            for i, r in enumerate(combined_results):
                r['rank'] = i + 1
                
            return combined_results
        except Exception as e:
            logger.error(f"合并策略结果失败: {str(e)}", exc_info=True)
            return []
    
    def _update_combined_results_ui(self, results):
        """更新组合策略结果UI"""
        if not isinstance(results, list) or not results:
            return
            
        # 清空当前结果
        for item in self.combined_tree.get_children():
            self.combined_tree.delete(item)
            
        # 添加新结果
        for result in results:
            try:
                values = (
                    result.get('ts_code', ''),
                    result.get('name', ''),
                    result.get('industry', ''),
                    f"{result.get('close', 0):.2f}",
                    f"{result.get('momentum_score', 0):.1f}",
                    result.get('ma_signal', '无信号'),
                    f"{result.get('combined_score', 0):.1f}",
                    str(result.get('rank', 0))
                )
                self.combined_tree.insert("", tk.END, values=values)
            except Exception as e:
                logger.error(f"插入组合策略结果时出错: {str(e)}", exc_info=True)
                
    def refresh_data(self):
        """刷新数据"""
        try:
            self.status_message.set("正在刷新数据...")
            
            # 更新行业列表
            self.industries = self.controller.get_stock_industries()
            
            self.status_message.set("数据刷新完成")
            messagebox.showinfo("刷新完成", "数据已成功刷新")
        except Exception as e:
            logger.error(f"刷新数据失败: {str(e)}", exc_info=True)
            self.status_message.set("刷新数据失败")
            messagebox.showerror("刷新失败", f"刷新数据时出错: {str(e)}")
            
    def refresh_market_overview(self):
        """刷新市场概览"""
        try:
            self.status_message.set("正在刷新市场概览...")
            
            # 清空当前结果
            for item in self.industry_tree.get_children():
                self.industry_tree.delete(item)
                
            # 获取市场概览数据
            self.controller.get_market_overview(self.gui_callback)
            
        except Exception as e:
            logger.error(f"刷新市场概览失败: {str(e)}", exc_info=True)
            self.status_message.set("刷新市场概览失败")
            messagebox.showerror("刷新失败", f"刷新市场概览时出错: {str(e)}")
            
    def on_momentum_tree_double_click(self, event):
        """双击动量分析结果表格的处理函数"""
        item = self.momentum_tree.selection()[0]
        stock_code = self.momentum_tree.item(item, "values")[0]
        
        # 打开图表
        chart_path = os.path.join(RESULTS_DIR, "charts", f"{stock_code}_momentum.png")
        if os.path.exists(chart_path):
            self.show_chart(chart_path)
        else:
            messagebox.showinfo("提示", f"未找到{stock_code}的图表文件")
    
    def on_ma_tree_double_click(self, event):
        """双击均线交叉结果表格的处理函数"""
        item = self.ma_tree.selection()[0]
        stock_code = self.ma_tree.item(item, "values")[0]
        
        # 打开图表
        chart_path = os.path.join(RESULTS_DIR, "ma_charts", f"{stock_code}_ma_cross.png")
        if os.path.exists(chart_path):
            self.show_chart(chart_path)
        else:
            messagebox.showinfo("提示", f"未找到{stock_code}的图表文件")
    
    def on_combined_tree_double_click(self, event):
        """双击组合策略结果表格的处理函数"""
        if not self.combined_tree.selection():
            return
            
        item = self.combined_tree.selection()[0]
        stock_code = self.combined_tree.item(item, "values")[0]
        
        # 创建新窗口显示详细分析
        self.show_combined_analysis_details(stock_code)
    
    def show_chart(self, chart_path):
        """显示图表"""
        if not chart_path or not os.path.exists(chart_path):
            raise ValueError(f"图表路径无效: {chart_path}")
            
        try:
            # 创建一个新窗口显示图表
            chart_window = tk.Toplevel(self.root)
            chart_window.title("股票分析图表")
            chart_window.geometry("1000x600")
            
            # 加载图像
            image = Image.open(chart_path)
            # 调整图像大小以适应窗口
            image = image.resize((980, 580), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # 在Label中显示图像
            label = tk.Label(chart_window, image=photo)
            label.image = photo  # 保持引用以防止被垃圾回收
            label.pack(padx=10, pady=10)
            
        except Exception as e:
            logger.error(f"显示图表时出错: {str(e)}", exc_info=True)
            raise
    
    def gui_callback(self, type, data):
        """处理从后台线程接收的回调并在主线程中更新GUI"""
        print(f"收到GUI回调: 类型 = {type}, 数据长度 = {len(data) if isinstance(data, list) else '非列表'}")
        logger.info(f"收到GUI回调: 类型 = {type}, 数据长度 = {len(data) if isinstance(data, list) else '非列表'}")
        
        # 使用after方法确保在主线程中更新GUI
        def update_gui():
            if type == "状态":
                self.status_message.set(data)
                print(f"状态更新: {data}")
                logger.info(f"状态更新: {data}")
            elif type == "结果":
                # 确定是哪种分析的结果
                if isinstance(data, list) and len(data) > 0:
                    print(f"收到结果数据: {len(data)}条")
                    logger.info(f"收到结果数据: {len(data)}条")
                    print(f"数据样例: {data[0]}")
                    logger.info(f"数据样例: {data[0]}")
                    
                    # 获取当前选中的选项卡名称
                    current_tab_id = self.notebook.select()
                    if not current_tab_id:
                        logger.warning("无法获取当前选项卡")
                        return
                        
                    current_tab_text = self.notebook.tab(current_tab_id, "text")
                    print(f"当前选项卡: {current_tab_text}")
                    logger.info(f"当前选项卡: {current_tab_text}")
                    
                    # 检查结果类型
                    first_item = data[0]
                    has_momentum = 'momentum_score' in first_item or 'score' in first_item
                    has_ma_signal = 'current_signal' in first_item
                    has_combined = 'combined_score' in first_item
                    
                    print(f"结果类型检测: 动量={has_momentum}, 均线={has_ma_signal}, 组合={has_combined}")
                    logger.info(f"结果类型检测: 动量={has_momentum}, 均线={has_ma_signal}, 组合={has_combined}")
                    print(f"数据键: {first_item.keys()}")
                    logger.info(f"数据键: {first_item.keys()}")
                    
                    # 根据结果类型更新相应的UI
                    if has_combined:
                        print("更新组合策略UI")
                        self._update_combined_results_ui(data)
                        self.status_message.set(f"组合策略分析完成，共分析{len(data)}支股票")
                        # 确保选中组合策略选项卡
                        for i in range(self.notebook.index("end")):
                            if self.notebook.tab(i, "text") == "组合策略":
                                self.notebook.select(i)
                                break
                    elif has_momentum and not has_ma_signal:
                        print("更新动量分析UI")
                        self._update_momentum_results_ui(data)
                        self.status_message.set(f"动量分析完成，发现{len(data)}支强势股票")
                        # 确保选中动量分析选项卡
                        for i in range(self.notebook.index("end")):
                            if self.notebook.tab(i, "text") == "动量分析":
                                self.notebook.select(i)
                                break
                    elif has_ma_signal:
                        print("更新均线交叉UI")
                        self._update_ma_results_ui(data)
                        self.status_message.set(f"均线交叉分析完成，发现{len(data)}支符合条件的股票")
                        # 确保选中均线交叉选项卡
                        for i in range(self.notebook.index("end")):
                            if self.notebook.tab(i, "text") == "均线交叉":
                                self.notebook.select(i)
                                break
                    else:
                        print(f"未知结果类型: {first_item.keys()}")
                        logger.warning(f"未知结果类型: {first_item.keys()}")
            elif type == "图表":
                # 处理图表显示
                try:
                    print(f"显示图表: {data}")
                    self._display_chart(data)
                    self.status_message.set("图表加载完成")
                except Exception as e:
                    self.status_message.set(f"加载图表时出错: {str(e)}")
                    logger.error(f"加载图表时出错: {str(e)}", exc_info=True)
                    
        # 在主线程中执行UI更新
        self.root.after(0, update_gui)
    
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
        
    def show_help(self):
        """显示帮助信息"""
        help_text = """
        股票分析系统使用帮助：
        
        1. 动量分析：
           - 选择样本数量和行业
           - 设置最低得分阈值
           - 点击"运行分析"按钮执行
           - 结果将显示在表格中
           - 双击表格行查看详细图表
           
        2. 均线交叉策略：
           - 设置短期均线和长期均线参数
           - 设置止损比例和初始资金
           - 选择行业和样本数量
           - 点击"运行策略"按钮执行
           - 结果将显示在表格中
           - 双击表格行查看详细图表
           
        3. 市场概览：
           - 点击"刷新市场数据"获取最新市场情况
           - 查看行业表现和市场统计数据
           
        4. 导出结果：
           - 在需要导出的选项卡中，从"文件"菜单选择"导出结果"
           - 选择保存位置和格式
        """
        messagebox.showinfo("使用帮助", help_text)
        
    def show_about(self):
        """显示关于信息"""
        about_text = """
        股票分析系统 v1.0.0
        
        集成动量分析和均线交叉策略的综合分析工具
        
        功能特点:
        - 基于技术指标的动量分析
        - 基于均线交叉的趋势跟踪策略
        - 市场和行业概览
        - 详细的图表和回测数据
        
        技术支持: tech@stockanalysis.com
        """
        messagebox.showinfo("关于", about_text)

    def show_combined_analysis_details(self, stock_code):
        """显示股票的组合分析详情"""
        details_window = tk.Toplevel(self.root)
        details_window.title(f"{stock_code} 组合分析详情")
        details_window.geometry("1000x800")
        
        # 创建选项卡
        details_notebook = ttk.Notebook(details_window)
        details_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 动量分析选项卡
        momentum_frame = ttk.Frame(details_notebook)
        details_notebook.add(momentum_frame, text="动量分析")
        
        # 均线交叉选项卡
        ma_frame = ttk.Frame(details_notebook)
        details_notebook.add(ma_frame, text="均线交叉")
        
        # 获取股票数据
        try:
            # 这里添加获取详细数据和图表的代码
            # 为简化，我们直接显示已有的图表文件（如果存在）
            
            # 动量图表文件
            momentum_chart = os.path.join(RESULTS_DIR, "charts", f"{stock_code}_momentum.png")
            if os.path.exists(momentum_chart):
                img1 = tk.PhotoImage(file=momentum_chart)
                canvas1 = tk.Canvas(momentum_frame, width=img1.width(), height=img1.height())
                canvas1.pack(fill=tk.BOTH, expand=True)
                canvas1.create_image(0, 0, anchor=tk.NW, image=img1)
                canvas1.image = img1
            else:
                ttk.Label(momentum_frame, text=f"未找到{stock_code}的动量分析图表").pack(pady=20)
            
            # 均线交叉图表文件
            ma_chart = os.path.join(RESULTS_DIR, "ma_charts", f"{stock_code}_ma_cross.png")
            if os.path.exists(ma_chart):
                img2 = tk.PhotoImage(file=ma_chart)
                canvas2 = tk.Canvas(ma_frame, width=img2.width(), height=img2.height())
                canvas2.pack(fill=tk.BOTH, expand=True)
                canvas2.create_image(0, 0, anchor=tk.NW, image=img2)
                canvas2.image = img2
            else:
                ttk.Label(ma_frame, text=f"未找到{stock_code}的均线交叉图表").pack(pady=20)
                
        except Exception as e:
            logger.error(f"显示组合分析详情失败: {str(e)}", exc_info=True)
            ttk.Label(details_window, text=f"加载详情失败: {str(e)}").pack(pady=20)

    def _update_momentum_results_ui(self, results):
        """更新动量分析结果UI"""
        if not isinstance(results, list) or not results:
            return
            
        # 清空当前结果
        for item in self.momentum_tree.get_children():
            self.momentum_tree.delete(item)
            
        # 添加新结果
        for result in results:
            try:
                # 确保所有数据都有默认值，防止出错
                values = (
                    result.get('ts_code', ''),
                    result.get('name', ''),
                    result.get('industry', ''),
                    f"{result.get('close', 0):.2f}",
                    f"{result.get('momentum_20d', 0):.2%}" if 'momentum_20d' in result else f"{result.get('momentum_20', 0):.2%}",
                    f"{result.get('rsi', 0):.1f}",
                    f"{result.get('macd_hist', 0):.3f}" if 'macd_hist' in result else f"{result.get('macd', 0):.3f}",
                    f"{result.get('volume_ratio', 1):.2f}",
                    f"{result.get('momentum_score', 0):.1f}" if 'momentum_score' in result else f"{result.get('score', 0):.1f}"
                )
                self.momentum_tree.insert("", tk.END, values=values)
            except Exception as e:
                logger.error(f"插入动量分析结果时出错: {str(e)}", exc_info=True)
                
    def _update_ma_results_ui(self, results):
        """更新均线交叉结果UI"""
        if not isinstance(results, list) or not results:
            return
            
        # 清空当前结果
        for item in self.ma_tree.get_children():
            self.ma_tree.delete(item)
            
        # 添加新结果
        for result in results:
            try:
                # 确保所有数据都有默认值，防止出错
                values = (
                    result.get('ts_code', ''),
                    result.get('name', ''),
                    result.get('industry', ''),
                    f"{result.get('close', 0):.2f}",
                    result.get('current_signal', '无信号'),
                    f"{result.get('short_ma', 0)}",
                    f"{result.get('long_ma', 0)}",
                    f"{result.get('total_return', 0):.2%}",
                    f"{result.get('annual_return', 0):.2%}",
                    f"{result.get('max_drawdown', 0):.2%}",
                    f"{result.get('win_rate', 0):.2%}"
                )
                self.ma_tree.insert("", tk.END, values=values)
            except Exception as e:
                logger.error(f"插入均线交叉结果时出错: {str(e)}", exc_info=True)
                
    def _display_chart(self, chart_path):
        """显示图表"""
        if not chart_path or not os.path.exists(chart_path):
            raise ValueError(f"图表路径无效: {chart_path}")
            
        try:
            # 创建一个新窗口显示图表
            chart_window = tk.Toplevel(self.root)
            chart_window.title("股票分析图表")
            chart_window.geometry("1000x600")
            
            # 加载图像
            image = Image.open(chart_path)
            # 调整图像大小以适应窗口
            image = image.resize((980, 580), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # 在Label中显示图像
            label = tk.Label(chart_window, image=photo)
            label.image = photo  # 保持引用以防止被垃圾回收
            label.pack(padx=10, pady=10)
            
        except Exception as e:
            logger.error(f"显示图表时出错: {str(e)}", exc_info=True)
            raise

# 自定义日志处理器
class LogHandler(logging.Handler):
    """将日志消息发送到Tkinter文本框的处理器"""
    
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        
    def emit(self, record):
        msg = self.format(record)
        
        def append():
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.insert(tk.END, msg + "\n")
            self.text_widget.see(tk.END)
            self.text_widget.config(state=tk.DISABLED)
            
        # 在UI线程中执行，以防止Tkinter线程问题
        self.text_widget.after(0, append)

def main():
    """程序入口点"""
    root = tk.Tk()
    app = StockAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 