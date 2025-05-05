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
# 导入增强版GUI控制器
from enhanced_gui_controller import EnhancedGuiController
# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))
# 导入项目配置
from src.enhanced.config.settings import LOG_DIR, DATA_DIR, RESULTS_DIR
# 导入财务分析模块
from financial_analysis import FinancialAnalyzer
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
        # 初始化控制器 - 使用增强版控制器
        self.controller = EnhancedGuiController()
        # 初始化数据
        self.status_message = tk.StringVar()
        self.status_message.set("就绪")
        
        # 热门板块分类
        self.hot_sectors = [
            "全部", 
            "人工智能", 
            "半导体芯片", 
            "新能源汽车", 
            "医疗器械", 
            "云计算", 
            "5G通信", 
            "生物医药",
            "医药生物", 
            "电子", 
            "计算机", 
            "有色金属", 
            "通信", 
            "传媒", 
            "电气设备", 
            "汽车", 
            "机械设备",
            "食品饮料", 
            "银行", 
            "房地产", 
            "钢铁", 
            "煤炭", 
            "石油石化"
        ]
        
        # 未来热门板块预测
        self.future_hot_sectors = [
            "量子计算",
            "生物技术",
            "绿色能源",
            "元宇宙",
            "高端制造"
        ]
        
        # 使用热门板块作为行业选择
        self.industries = self.hot_sectors
        
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
        # 创建财务分析选项卡
        self.financial_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.financial_tab, text="财务分析")
        self.setup_financial_tab()
        # 创建筹码分析选项卡
        self.chip_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.chip_tab, text="筹码分析")
        self.setup_chip_tab()
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
        ttk.Button(param_frame, text="运行增强分析", command=self.run_momentum_analysis).grid(
            row=3, column=0, columnspan=2, pady=10)
        # 右侧结果区域
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)
        # 结果表格
        self.momentum_result_frame = ttk.LabelFrame(right_frame, text="增强版动量分析结果")
        self.momentum_result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # 创建表格
        columns = ("股票代码", "股票名称", "行业", "当前价格", "20日动量", "RSI", "MACD", "成交量比", "行业因子", "得分")
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
        self.momentum_tree.column("MACD", width=60)
        self.momentum_tree.column("成交量比", width=80)
        self.momentum_tree.column("行业因子", width=80)
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
    def setup_financial_tab(self):
        """设置财务分析选项卡"""
        # 创建左右分割窗口
        paned = ttk.PanedWindow(self.financial_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧参数设置区域
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # 参数设置框架
        param_frame = ttk.LabelFrame(left_frame, text="分析参数")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 样本大小设置
        ttk.Label(param_frame, text="样本数量:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.fin_sample_size_var = tk.StringVar(value="100")
        ttk.Entry(param_frame, textvariable=self.fin_sample_size_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # 行业选择
        ttk.Label(param_frame, text="行业选择:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.fin_industry_var = tk.StringVar(value="全部")
        fin_industry_combo = ttk.Combobox(param_frame, textvariable=self.fin_industry_var, values=self.industries)
        fin_industry_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # 最低财务得分
        ttk.Label(param_frame, text="最低财务得分:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.fin_min_score_var = tk.StringVar(value="60")
        ttk.Entry(param_frame, textvariable=self.fin_min_score_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # 财务分析权重（用于综合分析）
        ttk.Label(param_frame, text="财务分析权重:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.fin_weight_var = tk.StringVar(value="0.5")
        ttk.Entry(param_frame, textvariable=self.fin_weight_var, width=10).grid(row=3, column=1, padx=5, pady=5)
        
        # 分析按钮
        ttk.Button(param_frame, text="单独财务分析", command=self.run_financial_analysis).grid(
            row=4, column=0, pady=10, padx=5)
        ttk.Button(param_frame, text="综合技术财务分析", command=self.run_combined_financial_analysis).grid(
            row=4, column=1, pady=10, padx=5)
        
        # 财务详情框架
        details_frame = ttk.LabelFrame(left_frame, text="财务详情")
        details_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建财务详情文本框
        self.financial_details_text = ScrolledText(details_frame, height=20)
        self.financial_details_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右侧结果区域
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)
        
        # 结果表格
        self.financial_result_frame = ttk.LabelFrame(right_frame, text="财务分析结果")
        self.financial_result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建表格
        columns = ("股票代码", "股票名称", "行业", "每股收益", "净资产收益率", "毛利率", "资产负债率", "最新财报", "财务得分")
        self.financial_tree = ttk.Treeview(self.financial_result_frame, columns=columns, show="headings", height=20)
        
        # 设置列标题
        for col in columns:
            self.financial_tree.heading(col, text=col)
        
        # 设置列宽
        self.financial_tree.column("股票代码", width=90)
        self.financial_tree.column("股票名称", width=90)
        self.financial_tree.column("行业", width=90)
        self.financial_tree.column("每股收益", width=80)
        self.financial_tree.column("净资产收益率", width=90)
        self.financial_tree.column("毛利率", width=70)
        self.financial_tree.column("资产负债率", width=80)
        self.financial_tree.column("最新财报", width=80)
        self.financial_tree.column("财务得分", width=70)
        
        # 添加滚动条
        financial_scrollbar = ttk.Scrollbar(self.financial_result_frame, orient=tk.VERTICAL, command=self.financial_tree.yview)
        self.financial_tree.configure(yscrollcommand=financial_scrollbar.set)
        
        # 布局
        self.financial_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        financial_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 表格双击事件
        self.financial_tree.bind("<Double-1>", self.on_financial_tree_double_click)
        
    def setup_chip_tab(self):
        """设置筹码分析选项卡"""
        # 创建左右分割窗口
        paned = ttk.PanedWindow(self.chip_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧参数设置区域
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # 参数设置框架
        param_frame = ttk.LabelFrame(left_frame, text="筹码分析参数")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 股票代码输入
        ttk.Label(param_frame, text="股票代码:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.chip_stock_code_var = tk.StringVar()
        self.chip_stock_entry = ttk.Entry(param_frame, textvariable=self.chip_stock_code_var, width=15)
        self.chip_stock_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # 分析按钮
        ttk.Button(param_frame, text="分析筹码", command=self.run_chip_analysis).grid(
            row=1, column=0, columnspan=2, pady=10)
        
        # 历史记录框架
        history_frame = ttk.LabelFrame(left_frame, text="历史记录")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 历史记录列表
        self.chip_history_list = tk.Listbox(history_frame, height=15, width=30)
        self.chip_history_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        chip_history_scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.chip_history_list.yview)
        self.chip_history_list.configure(yscrollcommand=chip_history_scrollbar.set)
        chip_history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 双击事件绑定
        self.chip_history_list.bind("<Double-1>", self.on_chip_history_double_click)
        
        # 右侧结果区域
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)
        
        # 结果表格
        self.chip_result_frame = ttk.LabelFrame(right_frame, text="筹码分析结果")
        self.chip_result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建图表显示区域
        self.chip_canvas_frame = ttk.Frame(self.chip_result_frame)
        self.chip_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加初始提示文本
        self.chip_info_label = ttk.Label(self.chip_canvas_frame, 
                                        text="请输入股票代码进行筹码分析\n支持沪深市场股票，请输入正确的6位代码", 
                                        font=("微软雅黑", 12))
        self.chip_info_label.pack(expand=True)
    
    def setup_overview_tab(self):
        """设置市场概览选项卡"""
        # 创建框架容器
        frame = ttk.Frame(self.overview_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 顶部控制区域
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 刷新按钮
        ttk.Button(control_frame, text="刷新市场数据", command=self.refresh_market_overview).pack(side=tk.LEFT, padx=5)
        
        # 市场情绪显示
        self.market_sentiment_frame = ttk.LabelFrame(control_frame, text="市场情绪")
        self.market_sentiment_frame.pack(side=tk.RIGHT, padx=5, fill=tk.X)
        
        self.sentiment_label = ttk.Label(self.market_sentiment_frame, text="--", font=("Arial", 10, "bold"))
        self.sentiment_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # 创建分隔线
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)
        
        # 创建市场概览内容区
        content_frame = ttk.Frame(frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧市场指数区域
        index_frame = ttk.LabelFrame(content_frame, text="市场指数")
        index_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建指数表格 - 添加5日涨跌幅和趋势列
        columns = ("指数名称", "最新价", "涨跌幅", "5日涨跌", "趋势", "成交量", "成交额")
        self.index_tree = ttk.Treeview(index_frame, columns=columns, show="headings", height=10)
        
        # 设置列标题
        for col in columns:
            self.index_tree.heading(col, text=col)
            self.index_tree.column(col, width=80)
        
        # 添加滚动条
        index_scrollbar = ttk.Scrollbar(index_frame, orient=tk.VERTICAL, command=self.index_tree.yview)
        self.index_tree.configure(yscrollcommand=index_scrollbar.set)
        
        # 布局
        self.index_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        index_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 右侧行业板块区域
        sector_frame = ttk.LabelFrame(content_frame, text="行业板块")
        sector_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建行业板块表格 - 添加强度指数和修改列布局
        columns = ("行业名称", "强度指数", "平均涨跌幅", "上涨家数", "下跌家数", "领涨股", "领涨股涨幅")
        self.sector_tree = ttk.Treeview(sector_frame, columns=columns, show="headings", height=10)
        
        # 设置列标题
        for col in columns:
            self.sector_tree.heading(col, text=col)
        
        # 设置列宽
        self.sector_tree.column("行业名称", width=90)
        self.sector_tree.column("强度指数", width=70)
        self.sector_tree.column("平均涨跌幅", width=80)
        self.sector_tree.column("上涨家数", width=70)
        self.sector_tree.column("下跌家数", width=70)
        self.sector_tree.column("领涨股", width=90)
        self.sector_tree.column("领涨股涨幅", width=80)
        
        # 添加滚动条
        sector_scrollbar = ttk.Scrollbar(sector_frame, orient=tk.VERTICAL, command=self.sector_tree.yview)
        self.sector_tree.configure(yscrollcommand=sector_scrollbar.set)
        
        # 布局
        self.sector_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sector_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 中部热门板块区域
        hot_frames_container = ttk.Frame(frame)
        hot_frames_container.pack(fill=tk.X, padx=5, pady=5)
        
        # 当前热门板块区域
        current_hot_frame = ttk.LabelFrame(hot_frames_container, text="当前热门板块")
        current_hot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建当前热门板块表格
        columns = ("板块名称", "涨跌幅", "成交额(亿)", "上涨家数", "下跌家数", "领涨股")
        self.current_hot_tree = ttk.Treeview(current_hot_frame, columns=columns, show="headings", height=5)
        
        # 设置列标题
        for col in columns:
            self.current_hot_tree.heading(col, text=col)
            self.current_hot_tree.column(col, width=80)
            
        # 添加滚动条
        current_hot_scrollbar = ttk.Scrollbar(current_hot_frame, orient=tk.VERTICAL, command=self.current_hot_tree.yview)
        self.current_hot_tree.configure(yscrollcommand=current_hot_scrollbar.set)
        
        # 布局
        self.current_hot_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        current_hot_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 未来热门板块预测区域 - 更改名称
        future_hot_frame = ttk.LabelFrame(hot_frames_container, text="未来热门板块AI预测")
        future_hot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建未来热门板块表格
        columns = ("板块名称", "预测涨幅", "关注指数", "主力资金", "成长性", "推荐理由")
        self.future_hot_tree = ttk.Treeview(future_hot_frame, columns=columns, show="headings", height=5)
        
        # 设置列标题
        for col in columns:
            self.future_hot_tree.heading(col, text=col)
            
        # 设置列宽
        self.future_hot_tree.column("板块名称", width=80)
        self.future_hot_tree.column("预测涨幅", width=70)
        self.future_hot_tree.column("关注指数", width=70)
        self.future_hot_tree.column("主力资金", width=70)
        self.future_hot_tree.column("成长性", width=70)
        self.future_hot_tree.column("推荐理由", width=200)
            
        # 添加滚动条
        future_hot_scrollbar = ttk.Scrollbar(future_hot_frame, orient=tk.VERTICAL, command=self.future_hot_tree.yview)
        self.future_hot_tree.configure(yscrollcommand=future_hot_scrollbar.set)
        
        # 布局
        self.future_hot_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        future_hot_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 底部市场统计区域
        stats_frame = ttk.LabelFrame(frame, text="市场统计")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建统计信息框
        stats_inner_frame = ttk.Frame(stats_frame)
        stats_inner_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # 统计指标
        stats = [
            ("上涨家数:", "上涨数量标签"),
            ("下跌家数:", "下跌数量标签"),
            ("平盘家数:", "平盘数量标签"),
            ("涨停家数:", "涨停数量标签"),
            ("跌停家数:", "跌停数量标签"),
            ("换手率:", "换手率标签"),
            ("总成交额:", "成交额标签")
        ]
        
        # 动态创建统计标签
        for i, (label_text, var_name) in enumerate(stats):
            col = i % 4
            row = i // 4
            ttk.Label(stats_inner_frame, text=label_text).grid(row=row, column=col*2, sticky=tk.W, padx=5, pady=5)
            label = ttk.Label(stats_inner_frame, text="--")
            label.grid(row=row, column=col*2+1, sticky=tk.W, padx=5, pady=5)
            setattr(self, var_name, label)
        
        # 更新时间和数据来源
        update_frame = ttk.Frame(stats_frame)
        update_frame.pack(side=tk.RIGHT, padx=5, pady=2)
        
        self.last_update_label = ttk.Label(update_frame, text="最后更新: --")
        self.last_update_label.pack(side=tk.RIGHT, padx=5)
        ttk.Label(update_frame, text="数据来源: 实时行情API").pack(side=tk.RIGHT, padx=5)
    
    def setup_log_tab(self):
        """设置日志选项卡"""
        # 创建框架
        frame = ttk.Frame(self.log_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # 日志文本框
        self.log_text = ScrolledText(frame, height=15, width=60)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # 日志滚动条
        log_scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # 日志添加按钮
        ttk.Button(frame, text="清空日志", command=self.clear_log).pack(pady=5)
        
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
            self.status_message.set("正在运行增强版动量分析...")
            # 使用增强版动量分析方法
            results = self.controller.get_enhanced_momentum_analysis(
                industry=None if industry == "全部" else industry,
                sample_size=sample_size,
                min_score=min_score
            )
            
            # 更新UI显示结果
            if results:
                self._update_momentum_results_ui(results)
                self.status_message.set(f"增强版动量分析完成，找到 {len(results)} 支符合条件的股票")
            else:
                self.status_message.set("增强版动量分析完成，但未找到符合条件的股票")
        except Exception as e:
            logger.error(f"增强版动量分析失败: {str(e)}", exc_info=True)
            self.status_message.set("增强版动量分析失败")
            messagebox.showerror("分析失败", f"增强版动量分析时出错: {str(e)}")
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
            # 不再从控制器更新行业列表，使用我们定义的热门板块分类
            # self.industries = self.controller.get_stock_industries()
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
            for item in self.sector_tree.get_children():
                self.sector_tree.delete(item)
            for item in self.index_tree.get_children():
                self.index_tree.delete(item)
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
        # 尝试打开增强版图表
        enhanced_chart_path = os.path.join(RESULTS_DIR, "enhanced_charts", f"{stock_code}_enhanced.png")
        # 如果增强版图表存在，则显示它
        if os.path.exists(enhanced_chart_path):
            self.show_chart(enhanced_chart_path)
        else:
            # 否则尝试打开标准图表
            standard_chart_path = os.path.join(RESULTS_DIR, "charts", f"{stock_code}_momentum.png")
            if os.path.exists(standard_chart_path):
                self.show_chart(standard_chart_path)
            else:
                messagebox.showinfo("提示", f"未找到股票{stock_code}的图表")
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
                elif isinstance(data, dict) and "indices" in data and "industry_performance" in data:
                    # 处理市场概览数据
                    print("更新市场概览UI")
                    self._update_market_overview_ui(data)
                    self.status_message.set("市场概览数据更新完成")
                    # 确保选中市场概览选项卡
                    for i in range(self.notebook.index("end")):
                        if self.notebook.tab(i, "text") == "市场概览":
                            self.notebook.select(i)
                            break
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
                    result.get('ts_code', '') if 'ts_code' in result else result.get('code', ''),
                    result.get('name', ''),
                    result.get('industry', ''),
                    f"{result.get('close', 0):.2f}",
                    f"{result.get('momentum_20d', 0):.2%}" if 'momentum_20d' in result else f"{result.get('momentum_20', 0):.2%}",
                    f"{result.get('rsi', 0):.1f}",
                    f"{result.get('macd_hist', 0):.3f}" if 'macd_hist' in result else f"{result.get('macd', 0):.3f}",
                    f"{result.get('volume_ratio', 1):.2f}",
                    f"{result.get('industry_factor', 1.0):.2f}", # 增强版 - 行业因子
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
    def run_financial_analysis(self):
        """运行财务分析"""
        try:
            # 获取参数
            sample_size = int(self.fin_sample_size_var.get())
            industry = self.fin_industry_var.get()
            min_score = int(self.fin_min_score_var.get())
            
            # 更新状态
            self.status_message.set(f"正在进行财务分析，样本大小: {sample_size}，行业: {industry}，最低得分: {min_score}")
            self.root.update()
            
            # 获取股票列表
            if self.controller.stock_list is None or (industry is not None and industry != "全部"):
                self.controller.stock_list = self.controller.get_stock_list(industry)
            
            if self.controller.stock_list.empty:
                messagebox.showerror("错误", "获取股票列表失败，请检查网络连接或数据源")
                return
            
            # 定义回调函数
            def financial_callback(status_type, data):
                if status_type == "状态":
                    self.status_message.set(data)
                    self.root.update()
                elif status_type == "结果":
                    self._update_financial_results_ui(data)
            
            # 运行财务分析
            def run_analysis_thread():
                try:
                    # 创建财务分析器
                    financial_analyzer = FinancialAnalyzer(use_tushare=True)
                    
                    # 分析股票
                    results = financial_analyzer.analyze_financial_stocks(
                        self.controller.stock_list, sample_size=sample_size, min_score=min_score)
                    
                    # 更新UI
                    if results:
                        financial_callback("结果", results)
                    else:
                        financial_callback("状态", "财务分析完成，未找到符合条件的股票")
                except Exception as e:
                    logger.error(f"财务分析过程中出错: {str(e)}", exc_info=True)
                    financial_callback("状态", f"财务分析过程中出错: {str(e)}")
            
            # 在新线程中运行分析
            analysis_thread = threading.Thread(target=run_analysis_thread)
            analysis_thread.daemon = True
            analysis_thread.start()
            
        except Exception as e:
            logger.error(f"启动财务分析失败: {str(e)}", exc_info=True)
            self.status_message.set(f"启动财务分析失败: {str(e)}")
    
    def run_combined_financial_analysis(self):
        """运行结合技术和财务的综合分析"""
        try:
            # 获取参数
            sample_size = int(self.fin_sample_size_var.get())
            industry = self.fin_industry_var.get()
            min_score = int(self.fin_min_score_var.get())
            fin_weight = float(self.fin_weight_var.get())
            
            # 更新状态
            self.status_message.set(f"正在进行综合分析，样本大小: {sample_size}，行业: {industry}")
            self.root.update()
            
            # 获取股票列表
            if self.controller.stock_list is None or (industry is not None and industry != "全部"):
                self.controller.stock_list = self.controller.get_stock_list(industry)
            
            if self.controller.stock_list.empty:
                messagebox.showerror("错误", "获取股票列表失败，请检查网络连接或数据源")
                return
            
            # 定义回调函数
            def combined_callback(status_type, data):
                if status_type == "状态":
                    self.status_message.set(data)
                    self.root.update()
                elif status_type == "结果":
                    self._update_combined_financial_results_ui(data)
            
            # 运行综合分析
            def run_analysis_thread():
                try:
                    # 创建分析器
                    financial_analyzer = FinancialAnalyzer(use_tushare=True)
                    
                    # 分析股票财务
                    fin_results = financial_analyzer.analyze_financial_stocks(
                        self.controller.stock_list, sample_size=sample_size, min_score=0)  # 不设最低分，后面综合时筛选
                    
                    # 运行动量分析
                    self.controller.run_momentum_analysis(
                        sample_size=sample_size, 
                        industry=industry, 
                        min_score=0,  # 不设最低分，后面综合时筛选
                        gui_callback=lambda status_type, data: None  # 静默回调
                    )
                    
                    # 等待动量分析完成
                    wait_count = 0
                    while self.controller.momentum_results is None and wait_count < 30:
                        time.sleep(1)
                        wait_count += 1
                    
                    # 检查动量分析是否成功
                    if self.controller.momentum_results is None:
                        combined_callback("状态", "动量分析超时或失败，无法完成综合分析")
                        return
                    
                    # 综合两个结果
                    combined_results = financial_analyzer.combine_financial_technical(
                        fin_results, self.controller.momentum_results, fin_weight)
                    
                    # 筛选符合最低综合得分的结果
                    filtered_results = [r for r in combined_results if r['combined_score'] >= min_score]
                    
                    # 更新UI
                    if filtered_results:
                        combined_callback("结果", filtered_results)
                    else:
                        combined_callback("状态", "综合分析完成，未找到符合条件的股票")
                except Exception as e:
                    logger.error(f"综合分析过程中出错: {str(e)}", exc_info=True)
                    combined_callback("状态", f"综合分析过程中出错: {str(e)}")
            
            # 在新线程中运行分析
            analysis_thread = threading.Thread(target=run_analysis_thread)
            analysis_thread.daemon = True
            analysis_thread.start()
            
        except Exception as e:
            logger.error(f"启动综合分析失败: {str(e)}", exc_info=True)
            self.status_message.set(f"启动综合分析失败: {str(e)}")
    
    def run_chip_analysis(self):
        """运行筹码分析"""
        try:
            # 获取股票代码
            ts_code = self.chip_stock_code_var.get().strip()
            if not ts_code:
                messagebox.showerror("错误", "请输入股票代码")
                return
            
            # 标准化股票代码格式（如果需要）
            if '.' not in ts_code:
                # 尝试推断交易所
                if ts_code.startswith('6'):
                    ts_code = ts_code + '.SH'
                else:
                    ts_code = ts_code + '.SZ'
            
            # 获取股票名称
            stock_name = ""
            if self.controller.stock_list is not None and not self.controller.stock_list.empty:
                stock_info = self.controller.stock_list[self.controller.stock_list['ts_code'] == ts_code]
                if not stock_info.empty:
                    stock_name = stock_info.iloc[0].get('name', '')
            
            # 更新状态
            self.status_message.set(f"正在分析 {ts_code} {stock_name} 的筹码分布...")
            self.root.update()
            
            # 运行筹码分析
            def run_analysis_thread():
                try:
                    # 创建财务分析器
                    financial_analyzer = FinancialAnalyzer(use_tushare=True)
                    
                    # 分析筹码分布
                    result = financial_analyzer.analyze_chip_distribution(ts_code, stock_name)
                    
                    # 更新UI
                    if result:
                        # 在主线程中更新UI
                        self.root.after(0, lambda: self._update_chip_results_ui(ts_code, stock_name, result))
                        
                        # 添加到历史记录
                        self.root.after(0, lambda: self._add_to_chip_history(ts_code, stock_name))
                    else:
                        self.root.after(0, lambda: self.status_message.set(f"获取 {ts_code} 的筹码数据失败"))
                except Exception as e:
                    logger.error(f"筹码分析过程中出错: {str(e)}", exc_info=True)
                    self.root.after(0, lambda: self.status_message.set(f"筹码分析过程中出错: {str(e)}"))
            
            # 在新线程中运行分析
            analysis_thread = threading.Thread(target=run_analysis_thread)
            analysis_thread.daemon = True
            analysis_thread.start()
            
        except Exception as e:
            logger.error(f"启动筹码分析失败: {str(e)}", exc_info=True)
            self.status_message.set(f"启动筹码分析失败: {str(e)}")
    
    def on_financial_tree_double_click(self, event):
        """财务分析表格双击事件处理"""
        item = self.financial_tree.selection()[0]
        item_data = self.financial_tree.item(item, "values")
        ts_code = item_data[0]
        stock_name = item_data[1]
        
        # 显示财务详情
        self.show_financial_details(ts_code, stock_name)
    
    def on_chip_history_double_click(self, event):
        """筹码分析历史列表双击事件处理"""
        if self.chip_history_list.curselection():
            index = self.chip_history_list.curselection()[0]
            value = self.chip_history_list.get(index)
            if value:
                # 提取股票代码
                ts_code = value.split()[0]
                self.chip_stock_code_var.set(ts_code)
                self.run_chip_analysis()
    
    def show_financial_details(self, ts_code, stock_name):
        """显示股票财务详情"""
        try:
            # 清空详情文本框
            self.financial_details_text.delete(1.0, tk.END)
            
            # 显示加载信息
            self.financial_details_text.insert(tk.END, f"正在获取 {ts_code} {stock_name} 的财务详情...\n")
            self.root.update()
            
            # 获取财务数据
            def get_details_thread():
                try:
                    # 创建财务分析器
                    financial_analyzer = FinancialAnalyzer(use_tushare=True)
                    
                    # 获取财务指标数据
                    financial_data = financial_analyzer.get_financial_indicator(ts_code)
                    
                    # 获取股东人数数据
                    holder_data = financial_analyzer.get_stk_holders(ts_code)
                    
                    # 获取机构调研数据
                    survey_data = financial_analyzer.get_institutional_survey(ts_code)
                    
                    # 在主线程中更新UI
                    self.root.after(0, lambda: self._update_financial_details_ui(
                        ts_code, stock_name, financial_data, holder_data, survey_data))
                except Exception as e:
                    logger.error(f"获取财务详情过程中出错: {str(e)}", exc_info=True)
                    self.root.after(0, lambda: self.financial_details_text.insert(
                        tk.END, f"\n获取财务详情失败: {str(e)}"))
            
            # 在新线程中获取数据
            details_thread = threading.Thread(target=get_details_thread)
            details_thread.daemon = True
            details_thread.start()
            
        except Exception as e:
            logger.error(f"显示财务详情失败: {str(e)}", exc_info=True)
            self.financial_details_text.insert(tk.END, f"\n显示财务详情失败: {str(e)}")
    
    def _update_financial_results_ui(self, results):
        """更新财务分析结果UI"""
        # 清空表格
        for i in self.financial_tree.get_children():
            self.financial_tree.delete(i)
        
        # 添加新数据
        for result in results:
            self.financial_tree.insert("", tk.END, values=(
                result['ts_code'],
                result['name'],
                result['industry'],
                f"{result['eps']:.2f}",
                f"{result['roe']:.2f}%",
                f"{result['grossprofit_margin']:.2f}%",
                f"{result['debt_to_assets']:.2f}%",
                result['latest_report'],
                f"{result['score']:.0f}"
            ))
        
        # 更新状态
        self.status_message.set(f"财务分析完成，找到 {len(results)} 支符合条件的股票")
    
    def _update_combined_financial_results_ui(self, results):
        """更新综合财务分析结果UI"""
        # 清空表格
        for i in self.financial_tree.get_children():
            self.financial_tree.delete(i)
        
        # 添加新数据
        for result in results:
            self.financial_tree.insert("", tk.END, values=(
                result['ts_code'],
                result['name'],
                result['industry'],
                f"{result['eps']:.2f}",
                f"{result['roe']:.2f}%",
                f"技术:{result['technical_score']:.0f}",
                f"财务:{result['financial_score']:.0f}",
                "",  # 最新财报列留空
                f"{result['combined_score']:.0f}"
            ))
        
        # 更新状态
        self.status_message.set(f"综合分析完成，找到 {len(results)} 支符合条件的股票")
    
    def _update_financial_details_ui(self, ts_code, stock_name, financial_data, holder_data, survey_data):
        """更新财务详情UI"""
        # 清空详情文本框
        self.financial_details_text.delete(1.0, tk.END)
        
        # 添加标题
        self.financial_details_text.insert(tk.END, f"{ts_code} {stock_name} 财务详情\n", "title")
        self.financial_details_text.insert(tk.END, "="*50 + "\n\n")
        
        # 添加财务指标数据
        if not financial_data.empty:
            # 按报告期降序排序
            financial_data = financial_data.sort_values(by='end_date', ascending=False)
            
            # 显示最近的财务指标
            latest = financial_data.iloc[0]
            self.financial_details_text.insert(tk.END, "最新财务指标（报告期：" + str(latest.get('end_date', '')) + "）\n", "subtitle")
            self.financial_details_text.insert(tk.END, "-"*50 + "\n")
            
            # 显示关键指标
            key_indicators = [
                ('每股收益(EPS)', 'eps', '元'),
                ('每股净资产', 'bps', '元'),
                ('净资产收益率', 'roe', '%'),
                ('毛利率', 'grossprofit_margin', '%'),
                ('资产负债率', 'debt_to_assets', '%'),
                ('每股营收', 'revenue_ps', '元'),
                ('销售净利率', 'netprofit_margin', '%'),
                ('总资产周转率', 'assets_turn', '次')
            ]
            
            for name, key, unit in key_indicators:
                value = latest.get(key, None)
                if value is not None:
                    self.financial_details_text.insert(tk.END, f"{name}: {value:.2f}{unit}\n")
            
            # 显示历史趋势
            self.financial_details_text.insert(tk.END, "\n历史财务指标趋势\n", "subtitle")
            self.financial_details_text.insert(tk.END, "-"*50 + "\n")
            
            # 创建简单的趋势表
            self.financial_details_text.insert(tk.END, "报告期\t\tEPS\t\tROE\t\t毛利率\n")
            
            for _, row in financial_data.head(8).iterrows():
                end_date = row.get('end_date', '')
                eps = row.get('eps', 0)
                roe = row.get('roe', 0)
                gp_margin = row.get('grossprofit_margin', 0)
                
                self.financial_details_text.insert(tk.END, 
                    f"{end_date}\t\t{eps:.2f}\t\t{roe:.2f}%\t\t{gp_margin:.2f}%\n")
        else:
            self.financial_details_text.insert(tk.END, "未找到财务指标数据\n")
        
        # 添加股东人数数据
        self.financial_details_text.insert(tk.END, "\n股东人数数据\n", "subtitle")
        self.financial_details_text.insert(tk.END, "-"*50 + "\n")
        
        if not holder_data.empty:
            # 按日期降序排序
            holder_data = holder_data.sort_values(by='end_date', ascending=False)
            
            # 显示股东人数变化趋势
            self.financial_details_text.insert(tk.END, "日期\t\t股东人数\n")
            
            for _, row in holder_data.head(5).iterrows():
                end_date = row.get('end_date', '')
                holder_num = row.get('holder_num', 0)
                
                self.financial_details_text.insert(tk.END, f"{end_date}\t\t{holder_num}\n")
        else:
            self.financial_details_text.insert(tk.END, "未找到股东人数数据\n")
        
        # 添加机构调研数据
        self.financial_details_text.insert(tk.END, "\n近期机构调研情况\n", "subtitle")
        self.financial_details_text.insert(tk.END, "-"*50 + "\n")
        
        if not survey_data.empty:
            # 按日期降序排序
            survey_data = survey_data.sort_values(by='surv_date', ascending=False)
            
            # 计算调研机构数量
            surv_dates = survey_data['surv_date'].unique()
            
            for date in surv_dates[:3]:  # 显示最近3次调研
                date_surveys = survey_data[survey_data['surv_date'] == date]
                orgs = date_surveys['rece_org'].unique()
                
                self.financial_details_text.insert(tk.END, 
                    f"调研日期: {date}, 参与机构数: {len(orgs)}\n")
                self.financial_details_text.insert(tk.END, f"机构类型: {', '.join(date_surveys['org_type'].unique())}\n")
                self.financial_details_text.insert(tk.END, "-"*30 + "\n")
        else:
            self.financial_details_text.insert(tk.END, "未找到机构调研数据\n")
        
        # 设置标题样式
        self.financial_details_text.tag_configure("title", font=("Helvetica", 12, "bold"))
        self.financial_details_text.tag_configure("subtitle", font=("Helvetica", 10, "bold"))
    
    def _update_chip_results_ui(self, ts_code, stock_name, result):
        """更新筹码分析结果UI"""
        # 清空画布框架
        for widget in self.chip_canvas_frame.winfo_children():
            widget.destroy()
        
        # 获取图表路径
        chart_path = result.get('chart_path')
        
        if chart_path and os.path.exists(chart_path):
            try:
                # 创建图像标签
                img = Image.open(chart_path)
                img = img.resize((900, 700), Image.LANCZOS)  # 调整大小适应界面
                photo = ImageTk.PhotoImage(img)
                
                # 保存引用，防止垃圾回收
                self.chip_image = photo
                
                # 显示图像
                label = ttk.Label(self.chip_canvas_frame, image=photo)
                label.pack(fill=tk.BOTH, expand=True)
                
                # 显示关键数据
                info_frame = ttk.Frame(self.chip_canvas_frame)
                info_frame.pack(fill=tk.X, padx=10, pady=5)
                
                ttk.Label(info_frame, text=f"平均成本: {result['avg_cost']:.2f}", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=10)
                ttk.Label(info_frame, text=f"胜率: {result['winner_rate']:.2f}%", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=10)
                ttk.Label(info_frame, text=f"低档成本: {result['low_cost']:.2f}", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=10)
                ttk.Label(info_frame, text=f"高档成本: {result['high_cost']:.2f}", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=10)
                
                # 更新状态
                self.status_message.set(f"{ts_code} {stock_name} 筹码分布分析完成")
            except Exception as e:
                logger.error(f"显示筹码分析图表失败: {str(e)}", exc_info=True)
                ttk.Label(self.chip_canvas_frame, text=f"显示图表失败: {str(e)}").pack(pady=100)
        else:
            ttk.Label(self.chip_canvas_frame, text="获取筹码分布数据失败或图表生成失败").pack(pady=100)
    
    def _add_to_chip_history(self, ts_code, stock_name):
        """添加到筹码分析历史列表"""
        history_text = f"{ts_code} {stock_name}"
        
        # 检查是否已存在
        found = False
        for i in range(self.chip_history_list.size()):
            if self.chip_history_list.get(i).startswith(ts_code):
                found = True
                break
        
        # 如果不存在，添加到列表顶部
        if not found:
            self.chip_history_list.insert(0, history_text)
            
            # 限制历史记录数量
            while self.chip_history_list.size() > 20:
                self.chip_history_list.delete(tk.END)
    
    def _update_market_overview_ui(self, overview_data):
        """更新市场概览UI"""
        if not isinstance(overview_data, dict):
            logger.warning("市场概览数据格式不正确")
            return
        
        try:
            # 更新市场情绪
            if "market_stats" in overview_data and "market_sentiment" in overview_data["market_stats"]:
                sentiment = overview_data["market_stats"]["market_sentiment"]
                self.sentiment_label.config(text=sentiment)
                
                # 根据情绪设置不同的颜色
                if "极度乐观" in sentiment:
                    self.sentiment_label.config(foreground="red", font=("Arial", 10, "bold"))
                elif "乐观" in sentiment:
                    self.sentiment_label.config(foreground="#FF4500", font=("Arial", 10, "bold"))  # 橙红色
                elif "偏乐观" in sentiment:
                    self.sentiment_label.config(foreground="#FF8C00", font=("Arial", 10, "bold"))  # 深橙色
                elif "极度悲观" in sentiment:
                    self.sentiment_label.config(foreground="green", font=("Arial", 10, "bold"))
                elif "悲观" in sentiment:
                    self.sentiment_label.config(foreground="#006400", font=("Arial", 10, "bold"))  # 深绿色
                elif "偏悲观" in sentiment:
                    self.sentiment_label.config(foreground="#228B22", font=("Arial", 10, "bold"))  # 森林绿
                else:
                    self.sentiment_label.config(foreground="black", font=("Arial", 10, "bold"))
            
            # 更新指数表格
            if "indices" in overview_data:
                # 清空表格
                for item in self.index_tree.get_children():
                    self.index_tree.delete(item)
                    
                # 添加指数数据
                for index in overview_data["indices"]:
                    change_str = f"{index.get('change', 0):.2f}%"
                    if index.get('change', 0) > 0:
                        change_str = f"+{change_str}"
                    
                    change_5d_str = f"{index.get('change_5d', 0):.2f}%"
                    if index.get('change_5d', 0) > 0:
                        change_5d_str = f"+{change_5d_str}"
                    
                    values = (
                        index.get('name', ''),
                        f"{index.get('close', 0):.2f}",
                        change_str,
                        change_5d_str,
                        index.get('trend', '中性'),
                        f"{index.get('volume', 0)/10000:.0f}万",
                        f"{index.get('amount', 0)/100000000:.2f}亿"
                    )
                    
                    item_id = self.index_tree.insert("", tk.END, values=values)
                    
                    # 为涨跌幅设置颜色
                    if index.get('change', 0) > 0:
                        self.index_tree.item(item_id, tags=('up',))
                    elif index.get('change', 0) < 0:
                        self.index_tree.item(item_id, tags=('down',))
                    
                # 设置颜色
                self.index_tree.tag_configure('up', foreground='red')
                self.index_tree.tag_configure('down', foreground='green')
            
            # 更新行业板块表格
            if "industry_performance" in overview_data:
                # 清空表格
                for item in self.sector_tree.get_children():
                    self.sector_tree.delete(item)
                    
                # 添加行业数据
                for industry in overview_data["industry_performance"]:
                    change_str = f"{industry.get('change', 0):.2f}%"
                    if industry.get('change', 0) > 0:
                        change_str = f"+{change_str}"
                    
                    leading_up_change = industry.get('leading_up_change', 0)
                    leading_up_change_str = f"{leading_up_change:.2f}%"
                    if leading_up_change > 0:
                        leading_up_change_str = f"+{leading_up_change_str}"
                    
                    values = (
                        industry.get('name', ''),
                        f"{industry.get('strength_index', 0):.0f}",
                        change_str,
                        industry.get('up_count', 0),
                        industry.get('down_count', 0),
                        industry.get('leading_up', ''),
                        leading_up_change_str
                    )
                    
                    item_id = self.sector_tree.insert("", tk.END, values=values)
                    
                    # 为涨跌幅设置颜色
                    if industry.get('change', 0) > 0:
                        self.sector_tree.item(item_id, tags=('up',))
                    elif industry.get('change', 0) < 0:
                        self.sector_tree.item(item_id, tags=('down',))
                
                # 设置颜色
                self.sector_tree.tag_configure('up', foreground='red')
                self.sector_tree.tag_configure('down', foreground='green')
            
            # 更新当前热门板块表格
            if "hot_sectors" in overview_data:
                # 清空表格
                for item in self.current_hot_tree.get_children():
                    self.current_hot_tree.delete(item)
                    
                # 添加热门板块数据
                for sector in overview_data.get("hot_sectors", []):
                    change_str = f"{sector.get('change', 0):.2f}%"
                    if sector.get('change', 0) > 0:
                        change_str = f"+{change_str}"
                    
                    values = (
                        sector.get('name', ''),
                        change_str,
                        f"{sector.get('turnover', 0):.2f}",
                        str(sector.get('up_count', 0)),
                        str(sector.get('down_count', 0)),
                        sector.get('leading_stock', '')
                    )
                    
                    item_id = self.current_hot_tree.insert("", tk.END, values=values)
                    
                    # 为涨跌幅设置颜色
                    if sector.get('change', 0) > 0:
                        self.current_hot_tree.item(item_id, tags=('up',))
                    elif sector.get('change', 0) < 0:
                        self.current_hot_tree.item(item_id, tags=('down',))
                
                # 设置颜色
                self.current_hot_tree.tag_configure('up', foreground='red')
                self.current_hot_tree.tag_configure('down', foreground='green')
            
            # 更新未来热门板块预测表格
            if "future_hot_sectors" in overview_data:
                # 清空表格
                for item in self.future_hot_tree.get_children():
                    self.future_hot_tree.delete(item)
                    
                # 添加未来热门板块预测数据
                for sector in overview_data.get("future_hot_sectors", []):
                    pred_change_str = f"{sector.get('predicted_change', 0):.2f}%"
                    if sector.get('predicted_change', 0) > 0:
                        pred_change_str = f"+{pred_change_str}"
                    
                    values = (
                        sector.get('name', ''),
                        pred_change_str,
                        f"{sector.get('attention_index', 0):.0f}",
                        f"{sector.get('fund_inflow', 0):.2f}亿",
                        f"{sector.get('growth_score', 0):.0f}",
                        sector.get('recommendation', '')
                    )
                    self.future_hot_tree.insert("", tk.END, values=values)
            
            # 更新市场统计数据
            if "market_stats" in overview_data:
                stats = overview_data["market_stats"]
                # 上涨家数
                if hasattr(self, "上涨数量标签"):
                    self.上涨数量标签.config(text=f"{stats.get('up_count', 0)}", foreground="red")
                
                # 下跌家数
                if hasattr(self, "下跌数量标签"):
                    self.下跌数量标签.config(text=f"{stats.get('down_count', 0)}", foreground="green")
                    
                # 平盘家数
                if hasattr(self, "平盘数量标签"):
                    self.平盘数量标签.config(text=f"{stats.get('flat_count', 0)}")
                    
                # 涨停家数
                if hasattr(self, "涨停数量标签"):
                    self.涨停数量标签.config(text=f"{stats.get('limit_up_count', 0)}", foreground="red")
                    
                # 跌停家数
                if hasattr(self, "跌停数量标签"):
                    self.跌停数量标签.config(text=f"{stats.get('limit_down_count', 0)}", foreground="green")
                    
                # 换手率
                if hasattr(self, "换手率标签") and "turnover_rate" in stats:
                    self.换手率标签.config(text=f"{stats.get('turnover_rate', 0):.2f}%")
                    
                # 总成交额
                if hasattr(self, "成交额标签") and "total_turnover" in stats:
                    self.成交额标签.config(text=f"{stats.get('total_turnover', 0)/100000000:.2f}亿")
            
            # 更新最后更新时间
            if hasattr(self, "last_update_label") and "date" in overview_data:
                self.last_update_label.config(text=f"最后更新: {overview_data.get('date', '')}")
        
        except Exception as e:
            logger.error(f"更新市场概览UI失败: {str(e)}", exc_info=True)

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
    try:
        # 设置Matplotlib后端为非交互式，可能会减少崩溃
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        
        root = tk.Tk()
        app = StockAnalysisGUI(root)
        root.mainloop()
    except Exception as e:
        # 记录全局异常
        logger.error(f"程序发生严重错误: {str(e)}", exc_info=True)
        # 尝试显示错误对话框
        try:
            import tkinter.messagebox as mb
            mb.showerror("严重错误", f"程序发生错误，请联系开发人员:\n{str(e)}")
        except:
            print(f"严重错误: {str(e)}")

if __name__ == "__main__":
    main()