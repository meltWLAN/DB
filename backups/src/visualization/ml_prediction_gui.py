#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
机器学习预测模型GUI界面
用于选择股票、设置参数并可视化预测结果
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
import logging
from pathlib import Path
from threading import Thread
from datetime import datetime, timedelta
import time
from src.utils.logger import get_logger

# Set logger for this module explicitly
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

# 确保模块路径正确
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# 添加项目根目录到搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# 导入自定义模块
from src.data.industry_data import get_simplified_industry_list, get_stocks_by_industry_name

# 导入系统模块
try:
    from src.strategies.ml_prediction_model import MLPredictionModel
    from src.indicators.advanced_indicators import AdvancedIndicators
    from src.data.data_fetcher import DataFetcher
except ImportError as e:
    print(f"导入模块时出错: {e}")
    logger = logging.getLogger("ml_prediction_gui")
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class MLPredictionGUI:
    """机器学习预测模型图形界面"""
    
    def __init__(self, root):
        """初始化ML预测GUI"""
        self.root = root
        self.root.title("机器学习预测系统")
        self.root.geometry("1400x800")
        
        # 初始化变量
        self.default_stocks = []  # 清空默认股票，不再预设股票
        
        # 初始化UI需要的变量
        self.industry_var = tk.StringVar()
        self.search_var = tk.StringVar()  # 新增搜索变量
        self.horizon_var = tk.StringVar(value="5")  # 默认预测周期5天
        self.threshold_var = tk.StringVar(value="0.6")  # 默认置信度阈值0.6
        self.lookback_var = tk.StringVar(value="365")  # 默认回溯365天
        self.status_var = tk.StringVar()
        
        # 初始化模型
        try:
            self.model = MLPredictionModel()
            self.indicators = AdvancedIndicators()
            self.data_fetcher = DataFetcher()
        except Exception as e:
            logger.error(f"初始化模型出错: {e}")
            messagebox.showerror("错误", f"初始化模型失败: {e}")
        
        # 创建主框架，包含三个部分
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建三个主要部分: 左(参数设置)、中(预测结果)、右(可视化图表)
        left_frame = tk.Frame(main_frame, width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5)
        
        middle_frame = tk.Frame(main_frame, width=350)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        right_frame = tk.Frame(main_frame, width=700)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # ===== 左侧面板: 参数设置 =====
        param_frame = ttk.LabelFrame(left_frame, text="参数设置")
        param_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        # 预测周期
        horizon_frame = tk.Frame(param_frame)
        horizon_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(horizon_frame, text="预测天数:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        horizon_values = ["5", "10", "20", "30"]
        horizon_combobox = ttk.Combobox(horizon_frame, textvariable=self.horizon_var, values=horizon_values, width=5)
        horizon_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        horizon_combobox.current(0)
        
        # 置信度阈值
        threshold_frame = tk.Frame(param_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(threshold_frame, text="信心值阈:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        threshold_values = ["0.5", "0.6", "0.7", "0.8", "0.9"]
        threshold_combobox = ttk.Combobox(threshold_frame, textvariable=self.threshold_var, values=threshold_values, width=5)
        threshold_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        threshold_combobox.current(1)  # 默认选择0.6
        
        # 数据范围
        lookback_frame = tk.Frame(param_frame)
        lookback_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(lookback_frame, text="数据范围(天):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        lookback_values = ["120", "240", "365", "500"]
        lookback_combobox = ttk.Combobox(lookback_frame, textvariable=self.lookback_var, values=lookback_values, width=5)
        lookback_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        lookback_combobox.current(2)  # 默认选择365天
        
        # 按钮框架
        button_frame = tk.Frame(left_frame)
        button_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        # 预测按钮
        predict_button = tk.Button(
            button_frame, text="预测", command=self._predict_selected, 
            bg="#4CAF50", fg="white", height=2
        )
        predict_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        # 多模型比较按钮
        compare_button = tk.Button(
            button_frame, text="多模型比较", command=self._compare_models, 
            bg="#3498DB", fg="white", height=2
        )
        compare_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        # 清空按钮
        clear_button = tk.Button(
            button_frame, text="清空列表", command=self._clear_stock_list,
            bg="#FF5733", fg="white", height=2
        )
        clear_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        # 已选择股票列表
        stock_list_frame = ttk.LabelFrame(left_frame, text="已选择股票")
        stock_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建一个滚动条
        scrollbar = tk.Scrollbar(stock_list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建一个Listbox并与滚动条关联
        self.stock_listbox = tk.Listbox(stock_list_frame, yscrollcommand=scrollbar.set, height=15)
        self.stock_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.stock_listbox.yview)
        
        # 行业分析区域
        industry_frame = ttk.LabelFrame(left_frame, text="行业分析")
        industry_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 行业选择
        industry_select_frame = tk.Frame(industry_frame)
        industry_select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(industry_select_frame, text="行业:").pack(side=tk.LEFT, padx=5, pady=5)
        self.industry_combobox = ttk.Combobox(industry_select_frame, textvariable=self.industry_var, state="readonly", width=20)
        self.industry_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        load_industry_button = tk.Button(industry_select_frame, text="加载成分股", command=self._load_industry_stocks)
        load_industry_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 成分股搜索框架
        search_frame = tk.Frame(industry_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(search_frame, text="搜索:").pack(side=tk.LEFT, padx=5, pady=5)
        search_entry = tk.Entry(search_frame, textvariable=self.search_var, width=15)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        search_entry.bind('<KeyRelease>', self._filter_stocks)
        
        # 行业成分股列表
        industry_stocks_frame = tk.Frame(industry_frame)
        industry_stocks_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建一个滚动条
        industry_scrollbar = tk.Scrollbar(industry_stocks_frame)
        industry_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 行业成分股列表框
        self.industry_stocks_listbox = tk.Listbox(industry_stocks_frame, yscrollcommand=industry_scrollbar.set, height=15, selectmode=tk.MULTIPLE)
        self.industry_stocks_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        industry_scrollbar.config(command=self.industry_stocks_listbox.yview)
        
        # 行业股票操作按钮
        industry_button_frame = tk.Frame(industry_frame)
        industry_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        add_to_list_button = tk.Button(industry_button_frame, text="添加到列表", command=self._add_selected_industry_stocks)
        add_to_list_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        predict_industry_button = tk.Button(industry_button_frame, text="直接预测", command=self._predict_industry_stocks, bg="#4CAF50", fg="white")
        predict_industry_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # 行业状态标签
        self.industry_status_label = tk.Label(industry_frame, text="请选择行业并点击加载成分股", anchor="w")
        self.industry_status_label.pack(fill=tk.X, padx=5, pady=5)
        
        # ===== 中间面板: 预测结果表格 =====
        result_frame = ttk.LabelFrame(middle_frame, text="预测结果")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建Treeview来显示预测结果
        self.result_tree = ttk.Treeview(result_frame, columns=("stock", "prediction", "confidence", "target_date"),
                                         show="headings", height=10)
        self.result_tree.heading("stock", text="股票代码")
        self.result_tree.heading("prediction", text="预测")
        self.result_tree.heading("confidence", text="置信度")
        self.result_tree.heading("target_date", text="目标日期")
        
        self.result_tree.column("stock", width=100)
        self.result_tree.column("prediction", width=80)
        self.result_tree.column("confidence", width=80)
        self.result_tree.column("target_date", width=100)
        
        self.result_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        
        # 特征重要性部分
        feature_frame = ttk.LabelFrame(middle_frame, text="特征重要性")
        feature_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.feature_text = ScrolledText(feature_frame, width=40, height=10)
        self.feature_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ===== 右侧面板: 可视化图表 =====
        vis_frame = ttk.LabelFrame(right_frame, text="可视化图表")
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建上中下三个区域，同时显示所有图表，每个高度比例合理
        price_frame = ttk.LabelFrame(vis_frame, text="价格走势")
        price_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        indicators_frame = ttk.LabelFrame(vis_frame, text="技术指标")
        indicators_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        confidence_frame = ttk.LabelFrame(vis_frame, text="预测置信度")
        confidence_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建价格图表
        self.price_fig, self.price_ax = plt.subplots(figsize=(8, 3.5))
        self.price_fig.tight_layout()
        self.price_canvas = FigureCanvasTkAgg(self.price_fig, master=price_frame)
        self.price_canvas.draw()
        self.price_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建技术指标图表
        self.indicators_fig = plt.figure(figsize=(8, 3.5))
        self.indicators_canvas = FigureCanvasTkAgg(self.indicators_fig, master=indicators_frame)
        self.indicators_canvas.draw()
        self.indicators_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建预测置信度图表
        self.confidence_fig, self.confidence_ax = plt.subplots(figsize=(8, 3.5))
        self.confidence_fig.tight_layout()
        self.confidence_canvas = FigureCanvasTkAgg(self.confidence_fig, master=confidence_frame)
        self.confidence_canvas.draw()
        self.confidence_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 加载行业列表
        self._load_industry_list()
    
    def _clear_stock_list(self):
        """清空股票列表"""
        self.stock_listbox.delete(0, tk.END)
        
    def _predict_industry_stocks(self):
        """直接预测所选行业成分股"""
        # 获取所选的行业成分股
        selected_indices = self.industry_stocks_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("提示", "请先选择要预测的行业成分股")
            return
        
        # 获取所选股票代码
        selected_stocks = []
        for i in selected_indices:
            item = self.industry_stocks_listbox.get(i)
            if " - " in item:
                stock_code = item.split(" - ")[0].strip()
            else:
                stock_code = item.strip()
            selected_stocks.append(stock_code)
        
        if not selected_stocks:
            messagebox.showinfo("提示", "未能获取有效的股票代码")
            return
        
        # 获取参数
        try:
            prediction_horizon = int(self.horizon_var.get())
            confidence_threshold = float(self.threshold_var.get())
            lookback_days = int(self.lookback_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的参数")
            return
        
        # 清空以前的结果
        for i in self.result_tree.get_children():
            self.result_tree.delete(i)
        self.feature_text.delete("1.0", tk.END)
        
        # 更新状态
        self.status_var.set("正在预测行业成分股...")
        
        # 使用线程进行预测
        thread = Thread(target=self._prediction_thread, 
                       args=(selected_stocks, prediction_horizon, confidence_threshold, lookback_days))
        thread.daemon = True
        thread.start()
        
    def _load_industry_stocks(self):
        """根据选择的行业加载成分股，并直接添加到已选择股票池"""
        industry = self.industry_var.get()
        if not industry or industry in ["无数据", "加载失败", "加载中..."]:
            messagebox.showinfo("提示", "请先选择有效的行业")
            return
            
        try:
            # 更新状态标签
            self.industry_status_label.config(text=f"正在加载 {industry} 行业成分股...")
            self.root.update()
            
            # 使用industry_data模块获取所选行业的成分股
            stocks_df = get_stocks_by_industry_name(industry)
            
            if stocks_df.empty:
                self.industry_status_label.config(text=f"未找到 {industry} 行业的成分股数据")
                messagebox.showinfo("提示", f"未找到{industry}行业的成分股数据")
                return
                
            # 清空行业成分股列表和已选择股票列表
            self.industry_stocks_listbox.delete(0, tk.END)
            self.stock_listbox.delete(0, tk.END)  # 清空已选择股票列表
            
            # 初始化all_industry_stocks列表
            self.all_industry_stocks = []
            
            # 从DataFrame中提取股票代码和名称添加到列表中
            for _, row in stocks_df.iterrows():
                stock_code = row['stock_code']
                stock_name = row['stock_name'] if 'stock_name' in row and not pd.isna(row['stock_name']) else ""
                display_text = f"{stock_code} - {stock_name}" if stock_name else stock_code
                
                # 添加到所有行业股票列表
                self.all_industry_stocks.append(display_text)
                
                # 添加到行业成分股显示列表
                self.industry_stocks_listbox.insert(tk.END, display_text)
                
                # 同时添加到已选择股票列表
                self.stock_listbox.insert(tk.END, display_text)
                
            # 更新状态标签
            count = stocks_df.shape[0]
            self.industry_status_label.config(text=f"成功加载 {industry} 行业的 {count} 只成分股")
            
            # 清空搜索框
            self.search_var.set("")
            
            logger.info(f"成功加载{industry}行业的{count}只成分股并添加到已选择股票池")
            messagebox.showinfo("成功", f"已加载{industry}行业的{count}只成分股并添加到已选择股票池")
        except Exception as e:
            self.industry_status_label.config(text=f"加载行业成分股出错: {e}")
            logger.error(f"加载行业成分股出错: {e}")
            messagebox.showerror("错误", f"加载行业成分股失败: {e}")
        
    def _add_selected_industry_stocks(self):
        """将所有行业成分股添加到主股票列表"""
        # 获取所有行业成分股
        all_stocks = self.industry_stocks_listbox.get(0, tk.END)
        if not all_stocks:
            messagebox.showinfo("提示", "当前没有行业成分股可添加")
            return
            
        # 将所有股票添加到主列表（避免重复）
        existing_stocks = list(self.stock_listbox.get(0, tk.END))
        added_count = 0
        
        for stock in all_stocks:
            if stock not in existing_stocks:
                self.stock_listbox.insert(tk.END, stock)
                added_count += 1
        
        messagebox.showinfo("成功", f"已添加 {added_count} 只股票到主列表")
    
    def _predict_selected(self):
        """运行预测任务"""
        # 获取所有股票代码
        stock_items = list(self.stock_listbox.get(0, tk.END))
        if not stock_items:
            messagebox.showwarning("警告", "请先添加股票")
            return
        
        # 从股票项中提取股票代码（去掉可能的股票名称部分）
        stocks = []
        for item in stock_items:
            if " - " in item:
                stock_code = item.split(" - ")[0].strip()  # 提取股票代码部分
            else:
                stock_code = item.strip()
            stocks.append(stock_code)
            
        # 获取参数
        try:
            prediction_horizon = int(self.horizon_var.get())
            confidence_threshold = float(self.threshold_var.get())
            lookback_days = int(self.lookback_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的参数")
            return
            
        # 清空以前的结果
        for i in self.result_tree.get_children():
            self.result_tree.delete(i)
        self.feature_text.delete("1.0", tk.END)
        
        # 更新状态
        self.status_var.set("正在预测...")
        
        # 询问是否进行模型评估
        if len(stocks) == 1 and messagebox.askyesno("模型评估", 
                                             f"是否为股票 {stocks[0]} 进行模型历史评估？\n"
                                             f"这将显示过去{lookback_days//2}天的模型表现。"):
            # 使用线程避免UI冻结
            thread = Thread(target=self._model_evaluation_thread, 
                           args=(stocks[0], prediction_horizon, confidence_threshold, lookback_days))
            thread.daemon = True
            thread.start()
        else:
            # 使用线程避免UI冻结
            thread = Thread(target=self._prediction_thread, 
                           args=(stocks, prediction_horizon, confidence_threshold, lookback_days))
            thread.daemon = True
            thread.start()
            
    def _model_evaluation_thread(self, stock, prediction_horizon, confidence_threshold, lookback_days):
        """
        模型评估线程，分析历史预测准确率
        """
        try:
            # 更新状态
            self.root.after(0, lambda: self.status_var.set(f"正在进行 {stock} 的模型历史评估..."))
            
            # 获取更长时间的历史数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days * 2)  # 使用两倍回溯期进行评估
            
            try:
                data = self.data_fetcher.get_stock_data(
                    stock, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                
                if data is None or len(data) < lookback_days:
                    self.root.after(0, lambda s=stock: messagebox.showwarning(
                        "警告", f"股票 {s} 的数据不足，无法进行历史评估"))
                    return
            except Exception as e:
                self.root.after(0, lambda s=stock, err=str(e): messagebox.showerror(
                    "错误", f"获取股票 {s} 数据失败: {err}"))
                return
                
            # 添加高级指标
            try:
                data = AdvancedIndicators.add_advanced_indicators(data)
            except Exception as e:
                logger.error(f"添加指标时出错: {e}")
                # 继续执行
                pass
            
            # 创建模型实例
            ml_model = MLPredictionModel(prediction_horizon=prediction_horizon)
            
            # 进行历史性能评估
            eval_results = self._evaluate_model_history(ml_model, data, prediction_horizon)
            
            # 在GUI中显示评估结果
            self.root.after(0, lambda r=eval_results, s=stock: self._display_evaluation_results(r, s))
            
            # 同时也进行常规预测
            prediction = ml_model.predict(data)
            
            # 将结果添加到结果表格
            last_date = prediction.index[-1]
            target_date = last_date + pd.Timedelta(days=prediction_horizon)
            target_date_str = target_date.strftime('%Y-%m-%d')
            
            prediction_value = "上涨" if prediction['prediction'].iloc[-1] == 1 else "下跌"
            confidence_score = f"{float(prediction['confidence'].iloc[-1]):.2f}"
            
            # 更新结果表格
            self.root.after(0, lambda s=stock, p=prediction_value, conf=confidence_score, 
                           d=target_date_str: self._update_result_table(s, p, conf, d))
            
            # 绘制图表
            self.root.after(0, lambda s=stock, d=data, p=prediction: self._update_all_charts(s, d, p))
            
            # 显示特征重要性
            self.root.after(0, lambda m=ml_model: self._show_feature_importance(m))
            
            # 更新状态
            self.root.after(0, lambda: self.status_var.set(f"{stock} 的模型评估已完成"))
            
        except Exception as e:
            logger.error(f"模型评估时出错: {e}")
            self.root.after(0, lambda err=str(e): messagebox.showerror(
                "错误", f"模型评估失败: {err}"))
            self.root.after(0, lambda: self.status_var.set("评估失败"))
    
    def _evaluate_model_history(self, model, data, prediction_horizon):
        """评估模型在历史数据上的表现"""
        results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'profit_factor': [],
            'win_rate': [],
            'dates': [],
            'predictions': [],
            'actuals': [],
            'confidences': []
        }
        
        # 至少需要60天数据进行评估
        min_train_days = 60
        
        # 将数据分割成多个滚动窗口进行评估
        test_start_idx = min_train_days
        max_test_idx = len(data) - prediction_horizon - 1  # 确保有足够的未来数据验证预测
        
        # 选择合适的评估点间隔，避免过多计算
        step = max(1, (max_test_idx - test_start_idx) // 30)  # 最多30个评估点
        
        for i in range(test_start_idx, max_test_idx, step):
            # 训练数据
            train_data = data.iloc[:i].copy()
            
            # 根据实际价格变化计算真实标签
            actual_future_price = data.iloc[i + prediction_horizon]['close']
            actual_current_price = data.iloc[i]['close']
            actual_label = 1 if actual_future_price > actual_current_price else 0
            
            # 使用模型预测
            try:
                prediction = model.predict(train_data)
                if prediction is not None and not prediction.empty:
                    pred_value = prediction['prediction'].iloc[-1]
                    confidence = prediction['confidence'].iloc[-1]
                    
                    # 保存预测结果
                    results['dates'].append(train_data.index[i])
                    results['predictions'].append(pred_value)
                    results['actuals'].append(actual_label)
                    results['confidences'].append(confidence)
            except Exception as e:
                logger.warning(f"在历史点 {i} 进行预测时出错: {e}")
                continue
                
        # 如果有足够的预测点，计算性能指标
        if len(results['predictions']) > 5:
            # 计算准确率
            correct = sum(1 for p, a in zip(results['predictions'], results['actuals']) if p == a)
            total = len(results['predictions'])
            accuracy = correct / total if total > 0 else 0
            
            # 计算精确率和召回率
            true_pos = sum(1 for p, a in zip(results['predictions'], results['actuals']) if p == 1 and a == 1)
            false_pos = sum(1 for p, a in zip(results['predictions'], results['actuals']) if p == 1 and a == 0)
            false_neg = sum(1 for p, a in zip(results['predictions'], results['actuals']) if p == 0 and a == 1)
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            
            # 计算F1得分
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # 计算简单的盈利因子
            winning_trades = sum(1 for p, a in zip(results['predictions'], results['actuals']) if p == a and p == 1)
            losing_trades = sum(1 for p, a in zip(results['predictions'], results['actuals']) if p != a and p == 1)
            profit_factor = winning_trades / losing_trades if losing_trades > 0 else float('inf')
            
            # 计算胜率
            win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
            
            # 保存结果
            results['accuracy'] = accuracy
            results['precision'] = precision
            results['recall'] = recall
            results['f1_score'] = f1
            results['profit_factor'] = profit_factor
            results['win_rate'] = win_rate
        
        return results
    
    def _display_evaluation_results(self, results, stock_code):
        """在GUI中显示模型评估结果"""
        # 清空特征重要性文本框，用于显示评估结果
        self.feature_text.delete("1.0", tk.END)
        
        # 设置结果文本
        eval_text = f"模型评估结果: {stock_code}\n"
        eval_text += "="*30 + "\n\n"
        
        if 'accuracy' in results and isinstance(results['accuracy'], (int, float)):
            eval_text += f"准确率: {results['accuracy']:.2%}\n"
            eval_text += f"精确率: {results['precision']:.2%}\n"
            eval_text += f"召回率: {results['recall']:.2%}\n"
            eval_text += f"F1得分: {results['f1_score']:.2f}\n"
            eval_text += f"盈利因子: {results['profit_factor']:.2f}\n"
            eval_text += f"胜率: {results['win_rate']:.2%}\n\n"
            
            # 显示评估结果的简单解释
            if results['accuracy'] > 0.65:
                eval_text += "模型性能良好，预测准确率高于65%\n"
            elif results['accuracy'] > 0.55:
                eval_text += "模型性能一般，预测准确率略高于随机猜测\n"
            else:
                eval_text += "模型性能不佳，预测准确率接近或低于随机猜测\n"
                
            if results['profit_factor'] > 2:
                eval_text += "盈利因子较高，模型有较好的盈利潜力\n"
            elif results['profit_factor'] > 1:
                eval_text += "盈利因子大于1，模型可能有盈利潜力\n"
            else:
                eval_text += "盈利因子较低，模型可能难以产生稳定收益\n"
        else:
            eval_text += "没有足够的历史数据点进行有效评估。\n"
            eval_text += "请考虑使用更长的数据回溯期进行评估。\n"
        
        # 显示评估结果
        self.feature_text.insert(tk.END, eval_text)
        
        # 如果有足够的预测点，还可以绘制历史预测准确率图表
        if 'dates' in results and len(results['dates']) > 5:
            self._plot_evaluation_results(results, stock_code)
        
    def _prediction_thread(self, stocks, prediction_horizon, confidence_threshold, lookback_days):
        """
        预测线程
        
        Args:
            stocks: 股票列表
            prediction_horizon: 预测天数
            confidence_threshold: 信心阈值
            lookback_days: 回溯天数
        """
        results = []
        
        # 为预测创建新的模型实例
        try:
            ml_model = MLPredictionModel(prediction_horizon=prediction_horizon)
        except Exception as e:
            logger.error(f"创建模型实例时出错: {e}")
            self.root.after(0, lambda err=str(e): messagebox.showerror(
                "错误", f"创建预测模型时出错: {err}"))
            self.root.after(0, lambda: self.status_var.set("预测失败"))
            return
        
        for stock in stocks:
            try:
                # 更新状态
                self.root.after(0, lambda s=stock: self.status_var.set(f"正在加载和预测 {s} 的数据..."))
                
                # 获取历史数据
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)
                
                try:
                    data = self.data_fetcher.get_stock_data(
                        stock, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                    
                    if data is None or len(data) < 60:
                        self.root.after(0, lambda s=stock: messagebox.showwarning(
                            "警告", f"股票 {s} 的数据不足，跳过预测"))
                        continue
                except Exception as e:
                    self.root.after(0, lambda s=stock, err=str(e): messagebox.showerror(
                        "错误", f"获取股票 {s} 数据失败: {err}"))
                    continue
                    
                # 添加高级指标
                try:
                    data = AdvancedIndicators.add_advanced_indicators(data)
                except Exception as e:
                    logger.error(f"重新获取数据并添加指标时出错: {e}")
                    # 继续执行，不显示错误对话框
                    pass
                
                # 预测
                try:
                    prediction = ml_model.predict(data)
                    # --- DEBUG LOGGING START ---
                    logger.debug(f"模型预测原始返回 (Stock: {stock}):\n{prediction}")
                    if prediction is not None and 'confidence' in prediction.columns:
                        conf_series = prediction['confidence']
                        logger.debug(f"置信度列类型: {conf_series.dtype}")
                        logger.debug(f"置信度列尾部值:\n{conf_series.tail()}")
                    # --- DEBUG LOGGING END ---
                except Exception as e:
                    logger.error(f"运行预测时出错: {e}")
                    # Optional: Display error in GUI status or a message box
                    self.root.after(0, lambda s=stock, err=str(e): self._update_status(f"错误：预测 {s} 失败: {err}"))
                    continue # Skip to next stock if prediction fails

                # Check if prediction DataFrame is None or empty after predict call
                if prediction is None or prediction.empty:
                    logger.warning(f"模型预测返回空结果 for stock {stock}. 跳过。")
                    self.root.after(0, lambda s=stock: self._update_status(f"警告：{s} 预测结果为空."))
                    continue

                # 获取预测结果DataFrame的最后一个索引（代表最新日期）
                last_date = prediction.index[-1]
                # 计算目标日期 - 直接使用传入的 prediction_horizon
                try:
                     target_date = last_date + pd.Timedelta(days=prediction_horizon)
                     target_date_str = target_date.strftime('%Y-%m-%d')
                except Exception as date_err:
                     logger.error(f"计算目标日期时出错: {date_err}")
                     target_date_str = "Error"

                # 将结果添加到列表
                prediction_value = "上涨" if prediction['prediction'].iloc[-1] == 1 else "下跌"
                # Add error handling for confidence score extraction
                try:
                    raw_confidence = prediction['confidence'].iloc[-1]
                    if pd.isna(raw_confidence):
                        confidence_score = "N/A"
                    else:
                        confidence_score = f"{float(raw_confidence):.2f}"
                except (KeyError, IndexError, ValueError, TypeError) as e:
                    logger.warning(f"无法提取或格式化置信度分数 for {stock}: {e}")
                    confidence_score = "N/A"
                
                # Append with calculated target_date_str and confidence_score
                results.append((stock, prediction_value, confidence_score, target_date_str, data, ml_model))
                
                # 更新结果表格 (use target_date_str and confidence_score)
                self.root.after(0, lambda s=stock, p=prediction_value, conf=confidence_score, 
                               d=target_date_str: self._update_result_table(s, p, conf, d))
                
                # 如果是第一个结果，则显示图表和特征重要性
                if len(results) == 1:
                    self.root.after(0, lambda r=results[0]: self._display_first_result(r))
                
            except Exception as e:
                logger.error(f"预测股票 {stock} 时出错: {e}")
                self.root.after(0, lambda s=stock, err=str(e): messagebox.showerror(
                    "错误", f"预测股票 {s} 时出错: {err}"))
        
        self.root.after(0, lambda: self.status_var.set(f"预测完成，共 {len(results)} 个结果"))
        
        # 如果没有结果，显示相应的消息
        if not results:
            self.root.after(0, lambda: messagebox.showinfo("结果", "没有成功的预测结果"))
        
    def _update_result_table(self, stock, prediction, confidence, target_date):
        """更新结果表格"""
        # 为预测结果设置适当的颜色
        if prediction == "上涨":
            prediction_text = "↑ 上涨"
            tag = "bullish"
        else:
            prediction_text = "↓ 下跌"
            tag = "bearish"
            
        # 为概率设置颜色, handle non-numeric confidence
        prob_tag = "default_prob" # Default tag
        if confidence != "N/A":
            try:
                prob_value = float(confidence)
                if prob_value > 0.7:
                    prob_tag = "high_prob"
                elif prob_value > 0.6:
                    prob_tag = "medium_prob"
                else:
                    prob_tag = "low_prob"
            except ValueError:
                logger.warning(f"无法将置信度 '{confidence}' 转换为浮点数进行颜色标记。")
                # Keep default tag

        # 添加到表格
        item_id = self.result_tree.insert("", tk.END, values=(stock, prediction_text, confidence, target_date))
        
        # 配置标签颜色
        self.result_tree.tag_configure("bullish", foreground="green")
        self.result_tree.tag_configure("bearish", foreground="red")
        self.result_tree.tag_configure("high_prob", foreground="blue")
        self.result_tree.tag_configure("medium_prob", foreground="purple")
        self.result_tree.tag_configure("low_prob", foreground="orange")
        self.result_tree.tag_configure("default_prob", foreground="black") # Add default tag color
        
        # 应用标签
        self.result_tree.item(item_id, tags=(tag, prob_tag))
        
        # 为结果项添加点击事件
        self.result_tree.bind('<ButtonRelease-1>', self._on_result_select)
        
    def _on_result_select(self, event):
        """当用户选择结果项时的回调"""
        # 获取选中的项
        selection = self.result_tree.selection()
        if not selection:
            return
            
        # 获取选中项的值
        item_id = selection[0]
        item_values = self.result_tree.item(item_id, 'values')
        
        if not item_values:
            return
            
        # 查找对应的结果
        stock = item_values[0]
        
        # 显示对应的结果
        for result in self._get_results():
            if result[0] == stock:
                self._display_result(result)
                break
                
    def _get_results(self):
        """获取所有结果项"""
        results = []
        for item_id in self.result_tree.get_children():
            item_values = self.result_tree.item(item_id, 'values')
            if item_values:
                stock = item_values[0]
                prediction = "上涨" if "上涨" in item_values[1] else "下跌"
                confidence = item_values[2]
                target_date = item_values[3]
                results.append((stock, prediction, confidence, target_date, None, None))
        return results
                
    def _display_first_result(self, result):
        """显示第一个结果的图表和特征重要性"""
        if not result:
            return
            
        self._display_result(result)
        
    def _display_result(self, result):
        """显示选中结果的图表和特征重要性"""
        if not result:
            return
            
        stock, prediction, confidence, target_date, data, model = result
        
        # 如果数据不完整，尝试重新获取
        if data is None or model is None:
            try:
                # 获取数据
                end_date = datetime.now()
                start_date = end_date - timedelta(days=int(self.lookback_var.get()))
                
                data = self.data_fetcher.get_stock_data(
                    stock, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                
                if data is None or len(data) < 60:
                    messagebox.showwarning("警告", f"无法获取股票 {stock} 的完整数据")
                    return
                    
                # 添加高级指标
                data = AdvancedIndicators.add_advanced_indicators(data)
                
                # 创建模型实例
                model = MLPredictionModel()
            except Exception as e:
                logger.error(f"重新获取数据时出错: {e}")
                messagebox.showerror("错误", f"无法显示股票 {stock} 的详细信息: {e}")
                return
        
        # 绘制图表
        self._update_all_charts(stock, data, prediction)
        
        # 显示特征重要性
        if model:
            self._show_feature_importance(model)
        
    def _update_all_charts(self, stock, data, prediction_result):
        """更新所有图表"""
        self._plot_price_chart(stock, data)
        self._plot_indicators_chart(stock, data)
        
        # 获取置信度值
        confidence_value = None
        if prediction_result is not None:
            if isinstance(prediction_result, pd.DataFrame) and 'confidence' in prediction_result.columns:
                confidence_value = prediction_result['confidence'].iloc[-1]
            elif isinstance(prediction_result, str) and prediction_result in ["上涨", "下跌"]:
                # 如果是字符串，则可能是预测结果本身
                prediction_str = prediction_result
            else:
                # 尝试从结果元组中获取置信度
                try:
                    prediction_str, confidence_value = prediction_result
                    confidence_value = float(confidence_value) if confidence_value != "N/A" else 0.5
                except (ValueError, TypeError):
                    logger.warning(f"无法解析预测结果: {prediction_result}")
                
        self._plot_confidence_chart(stock, confidence_value)
        
    def _plot_price_chart(self, stock_code, data):
        """绘制价格走势图"""
        if data is None or data.empty:
            return
            
        # 清除当前图形
        self.price_ax.clear()
        
        # 设置背景色和网格样式
        self.price_fig.set_facecolor('white')
        self.price_ax.set_facecolor('white')
        self.price_ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        
        # 绘制收盘价
        self.price_ax.plot(data.index[-120:], data['close'].iloc[-120:], 'blue', linewidth=2, label='价格')
        
        # 绘制均线
        if 'ma20' in data.columns:
            self.price_ax.plot(data.index[-120:], data['ma20'].iloc[-120:], 'red', linewidth=1, linestyle='--', label='MA20')
        if 'ma60' in data.columns:
            self.price_ax.plot(data.index[-120:], data['ma60'].iloc[-120:], 'green', linewidth=1, linestyle='--', label='MA60')
        
        # 绘制布林带，如果数据中有这些列
        if all(col in data.columns for col in ['boll_upper', 'boll_middle', 'boll_lower']):
            self.price_ax.plot(data.index[-120:], data['boll_upper'].iloc[-120:], 'purple', linewidth=1, linestyle='-.', alpha=0.5, label='布林上轨')
            self.price_ax.plot(data.index[-120:], data['boll_middle'].iloc[-120:], 'orange', linewidth=1, linestyle='-.', alpha=0.5, label='布林中轨')
            self.price_ax.plot(data.index[-120:], data['boll_lower'].iloc[-120:], 'purple', linewidth=1, linestyle='-.', alpha=0.5, label='布林下轨')
            # 填充布林带区域
            self.price_ax.fill_between(data.index[-120:], data['boll_upper'].iloc[-120:], data['boll_lower'].iloc[-120:], color='skyblue', alpha=0.1)
        
        # 添加上下箭头标记
        last_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2] if len(data) > 1 else last_price
        if last_price > prev_price:
            self.price_ax.annotate('↑', xy=(data.index[-1], last_price), xytext=(data.index[-1], last_price*1.02),
                              fontsize=12, color='red', ha='center')
        else:
            self.price_ax.annotate('↓', xy=(data.index[-1], last_price), xytext=(data.index[-1], last_price*1.02),
                              fontsize=12, color='red', ha='center')
        
        # 添加图例
        self.price_ax.legend(loc='best', fontsize=8)
        
        # 设置标题和标签
        self.price_ax.set_title(f'{stock_code} 价格走势', fontsize=14, fontweight='bold')
        self.price_ax.set_xlabel('date', fontsize=10)
        self.price_ax.set_ylabel('价格', fontsize=10)
        
        # 调整日期格式
        self.price_ax.tick_params(axis='x', rotation=45)
        
        # 调整布局
        self.price_fig.tight_layout()
        
        # 更新画布
        self.price_canvas.draw()
        
    def _plot_indicators_chart(self, stock_code, data):
        """绘制技术指标图"""
        if data is None or data.empty:
            return
            
        # 清除当前图形
        self.indicators_fig.clf()
        
        # 创建三个子图：RSI, MACD, KDJ
        gs = self.indicators_fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
        ax1 = self.indicators_fig.add_subplot(gs[0])  # RSI
        ax2 = self.indicators_fig.add_subplot(gs[1], sharex=ax1)  # MACD
        ax3 = self.indicators_fig.add_subplot(gs[2], sharex=ax1)  # KDJ
        
        # 设置背景色和样式
        self.indicators_fig.set_facecolor('white')
        ax1.set_facecolor('white')
        ax2.set_facecolor('white')
        ax3.set_facecolor('white')
        ax1.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        ax2.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        ax3.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        
        # 最近的数据点数量
        n_points = 90
        
        # 绘制RSI
        if 'rsi14' in data.columns:
            ax1.plot(data.index[-n_points:], data['rsi14'].iloc[-n_points:], 'blue', linewidth=1.5, label='RSI(14)')
            ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax1.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
            ax1.set_ylim(0, 100)
            ax1.legend(loc='upper left', fontsize=8)
            ax1.set_title('RSI指标', fontsize=10)
            ax1.set_ylabel('RSI', fontsize=9)
            
        # 绘制MACD
        if all(col in data.columns for col in ['macd', 'macd_signal']):
            line1 = ax2.plot(data.index[-n_points:], data['macd'].iloc[-n_points:], 'blue', linewidth=1.5, label='MACD')
            line2 = ax2.plot(data.index[-n_points:], data['macd_signal'].iloc[-n_points:], 'red', linewidth=1.5, label='Signal')
            
            # 绘制MACD柱状图
            if 'macd_hist' in data.columns:
                hist_values = data['macd_hist'].iloc[-n_points:]
                for i, val in enumerate(hist_values):
                    if i < len(data.index[-n_points:]):  # 确保索引有效
                        color = 'green' if val >= 0 else 'red'
                        ax2.bar(data.index[-n_points:][i], val, width=1.0, color=color, alpha=0.5)
            
            # 添加零线
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax2.legend(loc='upper left', fontsize=8)
            ax2.set_title('MACD指标', fontsize=10)
            ax2.set_ylabel('MACD', fontsize=9)
            
        # 绘制KDJ
        if all(col in data.columns for col in ['stoch_k', 'stoch_d']):
            ax3.plot(data.index[-n_points:], data['stoch_k'].iloc[-n_points:], 'blue', linewidth=1.5, label='%K')
            ax3.plot(data.index[-n_points:], data['stoch_d'].iloc[-n_points:], 'red', linewidth=1.5, label='%D')
            
            # 添加KDJ阈值线
            ax3.axhline(y=80, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(y=20, color='green', linestyle='--', alpha=0.5)
            ax3.set_ylim(0, 100)
            ax3.legend(loc='upper left', fontsize=8)
            ax3.set_title('KDJ指标', fontsize=10)
            ax3.set_ylabel('KDJ', fontsize=9)
            ax3.set_xlabel('日期', fontsize=9)
        
        # 调整日期格式
        ax3.tick_params(axis='x', rotation=45)
        
        # 调整布局
        self.indicators_fig.tight_layout()
        
        # 更新画布
        self.indicators_canvas.draw()
        
    def _plot_confidence_chart(self, stock_code, confidence_value):
        """绘制预测置信度图"""
        # 清除当前图形
        self.confidence_ax.clear()
        
        # 设置背景色
        self.confidence_fig.set_facecolor('white')
        self.confidence_ax.set_facecolor('white')
        self.confidence_ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        
        # 确保有有效的置信度值
        if confidence_value is not None:
            # 将置信度转换为0-1之间的值
            if isinstance(confidence_value, pd.Series):
                confidence_value = confidence_value.iloc[-1]
            elif isinstance(confidence_value, pd.DataFrame) and 'confidence' in confidence_value.columns:
                confidence_value = confidence_value['confidence'].iloc[-1]
            
            try:
                confidence_value = float(confidence_value)
            except (ValueError, TypeError):
                confidence_value = 0.5  # 默认值
                
            # 选择颜色 - 置信度超过0.7为深红色，超过0.6为红色，超过0.5为浅红色，低于0.5为绿色
            if confidence_value > 0.7:
                color = '#FF0000'  # 深红色
            elif confidence_value > 0.6:
                color = '#FF6347'  # 红色
            elif confidence_value > 0.5:
                color = '#FFA07A'  # 浅红色
            else:
                color = '#32CD32'  # 绿色
                
            # 绘制条形图
            bar = self.confidence_ax.bar(['看涨概率'], [confidence_value], color=color, width=0.4)
            
            # 在条形上方添加文本标签
            self.confidence_ax.text(0, confidence_value + 0.02, f'{confidence_value:.2f}', 
                                ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # 添加参考线
            self.confidence_ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
            self.confidence_ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7)
            self.confidence_ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
            
            # 设置图表属性
            self.confidence_ax.set_ylim(0, 1.0)
            self.confidence_ax.set_title(f'{stock_code} 预测信心', fontsize=14, fontweight='bold')
            self.confidence_ax.set_xlabel('看涨概率', fontsize=10)
            self.confidence_ax.set_yticks([0.00, 0.25, 0.50, 0.60, 0.70, 0.85, 1.00])
            
            # 添加阈值说明文本
            self.confidence_ax.text(0.85, 0.5, '中性', ha='center', va='bottom', fontsize=8)
            self.confidence_ax.text(0.85, 0.6, '偏多', ha='center', va='bottom', fontsize=8)
            self.confidence_ax.text(0.85, 0.7, '强多', ha='center', va='bottom', fontsize=8)
            
        # 调整布局
        self.confidence_fig.tight_layout()
        
        # 更新画布
        self.confidence_canvas.draw()
        
    def _show_feature_importance(self, model):
        """显示特征重要性"""
        # 清空文本框
        self.feature_text.delete("1.0", tk.END)
        
        # 获取特征重要性
        try:
            feature_importance = model.get_feature_importance()
        except Exception as e:
            logger.error(f"获取特征重要性时出错: {e}")
            self.feature_text.insert(tk.END, f"无法获取特征重要性: {e}")
            return
        
        # 检查返回结果
        if feature_importance is None or feature_importance.empty:
            self.feature_text.insert(tk.END, "模型未提供特征重要性数据。")
            return
            
        # 格式化并显示 Top 10 特征
        # Select top 10 features here
        top_features = feature_importance.head(10)
        feature_text_content = "特征重要性 (Top 10):\n" + "-"*20 + "\n"
        for feature, importance in top_features.items():
            feature_text_content += f"{feature}: {importance:.4f}\n"
            
        self.feature_text.insert(tk.END, feature_text_content)
        
    def _load_industry_list(self):
        """加载行业列表并更新下拉菜单"""
        try:
            # 使用industry_data模块获取行业列表
            industries = get_simplified_industry_list()
            
            # 更新下拉菜单
            self.industry_combobox['values'] = industries
            self.industry_var.set(industries[0] if industries else "无数据")
            
            logger.info(f"成功加载 {len(industries)} 个行业")
        except Exception as e:
            logger.error(f"加载行业列表出错: {e}")
            messagebox.showerror("错误", f"加载行业列表失败: {e}")
            self.industry_var.set("加载失败")
    
    def _update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)
        
    def _filter_stocks(self, event=None):
        """根据输入的文本过滤行业成分股列表"""
        search_text = self.search_var.get().lower()
        if not hasattr(self, 'all_industry_stocks'):
            # 如果还没有加载过行业成分股，直接返回
            return
            
        # 清空当前列表
        self.industry_stocks_listbox.delete(0, tk.END)
        
        # 重新添加匹配的股票
        for item in self.all_industry_stocks:
            if search_text in item.lower():
                self.industry_stocks_listbox.insert(tk.END, item)
                
        # 如果搜索框为空，显示所有股票
        if not search_text:
            for item in self.all_industry_stocks:
                self.industry_stocks_listbox.insert(tk.END, item)
                
    def _compare_models(self):
        """比较不同机器学习模型对所有股票的预测结果"""
        # 获取所有股票
        stock_items = list(self.stock_listbox.get(0, tk.END))
        if not stock_items:
            messagebox.showwarning("警告", "请先添加股票")
            return
            
        # 获取参数
        try:
            prediction_horizon = int(self.horizon_var.get())
            lookback_days = int(self.lookback_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的参数")
            return
        
        # 选择要比较的模型
        model_types = ["rf", "gbm", "xgb", "lgb", "ensemble"]
        
        # 清空以前的结果
        for i in self.result_tree.get_children():
            self.result_tree.delete(i)
        self.feature_text.delete("1.0", tk.END)
        
        # 更新状态
        self.status_var.set(f"正在对{len(stock_items)}只股票进行多模型比较...")
        
        # 为每个股票启动比较线程
        stocks = []
        for item in stock_items:
            if " - " in item:
                stock_code = item.split(" - ")[0].strip()
            else:
                stock_code = item.strip()
            stocks.append(stock_code)
            
        # 启动比较线程
        thread = Thread(target=self._run_models_comparison, 
                       args=(stocks, prediction_horizon, lookback_days, model_types))
        thread.daemon = True
        thread.start()
        
    def _run_models_comparison(self, stocks, prediction_horizon, lookback_days, model_types):
        """对多只股票运行不同模型进行比较"""
        try:
            # 创建结果字典保存所有预测
            all_results = {}
            
            # 记录第一只股票的数据用于后续展示
            first_stock_data = None
            first_stock_prediction = None
            first_stock_model = None
            
            # 对每只股票分别进行预测
            for stock_code in stocks:
                try:
                    # 更新状态
                    self.root.after(0, lambda s=stock_code: self.status_var.set(f"正在加载 {s} 的数据..."))
                    
                    # 获取数据
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=lookback_days)
                    
                    data = self.data_fetcher.get_stock_data(
                        stock_code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                        
                    if data is None or len(data) < 60:
                        self.root.after(0, lambda s=stock_code: messagebox.showwarning(
                            "警告", f"股票 {s} 的数据不足，跳过预测"))
                        continue
                        
                    # 添加指标
                    data = AdvancedIndicators.add_advanced_indicators(data)
                    
                    # 保存第一只股票的数据
                    if first_stock_data is None:
                        first_stock_data = data
                    
                    # 对每种模型进行预测
                    for model_type in model_types:
                        try:
                            # 更新状态
                            self.root.after(0, lambda s=stock_code, m=model_type: 
                                           self.status_var.set(f"正在使用{m}模型预测{s}..."))
                            
                            # 创建模型实例
                            model = MLPredictionModel(prediction_horizon=prediction_horizon, model_type=model_type)
                            
                            # 预测
                            prediction = model.predict(data)
                            
                            if prediction is not None and not prediction.empty:
                                # 获取预测结果
                                last_date = prediction.index[-1]
                                target_date = last_date + pd.Timedelta(days=prediction_horizon)
                                target_date_str = target_date.strftime('%Y-%m-%d')
                                
                                prediction_value = "上涨" if prediction['prediction'].iloc[-1] == 1 else "下跌"
                                confidence_score = f"{float(prediction['confidence'].iloc[-1]):.2f}"
                                
                                # 添加到结果表格
                                self.root.after(0, lambda s=f"{stock_code} ({model_type})", 
                                               p=prediction_value, conf=confidence_score, d=target_date_str: 
                                               self._update_result_table(s, p, conf, d))
                                
                                # 保存第一个模型对第一只股票的预测
                                if first_stock_prediction is None and first_stock_model is None:
                                    first_stock_prediction = prediction
                                    first_stock_model = model
                                
                        except Exception as e:
                            logger.error(f"使用 {model_type} 模型预测 {stock_code} 时出错: {e}")
                            continue
                        
                except Exception as e:
                    logger.error(f"处理股票 {stock_code} 时出错: {e}")
                    continue
            
            # 如果有结果，显示第一个模型的图表
            if first_stock_data is not None and first_stock_prediction is not None:
                stock_code = stocks[0] if stocks else "Unknown"
                self.root.after(0, lambda s=stock_code, d=first_stock_data, p=first_stock_prediction: 
                               self._update_all_charts(s, d, p))
                if first_stock_model is not None:
                    self.root.after(0, lambda m=first_stock_model: self._show_feature_importance(m))
            
            # 更新状态
            self.root.after(0, lambda: self.status_var.set(f"多模型比较已完成，共处理{len(stocks)}只股票"))
            
        except Exception as e:
            logger.error(f"进行多模型比较时出错: {e}")
            self.root.after(0, lambda err=str(e): messagebox.showerror(
                "错误", f"多模型比较失败: {err}"))
            self.root.after(0, lambda: self.status_var.set("多模型比较失败"))
    
    def _plot_evaluation_results(self, results, stock_code):
        """绘制模型评估结果图表"""
        # 检查有足够的数据点
        if len(results['dates']) < 5:
            return
            
        # 创建一个弹出窗口来显示评估图表
        eval_window = tk.Toplevel(self.root)
        eval_window.title(f"{stock_code} - 模型历史评估结果")
        eval_window.geometry("900x600")
        
        # 创建图表框架
        fig_frame = ttk.Frame(eval_window)
        fig_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建带有三个子图的图表
        fig = plt.Figure(figsize=(10, 8))
        fig.subplots_adjust(hspace=0.3)
        
        # 第一个子图：预测结果比较
        ax1 = fig.add_subplot(3, 1, 1)
        
        # 转换预测和实际值为更易读的格式
        pred_labels = ["上涨" if p == 1 else "下跌" for p in results['predictions']]
        actual_labels = ["上涨" if a == 1 else "下跌" for a in results['actuals']]
        
        # 计算预测是否正确
        correct_pred = [p == a for p, a in zip(results['predictions'], results['actuals'])]
        
        # 创建数据点颜色列表
        colors = ['green' if c else 'red' for c in correct_pred]
        
        # 设置预测点的Y坐标
        y_pred = [1] * len(results['dates'])
        y_actual = [0] * len(results['dates'])
        
        # 绘制散点图
        for i, (date, color) in enumerate(zip(results['dates'], colors)):
            ax1.scatter(date, 1, color=color, s=50, marker='o')  # 预测值
            ax1.scatter(date, 0, color='blue', s=30, marker='x')  # 实际值
            
            # 在正确的预测点之间绘制连接线
            if correct_pred[i]:
                ax1.plot([date, date], [0, 1], 'g-', alpha=0.3)
            else:
                ax1.plot([date, date], [0, 1], 'r-', alpha=0.3)
                
        # 设置Y轴标签
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['实际', '预测'])
        
        # 设置标题和标签
        ax1.set_title(f'{stock_code} 历史预测准确性', fontsize=12, fontweight='bold')
        ax1.set_ylabel('预测 vs 实际', fontsize=10)
        
        # 添加网格
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # 第二个子图：预测置信度
        ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
        
        # 使用颜色表示正确/错误的预测
        for i, (date, conf, color) in enumerate(zip(results['dates'], results['confidences'], colors)):
            ax2.bar(date, conf, color=color, width=5, alpha=0.7)
        
        # 添加置信度阈值线
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
        
        # 设置标题和标签
        ax2.set_title('预测置信度', fontsize=12, fontweight='bold')
        ax2.set_ylabel('置信度', fontsize=10)
        ax2.set_ylim(0, 1.0)
        
        # 添加网格
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # 第三个子图：累积性能
        ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
        
        # 计算累积正确预测数量
        cum_correct = np.cumsum(correct_pred)
        total_predictions = np.arange(1, len(correct_pred) + 1)
        cum_accuracy = cum_correct / total_predictions
        
        # 绘制累积准确率
        ax3.plot(results['dates'], cum_accuracy, 'b-', linewidth=2, label='累积准确率')
        
        # 添加基准线
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='随机猜测')
        
        # 设置标题和标签
        ax3.set_title('累积准确率', fontsize=12, fontweight='bold')
        ax3.set_ylabel('准确率', fontsize=10)
        ax3.set_ylim(0, 1.0)
        ax3.legend()
        
        # 添加网格
        ax3.grid(True, linestyle='--', alpha=0.3)
        
        # 格式化X轴日期
        fig.autofmt_xdate()
        
        # 创建图形窗口
        canvas = FigureCanvasTkAgg(fig, master=fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, fig_frame)
        toolbar.update()
        
    def _show_model_comparison_dialog(self, stock_code):
        """显示模型比较参数设置对话框"""
        # 创建对话框
        dialog = tk.Toplevel(self.root)
        dialog.title(f"多模型比较设置 - {stock_code}")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()  # 模态对话框
        
        # 参数框架
        param_frame = ttk.LabelFrame(dialog, text="参数设置")
        param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 预测天数
        tk.Label(param_frame, text="预测天数:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        horizon_var = tk.StringVar(value=self.horizon_var.get())
        horizon_values = ["5", "10", "20", "30"]
        horizon_combobox = ttk.Combobox(param_frame, textvariable=horizon_var, values=horizon_values, width=10)
        horizon_combobox.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # 回溯天数
        tk.Label(param_frame, text="回溯天数:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        lookback_var = tk.StringVar(value=self.lookback_var.get())
        lookback_values = ["120", "240", "365", "500"]
        lookback_combobox = ttk.Combobox(param_frame, textvariable=lookback_var, values=lookback_values, width=10)
        lookback_combobox.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # 选择要比较的模型
        tk.Label(param_frame, text="选择模型:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        
        # 创建复选框变量
        model_vars = {
            "随机森林 (RF)": tk.BooleanVar(value=True),
            "梯度提升树 (GBM)": tk.BooleanVar(value=True),
            "支持向量机 (SVM)": tk.BooleanVar(value=False),
            "XGBoost": tk.BooleanVar(value=True),
            "LightGBM": tk.BooleanVar(value=True),
            "集成模型": tk.BooleanVar(value=True)
        }
        
        # 创建复选框
        models_frame = ttk.Frame(param_frame)
        models_frame.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        row = 0
        for model_name, var in model_vars.items():
            tk.Checkbutton(models_frame, text=model_name, variable=var).grid(row=row, column=0, sticky="w")
            row += 1
        
        # 按钮框架
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 确定按钮
        def on_ok():
            # 获取选择的模型
            selected_models = [model_name.split(" ")[0].lower() for model_name, var in model_vars.items() if var.get()]
            
            # 关闭对话框
            dialog.destroy()
            
            # 运行模型比较
            try:
                prediction_horizon = int(horizon_var.get())
                lookback_days = int(lookback_var.get())
                
                # 启动比较线程
                thread = Thread(target=self._run_model_comparison, 
                               args=(stock_code, prediction_horizon, lookback_days, selected_models))
                thread.daemon = True
                thread.start()
            except ValueError as e:
                messagebox.showerror("错误", f"参数设置错误: {e}")
                
        # 取消按钮
        def on_cancel():
            dialog.destroy()
            
        ok_button = ttk.Button(button_frame, text="确定", command=on_ok)
        ok_button.pack(side=tk.RIGHT, padx=5)
        
        cancel_button = ttk.Button(button_frame, text="取消", command=on_cancel)
        cancel_button.pack(side=tk.RIGHT, padx=5)
        
    def _show_model_comparison_results(self, model_results, stock_code, data, prediction_horizon):
        """显示多模型比较结果"""
        # 检查结果
        if not model_results:
            messagebox.showwarning("警告", "没有成功的模型预测结果")
            return
            
        # 创建对话框
        dialog = tk.Toplevel(self.root)
        dialog.title(f"多模型比较结果 - {stock_code}")
        dialog.geometry("1000x700")
        
        # 创建笔记本控件
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 添加结果比较标签页
        summary_tab = ttk.Frame(notebook)
        notebook.add(summary_tab, text="结果比较")
        
        # 创建结果表格
        columns = ("model", "prediction", "confidence", "accuracy", "profit_factor", "win_rate")
        tree = ttk.Treeview(summary_tab, columns=columns, show="headings")
        
        # 设置列标题
        tree.heading("model", text="模型")
        tree.heading("prediction", text="预测结果")
        tree.heading("confidence", text="置信度")
        tree.heading("accuracy", text="历史准确率")
        tree.heading("profit_factor", text="盈利因子")
        tree.heading("win_rate", text="胜率")
        
        # 设置列宽
        tree.column("model", width=120)
        tree.column("prediction", width=100)
        tree.column("confidence", width=100)
        tree.column("accuracy", width=100)
        tree.column("profit_factor", width=100)
        tree.column("win_rate", width=100)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(summary_tab, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=0, pady=5)
        
        # 添加每个模型的结果
        model_names = {
            'rf': '随机森林 (RF)',
            'gbm': '梯度提升树 (GBM)',
            'svm': '支持向量机 (SVM)',
            'xgb': 'XGBoost',
            'lgb': 'LightGBM',
            'ensemble': '集成模型'
        }
        
        # 填充表格
        for model_code, result in model_results.items():
            prediction = result['prediction']
            evaluation = result['evaluation']
            
            # 获取最新预测
            pred_value = None
            confidence = None
            
            if prediction is not None and not prediction.empty:
                pred_value = prediction['prediction'].iloc[-1]
                confidence = prediction['confidence'].iloc[-1]
                
            # 格式化显示值
            display_prediction = "上涨" if pred_value == 1 else "下跌" if pred_value == 0 else "未知"
            display_confidence = f"{confidence:.2f}" if confidence is not None else "N/A"
            
            # 获取评估指标
            accuracy = evaluation.get('accuracy', None)
            profit_factor = evaluation.get('profit_factor', None)
            win_rate = evaluation.get('win_rate', None)
            
            display_accuracy = f"{accuracy:.2%}" if accuracy is not None else "N/A"
            display_profit = f"{profit_factor:.2f}" if profit_factor is not None else "N/A"
            display_winrate = f"{win_rate:.2%}" if win_rate is not None else "N/A"
            
            # 获取友好的模型名称
            model_name = model_names.get(model_code, model_code)
            
            # 添加到表格
            item_id = tree.insert("", tk.END, values=(
                model_name, display_prediction, display_confidence, 
                display_accuracy, display_profit, display_winrate
            ))
            
            # 设置颜色标签
            if pred_value == 1:
                tree.tag_configure(f"{model_code}_up", foreground="green")
                tree.item(item_id, tags=(f"{model_code}_up",))
            elif pred_value == 0:
                tree.tag_configure(f"{model_code}_down", foreground="red")
                tree.item(item_id, tags=(f"{model_code}_down",))
                
            # 为每个模型添加详细标签页
            details_tab = ttk.Frame(notebook)
            notebook.add(details_tab, text=f"{model_name}")
            
            # 在详细页中添加模型评估图表
            if evaluation and 'dates' in evaluation and len(evaluation['dates']) > 5:
                self._create_model_evaluation_chart(details_tab, evaluation, model_name, stock_code)
                
        # 添加价格图表标签页
        price_tab = ttk.Frame(notebook)
        notebook.add(price_tab, text="价格走势")
        
        # 绘制价格图表
        self._create_price_chart(price_tab, data, stock_code)
        
        # 计算目标日期
        last_date = data.index[-1]
        target_date = last_date + pd.Timedelta(days=prediction_horizon)
        target_date_str = target_date.strftime('%Y-%m-%d')
        
        # 添加结果说明
        info_frame = ttk.Frame(dialog)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        info_label = ttk.Label(info_frame, 
                              text=f"预测目标日期: {target_date_str}   |   " +
                                   f"分析基于 {len(data)} 天的历史数据   |   " +
                                   f"预测周期: {prediction_horizon} 天",
                              justify=tk.LEFT)
        info_label.pack(side=tk.LEFT, padx=5)
        
        # 添加关闭按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        close_button = ttk.Button(button_frame, text="关闭", command=dialog.destroy)
        close_button.pack(side=tk.RIGHT, padx=5)
        
    def _create_model_evaluation_chart(self, parent, evaluation, model_name, stock_code):
        """创建模型评估图表"""
        # 创建图表框架
        chart_frame = ttk.Frame(parent)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建图表
        fig = plt.Figure(figsize=(10, 8))
        fig.subplots_adjust(hspace=0.3)
        
        # 第一个子图：预测结果比较
        ax1 = fig.add_subplot(3, 1, 1)
        
        # 计算预测是否正确
        correct_pred = [p == a for p, a in zip(evaluation['predictions'], evaluation['actuals'])]
        
        # 创建数据点颜色列表
        colors = ['green' if c else 'red' for c in correct_pred]
        
        # 绘制散点图
        for i, (date, color) in enumerate(zip(evaluation['dates'], colors)):
            ax1.scatter(date, 1, color=color, s=50, marker='o')  # 预测值
            ax1.scatter(date, 0, color='blue', s=30, marker='x')  # 实际值
            
            # 在正确的预测点之间绘制连接线
            if correct_pred[i]:
                ax1.plot([date, date], [0, 1], 'g-', alpha=0.3)
            else:
                ax1.plot([date, date], [0, 1], 'r-', alpha=0.3)
                
        # 设置Y轴标签
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['实际', '预测'])
        
        # 设置标题和标签
        ax1.set_title(f'{model_name} 的历史预测准确性 ({stock_code})', fontsize=12, fontweight='bold')
        ax1.set_ylabel('预测 vs 实际', fontsize=10)
        
        # 添加网格
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # 第二个子图：预测置信度
        ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
        
        # 使用颜色表示正确/错误的预测
        for i, (date, conf, color) in enumerate(zip(evaluation['dates'], evaluation['confidences'], colors)):
            ax2.bar(date, conf, color=color, width=5, alpha=0.7)
        
        # 添加置信度阈值线
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
        
        # 设置标题和标签
        ax2.set_title('预测置信度', fontsize=12, fontweight='bold')
        ax2.set_ylabel('置信度', fontsize=10)
        ax2.set_ylim(0, 1.0)
        
        # 添加网格
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # 第三个子图：累积性能
        ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
        
        # 计算累积正确预测数量
        cum_correct = np.cumsum(correct_pred)
        total_predictions = np.arange(1, len(correct_pred) + 1)
        cum_accuracy = cum_correct / total_predictions
        
        # 绘制累积准确率
        ax3.plot(evaluation['dates'], cum_accuracy, 'b-', linewidth=2, label='累积准确率')
        
        # 添加基准线
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='随机猜测')
        
        # 设置标题和标签
        ax3.set_title('累积准确率', fontsize=12, fontweight='bold')
        ax3.set_ylabel('准确率', fontsize=10)
        ax3.set_ylim(0, 1.0)
        ax3.legend()
        
        # 添加网格
        ax3.grid(True, linestyle='--', alpha=0.3)
        
        # 格式化X轴日期
        fig.autofmt_xdate()
        
        # 创建图形窗口
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, chart_frame)
        toolbar.update()
        
    def _create_price_chart(self, parent, data, stock_code):
        """创建价格走势图"""
        # 创建图表框架
        chart_frame = ttk.Frame(parent)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 设置背景色和网格
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 绘制收盘价
        ax.plot(data.index, data['close'], 'b-', linewidth=2, label='收盘价')
        
        # 绘制均线
        if 'ma20' in data.columns:
            ax.plot(data.index, data['ma20'], 'r-', linewidth=1, label='MA20')
        if 'ma60' in data.columns:
            ax.plot(data.index, data['ma60'], 'g-', linewidth=1, label='MA60')
            
        # 设置标题和标签
        ax.set_title(f'{stock_code} 价格走势', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('价格', fontsize=12)
        
        # 添加图例
        ax.legend()
        
        # 格式化X轴日期
        fig.autofmt_xdate()
        
        # 创建图形窗口
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, chart_frame)
        toolbar.update()

def main():
    """主函数"""
    root = tk.Tk()
    app = MLPredictionGUI(root)
    root.mainloop()
    
if __name__ == "__main__":
    main() 