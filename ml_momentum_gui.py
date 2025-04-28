#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

# 确保自定义模块可以被导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_momentum_model import MLMomentumModel
from momentum_analysis import MomentumAnalyzer
from enhanced_momentum_analysis import EnhancedMomentumAnalyzer

# 设置日志
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logger = logging.getLogger("ml_momentum_gui")
logger.setLevel(logging.INFO)

# 文件处理器
file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, f"ml_momentum_gui_{datetime.now().strftime('%Y%m%d')}.log"),
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class MLMomentumGUI:
    """基于机器学习的动量模型GUI"""
    
    def __init__(self, root):
        """初始化
        
        Args:
            root: tkinter根窗口
        """
        self.root = root
        self.root.title("ML动量分析系统")
        self.root.geometry("1200x800")
        
        # 初始化模型
        self.ml_model = MLMomentumModel(use_enhanced=True)
        self.current_market_state = "neutral"  # 默认市场状态
        self.analyzer = self.ml_model.analyzer
        
        # 当前线程
        self.current_thread = None
        
        # 设置风格
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", foreground="#333333")
        style.configure("TLabel", padding=6, foreground="#333333")
        style.configure("TFrame", background="#f5f5f5")
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建顶部控制栏
        self.create_control_panel()
        
        # 创建状态栏
        self.create_status_bar()
        
        # 创建数据显示区域
        self.create_data_display()
        
        # 创建图表区域
        self.create_chart_area()
        
        # 创建参数配置区域
        self.create_parameter_panel()
        
        # 初始化股票列表
        self.stock_list = pd.DataFrame()
        self.load_stock_list()
        
        # 其他初始化
        self.initialize_market_state()
        
        logger.info("ML动量分析GUI初始化完成")
    
    def load_stock_list(self):
        """加载股票列表"""
        try:
            self.stock_list = self.analyzer.get_stock_list()
            if not self.stock_list.empty:
                logger.info(f"已加载 {len(self.stock_list)} 支股票")
                self.update_status(f"已加载 {len(self.stock_list)} 支股票")
            else:
                logger.warning("股票列表为空")
                self.update_status("警告：股票列表为空")
        except Exception as e:
            logger.error(f"加载股票列表失败: {str(e)}")
            self.update_status(f"错误：加载股票列表失败 - {str(e)}")
    
    def initialize_market_state(self):
        """初始化市场状态"""
        try:
            # 获取指数数据
            index_code = '000001.SH'  # 上证指数
            index_data = self.analyzer.get_stock_daily_data(index_code)
            
            if not index_data.empty:
                # 判断市场状态
                self.current_market_state = self.ml_model.determine_market_state(index_data)
                logger.info(f"当前市场状态: {self.current_market_state}")
                self.update_status(f"当前市场状态: {self.current_market_state}")
                
                # 更新市场状态下拉框
                self.market_state_combobox.set(self.current_market_state)
        except Exception as e:
            logger.error(f"初始化市场状态失败: {str(e)}")
            self.update_status(f"错误：初始化市场状态失败 - {str(e)}")
    
    def create_control_panel(self):
        """创建控制面板"""
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建按钮
        ttk.Button(control_frame, text="训练模型", command=self.train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="分析股票", command=self.analyze_stocks).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="可视化指标权重", command=self.visualize_weights).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="导出结果", command=self.export_results).pack(side=tk.LEFT, padx=5)
        
        # 市场状态选择
        ttk.Label(control_frame, text="市场状态:").pack(side=tk.LEFT, padx=5)
        self.market_state_combobox = ttk.Combobox(control_frame, values=list(self.ml_model.market_models.keys()), width=10)
        self.market_state_combobox.pack(side=tk.LEFT, padx=5)
        self.market_state_combobox.set("neutral")  # 默认为中性
        
        # 样本大小输入
        ttk.Label(control_frame, text="样本大小:").pack(side=tk.LEFT, padx=5)
        self.sample_size_var = tk.StringVar(value="100")
        ttk.Entry(control_frame, textvariable=self.sample_size_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # 最小分数输入
        ttk.Label(control_frame, text="最低得分:").pack(side=tk.LEFT, padx=5)
        self.min_score_var = tk.StringVar(value="60")
        ttk.Entry(control_frame, textvariable=self.min_score_var, width=5).pack(side=tk.LEFT, padx=5)
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_data_display(self):
        """创建数据显示区域"""
        # 创建表格框架
        table_frame = ttk.Frame(self.main_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建表格
        columns = ("code", "name", "industry", "close", "score", "base_score", "rsi", "macd")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        
        # 定义列标题
        self.tree.heading("code", text="代码")
        self.tree.heading("name", text="名称")
        self.tree.heading("industry", text="行业")
        self.tree.heading("close", text="收盘价")
        self.tree.heading("score", text="ML得分")
        self.tree.heading("base_score", text="基础得分")
        self.tree.heading("rsi", text="RSI")
        self.tree.heading("macd", text="MACD")
        
        # 定义列宽度
        self.tree.column("code", width=100)
        self.tree.column("name", width=100)
        self.tree.column("industry", width=120)
        self.tree.column("close", width=80)
        self.tree.column("score", width=80)
        self.tree.column("base_score", width=80)
        self.tree.column("rsi", width=80)
        self.tree.column("macd", width=80)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # 绑定点击事件
        self.tree.bind("<ButtonRelease-1>", self.on_tree_select)
    
    def create_chart_area(self):
        """创建图表区域"""
        self.chart_frame = ttk.Frame(self.main_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 初始化图表
        self.fig = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_parameter_panel(self):
        """创建参数配置面板"""
        param_frame = ttk.LabelFrame(self.main_frame, text="模型训练参数")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 训练数据参数
        train_frame = ttk.Frame(param_frame)
        train_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(train_frame, text="训练样本大小:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.train_sample_var = tk.StringVar(value="50")
        ttk.Entry(train_frame, textvariable=self.train_sample_var, width=8).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(train_frame, text="回溯天数:").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.lookback_days_var = tk.StringVar(value="180")
        ttk.Entry(train_frame, textvariable=self.lookback_days_var, width=8).grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(train_frame, text="预测天数:").grid(row=0, column=4, padx=5, pady=2, sticky=tk.W)
        self.forward_days_var = tk.StringVar(value="20")
        ttk.Entry(train_frame, textvariable=self.forward_days_var, width=8).grid(row=0, column=5, padx=5, pady=2)
        
        # 特定市场状态训练选项
        self.train_market_state_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(train_frame, text="为当前市场状态单独训练模型", 
                       variable=self.train_market_state_var).grid(row=1, column=0, columnspan=4, padx=5, pady=2, sticky=tk.W)
    
    def update_status(self, message):
        """更新状态栏
        
        Args:
            message: 状态消息
        """
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def train_model(self):
        """训练模型"""
        # 获取参数
        try:
            sample_size = int(self.train_sample_var.get())
            lookback_days = int(self.lookback_days_var.get())
            forward_days = int(self.forward_days_var.get())
            train_for_market = self.train_market_state_var.get()
            market_state = self.market_state_combobox.get() if train_for_market else None
            
            # 检查股票列表
            if self.stock_list.empty:
                messagebox.showwarning("警告", "股票列表为空，无法训练模型")
                return
            
            # 启动训练线程
            self.update_status("正在收集训练数据...")
            
            def train_thread():
                try:
                    # 收集训练数据
                    X, y = self.ml_model.collect_training_data(
                        self.stock_list, 
                        lookback_days=lookback_days,
                        forward_days=forward_days,
                        sample_size=sample_size
                    )
                    
                    if len(X) > 0:
                        self.root.after(0, lambda: self.update_status(f"正在训练模型 (数据样本: {len(X)})..."))
                        
                        # 训练模型
                        model = self.ml_model.train_model(X, y, market_state=market_state)
                        
                        if model:
                            message = f"模型训练完成 ({len(X)} 个样本)"
                            if market_state:
                                message += f" - 市场状态: {market_state}"
                            
                            self.root.after(0, lambda: self.update_status(message))
                            self.root.after(0, lambda: messagebox.showinfo("成功", message))
                            
                            # 可视化特征重要性
                            title = f"Feature Importance - {market_state}" if market_state else "Feature Importance"
                            self.ml_model.visualize_feature_importance(model, title=title)
                        else:
                            error_msg = "模型训练失败"
                            self.root.after(0, lambda: self.update_status(error_msg))
                            self.root.after(0, lambda: messagebox.showerror("错误", error_msg))
                    else:
                        error_msg = "无法收集足够的训练数据"
                        self.root.after(0, lambda: self.update_status(error_msg))
                        self.root.after(0, lambda: messagebox.showwarning("警告", error_msg))
                        
                except Exception as e:
                    error_msg = f"训练过程出错: {str(e)}"
                    logger.error(error_msg)
                    self.root.after(0, lambda: self.update_status(error_msg))
                    self.root.after(0, lambda: messagebox.showerror("错误", error_msg))
            
            self.current_thread = Thread(target=train_thread)
            self.current_thread.daemon = True
            self.current_thread.start()
            
        except ValueError:
            messagebox.showerror("错误", "参数格式错误，请输入有效的数字")
    
    def analyze_stocks(self):
        """分析股票"""
        # 获取参数
        try:
            sample_size = int(self.sample_size_var.get())
            min_score = int(self.min_score_var.get())
            market_state = self.market_state_combobox.get()
            
            # 检查股票列表
            if self.stock_list.empty:
                messagebox.showwarning("警告", "股票列表为空，无法进行分析")
                return
            
            # 清空表格
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # 清空图表
            self.fig.clear()
            self.canvas.draw()
            
            # 启动分析线程
            self.results = []
            
            def update_gui(message, progress):
                self.update_status(f"{message} ({progress:.1f}%)")
            
            def analyze_thread():
                try:
                    # 分析股票
                    self.results = self.ml_model.analyze_stocks_ml(
                        self.stock_list,
                        market_state=market_state,
                        sample_size=sample_size,
                        min_score=min_score,
                        gui_callback=lambda msg, prog: self.root.after(0, lambda: update_gui(msg, prog))
                    )
                    
                    # 更新UI
                    self.root.after(0, self.update_results_table)
                    
                except Exception as e:
                    error_msg = f"分析过程出错: {str(e)}"
                    logger.error(error_msg)
                    self.root.after(0, lambda: self.update_status(error_msg))
                    self.root.after(0, lambda: messagebox.showerror("错误", error_msg))
            
            self.current_thread = Thread(target=analyze_thread)
            self.current_thread.daemon = True
            self.current_thread.start()
            
        except ValueError:
            messagebox.showerror("错误", "参数格式错误，请输入有效的数字")
    
    def update_results_table(self):
        """更新结果表格"""
        # 清空表格
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 添加结果
        for idx, result in enumerate(self.results):
            self.tree.insert(
                "", tk.END, 
                values=(
                    result['ts_code'],
                    result['name'],
                    result.get('industry', ''),
                    f"{result['close']:.2f}",
                    f"{result['score']:.1f}",
                    f"{result.get('base_score', 0):.1f}",
                    f"{result['rsi']:.1f}",
                    f"{result['macd']:.2f}"
                )
            )
        
        self.update_status(f"分析完成，发现 {len(self.results)} 支符合条件的股票")
    
    def on_tree_select(self, event):
        """处理表格选择事件
        
        Args:
            event: 事件对象
        """
        # 获取选中项
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        ts_code = item['values'][0]
        
        # 查找对应的结果数据
        selected_stock = None
        for result in self.results:
            if result['ts_code'] == ts_code:
                selected_stock = result
                break
        
        if selected_stock:
            self.plot_stock_chart(selected_stock)
    
    def plot_stock_chart(self, stock_result):
        """绘制股票图表
        
        Args:
            stock_result: 股票分析结果
        """
        try:
            # 获取数据
            data = stock_result['data']
            score_details = stock_result['score_details']
            
            # 清空图表
            self.fig.clear()
            
            # 创建子图
            gs = self.fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
            ax1 = self.fig.add_subplot(gs[0, 0])  # K线图
            ax2 = self.fig.add_subplot(gs[1, 0])  # MACD
            ax3 = self.fig.add_subplot(gs[2, 0])  # RSI
            
            # 设置标题
            title = f"{stock_result['name']}({stock_result['ts_code']}) - ML得分: {stock_result['score']:.1f}"
            ax1.set_title(title, fontsize=12)
            
            # 获取最近的交易日期
            dates = data.index[-60:]
            
            # 绘制K线图
            ax1.plot(dates, data['close'][-60:], label='收盘价', color='blue')
            ax1.plot(dates, data['ma20'][-60:], label='MA20', color='red', linestyle='--')
            ax1.plot(dates, data['ma60'][-60:], label='MA60', color='green', linestyle='-.')
            
            # 添加成交量
            volume_data = data['volume'][-60:]
            volume_norm = volume_data / volume_data.max() * data['close'][-60:].min() * 0.3
            ax1.bar(dates, volume_norm, color='gray', alpha=0.3, label='成交量')
            
            # 设置x轴格式
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            
            # 添加图例
            ax1.legend(loc='upper left')
            
            # 绘制MACD
            ax2.plot(dates, data['macd'][-60:], label='MACD', color='blue')
            ax2.plot(dates, data['macd_signal'][-60:], label='Signal', color='red')
            ax2.bar(dates, data['macd_hist'][-60:], label='Hist', color='green', alpha=0.5)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax2.legend(loc='upper left')
            ax2.set_title('MACD', fontsize=10)
            
            # 绘制RSI
            ax3.plot(dates, data['rsi'][-60:], label='RSI', color='purple')
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.3)
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.3)
            ax3.set_ylim(0, 100)
            ax3.legend(loc='upper left')
            ax3.set_title('RSI', fontsize=10)
            
            # 显示得分明细
            score_text = "\n".join([
                f"{k}: {v:.1f}" for k, v in score_details.items() 
                if k not in ['ml_total', 'enhanced_total']
            ])
            
            if 'enhanced_total' in score_details:
                total_label = 'enhanced_total'
            elif 'ml_total' in score_details:
                total_label = 'ml_total'
            else:
                total_label = None
                
            if total_label:
                score_text += f"\n\n总分: {score_details[total_label]:.1f}"
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.02, 0.05, score_text, transform=ax1.transAxes, fontsize=8,
                     verticalalignment='bottom', bbox=props)
            
            # 调整布局
            self.fig.tight_layout()
            
            # 更新画布
            self.canvas.draw()
            
        except Exception as e:
            logger.error(f"绘制图表失败: {str(e)}")
            self.update_status(f"错误：绘制图表失败 - {str(e)}")
    
    def visualize_weights(self):
        """可视化指标权重"""
        # 获取当前市场状态
        market_state = self.market_state_combobox.get()
        
        # 获取权重
        weights = self.ml_model.get_optimal_weights(market_state)
        
        if weights is None:
            messagebox.showwarning("警告", f"没有找到{market_state}市场的模型，请先训练模型")
            return
        
        # 创建新窗口
        weights_window = tk.Toplevel(self.root)
        weights_window.title(f"指标权重 - {market_state}市场")
        weights_window.geometry("600x400")
        
        # 创建图表
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        
        # 排序权重
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_weights]
        values = [item[1] for item in sorted_weights]
        
        # 绘制条形图
        bars = ax.barh(labels, values, color='skyblue')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f"{width:.3f}", ha='left', va='center')
        
        ax.set_xlabel('权重值')
        ax.set_title(f"{market_state}市场指标权重")
        
        # 添加到窗口
        canvas = FigureCanvasTkAgg(fig, master=weights_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加关闭按钮
        ttk.Button(weights_window, text="关闭", command=weights_window.destroy).pack(pady=10)
    
    def export_results(self):
        """导出分析结果"""
        if not self.results:
            messagebox.showwarning("警告", "没有可导出的结果")
            return
        
        try:
            # 创建导出数据框
            export_data = []
            for result in self.results:
                # 过滤掉数据列
                filtered_result = {k: v for k, v in result.items() if k != 'data' and k != 'score_details'}
                export_data.append(filtered_result)
            
            export_df = pd.DataFrame(export_data)
            
            # 选择保存路径
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx")],
                title="保存分析结果"
            )
            
            if not file_path:
                return
            
            # 保存文件
            if file_path.endswith('.csv'):
                export_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            elif file_path.endswith('.xlsx'):
                export_df.to_excel(file_path, index=False)
            
            self.update_status(f"结果已导出到: {file_path}")
            messagebox.showinfo("成功", f"结果已导出到: {file_path}")
            
        except Exception as e:
            logger.error(f"导出结果失败: {str(e)}")
            self.update_status(f"错误：导出结果失败 - {str(e)}")
            messagebox.showerror("错误", f"导出结果失败: {str(e)}")


if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    app = MLMomentumGUI(root)
    root.mainloop() 