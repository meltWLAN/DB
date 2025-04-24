"""
股票分析系统GUI界面 - 行业分类版
整合动量分析与均线交叉策略的图形界面，支持按东方财富行业分类选择
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

# 尝试导入akshare库，用于获取东方财富行业分类
try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False
    logging.warning("未安装akshare库，将无法使用东方财富行业分类功能")

# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 导入分析模块
from momentum_analysis import MomentumAnalyzer
from ma_cross_strategy import MACrossStrategy

# 导入项目配置
try:
    from src.enhanced.config.settings import LOG_DIR, DATA_DIR, RESULTS_DIR
except ImportError:
    # 设置默认配置
    LOG_DIR = "./logs"
    DATA_DIR = "./data"
    RESULTS_DIR = "./results"

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志
log_file = os.path.join(LOG_DIR, f"industry_stock_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class IndustryStockAnalysisGUI:
    """行业股票分析系统GUI界面类"""
    
    def __init__(self, root):
        """初始化GUI界面"""
        self.root = root
        self.root.title("行业股票分析系统")
        self.root.geometry("1000x700")
        
        # 初始化分析器
        self.momentum_analyzer = MomentumAnalyzer(use_tushare=True)
        self.ma_strategy = MACrossStrategy(use_tushare=True)
        self.stock_list = None
        self.momentum_results = None
        self.ma_results = None
        self.industry_list = ["全部"] # 初始化行业列表
        
        # 加载行业列表
        self.load_industry_list()
        
        # 初始化UI
        self.setup_ui()
        
    def load_industry_list(self):
        """加载东方财富行业分类数据"""
        try:
            if not HAS_AKSHARE:
                logger.warning("未安装AKShare库，无法获取东方财富行业分类数据")
                return
                
            # 获取东方财富行业板块数据
            industry_df = ak.stock_board_industry_name_em()
            
            if industry_df is not None and not industry_df.empty:
                # 提取行业名称
                self.industry_list = ["全部"] + industry_df["板块名称"].tolist()
                logger.info(f"成功加载 {len(self.industry_list)-1} 个行业分类")
            else:
                logger.warning("获取行业列表为空")
        except Exception as e:
            logger.error(f"加载行业列表失败: {str(e)}")
            
    def get_stocks_by_industry(self, industry):
        """根据行业获取股票列表"""
        try:
            if industry == "全部":
                return self.stock_list
                
            if not HAS_AKSHARE:
                logger.warning("未安装AKShare库，无法获取行业股票数据")
                return self.stock_list
                
            # 获取该行业的股票列表
            industry_stocks = ak.stock_board_industry_cons_em(symbol=industry)
            
            if industry_stocks is not None and not industry_stocks.empty:
                # 转换代码格式以匹配
                industry_stocks["ts_code"] = industry_stocks["代码"].apply(
                    lambda x: f"{x}.SH" if x.startswith(("6", "9")) else f"{x}.SZ"
                )
                
                # 只保留需要的列
                industry_stocks = industry_stocks[["ts_code", "名称"]].rename(
                    columns={"名称": "name"}
                )
                
                # 添加行业信息
                industry_stocks["industry"] = industry
                
                logger.info(f"获取到 {len(industry_stocks)} 支{industry}行业的股票")
                return industry_stocks
            else:
                logger.warning(f"获取{industry}行业股票列表为空")
                return pd.DataFrame(columns=["ts_code", "name", "industry"])
                
        except Exception as e:
            logger.error(f"获取{industry}行业股票失败: {str(e)}")
            return self.stock_list
            
    def setup_ui(self):
        """设置UI界面"""
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
        
        # 创建状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
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
        
        # 行业选择
        ttk.Label(param_frame, text="行业选择:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.industry_var = tk.StringVar(value="全部")
        industry_combo = ttk.Combobox(param_frame, textvariable=self.industry_var, values=self.industry_list, width=15)
        industry_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # 样本大小设置
        ttk.Label(param_frame, text="样本数量:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.sample_size_var = tk.StringVar(value="20")
        ttk.Entry(param_frame, textvariable=self.sample_size_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
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
        columns = ("股票代码", "股票名称", "行业", "当前价格", "20日动量", "RSI", "得分")
        self.momentum_tree = ttk.Treeview(self.momentum_result_frame, columns=columns, show="headings")
        
        # 设置列标题
        for col in columns:
            self.momentum_tree.heading(col, text=col)
            self.momentum_tree.column(col, width=80)
        
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
        
        # 行业选择
        ttk.Label(param_frame, text="行业选择:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.ma_industry_var = tk.StringVar(value="全部")
        ma_industry_combo = ttk.Combobox(param_frame, textvariable=self.ma_industry_var, values=self.industry_list, width=15)
        ma_industry_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # 短期均线参数
        ttk.Label(param_frame, text="短期均线:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.short_ma_var = tk.StringVar(value="5")
        ttk.Entry(param_frame, textvariable=self.short_ma_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # 长期均线参数
        ttk.Label(param_frame, text="长期均线:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.long_ma_var = tk.StringVar(value="20")
        ttk.Entry(param_frame, textvariable=self.long_ma_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # 止损比例
        ttk.Label(param_frame, text="止损比例(%):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.stop_loss_var = tk.StringVar(value="5")
        ttk.Entry(param_frame, textvariable=self.stop_loss_var, width=10).grid(row=3, column=1, padx=5, pady=5)
        
        # 初始资金
        ttk.Label(param_frame, text="初始资金:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.initial_capital_var = tk.StringVar(value="100000")
        ttk.Entry(param_frame, textvariable=self.initial_capital_var, width=10).grid(row=4, column=1, padx=5, pady=5)
        
        # 样本大小
        ttk.Label(param_frame, text="样本数量:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.ma_sample_size_var = tk.StringVar(value="20")
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
        columns = ("股票代码", "股票名称", "行业", "收盘价", "当前信号", "回测收益", "年化收益", "最大回撤")
        self.ma_tree = ttk.Treeview(self.ma_result_frame, columns=columns, show="headings")
        
        # 设置列标题
        for col in columns:
            self.ma_tree.heading(col, text=col)
            self.ma_tree.column(col, width=80)
        
        # 添加滚动条
        ma_scrollbar = ttk.Scrollbar(self.ma_result_frame, orient=tk.VERTICAL, command=self.ma_tree.yview)
        self.ma_tree.configure(yscrollcommand=ma_scrollbar.set)
        
        # 布局
        self.ma_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ma_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 表格双击事件
        self.ma_tree.bind("<Double-1>", self.on_ma_tree_double_click)
        
    def run_momentum_analysis(self):
        """运行动量分析"""
        try:
            # 获取参数
            industry = self.industry_var.get()
            sample_size = int(self.sample_size_var.get())
            min_score = int(self.min_score_var.get())
            
            # 清空当前结果
            for item in self.momentum_tree.get_children():
                self.momentum_tree.delete(item)
                
            # 更新状态
            self.status_bar["text"] = f"正在运行{industry}行业的动量分析..."
            
            # 在新线程中运行分析
            def run_analysis():
                try:
                    # 获取股票列表
                    if self.stock_list is None:
                        self.status_bar["text"] = "正在获取股票列表..."
                        self.stock_list = self.momentum_analyzer.get_stock_list()
                    
                    # 获取行业股票
                    if industry != "全部":
                        industry_stocks = self.get_stocks_by_industry(industry)
                        if industry_stocks is not None and not industry_stocks.empty:
                            analyze_stocks = industry_stocks
                        else:
                            analyze_stocks = self.stock_list
                    else:
                        analyze_stocks = self.stock_list
                        
                    # 限制样本数量
                    if len(analyze_stocks) > sample_size:
                        analyze_stocks = analyze_stocks.head(sample_size)
                        
                    # 运行分析
                    self.status_bar["text"] = f"开始分析 {len(analyze_stocks)} 支{industry}行业股票的动量..."
                    self.momentum_results = self.momentum_analyzer.analyze_stocks(
                        analyze_stocks, min_score=min_score)
                    
                    # 更新结果
                    self.root.after(0, self.update_momentum_results)
                except Exception as e:
                    logger.error(f"分析过程中出错: {str(e)}")
                    self.root.after(0, lambda: self.show_error(f"分析过程中出错: {str(e)}"))
            
            # 启动分析线程
            analysis_thread = threading.Thread(target=run_analysis)
            analysis_thread.daemon = True
            analysis_thread.start()
            
        except Exception as e:
            logger.error(f"动量分析失败: {str(e)}")
            self.show_error(f"动量分析失败: {str(e)}")
            
    def update_momentum_results(self):
        """更新动量分析结果"""
        if self.momentum_results:
            # 更新表格
            for result in self.momentum_results:
                values = (
                    result['ts_code'],
                    result['name'],
                    result.get('industry', ''),
                    f"{result['close']:.2f}",
                    f"{result['momentum_20']:.2%}",
                    f"{result['rsi']:.2f}",
                    f"{result['score']}"
                )
                self.momentum_tree.insert("", tk.END, values=values)
                
            self.status_bar["text"] = f"动量分析完成，发现 {len(self.momentum_results)} 支强势股票"
        else:
            self.status_bar["text"] = "动量分析完成，未找到符合条件的股票"
            
    def run_ma_cross_strategy(self):
        """运行均线交叉策略"""
        try:
            # 获取参数
            industry = self.ma_industry_var.get()
            short_ma = int(self.short_ma_var.get())
            long_ma = int(self.long_ma_var.get())
            stop_loss_pct = float(self.stop_loss_var.get())
            initial_capital = float(self.initial_capital_var.get())
            sample_size = int(self.ma_sample_size_var.get())
            
            # 清空当前结果
            for item in self.ma_tree.get_children():
                self.ma_tree.delete(item)
                
            # 更新状态
            self.status_bar["text"] = f"正在运行{industry}行业的均线交叉策略..."
            
            # 在新线程中运行策略
            def run_strategy():
                try:
                    # 获取股票列表
                    if self.stock_list is None:
                        self.status_bar["text"] = "正在获取股票列表..."
                        self.stock_list = self.ma_strategy.get_stock_list()
                    
                    # 获取行业股票
                    if industry != "全部":
                        industry_stocks = self.get_stocks_by_industry(industry)
                        if industry_stocks is not None and not industry_stocks.empty:
                            analyze_stocks = industry_stocks
                        else:
                            analyze_stocks = self.stock_list
                    else:
                        analyze_stocks = self.stock_list
                        
                    # 限制样本数量
                    if len(analyze_stocks) > sample_size:
                        analyze_stocks = analyze_stocks.head(sample_size)
                        
                    # 运行策略
                    self.status_bar["text"] = f"开始分析 {len(analyze_stocks)} 支{industry}行业股票的均线交叉策略..."
                    self.ma_results = self.ma_strategy.run_strategy(
                        analyze_stocks, 
                        short_ma=short_ma,
                        long_ma=long_ma,
                        initial_capital=initial_capital,
                        stop_loss_pct=stop_loss_pct/100
                    )
                    
                    # 更新结果
                    self.root.after(0, self.update_ma_results)
                except Exception as e:
                    logger.error(f"策略执行过程中出错: {str(e)}")
                    self.root.after(0, lambda: self.show_error(f"策略执行过程中出错: {str(e)}"))
            
            # 启动策略线程
            strategy_thread = threading.Thread(target=run_strategy)
            strategy_thread.daemon = True
            strategy_thread.start()
            
        except Exception as e:
            logger.error(f"均线交叉策略失败: {str(e)}")
            self.show_error(f"均线交叉策略失败: {str(e)}")
            
    def update_ma_results(self):
        """更新均线交叉策略结果"""
        if self.ma_results:
            # 更新表格
            for result in self.ma_results:
                values = (
                    result['ts_code'],
                    result['name'],
                    result.get('industry', ''),
                    f"{result['close']:.2f}",
                    result['current_signal'],
                    f"{result['total_return']:.2%}",
                    f"{result['annual_return']:.2%}",
                    f"{result['max_drawdown']:.2%}"
                )
                self.ma_tree.insert("", tk.END, values=values)
                
            self.status_bar["text"] = f"均线交叉策略分析完成，分析了 {len(self.ma_results)} 支股票"
        else:
            self.status_bar["text"] = "均线交叉策略分析完成，未得到有效结果"
            
    def on_momentum_tree_double_click(self, event):
        """双击动量分析结果表格的处理函数"""
        selection = self.momentum_tree.selection()
        if not selection:
            return
            
        item = selection[0]
        stock_code = self.momentum_tree.item(item, "values")[0]
        
        # 显示图表路径，而不是直接打开图表
        chart_path = os.path.join(RESULTS_DIR, "charts", f"{stock_code}_momentum.png")
        if os.path.exists(chart_path):
            messagebox.showinfo("图表文件", f"图表文件已生成：\n{chart_path}")
        else:
            messagebox.showinfo("提示", f"未找到{stock_code}的图表文件")
    
    def on_ma_tree_double_click(self, event):
        """双击均线交叉结果表格的处理函数"""
        selection = self.ma_tree.selection()
        if not selection:
            return
            
        item = selection[0]
        stock_code = self.ma_tree.item(item, "values")[0]
        
        # 显示图表路径，而不是直接打开图表
        chart_path = os.path.join(RESULTS_DIR, "ma_charts", f"{stock_code}_ma_cross.png")
        if os.path.exists(chart_path):
            messagebox.showinfo("图表文件", f"图表文件已生成：\n{chart_path}")
        else:
            messagebox.showinfo("提示", f"未找到{stock_code}的图表文件")
    
    def show_error(self, message):
        """显示错误消息"""
        self.status_bar["text"] = "出错"
        messagebox.showerror("错误", message)

def main():
    """主函数"""
    try:
        # 创建GUI根窗口
        root = tk.Tk()
        app = IndustryStockAnalysisGUI(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"应用程序运行出错: {str(e)}", exc_info=True)
        messagebox.showerror("错误", f"应用程序运行出错: {str(e)}")

if __name__ == "__main__":
    main() 