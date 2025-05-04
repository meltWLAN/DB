"""
市场概览仪表盘UI组件
提供直观的市场数据可视化展示
"""
import os
import sys
import logging
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from datetime import datetime

# 设置日志
logger = logging.getLogger(__name__)

# 尝试导入市场概览模块
try:
    from src.enhanced.market.enhanced_market_overview import EnhancedMarketOverview
except ImportError:
    logger.error("无法导入EnhancedMarketOverview，市场仪表盘将无法正常工作")


class MarketDashboard:
    """市场仪表盘UI组件类"""
    
    def __init__(self, parent, width=1000, height=800):
        """初始化市场仪表盘
        
        Args:
            parent: 父级UI组件
            width: 宽度
            height: 高度
        """
        self.parent = parent
        self.width = width
        self.height = height
        
        # 初始化市场概览模块
        try:
            self.market_overview = EnhancedMarketOverview()
        except Exception as e:
            logger.error(f"初始化市场概览模块失败: {str(e)}")
            self.market_overview = None
            
        # 初始化UI
        self.init_ui()
        
    def init_ui(self):
        """初始化UI界面"""
        # 创建主框架
        self.main_frame = ttk.Frame(self.parent, width=self.width, height=self.height)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建状态栏
        self.status_var = tk.StringVar(value="就绪")
        self.status_bar = ttk.Label(self.parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 创建顶部控制区域
        self.create_top_controls()
        
        # 创建市场概览组件
        self.create_market_overview()
        
    def create_top_controls(self):
        """创建顶部控制区域"""
        # 创建顶部工具栏
        top_frame = ttk.Frame(self.main_frame)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 刷新按钮
        refresh_btn = ttk.Button(top_frame, text="刷新数据", command=self.refresh_data)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # 日期选择
        ttk.Label(top_frame, text="日期:").pack(side=tk.LEFT, padx=5)
        self.date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        date_entry = ttk.Entry(top_frame, textvariable=self.date_var, width=12)
        date_entry.pack(side=tk.LEFT, padx=5)
        
        # 视图选择
        ttk.Label(top_frame, text="视图:").pack(side=tk.LEFT, padx=5)
        self.view_var = tk.StringVar(value="综合")
        views = ["综合", "指数", "行业", "资金", "情绪"]
        view_combo = ttk.Combobox(top_frame, textvariable=self.view_var, values=views, width=10)
        view_combo.pack(side=tk.LEFT, padx=5)
        view_combo.bind("<<ComboboxSelected>>", self.change_view)
        
        # 导出按钮
        export_btn = ttk.Button(top_frame, text="导出报告", command=self.export_report)
        export_btn.pack(side=tk.RIGHT, padx=5)
        
    def create_market_overview(self):
        """创建市场概览组件"""
        # 创建主内容框架（使用Notebook组件实现多页面）
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建综合视图
        self.overview_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_tab, text="综合视图")
        
        # 创建指数视图
        self.indices_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.indices_tab, text="指数视图")
        
        # 创建行业视图
        self.industry_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.industry_tab, text="行业视图")
        
        # 创建资金视图
        self.money_flow_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.money_flow_tab, text="资金视图")
        
        # 创建市场情绪视图
        self.sentiment_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.sentiment_tab, text="情绪视图")
        
        # 设置综合视图内容
        self.setup_overview_tab()
        
        # 设置指数视图内容
        self.setup_indices_tab()
        
        # 设置行业视图内容
        self.setup_industry_tab()
        
        # 设置资金视图内容
        self.setup_money_flow_tab()
        
        # 设置市场情绪视图内容
        self.setup_sentiment_tab()
        
    def setup_overview_tab(self):
        """设置综合视图内容"""
        # 创建左右分割窗口
        paned = ttk.PanedWindow(self.overview_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧指数概览
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # 指数概览标题
        ttk.Label(left_frame, text="指数概览", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        
        # 指数列表
        columns = ("指数", "最新价", "涨跌幅", "趋势", "评分")
        self.index_tree = ttk.Treeview(left_frame, columns=columns, show="headings", height=10)
        
        # 设置列属性
        for col in columns:
            self.index_tree.heading(col, text=col)
            
        self.index_tree.column("指数", width=100)
        self.index_tree.column("最新价", width=80)
        self.index_tree.column("涨跌幅", width=80)
        self.index_tree.column("趋势", width=80)
        self.index_tree.column("评分", width=60)
        
        # 添加滚动条
        index_scroll = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.index_tree.yview)
        self.index_tree.configure(yscrollcommand=index_scroll.set)
        
        # 布局
        self.index_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        index_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # 右侧市场热点和统计
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # 市场统计框架
        stats_frame = ttk.LabelFrame(right_frame, text="市场统计")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 市场统计
        self.stats_vars = {
            "总成交额": tk.StringVar(value="--"),
            "上涨家数": tk.StringVar(value="--"),
            "下跌家数": tk.StringVar(value="--"),
            "涨停家数": tk.StringVar(value="--"),
            "跌停家数": tk.StringVar(value="--"),
            "市场状态": tk.StringVar(value="--")
        }
        
        row = 0
        for label, var in self.stats_vars.items():
            ttk.Label(stats_frame, text=f"{label}:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Label(stats_frame, textvariable=var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            row += 1
            
        # 热门板块框架
        hot_frame = ttk.LabelFrame(right_frame, text="热门板块")
        hot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 热门板块列表
        hot_columns = ("板块", "涨跌幅", "强度", "领涨股")
        self.hot_tree = ttk.Treeview(hot_frame, columns=hot_columns, show="headings", height=8)
        
        # 设置列属性
        for col in hot_columns:
            self.hot_tree.heading(col, text=col)
            
        self.hot_tree.column("板块", width=100)
        self.hot_tree.column("涨跌幅", width=80)
        self.hot_tree.column("强度", width=60)
        self.hot_tree.column("领涨股", width=100)
        
        # 添加滚动条
        hot_scroll = ttk.Scrollbar(hot_frame, orient=tk.VERTICAL, command=self.hot_tree.yview)
        self.hot_tree.configure(yscrollcommand=hot_scroll.set)
        
        # 布局
        self.hot_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        hot_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
    def setup_indices_tab(self):
        """设置指数视图内容"""
        # 此处添加指数详细视图的实现
        # 创建指数详情框架
        indices_frame = ttk.Frame(self.indices_tab)
        indices_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 在此处添加指数详情UI
        
    def setup_industry_tab(self):
        """设置行业视图内容"""
        # 此处添加行业视图的实现
        industry_frame = ttk.Frame(self.industry_tab)
        industry_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 在此处添加行业视图UI
        
    def setup_money_flow_tab(self):
        """设置资金视图内容"""
        # 此处添加资金视图的实现
        money_frame = ttk.Frame(self.money_flow_tab)
        money_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 在此处添加资金视图UI
        
    def setup_sentiment_tab(self):
        """设置市场情绪视图内容"""
        # 此处添加市场情绪视图的实现
        sentiment_frame = ttk.Frame(self.sentiment_tab)
        sentiment_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 在此处添加市场情绪视图UI
        
    def refresh_data(self):
        """刷新市场数据"""
        self.status_var.set("正在刷新市场数据...")
        
        # 获取选定日期
        trade_date = self.date_var.get()
        
        try:
            # 检查市场概览模块是否初始化
            if self.market_overview is None:
                self.status_var.set("错误：市场概览模块未初始化")
                return
                
            # 获取市场概览数据
            overview_data = self.market_overview.get_market_overview(trade_date)
            
            if not overview_data:
                self.status_var.set(f"无法获取 {trade_date} 的市场数据")
                return
                
            # 更新综合视图
            self.update_overview_tab(overview_data)
            
            # 根据当前选择的视图更新相应的内容
            current_view = self.view_var.get()
            if current_view == "指数":
                self.update_indices_tab(overview_data)
            elif current_view == "行业":
                self.update_industry_tab(overview_data)
            elif current_view == "资金":
                self.update_money_flow_tab(overview_data)
            elif current_view == "情绪":
                self.update_sentiment_tab(overview_data)
                
            self.status_var.set(f"成功获取 {trade_date} 的市场数据")
            
        except Exception as e:
            logger.error(f"刷新市场数据失败: {str(e)}")
            self.status_var.set(f"刷新市场数据失败: {str(e)}")
    
    def update_overview_tab(self, overview_data: Dict[str, Any]):
        """更新综合视图
        
        Args:
            overview_data: 市场概览数据
        """
        # 清空现有数据
        self.index_tree.delete(*self.index_tree.get_children())
        self.hot_tree.delete(*self.hot_tree.get_children())
        
        # 更新指数列表
        indices_data = overview_data.get('indices', [])
        for i, idx in enumerate(indices_data):
            if i >= 10:  # 只显示前10个指数
                break
                
            # 格式化涨跌幅，添加颜色标记
            change = idx.get('change', 0)
            change_str = f"{change:.2f}%" if change else "--"
            
            # 添加到树形视图
            self.index_tree.insert("", tk.END, values=(
                idx.get('name', ''),
                f"{idx.get('close', 0):.2f}",
                change_str,
                idx.get('trend', '未知'),
                f"{idx.get('score', 0):.1f}"
            ))
            
            # 根据涨跌幅设置行颜色
            if change > 0:
                self.index_tree.item(self.index_tree.get_children()[-1], tags=('up',))
            elif change < 0:
                self.index_tree.item(self.index_tree.get_children()[-1], tags=('down',))
        
        # 配置标签
        self.index_tree.tag_configure('up', foreground='red')
        self.index_tree.tag_configure('down', foreground='green')
        
        # 更新市场统计
        base_data = overview_data.get('market_base', {})
        
        if base_data:
            # 格式化成交额（单位：亿元）
            total_amount = base_data.get('total_amount', 0) / 100000000  # 转换为亿元
            self.stats_vars["总成交额"].set(f"{total_amount:.2f}亿")
            
            # 更新涨跌家数
            self.stats_vars["上涨家数"].set(f"{base_data.get('up_count', 0)}")
            self.stats_vars["下跌家数"].set(f"{base_data.get('down_count', 0)}")
            self.stats_vars["涨停家数"].set(f"{base_data.get('limit_up_count', 0)}")
            self.stats_vars["跌停家数"].set(f"{base_data.get('limit_down_count', 0)}")
            
            # 更新市场状态
            market_sentiment = overview_data.get('market_sentiment', {})
            market_status = market_sentiment.get('status', '中性')
            self.stats_vars["市场状态"].set(market_status)
        
        # 更新热门板块
        industry_data = overview_data.get('industry', [])
        
        # 按强度指数排序
        sorted_industries = sorted(industry_data, key=lambda x: x.get('strength_index', 0), reverse=True)
        
        for i, ind in enumerate(sorted_industries):
            if i >= 8:  # 只显示前8个行业
                break
                
            # 格式化涨跌幅
            change = ind.get('change', 0)
            change_str = f"{change:.2f}%" if change else "--"
            
            # 添加到树形视图
            self.hot_tree.insert("", tk.END, values=(
                ind.get('name', ''),
                change_str,
                f"{ind.get('strength_index', 0):.1f}",
                ind.get('leading_up', {}).get('name', '')
            ))
            
            # 根据涨跌幅设置行颜色
            if change > 0:
                self.hot_tree.item(self.hot_tree.get_children()[-1], tags=('up',))
            elif change < 0:
                self.hot_tree.item(self.hot_tree.get_children()[-1], tags=('down',))
        
        # 配置标签
        self.hot_tree.tag_configure('up', foreground='red')
        self.hot_tree.tag_configure('down', foreground='green')
            
    def update_indices_tab(self, overview_data: Dict[str, Any]):
        """更新指数视图
        
        Args:
            overview_data: 市场概览数据
        """
        # 待实现
        pass
        
    def update_industry_tab(self, overview_data: Dict[str, Any]):
        """更新行业视图
        
        Args:
            overview_data: 市场概览数据
        """
        # 待实现
        pass
        
    def update_money_flow_tab(self, overview_data: Dict[str, Any]):
        """更新资金视图
        
        Args:
            overview_data: 市场概览数据
        """
        # 待实现
        pass
        
    def update_sentiment_tab(self, overview_data: Dict[str, Any]):
        """更新市场情绪视图
        
        Args:
            overview_data: 市场概览数据
        """
        # 待实现
        pass
        
    def change_view(self, event):
        """切换视图
        
        Args:
            event: 事件对象
        """
        view = self.view_var.get()
        
        # 根据选中的视图切换选项卡
        if view == "综合":
            self.notebook.select(0)  # 综合视图
        elif view == "指数":
            self.notebook.select(1)  # 指数视图
        elif view == "行业":
            self.notebook.select(2)  # 行业视图
        elif view == "资金":
            self.notebook.select(3)  # 资金视图
        elif view == "情绪":
            self.notebook.select(4)  # 情绪视图
            
    def export_report(self):
        """导出市场报告"""
        self.status_var.set("正在生成市场报告...")
        
        # 获取选定日期
        trade_date = self.date_var.get()
        
        try:
            # 检查市场概览模块是否初始化
            if self.market_overview is None:
                self.status_var.set("错误：市场概览模块未初始化")
                return
                
            # 生成市场报告
            report = self.market_overview.generate_market_report(trade_date)
            
            if not report:
                self.status_var.set("生成市场报告失败")
                return
                
            # 保存报告到文件
            report_dir = "results/market_reports"
            os.makedirs(report_dir, exist_ok=True)
            
            filename = f"market_report_{trade_date.replace('-', '')}.md"
            report_path = os.path.join(report_dir, filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
                
            self.status_var.set(f"市场报告已保存到 {report_path}")
            
        except Exception as e:
            logger.error(f"导出市场报告失败: {str(e)}")
            self.status_var.set(f"导出市场报告失败: {str(e)}")
            
    def create_plot(self, frame, title, x_data, y_data, color='blue', width=6, height=4, kind='line'):
        """创建图表
        
        Args:
            frame: 父框架
            title: 图表标题
            x_data: X轴数据
            y_data: Y轴数据
            color: 线条颜色
            width: 图表宽度
            height: 图表高度
            kind: 图表类型 (line, bar, etc.)
            
        Returns:
            FigureCanvasTkAgg: 图表画布对象
        """
        fig = Figure(figsize=(width, height), dpi=100)
        ax = fig.add_subplot(111)
        
        # 设置图表标题
        ax.set_title(title)
        
        # 根据图表类型绘制
        if kind == 'line':
            ax.plot(x_data, y_data, color=color)
        elif kind == 'bar':
            ax.bar(x_data, y_data, color=color)
        elif kind == 'pie':
            ax.pie(y_data, labels=x_data, autopct='%1.1f%%')
        
        # 创建画布
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        
        # 添加到框架
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        return canvas 