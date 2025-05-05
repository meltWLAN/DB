#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票推荐系统完整GUI界面
提供全面的股票分析和推荐功能
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from queue import Queue
import traceback
import time

# 确保必要的目录存在
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

from src.enhanced.config.settings import LOG_DIR
from src.enhanced.strategies.stock_picker import StockPicker
from src.enhanced.data.data_manager import EnhancedDataManager
from src.enhanced.data.quality.data_quality_checker import DataQualityChecker

# 配置日志
log_file = os.path.join(LOG_DIR, f"stock_recommender_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class StockRecommenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("智能股票推荐系统")
        self.root.geometry("1200x800")
        
        # 初始化组件
        self.stock_picker = StockPicker()
        self.data_manager = EnhancedDataManager()
        self.quality_checker = DataQualityChecker()
        
        # 创建消息队列
        self.message_queue = Queue()
        
        # 创建主框架
        self.create_main_frame()
        
        # 创建菜单栏
        self.create_menu()
        
        # 创建状态栏
        self.create_status_bar()
        
        # 启动消息处理
        self.process_messages()
        
    def process_messages(self):
        """处理消息队列中的消息"""
        try:
            while not self.message_queue.empty():
                msg_type, msg = self.message_queue.get_nowait()
                if msg_type == "error":
                    messagebox.showerror("错误", msg)
                elif msg_type == "info":
                    messagebox.showinfo("提示", msg)
                elif msg_type == "status":
                    self.status_var.set(msg)
        except Exception as e:
            logger.error(f"处理消息时出错: {str(e)}", exc_info=True)
        finally:
            self.root.after(100, self.process_messages)
            
    def show_message(self, msg_type, msg):
        """显示消息"""
        self.message_queue.put((msg_type, msg))
        
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="导出数据", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 分析菜单
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="分析", menu=analysis_menu)
        analysis_menu.add_command(label="技术分析", command=self.show_technical_analysis)
        analysis_menu.add_command(label="基本面分析", command=self.show_fundamental_analysis)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
        
    def create_main_frame(self):
        """创建主框架"""
        # 创建左右分栏
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧框架
        left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(left_frame, weight=1)
        
        # 右侧框架
        right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(right_frame, weight=2)
        
        # 创建左侧内容
        self.create_left_frame(left_frame)
        
        # 创建右侧内容
        self.create_right_frame(right_frame)
        
    def create_left_frame(self, parent):
        """创建左侧框架内容"""
        # 参数设置区域
        param_frame = ttk.LabelFrame(parent, text="参数设置", padding="5")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 推荐数量设置
        ttk.Label(param_frame, text="推荐数量:").grid(row=0, column=0, padx=5, pady=5)
        self.top_n = tk.StringVar(value="5")
        entry = ttk.Entry(param_frame, textvariable=self.top_n, width=10)
        entry.grid(row=0, column=1, padx=5, pady=5)
        entry.bind('<KeyRelease>', self.validate_top_n)
        
        # 风险偏好设置
        ttk.Label(param_frame, text="风险偏好:").grid(row=1, column=0, padx=5, pady=5)
        self.risk_level = tk.StringVar(value="中等")
        risk_combo = ttk.Combobox(param_frame, textvariable=self.risk_level, values=["保守", "中等", "激进"])
        risk_combo.grid(row=1, column=1, padx=5, pady=5)
        risk_combo.bind('<<ComboboxSelected>>', self.on_risk_level_change)
        
        # 行业筛选
        ttk.Label(param_frame, text="行业筛选:").grid(row=2, column=0, padx=5, pady=5)
        self.industry = tk.StringVar(value="全部")
        industry_combo = ttk.Combobox(param_frame, textvariable=self.industry, values=["全部", "金融", "科技", "医药", "消费"])
        industry_combo.grid(row=2, column=1, padx=5, pady=5)
        industry_combo.bind('<<ComboboxSelected>>', self.on_industry_change)
        
        # 操作按钮
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建按钮并添加工具提示
        get_btn = ttk.Button(button_frame, text="获取推荐", command=self.get_recommendations)
        get_btn.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(get_btn, "获取股票推荐列表")
        
        refresh_btn = ttk.Button(button_frame, text="刷新数据", command=self.refresh_data)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(refresh_btn, "刷新最新市场数据")
        
        clear_btn = ttk.Button(button_frame, text="清空结果", command=self.clear_results)
        clear_btn.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(clear_btn, "清空当前显示的结果")
        
        # 推荐结果表格
        self.create_result_table(parent)
        
    def create_tooltip(self, widget, text):
        """创建工具提示"""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)  # 移除窗口边框
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, background="#ffffe0", relief=tk.SOLID, borderwidth=1)
            label.pack()
            
            def hide_tooltip(event):
                tooltip.destroy()
            
            widget.tooltip = tooltip
            widget.bind('<Leave>', hide_tooltip)
            
        widget.bind('<Enter>', show_tooltip)
        
    def validate_top_n(self, event):
        """验证推荐数量输入"""
        try:
            value = self.top_n.get()
            if not value:
                return
            num = int(value)
            if num <= 0:
                self.show_message("error", "推荐数量必须大于0")
                self.top_n.set("5")
        except ValueError:
            self.show_message("error", "请输入有效的数字")
            self.top_n.set("5")
            
    def on_risk_level_change(self, event):
        """处理风险偏好变化"""
        risk_level = self.risk_level.get()
        logger.info(f"风险偏好已更改为: {risk_level}")
        
    def on_industry_change(self, event):
        """处理行业筛选变化"""
        industry = self.industry.get()
        logger.info(f"行业筛选已更改为: {industry}")
        
    def create_right_frame(self, parent):
        """创建右侧框架内容"""
        # 创建notebook用于切换不同视图
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建各个标签页
        self.create_chart_tab()
        self.create_analysis_tab()
        self.create_details_tab()
        
    def create_result_table(self, parent):
        """创建结果表格"""
        table_frame = ttk.LabelFrame(parent, text="推荐结果", padding="5")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建表格
        columns = ("股票代码", "股票名称", "当前价格", "涨跌幅", "风险分数", "推荐理由")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        
        # 设置列标题和宽度
        column_widths = {
            "股票代码": 80,
            "股票名称": 100,
            "当前价格": 80,
            "涨跌幅": 80,
            "风险分数": 80,
            "推荐理由": 200
        }
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths.get(col, 100))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定选择事件
        self.tree.bind('<<TreeviewSelect>>', self.on_stock_select)
        
        # 添加右键菜单
        self.create_context_menu()
        
    def create_context_menu(self):
        """创建右键菜单"""
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="查看详情", command=self.show_stock_details)
        self.context_menu.add_command(label="导出选中", command=self.export_selected)
        
        # 绑定右键菜单
        self.tree.bind("<Button-3>", self.show_context_menu)
        
    def show_context_menu(self, event):
        """显示右键菜单"""
        try:
            item = self.tree.identify_row(event.y)
            if item:
                self.tree.selection_set(item)
                self.context_menu.post(event.x_root, event.y_root)
        except Exception as e:
            logger.error(f"显示右键菜单时出错: {str(e)}")
            
    def show_stock_details(self):
        """显示选中股票的详细信息"""
        selection = self.tree.selection()
        if selection:
            stock_code = self.tree.item(selection[0])['values'][0]
            self.update_stock_details(stock_code)
            
    def export_selected(self):
        """导出选中的股票数据"""
        try:
            selection = self.tree.selection()
            if not selection:
                self.show_message("warning", "请先选择要导出的股票")
                return
                
            data = []
            for item in selection:
                values = self.tree.item(item)['values']
                data.append({
                    '股票代码': values[0],
                    '股票名称': values[1],
                    '当前价格': values[2],
                    '涨跌幅': values[3],
                    '风险分数': values[4],
                    '推荐理由': values[5]
                })
            
            # 创建DataFrame并导出
            df = pd.DataFrame(data)
            output_file = os.path.join(LOG_DIR, f"selected_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            self.show_message("info", f"选中数据已导出到: {output_file}")
            
        except Exception as e:
            logger.error(f"导出选中数据时出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"导出数据时出错: {str(e)}")
        
    def create_chart_tab(self):
        """创建图表标签页"""
        chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(chart_frame, text="走势图")
        
        # 创建图表区域
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_analysis_tab(self):
        """创建分析标签页"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="分析报告")
        
        # 创建文本区域
        self.analysis_text = tk.Text(analysis_frame, wrap=tk.WORD, height=20)
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_details_tab(self):
        """创建详细信息标签页"""
        details_frame = ttk.Frame(self.notebook)
        self.notebook.add(details_frame, text="详细信息")
        
        # 创建详细信息显示区域
        self.details_text = tk.Text(details_frame, wrap=tk.WORD, height=20)
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_status_bar(self):
        """创建状态栏"""
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def get_recommendations(self):
        """获取股票推荐"""
        try:
            self.status_var.set("正在获取推荐...")
            self.root.update()
            
            # 清空现有结果
            self.clear_results()
            
            # 获取参数
            try:
                top_n = int(self.top_n.get())
                if top_n <= 0:
                    raise ValueError("推荐数量必须大于0")
            except ValueError as e:
                self.show_message("error", f"请输入有效的推荐数量: {str(e)}")
                return
            
            # 在新线程中获取推荐
            thread = threading.Thread(target=self._get_recommendations_thread, args=(top_n,))
            thread.daemon = True  # 设置为守护线程
            thread.start()
            
        except Exception as e:
            logger.error(f"获取推荐时出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"获取推荐时出错: {str(e)}")
            self.status_var.set("获取推荐失败")
            
    def _get_recommendations_thread(self, top_n):
        """在新线程中获取推荐"""
        try:
            # 记录参数信息
            logger.info(f"获取推荐 - 数量: {top_n}")
            
            # 获取推荐
            recommendations = self.stock_picker.get_stock_recommendations(
                top_n=top_n
            )
            
            if not recommendations:
                self.show_message("info", "未找到符合条件的股票")
                return
            
            # 更新UI
            self.root.after(0, self._update_recommendations, recommendations)
            
        except Exception as e:
            logger.error(f"获取推荐线程出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"获取推荐时出错: {str(e)}")
        finally:
            self.root.after(0, lambda: self.status_var.set("就绪"))
            
    def _update_recommendations(self, recommendations):
        """更新推荐结果"""
        try:
            for stock in recommendations:
                self.tree.insert("", tk.END, values=(
                    stock['stock_code'],
                    stock.get('stock_name', ''),
                    f"{stock['current_price']:.2f}",
                    f"{stock['change_pct']*100:.2f}%",
                    f"{stock['risk_score']:.2f}",
                    stock['recommendation_reason']
                ))
            
            self.status_var.set(f"成功获取 {len(recommendations)} 只股票推荐")
            
        except Exception as e:
            logger.error(f"更新推荐结果时出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"更新推荐结果时出错: {str(e)}")
        
    def on_stock_select(self, event):
        """处理股票选择事件"""
        try:
            selection = self.tree.selection()
            if not selection:
                return
                
            # 获取选中的股票代码
            stock_code = self.tree.item(selection[0])['values'][0]
            
            # 更新图表和分析
            self.update_stock_details(stock_code)
            
        except Exception as e:
            logger.error(f"处理股票选择事件时出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"处理股票选择时出错: {str(e)}")
        
    def update_stock_details(self, stock_code):
        """更新股票详细信息"""
        try:
            # 获取股票数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # 在新线程中获取数据
            thread = threading.Thread(
                target=self._update_stock_details_thread,
                args=(stock_code, start_date, end_date)
            )
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            logger.error(f"更新股票详情时出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"更新股票详情时出错: {str(e)}")
            
    def _update_stock_details_thread(self, stock_code, start_date, end_date):
        """在新线程中更新股票详情"""
        try:
            # 添加重试机制
            max_retries = 3
            retry_count = 0
            df = None
            
            while retry_count < max_retries:
                try:
                    df = self.data_manager.get_stock_data(
                        stock_code,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    if df is not None and not df.empty:
                        break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
                    logger.warning(f"获取股票数据失败，尝试重试 ({retry_count}/{max_retries}): {str(e)}")
                    time.sleep(1)  # 等待1秒后重试
            
            if df is not None and not df.empty:
                # 更新UI
                self.root.after(0, self._update_details_ui, stock_code, df)
            else:
                self.show_message("error", f"无法获取股票 {stock_code} 的数据")
                
        except Exception as e:
            logger.error(f"更新股票详情线程出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"更新股票详情时出错: {str(e)}")
            
    def _update_details_ui(self, stock_code, df):
        """更新UI显示"""
        try:
            # 更新图表
            self.update_chart(df)
            
            # 更新分析报告
            self.update_analysis(stock_code, df)
            
            # 更新详细信息
            self.update_details(stock_code, df)
            
        except Exception as e:
            logger.error(f"更新UI显示时出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"更新显示时出错: {str(e)}")
            
    def update_chart(self, df):
        """更新图表"""
        try:
            self.ax.clear()
            
            # 确保数据格式正确
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # 绘制收盘价
            self.ax.plot(df.index, df['close'], label='收盘价')
            
            # 添加均线
            ma5 = df['close'].rolling(window=5).mean()
            ma20 = df['close'].rolling(window=20).mean()
            self.ax.plot(df.index, ma5, label='5日均线', alpha=0.7)
            self.ax.plot(df.index, ma20, label='20日均线', alpha=0.7)
            
            # 设置图表属性
            self.ax.set_title("股价走势")
            self.ax.set_xlabel("日期")
            self.ax.set_ylabel("价格")
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
            # 调整x轴标签
            plt.xticks(rotation=45)
            self.fig.tight_layout()
            
            # 更新画布
            self.canvas.draw()
            
        except Exception as e:
            logger.error(f"更新图表时出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"更新图表时出错: {str(e)}")
        
    def update_analysis(self, stock_code, df):
        """更新分析报告"""
        try:
            self.analysis_text.delete(1.0, tk.END)
            
            # 计算技术指标
            ma5 = df['close'].rolling(window=5).mean()
            ma20 = df['close'].rolling(window=20).mean()
            
            # 计算其他技术指标
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # 计算RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 计算MACD
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            # 生成分析报告
            report = f"股票代码: {stock_code}\n\n"
            report += "技术分析报告:\n"
            report += f"当前价格: {close.iloc[-1]:.2f}\n"
            report += f"5日均线: {ma5.iloc[-1]:.2f}\n"
            report += f"20日均线: {ma20.iloc[-1]:.2f}\n"
            report += f"RSI(14): {rsi.iloc[-1]:.2f}\n"
            report += f"MACD: {macd.iloc[-1]:.2f}\n"
            report += f"MACD信号线: {signal.iloc[-1]:.2f}\n"
            report += f"成交量: {volume.iloc[-1]:,.0f}\n"
            
            # 添加趋势判断
            if close.iloc[-1] > ma5.iloc[-1] > ma20.iloc[-1]:
                report += "\n趋势判断: 上升趋势\n"
            elif close.iloc[-1] < ma5.iloc[-1] < ma20.iloc[-1]:
                report += "\n趋势判断: 下降趋势\n"
            else:
                report += "\n趋势判断: 震荡整理\n"
            
            self.analysis_text.insert(tk.END, report)
            
        except Exception as e:
            logger.error(f"更新分析报告时出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"更新分析报告时出错: {str(e)}")
        
    def update_details(self, stock_code, df):
        """更新详细信息"""
        try:
            self.details_text.delete(1.0, tk.END)
            
            # 生成详细信息
            details = f"股票代码: {stock_code}\n\n"
            details += "交易数据:\n"
            details += df.describe().to_string()
            
            self.details_text.insert(tk.END, details)
            
        except Exception as e:
            logger.error(f"更新详细信息时出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"更新详细信息时出错: {str(e)}")
        
    def refresh_data(self):
        """刷新数据"""
        try:
            self.status_var.set("正在刷新数据...")
            self.root.update()
            
            # 在新线程中刷新数据
            thread = threading.Thread(target=self._refresh_data_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            logger.error(f"刷新数据时出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"刷新数据时出错: {str(e)}")
            self.status_var.set("刷新数据失败")
            
    def _refresh_data_thread(self):
        """在新线程中刷新数据"""
        try:
            # 刷新数据
            self.data_manager.refresh_data()
            
            # 更新UI
            self.root.after(0, lambda: self.status_var.set("数据刷新完成"))
            
        except Exception as e:
            logger.error(f"刷新数据线程出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"刷新数据时出错: {str(e)}")
            
    def clear_results(self):
        """清空结果"""
        try:
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.analysis_text.delete(1.0, tk.END)
            self.details_text.delete(1.0, tk.END)
            self.ax.clear()
            self.canvas.draw()
            self.status_var.set("就绪")
            
        except Exception as e:
            logger.error(f"清空结果时出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"清空结果时出错: {str(e)}")
        
    def export_data(self):
        """导出数据"""
        try:
            # 获取当前推荐结果
            data = []
            for item in self.tree.get_children():
                values = self.tree.item(item)['values']
                data.append({
                    '股票代码': values[0],
                    '股票名称': values[1],
                    '当前价格': values[2],
                    '涨跌幅': values[3],
                    '风险分数': values[4],
                    '推荐理由': values[5]
                })
            
            if not data:
                self.show_message("warning", "没有可导出的数据")
                return
                
            # 创建DataFrame并导出
            df = pd.DataFrame(data)
            output_file = os.path.join(LOG_DIR, f"stock_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            self.show_message("info", f"数据已导出到: {output_file}")
            
        except Exception as e:
            logger.error(f"导出数据时出错: {str(e)}\n{traceback.format_exc()}")
            self.show_message("error", f"导出数据时出错: {str(e)}")
            
    def show_technical_analysis(self):
        """显示技术分析"""
        self.show_message("info", "技术分析功能正在开发中...")
        
    def show_fundamental_analysis(self):
        """显示基本面分析"""
        self.show_message("info", "基本面分析功能正在开发中...")
        
    def show_help(self):
        """显示帮助信息"""
        help_text = """
使用说明:
1. 在参数设置区域设置推荐参数
2. 点击"获取推荐"按钮获取股票推荐
3. 点击推荐结果中的股票查看详细信息
4. 使用"刷新数据"按钮更新最新数据
5. 使用"导出数据"保存推荐结果
        """
        self.show_message("info", help_text)
        
    def show_about(self):
        """显示关于信息"""
        about_text = """
智能股票推荐系统 v1.0
基于数据增强的股票分析和推荐系统
        """
        self.show_message("info", about_text)

def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = StockRecommenderGUI(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"GUI程序运行出错: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main() 