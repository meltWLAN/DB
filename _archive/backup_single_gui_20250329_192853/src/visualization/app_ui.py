"""
应用程序主界面示例
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random
from datetime import datetime, timedelta
import sys

from .ui_layout import (
    ThemeManager, 
    UIFactory, 
    ResponsiveGridLayout, 
    ResponsiveFrame, 
    Notification,
    StatusBar, 
    DialogFactory
)

class StockAppUI:
    """股票推荐系统UI"""
    
    def __init__(self, root):
        """
        初始化UI
        
        Args:
            root: Tkinter根窗口
        """
        self.root = root
        self.root.title("股票推荐系统")
        self.root.geometry("1280x800")
        
        # 创建主题管理器 (默认主题)
        self.theme_manager = ThemeManager("default")
        
        # 应用主题到窗口部件
        self.theme_manager.apply_theme_to_widgets(root)
        
        # 创建UI工厂
        self.ui_factory = UIFactory(self.theme_manager)
        
        # 创建通知管理器
        self.notification = Notification(root, self.theme_manager)
        
        # 创建对话框工厂
        self.dialog_factory = DialogFactory(root, self.theme_manager)
        
        # 创建主布局
        self.create_layout()
        
        # 初始化示例数据
        self.initialize_sample_data()
        
        # 填充UI
        self.populate_ui()
        
        # 创建状态栏
        self.status_bar = StatusBar(root, self.theme_manager)
        self.status_bar.pack(side="bottom", fill="x")
        self.status_bar.set_status("就绪")
        
    def create_layout(self):
        """创建应用布局"""
        # 创建菜单栏
        self.create_menu()
        
        # 创建主框架
        self.main_frame = self.ui_factory.create_frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 配置响应式网格
        self.layout = ResponsiveGridLayout(self.main_frame, columns=12)
        
        # 创建侧边栏
        self.sidebar_frame = self.ui_factory.create_frame(self.main_frame)
        self.layout.place_widget(self.sidebar_frame, row=0, column=0, columnspan=3, rowspan=1, sticky="nsew")
        
        # 创建内容区
        self.content_frame = self.ui_factory.create_frame(self.main_frame)
        self.layout.place_widget(self.content_frame, row=0, column=3, columnspan=9, rowspan=1, sticky="nsew")
        
        # 配置内容区网格
        self.content_layout = ResponsiveGridLayout(self.content_frame, columns=9)
        
        # 创建图表区域
        self.chart_frame = ResponsiveFrame(self.content_frame)
        self.content_layout.place_widget(self.chart_frame, row=0, column=0, columnspan=9, rowspan=6, sticky="nsew")
        
        # 创建数据表格区域
        self.table_frame = ResponsiveFrame(self.content_frame)
        self.content_layout.place_widget(self.table_frame, row=6, column=0, columnspan=9, rowspan=4, sticky="nsew")
        
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="导入数据", command=self.menu_import_data)
        file_menu.add_command(label="导出报告", command=self.menu_export_report)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 视图菜单
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="默认主题", command=lambda: self.change_theme("default"))
        view_menu.add_command(label="暗色主题", command=lambda: self.change_theme("dark"))
        view_menu.add_command(label="亮色主题", command=lambda: self.change_theme("light"))
        view_menu.add_separator()
        view_menu.add_command(label="刷新数据", command=self.refresh_data)
        menubar.add_cascade(label="视图", menu=view_menu)
        
        # 工具菜单
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="设置", command=self.menu_settings)
        tools_menu.add_command(label="回测系统", command=self.menu_backtest)
        menubar.add_cascade(label="工具", menu=tools_menu)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="帮助文档", command=self.menu_help)
        help_menu.add_command(label="关于", command=self.menu_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        
        self.root.config(menu=menubar)
        
    def initialize_sample_data(self):
        """初始化示例数据"""
        # 创建示例股票列表
        self.stocks = [
            {"code": "AAPL", "name": "苹果公司", "price": 150.25, "change": 2.5, "change_percent": 1.67, 
             "volume": 32500000, "recommendation": "买入", "confidence": 0.85},
            {"code": "MSFT", "name": "微软公司", "price": 285.11, "change": -1.2, "change_percent": -0.42, 
             "volume": 22100000, "recommendation": "持有", "confidence": 0.65},
            {"code": "AMZN", "name": "亚马逊", "price": 128.53, "change": 3.1, "change_percent": 2.41, 
             "volume": 41200000, "recommendation": "买入", "confidence": 0.78},
            {"code": "GOOGL", "name": "谷歌", "price": 112.87, "change": -0.5, "change_percent": -0.44, 
             "volume": 18900000, "recommendation": "持有", "confidence": 0.60},
            {"code": "TSLA", "name": "特斯拉", "price": 228.93, "change": 10.8, "change_percent": 4.72, 
             "volume": 52800000, "recommendation": "强烈买入", "confidence": 0.92},
        ]
        
        # 创建示例图表数据
        days = 30
        date_today = datetime.now()
        dates = [(date_today - timedelta(days=days-i)).strftime('%Y-%m-%d') for i in range(days)]
        
        # 模拟股价数据
        start_price = 100
        price_data = []
        current_price = start_price
        
        for i in range(days):
            price_change = random.uniform(-3, 3)
            current_price = max(current_price + price_change, current_price * 0.95)
            price_data.append(current_price)
            
        # 创建DataFrame
        self.chart_data = pd.DataFrame({
            'date': dates,
            'price': price_data,
            'volume': [random.randint(1000000, 5000000) for _ in range(days)]
        })
        
        # 计算移动平均线
        self.chart_data['MA5'] = self.chart_data['price'].rolling(5).mean()
        self.chart_data['MA10'] = self.chart_data['price'].rolling(10).mean()
        self.chart_data['MA20'] = self.chart_data['price'].rolling(20).mean()
        
    def populate_ui(self):
        """填充UI元素"""
        # 填充侧边栏
        self.populate_sidebar()
        
        # 填充图表区域
        self.populate_chart_area()
        
        # 填充数据表格
        self.populate_table()
        
    def populate_sidebar(self):
        """填充侧边栏"""
        # 创建搜索框
        search_frame = self.ui_factory.create_frame(self.sidebar_frame)
        search_frame.pack(fill="x", pady=10)
        
        search_label = self.ui_factory.create_label(search_frame, "搜索股票:")
        search_label.pack(anchor="w")
        
        self.search_var = tk.StringVar()
        search_entry = self.ui_factory.create_entry(search_frame, textvariable=self.search_var)
        search_entry.pack(fill="x", pady=5)
        
        search_button = self.ui_factory.create_button(search_frame, "搜索", command=self.search_stock)
        search_button.pack(anchor="e")
        
        # 创建筛选区域
        filter_frame = self.ui_factory.create_frame(self.sidebar_frame)
        filter_frame.pack(fill="x", pady=10)
        
        filter_label = self.ui_factory.create_heading(filter_frame, "筛选条件")
        filter_label.pack(anchor="w", pady=(0, 10))
        
        # 风险等级筛选
        risk_label = self.ui_factory.create_label(filter_frame, "风险等级:")
        risk_label.pack(anchor="w")
        
        self.risk_var = tk.StringVar(value="全部")
        risk_combo = self.ui_factory.create_combobox(filter_frame, 
                                                  values=["全部", "低风险", "中等风险", "高风险"],
                                                  textvariable=self.risk_var)
        risk_combo.pack(fill="x", pady=(0, 10))
        
        # 行业筛选
        industry_label = self.ui_factory.create_label(filter_frame, "行业:")
        industry_label.pack(anchor="w")
        
        self.industry_var = tk.StringVar(value="全部")
        industry_combo = self.ui_factory.create_combobox(filter_frame, 
                                                     values=["全部", "金融", "科技", "医药", "消费"],
                                                     textvariable=self.industry_var)
        industry_combo.pack(fill="x", pady=(0, 10))
        
        # 推荐类型筛选
        rec_label = self.ui_factory.create_label(filter_frame, "推荐类型:")
        rec_label.pack(anchor="w")
        
        self.rec_var = tk.StringVar(value="全部")
        rec_combo = self.ui_factory.create_combobox(filter_frame, 
                                                 values=["全部", "强烈买入", "买入", "持有", "卖出", "强烈卖出"],
                                                 textvariable=self.rec_var)
        rec_combo.pack(fill="x", pady=(0, 10))
        
        # 应用筛选按钮
        apply_button = self.ui_factory.create_button(filter_frame, "应用筛选", command=self.apply_filters)
        apply_button.pack(fill="x", pady=10)
        
        # 创建操作区域
        action_frame = self.ui_factory.create_frame(self.sidebar_frame)
        action_frame.pack(fill="x", pady=10)
        
        action_label = self.ui_factory.create_heading(action_frame, "操作")
        action_label.pack(anchor="w", pady=(0, 10))
        
        refresh_button = self.ui_factory.create_button(action_frame, "刷新数据", command=self.refresh_data)
        refresh_button.pack(fill="x", pady=5)
        
        analyze_button = self.ui_factory.create_button(action_frame, "分析所选股票", command=self.analyze_stock)
        analyze_button.pack(fill="x", pady=5)
        
        report_button = self.ui_factory.create_button(action_frame, "生成报告", command=self.generate_report)
        report_button.pack(fill="x", pady=5)
        
    def populate_chart_area(self):
        """填充图表区域"""
        # 创建标题
        chart_title = self.ui_factory.create_heading(self.chart_frame, "股票走势图")
        chart_title.pack(anchor="w", pady=(0, 10))
        
        # 创建图表
        self.create_stock_chart()
        
    def create_stock_chart(self):
        """创建股票图表"""
        # 创建Matplotlib图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # 股价图
        ax1.plot(self.chart_data['date'], self.chart_data['price'], label='价格', color='black', linewidth=1.5)
        ax1.plot(self.chart_data['date'], self.chart_data['MA5'], label='MA5', color='#FF9800', linewidth=1)
        ax1.plot(self.chart_data['date'], self.chart_data['MA10'], label='MA10', color='#9C27B0', linewidth=1)
        ax1.plot(self.chart_data['date'], self.chart_data['MA20'], label='MA20', color='#00BCD4', linewidth=1)
        
        # 设置标签和标题
        ax1.set_title('AAPL 股价走势')
        ax1.set_ylabel('价格')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        # 成交量图
        ax2.bar(self.chart_data['date'], self.chart_data['volume'], color='#007BFF', alpha=0.7)
        ax2.set_xlabel('日期')
        ax2.set_ylabel('成交量')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # 格式化x轴日期
        fig.autofmt_xdate()
        
        # 调整布局
        plt.tight_layout()
        
        # 创建Canvas
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def populate_table(self):
        """填充数据表格"""
        # 创建标题
        table_title = self.ui_factory.create_heading(self.table_frame, "推荐股票列表")
        table_title.pack(anchor="w", pady=(10, 10))
        
        # 创建表格
        columns = ("code", "name", "price", "change", "change_percent", "volume", "recommendation", "confidence")
        headings = ("代码", "名称", "价格", "涨跌额", "涨跌幅", "成交量", "推荐", "置信度")
        
        self.stock_table = self.ui_factory.create_treeview(self.table_frame, columns=columns, headings=headings)
        self.stock_table.pack(fill=tk.BOTH, expand=True)
        
        # 设置列宽
        self.stock_table.column("code", width=80)
        self.stock_table.column("name", width=120)
        self.stock_table.column("price", width=80)
        self.stock_table.column("change", width=80)
        self.stock_table.column("change_percent", width=80)
        self.stock_table.column("volume", width=100)
        self.stock_table.column("recommendation", width=80)
        self.stock_table.column("confidence", width=80)
        
        # 添加数据
        for i, stock in enumerate(self.stocks):
            values = (
                stock["code"],
                stock["name"],
                f"{stock['price']:.2f}",
                f"{stock['change']:+.2f}",
                f"{stock['change_percent']:+.2f}%",
                f"{stock['volume']:,}",
                stock["recommendation"],
                f"{stock['confidence']:.2f}"
            )
            
            # 根据涨跌设置颜色标签
            if stock["change"] > 0:
                tag = "up"
            elif stock["change"] < 0:
                tag = "down"
            else:
                tag = "neutral"
                
            item_id = self.stock_table.insert("", tk.END, values=values, tags=(tag,))
            
        # 配置行颜色
        self.stock_table.tag_configure("up", foreground=self.theme_manager.theme["success_color"])
        self.stock_table.tag_configure("down", foreground=self.theme_manager.theme["danger_color"])
        
        # 绑定点击事件
        self.stock_table.bind("<ButtonRelease-1>", self.on_table_click)
        
        # 添加滚动条
        scrollbar = self.ui_factory.create_scrollbar(self.table_frame, self.stock_table)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def on_table_click(self, event):
        """表格点击事件处理"""
        item = self.stock_table.identify_row(event.y)
        if item:
            # 获取所选行数据
            values = self.stock_table.item(item, "values")
            self.status_bar.set_status(f"已选择: {values[1]} ({values[0]})")
            
    def change_theme(self, theme_name):
        """更改主题"""
        # 更新主题
        self.theme_manager.set_theme(theme_name)
        
        # 应用主题到窗口
        self.theme_manager.apply_theme_to_widgets(self.root)
        
        # 重建UI
        # 注意：在实际应用中，这里可能需要更复杂的逻辑来保持当前状态
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        self.create_layout()
        self.populate_ui()
        
        # 显示通知
        self.notification.show(f"已切换到{theme_name}主题", type="success")
        
    def refresh_data(self):
        """刷新数据"""
        # 在实际应用中，这里会从数据源获取最新数据
        self.status_bar.set_status("正在刷新数据...")
        
        # 模拟数据刷新（随机更新价格）
        for stock in self.stocks:
            old_price = stock["price"]
            # 随机增减价格 (-3% 到 +3%)
            price_change_percent = random.uniform(-3.0, 3.0) / 100
            price_change = old_price * price_change_percent
            new_price = old_price + price_change
            
            stock["price"] = new_price
            stock["change"] = price_change
            stock["change_percent"] = price_change_percent * 100
            stock["volume"] = random.randint(10000000, 50000000)
        
        # 清空表格
        for item in self.stock_table.get_children():
            self.stock_table.delete(item)
            
        # 重新添加数据
        for stock in self.stocks:
            values = (
                stock["code"],
                stock["name"],
                f"{stock['price']:.2f}",
                f"{stock['change']:+.2f}",
                f"{stock['change_percent']:+.2f}%",
                f"{stock['volume']:,}",
                stock["recommendation"],
                f"{stock['confidence']:.2f}"
            )
            
            # 根据涨跌设置颜色标签
            if stock["change"] > 0:
                tag = "up"
            elif stock["change"] < 0:
                tag = "down"
            else:
                tag = "neutral"
                
            self.stock_table.insert("", tk.END, values=values, tags=(tag,))
            
        # 更新状态
        current_time = datetime.now().strftime("%H:%M:%S")
        self.status_bar.set_status(f"数据已更新 ({current_time})")
        self.notification.show("数据已刷新", type="success")
        
    def search_stock(self):
        """搜索股票"""
        search_term = self.search_var.get().strip().upper()
        if not search_term:
            return
            
        # 在实际应用中，这里会执行实际的股票搜索
        # 这里仅做简单的模拟
        found = False
        for stock in self.stocks:
            if search_term in stock["code"] or search_term in stock["name"].upper():
                # 选中匹配的行
                for item in self.stock_table.get_children():
                    values = self.stock_table.item(item, "values")
                    if values[0] == stock["code"]:
                        self.stock_table.selection_set(item)
                        self.stock_table.focus(item)
                        self.stock_table.see(item)
                        found = True
                        break
                if found:
                    self.status_bar.set_status(f"已找到: {stock['name']} ({stock['code']})")
                    break
                    
        if not found:
            self.notification.show(f"未找到股票: {search_term}", type="warning")
            
    def apply_filters(self):
        """应用筛选条件"""
        risk = self.risk_var.get()
        industry = self.industry_var.get()
        recommendation = self.rec_var.get()
        
        self.status_bar.set_status(f"应用筛选: 风险={risk}, 行业={industry}, 推荐={recommendation}")
        self.notification.show("筛选条件已应用", type="info")
        
    def analyze_stock(self):
        """分析所选股票"""
        selected = self.stock_table.selection()
        if not selected:
            self.notification.show("请先选择一支股票", type="warning")
            return
            
        # 获取所选行数据
        values = self.stock_table.item(selected[0], "values")
        
        # 创建消息对话框
        self.dialog_factory.create_message_dialog(
            title="股票分析",
            message=f"正在分析 {values[1]} ({values[0]}) 的数据...\n分析结果将在准备好后显示。",
            type="info"
        )
        
    def generate_report(self):
        """生成报告"""
        # 创建确认对话框
        def on_confirm(confirmed):
            if confirmed:
                self.status_bar.set_status("正在生成报告...")
                self.notification.show("报告生成完成", type="success")
                
        self.dialog_factory.create_confirm_dialog(
            title="生成报告",
            message="是否要生成当前分析的完整报告？",
            callback=on_confirm
        )
        
    def menu_import_data(self):
        """菜单：导入数据"""
        self.dialog_factory.create_input_dialog(
            title="导入数据",
            prompt="请输入数据文件路径:",
            default="data/stocks.csv",
            callback=lambda path: self.notification.show(f"已导入数据: {path}", type="success")
        )
        
    def menu_export_report(self):
        """菜单：导出报告"""
        self.dialog_factory.create_input_dialog(
            title="导出报告",
            prompt="请输入报告保存路径:",
            default="reports/stock_report.pdf",
            callback=lambda path: self.notification.show(f"报告已导出到: {path}", type="success")
        )
        
    def menu_settings(self):
        """菜单：设置"""
        self.notification.show("设置功能尚未实现", type="info")
        
    def menu_backtest(self):
        """菜单：回测系统"""
        self.notification.show("回测系统尚未实现", type="info")
        
    def menu_help(self):
        """菜单：帮助文档"""
        self.dialog_factory.create_message_dialog(
            title="帮助文档",
            message="请访问 docs/help.md 获取详细的使用说明。",
            type="info"
        )
        
    def menu_about(self):
        """菜单：关于"""
        self.dialog_factory.create_message_dialog(
            title="关于",
            message="股票推荐系统 v1.0\n© 2023 股票推荐系统团队",
            type="info"
        )
        

def main():
    """主函数"""
    root = tk.Tk()
    app = StockAppUI(root)
    root.mainloop()
    
if __name__ == "__main__":
    main()