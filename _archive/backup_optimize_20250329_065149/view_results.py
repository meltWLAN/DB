#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版分析结果展示程序
"""

import os
import sys
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

# 设置根目录
ROOT_DIR = Path(__file__).parent
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
MA_CHARTS_DIR = os.path.join(RESULTS_DIR, "ma_charts")

class ResultViewer:
    """简化版结果查看器"""
    
    def __init__(self, root):
        """初始化界面"""
        self.root = root
        self.root.title("股票分析结果查看器")
        self.root.geometry("1200x700")
        
        # 创建菜单
        self.create_menu()
        
        # 创建框架
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建选项卡
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 创建各选项卡
        self.create_momentum_tab()
        self.create_ma_tab()
        
        # 状态栏
        self.status_message = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_message, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 加载数据
        self.status_message.set("正在加载数据...")
        self.load_data()
        
    def create_menu(self):
        """创建菜单"""
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        
        # 文件菜单
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="刷新数据", command=self.load_data)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menu_bar.add_cascade(label="文件", menu=file_menu)
        
        # 帮助菜单
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        menu_bar.add_cascade(label="帮助", menu=help_menu)
    
    def create_momentum_tab(self):
        """创建动量分析选项卡"""
        momentum_frame = ttk.Frame(self.notebook)
        self.notebook.add(momentum_frame, text="动量分析结果")
        
        # 创建Treeview
        columns = ("ts_code", "name", "industry", "close", "momentum", "rsi", "macd", "volume_ratio", "score")
        self.momentum_tree = ttk.Treeview(momentum_frame, columns=columns, show="headings", selectmode="browse")
        
        # 设置列标题
        self.momentum_tree.heading("ts_code", text="股票代码")
        self.momentum_tree.heading("name", text="股票名称") 
        self.momentum_tree.heading("industry", text="行业")
        self.momentum_tree.heading("close", text="收盘价")
        self.momentum_tree.heading("momentum", text="动量")
        self.momentum_tree.heading("rsi", text="RSI")
        self.momentum_tree.heading("macd", text="MACD")
        self.momentum_tree.heading("volume_ratio", text="成交量比")
        self.momentum_tree.heading("score", text="得分")
        
        # 设置列宽
        self.momentum_tree.column("ts_code", width=100)
        self.momentum_tree.column("name", width=100)
        self.momentum_tree.column("industry", width=100)
        self.momentum_tree.column("close", width=70)
        self.momentum_tree.column("momentum", width=70)
        self.momentum_tree.column("rsi", width=70)
        self.momentum_tree.column("macd", width=70)
        self.momentum_tree.column("volume_ratio", width=70)
        self.momentum_tree.column("score", width=70)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(momentum_frame, orient=tk.VERTICAL, command=self.momentum_tree.yview)
        self.momentum_tree.configure(yscroll=scrollbar.set)
        
        # 放置组件
        self.momentum_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定双击事件
        self.momentum_tree.bind("<Double-1>", self.on_momentum_tree_double_click)
    
    def create_ma_tab(self):
        """创建均线交叉选项卡"""
        ma_frame = ttk.Frame(self.notebook)
        self.notebook.add(ma_frame, text="均线交叉结果")
        
        # 创建Treeview
        columns = ("ts_code", "name", "industry", "close", "signal", "short_ma", "long_ma", 
                   "total_return", "annual_return", "max_drawdown", "win_rate")
        self.ma_tree = ttk.Treeview(ma_frame, columns=columns, show="headings", selectmode="browse")
        
        # 设置列标题
        self.ma_tree.heading("ts_code", text="股票代码")
        self.ma_tree.heading("name", text="股票名称")
        self.ma_tree.heading("industry", text="行业")
        self.ma_tree.heading("close", text="收盘价")
        self.ma_tree.heading("signal", text="信号")
        self.ma_tree.heading("short_ma", text="短期均线")
        self.ma_tree.heading("long_ma", text="长期均线")
        self.ma_tree.heading("total_return", text="总收益")
        self.ma_tree.heading("annual_return", text="年化收益")
        self.ma_tree.heading("max_drawdown", text="最大回撤")
        self.ma_tree.heading("win_rate", text="胜率")
        
        # 设置列宽
        self.ma_tree.column("ts_code", width=100)
        self.ma_tree.column("name", width=100)
        self.ma_tree.column("industry", width=100)
        self.ma_tree.column("close", width=70)
        self.ma_tree.column("signal", width=70)
        self.ma_tree.column("short_ma", width=70)
        self.ma_tree.column("long_ma", width=70)
        self.ma_tree.column("total_return", width=70)
        self.ma_tree.column("annual_return", width=70)
        self.ma_tree.column("max_drawdown", width=70)
        self.ma_tree.column("win_rate", width=70)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(ma_frame, orient=tk.VERTICAL, command=self.ma_tree.yview)
        self.ma_tree.configure(yscroll=scrollbar.set)
        
        # 放置组件
        self.ma_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定双击事件
        self.ma_tree.bind("<Double-1>", self.on_ma_tree_double_click)
        
    def load_data(self):
        """加载分析结果数据"""
        try:
            # 查找最新的结果文件
            momentum_files = self.find_csv_files(os.path.join(RESULTS_DIR, "momentum"))
            ma_files = self.find_csv_files(os.path.join(RESULTS_DIR, "ma_cross"))
            
            # 加载动量分析结果
            if momentum_files:
                latest_momentum = momentum_files[0]
                print(f"加载动量分析结果文件: {latest_momentum}")
                momentum_df = pd.read_csv(latest_momentum)
                self.update_momentum_tree(momentum_df)
                self.status_message.set(f"已加载动量分析结果: {os.path.basename(latest_momentum)}")
            else:
                print("未找到动量分析结果文件")
                self.status_message.set("未找到动量分析结果文件")
            
            # 加载均线交叉结果
            if ma_files:
                latest_ma = ma_files[0]
                print(f"加载均线交叉结果文件: {latest_ma}")
                ma_df = pd.read_csv(latest_ma)
                self.update_ma_tree(ma_df)
                self.status_message.set(f"已加载均线交叉结果: {os.path.basename(latest_ma)}")
            else:
                print("未找到均线交叉结果文件")
                self.status_message.set("未找到均线交叉结果文件")
                
            # 检查是否有任何文件被加载
            if not momentum_files and not ma_files:
                # 尝试搜索整个结果目录
                all_csv_files = self.find_csv_files(RESULTS_DIR)
                if all_csv_files:
                    latest_file = all_csv_files[0]
                    print(f"尝试加载通用结果文件: {latest_file}")
                    df = pd.read_csv(latest_file)
                    # 检查文件内容来确定类型
                    if 'score' in df.columns or 'momentum_score' in df.columns:
                        self.update_momentum_tree(df)
                        self.status_message.set(f"已加载动量分析结果: {os.path.basename(latest_file)}")
                    elif 'current_signal' in df.columns:
                        self.update_ma_tree(df)
                        self.status_message.set(f"已加载均线交叉结果: {os.path.basename(latest_file)}")
                    else:
                        print(f"找到结果文件，但无法识别类型: {latest_file}")
                        self.status_message.set(f"找到结果文件，但无法识别类型: {os.path.basename(latest_file)}")
                else:
                    print("在整个结果目录中未找到任何CSV文件")
                    self.status_message.set("未找到任何分析结果文件")
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            self.status_message.set(f"加载数据时出错: {str(e)}")
            messagebox.showerror("错误", f"加载数据时出错: {str(e)}")
            
    def find_csv_files(self, directory):
        """查找指定目录中的CSV文件，按修改时间排序"""
        try:
            if not os.path.exists(directory):
                print(f"目录不存在: {directory}")
                return []
                
            csv_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.csv'):
                        full_path = os.path.join(root, file)
                        csv_files.append(full_path)
            
            # 按修改时间排序，最新的在前
            csv_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return csv_files
        except Exception as e:
            print(f"查找CSV文件时出错: {e}")
            return []
        
    def update_momentum_tree(self, df):
        """更新动量分析结果表格"""
        # 清空表格
        for item in self.momentum_tree.get_children():
            self.momentum_tree.delete(item)
            
        # 检查DataFrame是否为空
        if df.empty:
            print("动量分析结果为空")
            return
            
        # 打印DataFrame的列名，以便调试
        print(f"动量结果数据框列名: {df.columns.tolist()}")
        
        # 添加数据
        for _, row in df.iterrows():
            try:
                # 创建一个值的列表
                values = []
                
                # 股票代码
                values.append(row.get('ts_code', ''))
                
                # 股票名称
                values.append(row.get('name', ''))
                
                # 行业
                values.append(row.get('industry', ''))
                
                # 收盘价
                values.append(f"{row.get('close', 0):.2f}")
                
                # 动量
                momentum_value = row.get('momentum_20d', row.get('momentum_20', row.get('momentum', 0)))
                values.append(f"{momentum_value:.2%}" if isinstance(momentum_value, (int, float)) else momentum_value)
                
                # RSI
                values.append(f"{row.get('rsi', 0):.1f}")
                
                # MACD
                macd_value = row.get('macd_hist', row.get('macd', 0))
                values.append(f"{macd_value:.3f}")
                
                # 成交量比
                values.append(f"{row.get('volume_ratio', 1):.2f}")
                
                # 得分
                score_value = row.get('momentum_score', row.get('score', 0))
                values.append(f"{score_value:.1f}")
                
                # 插入到树形控件
                self.momentum_tree.insert("", tk.END, values=values)
                
            except Exception as e:
                print(f"插入动量分析数据时出错: {e}, 行数据: {row}")
                
    def update_ma_tree(self, df):
        """更新均线交叉结果表格"""
        # 清空表格
        for item in self.ma_tree.get_children():
            self.ma_tree.delete(item)
            
        # 检查DataFrame是否为空
        if df.empty:
            print("均线交叉结果为空")
            return
            
        # 打印DataFrame的列名，以便调试
        print(f"均线交叉结果数据框列名: {df.columns.tolist()}")
        
        # 添加数据
        for _, row in df.iterrows():
            try:
                # 创建一个值的列表
                values = []
                
                # 股票代码
                values.append(row.get('ts_code', ''))
                
                # 股票名称
                values.append(row.get('name', ''))
                
                # 行业
                values.append(row.get('industry', ''))
                
                # 收盘价
                values.append(f"{row.get('close', 0):.2f}")
                
                # 信号
                values.append(row.get('current_signal', '无信号'))
                
                # 短期均线
                values.append(f"{row.get('short_ma', 0)}")
                
                # 长期均线
                values.append(f"{row.get('long_ma', 0)}")
                
                # 总收益
                total_return = row.get('total_return', 0)
                values.append(f"{total_return:.2%}" if isinstance(total_return, (int, float)) else total_return)
                
                # 年化收益
                annual_return = row.get('annual_return', 0)
                values.append(f"{annual_return:.2%}" if isinstance(annual_return, (int, float)) else annual_return)
                
                # 最大回撤
                max_drawdown = row.get('max_drawdown', 0)
                values.append(f"{max_drawdown:.2%}" if isinstance(max_drawdown, (int, float)) else max_drawdown)
                
                # 胜率
                win_rate = row.get('win_rate', 0)
                values.append(f"{win_rate:.2%}" if isinstance(win_rate, (int, float)) else win_rate)
                
                # 插入到树形控件
                self.ma_tree.insert("", tk.END, values=values)
                
            except Exception as e:
                print(f"插入均线交叉数据时出错: {e}, 行数据: {row}")
    
    def on_momentum_tree_double_click(self, event):
        """双击动量分析表格项的处理函数"""
        if not self.momentum_tree.selection():
            return
            
        item = self.momentum_tree.selection()[0]
        stock_code = self.momentum_tree.item(item, "values")[0]
        
        # 查找并显示图表
        chart_path = os.path.join(CHARTS_DIR, f"{stock_code}_momentum.png")
        if os.path.exists(chart_path):
            self.show_chart(chart_path)
        else:
            messagebox.showinfo("提示", f"未找到{stock_code}的图表文件")
    
    def on_ma_tree_double_click(self, event):
        """双击均线交叉表格项的处理函数"""
        if not self.ma_tree.selection():
            return
            
        item = self.ma_tree.selection()[0]
        stock_code = self.ma_tree.item(item, "values")[0]
        
        # 查找并显示图表
        chart_path = os.path.join(MA_CHARTS_DIR, f"{stock_code}_ma_cross.png")
        if os.path.exists(chart_path):
            self.show_chart(chart_path)
        else:
            messagebox.showinfo("提示", f"未找到{stock_code}的图表文件")
            
    def show_chart(self, chart_path):
        """显示图表"""
        try:
            # 创建新窗口
            chart_window = tk.Toplevel(self.root)
            chart_window.title("图表查看")
            chart_window.geometry("1000x800")
            
            # 加载图像
            try:
                from PIL import Image, ImageTk
                
                # 使用Pillow打开图像
                image = Image.open(chart_path)
                # 调整图像大小以适应窗口
                image = image.resize((980, 780), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                # 在Label中显示图像
                label = tk.Label(chart_window, image=photo)
                label.image = photo  # 保持引用以防止被垃圾回收
                label.pack(padx=10, pady=10)
                
            except ImportError:
                # 如果没有Pillow，使用原生PhotoImage
                img = tk.PhotoImage(file=chart_path)
                canvas = tk.Canvas(chart_window, width=img.width(), height=img.height())
                canvas.pack(fill=tk.BOTH, expand=True)
                canvas.create_image(0, 0, anchor=tk.NW, image=img)
                canvas.image = img
                
        except Exception as e:
            print(f"显示图表时出错: {e}")
            messagebox.showerror("错误", f"显示图表时出错: {str(e)}")
            
    def show_about(self):
        """显示关于对话框"""
        about_text = """
        股票分析结果查看器
        
        用于查看和管理股票分析结果的简单工具
        
        功能:
        - 查看动量分析结果
        - 查看均线交叉策略结果
        - 查看分析图表
        """
        messagebox.showinfo("关于", about_text)

def main():
    """主函数"""
    root = tk.Tk()
    app = ResultViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main() 