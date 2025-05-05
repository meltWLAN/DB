#!/usr/bin/env python3
"""
整合版股票分析系统 - 结合分析和结果查看功能
绕过pyenv问题的版本
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

# 创建必要的目录
os.makedirs("logs", exist_ok=True)
os.makedirs("results/charts", exist_ok=True)
os.makedirs("results/ma_charts", exist_ok=True)
os.makedirs("results/momentum", exist_ok=True)
os.makedirs("results/ma_cross", exist_ok=True)
os.makedirs("data", exist_ok=True)

class IntegratedSystemLauncher:
    """整合版系统启动器"""
    
    def __init__(self, root):
        """初始化界面"""
        self.root = root
        self.root.title("股票分析系统 - 整合版启动器")
        self.root.geometry("600x400")
        
        # 设置样式
        self.setup_styles()
        
        # 创建框架
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题标签
        title_label = ttk.Label(
            main_frame, 
            text="股票分析系统", 
            style="Title.TLabel"
        )
        title_label.pack(pady=(0, 20))
        
        # 说明标签
        instructions_label = ttk.Label(
            main_frame,
            text="请选择要启动的功能:",
            style="Instructions.TLabel"
        )
        instructions_label.pack(pady=(0, 20), anchor=tk.W)
        
        # 选项按钮
        self.create_option_buttons(main_frame)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_styles(self):
        """设置控件样式"""
        style = ttk.Style()
        
        # 标题样式
        style.configure("Title.TLabel", font=("Arial", 20, "bold"))
        
        # 说明文字样式
        style.configure("Instructions.TLabel", font=("Arial", 12))
        
        # 按钮样式
        style.configure("Option.TButton", font=("Arial", 12), padding=10)
        
    def create_option_buttons(self, parent):
        """创建选项按钮"""
        # 分析按钮
        analysis_frame = ttk.Frame(parent)
        analysis_frame.pack(fill=tk.X, pady=10)
        
        analysis_button = ttk.Button(
            analysis_frame,
            text="启动分析系统",
            style="Option.TButton",
            command=self.launch_analysis_system
        )
        analysis_button.pack(side=tk.LEFT, padx=5)
        
        analysis_desc = ttk.Label(
            analysis_frame,
            text="执行股票动量分析、均线交叉策略和组合策略",
            wraplength=350
        )
        analysis_desc.pack(side=tk.LEFT, padx=10)
        
        # 结果查看按钮
        viewer_frame = ttk.Frame(parent)
        viewer_frame.pack(fill=tk.X, pady=10)
        
        viewer_button = ttk.Button(
            viewer_frame,
            text="启动结果查看器",
            style="Option.TButton",
            command=self.launch_result_viewer
        )
        viewer_button.pack(side=tk.LEFT, padx=5)
        
        viewer_desc = ttk.Label(
            viewer_frame,
            text="查看已完成的分析结果和图表",
            wraplength=350
        )
        viewer_desc.pack(side=tk.LEFT, padx=10)
        
        # 修复问题按钮
        fix_frame = ttk.Frame(parent)
        fix_frame.pack(fill=tk.X, pady=10)
        
        fix_button = ttk.Button(
            fix_frame,
            text="修复界面问题",
            style="Option.TButton",
            command=self.fix_display_issues
        )
        fix_button.pack(side=tk.LEFT, padx=5)
        
        fix_desc = ttk.Label(
            fix_frame,
            text="整理结果文件并修复界面展示问题",
            wraplength=350
        )
        fix_desc.pack(side=tk.LEFT, padx=10)
        
        # 整合模式按钮
        integrated_frame = ttk.Frame(parent)
        integrated_frame.pack(fill=tk.X, pady=10)
        
        integrated_button = ttk.Button(
            integrated_frame,
            text="一键启动整合模式",
            style="Option.TButton",
            command=self.launch_integrated_mode
        )
        integrated_button.pack(side=tk.LEFT, padx=5)
        
        integrated_desc = ttk.Label(
            integrated_frame,
            text="同时启动分析系统和结果查看器",
            wraplength=350
        )
        integrated_desc.pack(side=tk.LEFT, padx=10)
        
    def launch_analysis_system(self):
        """启动分析系统"""
        self.status_var.set("正在启动分析系统...")
        
        # 确保必要的文件存在
        if not os.path.exists("override_run.py"):
            self.create_override_run_script()
            
        try:
            # 使用子进程启动
            subprocess.Popen([sys.executable, "override_run.py"])
            self.status_var.set("分析系统已启动")
        except Exception as e:
            messagebox.showerror("启动错误", f"启动分析系统失败: {str(e)}")
            self.status_var.set(f"启动失败: {str(e)}")
    
    def launch_result_viewer(self):
        """启动结果查看器"""
        self.status_var.set("正在启动结果查看器...")
        
        # 确保必要的文件存在
        if not os.path.exists("view_results.py"):
            self.create_view_results_script()
            
        try:
            # 使用子进程启动
            subprocess.Popen([sys.executable, "view_results.py"])
            self.status_var.set("结果查看器已启动")
        except Exception as e:
            messagebox.showerror("启动错误", f"启动结果查看器失败: {str(e)}")
            self.status_var.set(f"启动失败: {str(e)}")
    
    def fix_display_issues(self):
        """修复界面展示问题"""
        self.status_var.set("正在修复界面展示问题...")
        
        # 确保必要的文件存在
        if not os.path.exists("fix_gui_display.py"):
            self.create_fix_gui_script()
            
        try:
            # 使用子进程启动并等待完成
            process = subprocess.Popen(
                [sys.executable, "fix_gui_display.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                self.status_var.set("界面问题修复完成")
                messagebox.showinfo("修复完成", "界面展示问题已修复")
            else:
                self.status_var.set("修复失败")
                messagebox.showerror("修复错误", f"修复失败: {stderr}")
        except Exception as e:
            messagebox.showerror("修复错误", f"修复失败: {str(e)}")
            self.status_var.set(f"修复失败: {str(e)}")
    
    def launch_integrated_mode(self):
        """启动整合模式"""
        self.status_var.set("正在启动整合模式...")
        
        # 先启动分析系统
        try:
            if not os.path.exists("override_run.py"):
                self.create_override_run_script()
            subprocess.Popen([sys.executable, "override_run.py"])
            
            # 等待一秒后启动结果查看器
            self.root.after(1000, self._delayed_launch_viewer)
            
        except Exception as e:
            messagebox.showerror("启动错误", f"启动整合模式失败: {str(e)}")
            self.status_var.set(f"启动失败: {str(e)}")
    
    def _delayed_launch_viewer(self):
        """延迟启动结果查看器"""
        try:
            if not os.path.exists("view_results.py"):
                self.create_view_results_script()
            subprocess.Popen([sys.executable, "view_results.py"])
            self.status_var.set("整合模式已启动")
        except Exception as e:
            messagebox.showerror("启动错误", f"启动结果查看器失败: {str(e)}")
            self.status_var.set(f"结果查看器启动失败: {str(e)}")
    
    def create_override_run_script(self):
        """创建override_run.py脚本"""
        script_content = """#!/usr/bin/env python3
\"\"\"
特殊的启动脚本，用于绕过pyenv拦截，启动主系统
\"\"\"

import os
import sys
import subprocess
import platform

def main():
    \"\"\"主函数\"\"\"
    print("========================================")
    print("   股票分析系统 - 特殊启动脚本")
    print("========================================")
    print()
    
    # 创建必要的目录
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results/charts", exist_ok=True)
    os.makedirs("results/ma_charts", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 系统信息
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    print()
    
    # 设置环境变量
    os.environ["TUSHARE_TOKEN"] = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    
    print("正在启动主系统...")
    
    # 直接使用当前Python解释器启动主系统
    try:
        # 使用子进程直接运行主系统
        subprocess.run([sys.executable, "stock_analysis_gui.py"])
    except Exception as e:
        print(f"启动失败: {e}")
        # 尝试备选方案
        try:
            print("尝试使用备选启动方式...")
            # 直接导入和运行模块
            sys.path.insert(0, os.getcwd())
            from stock_analysis_gui import main
            main()
        except Exception as e2:
            print(f"备选启动也失败: {e2}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
        with open("override_run.py", "w", encoding="utf-8") as f:
            f.write(script_content)
        os.chmod("override_run.py", 0o755)
    
    def create_view_results_script(self):
        """创建view_results.py脚本"""
        # 由于脚本较长，这里只创建一个简化版
        script_content = """#!/usr/bin/env python3
\"\"\"
简化版分析结果展示程序
\"\"\"

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from pathlib import Path
import glob

# 设置根目录
ROOT_DIR = Path(__file__).parent
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
MA_CHARTS_DIR = os.path.join(RESULTS_DIR, "ma_charts")

def find_latest_csv(directory):
    files = glob.glob(os.path.join(directory, "*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

class ResultViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("股票分析结果查看器")
        self.root.geometry("1200x700")
        
        # 创建选项卡
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 动量分析选项卡
        self.momentum_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.momentum_frame, text="动量分析结果")
        
        # 均线交叉选项卡
        self.ma_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ma_frame, text="均线交叉结果")
        
        # 创建表格和加载数据
        self.create_momentum_table()
        self.create_ma_table()
        self.load_data()
        
        # 状态栏
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 创建菜单
        self.create_menu()
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="刷新数据", command=self.load_data)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        
    def create_momentum_table(self):
        columns = ("ts_code", "name", "industry", "close", "momentum", "rsi", "macd", "volume_ratio", "score")
        
        self.momentum_tree = ttk.Treeview(self.momentum_frame, columns=columns, show="headings")
        
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
        for col in columns:
            self.momentum_tree.column(col, width=100)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(self.momentum_frame, orient=tk.VERTICAL, command=self.momentum_tree.yview)
        self.momentum_tree.configure(yscroll=scrollbar.set)
        
        # 放置组件
        self.momentum_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_ma_table(self):
        columns = ("ts_code", "name", "industry", "close", "signal", "total_return", "annual_return", "max_drawdown", "win_rate")
        
        self.ma_tree = ttk.Treeview(self.ma_frame, columns=columns, show="headings")
        
        # 设置列标题
        self.ma_tree.heading("ts_code", text="股票代码")
        self.ma_tree.heading("name", text="股票名称")
        self.ma_tree.heading("industry", text="行业")
        self.ma_tree.heading("close", text="收盘价")
        self.ma_tree.heading("signal", text="信号")
        self.ma_tree.heading("total_return", text="总收益")
        self.ma_tree.heading("annual_return", text="年化收益")
        self.ma_tree.heading("max_drawdown", text="最大回撤")
        self.ma_tree.heading("win_rate", text="胜率")
        
        # 设置列宽
        for col in columns:
            self.ma_tree.column(col, width=100)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(self.ma_frame, orient=tk.VERTICAL, command=self.ma_tree.yview)
        self.ma_tree.configure(yscroll=scrollbar.set)
        
        # 放置组件
        self.ma_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def load_data(self):
        self.status_var.set("正在加载数据...")
        
        # 加载动量分析结果
        momentum_dir = os.path.join(RESULTS_DIR, "momentum")
        momentum_file = find_latest_csv(momentum_dir)
        if not momentum_file:
            momentum_file = find_latest_csv(RESULTS_DIR)
            
        if momentum_file:
            try:
                df = pd.read_csv(momentum_file)
                self.load_momentum_data(df)
                self.status_var.set(f"已加载动量分析结果: {os.path.basename(momentum_file)}")
            except Exception as e:
                self.status_var.set(f"加载动量分析结果失败: {str(e)}")
        
        # 加载均线交叉结果
        ma_dir = os.path.join(RESULTS_DIR, "ma_cross")
        ma_file = find_latest_csv(ma_dir)
        if not ma_file:
            ma_file = find_latest_csv(RESULTS_DIR)
            
        if ma_file:
            try:
                df = pd.read_csv(ma_file)
                self.load_ma_data(df)
                self.status_var.set(f"已加载均线交叉结果: {os.path.basename(ma_file)}")
            except Exception as e:
                self.status_var.set(f"加载均线交叉结果失败: {str(e)}")
    
    def load_momentum_data(self, df):
        # 清空表格
        for i in self.momentum_tree.get_children():
            self.momentum_tree.delete(i)
            
        # 添加数据
        for _, row in df.iterrows():
            values = []
            for col in self.momentum_tree["columns"]:
                if col in row:
                    values.append(str(row[col]))
                else:
                    values.append("")
            self.momentum_tree.insert("", tk.END, values=values)
    
    def load_ma_data(self, df):
        # 清空表格
        for i in self.ma_tree.get_children():
            self.ma_tree.delete(i)
            
        # 添加数据
        for _, row in df.iterrows():
            values = []
            for col in self.ma_tree["columns"]:
                if col in row:
                    values.append(str(row[col]))
                else:
                    values.append("")
            self.ma_tree.insert("", tk.END, values=values)
    
    def show_about(self):
        messagebox.showinfo("关于", "股票分析结果查看器\\n\\n用于查看股票分析结果的简单工具")

def main():
    root = tk.Tk()
    app = ResultViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
"""
        with open("view_results.py", "w", encoding="utf-8") as f:
            f.write(script_content)
        os.chmod("view_results.py", 0o755)
    
    def create_fix_gui_script(self):
        """创建fix_gui_display.py脚本"""
        script_content = """#!/usr/bin/env python3
\"\"\"
优化GUI界面展示的辅助脚本
\"\"\"

import os
import sys
import subprocess
import glob
import shutil

# 设置结果目录
RESULTS_DIR = "results"
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
MA_CHARTS_DIR = os.path.join(RESULTS_DIR, "ma_charts")

def organize_results():
    \"\"\"整理分析结果文件到指定目录\"\"\"
    # 创建必要的目录
    os.makedirs(os.path.join(RESULTS_DIR, "momentum"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "ma_cross"), exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(MA_CHARTS_DIR, exist_ok=True)
    
    # 整理动量分析结果
    momentum_patterns = ["momentum_*.csv", "*momentum*.csv"]
    for pattern in momentum_patterns:
        for file_path in glob.glob(os.path.join(RESULTS_DIR, pattern)):
            target_path = os.path.join(RESULTS_DIR, "momentum", os.path.basename(file_path))
            try:
                shutil.copy2(file_path, target_path)
                print(f"已复制 {file_path} 到 {target_path}")
            except Exception as e:
                print(f"复制 {file_path} 时出错: {e}")
    
    # 整理均线交叉结果
    ma_patterns = ["ma_*.csv", "*ma_strategy*.csv", "*crossover*.csv"]
    for pattern in ma_patterns:
        for file_path in glob.glob(os.path.join(RESULTS_DIR, pattern)):
            target_path = os.path.join(RESULTS_DIR, "ma_cross", os.path.basename(file_path))
            try:
                shutil.copy2(file_path, target_path)
                print(f"已复制 {file_path} 到 {target_path}")
            except Exception as e:
                print(f"复制 {file_path} 时出错: {e}")
    
    # 整理图表文件
    for file_path in glob.glob(os.path.join(RESULTS_DIR, "*.png")):
        if "_momentum" in file_path.lower():
            target_path = os.path.join(CHARTS_DIR, os.path.basename(file_path))
        elif "_ma_" in file_path.lower() or "_cross" in file_path.lower():
            target_path = os.path.join(MA_CHARTS_DIR, os.path.basename(file_path))
        else:
            continue
            
        try:
            shutil.copy2(file_path, target_path)
            print(f"已复制 {file_path} 到 {target_path}")
        except Exception as e:
            print(f"复制 {file_path} 时出错: {e}")

def main():
    \"\"\"主函数\"\"\"
    print("========================================")
    print("   优化界面展示助手")
    print("========================================")
    
    print("\\n整理分析结果文件...")
    organize_results()
    
    print("\\n界面优化完成!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
        with open("fix_gui_display.py", "w", encoding="utf-8") as f:
            f.write(script_content)
        os.chmod("fix_gui_display.py", 0o755)

def main():
    """主函数"""
    # 创建并启动GUI
    root = tk.Tk()
    app = IntegratedSystemLauncher(root)
    root.mainloop()

if __name__ == "__main__":
    main() 