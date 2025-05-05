#!/usr/bin/env python3
"""
直接启动整合系统的脚本
绕过所有Python环境问题
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import platform

# 创建必要的目录
os.makedirs("logs", exist_ok=True)
os.makedirs("results/charts", exist_ok=True)
os.makedirs("results/ma_charts", exist_ok=True)
os.makedirs("results/momentum", exist_ok=True)
os.makedirs("results/ma_cross", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 获取系统Python路径
def get_system_python():
    """获取系统Python路径"""
    potential_paths = [
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3"
    ]
    
    for path in potential_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    return sys.executable

# 使用确定的系统Python路径
SYSTEM_PYTHON = get_system_python()

class DirectLauncher:
    """直接启动器"""
    
    def __init__(self, root):
        """初始化界面"""
        self.root = root
        self.root.title("股票分析系统 - 启动器")
        self.root.geometry("600x400")
        
        # 创建框架
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题标签
        title_label = ttk.Label(
            main_frame, 
            text="股票分析系统",
            font=("Arial", 20, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # 系统信息
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=10)
        
        system_info = f"系统: {platform.system()} {platform.release()}"
        python_info = f"Python: {sys.version.split()[0]}"
        system_python_info = f"系统Python: {SYSTEM_PYTHON}"
        
        info_label = ttk.Label(
            info_frame,
            text=f"{system_info}\n{python_info}\n{system_python_info}",
            font=("Arial", 10)
        )
        info_label.pack(anchor=tk.W)
        
        # 按钮
        self.create_buttons(main_frame)
        
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
    
    def create_buttons(self, parent):
        """创建按钮"""
        # 直接启动分析系统
        analysis_btn = ttk.Button(
            parent,
            text="启动分析系统",
            command=self.launch_analysis,
            padding=10
        )
        analysis_btn.pack(fill=tk.X, pady=10)
        
        # 直接启动结果查看器
        viewer_btn = ttk.Button(
            parent,
            text="启动结果查看器",
            command=self.launch_viewer,
            padding=10
        )
        viewer_btn.pack(fill=tk.X, pady=10)
        
        # 整理结果文件
        organize_btn = ttk.Button(
            parent,
            text="整理结果文件",
            command=self.organize_results,
            padding=10
        )
        organize_btn.pack(fill=tk.X, pady=10)
        
        # 一键启动全部
        all_btn = ttk.Button(
            parent,
            text="一键启动全部",
            command=self.launch_all,
            padding=10
        )
        all_btn.pack(fill=tk.X, pady=10)
    
    def launch_analysis(self):
        """启动分析系统"""
        self.status_var.set("正在启动分析系统...")
        
        # 使用直接内嵌代码而非依赖外部脚本
        try:
            # 设置环境变量
            os.environ["TUSHARE_TOKEN"] = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
            
            # 创建临时启动脚本
            temp_script = "temp_launch_stock_analysis.py"
            with open(temp_script, "w") as f:
                f.write("""
import os
import sys
import importlib.util
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('StockAnalysisLauncher')

# 确保Tushare Token已设置
os.environ["TUSHARE_TOKEN"] = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"

# 尝试导入主模块
try:
    logger.info("正在启动股票分析系统...")
    
    # 如果文件存在，直接导入并运行
    if os.path.exists("stock_analysis_gui.py"):
        spec = importlib.util.spec_from_file_location("stock_analysis_gui", "stock_analysis_gui.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.info("股票分析系统启动成功")
    else:
        logger.error("找不到股票分析GUI文件")
except Exception as e:
    logger.error(f"启动失败: {str(e)}")
""")
            
            # 使用系统Python直接运行临时脚本
            subprocess.Popen([SYSTEM_PYTHON, temp_script])
            self.status_var.set("分析系统已启动")
        except Exception as e:
            messagebox.showerror("启动错误", f"启动分析系统失败: {str(e)}")
            self.status_var.set(f"启动失败: {str(e)}")
    
    def launch_viewer(self):
        """启动结果查看器"""
        self.status_var.set("正在启动结果查看器...")
        
        # 创建临时结果查看器脚本
        try:
            # 解析结果目录
            results_dir = os.path.join(os.getcwd(), "results")
            csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
            
            if not csv_files:
                messagebox.showinfo("提示", "未找到分析结果文件")
                self.status_var.set("未找到分析结果")
                return
            
            # 创建临时结果查看器脚本
            temp_viewer = "temp_result_viewer.py"
            with open(temp_viewer, "w") as f:
                f.write("""
import os
import sys
import tkinter as tk
from tkinter import ttk
import pandas as pd
import glob

# 设置结果目录
RESULTS_DIR = "results"

# 查找最新结果文件
def find_latest_csv(pattern):
    files = glob.glob(os.path.join(RESULTS_DIR, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

# 主窗口类
class ResultViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("分析结果查看器")
        self.root.geometry("1000x600")
        
        # 创建选项卡
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 加载结果文件
        self.load_results()
    
    def load_results(self):
        # 查找结果文件
        momentum_file = find_latest_csv("*momentum*.csv")
        ma_file = find_latest_csv("*ma*.csv")
        
        # 添加动量分析结果选项卡
        if momentum_file:
            try:
                df = pd.read_csv(momentum_file)
                self.add_result_tab("动量分析结果", df)
            except Exception as e:
                print(f"加载动量分析结果失败: {e}")
        
        # 添加均线交叉结果选项卡
        if ma_file:
            try:
                df = pd.read_csv(ma_file)
                self.add_result_tab("均线交叉结果", df)
            except Exception as e:
                print(f"加载均线交叉结果失败: {e}")
    
    def add_result_tab(self, title, df):
        # 创建选项卡
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        
        # 创建表格
        columns = list(df.columns)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        
        # 设置列标题
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # 添加数据
        for _, row in df.iterrows():
            values = [str(row[col]) for col in columns]
            tree.insert("", tk.END, values=values)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        # 放置组件
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# 启动应用
root = tk.Tk()
app = ResultViewerApp(root)
root.mainloop()
""")
            
            # 直接使用系统Python运行临时查看器
            subprocess.Popen([SYSTEM_PYTHON, temp_viewer])
            self.status_var.set("结果查看器已启动")
        except Exception as e:
            messagebox.showerror("启动错误", f"启动结果查看器失败: {str(e)}")
            self.status_var.set(f"启动失败: {str(e)}")
    
    def organize_results(self):
        """整理结果文件"""
        self.status_var.set("正在整理结果文件...")
        
        try:
            # 创建必要的目录
            os.makedirs("results/momentum", exist_ok=True)
            os.makedirs("results/ma_cross", exist_ok=True)
            
            # 整理结果文件
            import glob
            import shutil
            
            # 整理动量分析结果
            for file_path in glob.glob("results/*momentum*.csv"):
                target_path = os.path.join("results/momentum", os.path.basename(file_path))
                shutil.copy2(file_path, target_path)
            
            # 整理均线交叉结果
            for file_path in glob.glob("results/*ma*.csv"):
                target_path = os.path.join("results/ma_cross", os.path.basename(file_path))
                shutil.copy2(file_path, target_path)
            
            messagebox.showinfo("整理完成", "结果文件整理完成")
            self.status_var.set("结果文件整理完成")
        except Exception as e:
            messagebox.showerror("整理错误", f"整理结果文件失败: {str(e)}")
            self.status_var.set(f"整理失败: {str(e)}")
    
    def launch_all(self):
        """一键启动全部"""
        self.status_var.set("正在启动所有功能...")
        
        # 先整理结果文件
        self.organize_results()
        
        # 启动分析系统
        self.launch_analysis()
        
        # 等待1秒后启动结果查看器
        self.root.after(1000, self.launch_viewer)
        
        self.status_var.set("所有功能已启动")

def main():
    """主函数"""
    # 如果不是以系统Python运行，则重启
    if sys.executable != SYSTEM_PYTHON:
        print(f"正在切换到系统Python: {SYSTEM_PYTHON}")
        try:
            # 重新以系统Python运行自己
            os.execv(SYSTEM_PYTHON, [SYSTEM_PYTHON, __file__])
        except Exception as e:
            print(f"切换失败: {e}")
    
    root = tk.Tk()
    app = DirectLauncher(root)
    root.mainloop()

if __name__ == "__main__":
    main() 