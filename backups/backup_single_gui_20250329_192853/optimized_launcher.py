#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
优化版启动器 - 提供更快的启动速度和更好的用户体验
"""

import os
import sys
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import platform
from datetime import datetime

# 确保必要的目录存在
for dirname in ["data", "logs", "results", "charts"]:
    os.makedirs(dirname, exist_ok=True)

# 配置日志
log_file = os.path.join("logs", f"launcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file)]
)

class OptimizedLauncher:
    """优化版启动器类"""
    
    def __init__(self, root):
        """初始化启动器"""
        self.root = root
        self.root.title("股票分析系统 - 优化版")
        self.root.geometry("600x400")
        
        # 设置主题
        self.setup_theme()
        
        # 创建界面元素
        self.create_widgets()
        
        # 检查依赖
        self.check_dependencies()
    
    def setup_theme(self):
        """设置主题"""
        style = ttk.Style()
        
        # 检测系统
        if platform.system() == "Darwin":  # macOS
            style.theme_use("aqua")
        elif platform.system() == "Windows":
            style.theme_use("vista")
        else:
            style.theme_use("clam")
        
        # 自定义按钮样式
        style.configure(
            "TButton",
            font=("Helvetica", 12),
            padding=6
        )
        style.configure(
            "Header.TLabel",
            font=("Helvetica", 24, "bold")
        )
        style.configure(
            "SubHeader.TLabel",
            font=("Helvetica", 12)
        )
    
    def create_widgets(self):
        """创建界面元素"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title = ttk.Label(
            main_frame,
            text="股票分析系统",
            style="Header.TLabel"
        )
        title.pack(pady=(0, 20))
        
        # 子标题
        subtitle = ttk.Label(
            main_frame,
            text="选择一个功能模块启动",
            style="SubHeader.TLabel"
        )
        subtitle.pack(pady=(0, 30))
        
        # 模块按钮
        modules_frame = ttk.Frame(main_frame)
        modules_frame.pack(fill=tk.BOTH, expand=True)
        
        modules = [
            ("完整分析界面", "stock_analysis_gui.py", "集成了所有分析功能的完整界面"),
            ("简易分析界面", "simple_gui.py", "简化版分析界面，适合新用户"),
            ("无界面分析", "headless_gui.py", "命令行分析工具，适合高级用户")
        ]
        
        for i, (name, script, desc) in enumerate(modules):
            module_frame = ttk.Frame(modules_frame, padding=5)
            module_frame.pack(fill=tk.X, pady=10)
            
            button = ttk.Button(
                module_frame,
                text=name,
                command=lambda s=script: self.launch_module(s)
            )
            button.pack(side=tk.LEFT, padx=(0, 10))
            
            label = ttk.Label(module_frame, text=desc)
            label.pack(side=tk.LEFT, fill=tk.X)
        
        # 底部状态栏
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar()
        self.status_var.set("系统就绪")
        
        status_bar = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 系统信息
        info_label = ttk.Label(
            status_frame,
            text=f"Python {platform.python_version()}",
            relief=tk.SUNKEN,
            anchor=tk.E
        )
        info_label.pack(side=tk.RIGHT)
    
    def check_dependencies(self):
        """检查依赖包"""
        required_packages = {
            "numpy": "数据计算",
            "pandas": "数据处理",
            "matplotlib": "图表显示"
        }
        
        missing = []
        
        for package, desc in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing.append(f"{package} ({desc})")
        
        if missing:
            message = "缺少以下依赖包:\n\n" + "\n".join(missing)
            message += "\n\n建议使用以下命令安装:\npip install " + " ".join([p.split()[0] for p in missing])
            
            self.status_var.set("缺少依赖")
            messagebox.warning("缺少依赖", message)
    
    def launch_module(self, script):
        """启动模块"""
        self.status_var.set(f"正在启动 {script}...")
        
        def run():
            try:
                # 准备环境变量
                env = os.environ.copy()
                
                # 移除可能干扰的环境变量
                for var in list(env.keys()):
                    if var.startswith("PYENV"):
                        del env[var]
                
                # 设置Python路径
                env["PYTHONPATH"] = os.getcwd()
                
                # 启动进程
                python_exe = sys.executable
                cmd = [python_exe, script]
                
                subprocess.Popen(cmd, env=env)
                self.status_var.set(f"已启动 {script}")
            except Exception as e:
                self.status_var.set(f"启动失败: {str(e)}")
                messagebox.showerror("启动错误", f"启动 {script} 失败:\n{str(e)}")
        
        threading.Thread(target=run, daemon=True).start()

def main():
    """主函数"""
    # 创建主窗口
    root = tk.Tk()
    
    # 设置图标
    try:
        icon_path = os.path.join("assets", "icon.png")
        if os.path.exists(icon_path):
            img = tk.PhotoImage(file=icon_path)
            root.iconphoto(True, img)
    except Exception as e:
        logging.warning(f"无法加载图标: {str(e)}")
    
    # 创建启动器
    app = OptimizedLauncher(root)
    
    # 主循环
    root.mainloop()

if __name__ == "__main__":
    main()
