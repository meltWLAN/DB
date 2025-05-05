#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票分析系统 - 主程序入口
整合了图形界面和命令行功能
"""
import os
import sys
import logging
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime
import threading
import platform
# 确保必要的目录存在
for dirname in ["data", "logs", "results", "charts"]:
    os.makedirs(dirname, exist_ok=True)
# 配置日志
log_file = os.path.join("logs", f"main_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MainGUI")
class MainGUI:
    """股票分析系统主界面类"""
    def __init__(self, root):
        """初始化主界面"""
        self.root = roo
        self.root.title("股票分析系统")
        self.root.geometry("800x600")
        # 设置图标
        try:
            if os.path.exists("assets/icon.png"):
                icon = tk.PhotoImage(file="assets/icon.png")
                self.root.iconphoto(True, icon)
        except Exception as e:
            logger.error(f"加载图标出错: {str(e)}")
        # 创建UI组件
        self.create_widgets()
    def create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # 标题
        title_label = ttk.Label(
            main_frame,
            text="股票分析系统",
            font=("Helvetica", 24, "bold")
        )
        title_label.pack(pady=20)
        # 系统信息
        info_frame = ttk.LabelFrame(main_frame, text="系统信息")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        # 显示Python版本
        python_info = f"Python版本: {platform.python_version()}"
        ttk.Label(info_frame, text=python_info).pack(anchor=tk.W, padx=10, pady=5)
        # 显示系统版本
        system_info = f"系统: {platform.system()} {platform.release()}"
        ttk.Label(info_frame, text=system_info).pack(anchor=tk.W, padx=10, pady=5)
        # 显示工作目录
        work_dir = f"工作目录: {os.getcwd()}"
        ttk.Label(info_frame, text=work_dir).pack(anchor=tk.W, padx=10, pady=5)
        # 功能按钮区域
        button_frame = ttk.LabelFrame(main_frame, text="选择功能")
        button_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # 按钮样式
        style = ttk.Style()
        style.configure("Big.TButton", font=("Helvetica", 12))
        # 创建主功能按钮
        buttons = [
            ("完整分析界面", "stock_analysis_gui.py", "启动完整的股票分析界面，包含动量分析、均线交叉和组合策略"),
            ("简易分析界面", "simple_gui.py", "启动简易的股票分析界面，适合初学者使用"),
            ("无界面分析工具", "headless_gui.py", "启动无界面分析工具，适合没有图形界面环境"),
            ("生成样例数据", self.generate_sample_data, "生成示例股票数据用于测试"),
            ("退出", self.root.quit, "退出程序")
        ]
        # 添加按钮
        for i, (text, command, desc) in enumerate(buttons):
            frame = ttk.Frame(button_frame)
            frame.pack(fill=tk.X, padx=20, pady=10)
            btn = ttk.Button(
                frame,
                text=text,
                style="Big.TButton",
                command=lambda cmd=command: self.run_command(cmd) if isinstance(cmd, str) else cmd()
            )
            btn.pack(side=tk.LEFT, padx=10)
            # 添加描述标签
            ttk.Label(frame, text=desc).pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("系统就绪")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    def run_command(self, script_name):
        """运行指定的脚本"""
        if not os.path.exists(script_name):
            messagebox.showerror("错误", f"找不到脚本文件: {script_name}")
            return
        # 更新状态
        self.status_var.set(f"正在启动 {script_name}...")
        # 创建启动线程
        def run_script():
            try:
                # 判断当前Python解释器
                if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                    # 虚拟环境中
                    python_exe = sys.executable
                else:
                    # 尝试使用系统Python
                    if platform.system() == "Darwin" or platform.system() == "Linux":
                        python_exe = "/usr/bin/python3"
                    else:
                        python_exe = "python"
                # 设置环境变量
                env = os.environ.copy()
                env["PYTHONPATH"] = os.getcwd()
                # 移除会导致问题的环境变量
                if "PYTHONHOME" in env:
                    del env["PYTHONHOME"]
                # 设置执行权限(Unix系统)
                try:
                    if platform.system() != "Windows":
                        os.chmod(script_name, 0o755)
                except:
                    pass
                # 启动进程
                subprocess.Popen([python_exe, script_name], env=env)
                self.status_var.set(f"已启动 {script_name}")
            except Exception as e:
                logger.error(f"启动脚本出错: {str(e)}")
                self.status_var.set(f"启动失败: {str(e)}")
                messagebox.showerror("启动错误", f"启动脚本出错: {str(e)}")
        # 启动线程
        threading.Thread(target=run_script, daemon=True).start()
    def generate_sample_data(self):
        """生成样例数据"""
        # 询问生成多少只股票数据
        try:
            from headless_gui import StockAnalyzer
            analyzer = StockAnalyzer()
            # 生成10只股票的示例数据
            analyzer.generate_sample_data(10)
            messagebox.showinfo("成功", "已成功生成10只股票的样例数据")
            self.status_var.set("样例数据生成完成")
        except Exception as e:
            logger.error(f"生成样例数据出错: {str(e)}")
            messagebox.showerror("错误", f"生成样例数据出错: {str(e)}")
            self.status_var.set("样例数据生成失败")
def check_packages():
    """检查必要的包是否已安装"""
    required_packages = {
        "numpy": "数值计算",
        "pandas": "数据分析",
        "matplotlib": "数据可视化"
    }
    missing_packages = []
    for package, desc in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(f"{package} ({desc})")
    if missing_packages:
        logger.warning(f"缺少以下包: {', '.join(missing_packages)}")
        msg = "缺少以下必要的包:\n\n"
        for pkg in missing_packages:
            msg += f"- {pkg}\n"
        msg += "\n是否继续启动程序？"
        return messagebox.askyesno("警告", msg)
    return True
def main():
    """主函数"""
    # 打印启动信息
    logger.info("系统启动")
    print("=" * 60)
    print("股票分析系统 - 主程序")
    print("=" * 60)
    print(f"Python版本: {platform.python_version()}")
    print(f"工作目录: {os.getcwd()}")
    print("-" * 60)
    # 创建GUI
    root = tk.Tk()
    # 检查包
    if not check_packages():
        return
    app = MainGUI(root)
    # 设置窗口关闭事件
    def on_closing():
        if messagebox.askokcancel("退出", "确定要退出程序吗？"):
            logger.info("系统正常关闭")
            root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    # 启动主循环
    try:
        root.mainloop()
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
        print(f"程序运行出错: {str(e)}")
if __name__ == "__main__":
    main()