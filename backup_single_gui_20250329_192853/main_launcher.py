#!/usr/bin/env python3
"""
股票分析系统主启动器
提供更好的异常处理和界面反馈
"""
import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import logging
from pathlib import Path
from datetime import datetime

# 配置日志
logs_dir = "./logs"
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, f"launcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

class SystemLauncher:
    """股票分析系统启动器"""
    def __init__(self, root):
        self.root = root
        self.root.title("股票分析系统启动器")
        self.root.geometry("500x400")
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="股票分析系统", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)
        
        # 描述
        desc_text = "这是一个集成了动量分析和均线交叉策略的股票分析系统，\n提供了可视化的分析工具和回测功能。"
        desc_label = ttk.Label(main_frame, text=desc_text, justify=tk.CENTER)
        desc_label.pack(pady=10)
        
        # 状态框架
        status_frame = ttk.LabelFrame(main_frame, text="系统状态")
        status_frame.pack(fill=tk.X, pady=10)
        
        # 检查基本组件
        components = [
            ("主界面", "stock_analysis_gui.py"),
            ("动量分析模块", "momentum_analysis.py"),
            ("均线交叉策略", "ma_cross_strategy.py"),
            ("GUI控制器", "gui_controller.py")
        ]
        
        self.status_labels = {}
        for i, (name, file) in enumerate(components):
            ttk.Label(status_frame, text=f"{name}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            status_var = tk.StringVar(value="检查中...")
            status_label = ttk.Label(status_frame, textvariable=status_var)
            status_label.grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            self.status_labels[file] = status_var
            
        # 检查组件状态
        self.check_components()
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        # 启动按钮
        launch_button = ttk.Button(button_frame, text="启动系统", command=self.launch_system)
        launch_button.pack(side=tk.LEFT, padx=10)
        
        # 退出按钮
        exit_button = ttk.Button(button_frame, text="退出", command=self.root.destroy)
        exit_button.pack(side=tk.LEFT, padx=10)
        
        # 日志框架
        log_frame = ttk.LabelFrame(main_frame, text="启动日志")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 日志文本框
        self.log_text = tk.Text(log_frame, height=6, width=50)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # 设置日志处理器
        self.setup_log_handler()
        
    def setup_log_handler(self):
        """设置日志处理器"""
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                logging.Handler.__init__(self)
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                def append():
                    self.text_widget.configure(state='normal')
                    self.text_widget.insert(tk.END, msg + '\n')
                    self.text_widget.see(tk.END)
                    self.text_widget.configure(state='disabled')
                self.text_widget.after(0, append)
                
        handler = TextHandler(self.log_text)
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(handler)
        
    def check_components(self):
        """检查系统组件是否存在"""
        for file, status_var in self.status_labels.items():
            if os.path.exists(file):
                status_var.set("✓ 已找到")
            else:
                status_var.set("✗ 未找到")
                logger.error(f"缺少组件: {file}")
    
    def check_dependencies(self):
        """检查系统依赖"""
        try:
            logger.info("检查系统依赖...")
            import pandas
            import numpy
            import matplotlib
            logger.info("核心依赖检查通过")
            return True
        except ImportError as e:
            logger.error(f"依赖检查失败: {str(e)}")
            messagebox.showerror("依赖错误", f"缺少必要的依赖: {str(e)}\n请运行 'pip install -r requirements.txt' 安装依赖")
            return False
    
    def launch_system(self):
        """启动主系统"""
        # 检查组件
        missing_components = []
        for file, status_var in self.status_labels.items():
            if not os.path.exists(file):
                missing_components.append(file)
        
        if missing_components:
            error_msg = f"缺少必要组件: {', '.join(missing_components)}\n请确保所有组件都存在后再启动系统"
            logger.error(error_msg)
            messagebox.showerror("组件错误", error_msg)
            return
        
        # 检查依赖
        if not self.check_dependencies():
            return
        
        # 启动主系统
        try:
            logger.info("正在启动股票分析系统...")
            os.environ["SYSTEM_VERSION_COMPAT"] = "1"  # 设置macOS兼容性环境变量
            
            # 使用当前Python解释器启动系统
            python_executable = sys.executable
            command = [python_executable, "stock_analysis_gui.py"]
            
            # 在新进程中启动主系统
            logger.info(f"执行命令: {' '.join(command)}")
            process = subprocess.Popen(command)
            
            # 关闭启动器
            logger.info("系统已启动，关闭启动器")
            self.root.destroy()
        except Exception as e:
            error_msg = f"启动系统时出错: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("启动错误", error_msg)

def main():
    """主函数"""
    logger.info("启动股票分析系统启动器")
    
    root = tk.Tk()
    app = SystemLauncher(root)
    
    # 设置窗口图标(如果存在)
    icon_path = os.path.join("assets", "icon.png")
    if os.path.exists(icon_path):
        try:
            img = tk.PhotoImage(file=icon_path)
            root.iconphoto(True, img)
        except Exception as e:
            logger.warning(f"无法设置窗口图标: {str(e)}")
    
    root.mainloop()

if __name__ == "__main__":
    main() 