#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票分析系统统一启动器
集成所有功能模块到一个界面
"""

import os
import sys
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
from datetime import datetime
from pathlib import Path
import importlib.util

# 设置日志
logs_dir = "./logs"
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, f"unified_launcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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

# 定义模块列表
MODULES = [
    {
        "name": "主分析系统",
        "description": "基础股票分析系统，包含动量分析、均线交叉等策略",
        "icon": "assets/main_icon.png",
        "module_file": "stock_analysis_gui.py",
        "class_name": "StockAnalysisGUI",
        "required_files": ["momentum_analysis.py", "ma_cross_strategy.py", "financial_analysis.py"]
    },
    {
        "name": "机器学习分析",
        "description": "基于机器学习的动量分析系统",
        "icon": "assets/ml_icon.png",
        "module_file": "ml_momentum_gui.py",
        "class_name": "MLMomentumGUI",
        "required_files": ["ml_momentum_model.py", "momentum_analysis.py"]
    },
    {
        "name": "热门股票分析",
        "description": "涨停股和热门板块分析系统",
        "icon": "assets/hot_icon.png",
        "module_file": "hot_stock_gui.py",
        "class_name": "HotStockGUI",
        "required_files": ["hot_stock_scanner.py"]
    },
    {
        "name": "增强API系统",
        "description": "使用增强API的股票分析系统",
        "icon": "assets/enhanced_icon.png",
        "module_file": "stock_analysis_gui.py",
        "class_name": "StockAnalysisGUI",
        "required_files": ["enhance_api_reliability.py", "enhanced_momentum_analysis.py"]
    }
]

class UnifiedLauncher:
    """统一启动器类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("股票分析系统 - 统一启动器")
        self.root.geometry("800x600")
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI界面"""
        # 主框架
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(
            self.main_frame, 
            text="股票分析系统", 
            font=("Helvetica", 20, "bold")
        )
        title_label.pack(pady=10)
        
        # 描述
        desc_text = "欢迎使用股票分析系统！请选择要启动的模块:"
        desc_label = ttk.Label(
            self.main_frame, 
            text=desc_text, 
            font=("Helvetica", 12)
        )
        desc_label.pack(pady=5)
        
        # 创建模块选择区域
        self.create_module_selection()
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_module_selection(self):
        """创建模块选择区域"""
        # 创建滚动区域
        module_canvas = tk.Canvas(self.main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            self.main_frame, 
            orient=tk.VERTICAL, 
            command=module_canvas.yview
        )
        
        # 将滚动条和画布连接
        module_canvas.configure(yscrollcommand=scrollbar.set)
        
        # 放置滚动区域
        module_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建容器框架
        self.modules_frame = ttk.Frame(module_canvas)
        canvas_window = module_canvas.create_window(
            (0, 0), 
            window=self.modules_frame, 
            anchor=tk.NW, 
            width=module_canvas.winfo_width()
        )
        
        # 绑定事件以调整滚动区域大小
        def on_frame_configure(event):
            module_canvas.configure(scrollregion=module_canvas.bbox("all"))
            
        def on_canvas_configure(event):
            # 更新内部框架宽度以匹配画布宽度
            module_canvas.itemconfig(canvas_window, width=event.width)
            
        self.modules_frame.bind("<Configure>", on_frame_configure)
        module_canvas.bind("<Configure>", on_canvas_configure)
        
        # 添加模块卡片
        self.create_module_cards()
        
    def create_module_cards(self):
        """创建模块卡片"""
        for i, module in enumerate(MODULES):
            # 创建卡片框架
            card_frame = ttk.Frame(self.modules_frame, style="Card.TFrame")
            card_frame.pack(fill=tk.X, padx=10, pady=5, ipady=10)
            
            # 创建卡片内容
            # 图标和标题在一行
            header_frame = ttk.Frame(card_frame)
            header_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # 尝试加载图标
            icon_label = None
            if module.get("icon") and os.path.exists(module.get("icon")):
                try:
                    # 这里需要PIL库来加载和调整图标大小
                    from PIL import Image, ImageTk
                    img = Image.open(module.get("icon"))
                    img = img.resize((32, 32), Image.LANCZOS)
                    icon = ImageTk.PhotoImage(img)
                    icon_label = ttk.Label(header_frame, image=icon)
                    icon_label.image = icon  # 保持引用
                    icon_label.pack(side=tk.LEFT, padx=(0, 10))
                except Exception as e:
                    logger.warning(f"无法加载图标: {str(e)}")
            
            # 标题
            title_label = ttk.Label(
                header_frame, 
                text=module.get("name", "未命名模块"),
                font=("Helvetica", 14, "bold")
            )
            title_label.pack(side=tk.LEFT)
            
            # 描述
            desc_label = ttk.Label(
                card_frame, 
                text=module.get("description", "无描述"),
                wraplength=600
            )
            desc_label.pack(fill=tk.X, padx=10)
            
            # 状态标签
            status_var = tk.StringVar(value="检查中...")
            status_label = ttk.Label(
                card_frame, 
                textvariable=status_var
            )
            status_label.pack(fill=tk.X, padx=10, pady=5)
            
            # 启动按钮
            launch_button = ttk.Button(
                card_frame, 
                text="启动",
                command=lambda m=module: self.launch_module(m)
            )
            launch_button.pack(pady=5)
            
            # 检查模块状态
            self.check_module_status(module, status_var, launch_button)
            
    def check_module_status(self, module, status_var, button):
        """检查模块状态"""
        # 检查主模块文件
        module_file = module.get("module_file")
        if not os.path.exists(module_file):
            status_var.set(f"错误: 未找到模块文件 {module_file}")
            button.configure(state=tk.DISABLED)
            return
            
        # 检查依赖文件
        missing_files = []
        for req_file in module.get("required_files", []):
            if not os.path.exists(req_file):
                missing_files.append(req_file)
                
        if missing_files:
            status_var.set(f"错误: 缺少依赖文件 {', '.join(missing_files)}")
            button.configure(state=tk.DISABLED)
            return
            
        # 尝试加载模块
        try:
            spec = importlib.util.spec_from_file_location(
                "module", 
                module.get("module_file")
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                
                # 检查类是否存在
                if not hasattr(mod, module.get("class_name", "")):
                    status_var.set(f"错误: 未找到类 {module.get('class_name')}")
                    button.configure(state=tk.DISABLED)
                    return
                    
                status_var.set("可用")
                button.configure(state=tk.NORMAL)
            else:
                status_var.set("错误: 无法加载模块规格")
                button.configure(state=tk.DISABLED)
        except Exception as e:
            status_var.set(f"错误: {str(e)}")
            button.configure(state=tk.DISABLED)
            
    def launch_module(self, module):
        """启动选择的模块"""
        module_file = module.get("module_file")
        module_name = module.get("name")
        
        try:
            self.status_var.set(f"正在启动 {module_name}...")
            logger.info(f"启动模块: {module_name}")
            
            # 创建子进程运行模块
            process = subprocess.Popen([sys.executable, module_file])
            
            # 更新状态
            self.status_var.set(f"{module_name} 已启动")
            
            # 可选：最小化启动器窗口
            self.root.iconify()
            
            # 注意：不等待进程结束，以避免阻塞主界面
            
        except Exception as e:
            error_msg = f"启动 {module_name} 失败: {str(e)}"
            logger.error(error_msg)
            self.status_var.set(error_msg)
            messagebox.showerror("启动错误", error_msg)
            
def main():
    """主函数"""
    try:
        root = tk.Tk()
        
        # 设置应用图标
        if os.path.exists("assets/app_icon.ico"):
            root.iconbitmap("assets/app_icon.ico")
            
        # 尝试设置主题
        try:
            style = ttk.Style()
            style.configure("TFrame", background="#f5f5f5")
            style.configure("Card.TFrame", background="#ffffff", relief="raised")
            style.configure("TButton", padding=6, relief="flat")
        except Exception as e:
            logger.warning(f"设置主题失败: {str(e)}")
            
        app = UnifiedLauncher(root)
        root.mainloop()
        
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}", exc_info=True)
        if 'root' in locals():
            messagebox.showerror("启动失败", f"应用启动失败: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 