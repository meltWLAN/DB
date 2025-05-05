"""
UI布局管理模块
负责界面布局和样式定义
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from typing import Dict, Any, List, Optional, Callable
import os
import json
from pathlib import Path
import sys

# 尝试导入配置
try:
    from ..config.settings import GUI_CONFIG
except ImportError:
    # 回退配置
    GUI_CONFIG = {
        "window_title": "股票推荐系统",
        "window_width": 1280,
        "window_height": 800,
        "default_theme": "default",
        "font_family": "Arial"
    }

class ThemeManager:
    """主题管理器"""
    
    # 预定义主题
    THEMES = {
        "default": {
            "bg_color": "#F0F0F0",
            "fg_color": "#333333",
            "accent_color": "#4A6FE3",
            "accent_color_dark": "#3D5FC4",
            "success_color": "#28A745",
            "warning_color": "#FFC107",
            "danger_color": "#DC3545",
            "font_family": "Arial",
            "heading_font_size": 14,
            "text_font_size": 10,
            "small_font_size": 9,
            "button_bg": "#4A6FE3",
            "button_fg": "white",
            "table_header_bg": "#E5E5E5",
            "table_row_alt_bg": "#F9F9F9",
            "plot_style": "seaborn-v0_8-darkgrid"
        },
        "dark": {
            "bg_color": "#2D2D2D",
            "fg_color": "#E0E0E0",
            "accent_color": "#6C8CD5",
            "accent_color_dark": "#5A7BC2",
            "success_color": "#48C774",
            "warning_color": "#FFDD57",
            "danger_color": "#FF6B6B",
            "font_family": "Arial",
            "heading_font_size": 14,
            "text_font_size": 10,
            "small_font_size": 9,
            "button_bg": "#6C8CD5",
            "button_fg": "white",
            "table_header_bg": "#3D3D3D",
            "table_row_alt_bg": "#353535",
            "plot_style": "dark_background"
        },
        "light": {
            "bg_color": "#FFFFFF",
            "fg_color": "#333333",
            "accent_color": "#007BFF",
            "accent_color_dark": "#0069D9",
            "success_color": "#28A745",
            "warning_color": "#FFC107",
            "danger_color": "#DC3545",
            "font_family": "Arial",
            "heading_font_size": 14,
            "text_font_size": 10,
            "small_font_size": 9,
            "button_bg": "#007BFF",
            "button_fg": "white",
            "table_header_bg": "#F8F9FA",
            "table_row_alt_bg": "#F2F2F2",
            "plot_style": "seaborn-v0_8-whitegrid"
        }
    }
    
    def __init__(self, theme_name: str = "default"):
        """
        初始化主题管理器
        
        Args:
            theme_name: 主题名称，可选：default, dark, light
        """
        self.theme_name = theme_name
        self.theme = self.THEMES.get(theme_name, self.THEMES["default"])
        
    def get_theme(self) -> Dict[str, Any]:
        """获取当前主题"""
        return self.theme
        
    def set_theme(self, theme_name: str) -> None:
        """
        设置主题
        
        Args:
            theme_name: 主题名称
        """
        if theme_name in self.THEMES:
            self.theme_name = theme_name
            self.theme = self.THEMES[theme_name]
        else:
            self.theme_name = "default"
            self.theme = self.THEMES["default"]
            
    def apply_theme_to_widgets(self, root) -> None:
        """
        应用主题到窗口部件
        
        Args:
            root: Tkinter根窗口
        """
        # 设置窗口背景色
        root.configure(bg=self.theme["bg_color"])
        
        # 设置Ttk样式
        style = ttk.Style()
        
        # 设置Ttk主题
        if self.theme_name == "dark":
            try:
                style.theme_use("clam")
            except:
                pass
        else:
            try:
                style.theme_use("clam")
            except:
                pass
                
        # 配置Ttk小部件样式
        style.configure("TFrame", background=self.theme["bg_color"])
        style.configure("TLabel", background=self.theme["bg_color"], foreground=self.theme["fg_color"])
        style.configure("TButton", 
                       background=self.theme["button_bg"], 
                       foreground=self.theme["button_fg"])
        style.map("TButton",
                 background=[("active", self.theme["accent_color_dark"])])
        
        # 配置Treeview（表格）样式
        style.configure("Treeview", 
                       background=self.theme["bg_color"],
                       foreground=self.theme["fg_color"],
                       fieldbackground=self.theme["bg_color"])
        style.configure("Treeview.Heading", 
                       background=self.theme["table_header_bg"],
                       foreground=self.theme["fg_color"])
        style.map("Treeview", 
                 background=[("selected", self.theme["accent_color"])])
                 
        # 设置绘图样式
        plt.style.use(self.theme["plot_style"])
        
class UIFactory:
    """UI工厂类，用于创建一致风格的UI组件"""
    
    def __init__(self, theme_manager: ThemeManager):
        """
        初始化UI工厂
        
        Args:
            theme_manager: 主题管理器
        """
        self.theme_manager = theme_manager
        self.theme = theme_manager.get_theme()
        
    def create_frame(self, parent, **kwargs):
        """创建框架"""
        default_kwargs = {
            "style": "TFrame",
            "padding": 10
        }
        default_kwargs.update(kwargs)
        return ttk.Frame(parent, **default_kwargs)
        
    def create_label(self, parent, text, **kwargs):
        """创建标签"""
        default_kwargs = {
            "style": "TLabel",
            "text": text
        }
        default_kwargs.update(kwargs)
        return ttk.Label(parent, **default_kwargs)
        
    def create_heading(self, parent, text, **kwargs):
        """创建标题"""
        heading_font = (self.theme["font_family"], self.theme["heading_font_size"], "bold")
        label = self.create_label(parent, text, **kwargs)
        label.configure(font=heading_font)
        return label
        
    def create_button(self, parent, text, command=None, **kwargs):
        """创建按钮"""
        default_kwargs = {
            "text": text,
            "command": command,
            "style": "TButton"
        }
        default_kwargs.update(kwargs)
        return ttk.Button(parent, **default_kwargs)
        
    def create_entry(self, parent, **kwargs):
        """创建输入框"""
        default_kwargs = {}
        default_kwargs.update(kwargs)
        
        if self.theme_manager.theme_name == "dark":
            # 为深色主题创建自定义输入框
            entry = tk.Entry(parent, 
                            bg=self.theme["bg_color"],
                            fg=self.theme["fg_color"],
                            insertbackground=self.theme["fg_color"],
                            **default_kwargs)
        else:
            entry = ttk.Entry(parent, **default_kwargs)
            
        return entry
        
    def create_combobox(self, parent, values=None, **kwargs):
        """创建下拉框"""
        default_kwargs = {}
        if values:
            default_kwargs["values"] = values
        default_kwargs.update(kwargs)
        return ttk.Combobox(parent, **default_kwargs)
        
    def create_checkbutton(self, parent, text, **kwargs):
        """创建复选框"""
        default_kwargs = {
            "text": text
        }
        default_kwargs.update(kwargs)
        return ttk.Checkbutton(parent, **default_kwargs)
        
    def create_treeview(self, parent, columns, headings=None, **kwargs):
        """
        创建表格视图
        
        Args:
            parent: 父容器
            columns: 列ID列表
            headings: 列标题列表，如果为None则使用columns
            **kwargs: 其他参数
            
        Returns:
            ttk.Treeview: 表格视图
        """
        default_kwargs = {
            "columns": columns,
            "show": "headings"
        }
        default_kwargs.update(kwargs)
        
        tree = ttk.Treeview(parent, **default_kwargs)
        
        # 设置列标题
        if headings is None:
            headings = columns
            
        for col, heading in zip(columns, headings):
            tree.heading(col, text=heading)
            tree.column(col, anchor="center")
            
        return tree
        
    def create_tabs(self, parent, tab_info, **kwargs):
        """
        创建选项卡
        
        Args:
            parent: 父容器
            tab_info: 选项卡信息列表，每项为(tab_id, tab_text, tab_content)的元组
            **kwargs: 其他参数
            
        Returns:
            ttk.Notebook: 选项卡控件
        """
        default_kwargs = {}
        default_kwargs.update(kwargs)
        
        notebook = ttk.Notebook(parent, **default_kwargs)
        
        for tab_id, tab_text, tab_content in tab_info:
            notebook.add(tab_content, text=tab_text)
            
        return notebook
        
    def create_scrollbar(self, parent, target, **kwargs):
        """
        创建滚动条
        
        Args:
            parent: 父容器
            target: 目标控件
            **kwargs: 其他参数
            
        Returns:
            ttk.Scrollbar: 滚动条
        """
        default_kwargs = {
            "orient": "vertical",
            "command": target.yview
        }
        default_kwargs.update(kwargs)
        
        scrollbar = ttk.Scrollbar(parent, **default_kwargs)
        target.configure(yscrollcommand=scrollbar.set)
        
        return scrollbar
        
    def create_progress_bar(self, parent, **kwargs):
        """创建进度条"""
        default_kwargs = {
            "mode": "determinate"
        }
        default_kwargs.update(kwargs)
        return ttk.Progressbar(parent, **default_kwargs)
        
    def create_matplotlib_canvas(self, parent, figsize=(8, 6), **kwargs):
        """
        创建Matplotlib画布
        
        Args:
            parent: 父容器
            figsize: 图形大小
            **kwargs: 其他参数
            
        Returns:
            tuple: (figure, canvas, toolbar)
        """
        fig = plt.figure(figsize=figsize, **kwargs)
        
        # 创建画布
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas_widget = canvas.get_tk_widget()
        
        # 创建工具栏
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        
        return fig, canvas, toolbar
        
class ResponsiveGridLayout:
    """响应式网格布局管理器"""
    
    def __init__(self, parent, columns=12, padding=10):
        """
        初始化响应式网格布局
        
        Args:
            parent: 父容器
            columns: 总列数
            padding: 内边距
        """
        self.parent = parent
        self.columns = columns
        self.padding = padding
        self.configure_grid()
        
    def configure_grid(self):
        """配置网格权重"""
        for i in range(self.columns):
            self.parent.columnconfigure(i, weight=1)
            
    def place_widget(self, widget, row, column, columnspan=1, rowspan=1, sticky="nsew", padx=None, pady=None):
        """
        放置部件到网格
        
        Args:
            widget: 要放置的部件
            row: 行索引
            column: 列索引
            columnspan: 跨列数
            rowspan: 跨行数
            sticky: 粘贴方式
            padx: 水平内边距
            pady: 垂直内边距
        """
        if padx is None:
            padx = self.padding
        if pady is None:
            pady = self.padding
            
        widget.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, 
                   sticky=sticky, padx=padx, pady=pady)
                   
class ResponsiveFrame(ttk.Frame):
    """响应式框架，可以自适应父容器大小"""
    
    def __init__(self, parent, layout_engine="grid", **kwargs):
        """
        初始化响应式框架
        
        Args:
            parent: 父容器
            layout_engine: 布局引擎，可选：grid, pack
            **kwargs: 其他参数
        """
        super().__init__(parent, **kwargs)
        self.layout_engine = layout_engine
        
        # 配置自适应
        if layout_engine == "grid":
            self.columnconfigure(0, weight=1)
            self.rowconfigure(0, weight=1)
        elif layout_engine == "pack":
            self.pack_configure(fill=tk.BOTH, expand=True)
            
    def add_widget(self, widget, **kwargs):
        """
        添加部件到框架
        
        Args:
            widget: 要添加的部件
            **kwargs: 布局参数
        """
        if self.layout_engine == "grid":
            default_kwargs = {
                "sticky": "nsew",
                "padx": 5,
                "pady": 5
            }
            default_kwargs.update(kwargs)
            widget.grid(**default_kwargs)
        elif self.layout_engine == "pack":
            default_kwargs = {
                "fill": tk.BOTH,
                "expand": True,
                "padx": 5,
                "pady": 5
            }
            default_kwargs.update(kwargs)
            widget.pack(**default_kwargs)
            
class Notification:
    """通知管理器，显示消息通知"""
    
    def __init__(self, parent, theme_manager: ThemeManager):
        """
        初始化通知管理器
        
        Args:
            parent: 父窗口
            theme_manager: 主题管理器
        """
        self.parent = parent
        self.theme = theme_manager.get_theme()
        self.notifications = []
        self.notification_var = tk.StringVar()
        
        # 创建通知标签
        self.notification_label = ttk.Label(
            parent,
            textvariable=self.notification_var,
            background=self.theme["bg_color"],
            foreground=self.theme["fg_color"],
            padding=(10, 5)
        )
        
    def show(self, message, type="info", duration=3000):
        """
        显示通知
        
        Args:
            message: 消息内容
            type: 消息类型，可选：info, success, warning, error
            duration: 显示时长（毫秒）
        """
        # 设置颜色
        if type == "success":
            color = self.theme["success_color"]
        elif type == "warning":
            color = self.theme["warning_color"]
        elif type == "error":
            color = self.theme["danger_color"]
        else:  # info
            color = self.theme["accent_color"]
            
        # 更新通知标签
        self.notification_label.configure(foreground=color)
        self.notification_var.set(message)
        
        # 显示通知
        self.notification_label.place(relx=0.5, rely=0.95, anchor="center")
        
        # 安排自动隐藏
        self.parent.after(duration, self.hide)
        
    def hide(self):
        """隐藏通知"""
        self.notification_label.place_forget()
        
class StatusBar(ttk.Frame):
    """状态栏"""
    
    def __init__(self, parent, theme_manager: ThemeManager):
        """
        初始化状态栏
        
        Args:
            parent: 父容器
            theme_manager: 主题管理器
        """
        super().__init__(parent)
        self.theme = theme_manager.get_theme()
        
        # 配置网格
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        
        # 创建状态文本标签
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(
            self,
            textvariable=self.status_var,
            padding=(5, 2)
        )
        self.status_label.grid(row=0, column=0, sticky="w")
        
        # 创建进度标签
        self.progress_var = tk.StringVar()
        self.progress_label = ttk.Label(
            self,
            textvariable=self.progress_var,
            padding=(5, 2)
        )
        self.progress_label.grid(row=0, column=1, sticky="e")
        
    def set_status(self, message):
        """设置状态消息"""
        self.status_var.set(message)
        
    def set_progress(self, message):
        """设置进度消息"""
        self.progress_var.set(message)
        
class DialogFactory:
    """对话框工厂，用于创建各种对话框"""
    
    def __init__(self, parent, theme_manager: ThemeManager):
        """
        初始化对话框工厂
        
        Args:
            parent: 父窗口
            theme_manager: 主题管理器
        """
        self.parent = parent
        self.theme_manager = theme_manager
        self.theme = theme_manager.get_theme()
        self.ui_factory = UIFactory(theme_manager)
        
    def _create_dialog_base(self, title, width=400, height=300):
        """创建对话框基础窗口"""
        dialog = tk.Toplevel(self.parent)
        dialog.title(title)
        dialog.geometry(f"{width}x{height}")
        dialog.configure(bg=self.theme["bg_color"])
        dialog.resizable(False, False)
        dialog.transient(self.parent)
        dialog.grab_set()
        
        return dialog
        
    def create_message_dialog(self, title, message, type="info"):
        """
        创建消息对话框
        
        Args:
            title: 标题
            message: 消息内容
            type: 消息类型，可选：info, success, warning, error
            
        Returns:
            tk.Toplevel: 对话框窗口
        """
        dialog = self._create_dialog_base(title, width=300, height=150)
        
        # 创建内容框架
        frame = self.ui_factory.create_frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 设置图标和颜色
        if type == "success":
            icon = "✓"
            color = self.theme["success_color"]
        elif type == "warning":
            icon = "⚠"
            color = self.theme["warning_color"]
        elif type == "error":
            icon = "✗"
            color = self.theme["danger_color"]
        else:  # info
            icon = "ℹ"
            color = self.theme["accent_color"]
            
        # 创建图标标签
        icon_label = tk.Label(
            frame, 
            text=icon, 
            font=(self.theme["font_family"], 24),
            fg=color,
            bg=self.theme["bg_color"]
        )
        icon_label.pack(pady=(0, 10))
        
        # 创建消息标签
        message_label = self.ui_factory.create_label(frame, message)
        message_label.pack(pady=10)
        
        # 创建按钮
        ok_button = self.ui_factory.create_button(frame, "确定", command=dialog.destroy)
        ok_button.pack(pady=10)
        
        return dialog
        
    def create_input_dialog(self, title, prompt, default="", callback=None):
        """
        创建输入对话框
        
        Args:
            title: 标题
            prompt: 提示文本
            default: 默认值
            callback: 回调函数，接收用户输入作为参数
            
        Returns:
            tk.Toplevel: 对话框窗口
        """
        dialog = self._create_dialog_base(title, width=350, height=180)
        
        # 创建内容框架
        frame = self.ui_factory.create_frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 创建提示标签
        prompt_label = self.ui_factory.create_label(frame, prompt)
        prompt_label.pack(anchor="w", pady=(0, 5))
        
        # 创建输入框
        entry_var = tk.StringVar(value=default)
        entry = self.ui_factory.create_entry(frame, textvariable=entry_var, width=40)
        entry.pack(fill="x", pady=5)
        entry.focus_set()
        
        # 创建按钮框架
        button_frame = self.ui_factory.create_frame(frame)
        button_frame.pack(fill="x", pady=(10, 0))
        
        # 确定按钮
        def on_ok():
            if callback:
                callback(entry_var.get())
            dialog.destroy()
            
        ok_button = self.ui_factory.create_button(button_frame, "确定", command=on_ok)
        ok_button.pack(side="right", padx=5)
        
        # 取消按钮
        cancel_button = self.ui_factory.create_button(button_frame, "取消", command=dialog.destroy)
        cancel_button.pack(side="right", padx=5)
        
        # 绑定回车键
        entry.bind("<Return>", lambda event: on_ok())
        
        return dialog
        
    def create_confirm_dialog(self, title, message, callback=None):
        """
        创建确认对话框
        
        Args:
            title: 标题
            message: 消息内容
            callback: 回调函数，接收布尔值参数表示用户是否确认
            
        Returns:
            tk.Toplevel: 对话框窗口
        """
        dialog = self._create_dialog_base(title, width=350, height=180)
        
        # 创建内容框架
        frame = self.ui_factory.create_frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 创建消息标签
        message_label = self.ui_factory.create_label(frame, message)
        message_label.pack(pady=20)
        
        # 创建按钮框架
        button_frame = self.ui_factory.create_frame(frame)
        button_frame.pack(fill="x", pady=(10, 0))
        
        # 确定按钮
        def on_yes():
            if callback:
                callback(True)
            dialog.destroy()
            
        yes_button = self.ui_factory.create_button(button_frame, "确定", command=on_yes)
        yes_button.pack(side="right", padx=5)
        
        # 取消按钮
        def on_no():
            if callback:
                callback(False)
            dialog.destroy()
            
        no_button = self.ui_factory.create_button(button_frame, "取消", command=on_no)
        no_button.pack(side="right", padx=5)
        
        return dialog 