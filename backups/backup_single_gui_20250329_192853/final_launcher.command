#!/bin/bash

# 完全绕过pyenv干扰的启动器
cd "$(dirname "$0")"
clear

# 在新的shell中执行
cat > /tmp/run_gui_$$_.py << 'EOF'
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import subprocess
import platform

# 创建GUI
root = tk.Tk()
root.title("股票分析系统启动器")
root.geometry("500x350")

# 设置标题
title = ttk.Label(
    root, 
    text="股票分析系统", 
    font=("Helvetica", 24, "bold")
)
title.pack(pady=20)

# 状态变量
status_var = tk.StringVar()
status_var.set("就绪")

def run_app(script_name):
    """启动应用程序"""
    status_var.set(f"正在启动 {script_name}...")
    
    try:
        # 创建新的环境变量
        env = {}
        env["PATH"] = "/usr/bin:/usr/local/bin:/bin:/sbin"
        env["PYTHONPATH"] = os.getcwd()
        
        # 启动进程
        cmd = f"/usr/bin/env python3 {script_name}"
        subprocess.Popen(cmd, shell=True, env=env)
        status_var.set(f"已启动 {script_name}")
    except Exception as e:
        status_var.set(f"启动失败: {str(e)}")
        messagebox.showerror("错误", f"启动失败: {str(e)}")

# 创建按钮框架
button_frame = ttk.Frame(root)
button_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# 应用列表
apps = [
    ("完整分析界面", "stock_analysis_gui.py", "启动完整的股票分析界面"),
    ("简易分析界面", "simple_gui.py", "启动简易的股票分析界面"),
    ("无界面分析", "headless_gui.py", "启动命令行分析工具")
]

# 添加应用按钮
for text, script, desc in apps:
    frame = ttk.Frame(button_frame)
    frame.pack(fill=tk.X, pady=10)
    
    button = ttk.Button(
        frame,
        text=text,
        command=lambda s=script: run_app(s)
    )
    button.pack(side=tk.LEFT, padx=10)
    
    label = ttk.Label(frame, text=desc)
    label.pack(side=tk.LEFT, padx=10)

# 状态栏
status_bar = ttk.Label(
    root, 
    textvariable=status_var,
    relief=tk.SUNKEN,
    anchor=tk.W
)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# 版本信息
version_label = ttk.Label(
    root,
    text=f"系统信息: Python {sys.version.split()[0]} | {platform.system()} {platform.release()}",
    font=("Helvetica", 8)
)
version_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

# 主循环
root.mainloop()
EOF

# 执行临时脚本（在完全隔离的环境中）
echo "正在启动股票分析系统..."
echo "请稍候..."
env -i PATH="/usr/bin:/usr/local/bin:/bin:/sbin" HOME="$HOME" /usr/bin/python3 /tmp/run_gui_$$_.py
rm -f /tmp/run_gui_$$_.py 