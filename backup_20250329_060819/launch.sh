#!/bin/bash

# 完全屏蔽pyenv
export PATH=$(echo "$PATH" | sed 's|/Users/mac/.pyenv/shims:||g')
export PATH=$(echo "$PATH" | sed 's|:/Users/mac/.pyenv/shims||g')
unset PYENV_VERSION
unset PYENV_ROOT
unset PYENV_DIR
unset PYENV_HOOK_PATH
unset PYENV_SHELL

echo "=============================================="
echo "股票分析系统启动器"
echo "=============================================="
echo "正在绕过pyenv直接启动..."

# 直接执行系统Python并进入主界面
/usr/bin/python3 -c "
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import subprocess
import platform
import logging
from datetime import datetime
import threading

# 确保必要的目录存在
for dirname in ['data', 'logs', 'results', 'charts']:
    os.makedirs(dirname, exist_ok=True)

# 配置日志
log_file = os.path.join('logs', f'launcher_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file)]
)

def run_module(module_name):
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        # 移除pyenv相关的环境变量
        for var in list(env.keys()):
            if 'PYENV' in var:
                del env[var]
        
        # 启动进程
        subprocess.Popen(['/usr/bin/python3', module_name], env=env)
        logging.info(f'启动模块: {module_name}')
    except Exception as e:
        logging.error(f'启动失败: {str(e)}')
        messagebox.showerror('启动错误', f'启动失败: {str(e)}')

# 创建主窗口
root = tk.Tk()
root.title('股票分析系统启动器')
root.geometry('500x300')

# 设置标题
title_label = ttk.Label(
    root, 
    text='股票分析系统',
    font=('Helvetica', 20, 'bold')
)
title_label.pack(pady=20)

# 创建按钮框架
button_frame = ttk.Frame(root)
button_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

# 创建功能按钮
buttons = [
    ('完整分析界面', 'stock_analysis_gui.py', '启动完整的股票分析界面'),
    ('简易分析界面', 'simple_gui.py', '启动简化版分析界面'),
    ('无界面分析工具', 'headless_gui.py', '启动命令行分析工具')
]

# 添加按钮
for i, (text, module, desc) in enumerate(buttons):
    frame = ttk.Frame(button_frame)
    frame.pack(fill=tk.X, pady=10)
    
    btn = ttk.Button(
        frame,
        text=text,
        command=lambda m=module: run_module(m)
    )
    btn.pack(side=tk.LEFT, padx=10)
    
    ttk.Label(frame, text=desc).pack(side=tk.LEFT, padx=10)

# 添加退出按钮
exit_btn = ttk.Button(root, text='退出', command=root.quit)
exit_btn.pack(pady=20)

# 显示信息
ttk.Label(
    root, 
    text=f'系统Python版本: {platform.python_version()}',
    font=('Helvetica', 8)
).pack(side=tk.BOTTOM, fill=tk.X)

# 运行主循环
root.mainloop()
" 