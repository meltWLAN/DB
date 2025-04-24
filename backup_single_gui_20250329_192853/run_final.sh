#!/bin/bash

# 完全绕过pyenv的干扰
echo "=============================================="
echo "股票分析系统 - 终极启动器"
echo "=============================================="

# 创建临时Python脚本
TMP_FILE=$(mktemp)

# 写入内容到临时文件
cat > $TMP_FILE << 'EOF'
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import subprocess
import platform
import threading

# 创建主窗口
root = tk.Tk()
root.title('股票分析系统')
root.geometry('600x400')

# 设置标题
title = ttk.Label(root, text='股票分析系统', font=('Helvetica', 24, 'bold'))
title.pack(pady=20)

# 状态变量
status_var = tk.StringVar()
status_var.set('就绪')

def run_script(script_name):
    """运行指定的脚本"""
    status_var.set(f'正在启动 {script_name}...')
    
    def run():
        try:
            env = os.environ.copy()
            
            # 清理环境变量
            if 'PYENV_ROOT' in env:
                del env['PYENV_ROOT']
            if 'PYENV_VERSION' in env:
                del env['PYENV_VERSION']
                
            # 设置干净的PATH
            path_parts = env['PATH'].split(':')
            clean_path = ':'.join([p for p in path_parts if 'pyenv' not in p])
            env['PATH'] = '/usr/bin:/usr/local/bin:/bin:/sbin:' + clean_path
            
            # 启动进程
            subprocess.Popen(['/usr/bin/python3', script_name], env=env)
            status_var.set(f'已启动 {script_name}')
        except Exception as e:
            status_var.set(f'启动失败: {str(e)}')
            messagebox.showerror('错误', f'启动失败: {str(e)}')
    
    # 在线程中运行
    threading.Thread(target=run, daemon=True).start()

# 创建按钮框架
button_frame = ttk.Frame(root)
button_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# 创建按钮
programs = [
    ('完整分析界面', 'stock_analysis_gui.py', '启动完整的股票分析界面，包含所有功能'),
    ('简易分析界面', 'simple_gui.py', '启动简化版分析界面，适合新手使用'),
    ('无界面分析工具', 'headless_gui.py', '启动命令行分析工具，适合高级用户')
]

for text, script, desc in programs:
    frame = ttk.Frame(button_frame)
    frame.pack(fill=tk.X, pady=10)
    
    btn = ttk.Button(
        frame, 
        text=text,
        command=lambda s=script: run_script(s)
    )
    btn.pack(side=tk.LEFT, padx=10)
    
    ttk.Label(frame, text=desc).pack(side=tk.LEFT, padx=10)

# 状态栏
status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# 底部信息
ttk.Label(
    root, 
    text=f'Python版本: {platform.python_version()} | 系统: {platform.system()} {platform.release()}',
    font=('Helvetica', 8)
).pack(side=tk.BOTTOM, fill=tk.X, pady=5)

# 主循环
root.mainloop()
EOF

echo "正在启动..."

# 直接运行临时文件，绕过所有拦截
/usr/bin/python3 $TMP_FILE

# 清理临时文件
rm -f $TMP_FILE 