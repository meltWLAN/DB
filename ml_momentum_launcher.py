#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import tkinter as tk
import resource
from ml_momentum_gui import MLMomentumGUI

# 确保自定义模块可以被导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "ml_momentum_launcher.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ml_momentum_launcher")

# 优化内存和CPU使用
def optimize_resources():
    """
    优化系统资源使用
    """
    try:
        # 尝试限制内存使用（在某些系统上可能不工作）
        # Linux/Mac
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        # 设置内存限制为4GB或系统允许的最大值
        memory_limit = min(4 * 1024 * 1024 * 1024, hard) if hard != resource.RLIM_INFINITY else 4 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, hard))
        logger.info(f"设置内存限制为: {memory_limit / (1024*1024*1024):.2f}GB")
    except Exception as e:
        logger.warning(f"无法设置资源限制: {str(e)}")

def main():
    """
    启动ML动量分析应用
    """
    try:
        logger.info("启动ML动量分析应用")
        
        # 优化资源使用
        optimize_resources()
        
        # 创建主窗口
        root = tk.Tk()
        app = MLMomentumGUI(root)
        
        # 设置默认更保守的样本大小
        app.sample_size_var.set("30")
        
        # 运行应用
        root.mainloop()
        
        logger.info("ML动量分析应用已关闭")
        
    except Exception as e:
        logger.error(f"启动ML动量分析应用时出错: {str(e)}")
        print(f"启动失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 