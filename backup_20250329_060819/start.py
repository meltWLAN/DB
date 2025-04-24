#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票分析系统启动脚本
"""

import os
import sys
import subprocess
import platform
import logging
from datetime import datetime

# 确保日志目录存在
os.makedirs("logs", exist_ok=True)

# 配置日志
log_file = os.path.join("logs", f"start_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("启动器")

def get_system_python():
    """获取系统Python路径"""
    if platform.system() == "Windows":
        return "python"
    elif platform.system() == "Darwin":  # macOS
        return "/usr/bin/python3"
    else:  # Linux and others
        return "/usr/bin/python3"

def main():
    """主函数"""
    print("=" * 60)
    print("股票分析系统启动器")
    print("=" * 60)
    
    try:
        # 获取系统Python路径
        python_exe = get_system_python()
        logger.info(f"使用Python解释器: {python_exe}")
        
        # 设置环境变量
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        
        # 移除可能导致问题的环境变量
        for var in ["PYTHONHOME", "PYENV_VERSION"]:
            if var in env:
                del env[var]
        
        # 设置启动文件路径
        main_script = "main_gui.py"
        
        # 设置执行权限(Unix系统)
        if platform.system() != "Windows":
            try:
                os.chmod(main_script, 0o755)
                logger.info(f"已设置{main_script}的执行权限")
            except Exception as e:
                logger.warning(f"设置执行权限失败: {str(e)}")
        
        # 启动主程序
        logger.info(f"正在启动主程序: {main_script}")
        print(f"正在启动股票分析系统...")
        subprocess.run([python_exe, main_script], env=env)
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("程序被用户中断")
        print("\n程序被用户中断")
        return 1
    except Exception as e:
        logger.error(f"启动失败: {str(e)}", exc_info=True)
        print(f"启动失败: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 