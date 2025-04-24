#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清理系统 - 删除多余的启动文件，只保留主要接口文件
保留文件:
- main_gui.py (主入口)
- stock_analysis_gui.py (完整分析界面)
- simple_gui.py (简易分析界面)
- headless_gui.py (无界面分析工具)
- 必要的数据和功能文件

删除文件:
- 所有其他启动器和入口脚本
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime

# 配置日志
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", f"clean_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("CleanSystem")

# 要保留的文件列表
KEEP_FILES = [
    "main_gui.py",
    "stock_analysis_gui.py",
    "simple_gui.py",
    "headless_gui.py",
    "README.md",
    "requirements.txt",
    "gui_controller.py",
    "ma_cross_strategy.py",
    "momentum_analysis.py"
]

# 要保留的目录列表
KEEP_DIRS = [
    "data",
    "logs",
    "results",
    "charts",
    "src",
    "assets"
]

# 要删除的启动文件列表(模式匹配)
REMOVE_PATTERNS = [
    "start_*.py",
    "start_*.sh",
    "run_*.py",
    "run_*.sh",
    "launcher*.py",
    "launcher*.sh",
    "*_launcher.py",
    "*_launcher.sh",
    "*_start.py",
    "*_start.sh",
    "direct_*.py",
    "direct_*.sh",
    "*fix*.py",
    "*fix*.sh",
    "bypass*.py",
    "bypass*.sh"
]

def should_remove(file_path):
    """判断文件是否应该移除"""
    # 始终保留指定的文件
    if os.path.basename(file_path) in KEEP_FILES:
        return False
    
    # 检查模式匹配
    file_name = os.path.basename(file_path)
    for pattern in REMOVE_PATTERNS:
        if Path(file_path).match(pattern):
            return True
    
    return False

def create_backup():
    """创建备份目录"""
    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    logger.info(f"已创建备份目录: {backup_dir}")
    return backup_dir

def main():
    """主函数"""
    print("=" * 60)
    print("系统清理工具")
    print("=" * 60)
    print("此工具将删除多余的启动文件，只保留主要接口文件")
    print("将创建备份目录以防止意外删除")
    print("-" * 60)
    
    # 创建备份目录
    backup_dir = create_backup()
    
    # 统计数据
    removed_count = 0
    backed_up_count = 0
    
    # 处理当前目录中的文件
    for item in os.listdir("."):
        # 跳过目录
        if os.path.isdir(item) and not item.startswith("."):
            if item not in KEEP_DIRS and not item.startswith("backup_"):
                # 备份整个目录
                shutil.copytree(item, os.path.join(backup_dir, item))
                backed_up_count += 1
                logger.info(f"备份目录: {item}")
            continue
        
        # 处理文件
        if os.path.isfile(item) and should_remove(item):
            # 备份文件
            shutil.copy2(item, os.path.join(backup_dir, item))
            backed_up_count += 1
            logger.info(f"备份文件: {item}")
            
            # 删除文件
            os.remove(item)
            removed_count += 1
            logger.info(f"删除文件: {item}")
    
    # 报告结果
    print(f"\n处理完成!")
    print(f"- 已删除文件: {removed_count}")
    print(f"- 已备份文件: {backed_up_count}")
    print(f"- 备份目录: {backup_dir}")
    print("\n系统现在应该只保留主要的接口文件")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
        print(f"程序运行出错: {str(e)}") 