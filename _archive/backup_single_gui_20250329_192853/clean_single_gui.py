#!/usr/bin/env python3
"""
清理系统，仅保留一个主要的GUI界面(stock_analysis_gui.py)
"""
import os
import sys
import shutil
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clean_single_gui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 需要保留的文件列表
ESSENTIAL_FILES = [
    'stock_analysis_gui.py',  # 主界面
    'momentum_analysis.py',   # 动量分析模块
    'ma_cross_strategy.py',   # 均线交叉策略模块
    'gui_controller.py',      # GUI控制器
    'run_optimized.sh',       # 启动脚本
    'requirements.txt',       # 依赖
    'README.md',              # 文档
    '.gitignore',             # Git配置
    'config.json',            # 配置文件
]

# 需要保留的目录
ESSENTIAL_DIRS = [
    'src',                    # 源代码目录
    'logs',                   # 日志目录 
    'data',                   # 数据目录
    'results',                # 结果目录
    'charts',                 # 图表目录
    'assets',                 # 资源目录
]

def backup_current_system():
    """备份当前系统"""
    from datetime import datetime
    
    backup_dir = f"backup_single_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"创建备份目录: {backup_dir}")
    
    # 创建备份目录
    os.makedirs(backup_dir, exist_ok=True)
    
    # 复制当前目录中的文件到备份目录(不包括备份目录本身和已有的备份目录)
    for item in os.listdir("."):
        if os.path.isfile(item) and not item.startswith(".") and not item.startswith("backup_"):
            shutil.copy2(item, os.path.join(backup_dir, item))
            logger.debug(f"已备份文件: {item}")
        elif os.path.isdir(item) and not item.startswith(".") and not item.startswith("backup_"):
            # 跳过 __pycache__ 目录
            if item == "__pycache__":
                continue
            shutil.copytree(item, os.path.join(backup_dir, item), dirs_exist_ok=True)
            logger.debug(f"已备份目录: {item}")
    
    logger.info(f"备份完成: {backup_dir}")
    return backup_dir

def clean_non_essential_files():
    """清理非必要文件"""
    for item in os.listdir("."):
        if os.path.isfile(item) and item not in ESSENTIAL_FILES and not item.startswith("backup_") and not item.startswith("."):
            # 跳过当前脚本和日志文件
            if item == os.path.basename(__file__) or item == "clean_single_gui.log" or item.endswith(".log"):
                continue
                
            logger.info(f"删除文件: {item}")
            os.remove(item)

def create_main_launcher():
    """创建主启动脚本"""
    launcher_content = """#!/bin/bash
# 股票分析系统主启动脚本
# 解决macOS环境下的兼容性问题并启动主界面

# 设置环境变量解决macOS兼容性问题
export SYSTEM_VERSION_COMPAT=1

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 检查Python可执行文件
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "错误: 未找到Python可执行文件"
    exit 1
fi

# 启动主界面
$PYTHON stock_analysis_gui.py
"""
    
    with open("run_main_gui.sh", "w") as f:
        f.write(launcher_content)
    
    # 设置可执行权限
    os.chmod("run_main_gui.sh", 0o755)
    logger.info("创建启动脚本: run_main_gui.sh")

def create_readme():
    """创建新的README文件"""
    readme_content = """# 股票分析系统

## 简介
这是一个集成了动量分析和均线交叉策略的股票分析系统，提供了可视化的分析工具和回测功能。

## 功能
- **动量分析**: 基于多种技术指标(RSI、MACD等)的动量评分系统
- **均线交叉策略**: 短期/长期均线交叉信号生成和回测
- **组合策略**: 动量分析与均线交叉策略的加权组合
- **市场概览**: 市场指数和行业表现统计

## 安装
```bash
pip install -r requirements.txt
```

## 运行
```bash
# 在macOS/Linux上
./run_main_gui.sh

# 在Windows上
python stock_analysis_gui.py
```

## 系统要求
- Python 3.7+
- 依赖库: pandas, numpy, matplotlib, tkinter, pillow等

## 数据源
系统支持从Tushare获取数据，也可以使用本地数据文件。

## 目录结构
- `stock_analysis_gui.py`: 主界面
- `momentum_analysis.py`: 动量分析模块
- `ma_cross_strategy.py`: 均线交叉策略模块
- `gui_controller.py`: GUI控制器
- `data/`: 数据目录
- `results/`: 结果目录
- `logs/`: 日志目录
"""
    
    with open("README_new.md", "w") as f:
        f.write(readme_content)
    
    logger.info("创建新的README文件: README_new.md")

def main():
    """主函数"""
    logger.info("开始清理系统...")
    
    # 备份当前系统
    backup_dir = backup_current_system()
    logger.info(f"已创建备份: {backup_dir}")
    
    # 清理非必要文件
    clean_non_essential_files()
    
    # 创建主启动脚本
    create_main_launcher()
    
    # 创建新的README
    create_readme()
    
    logger.info("系统清理完成，现在只保留了主要的GUI界面和必要的组件")
    logger.info("使用 ./run_main_gui.sh 启动系统")

if __name__ == "__main__":
    main() 