#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票推荐系统主程序入口
"""

import os
import sys
import tkinter as tk
import logging
from pathlib import Path
import json
import argparse

# 确保src包可以被导入
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir.parent))

# 导入配置
from src.config.settings import GUI_CONFIG, SYSTEM_CONFIG

# 导入UI模块
from src.visualization.app_ui import StockAppUI

# 配置日志
def setup_logging():
    """配置日志系统"""
    log_dir = Path(SYSTEM_CONFIG.get("log_dir", "logs"))
    log_file = log_dir / SYSTEM_CONFIG.get("log_file", "stockrecsys.log")
    
    # 确保日志目录存在
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # 配置日志
    log_level = getattr(logging, SYSTEM_CONFIG.get("log_level", "INFO"))
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("StockRecommendationSystem")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="股票推荐系统")
    parser.add_argument(
        "--theme", 
        choices=["default", "dark", "light"], 
        default=GUI_CONFIG.get("default_theme", "default"),
        help="UI主题"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="启用调试模式"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="自定义配置文件路径"
    )
    
    return parser.parse_args()

def load_custom_config(config_path):
    """加载自定义配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载自定义配置失败: {e}")
        return {}

def create_data_dirs():
    """创建必要的数据目录"""
    dirs = [
        SYSTEM_CONFIG.get("data_dir", "data"),
        SYSTEM_CONFIG.get("cache_dir", "cache"),
        SYSTEM_CONFIG.get("output_dir", "output")
    ]
    
    for d in dirs:
        Path(d).mkdir(exist_ok=True, parents=True)
        logger.info(f"确保目录存在: {d}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 启用调试模式
    if args.debug:
        SYSTEM_CONFIG["enable_debug"] = True
        SYSTEM_CONFIG["log_level"] = "DEBUG"
    
    # 配置日志
    global logger
    logger = setup_logging()
    logger.info("股票推荐系统启动")
    
    # 加载自定义配置
    if args.config:
        custom_config = load_custom_config(args.config)
        # 合并配置 (实际应用中可能需要更复杂的合并逻辑)
        logger.info(f"已加载自定义配置: {args.config}")
    
    # 创建数据目录
    create_data_dirs()
    
    # 创建UI
    root = tk.Tk()
    root.title(GUI_CONFIG.get("window_title", "股票推荐系统"))
    root.geometry(f"{GUI_CONFIG.get('window_width', 1280)}x{GUI_CONFIG.get('window_height', 800)}")
    
    # 设置图标 (如果存在)
    icon_path = Path("assets/icon.ico")
    if icon_path.exists():
        root.iconbitmap(icon_path)
    
    # 创建应用程序界面
    app = StockAppUI(root)
    
    # 应用命令行指定的主题
    if args.theme and args.theme != app.theme_manager.theme_name:
        app.change_theme(args.theme)
    
    # 运行应用程序
    root.mainloop()
    
    logger.info("股票推荐系统关闭")

if __name__ == "__main__":
    main() 