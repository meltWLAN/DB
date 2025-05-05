#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志工具模块
提供统一的日志配置和管理
"""

import logging
import os
from datetime import datetime
from pathlib import Path

def get_logger(name, level=logging.INFO, log_file=None):
    """
    获取配置好的logger实例
    
    Args:
        name: logger名称
        level: 日志级别
        log_file: 日志文件路径，如果为None则自动生成
        
    Returns:
        配置好的logger实例
    """
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除之前的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件或者自动创建日志文件
    if log_file is not None or name:
        # 如果没有指定日志文件，则自动生成
        if log_file is None:
            # 获取日志目录
            log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "../../logs"
            os.makedirs(log_dir, exist_ok=True)
            
            # 生成日志文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{name}_{timestamp}.log"
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 