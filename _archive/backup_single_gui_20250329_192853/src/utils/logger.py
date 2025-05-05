"""
日志工具模块
提供统一的日志记录功能
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from ..config.settings import LOG_LEVEL, LOG_FORMAT, LOG_FILE

class SystemLogger:
    """系统日志类"""
    
    def __init__(self, name: str):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
        """
        self.logger = logger.bind(name=name)
        self._setup_logger()
        
    def _setup_logger(self) -> None:
        """配置日志记录器"""
        # 移除默认处理器
        logger.remove()
        
        # 添加控制台处理器
        logger.add(
            sys.stderr,
            format=LOG_FORMAT,
            level=LOG_LEVEL,
            colorize=True
        )
        
        # 添加文件处理器
        logger.add(
            LOG_FILE,
            format=LOG_FORMAT,
            level=LOG_LEVEL,
            rotation="500 MB",
            retention="10 days",
            encoding="utf-8"
        )
        
    def debug(self, message: str) -> None:
        """记录调试信息"""
        self.logger.debug(message)
        
    def info(self, message: str) -> None:
        """记录一般信息"""
        self.logger.info(message)
        
    def warning(self, message: str) -> None:
        """记录警告信息"""
        self.logger.warning(message)
        
    def error(self, message: str, exc_info: Optional[Exception] = None) -> None:
        """
        记录错误信息
        
        Args:
            message: 错误信息
            exc_info: 异常对象
        """
        if exc_info:
            self.logger.exception(f"{message}: {str(exc_info)}")
        else:
            self.logger.error(message)
            
    def critical(self, message: str) -> None:
        """记录严重错误信息"""
        self.logger.critical(message)
        
    def success(self, message: str) -> None:
        """记录成功信息"""
        self.logger.success(message)
        
    def exception(self, message: str) -> None:
        """记录异常信息"""
        self.logger.exception(message)

def get_logger(name: str) -> SystemLogger:
    """
    获取日志记录器实例
    
    Args:
        name: 日志记录器名称
        
    Returns:
        SystemLogger: 日志记录器实例
    """
    return SystemLogger(name) 