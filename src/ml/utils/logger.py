import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
import os

class Logger:
    """日志工具类"""
    
    @staticmethod
    def setup_logger(
        name: str,
        log_path: str,
        level: str = 'INFO',
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> logging.Logger:
        """
        配置日志记录器
        
        Args:
            name: 日志记录器名称
            log_path: 日志文件路径
            level: 日志级别
            max_bytes: 单个日志文件最大大小
            backup_count: 保留的日志文件数量
            
        Returns:
            配置好的日志记录器
        """
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # 创建日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # 如果已经有处理器,不重复添加
        if logger.handlers:
            return logger
        
        # 文件处理器
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console_handler)
        
        return logger 