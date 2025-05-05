import time
import logging
from functools import wraps
from typing import Callable, Any
import psutil
import os

class PerformanceMonitor:
    """性能监控工具类"""
    
    def __init__(self, logger: logging.Logger):
        """
        初始化性能监控器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger
        self.process = psutil.Process(os.getpid())
    
    def monitor_memory(self, func: Callable) -> Callable:
        """
        监控函数内存使用
        
        Args:
            func: 要监控的函数
            
        Returns:
            包装后的函数
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            result = func(*args, **kwargs)
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_used = end_memory - start_memory
            
            self.logger.info(
                f"函数 {func.__name__} 内存使用: {memory_used:.2f}MB"
            )
            return result
        return wrapper
    
    def monitor_time(self, func: Callable) -> Callable:
        """
        监控函数执行时间
        
        Args:
            func: 要监控的函数
            
        Returns:
            包装后的函数
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            self.logger.info(
                f"函数 {func.__name__} 执行时间: {execution_time:.4f}秒"
            )
            return result
        return wrapper
    
    def monitor_cpu(self, func: Callable) -> Callable:
        """
        监控函数CPU使用
        
        Args:
            func: 要监控的函数
            
        Returns:
            包装后的函数
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_cpu = self.process.cpu_percent()
            result = func(*args, **kwargs)
            end_cpu = self.process.cpu_percent()
            cpu_used = end_cpu - start_cpu
            
            self.logger.info(
                f"函数 {func.__name__} CPU使用: {cpu_used:.2f}%"
            )
            return result
 