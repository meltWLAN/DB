#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
异步数据预加载模块
提供数据预取和异步处理功能，显著提升数据密集型操作性能
"""

import os
import sys
import logging
import time
import threading
import queue
import asyncio
import concurrent.futures
import functools
from collections import deque
from typing import List, Dict, Any, Callable, Optional, Tuple, Union

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 确保当前目录在Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 尝试导入本地优化模块
try:
    from .memory_optimizer import cleanup_memory
except ImportError:
    cleanup_memory = None

# 全局预取缓存
_prefetch_cache = {}
_prefetch_locks = {}
_prefetch_stats = {
    'hits': 0,
    'misses': 0,
    'prefetch_requests': 0,
    'completed_prefetch': 0
}

class AsyncDataPrefetcher:
    """
    异步数据预加载器
    在后台线程中预先加载和处理数据，避免在主线程中的阻塞等待
    """
    
    def __init__(self, max_workers=None, max_queue_size=10, cleanup_threshold=70):
        """
        初始化异步数据预加载器
        
        Args:
            max_workers: 最大工作线程数，None表示自动选择
            max_queue_size: 队列最大大小
            cleanup_threshold: 内存清理阈值（百分比）
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.cleanup_threshold = cleanup_threshold
        
        # 创建线程池执行器
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # 任务队列和结果缓存
        self.tasks = queue.Queue(maxsize=max_queue_size)
        self.results = {}
        self.futures = {}
        self.task_lock = threading.Lock()
        
        # 统计数据
        self.stats = {
            'requested': 0,
            'completed': 0,
            'cache_hits': 0,
            'errors': 0,
            'avg_time': 0,
            'total_time': 0
        }
        
        # 启动工作线程
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info(f"异步数据预加载器初始化完成: max_workers={max_workers}, queue_size={max_queue_size}")
    
    def _worker_loop(self):
        """工作线程主循环"""
        while self.running:
            try:
                # 从队列获取任务
                try:
                    task_id, func, args, kwargs = self.tasks.get(timeout=1)
                except queue.Empty:
                    continue
                
                # 检查缓存
                if task_id in self.results:
                    self.tasks.task_done()
                    self.stats['cache_hits'] += 1
                    continue
                
                # 提交到线程池处理
                future = self.executor.submit(func, *args, **kwargs)
                with self.task_lock:
                    self.futures[task_id] = future
                
                # 等待结果
                try:
                    start_time = time.time()
                    result = future.result()
                    elapsed = time.time() - start_time
                    
                    # 更新统计信息
                    self.stats['completed'] += 1
                    self.stats['total_time'] += elapsed
                    self.stats['avg_time'] = self.stats['total_time'] / self.stats['completed']
                    
                    # 存储结果
                    with self.task_lock:
                        self.results[task_id] = (result, None)
                        if task_id in self.futures:
                            del self.futures[task_id]
                except Exception as e:
                    logger.error(f"执行任务 {task_id} 失败: {str(e)}")
                    with self.task_lock:
                        self.results[task_id] = (None, str(e))
                        self.stats['errors'] += 1
                        if task_id in self.futures:
                            del self.futures[task_id]
                
                self.tasks.task_done()
                
                # 检查内存使用情况，必要时清理
                if cleanup_memory is not None and self.cleanup_threshold > 0:
                    cleanup_memory(threshold_percent=self.cleanup_threshold)
                
            except Exception as e:
                logger.error(f"预取工作线程出错: {str(e)}")
                time.sleep(1)  # 避免过快循环
    
    def prefetch(self, task_id, func, *args, **kwargs):
        """
        预加载数据
        
        Args:
            task_id: 任务ID，必须唯一
            func: 要执行的函数
            *args, **kwargs: 函数参数
        
        Returns:
            bool: 是否成功提交预取请求
        """
        # 更新统计
        self.stats['requested'] += 1
        
        # 检查结果是否已存在
        if task_id in self.results:
            return True
        
        # 提交任务到队列
        try:
            self.tasks.put((task_id, func, args, kwargs), block=False)
            return True
        except queue.Full:
            logger.warning(f"预取队列已满，任务 {task_id} 被拒绝")
            return False
    
    def get_result(self, task_id, timeout=None, delete=True):
        """
        获取预加载的结果
        
        Args:
            task_id: 任务ID
            timeout: 等待超时时间（秒），None表示一直等待
            delete: 获取后是否从缓存删除结果
        
        Returns:
            tuple: (result, error)，如果出错则error不为None
        """
        # 检查结果是否已存在
        if task_id in self.results:
            result = self.results[task_id]
            if delete:
                with self.task_lock:
                    if task_id in self.results:
                        del self.results[task_id]
            return result
        
        # 检查任务是否正在执行
        with self.task_lock:
            if task_id in self.futures:
                future = self.futures[task_id]
                try:
                    result = future.result(timeout=timeout)
                    self.results[task_id] = (result, None)
                    if delete:
                        del self.results[task_id]
                    return (result, None)
                except concurrent.futures.TimeoutError:
                    return (None, "等待超时")
                except Exception as e:
                    return (None, str(e))
        
        return (None, "任务不存在")
    
    def wait_all(self, timeout=None):
        """
        等待所有任务完成
        
        Args:
            timeout: 等待超时时间（秒），None表示一直等待
        
        Returns:
            bool: 是否所有任务都已完成
        """
        try:
            return self.tasks.join(timeout=timeout)
        except:
            return False
    
    def clear(self):
        """清空所有缓存的结果和统计信息"""
        with self.task_lock:
            self.results.clear()
            for future in self.futures.values():
                future.cancel()
            self.futures.clear()
            
            # 清空统计信息
            for key in self.stats:
                self.stats[key] = 0
        
        # 清空任务队列
        while not self.tasks.empty():
            try:
                self.tasks.get_nowait()
                self.tasks.task_done()
            except:
                break
    
    def shutdown(self):
        """关闭预加载器"""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)
        self.executor.shutdown(wait=False)
        logger.info("异步数据预加载器已关闭")
    
    def get_stats(self):
        """获取统计信息"""
        return self.stats.copy()

# 全局预取器实例
_global_prefetcher = None

def get_prefetcher(create_if_none=True):
    """
    获取全局预取器实例
    
    Args:
        create_if_none: 如果不存在是否创建新实例
    
    Returns:
        AsyncDataPrefetcher: 预取器实例
    """
    global _global_prefetcher
    if _global_prefetcher is None and create_if_none:
        _global_prefetcher = AsyncDataPrefetcher()
    return _global_prefetcher

def prefetch_data(func_or_key, *args, **kwargs):
    """
    预取数据的简便函数
    
    Args:
        func_or_key: 函数或字符串键
        *args, **kwargs: 函数参数
    
    Returns:
        str: 任务ID
    """
    global _prefetch_stats
    _prefetch_stats['prefetch_requests'] += 1
    
    # 创建任务ID
    if callable(func_or_key):
        func = func_or_key
        task_id = f"{func.__name__}_{hash((func.__name__, args, frozenset(kwargs.items())))}"
    else:
        task_id = func_or_key
        func = kwargs.pop('func', None)
        if func is None:
            raise ValueError("如果第一个参数不是可调用对象，必须提供func关键字参数")
    
    # 获取预取器
    prefetcher = get_prefetcher()
    
    # 提交预取任务
    success = prefetcher.prefetch(task_id, func, *args, **kwargs)
    if success:
        _prefetch_stats['completed_prefetch'] += 1
    
    return task_id

def get_prefetched_data(task_id, timeout=None, delete=True):
    """
    获取预取的数据
    
    Args:
        task_id: 任务ID
        timeout: 等待超时时间
        delete: 获取后是否删除
    
    Returns:
        tuple: (result, error)
    """
    global _prefetch_stats
    
    prefetcher = get_prefetcher(create_if_none=False)
    if prefetcher is None:
        return (None, "预取器未初始化")
    
    result, error = prefetcher.get_result(task_id, timeout=timeout, delete=delete)
    if error is None:
        _prefetch_stats['hits'] += 1
    else:
        _prefetch_stats['misses'] += 1
    
    return result, error

def prefetch_decorator(task_id_func=None):
    """
    预取装饰器
    
    Args:
        task_id_func: 用于生成任务ID的函数，接收与原函数相同的参数
    
    Returns:
        function: 装饰过的函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成任务ID
            if task_id_func is not None:
                task_id = task_id_func(*args, **kwargs)
            else:
                task_id = f"{func.__name__}_{hash((func.__name__, args, frozenset(kwargs.items())))}"
            
            # 尝试从预取缓存获取
            result, error = get_prefetched_data(task_id, timeout=None)
            if error is None:
                return result
            
            # 如果没有预取或预取失败，则直接执行
            return func(*args, **kwargs)
        
        # 添加预取方法
        wrapper.prefetch = lambda *args, **kwargs: prefetch_data(func, *args, **kwargs)
        return wrapper
    return decorator

def get_prefetch_stats():
    """获取预取统计信息"""
    global _prefetch_stats
    
    # 添加预取器的详细统计
    prefetcher = get_prefetcher(create_if_none=False)
    if prefetcher:
        stats = prefetcher.get_stats()
        _prefetch_stats.update(stats)
    
    return _prefetch_stats.copy()

def clear_prefetch_cache():
    """清空预取缓存"""
    prefetcher = get_prefetcher(create_if_none=False)
    if prefetcher:
        prefetcher.clear()
    
    global _prefetch_stats
    for key in _prefetch_stats:
        _prefetch_stats[key] = 0

# 导出主要函数
__all__ = [
    'AsyncDataPrefetcher',
    'get_prefetcher',
    'prefetch_data',
    'get_prefetched_data',
    'prefetch_decorator',
    'get_prefetch_stats',
    'clear_prefetch_cache'
]

if __name__ == "__main__":
    # 测试代码
    def load_data(symbol, days=10):
        """模拟加载数据的耗时操作"""
        time.sleep(1)  # 模拟耗时操作
        return {
            'symbol': symbol,
            'days': days,
            'data': [i for i in range(days)]
        }
    
    # 预取一些数据
    print("开始预取数据...")
    task1 = prefetch_data(load_data, "000001.SZ", days=30)
    task2 = prefetch_data(load_data, "000002.SZ", days=20)
    
    # 模拟做其他工作
    print("正在进行其他工作...")
    time.sleep(0.5)
    
    # 获取预取结果
    print("获取预取结果:")
    result1, error1 = get_prefetched_data(task1)
    if error1 is None:
        print(f"任务1结果: {result1}")
    else:
        print(f"任务1错误: {error1}")
    
    result2, error2 = get_prefetched_data(task2)
    if error2 is None:
        print(f"任务2结果: {result2}")
    else:
        print(f"任务2错误: {error2}")
    
    # 显示统计信息
    stats = get_prefetch_stats()
    print("\n预取统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 关闭预取器
    prefetcher = get_prefetcher()
    prefetcher.shutdown() 