#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预加载和批量获取模块

提供股票数据预加载、批量获取和数据维护功能
"""

import os
import time
import logging
import threading
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 导入缓存工具（如果存在）
try:
    from src.utils.cache import CacheManager, CacheLevel, cached
except ImportError:
    # 缓存不可用时的兼容性实现
    def cached(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# 设置日志
logger = logging.getLogger(__name__)

class DataPreloader:
    """数据预加载器
    
    在后台预先加载频繁使用的数据，减少实时查询等待时间
    """
    
    def __init__(self, max_workers: int = 4, 
                 preload_interval: int = 3600,  # 1小时预加载一次
                 data_manager = None):
        """
        初始化数据预加载器
        
        Args:
            max_workers: 最大工作线程数
            preload_interval: 预加载间隔(秒)
            data_manager: 数据管理器实例
        """
        self.max_workers = max_workers
        self.preload_interval = preload_interval
        self.data_manager = data_manager
        
        # 预加载状态
        self.is_preloading = False
        self.preload_status = {}
        self.preload_lock = threading.RLock()
        
        # 预加载线程
        self.preload_thread = None
        self.stop_event = threading.Event()
        
        # 已预加载数据
        self.preloaded_data = {}
        
    def start(self):
        """启动预加载服务"""
        if self.preload_thread is not None and self.preload_thread.is_alive():
            logger.warning("预加载服务已经在运行中")
            return False
            
        self.stop_event.clear()
        self.preload_thread = threading.Thread(target=self._preload_service, daemon=True)
        self.preload_thread.start()
        logger.info(f"数据预加载服务已启动，间隔：{self.preload_interval}秒")
        return True
        
    def stop(self):
        """停止预加载服务"""
        if self.preload_thread is None or not self.preload_thread.is_alive():
            return
            
        self.stop_event.set()
        self.preload_thread.join(timeout=5)
        logger.info("数据预加载服务已停止")
        
    def get_preloaded_data(self, data_key: str) -> Optional[Any]:
        """
        获取预加载的数据
        
        Args:
            data_key: 数据键名
            
        Returns:
            预加载的数据或None
        """
        with self.preload_lock:
            data = self.preloaded_data.get(data_key)
            
            # 如果数据存在但已过期，返回None
            if data is not None and "expires" in data:
                if datetime.now().timestamp() > data["expires"]:
                    return None
                    
            # 返回数据内容
            return data["data"] if data and "data" in data else None
            
    def get_status(self) -> Dict[str, Any]:
        """获取预加载状态"""
        with self.preload_lock:
            result = {
                "is_active": self.preload_thread is not None and self.preload_thread.is_alive(),
                "is_preloading": self.is_preloading,
                "preload_tasks": dict(self.preload_status),
                "preloaded_items": list(self.preloaded_data.keys()),
                "next_preload": 0
            }
            
            # 计算下次预加载时间
            if result["is_active"] and not result["is_preloading"]:
                for task, status in self.preload_status.items():
                    if "last_preload" in status:
                        next_time = status["last_preload"] + self.preload_interval
                        if result["next_preload"] == 0 or next_time < result["next_preload"]:
                            result["next_preload"] = next_time
                            
            return result
            
    def preload_stock_data(self, forced: bool = False):
        """
        预加载股票数据
        
        Args:
            forced: 是否强制重新加载，无视时间间隔
        """
        if self.data_manager is None:
            logger.error("预加载失败：未配置数据管理器")
            return False
            
        # 如果已在预加载中，则跳过
        if self.is_preloading and not forced:
            logger.info("预加载已在进行中，跳过本次请求")
            return False
            
        # 启动异步预加载
        threading.Thread(target=self._preload_all_data, args=(forced,), daemon=True).start()
        return True
        
    def _preload_service(self):
        """预加载服务主循环"""
        while not self.stop_event.is_set():
            try:
                # 执行预加载
                self._preload_all_data()
                
                # 等待下次预加载
                for _ in range(self.preload_interval):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"预加载服务出错: {str(e)}")
                time.sleep(60)  # 出错后暂停一分钟
                
    def _preload_all_data(self, forced: bool = False):
        """
        预加载所有数据
        
        Args:
            forced: 是否强制重新加载
        """
        if self.data_manager is None:
            return
            
        with self.preload_lock:
            if self.is_preloading and not forced:
                return
                
            self.is_preloading = True
            
        try:
            # 预加载基础股票列表
            self._preload_task("stock_list", self._load_stock_list, forced)
            
            # 预加载指数列表
            self._preload_task("index_list", self._load_index_list, forced)
            
            # 预加载行业数据
            self._preload_task("industry_data", self._load_industry_data, forced)
            
            # 预加载市场概览数据
            self._preload_task("market_overview", self._load_market_overview, forced)
            
            logger.info("数据预加载完成")
        except Exception as e:
            logger.error(f"数据预加载出错: {str(e)}")
        finally:
            with self.preload_lock:
                self.is_preloading = False
                
    def _preload_task(self, task_key: str, load_func: Callable, forced: bool = False) -> bool:
        """
        执行单个预加载任务
        
        Args:
            task_key: 任务键名
            load_func: 加载函数
            forced: 是否强制重新加载
            
        Returns:
            bool: 是否执行了加载
        """
        # 检查是否需要加载
        with self.preload_lock:
            now = datetime.now().timestamp()
            
            if task_key not in self.preload_status:
                self.preload_status[task_key] = {"last_preload": 0}
                
            if not forced and now - self.preload_status[task_key]["last_preload"] < self.preload_interval:
                return False
                
            # 更新状态
            self.preload_status[task_key]["status"] = "loading"
            self.preload_status[task_key]["start_time"] = now
            
        try:
            # 执行加载
            logger.info(f"开始预加载任务: {task_key}")
            result = load_func()
            
            # 保存结果
            with self.preload_lock:
                if result is not None:
                    self.preloaded_data[task_key] = {
                        "data": result,
                        "expires": now + self.preload_interval,
                        "loaded_at": now
                    }
                    
                # 更新状态
                self.preload_status[task_key]["status"] = "success"
                self.preload_status[task_key]["last_preload"] = now
                self.preload_status[task_key]["duration"] = datetime.now().timestamp() - now
                
            return True
        except Exception as e:
            logger.error(f"预加载任务出错 ({task_key}): {str(e)}")
            
            # 更新状态
            with self.preload_lock:
                self.preload_status[task_key]["status"] = "error"
                self.preload_status[task_key]["error"] = str(e)
                self.preload_status[task_key]["duration"] = datetime.now().timestamp() - now
                
            return False
    
    # 预加载任务实现
    
    def _load_stock_list(self) -> Dict[str, Any]:
        """预加载股票列表"""
        if hasattr(self.data_manager, 'get_all_stocks'):
            stocks = self.data_manager.get_all_stocks()
            return {
                "stocks": stocks,
                "count": len(stocks) if stocks is not None else 0
            }
        return None
        
    def _load_index_list(self) -> Dict[str, Any]:
        """预加载指数列表"""
        if hasattr(self.data_manager, 'get_index_list'):
            indices = self.data_manager.get_index_list()
            return {
                "indices": indices,
                "count": len(indices) if indices is not None else 0
            }
        return None
        
    def _load_industry_data(self) -> Dict[str, Any]:
        """预加载行业数据"""
        if hasattr(self.data_manager, 'get_industry_data'):
            industries = self.data_manager.get_industry_data()
            return {
                "industries": industries,
                "count": len(industries) if industries is not None else 0
            }
        return None
        
    def _load_market_overview(self) -> Dict[str, Any]:
        """预加载市场概览数据"""
        if hasattr(self.data_manager, 'get_market_overview'):
            return self.data_manager.get_market_overview()
        return None

class BatchDataFetcher:
    """批量数据获取器
    
    优化批量获取股票数据的性能，支持多线程下载和自动重试
    """
    
    def __init__(self, data_manager = None, max_workers: int = 4, 
                 retry_count: int = 3, retry_delay: float = 1.0):
        """
        初始化批量数据获取器
        
        Args:
            data_manager: 数据管理器实例
            max_workers: 最大工作线程数
            retry_count: 重试次数
            retry_delay: 重试延迟(秒)
        """
        self.data_manager = data_manager
        self.max_workers = max_workers
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        
    def batch_get_stock_data(self, stock_codes: List[str], 
                           start_date: str, end_date: str, 
                           fields: Optional[List[str]] = None,
                           progress_callback: Optional[Callable] = None) -> Dict[str, pd.DataFrame]:
        """
        批量获取股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 需要的字段列表
            progress_callback: 进度回调函数
            
        Returns:
            Dict[str, pd.DataFrame]: 股票数据字典，键为股票代码
        """
        if self.data_manager is None:
            logger.error("批量获取失败：未配置数据管理器")
            return {}
            
        # 用于存储结果
        results = {}
        
        # 已完成计数
        completed = 0
        total = len(stock_codes)
        
        # 线程安全锁
        lock = threading.RLock()
        
        def fetch_single_stock(stock_code):
            """获取单个股票数据的函数"""
            nonlocal completed
            
            # 多次重试
            for attempt in range(self.retry_count):
                try:
                    # 调用数据管理器获取股票数据
                    if hasattr(self.data_manager, 'get_stock_data'):
                        data = self.data_manager.get_stock_data(
                            stock_code, start_date, end_date, fields
                        )
                        
                        # 存储结果
                        with lock:
                            results[stock_code] = data
                            completed += 1
                            
                            # 调用进度回调
                            if progress_callback:
                                try:
                                    progress_callback(completed, total, stock_code)
                                except Exception as e:
                                    logger.error(f"进度回调出错: {str(e)}")
                                    
                        return data
                except Exception as e:
                    if attempt < self.retry_count - 1:
                        logger.warning(f"获取股票 {stock_code} 数据失败(尝试 {attempt+1}/{self.retry_count}): {str(e)}")
                        time.sleep(self.retry_delay * (attempt + 1))
                    else:
                        logger.error(f"获取股票 {stock_code} 数据失败: {str(e)}")
                        
                        # 更新进度
                        with lock:
                            completed += 1
                            
                            # 调用进度回调
                            if progress_callback:
                                try:
                                    progress_callback(completed, total, None)
                                except Exception as e:
                                    logger.error(f"进度回调出错: {str(e)}")
            
            return None
            
        # 使用线程池并行获取数据
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(fetch_single_stock, stock_codes)
            
        return results
        
    def batch_get_financial_data(self, stock_codes: List[str], 
                               report_date: str,
                               fields: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        批量获取财务数据
        
        Args:
            stock_codes: 股票代码列表
            report_date: 报告期
            fields: 需要的字段列表
            
        Returns:
            Dict[str, Dict]: 财务数据字典，键为股票代码
        """
        if self.data_manager is None or not hasattr(self.data_manager, 'get_financial_data'):
            logger.error("批量获取财务数据失败：未配置数据管理器或不支持该方法")
            return {}
            
        # 实现类似于批量获取股票数据的逻辑
        # 这里简化处理，直接逐个获取
        results = {}
        
        for stock_code in stock_codes:
            try:
                data = self.data_manager.get_financial_data(stock_code, report_date, fields)
                if data is not None:
                    results[stock_code] = data
            except Exception as e:
                logger.error(f"获取股票 {stock_code} 财务数据失败: {str(e)}")
                
        return results

# 辅助工具：线程池执行器（简化版）
class ThreadPoolExecutor:
    """线程池执行器"""
    
    def __init__(self, max_workers: int):
        """
        初始化线程池执行器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.workers = []
        self.tasks = []
        self.lock = threading.RLock()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        
    def map(self, func, items):
        """
        对列表项应用函数
        
        Args:
            func: 要应用的函数
            items: 项目列表
            
        Returns:
            List: 结果列表
        """
        # 创建任务
        tasks = []
        for item in items:
            task = threading.Thread(target=func, args=(item,))
            tasks.append(task)
            
        # 控制并发数量
        active = []
        results = []
        
        for task in tasks:
            # 检查活动线程，移除已完成的
            active = [t for t in active if t.is_alive()]
            
            # 如果活动线程数量达到最大值，等待
            while len(active) >= self.max_workers:
                time.sleep(0.1)
                active = [t for t in active if t.is_alive()]
                
            # 启动新线程
            task.start()
            active.append(task)
            
        # 等待所有任务完成
        for task in tasks:
            task.join()
            
        return results
        
    def shutdown(self):
        """关闭线程池"""
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1)
                
# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 简单测试
    class MockDataManager:
        def get_all_stocks(self):
            return ["000001.SZ", "000002.SZ", "000003.SZ"]
            
        def get_stock_data(self, stock_code, start_date, end_date, fields=None):
            time.sleep(0.1)  # 模拟网络延迟
            return pd.DataFrame({
                "date": [start_date, end_date],
                "open": [10, 11],
                "close": [11, 12]
            })
    
    # 创建预加载器
    data_manager = MockDataManager()
    preloader = DataPreloader(data_manager=data_manager)
    preloader.start()
    
    # 测试批量获取
    fetcher = BatchDataFetcher(data_manager=data_manager, max_workers=2)
    
    def progress(current, total, item):
        print(f"进度: {current}/{total} - {item}")
        
    results = fetcher.batch_get_stock_data(
        ["000001.SZ", "000002.SZ", "000003.SZ"],
        "2023-01-01", "2023-01-10",
        progress_callback=progress
    )
    
    print(f"获取结果数量: {len(results)}")
    preloader.stop() 