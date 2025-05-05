"""
数据预加载模块
负责后台预加载和预处理数据
"""

import threading
import queue
import time
from typing import List, Dict, Any, Optional, Callable, Set
import pandas as pd
from datetime import datetime, timedelta
from ..utils.logger import get_logger
from ..utils.parallel_processor import ParallelProcessor
from ..data.data_manager import DataManager
from ..data.cache_manager import CacheManager

class DataPreloader:
    """数据预加载器"""
    
    def __init__(self, data_manager: Optional[DataManager] = None, 
                 cache_manager: Optional[CacheManager] = None,
                 max_workers: int = 4):
        """
        初始化数据预加载器
        
        Args:
            data_manager: 数据管理器
            cache_manager: 缓存管理器
            max_workers: 最大工作线程数
        """
        self.logger = get_logger("DataPreloader")
        
        # 设置管理器
        self.data_manager = data_manager if data_manager else DataManager()
        self.cache_manager = cache_manager if cache_manager else CacheManager()
        
        # 设置并行处理器 (使用线程，因为主要是IO密集型任务)
        self.processor = ParallelProcessor(use_threads=True, max_workers=max_workers)
        
        # 工作队列
        self.task_queue = queue.Queue()
        
        # 正在处理的任务
        self.processing = set()
        self.processing_lock = threading.Lock()
        
        # 预加载线程
        self.preload_thread = None
        self.stop_event = threading.Event()
        
        # 统计信息
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0
        }
        
        self.logger.info(f"数据预加载器初始化完成，最大工作线程数：{max_workers}")
        
    def start(self) -> None:
        """启动预加载线程"""
        if self.preload_thread is None or not self.preload_thread.is_alive():
            self.stop_event.clear()
            self.preload_thread = threading.Thread(target=self._preload_worker, daemon=True)
            self.preload_thread.start()
            self.logger.info("数据预加载线程已启动")
        else:
            self.logger.warning("数据预加载线程已在运行")
            
    def stop(self) -> None:
        """停止预加载线程"""
        if self.preload_thread and self.preload_thread.is_alive():
            self.stop_event.set()
            self.preload_thread.join(timeout=2.0)
            self.logger.info("数据预加载线程已停止")
        else:
            self.logger.warning("数据预加载线程未运行")
            
    def _preload_worker(self) -> None:
        """预加载工作线程"""
        self.logger.info("预加载工作线程已启动")
        
        while not self.stop_event.is_set():
            try:
                # 尝试从队列获取任务，最多等待1秒
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # 处理任务
                start_time = time.time()
                task_id = task.get("id", "unknown")
                
                try:
                    # 添加到正在处理的集合
                    with self.processing_lock:
                        self.processing.add(task_id)
                        
                    # 执行任务
                    self._execute_task(task)
                    
                    # 更新统计信息
                    self.stats["tasks_completed"] += 1
                    
                except Exception as e:
                    self.logger.error(f"预加载任务执行失败 [{task_id}]: {str(e)}")
                    self.stats["tasks_failed"] += 1
                    
                finally:
                    # 任务完成，从处理集合中移除
                    with self.processing_lock:
                        self.processing.discard(task_id)
                        
                    # 标记任务完成
                    self.task_queue.task_done()
                    
                    # 更新统计信息
                    self.stats["total_processing_time"] += time.time() - start_time
                    
            except Exception as e:
                self.logger.error(f"预加载工作线程发生错误: {str(e)}")
                time.sleep(1.0)  # 避免错误循环消耗资源
                
        self.logger.info("预加载工作线程已退出")
        
    def _execute_task(self, task: Dict[str, Any]) -> None:
        """执行预加载任务"""
        task_type = task.get("type")
        
        if task_type == "stock_data":
            self._preload_stock_data(task)
        elif task_type == "stock_list":
            self._preload_stock_list(task)
        elif task_type == "batch_stock_data":
            self._preload_batch_stock_data(task)
        elif task_type == "custom":
            self._execute_custom_task(task)
        else:
            self.logger.warning(f"未知的任务类型: {task_type}")
            
    def _preload_stock_data(self, task: Dict[str, Any]) -> None:
        """预加载股票数据"""
        stock_code = task.get("stock_code")
        start_date = task.get("start_date")
        end_date = task.get("end_date")
        source = task.get("source", "tushare")
        
        if not stock_code:
            self.logger.error("预加载股票数据缺少股票代码")
            return
            
        try:
            # 生成缓存键
            cache_key = {
                "type": "stock_data",
                "stock_code": stock_code,
                "start_date": start_date,
                "end_date": end_date,
                "source": source
            }
            
            # 检查是否已经缓存
            if self.cache_manager.get_dataframe(cache_key) is not None:
                self.logger.debug(f"股票数据已存在缓存中: {stock_code}")
                return
                
            # 获取数据
            self.logger.info(f"预加载股票数据: {stock_code}")
            df = self.data_manager.get_stock_data(stock_code, start_date, end_date, source)
            
            if df is not None and not df.empty:
                # 缓存数据
                self.cache_manager.set_dataframe(cache_key, df)
                self.logger.info(f"股票数据已预加载并缓存: {stock_code}")
            else:
                self.logger.warning(f"股票数据预加载失败，数据为空: {stock_code}")
                
        except Exception as e:
            self.logger.error(f"预加载股票数据失败 [{stock_code}]: {str(e)}")
            
    def _preload_stock_list(self, task: Dict[str, Any]) -> None:
        """预加载股票列表"""
        source = task.get("source", "tushare")
        
        try:
            # 生成缓存键
            cache_key = {
                "type": "stock_list",
                "source": source
            }
            
            # 检查是否已经缓存
            if self.cache_manager.get_dataframe(cache_key) is not None:
                self.logger.debug(f"股票列表已存在缓存中")
                return
                
            # 获取数据
            self.logger.info(f"预加载股票列表")
            df = self.data_manager.get_stock_list(source)
            
            if df is not None and not df.empty:
                # 缓存数据
                self.cache_manager.set_dataframe(cache_key, df)
                self.logger.info(f"股票列表已预加载并缓存")
            else:
                self.logger.warning(f"股票列表预加载失败，数据为空")
                
        except Exception as e:
            self.logger.error(f"预加载股票列表失败: {str(e)}")
            
    def _preload_batch_stock_data(self, task: Dict[str, Any]) -> None:
        """批量预加载股票数据"""
        stock_codes = task.get("stock_codes", [])
        start_date = task.get("start_date")
        end_date = task.get("end_date")
        source = task.get("source", "tushare")
        
        if not stock_codes:
            self.logger.error("批量预加载股票数据缺少股票代码列表")
            return
            
        try:
            # 定义单个股票数据获取函数
            def _get_stock_data(stock_code: str) -> Optional[pd.DataFrame]:
                try:
                    cache_key = {
                        "type": "stock_data",
                        "stock_code": stock_code,
                        "start_date": start_date,
                        "end_date": end_date,
                        "source": source
                    }
                    
                    # 检查缓存
                    df = self.cache_manager.get_dataframe(cache_key)
                    if df is not None:
                        return df
                        
                    # 获取数据
                    df = self.data_manager.get_stock_data(stock_code, start_date, end_date, source)
                    
                    if df is not None and not df.empty:
                        # 缓存数据
                        self.cache_manager.set_dataframe(cache_key, df)
                        return df
                    return None
                    
                except Exception as e:
                    self.logger.error(f"获取股票数据失败 [{stock_code}]: {str(e)}")
                    return None
                    
            # 使用并行处理器获取数据
            self.logger.info(f"批量预加载{len(stock_codes)}只股票数据")
            results = self.processor.map(_get_stock_data, stock_codes, desc="预加载股票数据")
            
            # 统计结果
            success_count = sum(1 for df in results if df is not None)
            self.logger.info(f"批量预加载完成，成功加载{success_count}/{len(stock_codes)}只股票数据")
            
        except Exception as e:
            self.logger.error(f"批量预加载股票数据失败: {str(e)}")
            
    def _execute_custom_task(self, task: Dict[str, Any]) -> None:
        """执行自定义任务"""
        func = task.get("func")
        args = task.get("args", [])
        kwargs = task.get("kwargs", {})
        
        if not callable(func):
            self.logger.error("自定义任务缺少可调用函数")
            return
            
        try:
            self.logger.info(f"执行自定义预加载任务: {func.__name__}")
            result = func(*args, **kwargs)
            self.logger.info(f"自定义预加载任务完成: {func.__name__}")
            return result
            
        except Exception as e:
            self.logger.error(f"执行自定义任务失败 [{func.__name__}]: {str(e)}")
            
    def add_task(self, task: Dict[str, Any]) -> str:
        """
        添加预加载任务
        
        Args:
            task: 任务描述字典，必须包含type字段
            
        Returns:
            str: 任务ID
        """
        if "type" not in task:
            raise ValueError("任务必须包含type字段")
            
        # 生成任务ID
        if "id" not in task:
            task["id"] = f"{task['type']}_{int(time.time() * 1000)}"
            
        # 检查任务是否正在处理
        with self.processing_lock:
            if task["id"] in self.processing:
                self.logger.debug(f"任务已在处理中: {task['id']}")
                return task["id"]
                
        # 添加到队列
        self.task_queue.put(task)
        self.logger.debug(f"已添加预加载任务: {task['id']}")
        
        return task["id"]
        
    def preload_stock_data(self, stock_code: str, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None, source: str = "tushare") -> str:
        """
        预加载股票数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            source: 数据源，可选：tushare, akshare
            
        Returns:
            str: 任务ID
        """
        task = {
            "type": "stock_data",
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date,
            "source": source
        }
        
        return self.add_task(task)
        
    def preload_stock_list(self, source: str = "tushare") -> str:
        """
        预加载股票列表
        
        Args:
            source: 数据源，可选：tushare, akshare
            
        Returns:
            str: 任务ID
        """
        task = {
            "type": "stock_list",
            "source": source
        }
        
        return self.add_task(task)
        
    def preload_batch_stock_data(self, stock_codes: List[str], start_date: Optional[str] = None, 
                                end_date: Optional[str] = None, source: str = "tushare") -> str:
        """
        批量预加载股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            source: 数据源，可选：tushare, akshare
            
        Returns:
            str: 任务ID
        """
        task = {
            "type": "batch_stock_data",
            "stock_codes": stock_codes,
            "start_date": start_date,
            "end_date": end_date,
            "source": source
        }
        
        return self.add_task(task)
        
    def preload_custom(self, func: Callable, *args, **kwargs) -> str:
        """
        预加载自定义任务
        
        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数
            
        Returns:
            str: 任务ID
        """
        task = {
            "type": "custom",
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "id": f"custom_{func.__name__}_{int(time.time() * 1000)}"
        }
        
        return self.add_task(task)
        
    def is_task_processing(self, task_id: str) -> bool:
        """
        检查任务是否正在处理
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否正在处理
        """
        with self.processing_lock:
            return task_id in self.processing
            
    def get_stats(self) -> Dict[str, Any]:
        """
        获取预加载器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self.stats.copy()
        stats["queue_size"] = self.task_queue.qsize()
        stats["processing_count"] = len(self.processing)
        stats["is_active"] = self.preload_thread is not None and self.preload_thread.is_alive()
        
        return stats
        
    def preload_industry_stocks(self, industry: str, start_date: Optional[str] = None, 
                               end_date: Optional[str] = None, source: str = "tushare") -> str:
        """
        预加载某行业的所有股票数据
        
        Args:
            industry: 行业名称
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            source: 数据源，可选：tushare, akshare
            
        Returns:
            str: 任务ID
        """
        # 首先预加载股票列表
        self.preload_stock_list(source)
        
        # 定义获取行业股票并预加载的函数
        def _preload_industry_stocks() -> None:
            try:
                # 获取股票列表
                cache_key = {"type": "stock_list", "source": source}
                stock_list = self.cache_manager.get_dataframe(cache_key)
                
                if stock_list is None or stock_list.empty:
                    self.logger.warning(f"无法获取股票列表，预加载行业股票失败: {industry}")
                    return
                    
                # 过滤行业股票
                industry_stocks = stock_list[stock_list["industry"] == industry]["code"].tolist()
                
                if not industry_stocks:
                    self.logger.warning(f"找不到该行业的股票: {industry}")
                    return
                    
                # 批量预加载
                self.preload_batch_stock_data(industry_stocks, start_date, end_date, source)
                
            except Exception as e:
                self.logger.error(f"预加载行业股票失败 [{industry}]: {str(e)}")
                
        # 添加自定义任务
        return self.preload_custom(_preload_industry_stocks)
        
    def preload_top_stocks(self, top_n: int = 50, field: str = "market_cap", 
                          start_date: Optional[str] = None, end_date: Optional[str] = None,
                          source: str = "tushare") -> str:
        """
        预加载市值排名靠前的股票数据
        
        Args:
            top_n: 前N只股票
            field: 排序字段，如市值、成交量等
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            source: 数据源，可选：tushare, akshare
            
        Returns:
            str: 任务ID
        """
        # 定义获取排名靠前股票并预加载的函数
        def _preload_top_stocks() -> None:
            try:
                # 这里假设数据管理器有获取股票基本面信息的方法
                # 实际实现时可能需要调用专门的API
                # 此处仅作示例，实际项目中应根据具体数据源实现
                top_stocks = self.data_manager.get_top_stocks(top_n, field)
                
                if not top_stocks:
                    self.logger.warning(f"获取排名靠前的股票失败")
                    return
                    
                # 批量预加载
                self.preload_batch_stock_data(top_stocks, start_date, end_date, source)
                
            except Exception as e:
                self.logger.error(f"预加载排名靠前股票失败: {str(e)}")
                
        # 添加自定义任务
        return self.preload_custom(_preload_top_stocks) 