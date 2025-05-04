#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并行处理工具模块

提供多进程与多线程并行计算功能，针对股票分析优化
"""

import os
import time
import logging
import threading
import concurrent.futures
import psutil
import queue
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Tuple, Union, Optional, Set
from functools import wraps

# 日志配置
logger = logging.getLogger(__name__)

class ParallelMode(Enum):
    """并行模式枚举"""
    THREAD = "thread"  # 多线程模式
    PROCESS = "process"  # 多进程模式
    AUTO = "auto"  # 自动选择模式

@dataclass
class TaskResult:
    """任务结果结构"""
    success: bool
    result: Any = None  
    error: Optional[Exception] = None
    task_id: Optional[str] = None
    execution_time: float = 0.0

class TaskQueue:
    """任务队列管理器
    
    管理并行执行的任务队列，支持优先级和取消
    """
    
    def __init__(self, max_size: int = 1000):
        """初始化任务队列
        
        Args:
            max_size: 队列最大大小
        """
        self.task_queue = queue.PriorityQueue(max_size)
        self.active_tasks = set()  # 当前活动任务ID集合
        self.canceled_tasks = set()  # 已取消的任务ID集合
        self.lock = threading.RLock()
        
    def add_task(self, task_id: str, task_func: Callable, 
                priority: int = 5, *args, **kwargs) -> bool:
        """添加任务到队列
        
        Args:
            task_id: 任务ID
            task_func: 任务函数
            priority: 优先级(0-10)，数字越小优先级越高
            *args, **kwargs: 传递给任务函数的参数
            
        Returns:
            bool: 是否成功添加
        """
        try:
            with self.lock:
                if task_id in self.active_tasks:
                    return False
                    
                self.active_tasks.add(task_id)
                
            # 将任务加入队列，包含: (优先级, 任务ID, 函数, 参数, 关键字参数)
            self.task_queue.put((priority, task_id, task_func, args, kwargs))
            return True
        except Exception as e:
            logger.error(f"添加任务到队列失败: {str(e)}")
            return False
            
    def cancel_task(self, task_id: str) -> bool:
        """取消排队中的任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否成功取消
        """
        with self.lock:
            if task_id not in self.active_tasks:
                return False
                
            self.canceled_tasks.add(task_id)
            return True
            
    def is_canceled(self, task_id: str) -> bool:
        """检查任务是否已取消
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否已取消
        """
        with self.lock:
            return task_id in self.canceled_tasks
            
    def get_task(self) -> Tuple[int, str, Callable, tuple, dict]:
        """获取下一个待执行任务
        
        Returns:
            Tuple: (优先级, 任务ID, 函数, 参数, 关键字参数)
        """
        return self.task_queue.get(block=True)
        
    def complete_task(self, task_id: str) -> None:
        """标记任务完成
        
        Args:
            task_id: 任务ID
        """
        with self.lock:
            if task_id in self.active_tasks:
                self.active_tasks.remove(task_id)
            if task_id in self.canceled_tasks:
                self.canceled_tasks.remove(task_id)
        
        self.task_queue.task_done()
        
    def get_active_count(self) -> int:
        """获取活动任务数量"""
        with self.lock:
            return len(self.active_tasks)
            
    def clear(self) -> None:
        """清空任务队列"""
        with self.lock:
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                    self.task_queue.task_done()
                except queue.Empty:
                    break
                    
            self.active_tasks.clear()
            self.canceled_tasks.clear()

class ParallelExecutor:
    """并行执行器
    
    根据任务特性自动或手动选择多进程/多线程执行模式
    支持任务队列、优先级和任务取消
    """
    
    def __init__(self, mode: ParallelMode = ParallelMode.AUTO, 
                 max_workers: Optional[int] = None,
                 timeout: int = 300,
                 cpu_threshold: float = 0.8,  # CPU使用率阈值
                 adaptive_mode: bool = True):  # 是否启用自适应模式切换
        """
        初始化并行执行器
        
        Args:
            mode: 并行模式，可选THREAD、PROCESS或AUTO
            max_workers: 最大工作线程/进程数，默认为None（自动选择）
            timeout: 任务超时时间（秒）
            cpu_threshold: CPU使用率阈值，超过此值时调整工作线程数
            adaptive_mode: 是否启用自适应模式切换
        """
        self.mode = mode
        self.timeout = timeout
        self.cpu_threshold = cpu_threshold
        self.adaptive_mode = adaptive_mode
        
        # 默认工作线程/进程数: 由系统CPU核心数智能决定
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            # 根据系统内存和CPU来优化worker数
            if self.mode == ParallelMode.PROCESS:
                # 进程模式占用更多内存，限制worker数量
                available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
                # 根据可用内存调整，每个进程估计使用500MB
                memory_based_workers = max(1, int(available_memory / 0.5))
                self.max_workers = min(cpu_count, memory_based_workers, 16)
            else:
                # 线程模式根据CPU核心数和负载调整
                cpu_percent = psutil.cpu_percent(interval=0.1)
                if cpu_percent > 85:  # CPU负载高时减少worker数
                    self.max_workers = max(2, int(cpu_count * 0.5))
                else:
                    self.max_workers = min(32, cpu_count + 4)
        else:
            self.max_workers = max_workers
            
        # 实际使用的worker数量，可通过自适应调整
        self.current_workers = self.max_workers
        
        # 任务队列
        self.task_queue = TaskQueue()
        
        # 初始化执行器
        self._thread_executor = None
        self._process_executor = None
        self._init_executors()
        
        # 统计信息
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
        }
        
        # 启动监控线程
        if self.adaptive_mode:
            self._start_resource_monitor()
            
    def _init_executors(self):
        """初始化执行器"""
        # 根据模式初始化相应的执行器
        if self.mode in (ParallelMode.THREAD, ParallelMode.AUTO):
            self._thread_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.current_workers,
                thread_name_prefix="ParallelWorker"
            )
            
        if self.mode in (ParallelMode.PROCESS, ParallelMode.AUTO):
            self._process_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=min(self.current_workers, os.cpu_count() or 4)
            )
    
    def _start_resource_monitor(self):
        """启动资源监控线程"""
        def monitor_resources():
            while True:
                try:
                    # 每30秒检查一次系统资源
                    time.sleep(30)
                    
                    # 检查CPU使用率
                    cpu_percent = psutil.cpu_percent(interval=1)
                    
                    # 根据CPU使用率调整worker数
                    if cpu_percent > self.cpu_threshold * 100:
                        # CPU使用率高，减少worker数
                        new_workers = max(2, int(self.current_workers * 0.8))
                        if new_workers < self.current_workers:
                            logger.info(f"系统负载高 (CPU: {cpu_percent}%)，减少worker数: {self.current_workers} -> {new_workers}")
                            self.current_workers = new_workers
                            # 重新初始化执行器
                            self._shutdown_executors()
                            self._init_executors()
                    elif cpu_percent < self.cpu_threshold * 50 and self.current_workers < self.max_workers:
                        # CPU使用率低，增加worker数
                        new_workers = min(self.max_workers, int(self.current_workers * 1.2))
                        if new_workers > self.current_workers:
                            logger.info(f"系统负载低 (CPU: {cpu_percent}%)，增加worker数: {self.current_workers} -> {new_workers}")
                            self.current_workers = new_workers
                            # 重新初始化执行器
                            self._shutdown_executors()
                            self._init_executors()
                            
                    # 如果是AUTO模式，还可以根据任务特性动态调整执行模式
                    if self.mode == ParallelMode.AUTO and self.stats["tasks_completed"] > 10:
                        # 根据历史任务执行情况分析最佳模式
                        pass
                        
                except Exception as e:
                    logger.error(f"资源监控线程出错: {str(e)}")
                    
        # 创建守护线程
        thread = threading.Thread(target=monitor_resources, daemon=True)
        thread.start()
        logger.debug("资源监控线程已启动")
    
    def _shutdown_executors(self):
        """关闭执行器"""
        if self._thread_executor:
            self._thread_executor.shutdown(wait=False)
            
        if self._process_executor:
            self._process_executor.shutdown(wait=False)
            
    def map(self, func: Callable, items: List[Any], *args, **kwargs) -> List[Any]:
        """
        对列表项并行应用函数，支持智能分配任务到最合适的执行器
        
        Args:
            func: 要应用的函数
            items: 项目列表
            *args: 传递给func的位置参数
            **kwargs: 传递给func的关键字参数
            
        Returns:
            List: 结果列表，与输入列表顺序相同
        """
        if not items:
            return []
            
        # 少于5个项目或总数据量较小时不使用并行
        if len(items) < 5:
            return [func(item, *args, **kwargs) for item in items]
        
        # 智能选择执行模式
        is_cpu_bound = kwargs.pop('is_cpu_bound', False)
        executor = self._select_executor(is_cpu_bound)
        
        # 创建部分应用的函数
        def process_item(item):
            start_time = time.time()
            try:
                result = func(item, *args, **kwargs)
                execution_time = time.time() - start_time
                return TaskResult(True, result=result, execution_time=execution_time)
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"处理项目出错 ({str(item)[:50]}): {str(e)}")
                return TaskResult(False, error=e, execution_time=execution_time)
        
        results = []
        start_time = time.time()
        
        try:
            with executor as exec_pool:
                futures = {exec_pool.submit(process_item, item): i for i, item in enumerate(items)}
                
                # 保证结果与输入顺序一致
                results = [None] * len(items)
                
                for future in concurrent.futures.as_completed(futures, timeout=self.timeout):
                    index = futures[future]
                    try:
                        task_result = future.result()
                        results[index] = task_result.result if task_result.success else None
                        
                        # 更新统计信息
                        with threading.RLock():
                            self.stats["tasks_completed"] += task_result.success
                            self.stats["tasks_failed"] += not task_result.success
                            self.stats["total_execution_time"] += task_result.execution_time
                    except Exception as e:
                        logger.error(f"任务执行失败: {str(e)}")
                        results[index] = None
                        self.stats["tasks_failed"] += 1
        except concurrent.futures.TimeoutError:
            logger.error(f"并行执行超时 (>{self.timeout}秒)")
        except Exception as e:
            logger.error(f"并行执行出错: {str(e)}")
        
        logger.debug(f"并行处理 {len(items)} 个项目耗时: {time.time() - start_time:.2f}秒")
        return results
    
    def _select_executor(self, is_cpu_bound: bool = False):
        """
        智能选择最合适的执行器
        
        Args:
            is_cpu_bound: 是否为CPU密集型任务
            
        Returns:
            执行器对象
        """
        if self.mode == ParallelMode.THREAD:
            return self._thread_executor
        elif self.mode == ParallelMode.PROCESS:
            return self._process_executor
        else:
            # 自动模式根据任务特性选择
            if is_cpu_bound:
                # CPU密集型任务使用进程池
                return self._process_executor
            else:
                # IO密集型任务使用线程池
                return self._thread_executor
    
    def execute_batch(self, func: Callable, items: List[Any], 
                     batch_size: int = 50, 
                     progress_callback: Optional[Callable] = None,
                     *args, **kwargs) -> List[Any]:
        """
        分批次并行执行任务，支持进度回调
        
        Args:
            func: 要应用的函数
            items: 项目列表
            batch_size: 每批次的项目数
            progress_callback: 进度回调函数，接收(已完成数, 总数, 当前批次)
            *args: 传递给func的位置参数
            **kwargs: 传递给func的关键字参数
            
        Returns:
            List: 合并后的结果列表
        """
        if not items:
            return []
            
        # 计算批次数
        total = len(items)
        batch_count = (total + batch_size - 1) // batch_size
        logger.info(f"任务分为 {batch_count} 批执行，共 {total} 个项目")
        
        all_results = []
        completed = 0
        
        for i in range(batch_count):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total)
            batch = items[start_idx:end_idx]
            
            logger.info(f"执行批次 {i+1}/{batch_count}, 项目数: {len(batch)}")
            batch_results = self.map(func, batch, *args, **kwargs)
            all_results.extend(batch_results)
            
            # 更新进度
            completed += len(batch)
            if progress_callback:
                try:
                    progress_callback(completed, total, i+1)
                except Exception as e:
                    logger.error(f"进度回调函数出错: {str(e)}")
            
            # 批次间短暂暂停，避免API限制
            if i < batch_count - 1:
                time.sleep(0.5)
        
        return all_results
        
    def execute_with_progress(self, func: Callable, items: List[Any], 
                             progress_callback: Optional[Callable] = None,
                             *args, **kwargs) -> List[Any]:
        """
        带进度回调的并行执行，提供实时进度更新
        
        Args:
            func: 要应用的函数
            items: 项目列表
            progress_callback: 进度回调函数，接收(当前项索引, 总项数, 当前项结果)
            *args: 传递给func的位置参数
            **kwargs: 传递给func的关键字参数
            
        Returns:
            List: 结果列表
        """
        if not items:
            return []
            
        total = len(items)
        results = [None] * total
        completed = 0
        
        # 线程安全的进度计数器
        lock = threading.RLock()
        
        def process_with_progress(idx, item):
            nonlocal completed
            start_time = time.time()
            
            try:
                result = func(item, *args, **kwargs)
                execution_time = time.time() - start_time
                
                # 更新进度
                with lock:
                    results[idx] = result
                    completed += 1
                    self.stats["tasks_completed"] += 1
                    self.stats["total_execution_time"] += execution_time
                    
                # 调用进度回调
                if progress_callback:
                    # 防止回调错误影响主流程
                    try:
                        progress_callback(idx, total, result)
                    except Exception as e:
                        logger.error(f"进度回调函数出错: {str(e)}")
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"处理项目 {str(item)[:50]} 出错: {str(e)}")
                
                # 更新进度
                with lock:
                    completed += 1
                    self.stats["tasks_failed"] += 1
                    self.stats["total_execution_time"] += execution_time
                    
                if progress_callback:
                    try:
                        progress_callback(idx, total, None)
                    except Exception as callback_e:
                        logger.error(f"进度回调函数出错: {str(callback_e)}")
                
                return None
                
        # 选择执行器
        is_cpu_bound = kwargs.pop('is_cpu_bound', False)
        executor = self._select_executor(is_cpu_bound)
        
        try:
            with executor as exec_pool:
                # 提交所有任务
                futures = [exec_pool.submit(process_with_progress, i, item) 
                          for i, item in enumerate(items)]
                
                # 等待所有任务完成
                concurrent.futures.wait(futures, timeout=self.timeout)
        except Exception as e:
            logger.error(f"并行执行出错: {str(e)}")
            
        return results
    
    def submit_task(self, func: Callable, task_id: str = None, 
                   priority: int = 5, *args, **kwargs) -> str:
        """
        提交任务到队列
        
        Args:
            func: 要执行的函数
            task_id: 任务ID，如果为None则自动生成
            priority: 优先级(0-10)，数字越小优先级越高
            *args: 传递给func的位置参数
            **kwargs: 传递给func的关键字参数
            
        Returns:
            str: 任务ID
        """
        # 生成任务ID
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000)}_{id(func)}"
            
        # 添加到队列
        success = self.task_queue.add_task(task_id, func, priority, *args, **kwargs)
        
        if success:
            self.stats["tasks_submitted"] += 1
            return task_id
        return None
        
    def process_queue(self, max_tasks: int = None, timeout: int = None) -> Dict[str, Any]:
        """
        处理队列中的任务
        
        Args:
            max_tasks: 最多处理的任务数，None表示全部
            timeout: 超时时间(秒)
            
        Returns:
            Dict: 处理结果统计
        """
        if max_tasks is None:
            max_tasks = self.task_queue.get_active_count()
            
        if timeout is None:
            timeout = self.timeout
            
        processed = 0
        succeeded = 0
        failed = 0
        start_time = time.time()
        
        # 选择执行器
        executor = self._select_executor()
        
        try:
            with executor as exec_pool:
                futures = {}
                
                while processed < max_tasks:
                    try:
                        # 非阻塞获取任务，如果队列为空则退出
                        if self.task_queue.task_queue.empty():
                            break
                            
                        priority, task_id, func, args, kwargs = self.task_queue.get_task()
                        
                        # 检查任务是否已取消
                        if self.task_queue.is_canceled(task_id):
                            self.task_queue.complete_task(task_id)
                            continue
                            
                        # 提交任务
                        future = exec_pool.submit(func, *args, **kwargs)
                        futures[future] = task_id
                        processed += 1
                        
                        # 检查是否超时
                        if time.time() - start_time > timeout:
                            logger.warning(f"队列处理超时 (>{timeout}秒)")
                            break
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.error(f"处理队列任务出错: {str(e)}")
                        failed += 1
                
                # 等待已提交的任务完成
                for future in concurrent.futures.as_completed(futures):
                    task_id = futures[future]
                    try:
                        result = future.result()
                        succeeded += 1
                    except Exception as e:
                        logger.error(f"任务 {task_id} 执行失败: {str(e)}")
                        failed += 1
                    finally:
                        self.task_queue.complete_task(task_id)
        except Exception as e:
            logger.error(f"处理任务队列出错: {str(e)}")
            
        # 更新统计信息
        self.stats["tasks_completed"] += succeeded
        self.stats["tasks_failed"] += failed
        
        return {
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed,
            "remaining": self.task_queue.get_active_count(),
            "execution_time": time.time() - start_time
        }
        
    def cancel_all_tasks(self) -> int:
        """
        取消所有排队中的任务
        
        Returns:
            int: 取消的任务数
        """
        with self.task_queue.lock:
            canceled = len(self.task_queue.active_tasks)
            # 将所有活动任务添加到已取消集合
            self.task_queue.canceled_tasks.update(self.task_queue.active_tasks)
            return canceled
    
    def shutdown(self, wait: bool = True):
        """
        关闭执行器
        
        Args:
            wait: 是否等待所有任务完成
        """
        # 关闭所有执行器
        if self._thread_executor:
            self._thread_executor.shutdown(wait=wait)
            
        if self._process_executor:
            self._process_executor.shutdown(wait=wait)
            
        # 清空任务队列
        self.task_queue.clear()
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取执行统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = dict(self.stats)
        
        # 计算平均执行时间
        if stats["tasks_completed"] > 0:
            stats["avg_execution_time"] = stats["total_execution_time"] / stats["tasks_completed"]
        else:
            stats["avg_execution_time"] = 0
            
        # 添加队列信息
        stats["queued_tasks"] = self.task_queue.get_active_count()
        stats["mode"] = self.mode.value
        stats["max_workers"] = self.max_workers
        stats["current_workers"] = self.current_workers
        
        return stats

# 优化的装饰器：使函数支持并行处理
def parallelize(mode: ParallelMode = ParallelMode.AUTO, max_workers: Optional[int] = None,
               batch_size: Optional[int] = None, is_cpu_bound: bool = False):
    """
    装饰器：使函数能够对列表输入并行处理
    
    Args:
        mode: 并行模式
        max_workers: 最大工作线程/进程数
        batch_size: 批处理大小，None表示不分批
        is_cpu_bound: 是否为CPU密集型任务，影响执行器选择
        
    Example:
        @parallelize(mode=ParallelMode.PROCESS, is_cpu_bound=True)
        def process_stock(stock_code):
            # 处理单只股票的代码
            return result
            
        # 调用时会自动并行处理
        results = process_stock(stock_list)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(items, *args, **kwargs):
            # 如果输入不是列表，直接调用原函数
            if not isinstance(items, (list, tuple)):
                return func(items, *args, **kwargs)
                
            # 确保kwargs中包含is_cpu_bound参数
            kwargs['is_cpu_bound'] = is_cpu_bound
                
            executor = ParallelExecutor(mode=mode, max_workers=max_workers)
            
            # 是否使用批处理
            if batch_size:
                return executor.execute_batch(func, items, batch_size, *args, **kwargs)
            else:
                return executor.map(func, items, *args, **kwargs)
                
        return wrapper
    return decorator

# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 示例函数：模拟处理单只股票
    def process_stock(stock_code):
        time.sleep(0.5)  # 模拟IO操作
        return f"处理结果: {stock_code}"
    
    # 测试代码
    stock_list = [f"00000{i}" for i in range(20)]
    
    # 使用执行器
    executor = ParallelExecutor(mode=ParallelMode.THREAD)
    results = executor.map(process_stock, stock_list)
    
    # 使用装饰器
    @parallelize(mode=ParallelMode.THREAD, batch_size=10)
    def batch_process(stock):
        return process_stock(stock)
        
    results_from_decorator = batch_process(stock_list)
    
    # 测试任务队列
    for i in range(5):
        executor.submit_task(process_stock, f"queue_task_{i}", priority=i, f"00000{i}")
    
    queue_stats = executor.process_queue()
    exec_stats = executor.get_statistics()
    
    print(f"处理完成: {len(results)} 项结果")
    print(f"队列处理统计: {queue_stats}")
    print(f"执行器统计: {exec_stats}")
    
    # 关闭执行器
    executor.shutdown() 