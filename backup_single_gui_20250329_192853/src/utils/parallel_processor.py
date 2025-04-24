"""
并行处理模块
使用多进程加速数据获取和处理
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Iterable, TypeVar, Generic
from tqdm import tqdm
import time
import os
from ..utils.logger import get_logger

T = TypeVar('T')
R = TypeVar('R')

class ParallelProcessor:
    """并行处理器类"""
    
    def __init__(self, use_threads: bool = False, max_workers: int = None):
        """
        初始化并行处理器
        
        Args:
            use_threads: 是否使用线程而非进程（IO密集型任务建议使用线程）
            max_workers: 最大工作进程/线程数，默认为CPU核心数
        """
        self.logger = get_logger("ParallelProcessor")
        self.use_threads = use_threads
        
        if max_workers is None:
            self.max_workers = mp.cpu_count()
        else:
            self.max_workers = max_workers
            
        self.logger.info(f"并行处理器初始化完成，使用{'线程' if use_threads else '进程'}，工作数：{self.max_workers}")
        
    def map(self, 
            func: Callable[[T], R], 
            items: List[T], 
            show_progress: bool = True,
            desc: str = "处理中",
            chunksize: int = 1) -> List[R]:
        """
        并行映射函数到数据项
        
        Args:
            func: 要应用的函数
            items: 要处理的数据项列表
            show_progress: 是否显示进度条
            desc: 进度条描述
            chunksize: 每个进程/线程处理的数据块大小
            
        Returns:
            List[R]: 处理结果列表
        """
        if not items:
            self.logger.warning("输入数据为空，无法进行并行处理")
            return []
            
        results = []
        executor_class = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        
        try:
            with executor_class(max_workers=self.max_workers) as executor:
                futures = [executor.submit(func, item) for item in items]
                
                if show_progress:
                    # 使用tqdm显示进度
                    for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            self.logger.error(f"处理任务时发生错误: {str(e)}")
                else:
                    # 不显示进度
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            self.logger.error(f"处理任务时发生错误: {str(e)}")
                            
            return results
            
        except Exception as e:
            self.logger.error(f"并行处理失败: {str(e)}")
            return []
            
    def process_batches(self, 
                        func: Callable[[List[T]], List[R]], 
                        items: List[T], 
                        batch_size: int = 10,
                        show_progress: bool = True,
                        desc: str = "批处理中") -> List[R]:
        """
        批量并行处理数据
        
        Args:
            func: 处理批量数据的函数
            items: 要处理的数据项列表
            batch_size: 每批数据大小
            show_progress: 是否显示进度条
            desc: 进度条描述
            
        Returns:
            List[R]: 处理结果列表
        """
        if not items:
            self.logger.warning("输入数据为空，无法进行批处理")
            return []
            
        # 创建批次
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        self.logger.info(f"已将{len(items)}个项目分成{len(batches)}个批次，每批{batch_size}个")
        
        all_results = []
        
        try:
            # 使用map处理批次
            batch_results = self.map(func, batches, show_progress, desc)
            
            # 合并结果
            for results in batch_results:
                all_results.extend(results)
                
            return all_results
            
        except Exception as e:
            self.logger.error(f"批处理失败: {str(e)}")
            return []
            
    @staticmethod
    def process_with_retry(func: Callable, max_retries: int = 3, retry_delay: float = 1.0, *args, **kwargs) -> Any:
        """
        带重试的函数执行
        
        Args:
            func: 要执行的函数
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            *args, **kwargs: 函数参数
            
        Returns:
            Any: 函数执行结果
        """
        logger = get_logger("RetryProcessor")
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"执行失败，将在{retry_delay}秒后重试（{attempt+1}/{max_retries}）: {str(e)}")
                    time.sleep(retry_delay)
                    # 线性增加重试延迟
                    retry_delay *= 1.5
                else:
                    logger.error(f"已达到最大重试次数，操作失败: {str(e)}")
                    raise 