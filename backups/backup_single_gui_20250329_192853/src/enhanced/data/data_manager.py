#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版数据管理器
负责协调数据获取、处理、缓存和质量检查
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import hashlib
import json
from pathlib import Path
import concurrent.futures
from typing import Dict, List, Optional, Union, Any, Tuple

from src.enhanced.config.settings import (
    DATA_DIR, CACHE_DIR, ENHANCED_CACHE_DIR, 
    DATA_SOURCE_CONFIG, CACHE_CONFIG, 
    PROCESSING_CONFIG, INCREMENTAL_UPDATE_CONFIG,
    HEALTH_CHECK_CONFIG
)
from src.enhanced.data.fetchers.data_source_manager import DataSourceManager
from src.enhanced.data.processors.optimized_processor import OptimizedDataProcessor
from src.enhanced.data.quality.data_quality_checker import DataQualityChecker
from src.enhanced.data.cache.cache_manager import CacheManager

# 设置日志
logger = logging.getLogger(__name__)

class EnhancedDataManager:
    """增强版数据管理器，集中处理数据获取、缓存和处理"""
    
    def __init__(self, config: Dict = None):
        """
        初始化数据管理器
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config if config is not None else DATA_SOURCE_CONFIG
        self.processor = OptimizedDataProcessor()
        self.quality_checker = DataQualityChecker()
        self.cache_manager = CacheManager(
            cache_dir=ENHANCED_CACHE_DIR,
            memory_cache_size=CACHE_CONFIG.get("memory_cache_size", 128),
            max_age_hours=CACHE_CONFIG.get("disk_cache_max_age", 24)
        )
        
        # 创建数据源管理器
        self.data_source_manager = DataSourceManager(self.config)
        
        # 并行处理设置
        self.parallel = PROCESSING_CONFIG.get("parallel", True)
        self.num_workers = PROCESSING_CONFIG.get("num_workers", 4)
        self.chunk_size = PROCESSING_CONFIG.get("chunk_size", 5000)
        
        # 增量更新设置
        self.incremental_enabled = INCREMENTAL_UPDATE_CONFIG.get("enabled", True)
        self.batch_size = INCREMENTAL_UPDATE_CONFIG.get("batch_size", 1000)
        
        logger.info("增强版数据管理器初始化完成")
    
    def get_stock_data(self, stock_code: str, start_date: str, end_date: str = None, 
                      force_refresh: bool = False) -> pd.DataFrame:
        """
        获取股票数据，自动处理缓存和增量更新
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD，默认为今天
            force_refresh: 是否强制刷新缓存
            
        Returns:
            DataFrame: 处理后的股票数据
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 生成缓存键
        cache_key = f"stock_data_{stock_code}_{start_date}_{end_date}"
        
        # 如果不强制刷新，尝试从缓存获取
        if not force_refresh:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                logger.info(f"从缓存加载股票 {stock_code} 的数据")
                return cached_data
        
        # 如果启用增量更新，检查是否有部分数据已缓存
        if self.incremental_enabled and not force_refresh:
            # 尝试使用增量更新
            incremental_result = self._get_incremental_data(stock_code, start_date, end_date)
            if incremental_result is not None:
                logger.info(f"使用增量更新获取股票 {stock_code} 的数据")
                # 保存到缓存
                self.cache_manager.set(cache_key, incremental_result)
                return incremental_result
        
        # 直接获取完整数据
        logger.info(f"获取股票 {stock_code} 的完整数据")
        data = self.data_source_manager.get_daily_data(stock_code, start_date, end_date)
        
        if data is not None and not data.empty:
            # 保存到缓存
            self.cache_manager.set(cache_key, data)
            return data
        
        logger.warning(f"未能获取到股票 {stock_code} 的数据")
        return None
    
    def _get_incremental_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        增量获取数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame或None: 增量更新后的数据，如果无法增量更新则返回None
        """
        # 搜索可能的现有缓存
        existing_data = None
        
        # 查找包含部分数据的缓存
        for cache_key in self.cache_manager.list_keys():
            if f"stock_data_{stock_code}_" in cache_key:
                cache_info = cache_key.split('_')
                if len(cache_info) >= 5:  # stock_data_股票代码_开始日期_结束日期
                    cache_start = cache_info[-2]
                    cache_end = cache_info[-1]
                    
                    # 如果缓存的日期范围包含请求的部分范围，可以使用增量更新
                    if cache_start <= start_date:
                        logger.info(f"找到可用的缓存数据: {cache_key}")
                        existing_data = self.cache_manager.get(cache_key)
                        if existing_data is not None:
                            break
        
        if existing_data is None:
            return None
        
        # 找到现有数据的最后日期
        last_date = existing_data['date'].max() if 'date' in existing_data.columns else None
        if last_date is None:
            return None
        
        # 如果缓存数据已经覆盖了请求的时间范围，直接过滤并返回
        if last_date >= end_date:
            return existing_data[(existing_data['date'] >= start_date) & (existing_data['date'] <= end_date)]
        
        # 获取增量数据
        new_start_date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        incremental_data = self.data_source_manager.get_daily_data(stock_code, new_start_date, end_date)
        
        if incremental_data is None or incremental_data.empty:
            # 没有新数据，返回已有数据的相关部分
            return existing_data[(existing_data['date'] >= start_date) & (existing_data['date'] <= end_date)]
        
        # 合并数据
        combined_data = pd.concat([existing_data, incremental_data], ignore_index=True)
        # 去重和排序
        combined_data = combined_data.drop_duplicates(subset=['date']).sort_values('date')
        
        # 过滤时间范围
        filtered_data = combined_data[(combined_data['date'] >= start_date) & (combined_data['date'] <= end_date)]
        
        return filtered_data
    
    def process_stocks_in_parallel(self, stock_codes: List[str], start_date: str, end_date: str = None, 
                                 force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        并行处理多个股票的数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期，默认为今天
            force_refresh: 是否强制刷新缓存
            
        Returns:
            Dict: 股票代码到数据的映射
        """
        if not self.parallel or len(stock_codes) == 1:
            # 串行处理
            results = {}
            for code in stock_codes:
                results[code] = self.get_stock_data(code, start_date, end_date, force_refresh)
            return results
        
        # 并行处理
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_code = {
                executor.submit(self.get_stock_data, code, start_date, end_date, force_refresh): code 
                for code in stock_codes
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_code):
                stock_code = future_to_code[future]
                try:
                    results[stock_code] = future.result()
                except Exception as e:
                    logger.error(f"处理股票 {stock_code} 时出错: {str(e)}")
                    results[stock_code] = None
        
        return results
    
    def clear_cache(self, pattern: str = None):
        """
        清除缓存
        
        Args:
            pattern: 要清除的缓存键模式，None表示清除所有
        """
        self.cache_manager.clear(pattern)
        logger.info(f"已清除缓存: {pattern or '所有'}")
    
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        
        Returns:
            DataFrame: 股票列表
        """
        # 使用缓存
        cache_key = "stock_list"
        cached_data = self.cache_manager.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        # 获取数据
        data = self.data_source_manager.get_stock_list()
        if data is not None:
            self.cache_manager.set(cache_key, data)
        
        return data
    
    def save_to_file(self, data: pd.DataFrame, filename: str, format_type: str = "parquet"):
        """
        保存数据到文件
        
        Args:
            data: 要保存的数据
            filename: 文件名
            format_type: 文件格式，支持parquet、csv、hdf5
        """
        if data is None or data.empty:
            logger.warning(f"没有数据可保存到 {filename}")
            return
        
        # 确保文件扩展名正确
        if format_type == "parquet" and not filename.endswith(".parquet"):
            filename += ".parquet"
        elif format_type == "csv" and not filename.endswith(".csv"):
            filename += ".csv"
        elif format_type == "hdf5" and not filename.endswith(".h5"):
            filename += ".h5"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # 保存数据
        try:
            if format_type == "parquet":
                data.to_parquet(filename, compression=PROCESSING_CONFIG.get("cache_compression", "snappy"))
            elif format_type == "csv":
                data.to_csv(filename, index=False)
            elif format_type == "hdf5":
                data.to_hdf(filename, key='data', mode='w')
            else:
                logger.warning(f"不支持的文件格式: {format_type}")
                return
            
            logger.info(f"数据已保存到 {filename}")
        except Exception as e:
            logger.error(f"保存数据到 {filename} 失败: {str(e)}") 