#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据源管理器
管理多个数据源，自动进行健康检查和故障转移
"""

import time
import logging
import pandas as pd
import numpy as np
import os
import json
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from functools import wraps, partial

from src.enhanced.config.settings import HEALTH_CHECK_CONFIG, DATA_SOURCE_CONFIG, CACHE_CONFIG, ENHANCED_CACHE_DIR, PROCESSING_CONFIG
from src.enhanced.data.fetchers.tushare_fetcher import EnhancedTushareFetcher
from src.enhanced.data.fetchers.akshare_fetcher import EnhancedAKShareFetcher
from src.enhanced.data.fetchers.joinquant_fetcher import EnhancedJoinQuantFetcher

# 设置日志
logger = logging.getLogger(__name__)

# 独立的缓存装饰器函数
def with_cache(method_name: str):
    """
    装饰器: 为方法添加缓存功能
    
    Args:
        method_name: 方法名
        
    Returns:
        callable: 装饰后的方法
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.cache_enabled:
                return func(self, *args, **kwargs)
                
            # 从关键字参数中提取强制刷新标志
            force_refresh = kwargs.pop('force_refresh', False)
            
            # 生成缓存键
            cache_key = self._get_cache_key(method_name, *args, **kwargs)
            
            # 如果不需要强制刷新，尝试从缓存获取
            if not force_refresh:
                cached_data = self._get_from_cache(cache_key)
                if cached_data is not None:
                    return cached_data
            
            # 执行原始方法
            result = func(self, *args, **kwargs)
            
            # 如果有结果，保存到缓存
            if result is not None:
                self._save_to_cache(cache_key, result)
                
            return result
        return wrapper
    return decorator

# 独立的故障转移装饰器函数
def with_failover(method_name: str):
    """
    装饰器: 为方法添加故障转移逻辑
    
    Args:
        method_name: 数据源的方法名
        
    Returns:
        callable: 装饰后的方法
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 从关键字参数中提取首选数据源
            preferred_sources = kwargs.pop('preferred_sources', None)
            
            # 记录遇到的异常
            exceptions = {}
            
            # 如果提供了首选数据源，先尝试它们
            if preferred_sources:
                for source_name in preferred_sources:
                    if not self.health_check(source_name):
                        logger.debug(f"跳过不健康的首选数据源: {source_name}")
                        continue
                    
                    data_source = self.data_sources[source_name]
                    try:
                        logger.debug(f"尝试使用首选数据源 {source_name} 的 {method_name} 方法")
                        method = getattr(data_source, method_name)
                        return method(*args[1:], **kwargs)
                    except Exception as e:
                        exceptions[source_name] = str(e)
                        logger.warning(f"使用数据源 {source_name} 失败: {str(e)}")
                        # 标记为不健康
                        self.health_status[source_name] = False
            
            # 尝试主数据源(如果它不在首选数据源中)
            if self.primary_source and (not preferred_sources or self.primary_source not in preferred_sources):
                if self.health_check(self.primary_source):
                    try:
                        logger.debug(f"尝试使用主数据源 {self.primary_source} 的 {method_name} 方法")
                        method = getattr(self.data_sources[self.primary_source], method_name)
                        return method(*args[1:], **kwargs)
                    except Exception as e:
                        exceptions[self.primary_source] = str(e)
                        logger.warning(f"使用主数据源 {self.primary_source} 失败: {str(e)}")
                        # 标记为不健康
                        self.health_status[self.primary_source] = False
            
            # 尝试其他健康的数据源
            for source_name, data_source in self.data_sources.items():
                # 跳过已尝试过的
                if (preferred_sources and source_name in preferred_sources) or source_name == self.primary_source:
                    continue
                
                if not self.health_check(source_name):
                    logger.debug(f"跳过不健康的数据源: {source_name}")
                    continue
                
                try:
                    logger.debug(f"尝试使用备用数据源 {source_name} 的 {method_name} 方法")
                    method = getattr(data_source, method_name)
                    return method(*args[1:], **kwargs)
                except Exception as e:
                    exceptions[source_name] = str(e)
                    logger.warning(f"使用数据源 {source_name} 失败: {str(e)}")
                    # 标记为不健康
                    self.health_status[source_name] = False
            
            # 所有数据源都失败了
            logger.error(f"所有数据源的 {method_name} 方法都失败了")
            for source_name, error in exceptions.items():
                logger.error(f"  - {source_name}: {error}")
            
            return None
        
        return wrapper
    return decorator

class DataSourceManager:
    """
    数据源管理器
    管理多个数据源，提供健康检查和故障转移功能
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化数据源管理器
        
        Args:
            config: 数据源配置字典，如果为None则使用默认配置
        """
        self.config = config if config is not None else DATA_SOURCE_CONFIG
        self.data_sources = {}
        self.health_status = {}
        self.last_check_time = {}
        
        # 健康检查配置
        self.health_check_enabled = HEALTH_CHECK_CONFIG.get("enabled", True)
        self.check_interval = HEALTH_CHECK_CONFIG.get("check_interval", 300)  # 默认5分钟
        self.retry_interval = HEALTH_CHECK_CONFIG.get("retry_interval", 1800)  # 默认30分钟
        
        # 缓存配置
        self.cache_enabled = CACHE_CONFIG.get("enabled", True)
        self.cache_dir = ENHANCED_CACHE_DIR
        self.disk_cache_max_age = CACHE_CONFIG.get("disk_cache_max_age", 24) * 3600  # 转换为秒
        self.memory_cache = {}  # 内存缓存
        self.memory_cache_expires = {}  # 内存缓存过期时间
        self.memory_cache_size = CACHE_CONFIG.get("memory_cache_size", 128) * 1024 * 1024  # 转换为字节
        self.current_memory_cache_size = 0  # 当前内存缓存大小(字节)
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 初始化数据源
        self._init_data_sources()
        
        # 记录主数据源
        self.primary_source = self._get_primary_source_name()
        
        logger.info(f"数据源管理器初始化完成，已注册 {len(self.data_sources)} 个数据源，主数据源: {self.primary_source}")
    
    def _init_data_sources(self):
        """初始化所有启用的数据源"""
        for source_name, source_config in self.config.items():
            if not source_config.get('enabled', False):
                logger.info(f"数据源 {source_name} 未启用，跳过初始化")
                continue
                
            try:
                if source_name == 'tushare':
                    tushare_config = source_config.copy()
                    # 处理rate_limit配置
                    if 'rate_limit' in tushare_config:
                        rate_limit_config = tushare_config.pop('rate_limit')
                        tushare_config['rate_limit'] = rate_limit_config.get('calls_per_minute', 500) / 60  # 转换为每秒请求数
                    # 处理retry配置
                    if 'retry' in tushare_config:
                        retry_config = tushare_config.pop('retry')
                        tushare_config['connection_retries'] = retry_config.get('max_retries', 3)
                        tushare_config['retry_delay'] = retry_config.get('retry_interval', 5)
                    
                    self.data_sources['tushare'] = EnhancedTushareFetcher(tushare_config)
                    self.health_status['tushare'] = True
                    self.last_check_time['tushare'] = 0
                    
                elif source_name == 'akshare':
                    akshare_config = source_config.copy()
                    # 处理rate_limit配置
                    if 'rate_limit' in akshare_config:
                        rate_limit_config = akshare_config.pop('rate_limit')
                        akshare_config['rate_limit'] = rate_limit_config.get('calls_per_minute', 120) / 60  # 转换为每秒请求数
                    # 处理retry配置
                    if 'retry' in akshare_config:
                        retry_config = akshare_config.pop('retry')
                        akshare_config['connection_retries'] = retry_config.get('max_retries', 3)
                        akshare_config['retry_delay'] = retry_config.get('retry_interval', 5)
                    
                    self.data_sources['akshare'] = EnhancedAKShareFetcher(akshare_config)
                    self.health_status['akshare'] = True
                    self.last_check_time['akshare'] = 0
                    
                elif source_name == 'joinquant':
                    self.data_sources['joinquant'] = EnhancedJoinQuantFetcher(source_config)
                    self.health_status['joinquant'] = True
                    self.last_check_time['joinquant'] = 0
                    
            except Exception as e:
                logger.error(f"初始化数据源 {source_name} 失败: {str(e)}")
                self.health_status[source_name] = False
                
        logger.info(f"成功初始化 {len(self.data_sources)} 个数据源")
    
    def _get_primary_source_name(self) -> str:
        """获取主数据源名称"""
        # 查找配置为主数据源的
        for name, config in self.config.items():
            if config.get('is_primary', False) and name in self.data_sources:
                return name
        
        # 如果没有配置主数据源，返回第一个可用的数据源
        if self.data_sources:
            return next(iter(self.data_sources.keys()))
        
        return None
    
    def health_check(self, source_name: str) -> bool:
        """
        检查数据源健康状态
        
        Args:
            source_name: 数据源名称
            
        Returns:
            bool: 数据源是否健康
        """
        if not self.health_check_enabled:
            # 健康检查禁用，假设所有数据源都健康
            return source_name in self.data_sources
        
        now = time.time()
        
        # 如果最近刚检查过，直接返回状态
        if now - self.last_check_time.get(source_name, 0) < self.check_interval:
            return self.health_status.get(source_name, False)
        
        # 更新检查时间
        self.last_check_time[source_name] = now
        
        # 如果数据源不存在，返回False
        if source_name not in self.data_sources:
            return False
        
        # 执行健康检查
        try:
            data_source = self.data_sources[source_name]
            
            # 如果数据源有自己的健康检查方法，使用它
            if hasattr(data_source, 'check_health'):
                is_healthy = data_source.check_health()
            else:
                # 否则尝试获取股票列表作为健康检查
                test_result = data_source.get_stock_list()
                is_healthy = test_result is not None and not (isinstance(test_result, pd.DataFrame) and test_result.empty)
            
            self.health_status[source_name] = is_healthy
            
            if is_healthy:
                logger.debug(f"数据源 {source_name} 健康检查通过")
            else:
                logger.warning(f"数据源 {source_name} 健康检查失败")
            
            return is_healthy
        except Exception as e:
            logger.error(f"数据源 {source_name} 健康检查出错: {str(e)}")
            self.health_status[source_name] = False
            return False
    
    def get_healthy_source(self, preferred_sources: List[str] = None) -> Optional[str]:
        """
        获取健康的数据源名称
        
        Args:
            preferred_sources: 首选数据源列表
            
        Returns:
            str: 健康的数据源名称，如果没有则返回None
        """
        # 如果提供了首选数据源，先检查它们
        if preferred_sources:
            for source_name in preferred_sources:
                if self.health_check(source_name):
                    return source_name
        
        # 如果主数据源健康，返回主数据源
        if self.primary_source and self.health_check(self.primary_source):
            return self.primary_source
        
        # 否则遍历所有数据源，返回第一个健康的
        for source_name in self.data_sources:
            if source_name != self.primary_source and self.health_check(source_name):
                return source_name
        
        # 如果所有数据源都不健康，尝试重置一段时间前失败的数据源
        for source_name in self.data_sources:
            if not self.health_status.get(source_name, True) and time.time() - self.last_check_time.get(source_name, 0) > self.retry_interval:
                # 重置检查时间，触发下次健康检查
                self.last_check_time[source_name] = 0
        
        # 再次尝试获取健康的数据源
        for source_name in self.data_sources:
            if self.health_check(source_name):
                return source_name
        
        return None
    
    @property
    def available_sources(self) -> List[str]:
        """获取可用的数据源列表"""
        return [name for name in self.data_sources if self.health_check(name)]
    
    # 以下方法为通用数据获取接口，会自动进行故障转移
    
    @with_cache(method_name="get_stock_list")
    @with_failover(method_name="get_stock_list")
    def get_stock_list(self) -> Optional[pd.DataFrame]:
        """获取股票列表，自动故障转移"""
        pass
    
    @with_cache(method_name="get_daily_data")
    @with_failover(method_name="get_daily_data")
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str = None) -> Optional[pd.DataFrame]:
        """获取日线数据，自动故障转移"""
        pass
    
    @with_cache(method_name="get_industry_list")
    @with_failover(method_name="get_industry_list")
    def get_industry_list(self) -> Optional[pd.DataFrame]:
        """获取行业列表，自动故障转移"""
        pass
    
    @with_cache(method_name="get_stock_fund_flow")
    @with_failover(method_name="get_stock_fund_flow")
    def get_stock_fund_flow(self, stock_code: str, start_date: str, end_date: str = None) -> Optional[pd.DataFrame]:
        """获取个股资金流向，自动故障转移"""
        pass
    
    @with_cache(method_name="get_continuous_limit_up_stocks")
    @with_failover(method_name="get_continuous_limit_up_stocks")
    def get_continuous_limit_up_stocks(self, days: int = 1, end_date: str = None) -> Optional[pd.DataFrame]:
        """获取连续涨停股票，自动故障转移"""
        pass
    
    # 添加更多数据获取方法
    
    @with_cache(method_name="get_stock_financial_data")
    @with_failover(method_name="get_stock_financial_data")
    def get_stock_financial_data(self, stock_code: str, report_type: str = 'annual') -> Optional[pd.DataFrame]:
        """
        获取股票财务数据，自动故障转移
        
        Args:
            stock_code: 股票代码
            report_type: 报告类型，可选 'annual'(年报), 'quarterly'(季报)
            
        Returns:
            pd.DataFrame: 财务数据
        """
        pass
        
    @with_cache(method_name="get_stock_index_data")
    @with_failover(method_name="get_stock_index_data")
    def get_stock_index_data(self, index_code: str, start_date: str, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取指数数据，自动故障转移
        
        Args:
            index_code: 指数代码，如上证指数(000001.SH)
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 指数数据
        """
        pass
        
    @with_cache(method_name="get_stock_holder_change")
    @with_failover(method_name="get_stock_holder_change")
    def get_stock_holder_change(self, stock_code: str, start_date: str, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取股东变动情况，自动故障转移
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 股东变动数据
        """
        pass
        
    @with_cache(method_name="get_concept_stocks")
    @with_failover(method_name="get_concept_stocks")
    def get_concept_stocks(self, concept_name: str = None) -> Optional[pd.DataFrame]:
        """
        获取概念股，自动故障转移
        
        Args:
            concept_name: 概念名称，如None则返回所有概念
            
        Returns:
            pd.DataFrame: 概念股数据
        """
        pass
        
    @with_cache(method_name="get_industry_stocks")
    @with_failover(method_name="get_industry_stocks")
    def get_industry_stocks(self, industry_code: str) -> Optional[pd.DataFrame]:
        """
        获取行业成分股，自动故障转移
        
        Args:
            industry_code: 行业代码
            
        Returns:
            pd.DataFrame: 行业成分股数据
        """
        pass
        
    @with_cache(method_name="get_hk_stock_list")
    @with_failover(method_name="get_hk_stock_list")
    def get_hk_stock_list(self) -> Optional[pd.DataFrame]:
        """
        获取港股列表，自动故障转移
        
        Returns:
            pd.DataFrame: 港股列表
        """
        pass
        
    @with_cache(method_name="get_global_index_data")
    @with_failover(method_name="get_global_index_data")
    def get_global_index_data(self, index_code: str, start_date: str, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取全球指数数据，自动故障转移
        
        Args:
            index_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 指数数据
        """
        pass
        
    @with_cache(method_name="get_stock_news")
    @with_failover(method_name="get_stock_news")
    def get_stock_news(self, stock_code: str, limit: int = 50) -> Optional[pd.DataFrame]:
        """
        获取个股新闻，自动故障转移
        
        Args:
            stock_code: 股票代码
            limit: 获取条数
            
        Returns:
            pd.DataFrame: 新闻数据
        """
        pass
        
    # 高级数据获取方法
    
    def get_multi_stocks_daily_data(self, stock_codes: List[str], start_date: str, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        获取多只股票的日线数据，带有并行和故障转移
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, pd.DataFrame]: 股票代码到日线数据的映射
        """
        return self._parallel_fetch(
            self.get_daily_data, 
            stock_codes,
            additional_args=[start_date, end_date]
        )
        
    def _parallel_fetch(self, fetch_func: Callable, item_list: List, additional_args: List = None, additional_kwargs: Dict = None) -> Dict:
        """
        通用并行获取数据方法
        
        Args:
            fetch_func: 获取单个项目数据的函数
            item_list: 项目列表
            additional_args: 额外的位置参数
            additional_kwargs: 额外的关键字参数
            
        Returns:
            Dict: 项目到数据的映射
        """
        if additional_args is None:
            additional_args = []
        if additional_kwargs is None:
            additional_kwargs = {}
            
        results = {}
        parallel_enabled = PROCESSING_CONFIG.get("parallel", True)
        num_workers = PROCESSING_CONFIG.get("num_workers", min(32, os.cpu_count() + 4))
        
        # 如果禁用并行或项目少于5个，使用串行处理
        if not parallel_enabled or len(item_list) < 5:
            for item in item_list:
                try:
                    results[item] = fetch_func(item, *additional_args, **additional_kwargs)
                except Exception as e:
                    logger.error(f"获取项目 {item} 的数据失败: {str(e)}")
                    results[item] = None
            return results
            
        # 创建部分函数，包含额外参数
        partial_func = partial(self._fetch_single_item, fetch_func=fetch_func, additional_args=additional_args, additional_kwargs=additional_kwargs)
        
        # 使用线程池并行获取数据
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_item = {executor.submit(partial_func, item): item for item in item_list}
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    results[item] = future.result()
                except Exception as e:
                    logger.error(f"并行获取项目 {item} 的数据失败: {str(e)}")
                    results[item] = None
                    
        return results
        
    def _fetch_single_item(self, item, fetch_func, additional_args, additional_kwargs):
        """获取单个项目的数据"""
        try:
            return fetch_func(item, *additional_args, **additional_kwargs)
        except Exception as e:
            logger.error(f"获取项目 {item} 的数据失败: {str(e)}")
            return None
            
    def get_multi_stocks_financial_data(self, stock_codes: List[str], report_type: str = 'annual') -> Dict[str, pd.DataFrame]:
        """
        获取多只股票的财务数据，并行处理
        
        Args:
            stock_codes: 股票代码列表
            report_type: 报告类型
            
        Returns:
            Dict[str, pd.DataFrame]: 股票代码到财务数据的映射
        """
        return self._parallel_fetch(
            self.get_stock_financial_data,
            stock_codes,
            additional_kwargs={'report_type': report_type}
        )
        
    def get_multi_stocks_fund_flow(self, stock_codes: List[str], start_date: str, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        获取多只股票的资金流向，并行处理
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, pd.DataFrame]: 股票代码到资金流向数据的映射
        """
        return self._parallel_fetch(
            self.get_stock_fund_flow,
            stock_codes,
            additional_args=[start_date, end_date]
        )
        
    def get_stocks_by_industry(self, industry_codes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        获取多个行业的成分股，并行处理
        
        Args:
            industry_codes: 行业代码列表
            
        Returns:
            Dict[str, pd.DataFrame]: 行业代码到成分股数据的映射
        """
        return self._parallel_fetch(
            self.get_industry_stocks,
            industry_codes
        )
        
    def get_industry_performance(self, trade_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取行业表现数据
        
        Args:
            trade_date: 交易日期
            
        Returns:
            pd.DataFrame: 行业表现数据
        """
        try:
            # 获取行业列表
            industries = self.get_industry_list()
            if industries is None or industries.empty:
                logger.error("获取行业列表失败")
                return None
                
            # 获取当前所有股票数据
            all_stocks_data = self.get_all_stock_data_on_date(trade_date)
            if all_stocks_data is None or all_stocks_data.empty:
                logger.error(f"获取 {trade_date} 的所有股票行情数据失败")
                return None
                
            # 获取所有股票的行业归属
            stock_industries = {}
            industry_results = {}
            
            # 并行获取各行业成分股
            industry_codes = industries['industry_code'].tolist()
            industry_stocks_dict = self.get_stocks_by_industry(industry_codes)
            
            # 处理行业数据
            industry_perf_list = []
            
            for industry_code, stocks_df in industry_stocks_dict.items():
                if stocks_df is None or stocks_df.empty:
                    continue
                    
                # 获取行业名称
                industry_name = industries[industries['industry_code'] == industry_code]['industry_name'].iloc[0]
                
                # 获取该行业所有股票的行情数据
                industry_stock_codes = stocks_df['code'].tolist()
                industry_stocks_data = all_stocks_data[all_stocks_data['code'].isin(industry_stock_codes)]
                
                if len(industry_stocks_data) == 0:
                    continue
                    
                # 计算行业平均涨跌幅
                industry_stocks_data['change_pct'] = (industry_stocks_data['close'] - industry_stocks_data['open']) / industry_stocks_data['open'] * 100
                avg_change = industry_stocks_data['change_pct'].mean()
                
                # 计算行业总成交量和成交额
                total_volume = industry_stocks_data['volume'].sum() if 'volume' in industry_stocks_data.columns else 0
                total_amount = industry_stocks_data['amount'].sum() if 'amount' in industry_stocks_data.columns else 0
                
                # 计算行业涨跌家数
                up_count = len(industry_stocks_data[industry_stocks_data['change_pct'] > 0])
                down_count = len(industry_stocks_data[industry_stocks_data['change_pct'] < 0])
                flat_count = len(industry_stocks_data) - up_count - down_count
                
                # 找出行业领涨和领跌股
                if len(industry_stocks_data) > 0:
                    leading_stock = industry_stocks_data.loc[industry_stocks_data['change_pct'].idxmax()]
                    lagging_stock = industry_stocks_data.loc[industry_stocks_data['change_pct'].idxmin()]
                    leading_stock_info = {
                        'code': leading_stock['code'],
                        'name': leading_stock['name'] if 'name' in leading_stock else '',
                        'change_pct': leading_stock['change_pct']
                    }
                    lagging_stock_info = {
                        'code': lagging_stock['code'],
                        'name': lagging_stock['name'] if 'name' in lagging_stock else '',
                        'change_pct': lagging_stock['change_pct']
                    }
                else:
                    leading_stock_info = None
                    lagging_stock_info = None
                
                # 添加行业表现数据
                industry_perf = {
                    'date': trade_date,
                    'industry_code': industry_code,
                    'industry_name': industry_name,
                    'stock_count': len(industry_stocks_data),
                    'avg_change_pct': avg_change,
                    'total_volume': total_volume,
                    'total_amount': total_amount,
                    'up_count': up_count,
                    'down_count': down_count,
                    'flat_count': flat_count,
                    'leading_stock': leading_stock_info,
                    'lagging_stock': lagging_stock_info
                }
                
                industry_perf_list.append(industry_perf)
                
            # 创建结果DataFrame
            if industry_perf_list:
                # 将字典列表转换为DataFrame
                result_df = pd.DataFrame(industry_perf_list)
                
                # 按平均涨跌幅排序
                result_df = result_df.sort_values('avg_change_pct', ascending=False)
                
                logger.debug(f"成功获取 {trade_date} 的 {len(result_df)} 个行业表现数据")
                return result_df
                
            logger.warning(f"在 {trade_date} 没有获取到任何行业表现数据")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取行业表现数据失败: {str(e)}")
            return None
        
    def get_all_stock_data_on_date(self, trade_date: str) -> Optional[pd.DataFrame]:
        """
        获取指定交易日所有股票的行情数据
        
        Args:
            trade_date: 交易日期
            
        Returns:
            pd.DataFrame: 所有股票的行情数据
        """
        try:
            # 获取健康的数据源
            source_name = self.get_healthy_source()
            if not source_name:
                logger.error("没有可用的数据源来获取交易日行情数据")
                return None
                
            # 使用主数据源获取所有股票行情
            data_source = self.data_sources[source_name]
            if hasattr(data_source, 'get_all_stock_data_on_date'):
                logger.debug(f"使用数据源 {source_name} 获取 {trade_date} 的所有股票行情")
                return data_source.get_all_stock_data_on_date(trade_date)
            
            # 如果没有实现此方法，尝试获取股票列表然后一个个获取
            logger.debug(f"数据源 {source_name} 没有实现get_all_stock_data_on_date方法，使用替代方法")
            stocks = self.get_stock_list()
            if stocks is None or stocks.empty:
                logger.error("获取股票列表失败")
                return None
                
            # 收集所有股票的日线数据
            all_data = []
            for _, stock in stocks.iterrows():
                stock_code = stock['code']
                try:
                    daily_data = self.get_daily_data(stock_code, trade_date, trade_date)
                    if daily_data is not None and not daily_data.empty:
                        # 添加股票名称
                        daily_data['name'] = stock['name']
                        all_data.append(daily_data)
                except Exception as e:
                    logger.warning(f"获取股票 {stock_code} 在 {trade_date} 的数据失败: {str(e)}")
                    continue
                    
            # 合并所有数据
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                logger.debug(f"成功获取 {trade_date} 的 {len(combined_data)} 只股票行情数据")
                return combined_data
                
            logger.warning(f"在 {trade_date} 没有获取到任何股票行情数据")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取 {trade_date} 的所有股票行情数据失败: {str(e)}")
            return None
            
    @with_cache(method_name="get_market_overview")
    @with_failover(method_name="get_market_overview")
    def get_market_overview(self, trade_date: str = None) -> Dict[str, Any]:
        """
        获取市场概览数据
        
        Args:
            trade_date: 交易日期，默认为最新交易日
            
        Returns:
            Dict: 市场概览数据，包括涨跌家数、成交量、成交额等
        """
        try:
            # 获取所有股票当日行情
            all_data = self.get_all_stock_data_on_date(trade_date)
            if all_data is None or all_data.empty:
                logger.error(f"获取 {trade_date} 的市场概览失败：无法获取行情数据")
                return {}
                
            # 计算涨跌家数
            up_count = len(all_data[all_data['close'] > all_data['open']])
            down_count = len(all_data[all_data['close'] < all_data['open']])
            flat_count = len(all_data) - up_count - down_count
            
            # 计算总成交量和成交额
            total_volume = all_data['volume'].sum() if 'volume' in all_data.columns else 0
            total_amount = all_data['amount'].sum() if 'amount' in all_data.columns else 0
            
            # 计算平均涨跌幅
            all_data['change_pct'] = (all_data['close'] - all_data['open']) / all_data['open'] * 100
            avg_change_pct = all_data['change_pct'].mean()
            
            # 获取涨停和跌停股票
            limit_up_stocks = all_data[all_data['change_pct'] > 9.5][['code', 'name']].to_dict('records')
            limit_down_stocks = all_data[all_data['change_pct'] < -9.5][['code', 'name']].to_dict('records')
            
            # 组装结果
            result = {
                'date': trade_date,
                'up_count': up_count,
                'down_count': down_count,
                'flat_count': flat_count,
                'total_count': len(all_data),
                'total_volume': total_volume,
                'total_amount': total_amount,
                'avg_change_pct': avg_change_pct,
                'limit_up_count': len(limit_up_stocks),
                'limit_down_count': len(limit_down_stocks),
                'limit_up_stocks': limit_up_stocks[:10],  # 只返回前10只
                'limit_down_stocks': limit_down_stocks[:10],  # 只返回前10只
                'turnover_rate': total_volume / total_amount * 100 if total_amount > 0 else 0,
            }
            
            logger.debug(f"成功获取 {trade_date} 的市场概览数据")
            return result
            
        except Exception as e:
            logger.error(f"获取市场概览数据失败: {str(e)}")
            return {}
        
    # 缓存相关方法
    
    def _get_cache_key(self, method_name: str, *args, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            method_name: 方法名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            str: 缓存键
        """
        # 将所有参数序列化为字符串
        args_str = "_".join([str(arg) for arg in args if arg is not None])
        kwargs_str = "_".join([f"{k}_{v}" for k, v in sorted(kwargs.items()) if v is not None])
        
        # 组合成缓存键
        cache_key = f"{method_name}"
        if args_str:
            cache_key += f"_{args_str}"
        if kwargs_str:
            cache_key += f"_{kwargs_str}"
            
        # 替换不适合作为文件名的字符
        cache_key = cache_key.replace("/", "_").replace("\\", "_").replace(":", "_")
        
        return cache_key
        
    def _get_cache_file_path(self, cache_key: str) -> str:
        """
        根据缓存键获取缓存文件路径
        
        Args:
            cache_key: 缓存键
            
        Returns:
            str: 缓存文件路径
        """
        return os.path.join(self.cache_dir, f"{cache_key}.parquet")
        
    def _save_to_cache(self, cache_key: str, data: Union[pd.DataFrame, Dict]) -> bool:
        """
        保存数据到缓存
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
            
        Returns:
            bool: 是否成功保存
        """
        if not self.cache_enabled:
            return False
            
        try:
            # 保存到内存缓存
            if isinstance(data, pd.DataFrame):
                # 计算DataFrame大小(以字节为单位)
                data_size = data.memory_usage(deep=True).sum()
                
                # 如果数据太大，不保存到内存缓存
                if data_size > self.memory_cache_size * 0.5:  # 单个缓存项不超过总大小的50%
                    logger.debug(f"数据太大({data_size / 1024 / 1024:.2f}MB)，不保存到内存缓存")
                else:
                    # 如果内存缓存快满了，清理一些旧的缓存
                    if self.current_memory_cache_size + data_size > self.memory_cache_size:
                        self._clean_memory_cache()
                        
                    # 保存到内存缓存
                    self.memory_cache[cache_key] = data.copy()
                    self.memory_cache_expires[cache_key] = time.time() + self.disk_cache_max_age
                    self.current_memory_cache_size += data_size
                    logger.debug(f"数据已保存到内存缓存，键={cache_key}，大小={data_size / 1024 / 1024:.2f}MB")
                
                # 保存到磁盘缓存
                cache_file = self._get_cache_file_path(cache_key)
                data.to_parquet(cache_file, index=False)
                logger.debug(f"数据已保存到磁盘缓存，文件={cache_file}")
                return True
            elif isinstance(data, dict):
                # 将字典转换为DataFrame后缓存
                dict_df = pd.DataFrame([data])
                cache_file = self._get_cache_file_path(cache_key)
                dict_df.to_parquet(cache_file, index=False)
                logger.debug(f"字典数据已保存到磁盘缓存，文件={cache_file}")
                return True
                
            return False
        except Exception as e:
            logger.warning(f"保存缓存失败: {str(e)}")
            return False
            
    def _get_from_cache(self, cache_key: str) -> Optional[Union[pd.DataFrame, Dict]]:
        """
        从缓存获取数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Optional[Union[pd.DataFrame, Dict]]: 缓存的数据，如果没有则返回None
        """
        if not self.cache_enabled:
            return None
            
        try:
            # 先尝试从内存缓存获取
            if cache_key in self.memory_cache:
                if time.time() < self.memory_cache_expires[cache_key]:
                    logger.debug(f"从内存缓存获取数据，键={cache_key}")
                    return self.memory_cache[cache_key]
                else:
                    # 缓存过期，删除
                    logger.debug(f"内存缓存已过期，键={cache_key}")
                    data_size = self.memory_cache[cache_key].memory_usage(deep=True).sum()
                    self.current_memory_cache_size -= data_size
                    del self.memory_cache[cache_key]
                    del self.memory_cache_expires[cache_key]
            
            # 尝试从磁盘缓存获取
            cache_file = self._get_cache_file_path(cache_key)
            if os.path.exists(cache_file):
                # 检查文件修改时间，判断是否过期
                file_mtime = os.path.getmtime(cache_file)
                if time.time() - file_mtime < self.disk_cache_max_age:
                    logger.debug(f"从磁盘缓存获取数据，文件={cache_file}")
                    return pd.read_parquet(cache_file)
                else:
                    # 缓存过期，删除
                    logger.debug(f"磁盘缓存已过期，文件={cache_file}")
                    os.remove(cache_file)
                    
            return None
        except Exception as e:
            logger.warning(f"获取缓存失败: {str(e)}")
            return None
            
    def _clean_memory_cache(self):
        """清理内存缓存，删除最老的缓存项，直到空间足够"""
        if not self.memory_cache:
            return
            
        logger.debug(f"清理内存缓存，当前大小={self.current_memory_cache_size / 1024 / 1024:.2f}MB")
        
        # 按过期时间排序，删除最老的缓存项
        cache_items = sorted(self.memory_cache_expires.items(), key=lambda x: x[1])
        
        # 删除缓存项，直到空间减少到50%
        target_size = self.memory_cache_size * 0.5
        for key, _ in cache_items:
            if self.current_memory_cache_size <= target_size:
                break
                
            if key in self.memory_cache:
                data_size = self.memory_cache[key].memory_usage(deep=True).sum()
                self.current_memory_cache_size -= data_size
                del self.memory_cache[key]
                del self.memory_cache_expires[key]
                logger.debug(f"删除内存缓存项，键={key}，新大小={self.current_memory_cache_size / 1024 / 1024:.2f}MB")
                
    # 工具方法
    
    @with_cache(method_name='get_trading_dates')
    def get_trading_dates(self, start_date: str, end_date: str = None) -> Optional[List[str]]:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[str]: 交易日列表
        """
        try:
            # 获取健康的数据源
            source_name = self.get_healthy_source()
            if not source_name:
                logger.error("没有可用的数据源来获取交易日历")
                return None
                
            # 使用数据源获取交易日历
            data_source = self.data_sources[source_name]
            if hasattr(data_source, 'get_trading_dates'):
                logger.debug(f"使用数据源 {source_name} 获取交易日历")
                return data_source.get_trading_dates(start_date, end_date)
            
            # 如果没有实现此方法，使用替代方法
            logger.debug(f"数据源 {source_name} 没有实现get_trading_dates方法，使用替代方法")
            
            # 尝试获取上证指数数据作为交易日参考
            index_data = self.get_stock_index_data('000001.SH', start_date, end_date)
            if index_data is not None and not index_data.empty:
                trading_dates = index_data['date'].tolist()
                trading_dates.sort()
                return trading_dates
                
            logger.warning("无法获取交易日历")
            return None
            
        except Exception as e:
            logger.error(f"获取交易日历失败: {str(e)}")
            return None
            
    def get_latest_trading_date(self) -> Optional[str]:
        """
        获取最新交易日
        
        Returns:
            str: 最新交易日日期
        """
        # 查询最近10天的交易日历
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        
        trading_dates = self.get_trading_dates(start_date, end_date)
        if trading_dates:
            return trading_dates[-1]
            
        return None
        
    def is_trading_date(self, date_str: str) -> bool:
        """
        判断是否为交易日
        
        Args:
            date_str: 日期字符串
            
        Returns:
            bool: 是否为交易日
        """
        # 将日期转换为标准格式
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # 查询当月的交易日历
        start_date = date_obj.replace(day=1).strftime('%Y-%m-%d')
        end_date = (date_obj.replace(day=1) + timedelta(days=31)).replace(day=1) - timedelta(days=1)
        end_date = end_date.strftime('%Y-%m-%d')
        
        trading_dates = self.get_trading_dates(start_date, end_date)
        if trading_dates:
            return date_str in trading_dates
            
        return False
        
    def get_previous_trading_date(self, date_str: str, n: int = 1) -> Optional[str]:
        """
        获取前N个交易日
        
        Args:
            date_str: 日期字符串
            n: 前N个交易日
            
        Returns:
            str: 前N个交易日日期
        """
        # 将日期转换为标准格式
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # 查询前30天的交易日历
        end_date = date_str
        start_date = (date_obj - timedelta(days=30)).strftime('%Y-%m-%d')
        
        trading_dates = self.get_trading_dates(start_date, end_date)
        if trading_dates:
            trading_dates.sort()
            date_index = trading_dates.index(date_str) if date_str in trading_dates else len(trading_dates)
            
            if date_index >= n:
                return trading_dates[date_index - n]
                
        return None
        
    def get_next_trading_date(self, date_str: str, n: int = 1) -> Optional[str]:
        """
        获取后N个交易日
        
        Args:
            date_str: 日期字符串
            n: 后N个交易日
            
        Returns:
            str: 后N个交易日日期
        """
        # 将日期转换为标准格式
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # 查询后30天的交易日历
        start_date = date_str
        end_date = (date_obj + timedelta(days=30)).strftime('%Y-%m-%d')
        
        trading_dates = self.get_trading_dates(start_date, end_date)
        if trading_dates:
            trading_dates.sort()
            if date_str in trading_dates:
                date_index = trading_dates.index(date_str)
                if date_index + n < len(trading_dates):
                    return trading_dates[date_index + n]
            else:
                # 如果当前日期不是交易日，找到下一个交易日
                next_dates = [d for d in trading_dates if d > date_str]
                if len(next_dates) >= n:
                    return next_dates[n-1]
                    
        return None
        
    def get_data_sources_status(self) -> Dict[str, Dict]:
        """
        获取所有数据源的状态信息
        
        Returns:
            Dict[str, Dict]: 数据源名称到状态的映射
        """
        status = {}
        for source_name in self.data_sources:
            source_status = {
                'healthy': self.health_check(source_name),
                'last_check_time': datetime.fromtimestamp(self.last_check_time.get(source_name, 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'is_primary': source_name == self.primary_source
            }
            status[source_name] = source_status
            
        return status 
