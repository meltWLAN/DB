#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据源管理器
管理多个数据源，自动进行健康检查和故障转移
"""

import time as time_module
import logging
import pandas as pd
import numpy as np
import os
import json
import concurrent.futures
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from functools import wraps, partial

from src.enhanced.config.settings import HEALTH_CHECK_CONFIG, DATA_SOURCE_CONFIG, CACHE_CONFIG, ENHANCED_CACHE_DIR, PROCESSING_CONFIG
from src.enhanced.data.fetchers.tushare_fetcher import EnhancedTushareFetcher
from src.enhanced.data.fetchers.akshare_fetcher import EnhancedAKShareFetcher
from src.enhanced.data.fetchers.joinquant_fetcher import EnhancedJoinQuantFetcher

# 配置日志
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
    
    def __init__(self, config_file=None):
        """
        初始化数据源管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.data_sources = {}
        self.data_source_config = {}
        self._primary_source = None
        self._secondary_source = None
        
        # 添加健康状态跟踪
        self.health_status = {}
        self.last_check_time = {}
        
        # 添加健康检查配置
        self.health_check_enabled = True
        self.check_interval = 300  # 默认5分钟
        self.retry_interval = 1800  # 默认30分钟
        
        # 添加缓存相关配置
        self.cache_enabled = True
        self.cache_dir = "./cache"
        self.disk_cache_max_age = 24 * 3600  # 默认24小时，转换为秒
        self.memory_cache = {}  # 内存缓存
        self.memory_cache_expires = {}  # 内存缓存过期时间
        self.memory_cache_size = 128 * 1024 * 1024  # 默认128MB，转换为字节
        self.current_memory_cache_size = 0  # 当前内存缓存大小(字节)
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 加载配置
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    self.data_source_config = json.load(f)
            except Exception as e:
                logger.error(f"加载配置文件 {config_file} 失败: {e}")
                # 使用默认配置
                self.data_source_config = {}
        
        # 如果配置文件加载失败或未提供，使用默认配置
        if not self.data_source_config:
            try:
                self.data_source_config = DATA_SOURCE_CONFIG
            except Exception as e:
                logger.error(f"使用默认配置失败: {e}")
                self.data_source_config = {}
                
        # 初始化数据源
        self._init_data_sources()
        
        # 设置主数据源和备用数据源
        self._setup_primary_sources()
        
        logger.info(f"数据源管理器初始化完成，已注册 {len(self.data_sources)} 个数据源，主数据源: {self.primary_source}")
                
    def init_data_sources(self):
        """
        初始化数据源，公共方法，调用_init_data_sources
        """
        return self._init_data_sources()
    
    def _init_data_sources(self):
        """初始化所有启用的数据源"""
        success_count = 0
        
        # 遍历所有配置的数据源
        for source_name, source_config in self.data_source_config.items():
            if not source_config.get('enabled', False):
                logger.info(f"数据源 {source_name} 未启用，跳过初始化")
                continue
                
            try:
                # 初始化各类数据源
                if source_name == 'tushare':
                    # 初始化 TuShare 数据源
                    try:
                        # 从settings.py获取有效token
                        from src.enhanced.config.settings import DATA_SOURCE_CONFIG
                        token = DATA_SOURCE_CONFIG.get('tushare', {}).get('token', '')
                        if token:
                            source_config['token'] = token
                            
                        self.data_sources[source_name] = EnhancedTushareFetcher(source_config)
                        self.health_status[source_name] = True
                        self.last_check_time[source_name] = time_module.time()
                        success_count += 1
                    except Exception as e:
                        logger.error(f"初始化数据源 {source_name} 失败: {str(e)}")
                    
                elif source_name == 'akshare':
                    # 初始化 AKShare 数据源
                    try:
                        self.data_sources[source_name] = EnhancedAKShareFetcher(source_config)
                        self.health_status[source_name] = True
                        self.last_check_time[source_name] = time_module.time()
                        success_count += 1
                    except Exception as e:
                        logger.error(f"初始化数据源 {source_name} 失败: {str(e)}")
                    
                elif source_name == 'joinquant':
                    # 初始化 JoinQuant 数据源
                    try:
                        self.data_sources[source_name] = EnhancedJoinQuantFetcher(source_config)
                        self.health_status[source_name] = True
                        self.last_check_time[source_name] = time_module.time()
                        success_count += 1
                    except Exception as e:
                        logger.error(f"初始化数据源 {source_name} 失败: {str(e)}")
                    
                else:
                    logger.warning(f"未知数据源类型 {source_name}，跳过初始化")
                    
            except Exception as e:
                logger.error(f"初始化数据源 {source_name} 时发生错误: {str(e)}")
                
        logger.info(f"成功初始化 {success_count} 个数据源")
    
    def _setup_primary_sources(self):
        """设置主数据源和备用数据源"""
        # 获取主数据源名称
        primary_source_name = self._get_primary_source_name()
        self.primary_source = primary_source_name
        
        # 设置主数据源
        if primary_source_name in self.data_sources:
            self._primary_source = self.data_sources[primary_source_name]
            logger.info(f"设置 {primary_source_name} 为主数据源")
        else:
            # 如果未找到主数据源，使用第一个可用的
            if self.data_sources:
                first_source_name = next(iter(self.data_sources))
                self._primary_source = self.data_sources[first_source_name]
                self.primary_source = first_source_name
                logger.warning(f"未找到主数据源，使用 {first_source_name} 作为主数据源")
            else:
                self._primary_source = None
                self.primary_source = "none"
                logger.error("未找到可用的数据源")
                
        # 设置备用数据源
        if len(self.data_sources) > 1:
            # 使用非主数据源作为备用数据源
            for name, source in self.data_sources.items():
                if name != primary_source_name:
                    self._secondary_source = source
                    logger.info(f"设置 {name} 为备用数据源")
                    break
        else:
            # 如果只有一个数据源，备用数据源与主数据源相同
            self._secondary_source = self._primary_source
    
    def _get_primary_source_name(self) -> str:
        """获取主数据源名称"""
        # 查找配置为主数据源的
        for name, config in self.data_source_config.items():
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
        
        now = time_module.time()
        
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
            if not self.health_status.get(source_name, True) and time_module.time() - self.last_check_time.get(source_name, 0) > self.retry_interval:
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
    def get_stock_list(self, industry=None) -> Optional[pd.DataFrame]:
        """
        获取股票列表
        
        Args:
            industry: 行业名称或代码，如果为None则获取所有股票
            
        Returns:
            pandas.DataFrame: 股票列表，包含代码、名称等信息
        """
        try:
            # 尝试从主数据源获取
            if self._primary_source:
                if hasattr(self._primary_source, 'get_stock_list'):
                    data = self._primary_source.get_stock_list(industry)
                    if data is not None and len(data) > 0:
                        return data
            
            # 如果主数据源获取失败，尝试备用数据源
            if self._secondary_source and self._secondary_source != self._primary_source:
                if hasattr(self._secondary_source, 'get_stock_list'):
                    logger.warning("主数据源获取股票列表失败，尝试备用数据源")
                    data = self._secondary_source.get_stock_list(industry)
                    if data is not None and len(data) > 0:
                        return data
            
            # 如果都失败，返回空DataFrame
            logger.error("无法获取股票列表，返回空DataFrame")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            # 返回一个空的DataFrame
            return pd.DataFrame()
    
    @with_cache(method_name="get_daily_data")
    @with_failover(method_name="get_daily_data")
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str = None) -> Optional[pd.DataFrame]:
        """获取日线数据，自动故障转移"""
        pass
    
    @with_cache(method_name="get_industry_list")
    @with_failover(method_name="get_industry_list")
    def get_industry_list(self) -> Optional[pd.DataFrame]:
        """
        获取行业列表（适配 gui_controller 接口）
        
        Returns:
            pandas.DataFrame: 行业列表，如果获取失败则返回 None
        """
        try:
            # 尝试使用主数据源获取行业列表
            if self._primary_source and hasattr(self._primary_source, 'get_industry_list'):
                data = self._primary_source.get_industry_list()
                if data is not None and len(data) > 0:
                    return data
            
            # 如果主数据源获取失败，尝试备用数据源
            if self._secondary_source and self._secondary_source != self._primary_source and hasattr(self._secondary_source, 'get_industry_list'):
                logger.warning("主数据源获取行业列表失败，尝试备用数据源")
                data = self._secondary_source.get_industry_list()
                return data
                
            # 如果都失败，返回空的DataFrame
            logger.error("无法获取行业列表，返回空的DataFrame")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取行业列表失败: {e}")
            # 返回空的DataFrame
            return pd.DataFrame()
    
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
        try:
            # 尝试从主数据源获取
            if self._primary_source:
                try:
                    data = self._primary_source.get_stock_index_data(index_code, start_date, end_date)
                    if data is not None and not data.empty:
                        logger.info(f"成功从主数据源获取 {index_code} 的指数数据，包含 {len(data)} 条记录")
                        return data
                except Exception as e:
                    logger.warning(f"主数据源获取指数 {index_code} 数据失败: {str(e)}")
            
            # 尝试从备用数据源获取
            if self._secondary_source and self._secondary_source != self._primary_source:
                try:
                    data = self._secondary_source.get_stock_index_data(index_code, start_date, end_date)
                    if data is not None and not data.empty:
                        logger.info(f"成功从备用数据源获取 {index_code} 的指数数据，包含 {len(data)} 条记录")
                        return data
                except Exception as e:
                    logger.warning(f"备用数据源获取指数 {index_code} 数据失败: {str(e)}")
            
            # 如果所有源都失败，返回None
            logger.warning(f"所有数据源获取指数 {index_code} 数据失败")
            return None
        
        except Exception as e:
            logger.error(f"获取指数 {index_code} 数据出错: {str(e)}")
            return None

    def _generate_mock_index_data(self, index_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        生成模拟的指数数据
        
        Args:
            index_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 模拟的指数数据
        """
        try:
            # 解析日期
            from datetime import datetime, timedelta
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                # 默认生成30天的数据
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # 创建日期范围
            date_range = []
            curr_dt = start_dt
            while curr_dt <= end_dt:
                # 跳过周末
                if curr_dt.weekday() < 5:  # 0-4是周一到周五
                    date_range.append(curr_dt.strftime('%Y-%m-%d'))
                curr_dt += timedelta(days=1)
            
            # 使用稳定的种子确保同一指数每次生成的模拟数据一致
            import numpy as np
            
            # 修复: 确保种子在有效范围内 (0到2^32-1)
            # 使用指数代码的哈希值的绝对值，然后取模以确保在范围内
            seed_hash = abs(hash(index_code))
            seed = seed_hash % (2**32 - 1)  
            np.random.seed(seed)
            
            # 基础价格根据指数类型设置
            if '000001.SH' in index_code:  # 上证指数
                base_price = 3000 + np.random.uniform(-300, 300)
            elif '399001' in index_code:  # 深证成指
                base_price = 10000 + np.random.uniform(-1000, 1000)
            elif '399006' in index_code:  # 创业板指
                base_price = 2000 + np.random.uniform(-200, 200)
            elif '000300' in index_code:  # 沪深300
                base_price = 3500 + np.random.uniform(-350, 350)
            elif '000016' in index_code:  # 上证50
                base_price = 2500 + np.random.uniform(-250, 250)
            elif '000905' in index_code:  # 中证500
                base_price = 6000 + np.random.uniform(-600, 600)
            else:
                base_price = 1000 + np.random.uniform(-100, 100)
            
            # 生成价格和成交量
            prices = []
            volumes = []
            for i in range(len(date_range)):
                if i == 0:
                    prices.append(base_price)
                else:
                    change_pct = np.random.normal(0, 0.01)  # 每日涨跌幅，均值为0，标准差为1%
                    prices.append(prices[i-1] * (1 + change_pct))
                volumes.append(np.random.randint(100000, 10000000))  # 随机成交量
            
            # 创建DataFrame
            data = {
                'ts_code': index_code,
                'trade_date': date_range,
                'open': [price * (1 + np.random.normal(0, 0.003)) for price in prices],
                'high': [price * (1 + abs(np.random.normal(0, 0.005))) for price in prices],
                'low': [price * (1 - abs(np.random.normal(0, 0.005))) for price in prices],
                'close': prices,
                'vol': volumes,
                'amount': [vol * price / 1000 * np.random.uniform(0.95, 1.05) for vol, price in zip(volumes, prices)]
            }
            
            # 计算涨跌幅
            data['change'] = [0] + [prices[i] - prices[i-1] for i in range(1, len(prices))]
            data['pct_chg'] = [0] + [(prices[i] / prices[i-1] - 1) * 100 for i in range(1, len(prices))]
            
            mock_df = pd.DataFrame(data)
            
            # 重置随机数种子
            np.random.seed(None)
            
            logger.warning(f"生成 {index_code} 的模拟指数数据，包含 {len(mock_df)} 条记录")
            return mock_df
            
        except Exception as e:
            logger.error(f"生成模拟指数数据出错: {str(e)}")
            # 出错时返回至少包含基本结构的空DataFrame
            import pandas as pd
            empty_df = pd.DataFrame({
                'ts_code': [index_code],
                'trade_date': [datetime.now().strftime('%Y-%m-%d')],
                'open': [0], 'high': [0], 'low': [0], 'close': [0],
                'vol': [0], 'amount': [0], 'pct_chg': [0]
            })
            return empty_df
        
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
                    self.memory_cache_expires[cache_key] = time_module.time() + self.disk_cache_max_age
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
                if time_module.time() < self.memory_cache_expires[cache_key]:
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
                if time_module.time() - file_mtime < self.disk_cache_max_age:
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
        获取交易日历，按照中国股市交易规则
        
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
            # 中国股市交易日特征: 工作日(周一至周五)，排除法定节假日
            # 法定节假日包括: 元旦、春节、清明节、劳动节、端午节、中秋节、国庆节等
            index_data = self.get_stock_index_data('000001.SH', start_date, end_date)
            if index_data is not None and not index_data.empty and 'date' in index_data.columns:
                trading_dates = index_data['date'].tolist()
                trading_dates.sort()
                return trading_dates
            
            # 如果无法通过指数数据获取，尝试其他方法
            logger.warning("无法通过指数数据获取交易日历，尝试使用交易日API")
            
            # 尝试使用特定的交易日历API
            if self._primary_source and hasattr(self._primary_source, '_execute_api_call'):
                try:
                    # 转换日期格式
                    start_date_fmt = start_date.replace('-', '')
                    end_date_fmt = end_date.replace('-', '') if end_date else datetime.now().strftime('%Y%m%d')
                    
                    # 使用TuShare的trade_cal接口
                    trade_cal = self._primary_source._execute_api_call('trade_cal', 
                                                                       start_date=start_date_fmt, 
                                                                       end_date=end_date_fmt,
                                                                       is_open='1')  # is_open=1表示交易日
                    
                    if trade_cal is not None and not trade_cal.empty and 'cal_date' in trade_cal.columns:
                        # 转换日期格式
                        trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date']).dt.strftime('%Y-%m-%d')
                        trading_dates = trade_cal['cal_date'].tolist()
                        trading_dates.sort()
                        return trading_dates
                except Exception as e:
                    logger.error(f"获取交易日历API失败: {str(e)}")
            
            logger.warning("无法获取中国股市交易日历，返回None")
            return None
            
        except Exception as e:
            logger.error(f"获取交易日历失败: {str(e)}")
            return None
            
    def is_trading_day(self, date_str: str) -> bool:
        """
        检查给定日期是否为交易日，根据中国股市交易规则
        
        Args:
            date_str: 日期字符串，格式 YYYY-MM-DD
            
        Returns:
            bool: 是否为交易日
        """
        try:
            # 格式化日期
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # 快速检查：周末不是交易日
            if date_obj.weekday() >= 5:  # 5是周六，6是周日
                return False
                
            # 检查是否在交易日历中
            source_name = self.get_healthy_source()
            if not source_name:
                logger.warning("没有健康的数据源来检查交易日")
                # 使用备选方案: 直接检查工作日
                return date_obj.weekday() < 5  # 周一至周五
                
            # 使用数据源API检查
            data_source = self.data_sources[source_name]
            if hasattr(data_source, '_execute_api_call'):
                try:
                    # 转换日期格式
                    date_fmt = date_str.replace('-', '')
                    
                    # 查询该日期是否为交易日
                    is_trade_day = data_source._execute_api_call('trade_cal', 
                                                               exchange='SSE', 
                                                               start_date=date_fmt, 
                                                               end_date=date_fmt,
                                                               is_open='1')  # is_open=1表示交易日
                                                           
                    return is_trade_day is not None and not is_trade_day.empty
                except Exception as e:
                    logger.error(f"查询交易日失败: {str(e)}")
                    
            # 如果上述方法失败，尝试获取一个月的交易日历
            start_date = (date_obj - timedelta(days=15)).strftime('%Y-%m-%d')
            end_date = (date_obj + timedelta(days=15)).strftime('%Y-%m-%d')
            
            trading_dates = self.get_trading_dates(start_date, end_date)
            if trading_dates and date_str in trading_dates:
                return True
                
            return False
        except Exception as e:
            logger.error(f"检查交易日失败: {str(e)}")
            # 默认当作不是交易日处理
            return False

    def get_latest_trading_date(self) -> Optional[str]:
        """
        获取最近的交易日期（不含当日，如当日为交易日则返回上一交易日）
        按照中国股市交易规则
        
        Returns:
            str: 最近的交易日期，如果获取失败则返回None
        """
        try:
            # 获取当前日期
            today = datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.now().time()
            
            # 如果当天是交易日且为开盘时间(9:30-15:00)，则返回当天
            if self.is_trading_day(today) and \
               ((current_time >= datetime.strptime('09:30', '%H:%M').time() and current_time <= datetime.strptime('11:30', '%H:%M').time()) or \
                (current_time >= datetime.strptime('13:00', '%H:%M').time() and current_time <= datetime.strptime('15:00', '%H:%M').time())):
                return today
                
            # 否则返回最近一个交易日
            return self.get_previous_trading_date(today)
        except Exception as e:
            logger.error(f"获取最近交易日期失败: {str(e)}")
            return None
        
    def get_previous_trading_date(self, date_str: str, n: int = 1) -> Optional[str]:
        """
        获取前N个交易日，按照中国股市规则
        
        Args:
            date_str: 日期字符串，格式 YYYY-MM-DD
            n: 前N个交易日
            
        Returns:
            str: 前N个交易日日期，如果获取失败则返回None
        """
        # 将日期转换为标准格式
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"日期格式错误: {date_str}，应为YYYY-MM-DD格式: {str(e)}")
            return None
        
        # 查询前40天的交易日历，确保能覆盖n个交易日
        # 中国股市春节假期可能长达10天左右，保险起见取40天
        end_date = date_str
        start_date = (date_obj - timedelta(days=40)).strftime('%Y-%m-%d')
        
        trading_dates = self.get_trading_dates(start_date, end_date)
        if not trading_dates:
            logger.warning(f"获取从 {start_date} 到 {end_date} 的交易日历失败")
            
            # 如果获取交易日历失败，尝试基于工作日的简单推算（不含节假日考虑）
            fallback_date = date_obj
            business_days_count = 0
            max_iterations = 100  # 防止无限循环
            iterations = 0
            
            while business_days_count < n and iterations < max_iterations:
                fallback_date = fallback_date - timedelta(days=1)
                iterations += 1
                # 仅考虑周一至周五
                if fallback_date.weekday() < 5:  # 0-4是周一至周五
                    business_days_count += 1
            
            logger.warning(f"使用工作日简单推算，前{n}个工作日是 {fallback_date.strftime('%Y-%m-%d')}，未考虑法定节假日")
            return fallback_date.strftime('%Y-%m-%d')
        
        # 将日期列表排序
        trading_dates.sort()
        
        # 查找当前日期的位置
        if date_str in trading_dates:
            date_index = trading_dates.index(date_str)
            
            # 如果索引位置足够大，返回前n个交易日
            if date_index >= n:
                return trading_dates[date_index - n]
            else:
                logger.warning(f"交易日历中的日期不足以获取前{n}个交易日")
                return trading_dates[0] if trading_dates else None
        else:
            # 如果当前日期不是交易日，找到小于当前日期的最近交易日
            prev_dates = [d for d in trading_dates if d < date_str]
            if len(prev_dates) >= n:
                return prev_dates[-n]
            else:
                logger.warning(f"交易日历中的日期不足以获取前{n}个交易日")
                return prev_dates[0] if prev_dates else None
            
    def get_next_trading_date(self, date_str: str, n: int = 1) -> Optional[str]:
        """
        获取后N个交易日，按照中国股市规则
        
        Args:
            date_str: 日期字符串，格式 YYYY-MM-DD
            n: 后N个交易日
            
        Returns:
            str: 后N个交易日日期，如果获取失败则返回None
        """
        # 将日期转换为标准格式
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"日期格式错误: {date_str}，应为YYYY-MM-DD格式: {str(e)}")
            return None
        
        # 查询后40天的交易日历
        # 中国股市春节假期可能长达10天左右，保险起见取40天
        start_date = date_str
        end_date = (date_obj + timedelta(days=40)).strftime('%Y-%m-%d')
        
        trading_dates = self.get_trading_dates(start_date, end_date)
        if not trading_dates:
            logger.warning(f"获取从 {start_date} 到 {end_date} 的交易日历失败")
            
            # 如果获取交易日历失败，尝试基于工作日的简单推算（不含节假日考虑）
            fallback_date = date_obj
            business_days_count = 0
            max_iterations = 100  # 防止无限循环
            iterations = 0
            
            while business_days_count < n and iterations < max_iterations:
                fallback_date = fallback_date + timedelta(days=1)
                iterations += 1
                # 仅考虑周一至周五
                if fallback_date.weekday() < 5:  # 0-4是周一至周五
                    business_days_count += 1
            
            logger.warning(f"使用工作日简单推算，后{n}个工作日是 {fallback_date.strftime('%Y-%m-%d')}，未考虑法定节假日")
            return fallback_date.strftime('%Y-%m-%d')
        
        # 将日期列表排序
        trading_dates.sort()
        
        # 查找当前日期的位置
        if date_str in trading_dates:
            date_index = trading_dates.index(date_str)
            
            # 如果有足够的后续交易日
            if date_index + n < len(trading_dates):
                return trading_dates[date_index + n]
            else:
                logger.warning(f"交易日历中的日期不足以获取后{n}个交易日")
                return trading_dates[-1] if trading_dates else None
        else:
            # 如果当前日期不是交易日，找到大于当前日期的最近交易日
            next_dates = [d for d in trading_dates if d > date_str]
            
            # 如果有足够的后续交易日
            if len(next_dates) >= n:
                return next_dates[n-1]
            else:
                logger.warning(f"交易日历中的日期不足以获取后{n}个交易日")
                return next_dates[-1] if next_dates else None

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

    def get_index_daily(self, index_code, start_date=None, end_date=None, limit=None):
        """
        获取指数日线数据（适配 gui_controller 接口）
        
        Args:
            index_code: 指数代码，如 '000001.SH'
            start_date: 起始日期，格式为 YYYY-MM-DD，默认为 None
            end_date: 结束日期，格式为 YYYY-MM-DD，默认为 None
            limit: 获取条数限制，默认为 None
            
        Returns:
            pandas.DataFrame: 指数日线数据，如果获取失败则返回 None
        """
        try:
            # 如果只提供了 limit 但没有提供日期范围，设置合理的默认日期范围
            if start_date is None and limit is not None:
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=limit * 2)).strftime('%Y-%m-%d')
            
            # 调用底层接口
            data = self.get_stock_index_data(index_code, start_date, end_date)
            
            # 如果底层接口返回为空，尝试其他数据源
            if data is None or len(data) == 0:
                if self._secondary_source and self._secondary_source != self._primary_source:
                    logger.warning(f"主数据源获取指数 {index_code} 数据失败，尝试备用数据源")
                    data = self._secondary_source.get_stock_index_data(index_code, start_date, end_date)
            
            # 如果指定了limit且数据不为空，返回最近的limit条记录
            if data is not None and limit is not None and len(data) > limit:
                return data.head(limit)
            
            return data
        except Exception as e:
            logger.error(f"获取指数 {index_code} 日线数据失败: {e}")
            return None
            
    @with_cache(method_name="get_stock_data")
    @with_failover(method_name="get_daily_data")
    def get_stock_data(self, stock_code: str, start_date: str = None, end_date: str = None, limit: int = None) -> Optional[pd.DataFrame]:
        """
        获取股票日线数据（适配 gui_controller 接口）
        
        Args:
            stock_code: 股票代码，如 '000001.SZ'
            start_date: 起始日期，格式为 YYYY-MM-DD，默认为 None
            end_date: 结束日期，格式为 YYYY-MM-DD，默认为 None
            limit: 获取条数限制，默认为 None
            
        Returns:
            pandas.DataFrame: 股票日线数据，如果获取失败则返回 None
        """
        try:
            # 如果未提供起始日期但提供了limit，设置默认起始日期
            if start_date is None and limit is not None:
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime('%Y-%m-%d') if end_date is None else end_date
                start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=limit * 2)).strftime('%Y-%m-%d')
                
            # 调用 get_daily_data 方法
            data = self.get_daily_data(stock_code, start_date, end_date)
            
            # 如果获取成功，进行处理
            if data is not None and not data.empty:
                # 如果提供了limit，截取最近的数据
                if limit is not None and len(data) > limit:
                    data = data.sort_values('date', ascending=False).head(limit).sort_values('date')
                    
                return data
            else:
                # 如果没有获取到数据，返回None而不是生成模拟数据
                logger.warning(f"无法获取股票 {stock_code} 数据，未使用模拟数据替代")
                return None
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 日线数据失败: {e}")
            return None
            
    def _generate_mock_stock_data(self, stock_code: str, start_date: str = None, end_date: str = None, limit: int = None) -> pd.DataFrame:
        """
        生成模拟的股票日线数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            limit: 获取条数限制
            
        Returns:
            pandas.DataFrame: 模拟的股票日线数据
        """
        try:
            # 解析日期
            from datetime import datetime, timedelta
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                # 默认生成30天的数据
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # 创建日期范围
            date_range = []
            curr_dt = start_dt
            while curr_dt <= end_dt:
                # 跳过周末
                if curr_dt.weekday() < 5:  # 0-4是周一到周五
                    date_range.append(curr_dt.strftime('%Y%m%d'))
                curr_dt += timedelta(days=1)
                
            # 如果有限制条数，截取最近的数据
            if limit is not None and len(date_range) > limit:
                date_range = date_range[-limit:]
                
            # 确保种子在有效范围内
            import numpy as np
            seed = int(hash(stock_code) % (2**32 - 1))  # 确保种子在合法范围内
            np.random.seed(seed)
            
            # 生成随机价格和成交量
            base_price = np.random.uniform(10, 100)  # 基础价格
            prices = []
            volumes = []
            for i in range(len(date_range)):
                if i == 0:
                    prices.append(base_price)
                else:
                    change_pct = np.random.normal(0, 0.02)  # 每日涨跌幅，均值为0，标准差为2%
                    prices.append(prices[i-1] * (1 + change_pct))
                volumes.append(np.random.randint(10000, 1000000))  # 随机成交量
                
            # 创建DataFrame
            data = {
                'ts_code': stock_code,
                'trade_date': date_range,
                'open': [price * (1 + np.random.normal(0, 0.005)) for price in prices],
                'high': [price * (1 + abs(np.random.normal(0, 0.01))) for price in prices],
                'low': [price * (1 - abs(np.random.normal(0, 0.01))) for price in prices],
                'close': prices,
                'vol': volumes,
                'amount': [vol * price * np.random.uniform(0.9, 1.1) for vol, price in zip(volumes, prices)],
                'change': [0] + [prices[i] - prices[i-1] for i in range(1, len(prices))],
                'pct_chg': [0] + [(prices[i] / prices[i-1] - 1) * 100 for i in range(1, len(prices))]
            }
            
            mock_df = pd.DataFrame(data)
            
            # 重置随机数种子
            np.random.seed(None)
            
            logger.warning(f"返回 {stock_code} 的模拟股票数据")
            return mock_df
            
        except Exception as e:
            logger.error(f"生成模拟股票数据失败: {e}")
            # 出错时返回空DataFrame
            return pd.DataFrame()

    def get_stocks_in_industry(self, industry_code):
        """
        获取行业成分股（适配 gui_controller 接口）
        
        Args:
            industry_code: 行业代码
            
        Returns:
            pandas.DataFrame: 行业成分股，如果获取失败则返回 None
        """
        try:
            if self._primary_source and hasattr(self._primary_source, 'get_stocks_in_industry'):
                data = self._primary_source.get_stocks_in_industry(industry_code)
                if data is not None and len(data) > 0:
                    return data
            
            # 如果主数据源获取失败，尝试备用数据源
            if self._secondary_source and self._secondary_source != self._primary_source and hasattr(self._secondary_source, 'get_stocks_in_industry'):
                logger.warning(f"主数据源获取行业 {industry_code} 成分股失败，尝试备用数据源")
                data = self._secondary_source.get_stocks_in_industry(industry_code)
                return data
                
            # 如果都失败，返回空DataFrame
            logger.error(f"无法获取行业 {industry_code} 成分股，返回空DataFrame")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取行业 {industry_code} 成分股失败: {e}")
            return pd.DataFrame()
    
    def get_daily_market_data(self):
        """
        获取市场日线数据（适配 gui_controller 接口）
        
        Returns:
            pandas.DataFrame: 市场日线数据，如果获取失败则返回 None
        """
        try:
            # 获取股票列表
            stock_list = self.get_stock_list()
            if stock_list is None or len(stock_list) == 0:
                logger.error("获取股票列表失败")
                return pd.DataFrame()
            
            # 获取当前日期
            from datetime import datetime
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 随机抽样最多100支股票获取行情
            import pandas as pd
            
            if len(stock_list) > 100:
                sample_stocks = stock_list.sample(100)
            else:
                sample_stocks = stock_list
            
            # 存储所有股票当日行情
            all_data = []
            
            for _, stock in sample_stocks.iterrows():
                ts_code = stock['ts_code']
                try:
                    # 获取最近的一条数据
                    data = self.get_stock_data(ts_code, start_date=current_date, end_date=current_date)
                    
                    if data is not None and len(data) > 0:
                        all_data.append(data.iloc[0])
                except Exception:
                    continue
            
            if len(all_data) > 0:
                result_df = pd.DataFrame(all_data)
                return result_df
            
            # 如果无法获取任何数据，返回空DataFrame
            logger.warning("无法获取市场日线数据，返回空DataFrame")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取市场日线数据失败: {e}")
            return pd.DataFrame() 
