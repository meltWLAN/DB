#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据获取模块
支持多种数据源，优先使用Tushare
"""

import os
import logging
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures
from src.data.processor import StockDataProcessor

# 根据配置选择性导入相应的数据源
try:
    from .tushare_fetcher import TushareFetcher
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

from ..config import DATA_SOURCE_CONFIG

# 设置日志
logger = logging.getLogger(__name__)

# 尝试从配置中导入 CACHE_DIR，如果失败则使用默认值
try:
    from ..config import CACHE_DIR
except ImportError:
    # 使用当前目录下的 cache 目录作为缓存目录
    CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

class StockDataFetcher:
    """股票数据获取类，支持多种数据源"""
    
    def __init__(self):
        """初始化数据获取器，支持多数据源"""
        # 读取配置
        from src.config import DATA_SOURCE_CONFIG
        self.config = DATA_SOURCE_CONFIG
        
        # 初始化数据源
        self.data_sources = {}
        self.primary_source = None
        
        # 尝试初始化Tushare
        if 'tushare' in self.config and self.config['tushare'].get('enable', False):
            try:
                from src.data.tushare_fetcher import TushareFetcher
                self.data_sources['tushare'] = TushareFetcher(self.config['tushare'])
                if self.config['tushare'].get('is_primary', False):
                    self.primary_source = 'tushare'
                logger.info("Tushare数据源初始化成功")
            except Exception as e:
                logger.error(f"Tushare数据源初始化失败: {str(e)}")
        
        # 尝试初始化AKShare
        if 'akshare' in self.config and self.config['akshare'].get('enable', False):
            try:
                from src.data.akshare_fetcher import AKShareFetcher
                self.data_sources['akshare'] = AKShareFetcher(self.config['akshare'])
                if self.primary_source is None or self.config['akshare'].get('is_primary', False):
                    self.primary_source = 'akshare'
                logger.info("AKShare数据源初始化成功")
            except Exception as e:
                logger.error(f"AKShare数据源初始化失败: {str(e)}")
        
        # 尝试初始化BaoStock
        if 'baostock' in self.config and self.config['baostock'].get('enable', False):
            try:
                from src.data.baostock_fetcher import BaoStockFetcher
                self.data_sources['baostock'] = BaoStockFetcher(self.config['baostock'])
                if self.primary_source is None or self.config['baostock'].get('is_primary', False):
                    self.primary_source = 'baostock'
                logger.info("BaoStock数据源初始化成功")
            except Exception as e:
                logger.error(f"BaoStock数据源初始化失败: {str(e)}")
        
        # 确保至少有一个数据源可用
        if not self.data_sources:
            logger.warning("没有可用的数据源，请检查配置")
        else:
            logger.info(f"初始化了 {len(self.data_sources)} 个数据源，主数据源是 {self.primary_source}")
        
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        # 优先使用主要数据源
        if self.primary_source == "tushare":
            result = self.data_sources["tushare"].get_stock_list()
            if result is not None:
                return result
        
        # 如果主要数据源失败，尝试其他数据源
        if "akshare" in self.data_sources:
            try:
                df = ak.stock_info_a_code_name()
                logger.info(f"从AKShare获取到 {len(df)} 只股票")
                return df
            except Exception as e:
                logger.error(f"从AKShare获取股票列表失败: {e}")
        
        return None
    
    def get_daily_data(self, stock_code, start_date, end_date=None):
        """获取股票日线数据
        
        Args:
            stock_code: 股票代码
            start_date: 起始日期
            end_date: 结束日期，默认为今天
            
        Returns:
            DataFrame: 日线数据
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"获取股票 {stock_code} 的日线数据 ({start_date} 至 {end_date})")
        
        # 缓存文件名
        cache_file = os.path.join(CACHE_DIR, f"{stock_code}_{start_date}_{end_date}.csv")
        
        # 如果缓存存在且新鲜，直接使用缓存
        if os.path.exists(cache_file):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            # 如果缓存文件不超过1天，使用缓存
            if (datetime.now() - file_mtime).days < 1:
                try:
                    df = pd.read_csv(cache_file)
                    if not df.empty:
                        logger.info(f"从缓存读取股票 {stock_code} 的日线数据")
                        return df
                except Exception as e:
                    logger.warning(f"读取缓存文件 {cache_file} 失败: {str(e)}")
        
        # 使用数据源获取数据
        result = None
        error = None
        
        # 先尝试使用主数据源
        if "tushare" in self.data_sources and self.primary_source == "tushare":
            try:
                result = self.data_sources["tushare"].get_daily_data(stock_code, start_date, end_date)
                if result is not None and not result.empty:
                    # 保存到缓存
                    result.to_csv(cache_file, index=False)
                    logger.info(f"从Tushare获取到 {len(result)} 条日线数据")
                    return result
                else:
                    logger.warning(f"从Tushare获取股票 {stock_code} 日线数据失败: 空数据")
            except Exception as e:
                error = str(e)
                logger.warning(f"从Tushare获取股票 {stock_code} 日线数据失败: {error}")
        
        # 如果主数据源失败，尝试AKShare
        if "akshare" in self.data_sources:
            try:
                result = self.data_sources["akshare"].get_daily_data(stock_code, start_date, end_date)
                if result is not None and not result.empty:
                    # 保存到缓存
                    result.to_csv(cache_file, index=False)
                    logger.info(f"从AKShare获取到 {len(result)} 条日线数据")
                    return result
                else:
                    logger.warning(f"从AKShare获取股票 {stock_code} 日线数据失败: 空数据")
            except Exception as e:
                error = f"{error}; AKShare错误: {str(e)}" if error else f"AKShare错误: {str(e)}"
                logger.warning(f"从AKShare获取股票 {stock_code} 日线数据失败: {str(e)}")
        
        # 如果AKShare也失败，尝试BaoStock
        if "baostock" in self.data_sources:
            try:
                result = self.data_sources["baostock"].get_daily_data(stock_code, start_date, end_date)
                if result is not None and not result.empty:
                    # 保存到缓存
                    result.to_csv(cache_file, index=False)
                    logger.info(f"从BaoStock获取到 {len(result)} 条日线数据")
                    return result
                else:
                    logger.warning(f"从BaoStock获取股票 {stock_code} 日线数据失败: 空数据")
            except Exception as e:
                error = f"{error}; BaoStock错误: {str(e)}" if error else f"BaoStock错误: {str(e)}"
                logger.warning(f"从BaoStock获取股票 {stock_code} 日线数据失败: {str(e)}")
        
        # 所有数据源都失败
        logger.error(f"未能获取到股票 {stock_code} 的日线数据")
        logger.error(f"错误详情: {error}")
        return None
    
    def get_industry_list(self) -> pd.DataFrame:
        """获取行业列表"""
        # 优先使用主要数据源
        if self.primary_source == "tushare":
            result = self.data_sources["tushare"].get_industry_list()
            if result is not None:
                return result
        
        # 如果主要数据源失败，尝试其他数据源
        if "akshare" in self.data_sources:
            try:
                df = ak.stock_sector_spot()
                logger.info(f"从AKShare获取到 {len(df)} 个行业")
                return df
            except Exception as e:
                logger.error(f"从AKShare获取行业列表失败: {e}")
        
        return None
    
    def get_stock_fund_flow(self, stock_code: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """获取个股资金流向数据"""
        # 优先使用主要数据源
        if self.primary_source == "tushare":
            result = self.data_sources["tushare"].get_stock_fund_flow(stock_code, start_date, end_date)
            if result is not None:
                return result
        
        # 如果主要数据源失败，尝试其他数据源
        if "akshare" in self.data_sources:
            try:
                # 转换股票代码格式
                if stock_code.endswith('.SH') or stock_code.endswith('.SZ'):
                    symbol = stock_code.split('.')[0]
                else:
                    symbol = stock_code
                
                df = ak.stock_individual_fund_flow(stock=symbol)
                if not df.empty:
                    # 过滤日期范围
                    df['date'] = pd.to_datetime(df['日期'])
                    df = df[(df['date'] >= start_date) & (df['date'] <= (end_date or datetime.now().strftime('%Y-%m-%d')))]
                    df = df.sort_values('date')
                    logger.info(f"从AKShare获取到 {len(df)} 条资金流向数据")
                    return df
            except Exception as e:
                logger.error(f"从AKShare获取资金流向数据失败: {e}")
        
        return None
    
    def get_continuous_limit_up_stocks(self, days=1, end_date=None):
        """获取连续涨停股票
        
        Args:
            days: 连续涨停天数
            end_date: 结束日期，默认为最新交易日
            
        Returns:
            DataFrame: 连续涨停股票数据
        """
        result = None
        logger.info(f"获取连续{days}天涨停的股票")
        
        # 首先尝试从主数据源获取
        if self.primary_source == "tushare" and "tushare" in self.data_sources:
            try:
                result = self.data_sources["tushare"].get_continuous_limit_up_stocks(days, end_date)
                if result is not None and not result.empty:
                    logger.info(f"从Tushare成功获取到 {len(result)} 只连续{days}天涨停的股票")
                    return result
                else:
                    logger.warning(f"从Tushare获取连续{days}天涨停股票返回空结果")
            except Exception as e:
                logger.error(f"从Tushare获取连续{days}天涨停股票失败: {str(e)}")
        
        # 如果主数据源获取失败，尝试备用数据源
        if "akshare" in self.data_sources:
            try:
                result = self.data_sources["akshare"].get_continuous_limit_up_stocks(days, end_date)
                if result is not None and not result.empty:
                    logger.info(f"从AKShare成功获取到 {len(result)} 只连续{days}天涨停的股票")
                    return result
                else:
                    logger.warning(f"从AKShare获取连续{days}天涨停股票返回空结果")
            except Exception as e:
                logger.error(f"从AKShare获取连续{days}天涨停股票失败: {str(e)}")
        
        # 如果仍然获取失败，尝试备用数据源
        if "baostock" in self.data_sources:
            try:
                result = self.data_sources["baostock"].get_continuous_limit_up_stocks(days, end_date)
                if result is not None and not result.empty:
                    logger.info(f"从BaoStock成功获取到 {len(result)} 只连续{days}天涨停的股票")
                    return result
                else:
                    logger.warning(f"从BaoStock获取连续{days}天涨停股票返回空结果")
            except Exception as e:
                logger.error(f"从BaoStock获取连续{days}天涨停股票失败: {str(e)}")
        
        # 所有数据源都失败，返回空DataFrame
        logger.error(f"所有数据源获取连续{days}天涨停股票都失败")
        return pd.DataFrame()
    
    def get_realtime_quotes(self, stock_codes: List[str]) -> pd.DataFrame:
        """获取实时行情数据"""
        # 优先使用主要数据源
        if self.primary_source == "tushare":
            result = self.data_sources["tushare"].get_realtime_quotes(stock_codes)
            if result is not None:
                return result
        
        # 如果主要数据源失败，尝试其他数据源
        if "akshare" in self.data_sources:
            try:
                df = ak.stock_zh_a_spot()
                if not df.empty:
                    # 过滤指定的股票
                    df = df[df['代码'].isin([code.split('.')[0] for code in stock_codes])]
                    logger.info(f"从AKShare获取到 {len(df)} 只股票的实时行情")
                    return df
            except Exception as e:
                logger.error(f"从AKShare获取实时行情失败: {e}")
        
        return None
    
    def get_stock_indicators(self, stock_code: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """获取股票技术指标数据"""
        # 优先使用主要数据源
        if self.primary_source == "tushare":
            result = self.data_sources["tushare"].get_stock_indicators(stock_code, start_date, end_date)
            if result is not None:
                return result
        
        # 如果主要数据源失败，尝试其他数据源
        if "akshare" in self.data_sources:
            try:
                # 转换股票代码格式
                if stock_code.endswith('.SH') or stock_code.endswith('.SZ'):
                    symbol = stock_code.split('.')[0]
                else:
                    symbol = stock_code
                
                df = ak.stock_a_indicator_lg(symbol=symbol)
                if not df.empty:
                    # 过滤日期范围
                    df['date'] = pd.to_datetime(df['trade_date'])
                    df = df[(df['date'] >= start_date) & (df['date'] <= (end_date or datetime.now().strftime('%Y-%m-%d')))]
                    df = df.sort_values('date')
                    logger.info(f"从AKShare获取到 {len(df)} 条技术指标数据")
                    return df
            except Exception as e:
                logger.error(f"从AKShare获取技术指标数据失败: {e}")
        
        return None 