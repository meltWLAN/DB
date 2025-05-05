#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据获取模块
负责从Tushare获取股票数据
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import time
import random

# 设置日志
logger = logging.getLogger(__name__)

class DataFetcher:
    """数据获取类，用于从Tushare获取股票数据"""
    
    def __init__(self, data_dir=None):
        """
        初始化数据获取器
        
        Args:
            data_dir: 数据目录路径，如果为None则使用默认路径
        """
        # 设置数据目录
        if data_dir is None:
            current_dir = Path(__file__).parent
            self.data_dir = current_dir / "../../data/stock_data"
        else:
            self.data_dir = Path(data_dir)
            
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 初始化Tushare数据源
        self._init_tushare()
        
    def _init_tushare(self):
        """初始化Tushare数据源"""
        try:
            import tushare as ts
            self.ts = ts
            
            # 直接使用提供的Tushare token
            token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
            
            ts.set_token(token)
            self.pro = ts.pro_api()
            logger.info("成功初始化Tushare数据源，使用直接配置的token")

        except Exception as e:
            logger.error(f"无法初始化Tushare数据源: {e}")
            self.ts = None
            self.pro = None
            raise RuntimeError(f"Tushare数据源初始化失败: {e}")
            
    def get_stock_data(self, stock_code, start_date, end_date=None, adjust="qfq"):
        """
        获取股票数据
        
        Args:
            stock_code: 股票代码，如 '600000'
            start_date: 开始日期，如 '2020-01-01'
            end_date: 结束日期，如 '2020-12-31'，默认为当前日期
            adjust: 价格调整方式，'qfq'前复权，'hfq'后复权，None不复权
            
        Returns:
            包含OHLCV数据的DataFrame，索引为日期
        """
        # 设置默认值
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # 检查本地缓存
        data = self._get_data_from_cache(stock_code, start_date, end_date, adjust)
        if data is not None and not data.empty:
            logger.info(f"从缓存加载了 {stock_code} 的数据")
            return data
            
        # 从Tushare获取数据
        try:
            if self.ts is not None:
                data = self._get_data_from_tushare(stock_code, start_date, end_date, adjust)
                if data is not None and not data.empty:
                    # 缓存数据
                    self._cache_data(stock_code, data)
                    return data
        except Exception as e:
            logger.error(f"从Tushare获取数据时出错: {str(e)}")
                
        # 如果获取失败，返回空DataFrame
        logger.warning(f"无法获取股票 {stock_code} 的数据")
        return pd.DataFrame()
        
    def _get_data_from_cache(self, stock_code, start_date, end_date, adjust):
        """从本地缓存获取数据"""
        cache_file = self.data_dir / f"{stock_code}.csv"
        
        if not cache_file.exists():
            return None
            
        try:
            # 读取数据
            data = pd.read_csv(cache_file, parse_dates=['date'], index_col='date')
            
            # 转换日期格式
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # 过滤日期范围
            data = data.loc[(data.index >= start_date) & (data.index <= end_date)]
            
            # 检查数据是否足够
            if len(data) > 0:
                return data
                
            return None
        except Exception as e:
            logger.error(f"从缓存读取数据时出错: {str(e)}")
            return None
            
    def _cache_data(self, stock_code, data):
        """缓存数据到本地"""
        cache_file = self.data_dir / f"{stock_code}.csv"
        
        try:
            # 若已有缓存，读取并与新数据合并
            if cache_file.exists():
                old_data = pd.read_csv(cache_file, parse_dates=['date'], index_col='date')
                data = pd.concat([old_data, data[~data.index.isin(old_data.index)]])
                data = data.sort_index()
                
            # 保存到CSV
            data.to_csv(cache_file)
            logger.info(f"缓存了 {stock_code} 的数据到 {cache_file}")
        except Exception as e:
            logger.error(f"缓存数据时出错: {str(e)}")
            
    def _get_data_from_tushare(self, stock_code, start_date, end_date, adjust):
        """从Tushare获取数据"""
        if self.ts is None or self.pro is None:
            return None
    
        try:
            # 对于上交所、深交所，转换为tushare代码格式
            if stock_code.startswith('6'):
                ts_code = f"{stock_code}.SH"
            elif stock_code.startswith('0') or stock_code.startswith('3'):
                ts_code = f"{stock_code}.SZ"
            else:
                ts_code = stock_code
                
            # 获取日线数据
            if adjust == 'qfq':
                df = self.ts.pro_bar(ts_code=ts_code, adj='qfq',
                             start_date=start_date.replace('-', ''), 
                             end_date=end_date.replace('-', ''),
                             freq='D')
            elif adjust == 'hfq':
                df = self.ts.pro_bar(ts_code=ts_code, adj='hfq',
                             start_date=start_date.replace('-', ''), 
                             end_date=end_date.replace('-', ''),
                             freq='D')
            else:
                df = self.ts.pro_bar(ts_code=ts_code, adj=None,
                             start_date=start_date.replace('-', ''), 
                             end_date=end_date.replace('-', ''),
                             freq='D')
                
            # 如果获取不到数据，尝试使用基础行情接口
            if df is None or df.empty:
                if self.pro is not None:
                    df = self.pro.daily(ts_code=ts_code, 
                                start_date=start_date.replace('-', ''), 
                                end_date=end_date.replace('-', ''))
            
            # 处理数据
            if df is not None and not df.empty:
                # 重命名列
                df = df.rename(columns={
                    'trade_date': 'date',
                    'vol': 'volume'
                })
                
                # 转换日期格式并设为索引
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
                # 按日期排序
                df = df.sort_index()
                
                # 选取需要的列
                cols = ['open', 'high', 'low', 'close', 'volume']
                df = df[cols]
                
                return df
            
            return None
        except Exception as e:
            logger.error(f"从Tushare获取数据时出错: {str(e)}")
            return None
            
    def get_index_data(self, index_code, start_date, end_date=None):
        """获取指数数据"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        try:
            if self.pro is None:
                return None
                
            # 转换指数代码格式
            if index_code.startswith('0') and len(index_code) == 6:
                ts_code = f"{index_code}.SH"
            elif index_code.startswith('3') and len(index_code) == 6:
                ts_code = f"{index_code}.SZ"
            else:
                ts_code = index_code
                
            # 获取指数数据
            df = self.pro.index_daily(ts_code=ts_code,
                            start_date=start_date.replace('-', ''),
                            end_date=end_date.replace('-', ''))
                            
            # 处理数据
            if df is not None and not df.empty:
                # 重命名列
                df = df.rename(columns={
                    'trade_date': 'date',
                    'vol': 'volume'
                })
                
                # 转换日期格式并设为索引
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
                # 按日期排序
                df = df.sort_index()
                
                # 选取需要的列
                cols = ['open', 'high', 'low', 'close', 'volume']
                df = df[cols]
                
                return df
                
            return None
        except Exception as e:
            logger.error(f"获取指数 {index_code} 数据时出错: {str(e)}")
            return None
            
    def get_industry_list(self) -> pd.DataFrame:
        """获取行业列表"""
        try:
            if self.pro is None:
                return pd.DataFrame()
                
            # 调用tushare_fetcher中的方法获取行业列表
            result = self.pro.index_classify(level='L1', src='sw')
            
            if result is not None and not result.empty:
                return result[['index_code', 'industry_name']]
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取行业列表时出错: {str(e)}")
            return pd.DataFrame()