#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TuShare数据获取器
提供与TuShare API的交互接口
"""

import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

import tushare as ts

# 设置日志
logger = logging.getLogger(__name__)

class EnhancedTushareFetcher:
    """
    增强版TuShare数据获取器
    实现与TuShare API的交互，提供数据下载和转换功能
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化TuShare获取器
        
        Args:
            config: 配置字典，包含api_token等信息
        """
        self.config = config or {}
        self.token = self.config.get('token', '')
        self.api = None
        self.connection_retries = self.config.get('connection_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 2)
        self.rate_limit = self.config.get('rate_limit', 3)  # 每秒请求数
        self.request_interval = 1.0 / self.rate_limit if self.rate_limit > 0 else 0
        self.last_request_time = 0
        
        # 初始化API连接
        self._init_api()
        
        logger.info("增强版TuShare数据获取器初始化完成")
    
    def _init_api(self):
        """初始化TuShare API连接"""
        try:
            if self.token:
                ts.set_token(self.token)
                self.api = ts.pro_api()
                logger.info(f"TuShare API连接成功，Token前5位: {self.token[:5]}...")
            else:
                logger.warning("未提供TuShare Token，将使用受限功能")
                self.api = ts.pro_api()
        except Exception as e:
            logger.error(f"TuShare API连接失败: {str(e)}")
            self.api = None
    
    def _execute_api_call(self, api_name: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        执行API调用，带有重试和限速
        
        Args:
            api_name: API名称
            **kwargs: API参数
            
        Returns:
            Optional[pd.DataFrame]: API返回的数据，如果失败则返回None
        """
        if self.api is None:
            logger.error("TuShare API未初始化，无法执行调用")
            return None
        
        # 限速控制
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_interval:
            time.sleep(self.request_interval - time_since_last_request)
        
        # 执行API调用，带有重试
        for attempt in range(self.connection_retries):
            try:
                # 调用对应的API方法
                method = getattr(self.api, api_name)
                data = method(**kwargs)
                self.last_request_time = time.time()
                
                # 检查数据是否有效
                if data is not None and not data.empty:
                    return data
                else:
                    logger.warning(f"API调用 {api_name} 返回空数据")
                    return None
            except Exception as e:
                logger.warning(f"API调用 {api_name} 失败(尝试 {attempt+1}/{self.connection_retries}): {str(e)}")
                if attempt < self.connection_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # 指数退避
                else:
                    logger.error(f"API调用 {api_name} 在多次尝试后仍然失败")
        
        return None
    
    def check_health(self) -> bool:
        """
        检查API健康状态
        
        Returns:
            bool: API是否健康
        """
        try:
            # 尝试获取交易日历作为健康检查
            today = datetime.now().strftime('%Y%m%d')
            last_month = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            data = self._execute_api_call('trade_cal', start_date=last_month, end_date=today)
            return data is not None and not data.empty
        except Exception as e:
            logger.error(f"TuShare健康检查失败: {str(e)}")
            return False
    
    def get_stock_list(self) -> Optional[pd.DataFrame]:
        """
        获取股票列表
        
        Returns:
            Optional[pd.DataFrame]: 股票列表数据
        """
        try:
            # 获取股票基本信息
            data = self._execute_api_call('stock_basic', 
                                          exchange='', 
                                          list_status='L',
                                          fields='ts_code,symbol,name,area,industry,list_date')
            
            if data is not None and not data.empty:
                # 添加交易所列
                data['exchange'] = data['ts_code'].apply(lambda x: x.split('.')[-1])
                
                # 规范化列名和格式
                data = data.rename(columns={
                    'ts_code': 'code',
                    'symbol': 'symbol',
                    'name': 'name',
                    'area': 'area',
                    'industry': 'industry',
                    'list_date': 'list_date'
                })
                
                logger.info(f"获取到 {len(data)} 只股票基本信息")
                return data
            
            return None
        except Exception as e:
            logger.error(f"获取股票列表失败: {str(e)}")
            return None
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取日线数据
        
        Args:
            stock_code: 股票代码，如 000001.SZ
            start_date: 开始日期，格式 YYYY-MM-DD
            end_date: 结束日期，格式 YYYY-MM-DD，默认为今天
            
        Returns:
            Optional[pd.DataFrame]: 日线数据
        """
        try:
            # 转换日期格式
            start_date_fmt = start_date.replace('-', '')
            end_date_fmt = end_date.replace('-', '') if end_date else datetime.now().strftime('%Y%m%d')
            
            # 获取日线数据
            data = self._execute_api_call('daily', 
                                        ts_code=stock_code,
                                        start_date=start_date_fmt,
                                        end_date=end_date_fmt)
            
            if data is not None and not data.empty:
                # 规范化列名和格式
                data = data.rename(columns={
                    'trade_date': 'date',
                    'vol': 'volume',
                    'ts_code': 'code'
                })
                
                # 转换日期格式
                data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
                
                # 按日期升序排列
                data = data.sort_values('date')
                
                # 确保数值列为浮点数
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
                for col in numeric_cols:
                    if col in data.columns:
                        data[col] = data[col].astype(float)
                
                logger.debug(f"获取到 {stock_code} 的 {len(data)} 条日线数据")
                return data
            
            return None
        except Exception as e:
            logger.error(f"获取日线数据失败: {str(e)}")
            return None
    
    def get_industry_list(self) -> Optional[pd.DataFrame]:
        """
        获取行业列表
        
        Returns:
            Optional[pd.DataFrame]: 行业列表数据
        """
        try:
            # 获取申万行业分类
            data = self._execute_api_call('index_classify', src='SW')
            
            if data is not None and not data.empty:
                # 规范化列名
                data = data.rename(columns={
                    'index_code': 'industry_code',
                    'industry_name': 'industry_name',
                    'level': 'level',
                    'industry_id': 'industry_id'
                })
                
                logger.debug(f"获取到 {len(data)} 个行业分类")
                return data
            
            return None
        except Exception as e:
            logger.error(f"获取行业列表失败: {str(e)}")
            return None
    
    def get_stock_fund_flow(self, stock_code: str, start_date: str, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取个股资金流向
        
        Args:
            stock_code: 股票代码，如 000001.SZ
            start_date: 开始日期，格式 YYYY-MM-DD
            end_date: 结束日期，格式 YYYY-MM-DD，默认为今天
            
        Returns:
            Optional[pd.DataFrame]: 资金流向数据
        """
        try:
            # 转换日期格式
            start_date_fmt = start_date.replace('-', '')
            end_date_fmt = end_date.replace('-', '') if end_date else datetime.now().strftime('%Y%m%d')
            
            # 获取资金流向数据
            data = self._execute_api_call('moneyflow', 
                                         ts_code=stock_code,
                                         start_date=start_date_fmt,
                                         end_date=end_date_fmt)
            
            if data is not None and not data.empty:
                # 规范化列名和格式
                data = data.rename(columns={
                    'trade_date': 'date',
                    'ts_code': 'code',
                    'buy_sm_vol': 'small_buy_volume',
                    'sell_sm_vol': 'small_sell_volume',
                    'buy_md_vol': 'medium_buy_volume',
                    'sell_md_vol': 'medium_sell_volume',
                    'buy_lg_vol': 'large_buy_volume',
                    'sell_lg_vol': 'large_sell_volume',
                    'buy_elg_vol': 'extra_large_buy_volume',
                    'sell_elg_vol': 'extra_large_sell_volume',
                    'net_mf_vol': 'net_volume',
                    'net_mf_amount': 'net_amount'
                })
                
                # 转换日期格式
                data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
                
                # 按日期升序排列
                data = data.sort_values('date')
                
                logger.debug(f"获取到 {stock_code} 的 {len(data)} 条资金流向数据")
                return data
            
            return None
        except Exception as e:
            logger.error(f"获取 {stock_code} 的资金流向数据失败: {str(e)}")
            return None
    
    def get_continuous_limit_up_stocks(self, days: int = 1, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取连续涨停股票
        
        Args:
            days: 连续涨停天数
            end_date: 结束日期，格式 YYYY-MM-DD，默认为今天
            
        Returns:
            Optional[pd.DataFrame]: 涨停股票数据
        """
        try:
            # 处理日期
            end_date_fmt = end_date.replace('-', '') if end_date else datetime.now().strftime('%Y%m%d')
            
            # 构建结果DataFrame
            result_stocks = []
            
            # 获取当天的股票列表
            stocks = self.get_stock_list()
            if stocks is None or stocks.empty:
                return None
            
            # 计算开始日期，多获取几天来判断连续涨停
            days_to_fetch = days + 5  # 多获取几天以确保可以判断连续性
            start_date = (datetime.strptime(end_date_fmt, '%Y%m%d') - timedelta(days=days_to_fetch)).strftime('%Y%m%d')
            
            # 获取所有股票在这段时间的行情数据
            for _, stock in stocks.iterrows():
                stock_code = stock['code']
                
                # 获取日线数据
                daily_data = self.get_daily_data(stock_code, start_date, end_date)
                if daily_data is None or len(daily_data) < days:
                    continue
                
                # 计算每日涨跌幅
                daily_data['pct_change'] = daily_data['close'].pct_change() * 100
                daily_data = daily_data.dropna(subset=['pct_change'])
                
                # 判断是否涨停
                # 涨停标准: 涨幅超过9.5%
                daily_data['is_limit_up'] = daily_data['pct_change'] > 9.5
                
                # 检查最近几天是否连续涨停
                recent_data = daily_data.tail(days)
                if len(recent_data) == days and all(recent_data['is_limit_up']):
                    # 添加到结果中
                    result_stocks.append({
                        'code': stock_code,
                        'name': stock['name'],
                        'industry': stock.get('industry', ''),
                        'consecutive_days': days,
                        'last_date': recent_data['date'].iloc[-1]
                    })
            
            # 创建结果DataFrame
            if result_stocks:
                result_df = pd.DataFrame(result_stocks)
                logger.debug(f"找到 {len(result_df)} 只连续 {days} 天涨停的股票")
                return result_df
            else:
                logger.debug(f"没有找到连续 {days} 天涨停的股票")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取连续涨停股票失败: {str(e)}")
            return None 