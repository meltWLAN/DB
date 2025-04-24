#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版AKShare数据获取器
提供对AKShare API的封装和增强功能
"""

import logging
from typing import Dict, Optional, List
import time
import pandas as pd
import akshare as ak

logger = logging.getLogger(__name__)

class EnhancedAKShareFetcher:
    """增强版AKShare数据获取器"""
    
    def __init__(self, config: Dict):
        """
        初始化AKShare数据获取器
        
        Args:
            config: 配置字典，包含rate_limit和retry等参数
        """
        self.rate_limit = config.get('rate_limit', 2.0)  # 默认每秒2次请求
        self.connection_retries = config.get('connection_retries', 3)
        self.retry_delay = config.get('retry_delay', 5)
        self.last_request_time = 0
        logger.info("增强版AKShare数据获取器初始化完成")
        
    def _wait_for_rate_limit(self):
        """等待以遵守速率限制"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < 1.0 / self.rate_limit:
            time.sleep(1.0 / self.rate_limit - time_since_last_request)
        self.last_request_time = time.time()
        
    def _execute_api_call(self, func, *args, **kwargs) -> Optional[pd.DataFrame]:
        """
        执行API调用，包含重试逻辑
        
        Args:
            func: 要执行的API函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            DataFrame或None
        """
        for attempt in range(self.connection_retries):
            try:
                self._wait_for_rate_limit()
                result = func(*args, **kwargs)
                if isinstance(result, pd.DataFrame):
                    return result
                logger.warning(f"API调用返回非DataFrame结果: {type(result)}")
                return None
            except Exception as e:
                logger.warning(f"API调用失败(尝试 {attempt + 1}/{self.connection_retries}): {str(e)}")
                if attempt < self.connection_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"API调用在多次尝试后仍然失败")
                    return None
                    
    def check_health(self) -> bool:
        """
        检查API连接健康状态
        
        Returns:
            bool: 是否健康
        """
        try:
            # 尝试获取股票列表作为健康检查
            result = self._execute_api_call(ak.stock_info_a_code_name)
            return result is not None and len(result) > 0
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            return False
            
    def get_stock_list(self) -> Optional[pd.DataFrame]:
        """
        获取股票列表
        
        Returns:
            DataFrame或None: 包含股票代码和名称的DataFrame
        """
        try:
            df = self._execute_api_call(ak.stock_info_a_code_name)
            if df is not None:
                df.columns = ['code', 'name']
                logger.info(f"成功获取 {len(df)} 只股票的信息")
                return df
            return None
        except Exception as e:
            logger.error(f"获取股票列表失败: {str(e)}")
            return None
            
    def get_daily_data(self, symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取股票日线数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期，格式：YYYYMMDD
            end_date: 结束日期，格式：YYYYMMDD
            
        Returns:
            DataFrame或None: 包含日线数据的DataFrame
        """
        try:
            # 移除可能的后缀
            pure_symbol = symbol.split('.')[0]
            
            df = self._execute_api_call(
                ak.stock_zh_a_hist,
                symbol=pure_symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            
            if df is not None:
                # 标准化列名
                df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_chg', 'change', 'turnover']
                # 只保留需要的列
                df = df[['date', 'open', 'close', 'high', 'low', 'volume', 'amount']]
                # 确保日期格式正确
                df['date'] = pd.to_datetime(df['date'])
                # 确保数值列为float类型
                numeric_columns = ['open', 'close', 'high', 'low', 'volume', 'amount']
                df[numeric_columns] = df[numeric_columns].astype(float)
                
                logger.info(f"成功获取股票 {symbol} 的日线数据，共 {len(df)} 条记录")
                return df
                
            return None
        except Exception as e:
            logger.error(f"获取股票 {symbol} 的日线数据失败: {str(e)}")
            return None 