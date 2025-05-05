#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版AKShare数据获取器
提供对AKShare API的封装和增强功能
"""

import logging
from typing import Dict, Optional, List, Any
import time
import pandas as pd
import akshare as ak
from datetime import datetime

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
    
    def get_market_overview(self, trade_date: str = None) -> Dict[str, Any]:
        """
        获取市场概览数据
        
        Args:
            trade_date: 交易日期，格式 YYYY-MM-DD，默认为最新交易日
            
        Returns:
            Dict: 市场概览数据
        """
        try:
            # 获取市场宏观数据
            market_data = self._execute_api_call(ak.stock_zh_index_daily, symbol="sh000001")
            
            if market_data is None or market_data.empty:
                logger.warning(f"无法获取市场数据")
                return {}
            
            # 获取当天所有A股数据
            all_stocks = self._execute_api_call(ak.stock_zh_a_spot_em)
            
            if all_stocks is None or all_stocks.empty:
                logger.warning(f"无法获取A股实时行情数据")
                return {}
            
            # 计算涨跌家数
            all_stocks['涨跌幅'] = pd.to_numeric(all_stocks['涨跌幅'], errors='coerce')
            up_count = len(all_stocks[all_stocks['涨跌幅'] > 0])
            down_count = len(all_stocks[all_stocks['涨跌幅'] < 0])
            flat_count = len(all_stocks) - up_count - down_count
            
            # 计算涨停跌停数量
            limit_up_count = len(all_stocks[all_stocks['涨跌幅'] > 9.5])
            limit_down_count = len(all_stocks[all_stocks['涨跌幅'] < -9.5])
            
            # 计算成交量和成交额
            all_stocks['成交量'] = pd.to_numeric(all_stocks['成交量'], errors='coerce')
            all_stocks['成交额'] = pd.to_numeric(all_stocks['成交额'], errors='coerce')
            total_volume = all_stocks['成交量'].sum()
            total_amount = all_stocks['成交额'].sum()
            
            # 计算平均涨跌幅
            avg_change = all_stocks['涨跌幅'].mean()
            
            # 构建市场概览数据
            overview_data = {
                'date': trade_date if trade_date else datetime.now().strftime('%Y-%m-%d'),
                'up_count': int(up_count),
                'down_count': int(down_count),
                'flat_count': int(flat_count),
                'total_count': len(all_stocks),
                'limit_up_count': int(limit_up_count),
                'limit_down_count': int(limit_down_count),
                'total_volume': float(total_volume),
                'total_amount': float(total_amount),
                'avg_change_pct': float(avg_change),
                'turnover_rate': float(total_volume / total_amount * 100 if total_amount > 0 else 0)
            }
            
            logger.debug(f"成功获取市场概览数据")
            return overview_data
            
        except Exception as e:
            logger.error(f"获取市场概览数据失败: {str(e)}")
            return {}
            
    def get_stock_index_data(self, index_code: str, start_date: str, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取指数日线数据
        
        Args:
            index_code: 指数代码，如 000001.SH
            start_date: 开始日期，格式 YYYY-MM-DD
            end_date: 结束日期，格式 YYYY-MM-DD，默认为今天
            
        Returns:
            Optional[pd.DataFrame]: 指数日线数据
        """
        try:
            # 转换指数代码格式为AKShare格式
            if index_code.endswith('.SH'):
                symbol = 'sh' + index_code.split('.')[0]
            elif index_code.endswith('.SZ'):
                symbol = 'sz' + index_code.split('.')[0]
            else:
                symbol = index_code
                
            # 获取指数日线数据
            df = self._execute_api_call(
                ak.stock_zh_index_daily,
                symbol=symbol
            )
            
            if df is not None and not df.empty:
                # 标准化列名
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
                
                # 确保日期格式正确
                df['date'] = pd.to_datetime(df['date'])
                
                # 按日期过滤
                start_date_ts = pd.to_datetime(start_date)
                if end_date:
                    end_date_ts = pd.to_datetime(end_date)
                    df = df[(df['date'] >= start_date_ts) & (df['date'] <= end_date_ts)]
                else:
                    df = df[df['date'] >= start_date_ts]
                
                # 转换为标准格式
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                df['code'] = index_code
                
                # 确保数值列为浮点数
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                
                logger.debug(f"获取到 {index_code} 的 {len(df)} 条指数日线数据")
                return df
            
            return None
        except Exception as e:
            logger.error(f"获取 {index_code} 的指数日线数据失败: {str(e)}")
            return None 