"""
增强型数据获取模块
提供更可靠的股票数据获取功能
"""

import logging
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EnhancedFetcher:
    """增强型数据获取类，提供更可靠的API访问"""
    
    def __init__(self, api_token=None, use_cache=True, retry_count=3, retry_delay=2):
        """初始化增强型数据获取器
        
        Args:
            api_token: API令牌
            use_cache: 是否使用缓存
            retry_count: 重试次数
            retry_delay: 重试延迟（秒）
        """
        self.api_token = api_token
        self.use_cache = use_cache
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.cache = {}
        self.cache_hits = 0
        self.api_calls = 0
        self.failed_calls = 0
        
        logger.info("增强型数据获取器初始化完成")
    
    def with_retry(self, func, *args, **kwargs):
        """使用重试机制执行函数
        
        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数
            
        Returns:
            函数执行结果
        """
        last_error = None
        for attempt in range(self.retry_count):
            try:
                self.api_calls += 1
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"API调用失败 (尝试 {attempt+1}/{self.retry_count}): {str(e)}")
                if attempt < self.retry_count - 1:
                    # 添加随机延迟，避免同时重试
                    jitter = random.uniform(0, 1)
                    sleep_time = self.retry_delay * (2 ** attempt) + jitter
                    logger.info(f"等待 {sleep_time:.2f} 秒后重试...")
                    time.sleep(sleep_time)
        
        # 所有重试都失败
        self.failed_calls += 1
        logger.error(f"API调用失败 (已尝试 {self.retry_count} 次): {str(last_error)}")
        raise last_error
    
    def get_stock_data(self, stock_code, start_date=None, end_date=None):
        """获取股票数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            股票数据DataFrame
        """
        # 检查缓存
        cache_key = f"{stock_code}_{start_date}_{end_date}"
        if self.use_cache and cache_key in self.cache:
            self.cache_hits += 1
            logger.debug(f"缓存命中: {cache_key}")
            return self.cache[cache_key]
        
        # 这里应该调用实际的API，这里只是示例
        # 在实际应用中，这里应该使用with_retry包装实际的API调用
        logger.info(f"获取股票数据: {stock_code}, {start_date} - {end_date}")
        
        # 生成模拟数据作为示例
        data = self._generate_mock_data(stock_code, start_date, end_date)
        
        # 缓存结果
        if self.use_cache:
            self.cache[cache_key] = data
        
        return data
    
    def _generate_mock_data(self, stock_code, start_date, end_date):
        """生成模拟数据（仅用于示例）"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        # 解析日期
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # 生成日期范围（仅交易日）
        date_range = pd.date_range(start=start, end=end, freq='B')
        
        # 基础价格 - 使用股票代码的数字部分作为基础，确保不同股票有不同的价格
        try:
            base_price = float(stock_code.replace('.', '')[-4:]) / 100
        except:
            base_price = 10.0
            
        if base_price < 5:
            base_price = base_price + 5
            
        # 生成价格序列 - 使用随机游走
        np.random.seed(hash(stock_code) % 10000)
        
        n = len(date_range)
        returns = np.random.normal(0.0005, 0.015, n)
        prices = base_price * (1 + np.cumsum(returns))
        
        # 确保价格为正
        prices = np.maximum(prices, 0.1)
        
        # 生成其他数据
        high = prices * (1 + np.random.uniform(0, 0.05, n))
        low = prices * (1 - np.random.uniform(0, 0.05, n))
        open_prices = prices * (1 + np.random.normal(0, 0.01, n))
        volumes = np.random.uniform(1000000, 10000000, n)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': date_range,
            'open': open_prices,
            'high': high,
            'close': prices,
            'low': low,
            'volume': volumes,
            'ts_code': stock_code
        })
        
        return df
    
    def get_api_status(self):
        """获取API状态统计"""
        total_calls = self.api_calls
        success_rate = 0 if total_calls == 0 else (total_calls - self.failed_calls) / total_calls * 100
        cache_hit_rate = 0 if total_calls == 0 else self.cache_hits / total_calls * 100
        
        return {
            'total_calls': total_calls,
            'failed_calls': self.failed_calls,
            'cache_hits': self.cache_hits,
            'success_rate': success_rate,
            'cache_hit_rate': cache_hit_rate
        }
        
    def clear_cache(self):
        """清除缓存"""
        cache_size = len(self.cache)
        self.cache = {}
        logger.info(f"已清除 {cache_size} 条缓存数据")
        return cache_size

# 创建单例实例
enhanced_fetcher = EnhancedFetcher(use_cache=True)

# 方便导入的函数
def get_stock_data(stock_code, start_date=None, end_date=None):
    """获取股票数据的便捷函数"""
    return enhanced_fetcher.get_stock_data(stock_code, start_date, end_date)

def with_retry(func, *args, **kwargs):
    """重试机制的便捷函数"""
    return enhanced_fetcher.with_retry(func, *args, **kwargs)

def get_api_status():
    """获取API状态的便捷函数"""
    return enhanced_fetcher.get_api_status()

def clear_cache():
    """清除缓存的便捷函数"""
    return enhanced_fetcher.clear_cache() 