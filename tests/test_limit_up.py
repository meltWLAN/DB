#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from datetime import datetime, timedelta
from src.data.fetcher import StockDataFetcher
from src.utils.logger import get_logger
import pytest
import pandas as pd

# 设置日志
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG) # Explicitly set level if needed for tests

def test_limit_up_stocks():
    """测试连续涨停股票获取功能"""
    # 创建数据获取器实例
    fetcher = StockDataFetcher()
    
    # 设置日期范围 - 使用更长的时间范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    logger.info(f"测试日期范围: {start_date} 至 {end_date}")
    
    # 先获取一只股票的日线数据，检查是否能正确获取price_chg字段
    logger.info("测试日线数据获取...")
    stock_list = fetcher.get_stock_list()
    if stock_list is not None and not stock_list.empty:
        test_stock = stock_list.iloc[0]['ts_code']
        logger.info(f"测试股票: {test_stock}")
        
        daily_data = fetcher.get_daily_data(test_stock, start_date, end_date)
        if daily_data is not None and not daily_data.empty:
            logger.info(f"获取到 {len(daily_data)} 条日线数据")
            logger.info(f"日线数据列: {daily_data.columns.tolist()}")
            logger.info(f"日线数据示例:\n{daily_data.head()}")
        else:
            logger.error("获取日线数据失败")
    
    # 测试连续涨停股票获取
    logger.info("\n测试连续涨停股票获取...")
    
    for days in [1, 2, 3]:
        logger.info(f"测试连续{days}天涨停的股票...")
        try:
            limit_up_stocks = fetcher.get_continuous_limit_up_stocks(days=days)
            if limit_up_stocks is not None and not limit_up_stocks.empty:
                logger.info(f"成功获取到 {len(limit_up_stocks)} 只连续{days}天涨停的股票")
                logger.info(f"连续{days}天涨停股票示例:\n{limit_up_stocks.head()}")
            else:
                logger.info(f"未找到连续{days}天涨停的股票")
        except Exception as e:
            logger.error(f"获取连续{days}天涨停股票时出错: {str(e)}")

if __name__ == "__main__":
    test_limit_up_stocks() 