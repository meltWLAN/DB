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

@pytest.fixture
def fetcher():
    # 创建数据获取器实例
    return StockDataFetcher()

def test_data_fetching():
    """测试数据获取功能"""
    # 创建数据获取器实例
    fetcher = StockDataFetcher()
    
    # 设置日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # 1. 测试获取股票列表
    logger.info("测试获取股票列表...")
    stock_list = fetcher.get_stock_list()
    if stock_list is not None and not stock_list.empty:
        logger.info(f"成功获取到 {len(stock_list)} 只股票")
        logger.info("股票列表示例：")
        logger.info(stock_list.head())
    else:
        logger.error("获取股票列表失败")
    
    # 2. 测试获取行业列表
    logger.info("\n测试获取行业列表...")
    industry_list = fetcher.get_industry_list()
    if industry_list is not None and not industry_list.empty:
        logger.info(f"成功获取到 {len(industry_list)} 个行业")
        logger.info("行业列表示例：")
        logger.info(industry_list.head())
    else:
        logger.error("获取行业列表失败")
    
    # 3. 测试获取日线数据
    logger.info("\n测试获取日线数据...")
    if stock_list is not None and not stock_list.empty:
        test_stock = stock_list.iloc[0]['ts_code']
        daily_data = fetcher.get_daily_data(test_stock, start_date, end_date)
        if daily_data is not None and not daily_data.empty:
            logger.info(f"成功获取股票 {test_stock} 的日线数据")
            logger.info("日线数据示例：")
            logger.info(daily_data.head())
        else:
            logger.error(f"获取股票 {test_stock} 的日线数据失败")
    
    # 4. 测试获取连续涨停股票
    logger.info("\n测试获取连续涨停股票...")
    for days in [1, 2, 3]:
        limit_up_stocks = fetcher.get_continuous_limit_up_stocks(days=days, end_date=end_date)
        if limit_up_stocks is not None and not limit_up_stocks.empty:
            logger.info(f"成功获取到 {len(limit_up_stocks)} 只连续{days}天涨停的股票")
            logger.info(f"连续{days}天涨停股票示例：")
            logger.info(limit_up_stocks.head())
        else:
            logger.info(f"没有找到连续{days}天涨停的股票")

if __name__ == "__main__":
    test_data_fetching() 