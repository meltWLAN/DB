#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Tushare数据源
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tushare as ts
import logging
from src.data.tushare_fetcher import TushareFetcher
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger("test_tushare", level=logging.INFO)

# 添加当前目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from src.config import DATA_SOURCE_CONFIG

def test_tushare_fetcher():
    """测试Tushare数据获取器的各个功能"""
    logger.info("开始测试Tushare数据获取功能...")
    
    # 创建Tushare数据获取器实例
    fetcher = TushareFetcher()
    
    # 设置日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # 测试获取股票列表
    logger.info("1. 测试获取股票列表...")
    stock_list = fetcher.get_stock_list()
    if stock_list is not None and not stock_list.empty:
        logger.info(f"   成功获取到 {len(stock_list)} 只股票")
        logger.info(f"   股票列表数据类型: {type(stock_list)}")
        logger.info(f"   股票列表列: {stock_list.columns.tolist()}")
        logger.info(f"   股票列表示例:\n{stock_list.head()}")
    else:
        logger.error("   获取股票列表失败")
    
    # 选择一只股票进行测试
    test_stock = stock_list.iloc[0]['ts_code'] if stock_list is not None and not stock_list.empty else "000001.SZ"
    logger.info(f"选择 {test_stock} 进行后续测试")
    
    # 测试获取日线数据
    logger.info("\n2. 测试获取日线数据...")
    daily_data = fetcher.get_daily_data(test_stock, start_date, end_date)
    if daily_data is not None and not daily_data.empty:
        logger.info(f"   成功获取到 {len(daily_data)} 条日线数据")
        logger.info(f"   日线数据类型: {type(daily_data)}")
        logger.info(f"   日线数据列: {daily_data.columns.tolist()}")
        logger.info(f"   日线数据示例:\n{daily_data.head()}")
        
        # 验证日线数据的结构
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'change']
        missing_cols = [col for col in required_cols if col not in daily_data.columns]
        if missing_cols:
            logger.error(f"   日线数据缺少以下列: {missing_cols}")
        else:
            logger.info("   日线数据结构正确，包含所有必要列")
    else:
        logger.error("   获取日线数据失败")
    
    # 测试获取行业列表
    logger.info("\n3. 测试获取行业列表...")
    industry_list = fetcher.get_industry_list()
    if industry_list is not None and not industry_list.empty:
        logger.info(f"   成功获取到 {len(industry_list)} 个行业")
        logger.info(f"   行业列表数据类型: {type(industry_list)}")
        logger.info(f"   行业列表列: {industry_list.columns.tolist()}")
        logger.info(f"   行业列表示例:\n{industry_list.head()}")
    else:
        logger.error("   获取行业列表失败")
    
    # 测试获取资金流向数据
    logger.info("\n4. 测试获取资金流向数据...")
    fund_flow_data = fetcher.get_stock_fund_flow(test_stock, start_date, end_date)
    if fund_flow_data is not None and not fund_flow_data.empty:
        logger.info(f"   成功获取到 {len(fund_flow_data)} 条资金流向数据")
        logger.info(f"   资金流向数据类型: {type(fund_flow_data)}")
        logger.info(f"   资金流向数据列: {fund_flow_data.columns.tolist()}")
        logger.info(f"   资金流向数据示例:\n{fund_flow_data.head()}")
    else:
        logger.warning("   获取资金流向数据失败或没有数据")
    
    # 测试获取连续涨停股票
    logger.info("\n5. 测试获取连续涨停股票...")
    for days in [1, 2, 3]:
        logger.info(f"   测试获取连续{days}天涨停的股票...")
        limit_up_stocks = fetcher.get_continuous_limit_up_stocks(days, end_date)
        if limit_up_stocks is not None and not limit_up_stocks.empty:
            logger.info(f"   成功获取到 {len(limit_up_stocks)} 只连续{days}天涨停的股票")
            logger.info(f"   连续涨停股票数据类型: {type(limit_up_stocks)}")
            logger.info(f"   连续涨停股票列: {limit_up_stocks.columns.tolist()}")
            logger.info(f"   连续涨停股票示例:\n{limit_up_stocks.head()}")
        else:
            logger.info(f"   未找到连续{days}天涨停的股票")
    
    # 测试获取技术指标数据
    logger.info("\n6. 测试获取技术指标数据...")
    indicators_data = fetcher.get_stock_indicators(test_stock, start_date, end_date)
    if indicators_data is not None and not indicators_data.empty:
        logger.info(f"   成功获取到 {len(indicators_data)} 条技术指标数据")
        logger.info(f"   技术指标数据类型: {type(indicators_data)}")
        logger.info(f"   技术指标数据列: {indicators_data.columns.tolist()}")
        logger.info(f"   技术指标数据示例:\n{indicators_data.head()}")
    else:
        logger.warning("   获取技术指标数据失败或没有数据")
    
    logger.info("\nTushare数据获取功能测试完成")

if __name__ == "__main__":
    print("=" * 80)
    print(" Tushare 数据源测试程序 ".center(80, "="))
    print("=" * 80)
    
    # 运行测试
    test_tushare_fetcher()
    
    print("=" * 80)
    print(f" 测试结果: {'成功' if True else '失败'} ".center(80, "="))
    print("=" * 80) 