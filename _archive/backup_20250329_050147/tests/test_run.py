#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import time
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# 导入主系统
from main import StockAnalysisSystem

def test_system():
    """测试股票分析系统的基本功能"""
    logger.info("开始测试股票分析系统")
    
    # 创建系统实例
    system = StockAnalysisSystem()
    
    # 设置日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    logger.info(f"测试日期范围: {start_date} 至 {end_date}")
    
    # 1. 测试数据准备
    logger.info("1. 测试数据准备功能")
    success = system.prepare_data(start_date, end_date)
    if success:
        logger.info("数据准备成功")
    else:
        logger.error("数据准备失败")
        return
    
    # 2. 测试单个股票分析
    logger.info("2. 测试单个股票分析功能")
    stock_code = "000001.SZ"  # 平安银行
    stock_result = system.analyze_stock(stock_code, start_date, end_date)
    
    if stock_result:
        logger.info(f"分析股票 {stock_code} 成功")
        logger.info(f"股票名称: {stock_result['stock_name']}")
        logger.info(f"最新价格: {stock_result['latest_price']}")
        logger.info(f"评分: {stock_result['score']}")
        logger.info(f"信号数量: {len(stock_result['signals'])}")
    else:
        logger.warning(f"分析股票 {stock_code} 失败")
    
    # 3. 测试连续涨停股票
    logger.info("3. 测试连续涨停股票识别")
    for days in system.limit_up_stocks:
        limit_up_df = system.limit_up_stocks[days]
        if limit_up_df is not None and not limit_up_df.empty:
            logger.info(f"找到 {len(limit_up_df)} 只连续 {days} 天涨停的股票")
            # 打印第一只涨停股票的信息
            if len(limit_up_df) > 0:
                first_stock = limit_up_df.iloc[0]
                logger.info(f"示例: {first_stock['ts_code']} {first_stock['name']} - 涨幅: {first_stock['pct_chg']}%")
        else:
            logger.info(f"没有找到连续 {days} 天涨停的股票")
    
    # 4. 不运行完整的系统，只测试关键功能
    logger.info("测试完成")

if __name__ == "__main__":
    test_system() 