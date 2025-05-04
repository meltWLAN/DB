#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TuShare API 测试脚本
用于测试TuShare API连接是否正常，验证令牌有效性
"""

import logging
import sys
import pandas as pd
import time
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_tushare_api():
    """测试TuShare API连接和数据获取"""
    try:
        import tushare as ts
        
        # 使用已知的token
        token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
        logger.info(f"使用Token: {token[:5]}...{token[-5:]}")
        
        # 设置token并初始化API
        ts.set_token(token)
        pro = ts.pro_api()
        logger.info("TuShare Pro API初始化成功")
        
        # 测试1: 获取交易日历
        logger.info("测试1: 获取交易日历...")
        today = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        trade_cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=today)
        if trade_cal is not None and not trade_cal.empty:
            logger.info(f"成功获取交易日历，共{len(trade_cal)}条记录")
            logger.info(f"最近的交易日: {trade_cal.tail(3)['cal_date'].tolist()}")
        else:
            logger.error("获取交易日历失败")
        
        # 测试2: 获取上证指数数据
        logger.info("\n测试2: 获取上证指数数据...")
        index_data = pro.index_daily(ts_code='000001.SH', start_date=start_date, end_date=today)
        if index_data is not None and not index_data.empty:
            logger.info(f"成功获取上证指数数据，共{len(index_data)}条记录")
            logger.info(f"最新收盘价: {index_data.iloc[0]['close']}")
        else:
            logger.error("获取上证指数数据失败")
        
        # 测试3: 获取股票列表
        logger.info("\n测试3: 获取股票列表...")
        stocks = pro.stock_basic(exchange='', list_status='L')
        if stocks is not None and not stocks.empty:
            logger.info(f"成功获取股票列表，共{len(stocks)}只股票")
            sample_stocks = stocks.sample(5)
            logger.info(f"随机5只股票:\n{sample_stocks[['ts_code', 'name', 'industry']]}")
        else:
            logger.error("获取股票列表失败")
        
        # 测试4: 获取个股日线数据
        logger.info("\n测试4: 获取个股日线数据...")
        if stocks is not None and not stocks.empty:
            sample_stock = stocks.iloc[0]['ts_code']
            logger.info(f"获取股票 {sample_stock} 的日线数据...")
            stock_daily = pro.daily(ts_code=sample_stock, start_date=start_date, end_date=today)
            if stock_daily is not None and not stock_daily.empty:
                logger.info(f"成功获取{sample_stock}的日线数据，共{len(stock_daily)}条记录")
                logger.info(f"最近交易数据:\n{stock_daily.head(3)[['trade_date', 'open', 'high', 'low', 'close', 'vol']]}")
            else:
                logger.error(f"获取{sample_stock}的日线数据失败")
        
        # 测试5: 使用limit_list接口获取涨停数据
        logger.info("\n测试5: 获取涨停数据...")
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        try:
            limit_list = pro.limit_list(trade_date=yesterday, limit_type='U')
            if limit_list is not None and not limit_list.empty:
                logger.info(f"成功获取{yesterday}的涨停数据，共{len(limit_list)}条记录")
                logger.info(f"部分涨停股票:\n{limit_list.head(3)[['ts_code', 'name', 'pct_chg', 'amount']]}")
            else:
                logger.warning(f"没有找到{yesterday}的涨停数据")
        except Exception as e:
            logger.error(f"获取涨停数据失败: {str(e)}")
        
        logger.info("\nTuShare API 测试完成!")
        return True
    
    except Exception as e:
        logger.error(f"TuShare API 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_tushare_api() 