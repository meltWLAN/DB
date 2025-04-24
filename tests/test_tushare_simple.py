#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import tushare as ts

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('src.data.tushare_fetcher')

def test_tushare():
    """简单的Tushare功能测试"""
    print("=" * 60)
    print("=" * 20 + " 简单的Tushare功能测试 " + "=" * 20)
    print("=" * 60)
    
    print("开始基本Tushare测试...")
    
    # Tushare配置
    token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    print(f"Tushare Token: {token[:5]}...{token[-5:]} (长度: {len(token)})")
    
    # 设置token
    ts.set_token(token)
    pro = ts.pro_api()
    
    # 设置日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    print(f"测试日期范围: {start_date} 至 {end_date}")
    
    # 1. 测试获取股票列表
    print("\n1. 测试获取股票列表")
    stock_list = pro.stock_basic(exchange='', list_status='L')
    print(f"成功获取到 {len(stock_list)} 只股票")
    print(f"示例: {stock_list.iloc[0].to_dict()}")
    
    # 2. 测试获取日线数据
    print("\n2. 测试获取日线数据")
    stock_code = "000001.SZ"  # 平安银行
    daily_data = pro.daily(ts_code=stock_code, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
    print(f"成功获取到 {len(daily_data)} 条日线数据")
    print(f"列: {daily_data.columns.tolist()}")
    print(f"示例: {daily_data.iloc[0].to_dict()}")
    
    # 3. 测试获取连续涨停股票
    print("\n3. 测试获取连续涨停股票")
    # 方法1：使用limit_list接口获取当日涨停股票
    trade_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')  # 使用10天前的日期
    try:
        limit_up_df = pro.limit_list(trade_date=trade_date, limit_type='U')
        print(f"成功获取 {trade_date} 的涨停板数据，共 {len(limit_up_df)} 条记录")
        if not limit_up_df.empty and len(limit_up_df) > 0:
            print(f"涨停股票示例: {limit_up_df.iloc[0]['ts_code']} - {limit_up_df.iloc[0]['name']}")
            
            # 测试获取该股票的历史日线数据
            ts_code = limit_up_df.iloc[0]['ts_code']
            hist_data = pro.daily(ts_code=ts_code, start_date=(datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=10)).strftime('%Y%m%d'), end_date=trade_date)
            print(f"获取涨停股票 {ts_code} 的历史数据，共 {len(hist_data)} 条记录")
            
            # 检查涨停情况
            hist_data['is_limit_up'] = hist_data['pct_chg'] >= 9.5
            limit_up_days = hist_data['is_limit_up'].sum()
            print(f"该股票在此期间有 {limit_up_days} 天涨停")
        else:
            print(f"{trade_date} 没有涨停股票数据")
    except Exception as e:
        print(f"获取涨停数据出错: {e}")
        
        # 方法2：通过遍历股票筛选连续涨停
        print("尝试方法2：通过日线数据筛选涨停股票")
        # 随机抽取20支股票进行测试
        test_stocks = stock_list.sample(20)
        
        for _, stock in test_stocks.iterrows():
            try:
                ts_code = stock['ts_code']
                hist_data = pro.daily(ts_code=ts_code, start_date=(datetime.now() - timedelta(days=20)).strftime('%Y%m%d'), end_date=datetime.now().strftime('%Y%m%d'))
                
                if hist_data is not None and not hist_data.empty and len(hist_data) > 0:
                    # 检查是否有涨停
                    hist_data['is_limit_up'] = hist_data['pct_chg'] >= 9.5
                    limit_up_days = hist_data['is_limit_up'].sum()
                    
                    if limit_up_days > 0:
                        print(f"股票 {ts_code} - {stock['name']} 有 {limit_up_days} 天涨停")
                        # 检查最后一次涨停是哪一天
                        if hist_data.iloc[0]['is_limit_up']:
                            print(f"最新交易日 {hist_data.iloc[0]['trade_date']} 是涨停")
            except Exception as e:
                print(f"处理股票 {ts_code} 时出错: {e}")
                continue
    
    print("\n测试完成")
        
if __name__ == "__main__":
    test_tushare() 