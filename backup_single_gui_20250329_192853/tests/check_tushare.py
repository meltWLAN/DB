#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
import time
import tushare as ts

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger()

def test_tushare():
    """直接测试Tushare API功能"""
    print("=" * 60)
    print("=" * 20 + " Tushare API 功能测试 " + "=" * 20)
    print("=" * 60)
    
    # Tushare配置
    token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    
    print(f"Tushare Token: {token[:5]}...{token[-5:]} (长度: {len(token)})")
    
    # 设置token
    ts.set_token(token)
    
    # 初始化pro API
    try:
        pro = ts.pro_api()
        print("初始化Tushare API成功")
    except Exception as e:
        print(f"初始化Tushare API失败: {str(e)}")
        return
    
    # 1. 测试获取股票基本信息
    try:
        df = pro.stock_basic(exchange='', list_status='L')
        print(f"获取股票列表成功，共 {len(df)} 只股票")
        if not df.empty:
            print(f"示例股票: {df.iloc[0]['ts_code']} - {df.iloc[0]['name']}")
    except Exception as e:
        print(f"获取股票列表失败: {str(e)}")
    
    # 2. 测试获取日线数据
    try:
        stock_code = "000001.SZ"  # 平安银行
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        
        df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
        print(f"获取 {stock_code} 的日线数据成功，共 {len(df)} 条记录")
        if not df.empty:
            print(f"最新记录: {df.iloc[0]['trade_date']} 收盘价: {df.iloc[0]['close']}")
    except Exception as e:
        print(f"获取日线数据失败: {str(e)}")
    
    # 3. 测试获取涨停股票
    try:
        trade_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        df = pro.limit_list(trade_date=trade_date, limit_type='U')
        print(f"获取 {trade_date} 的涨停股票成功，共 {len(df)} 只股票")
        if not df.empty and len(df) > 0:
            print(f"涨停股票示例: {df.iloc[0]['ts_code']} - {df.iloc[0]['name']}")
    except Exception as e:
        print(f"获取涨停股票失败: {str(e)}")
    
    # 4. 测试获取行业数据
    try:
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        industries = df['industry'].unique()
        print(f"获取行业数据成功，共 {len(industries)} 个行业")
        if len(industries) > 0:
            print(f"行业示例: {industries[0:5]}")
    except Exception as e:
        print(f"获取行业数据失败: {str(e)}")
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_tushare() 