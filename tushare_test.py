#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tushare API测试脚本，验证API连接和数据获取功能
"""

import tushare as ts
import pandas as pd
import time
from datetime import datetime, timedelta

# 用户提供的token
USER_TOKEN = '0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10'

def main():
    print("开始测试Tushare API...")
    
    # 1. 验证Tushare版本
    print(f"Tushare版本: {ts.__version__}")
    
    # 2. 设置token并初始化API
    print(f"设置Tushare token...")
    ts.set_token(USER_TOKEN)
    pro_api = ts.pro_api()
    
    if pro_api is None:
        print("Tushare Pro API初始化失败")
        return
    
    print("Tushare Pro API初始化成功")
    
    # 3. 获取交易日历测试
    print("\n测试获取交易日历...")
    try:
        cal = pro_api.trade_cal(exchange='', start_date='20240501', end_date='20240531')
        if cal is not None and not cal.empty:
            print(f"成功获取交易日历数据: {len(cal)}条记录")
            print(cal.head(5))
        else:
            print("获取交易日历返回为空")
    except Exception as e:
        print(f"获取交易日历失败: {e}")
    
    # 4. 获取股票列表测试
    print("\n测试获取股票列表...")
    try:
        stocks = pro_api.stock_basic(exchange='', list_status='L')
        if stocks is not None and not stocks.empty:
            print(f"成功获取股票列表: {len(stocks)}支股票")
            print(stocks.head(5))
        else:
            print("获取股票列表返回为空")
    except Exception as e:
        print(f"获取股票列表失败: {e}")
    
    # 5. 获取股票日线数据测试-使用pro_bar
    print("\n测试获取股票日线数据(pro_bar)...")
    ts_code = '000001.SZ'  # 测试平安银行
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
    
    try:
        print(f"获取{ts_code}从{start_date}到{end_date}的日线数据(pro_bar)...")
        df = pro_api.pro_bar(ts_code=ts_code, adj='qfq', start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            print(f"成功获取日线数据(pro_bar): {len(df)}条记录")
            print(df.head())
        else:
            print("获取日线数据(pro_bar)返回为空")
    except Exception as e:
        print(f"获取日线数据(pro_bar)失败: {e}")
    
    # 6. 获取股票日线数据测试-使用daily
    print("\n测试获取股票日线数据(daily)...")
    try:
        print(f"获取{ts_code}从{start_date}到{end_date}的日线数据(daily)...")
        df = pro_api.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            print(f"成功获取日线数据(daily): {len(df)}条记录")
            print(df.head())
        else:
            print("获取日线数据(daily)返回为空")
    except Exception as e:
        print(f"获取日线数据(daily)失败: {e}")
    
    # 7. 测试旧版API (不建议使用，但测试兼容性)
    print("\n测试旧版API (仅测试)...")
    try:
        print(f"尝试使用旧版API获取{ts_code.split('.')[0]}的数据...")
        df_old = ts.get_k_data(ts_code.split('.')[0], start=start_date, end=end_date)
        if df_old is not None and not df_old.empty:
            print(f"成功获取旧版API数据: {len(df_old)}条记录")
            print(df_old.head())
        else:
            print("旧版API返回为空")
    except Exception as e:
        print(f"旧版API测试失败: {e}")
    
    print("\nTushare API测试完成")

if __name__ == "__main__":
    main() 