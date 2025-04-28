#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tushare API 测试脚本
验证Tushare连接和数据获取功能
"""

import os
import sys
import pandas as pd
import numpy as np
import tushare as ts
import datetime
import time

# 设置显示选项
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 120)

def test_tushare_functionality():
    print("==================================================")
    print("Tushare API 连接测试")
    print("==================================================")

    # 显示Tushare版本
    print(f"Tushare版本: {ts.__version__}")

    # 设置日期范围（最近10天）
    today = datetime.datetime.now()
    end_date = today.strftime('%Y%m%d')
    start_date = (today - datetime.timedelta(days=10)).strftime('%Y%m%d')
    print(f"测试日期范围: {start_date} 至 {end_date}")

    # 设置token
    token = '0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10'
    print(f"使用token: {token}")

    # 初始化Pro API
    print("\n测试1: 初始化Tushare Pro API")
    ts.set_token(token)
    try:
        pro = ts.pro_api()
        print("Tushare Pro API 初始化成功")
    except Exception as e:
        print(f"Tushare Pro API 初始化失败: {e}")
        sys.exit(1)

    # 测试交易日历获取
    print("\n测试2: 获取交易日历")
    try:
        df_cal = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
        if df_cal is not None and not df_cal.empty:
            print(f"交易日历获取成功, 共获取 {len(df_cal)} 条记录")
            print(df_cal.head())
        else:
            print("交易日历获取结果为空")
    except Exception as e:
        print(f"交易日历获取失败: {e}")

    # 测试股票列表获取
    print("\n测试3: 获取股票列表")
    try:
        df_stock_list = pro.stock_basic(exchange='', list_status='L')
        if df_stock_list is not None and not df_stock_list.empty:
            print(f"股票列表获取成功, 共获取 {len(df_stock_list)} 条记录")
            print(df_stock_list.head())
        else:
            print("股票列表获取结果为空")
    except Exception as e:
        print(f"股票列表获取失败: {e}")

    # 测试使用pro_bar获取股票日线数据
    print("\n测试4: 使用pro_bar获取股票日线数据 (000001.SZ)")
    try:
        df_daily_probar = ts.pro_bar(ts_code='000001.SZ', adj='qfq', start_date=start_date, end_date=end_date)
        if df_daily_probar is not None and not df_daily_probar.empty:
            print(f"股票日线数据(pro_bar)获取成功, 共获取 {len(df_daily_probar)} 条记录")
            print(df_daily_probar.head())
        else:
            print("股票日线数据(pro_bar)获取结果为空")
    except Exception as e:
        print(f"股票日线数据(pro_bar)获取失败: {e}")

    # 测试使用daily接口获取股票日线数据
    print("\n测试5: 使用daily接口获取股票日线数据 (000001.SZ)")
    try:
        df_daily = pro.daily(ts_code='000001.SZ', start_date=start_date, end_date=end_date)
        if df_daily is not None and not df_daily.empty:
            print(f"股票日线数据(daily)获取成功, 共获取 {len(df_daily)} 条记录")
            print(df_daily.head())
        else:
            print("股票日线数据(daily)获取结果为空")
    except Exception as e:
        print(f"股票日线数据(daily)获取失败: {e}")

    # 测试使用老版本API获取历史数据
    print("\n测试6: 使用get_hist_data接口获取历史数据 (000001)")
    try:
        df_hist = ts.get_hist_data('000001', start=start_date[:4] + '-' + start_date[4:6] + '-' + start_date[6:], 
                                  end=end_date[:4] + '-' + end_date[4:6] + '-' + end_date[6:])
        if df_hist is not None and not df_hist.empty:
            print(f"历史数据(get_hist_data)获取成功, 共获取 {len(df_hist)} 条记录")
            print(df_hist.head())
        else:
            print("历史数据(get_hist_data)获取结果为空")
    except Exception as e:
        print(f"历史数据(get_hist_data)获取失败: {e}")

    print("\n"+"="*50)
    print("测试完成")
    print("="*50)

if __name__ == "__main__":
    test_tushare_functionality() 