#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试不同日期的Tushare数据可用性
"""

import os
import sys
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
import tushare as ts
import random

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger()

def test_date_data(token, date, cache_dir='./cache/dates_test'):
    """测试特定日期的数据可用性"""
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    
    # 格式化日期
    date_str = date.replace('-', '')
    
    # 输出测试信息
    print(f"\n{'='*20} 测试日期: {date} {'='*20}")
    
    # 初始化tushare
    ts.set_token(token)
    pro = ts.pro_api()
    
    try:
        # 测试1：使用limit_list接口
        print(f"\n1. 使用limit_list接口测试 {date}")
        limit_cache_file = os.path.join(cache_dir, f"limit_list_{date_str}.csv")
        
        if os.path.exists(limit_cache_file):
            limit_up = pd.read_csv(limit_cache_file)
            print(f"从缓存读取 - limit_list接口返回 {len(limit_up)} 条记录")
        else:
            try:
                limit_up = pro.limit_list(trade_date=date_str, limit_type='U')
                if not isinstance(limit_up, pd.DataFrame) or limit_up.empty:
                    print(f"limit_list接口返回空数据")
                else:
                    limit_up.to_csv(limit_cache_file, index=False)
                    print(f"limit_list接口返回 {len(limit_up)} 条记录")
                    print(f"列: {limit_up.columns.tolist()}")
                    if len(limit_up) > 0:
                        print(f"样例数据: \n{limit_up.head(1)}")
            except Exception as e:
                print(f"limit_list接口调用失败: {e}")
        
        # 测试2：使用日线数据接口
        print(f"\n2. 使用指数日线数据测试交易日 {date}")
        index_cache_file = os.path.join(cache_dir, f"index_daily_{date_str}.csv")
        
        # 上证指数
        index_code = '000001.SH'
        
        if os.path.exists(index_cache_file):
            index_data = pd.read_csv(index_cache_file)
            print(f"从缓存读取 - 指数日线接口返回 {len(index_data)} 条记录")
        else:
            try:
                # 获取前后3天的数据，确保能覆盖到这个日期
                start_date = (datetime.strptime(date, '%Y%m%d') - timedelta(days=3)).strftime('%Y%m%d')
                end_date = (datetime.strptime(date, '%Y%m%d') + timedelta(days=3)).strftime('%Y%m%d')
                
                index_data = pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
                if not isinstance(index_data, pd.DataFrame) or index_data.empty:
                    print(f"指数日线接口返回空数据")
                else:
                    index_data.to_csv(index_cache_file, index=False)
                    print(f"指数日线接口返回 {len(index_data)} 条记录")
                    
                    # 检查是否包含指定日期
                    if date_str in index_data['trade_date'].values:
                        print(f"确认 {date} 是交易日")
                        day_data = index_data[index_data['trade_date'] == date_str]
                        print(f"当日指数数据: \n{day_data}")
                    else:
                        print(f"警告: {date} 可能不是交易日!")
                        print(f"可用交易日: {sorted(index_data['trade_date'].tolist())}")
            except Exception as e:
                print(f"指数日线接口调用失败: {e}")
        
        # 测试3：获取股票示例日线数据
        print(f"\n3. 使用单只股票数据测试 {date}")
        # 平安银行
        stock_code = '000001.SZ'
        stock_cache_file = os.path.join(cache_dir, f"stock_daily_{stock_code}_{date_str}.csv")
        
        if os.path.exists(stock_cache_file):
            stock_data = pd.read_csv(stock_cache_file)
            print(f"从缓存读取 - 股票日线接口返回 {len(stock_data)} 条记录")
        else:
            try:
                # 获取前后3天的数据
                start_date = (datetime.strptime(date, '%Y%m%d') - timedelta(days=3)).strftime('%Y%m%d')
                end_date = (datetime.strptime(date, '%Y%m%d') + timedelta(days=3)).strftime('%Y%m%d')
                
                stock_data = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
                if not isinstance(stock_data, pd.DataFrame) or stock_data.empty:
                    print(f"股票日线接口返回空数据")
                else:
                    stock_data.to_csv(stock_cache_file, index=False)
                    print(f"股票日线接口返回 {len(stock_data)} 条记录")
                    
                    # 检查是否包含指定日期
                    if date_str in stock_data['trade_date'].values:
                        print(f"确认 {date} 有股票交易数据")
                        day_data = stock_data[stock_data['trade_date'] == date_str]
                        print(f"当日股票数据: \n{day_data}")
                    else:
                        print(f"警告: {date} 没有该股票交易数据!")
                        print(f"可用交易日: {sorted(stock_data['trade_date'].tolist())}")
            except Exception as e:
                print(f"股票日线接口调用失败: {e}")
                
        print(f"\n{'='*50}")
        return True
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        return False

def main():
    """测试多个日期"""
    # 使用提供的token
    token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    
    # 测试日期范围（从2023年到2024年）
    test_dates = [
        # 2023年数据
        "20230605",  # 2023年6月
        "20230825",  # 2023年8月
        "20231016",  # 2023年10月
        "20231225",  # 2023年12月
        
        # 2024年数据
        "20240122",  # 2024年1月
        "20240226",  # 2024年2月
        "20240318",  # 2024年3月中旬
        "20240322",  # 3月22日 (之前测试用的日期)
        "20240325",  # 3月25日 (最近的数据)
    ]
    
    # 时间间隔，避免API调用频率限制
    sleep_between_dates = 30  # 秒
    
    # 循环测试每个日期
    for i, date in enumerate(test_dates):
        if i > 0:
            print(f"等待 {sleep_between_dates} 秒后继续测试...")
            time.sleep(sleep_between_dates)
            
        test_date_data(token, date)

if __name__ == "__main__":
    main() 