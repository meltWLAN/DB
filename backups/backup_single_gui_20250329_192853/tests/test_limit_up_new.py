#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd
from datetime import datetime, timedelta
from src.data.tushare_fetcher import TushareFetcher

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_limit_up_stocks():
    """测试获取连续涨停股票的功能"""
    print("=" * 60)
    print(" 连续涨停股票获取测试 ".center(60, "="))
    print("=" * 60)
    
    # 创建Tushare获取器
    fetcher = TushareFetcher()
    
    # 设置日期范围
    today = datetime.now().strftime('%Y-%m-%d')
    thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    ninety_days_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # 测试不同的日期范围
    date_ranges = [
        {'start': thirty_days_ago, 'end': today, 'desc': '最近30天'},
        {'start': ninety_days_ago, 'end': today, 'desc': '最近90天'},
    ]
    
    # 测试不同连续天数
    for days in [1, 2, 3]:
        print(f"\n测试 连续{days}天涨停 的股票获取:")
        print("-" * 40)
        
        for date_range in date_ranges:
            print(f"\n日期范围: {date_range['desc']} ({date_range['start']} 至 {date_range['end']})")
            
            # 获取连续涨停股票
            try:
                limit_up_stocks = fetcher.get_continuous_limit_up_stocks(days=days, end_date=date_range['end'])
                
                if limit_up_stocks is not None and not limit_up_stocks.empty:
                    print(f"成功获取到 {len(limit_up_stocks)} 只连续{days}天涨停股票")
                    print(f"数据列: {limit_up_stocks.columns.tolist()}")
                    print(f"示例数据:")
                    print(limit_up_stocks.head(3).to_string())
                else:
                    print(f"未找到连续{days}天涨停的股票")
            except Exception as e:
                print(f"获取连续{days}天涨停股票时出错: {e}")
    
    print("\n" + "=" * 60)
    print(" 测试完成 ".center(60, "="))
    print("=" * 60)

if __name__ == "__main__":
    test_limit_up_stocks() 