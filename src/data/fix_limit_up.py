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
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger()

def fix_limit_up_detection():
    """直接实现涨停检测功能"""
    print("=" * 60)
    print("=" * 20 + " 涨停检测测试 " + "=" * 20)
    print("=" * 60)
    
    # Tushare配置
    token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    print(f"Tushare Token: {token[:5]}...{token[-5:]} (长度: {len(token)})")
    
    # 设置token
    ts.set_token(token)
    pro = ts.pro_api()
    
    # 设置日期范围 - 修改为使用历史数据
    end_date = "20240322"  # A股3月22日的数据
    start_date = "20240220"  # 获取一个月左右的数据
    print(f"测试日期范围: {start_date} 至 {end_date}")
    
    # 获取股票列表
    stock_list = pro.stock_basic(exchange='', list_status='L')
    print(f"获取到 {len(stock_list)} 只股票")
    
    # 不再使用limit_list接口，直接用自定义逻辑查找涨停股
    print("\n1. 使用自定义逻辑查找涨停股票")
    
    try:
        # 随机选取20支股票进行测试
        sample_size = 20
        sample_stocks = stock_list.sample(min(sample_size, len(stock_list)))
        print(f"随机选取 {len(sample_stocks)} 支股票进行测试")
        
        # 涨停股票列表
        limit_up_stocks = []
        
        for idx, stock in sample_stocks.iterrows():
            ts_code = stock['ts_code']
            name = stock['name']
            
            # 获取历史数据
            hist_data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if hist_data is None or hist_data.empty:
                print(f"股票 {ts_code} ({name}) 没有历史数据")
                continue
                
            # 按日期降序排序
            hist_data = hist_data.sort_values('trade_date', ascending=False)
            
            # 根据股票类型确定涨停幅度：
            # - 主板、中小板、科创板、创业板（涨幅限制20%）: ST和*ST股票
            # - 主板、中小板（涨幅限制10%）: 非ST股票
            # - 科创板、创业板（涨幅限制20%）: 新股上市后前5个交易日不设涨幅限制
            
            # 简化判断：
            # - 创业板(300开头)、科创板(688开头): 阈值19.5%
            # - 其他股票: 阈值9.5%
            is_tech_stock = ts_code.startswith('300') or ts_code.startswith('688')
            threshold = 19.5 if is_tech_stock else 9.5
            
            # 添加涨停判断
            hist_data['is_limit_up'] = hist_data['pct_chg'] >= threshold
            
            # 检查是否有涨停
            if hist_data['is_limit_up'].any():
                limit_days = hist_data[hist_data['is_limit_up']]
                if not limit_days.empty:
                    latest_limit_day = limit_days.iloc[0]
                    print(f"股票 {ts_code} ({name}) 在 {latest_limit_day['trade_date']} 涨停，涨幅 {latest_limit_day['pct_chg']:.2f}%")
                    
                    # 添加到涨停股票列表
                    limit_up_stocks.append({
                        'ts_code': ts_code,
                        'name': name,
                        'trade_date': latest_limit_day['trade_date'],
                        'close': latest_limit_day['close'],
                        'pct_chg': latest_limit_day['pct_chg'],
                        'threshold': threshold
                    })
                    
                    # 检查连续涨停
                    consecutive_days = 0
                    consecutive_dates = []
                    
                    for i, row in hist_data.iterrows():
                        if row['is_limit_up']:
                            consecutive_days += 1
                            consecutive_dates.append(row['trade_date'])
                        else:
                            break
                    
                    if consecutive_days > 1:
                        print(f"   连续涨停 {consecutive_days} 天! 日期: {', '.join(consecutive_dates)}")
        
        print(f"\n找到 {len(limit_up_stocks)} 只涨停股票")
        if limit_up_stocks:
            limit_up_df = pd.DataFrame(limit_up_stocks)
            print("\n涨停股票列表:")
            print(limit_up_df)
            
            # 示例股票用于测试连续涨停
            if not limit_up_df.empty:
                example_stock = limit_up_df.iloc[0]
                ts_code = example_stock['ts_code']
                print(f"\n2. 使用 {ts_code} 测试连续涨停检测")
                
                # 获取该股票的历史数据
                hist_data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                
                if not hist_data.empty:
                    # 按日期降序排序
                    hist_data = hist_data.sort_values('trade_date', ascending=False)
                    
                    # 根据股票类型确定涨停幅度
                    is_tech_stock = ts_code.startswith('300') or ts_code.startswith('688')
                    threshold = 19.5 if is_tech_stock else 9.5
                    
                    # 添加涨停标志
                    hist_data['is_limit_up'] = hist_data['pct_chg'] >= threshold
                    
                    print(f"历史数据 (最近10天):")
                    print(hist_data[['trade_date', 'open', 'close', 'pct_chg', 'is_limit_up']].head(10))
                    
                    # 检查连续涨停
                    consecutive_days = 0
                    consecutive_dates = []
                    
                    for i, row in hist_data.iterrows():
                        if row['is_limit_up']:
                            consecutive_days += 1
                            consecutive_dates.append(row['trade_date'])
                        else:
                            break
                    
                    print(f"\n股票 {ts_code} 连续涨停 {consecutive_days} 天")
                    if consecutive_days > 0:
                        print(f"涨停日期: {', '.join(consecutive_dates)}")
        else:
            print("未找到任何涨停股票，增加样本量或更改日期范围可能会有所帮助")
    
    except Exception as e:
        print(f"测试过程中出错: {e}")
    
    print("\n3. 实现连续涨停检测功能")
    
    def get_continuous_limit_up_stocks(days=1, end_date=None):
        """获取连续涨停股票
        
        Args:
            days: 连续涨停天数
            end_date: 结束日期，默认为最新交易日
            
        Returns:
            DataFrame: 连续涨停股票数据
        """
        if end_date is None:
            end_date = "20240322"  # 默认使用3月22日
            
        print(f"获取连续{days}天涨停的股票, 截至 {end_date}")
        
        try:
            # 计算开始日期（往前多取几天的数据）
            start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=days*3)).strftime('%Y%m%d')
            
            # 获取所有股票列表
            all_stocks = pro.stock_basic(exchange='', list_status='L')
            if all_stocks is None or all_stocks.empty:
                print("获取股票列表失败")
                return pd.DataFrame()
                
            print(f"分析 {len(all_stocks)} 只股票...")
            
            # 为了效率，随机抽取100支股票进行测试
            sample_size = min(100, len(all_stocks))
            sample_stocks = all_stocks.sample(sample_size)
            print(f"抽样测试 {sample_size} 只股票")
            
            # 连续涨停股票列表
            continuous_limit_up_stocks = []
            
            # 遍历股票列表
            for idx, stock in sample_stocks.iterrows():
                ts_code = stock['ts_code']
                name = stock['name']
                
                # 获取历史数据
                hist_data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if hist_data is None or hist_data.empty or len(hist_data) < days:
                    continue
                
                # 按日期降序排序
                hist_data = hist_data.sort_values('trade_date', ascending=False)
                
                # 根据股票类型确定涨停幅度
                is_tech_stock = ts_code.startswith('300') or ts_code.startswith('688')
                threshold = 19.5 if is_tech_stock else 9.5
                
                # 添加涨停标志
                hist_data['is_limit_up'] = hist_data['pct_chg'] >= threshold
                
                # 检查连续涨停
                is_continuous = True
                limit_dates = []
                
                for i in range(days):
                    if i < len(hist_data):
                        if not hist_data.iloc[i]['is_limit_up']:
                            is_continuous = False
                            break
                        limit_dates.append(hist_data.iloc[i]['trade_date'])
                    else:
                        is_continuous = False
                        break
                
                if is_continuous:
                    # 发现连续涨停股票
                    print(f"发现连续{days}天涨停股票: {ts_code} - {name}")
                    continuous_limit_up_stocks.append({
                        'ts_code': ts_code,
                        'name': name,
                        'close': hist_data.iloc[0]['close'],
                        'pct_chg': hist_data.iloc[0]['pct_chg'],
                        'threshold': threshold,
                        'industry': stock['industry'] if 'industry' in stock else '',
                        'limit_dates': ','.join(limit_dates)
                    })
            
            # 转换为DataFrame
            result_df = pd.DataFrame(continuous_limit_up_stocks)
            
            print(f"找到 {len(result_df)} 只连续{days}天涨停的股票")
            return result_df
                
        except Exception as e:
            print(f"获取连续涨停股票失败: {e}")
            return pd.DataFrame()
    
    # 测试连续涨停函数
    for days in [1, 2, 3]:
        print(f"\n测试获取连续{days}天涨停的股票")
        result = get_continuous_limit_up_stocks(days)
        if not result.empty:
            print(f"成功获取到 {len(result)} 只连续{days}天涨停的股票")
            print(result.head())
        else:
            print(f"没有找到连续{days}天涨停的股票")
    
    print("\n测试完成")
    print("=" * 60)

if __name__ == "__main__":
    fix_limit_up_detection() 