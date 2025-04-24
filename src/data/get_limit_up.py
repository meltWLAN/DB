#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版涨停股票获取脚本，直接调用tushare API并使用缓存
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
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

class SimpleStockFetcher:
    """简化版股票数据获取类"""
    
    def __init__(self, token, cache_dir='./cache'):
        self.token = token
        self.cache_dir = cache_dir
        self.api_rate_limit = 40  # 每分钟最大请求次数
        self.last_call_time = 0
        self.call_times = []
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化tushare
        ts.set_token(token)
        self.pro = ts.pro_api()
        logger.info(f"初始化完成，token前5位: {token[:5]}...")
    
    def _check_rate_limit(self):
        """检查并控制API调用频率"""
        now = time.time()
        # 保留最近一分钟内的调用时间
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        # 如果最近一分钟内的调用次数超过限制，则等待
        if len(self.call_times) >= self.api_rate_limit:
            wait_time = 60 - (now - self.call_times[0]) + 0.5  # 额外等待0.5秒
            if wait_time > 0:
                logger.info(f"API调用频率限制，等待 {wait_time:.2f} 秒")
                time.sleep(wait_time)
        
        # 记录本次调用时间
        self.call_times.append(time.time())
        
        # 随机延迟，避免请求过于集中
        time.sleep(random.uniform(0.2, 0.5))
    
    def api_call(self, func, max_retries=3, retry_delay=5, **kwargs):
        """封装API调用，包含重试和频率控制"""
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 检查API调用频率
                self._check_rate_limit()
                
                # 调用API
                result = func(**kwargs)
                
                # 如果返回空DataFrame，视为失败
                if isinstance(result, pd.DataFrame) and result.empty:
                    logger.warning("API返回空DataFrame")
                    retry_count += 1
                    time.sleep(retry_delay)
                    continue
                
                return result
            
            except Exception as e:
                retry_count += 1
                logger.warning(f"API调用失败 ({retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    # 指数退避策略
                    wait_time = retry_delay * (2 ** (retry_count - 1))
                    logger.info(f"等待 {wait_time} 秒后重试")
                    time.sleep(wait_time)
                else:
                    logger.error(f"达到最大重试次数，API调用失败")
                    return None
    
    def get_stock_list(self):
        """获取股票列表"""
        cache_file = os.path.join(self.cache_dir, 'stock_list.csv')
        
        # 检查缓存
        if os.path.exists(cache_file):
            # 如果缓存文件不超过1天，使用缓存
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_time).days < 1:
                logger.info("从缓存读取股票列表")
                return pd.read_csv(cache_file)
        
        logger.info("从API获取股票列表")
        stocks = self.api_call(
            self.pro.stock_basic,
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,market,list_date'
        )
        
        if stocks is not None and not stocks.empty:
            # 保存到缓存
            stocks.to_csv(cache_file, index=False)
            logger.info(f"获取到 {len(stocks)} 只股票")
            return stocks
        else:
            logger.error("获取股票列表失败")
            return pd.DataFrame()
    
    def get_daily_data(self, ts_code, start_date, end_date):
        """获取日线数据"""
        # 格式化日期
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        
        # 缓存文件名
        cache_file = os.path.join(self.cache_dir, f"daily_{ts_code}_{start_date}_{end_date}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file):
            # 如果缓存文件不超过1天，使用缓存
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_time).days < 1:
                logger.debug(f"从缓存读取 {ts_code} 日线数据")
                return pd.read_csv(cache_file)
        
        logger.debug(f"从API获取 {ts_code} 日线数据")
        daily = self.api_call(
            self.pro.daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if daily is not None and not daily.empty:
            # 保存到缓存
            daily.to_csv(cache_file, index=False)
            logger.debug(f"获取到 {len(daily)} 条日线数据")
            return daily
        else:
            logger.warning(f"获取 {ts_code} 日线数据失败")
            return pd.DataFrame()
    
    def find_limit_up_stocks(self, date, sample_size=None):
        """查找指定日期的涨停股票"""
        logger.info(f"查找 {date} 的涨停股票")
        date = date.replace('-', '')
        
        # 缓存文件名
        cache_file = os.path.join(self.cache_dir, f"limit_up_{date}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file):
            # 如果缓存文件不超过1天，使用缓存
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_time).days < 1:
                logger.info(f"从缓存读取 {date} 涨停股票")
                limit_up = pd.read_csv(cache_file)
                logger.info(f"找到 {len(limit_up)} 只涨停股票")
                return limit_up
        
        # 方法1: 使用limit_list接口
        try:
            logger.info("尝试使用limit_list接口")
            limit_up = self.api_call(
                self.pro.limit_list,
                trade_date=date,
                limit_type='U'
            )
            
            if limit_up is not None and not limit_up.empty:
                limit_up.to_csv(cache_file, index=False)
                logger.info(f"使用limit_list接口找到 {len(limit_up)} 只涨停股票")
                return limit_up
            else:
                logger.warning("limit_list接口未返回数据，尝试方法2")
        except Exception as e:
            logger.warning(f"limit_list接口调用失败: {e}")
        
        # 方法2: 获取股票列表，然后检查每只股票的涨幅
        logger.info("使用自定义方法查找涨停股票")
        stocks = self.get_stock_list()
        if stocks.empty:
            logger.error("获取股票列表失败，无法查找涨停股票")
            return pd.DataFrame()
        
        # 如果指定了样本大小，随机抽样
        if sample_size and sample_size < len(stocks):
            logger.info(f"随机抽样 {sample_size} 只股票")
            stocks = stocks.sample(sample_size)
        
        # 获取前一天日期，用于计算涨停
        dt = datetime.strptime(date, '%Y%m%d')
        prev_date = (dt - timedelta(days=7)).strftime('%Y%m%d')  # 往前取7天，确保能获取到前一个交易日
        
        limit_up_stocks = []
        for i, stock in stocks.iterrows():
            ts_code = stock['ts_code']
            
            try:
                # 获取日线数据
                daily = self.get_daily_data(ts_code, prev_date, date)
                if daily.empty:
                    continue
                
                # 按日期降序排序
                daily = daily.sort_values('trade_date', ascending=False)
                
                # 只关注最新一天
                if len(daily) > 0 and daily.iloc[0]['trade_date'] == date:
                    last_day = daily.iloc[0]
                    
                    # 根据股票类型判断涨停标准
                    is_tech_stock = ts_code.startswith('300') or ts_code.startswith('688')
                    threshold = 19.5 if is_tech_stock else 9.5
                    
                    # 判断是否涨停
                    if last_day['pct_chg'] >= threshold:
                        limit_up_stocks.append({
                            'ts_code': ts_code,
                            'name': stock['name'],
                            'industry': stock.get('industry', ''),
                            'close': last_day['close'],
                            'pct_chg': last_day['pct_chg'],
                            'trade_date': date,
                            'is_tech_stock': is_tech_stock
                        })
                        logger.info(f"发现涨停股票: {ts_code} - {stock['name']}, 涨幅: {last_day['pct_chg']:.2f}%")
            
            except Exception as e:
                logger.debug(f"处理股票 {ts_code} 时出错: {e}")
        
        # 转换为DataFrame
        result = pd.DataFrame(limit_up_stocks)
        
        # 保存到缓存
        if not result.empty:
            result.to_csv(cache_file, index=False)
        
        logger.info(f"找到 {len(result)} 只涨停股票")
        return result
    
    def find_continuous_limit_up(self, days=2, end_date=None, sample_size=None):
        """查找连续涨停股票"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        end_date = end_date.replace('-', '')
        
        logger.info(f"查找截至 {end_date} 连续 {days} 天涨停的股票")
        
        # 缓存文件名
        cache_file = os.path.join(self.cache_dir, f"continuous_limit_up_{days}d_{end_date}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file):
            # 如果缓存文件不超过1天，使用缓存
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_time).days < 1:
                logger.info(f"从缓存读取连续涨停股票")
                result = pd.read_csv(cache_file)
                logger.info(f"找到 {len(result)} 只连续涨停股票")
                return result
        
        # 获取结束日期的涨停股票
        last_day_limit_up = self.find_limit_up_stocks(end_date, sample_size)
        if last_day_limit_up.empty:
            logger.warning(f"{end_date} 没有涨停股票")
            return pd.DataFrame()
        
        # 如果只需要1天涨停，直接返回
        if days == 1:
            last_day_limit_up.to_csv(cache_file, index=False)
            return last_day_limit_up
        
        # 计算开始日期 (往前多取几天，避免中间有非交易日)
        start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=days*3)).strftime('%Y%m%d')
        
        # 查找连续涨停
        continuous_limit_up = []
        for i, stock in last_day_limit_up.iterrows():
            ts_code = stock['ts_code']
            
            try:
                # 获取股票历史数据
                hist_data = self.get_daily_data(ts_code, start_date, end_date)
                if hist_data.empty or len(hist_data) < days:
                    continue
                
                # 按日期降序排序
                hist_data = hist_data.sort_values('trade_date', ascending=False)
                
                # 根据股票类型判断涨停标准
                is_tech_stock = ts_code.startswith('300') or ts_code.startswith('688')
                threshold = 19.5 if is_tech_stock else 9.5
                
                # 检查连续涨停
                is_continuous = True
                limit_dates = []
                
                for i in range(min(days, len(hist_data))):
                    row = hist_data.iloc[i]
                    if row['pct_chg'] >= threshold:
                        limit_dates.append(row['trade_date'])
                    else:
                        is_continuous = False
                        break
                
                if is_continuous and len(limit_dates) >= days:
                    logger.info(f"发现连续 {days} 天涨停股票: {ts_code}")
                    continuous_limit_up.append({
                        'ts_code': ts_code,
                        'name': stock['name'],
                        'industry': stock.get('industry', ''),
                        'close': hist_data.iloc[0]['close'],
                        'pct_chg': hist_data.iloc[0]['pct_chg'],
                        'limit_dates': ','.join(limit_dates[:days]),
                        'continuous_days': len(limit_dates)
                    })
            
            except Exception as e:
                logger.warning(f"处理股票 {ts_code} 时出错: {e}")
        
        # 转换为DataFrame
        result = pd.DataFrame(continuous_limit_up)
        
        # 保存到缓存
        if not result.empty:
            result.to_csv(cache_file, index=False)
        
        logger.info(f"找到 {len(result)} 只连续 {days} 天涨停的股票")
        return result

def main():
    """主函数"""
    # 使用指定token或从配置读取
    token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    fetcher = SimpleStockFetcher(token, cache_dir='./cache/tushare')
    
    # 设置时间范围
    end_date = "20240322"  # 使用3月22日
    
    # 查找单日涨停股票
    logger.info("\n1. 查找单日涨停股票")
    limit_up = fetcher.find_limit_up_stocks(end_date, sample_size=100)
    if not limit_up.empty:
        print(f"找到 {len(limit_up)} 只涨停股票 (前5只):")
        print(limit_up.head())
    else:
        print("没有找到涨停股票")
    
    # 查找连续涨停股票
    for days in [2, 3]:
        logger.info(f"\n{days}. 查找连续{days}天涨停股票")
        continuous = fetcher.find_continuous_limit_up(days, end_date, sample_size=100)
        if not continuous.empty:
            print(f"找到 {len(continuous)} 只连续{days}天涨停股票:")
            print(continuous)
        else:
            print(f"没有找到连续{days}天涨停股票")

if __name__ == "__main__":
    main() 