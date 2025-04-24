#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
涨停股票分析高级脚本
为高级Tushare VIP用户设计（15000+积分）
可查询2023年和2024年所有涨停股票
"""

import os
import sys
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
import tushare as ts
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("limit_up_analysis.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("LimitUpAnalysis")

class VipLimitUpFetcher:
    """VIP涨停股票数据获取和分析类"""
    
    def __init__(self, token=None, cache_dir='./cache/limit_up'):
        # 直接使用提供的Tushare token
        self.token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
        self.cache_dir = cache_dir
        self.results_dir = os.path.join(cache_dir, 'results')
        
        # VIP用户配置 - 高频调用
        self.api_rate_limit = 500  # 每分钟调用次数限制
        self.call_times = []
        
        # 创建缓存和结果目录
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 初始化tushare
        logger.info(f"初始化Tushare API (Token前5位: {self.token[:5]}...)")
        try:
            ts.set_token(self.token)
            self.pro = ts.pro_api()
            # 测试连接
            test_df = self.pro.stock_basic(exchange='', list_status='L', limit=5)
            if not test_df.empty:
                logger.info(f"Tushare API连接成功")
            else:
                logger.warning("API返回空数据，可能存在连接问题")
        except Exception as e:
            logger.error(f"Tushare初始化失败: {str(e)}")
            raise
    
    def _check_rate_limit(self):
        """VIP用户API调用频率控制"""
        now = time.time()
        # 保留最近一分钟内的调用时间
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        # 如果最近一分钟内的调用次数超过限制，则等待
        if len(self.call_times) >= self.api_rate_limit:
            wait_time = 60 - (now - self.call_times[0]) + 0.2
            if wait_time > 0:
                logger.warning(f"API调用频率限制，等待 {wait_time:.2f} 秒")
                time.sleep(wait_time)
        
        # 记录本次调用时间
        self.call_times.append(time.time())
    
    def api_call(self, func, max_retries=3, retry_delay=5, **kwargs):
        """封装API调用，包含重试和频率控制"""
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # 检查API调用频率
                self._check_rate_limit()
                
                # 调用API
                result = func(**kwargs)
                
                # 检查结果
                if isinstance(result, pd.DataFrame):
                    if result.empty and retry_count < max_retries:
                        retry_count += 1
                        logger.warning(f"API返回空DataFrame，重试 ({retry_count}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    return result
                else:
                    logger.warning(f"API返回了非DataFrame结果: {type(result)}")
                    return pd.DataFrame()
                
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    logger.warning(f"API调用失败，重试 ({retry_count}/{max_retries}): {str(e)}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"API调用达到最大重试次数，错误: {str(e)}")
                    return pd.DataFrame()
        
        return pd.DataFrame()
    
    def get_trade_calendar(self, year=None, start_date=None, end_date=None):
        """获取交易日历
        
        Args:
            year: 特定年份，如2023
            start_date: 开始日期，如'20230101'
            end_date: 结束日期，如'20231231'
        """
        if year is not None:
            start_date = f"{year}0101"
            end_date = f"{year}1231"
        
        if start_date is None or end_date is None:
            logger.error("必须提供年份或日期范围")
            return pd.DataFrame()
        
        cache_file = os.path.join(self.cache_dir, f"trade_cal_{start_date}_{end_date}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file):
            logger.info(f"从缓存读取交易日历")
            return pd.read_csv(cache_file)
        
        logger.info(f"获取交易日历: {start_date} 至 {end_date}")
        
        # 调用API
        trade_cal = self.api_call(
            self.pro.trade_cal,
            exchange='SSE',
            start_date=start_date,
            end_date=end_date,
            is_open='1'  # 只获取交易日
        )
        
        if not trade_cal.empty:
            # 保存到缓存
            trade_cal.to_csv(cache_file, index=False)
            logger.info(f"获取到 {len(trade_cal)} 个交易日")
            return trade_cal
        else:
            logger.error(f"获取交易日历失败")
            return pd.DataFrame()
    
    def get_limit_list(self, trade_date):
        """获取单个交易日的涨停板数据
        
        Args:
            trade_date: 交易日期，格式为'YYYYMMDD'
        """
        date_str = trade_date.replace('-', '')
        cache_file = os.path.join(self.cache_dir, f"limit_list_{date_str}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file):
            logger.info(f"从缓存读取 {trade_date} 涨停板数据")
            return pd.read_csv(cache_file)
        
        logger.info(f"获取 {trade_date} 涨停板数据")
        
        # 调用limit_list接口
        limit_data = self.api_call(
            self.pro.limit_list,
            trade_date=date_str,
            limit_type='U'  # U表示涨停
        )
        
        if not limit_data.empty:
            # 保存到缓存
            limit_data.to_csv(cache_file, index=False)
            logger.info(f"获取到 {len(limit_data)} 只涨停股票")
            
            # 同时保存到结果目录
            result_file = os.path.join(self.results_dir, f"limit_up_{date_str}.csv")
            limit_data.to_csv(result_file, index=False)
            
            return limit_data
        else:
            logger.warning(f"{trade_date} 未获取到涨停板数据")
            # 创建空文件作为标记
            pd.DataFrame().to_csv(cache_file, index=False)
            return pd.DataFrame()
    
    def get_monthly_limit_up(self, year, month):
        """获取特定月份的所有涨停股票
        
        Args:
            year: 年份，如2023
            month: 月份，1-12
        """
        logger.info(f"获取 {year}年{month}月 涨停股票数据")
        
        # 生成月份的起止日期
        if month < 10:
            start_date = f"{year}0{month}01"
        else:
            start_date = f"{year}{month}01"
        
        # 计算月末日期
        if month == 12:
            end_date = f"{year}1231"
        else:
            next_month = month + 1
            next_year = year
            if next_month > 12:
                next_month = 1
                next_year += 1
            
            if next_month < 10:
                end_date = f"{next_year}0{next_month}01"
            else:
                end_date = f"{next_year}{next_month}01"
            
            # 减去1天得到月末
            end_date_obj = datetime.strptime(end_date, '%Y%m%d') - timedelta(days=1)
            end_date = end_date_obj.strftime('%Y%m%d')
        
        # 获取月度交易日
        trade_cal = self.get_trade_calendar(start_date=start_date, end_date=end_date)
        if trade_cal.empty:
            logger.error(f"获取 {year}年{month}月 交易日历失败")
            return pd.DataFrame()
        
        # 按日期排序
        trade_dates = sorted(trade_cal['cal_date'].tolist())
        logger.info(f"{year}年{month}月共有 {len(trade_dates)} 个交易日")
        
        # 汇总文件
        summary_file = os.path.join(self.results_dir, f"limit_up_{year}_{month:02d}.csv")
        
        # 如果汇总文件已存在，直接返回
        if os.path.exists(summary_file):
            logger.info(f"从缓存读取 {year}年{month}月 涨停股票汇总")
            return pd.read_csv(summary_file)
        
        # 逐日获取涨停数据
        all_limit_up = []
        for i, date in enumerate(trade_dates):
            logger.info(f"处理 {date} ({i+1}/{len(trade_dates)})")
            
            # 获取当日涨停数据
            limit_data = self.get_limit_list(date)
            
            if not limit_data.empty:
                # 添加日期标记
                if 'trade_date' not in limit_data.columns:
                    limit_data['trade_date'] = date
                    
                all_limit_up.append(limit_data)
                logger.info(f"{date} 有 {len(limit_data)} 只涨停股票")
            else:
                logger.warning(f"{date} 没有涨停股票数据")
            
            # 每5个日期保存一次，避免数据丢失
            if (i + 1) % 5 == 0 or i == len(trade_dates) - 1:
                if all_limit_up:
                    # 合并数据
                    combined = pd.concat(all_limit_up, ignore_index=True)
                    # 保存阶段性结果
                    combined.to_csv(summary_file, index=False)
                    logger.info(f"已更新 {year}年{month}月 涨停股票汇总")
        
        # 最终合并
        if all_limit_up:
            result = pd.concat(all_limit_up, ignore_index=True)
            logger.info(f"{year}年{month}月 共有 {len(result)} 条涨停记录")
            return result
        else:
            logger.warning(f"{year}年{month}月 没有涨停数据")
            return pd.DataFrame()
    
    def get_yearly_limit_up(self, year):
        """获取特定年份的所有涨停股票
        
        Args:
            year: 年份，如2023
        """
        logger.info(f"获取 {year}年 涨停股票数据")
        
        # 汇总文件
        summary_file = os.path.join(self.results_dir, f"limit_up_{year}_summary.csv")
        
        # 如果汇总文件已存在，直接返回
        if os.path.exists(summary_file):
            logger.info(f"从缓存读取 {year}年 涨停股票汇总")
            return pd.read_csv(summary_file)
        
        # 按月处理
        all_months_data = []
        for month in range(1, 13):
            # 获取月度数据
            month_data = self.get_monthly_limit_up(year, month)
            
            if not month_data.empty:
                all_months_data.append(month_data)
                logger.info(f"{year}年{month}月 有 {len(month_data)} 条涨停记录")
            
            # 每处理一个月份暂停一下，避免API限制
            if month < 12:
                time.sleep(1)
        
        # 合并所有月份数据
        if all_months_data:
            result = pd.concat(all_months_data, ignore_index=True)
            result.to_csv(summary_file, index=False)
            logger.info(f"{year}年 共有 {len(result)} 条涨停记录")
            return result
        else:
            logger.warning(f"{year}年 没有涨停数据")
            return pd.DataFrame()
    
    def get_continuous_limit_up(self, year, min_days=2, max_days=5):
        """分析特定年份的连续涨停股票
        
        Args:
            year: 年份，如2023
            min_days: 最小连续天数，默认2
            max_days: 最大连续天数，默认5
        """
        logger.info(f"分析 {year}年 连续涨停股票 ({min_days}-{max_days}天)")
        
        # 先获取年度所有涨停数据
        yearly_data = self.get_yearly_limit_up(year)
        if yearly_data.empty:
            logger.error(f"获取 {year}年 涨停数据失败")
            return {}
        
        # 获取交易日历
        trade_cal = self.get_trade_calendar(year=year)
        if trade_cal.empty:
            logger.error(f"获取 {year}年 交易日历失败")
            return {}
        
        # 按日期排序
        all_trade_dates = sorted(trade_cal['cal_date'].tolist())
        
        # 统计每个股票在每个交易日是否涨停
        stock_daily_limit = {}
        for _, row in yearly_data.iterrows():
            ts_code = row['ts_code']
            trade_date = str(row['trade_date'])
            
            if ts_code not in stock_daily_limit:
                stock_daily_limit[ts_code] = {'name': row['name'], 'dates': []}
            
            if trade_date not in stock_daily_limit[ts_code]['dates']:
                stock_daily_limit[ts_code]['dates'].append(trade_date)
        
        # 分析连续涨停
        continuous_result = {}
        for days in range(min_days, max_days + 1):
            continuous_result[days] = []
        
        # 对每只股票分析连续涨停情况
        for ts_code, info in stock_daily_limit.items():
            # 按日期排序
            limit_dates = sorted(info['dates'])
            
            # 查找连续涨停
            for i in range(len(limit_dates)):
                max_continuous = 1
                current_date = limit_dates[i]
                
                # 向后检查连续日期
                for j in range(i + 1, len(limit_dates)):
                    # 检查是否连续交易日
                    next_date = limit_dates[j]
                    
                    # 获取两个日期在交易日历中的索引
                    try:
                        idx_current = all_trade_dates.index(current_date)
                        idx_next = all_trade_dates.index(next_date)
                        
                        # 如果是连续的交易日
                        if idx_next == idx_current + 1:
                            max_continuous += 1
                            current_date = next_date
                        else:
                            break
                    except ValueError:
                        # 日期不在交易日历中
                        break
                
                # 记录达到标准的连续涨停
                if max_continuous >= min_days:
                    for days in range(min_days, min(max_continuous + 1, max_days + 1)):
                        record = {
                            'ts_code': ts_code,
                            'name': info['name'],
                            'continuous_days': days,
                            'start_date': limit_dates[i],
                            'end_date': limit_dates[i + days - 1] if i + days - 1 < len(limit_dates) else limit_dates[-1]
                        }
                        continuous_result[days].append(record)
        
        # 将结果转换为DataFrame并保存
        for days, records in continuous_result.items():
            if records:
                df = pd.DataFrame(records)
                result_file = os.path.join(self.results_dir, f"continuous_limit_up_{year}_{days}days.csv")
                df.to_csv(result_file, index=False)
                logger.info(f"{year}年 连续{days}天涨停的股票有 {len(df)} 只")
            else:
                logger.warning(f"{year}年 没有连续{days}天涨停的股票")
        
        return continuous_result
    
    def analyze_limit_up_stats(self, year):
        """分析涨停统计数据
        
        Args:
            year: 年份，如2023
        """
        logger.info(f"分析 {year}年 涨停统计数据")
        
        # 获取年度涨停数据
        yearly_data = self.get_yearly_limit_up(year)
        if yearly_data.empty:
            logger.error(f"获取 {year}年 涨停数据失败")
            return None
        
        # 月度涨停数量统计
        monthly_count = yearly_data.groupby(yearly_data['trade_date'].astype(str).str[:6]).size()
        monthly_count.name = 'count'
        monthly_count = monthly_count.reset_index()
        monthly_count.columns = ['month', 'limit_up_count']
        
        # 股票涨停次数统计
        stock_count = yearly_data.groupby(['ts_code', 'name']).size()
        stock_count.name = 'limit_up_count'
        stock_count = stock_count.reset_index().sort_values('limit_up_count', ascending=False)
        
        # 行业涨停统计
        if 'industry' in yearly_data.columns:
            industry_count = yearly_data.groupby('industry').size()
            industry_count.name = 'limit_up_count'
            industry_count = industry_count.reset_index().sort_values('limit_up_count', ascending=False)
        else:
            # 获取股票行业信息
            stock_info = self.api_call(
                self.pro.stock_basic,
                exchange='',
                list_status='L',
                fields='ts_code,name,industry'
            )
            
            if not stock_info.empty:
                # 合并行业信息
                merged = pd.merge(yearly_data, stock_info[['ts_code', 'industry']], on='ts_code', how='left')
                
                # 行业统计
                industry_count = merged.groupby('industry').size()
                industry_count.name = 'limit_up_count'
                industry_count = industry_count.reset_index().sort_values('limit_up_count', ascending=False)
            else:
                industry_count = pd.DataFrame()
        
        # 保存统计结果
        monthly_count.to_csv(os.path.join(self.results_dir, f"monthly_limit_up_{year}.csv"), index=False)
        stock_count.to_csv(os.path.join(self.results_dir, f"stock_limit_up_count_{year}.csv"), index=False)
        
        if not industry_count.empty:
            industry_count.to_csv(os.path.join(self.results_dir, f"industry_limit_up_{year}.csv"), index=False)
        
        # 返回统计结果
        stats = {
            'monthly_count': monthly_count,
            'stock_count': stock_count,
            'industry_count': industry_count if not industry_count.empty else None
        }
        
        return stats
    
    def print_stats_summary(self, stats):
        """打印统计结果摘要
        
        Args:
            stats: 统计结果字典
        """
        if not stats:
            logger.warning("没有统计数据可供显示")
            return
        
        print("\n" + "="*50)
        print("涨停统计摘要")
        print("="*50)
        
        if 'monthly_count' in stats and not stats['monthly_count'].empty:
            print("\n月度涨停数量TOP5:")
            print(stats['monthly_count'].sort_values('limit_up_count', ascending=False).head(5))
        
        if 'stock_count' in stats and not stats['stock_count'].empty:
            print("\n股票涨停次数TOP10:")
            print(stats['stock_count'].head(10))
        
        if 'industry_count' in stats and stats['industry_count'] is not None:
            print("\n行业涨停次数TOP5:")
            print(stats['industry_count'].head(5))
        
        print("\n" + "="*50)

def get_parser():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='涨停股票分析工具')
    
    parser.add_argument('--token', type=str,
                        default="0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10",
                        help='Tushare API令牌')
    
    parser.add_argument('--year', type=int, default=2023,
                        help='分析年份 (默认: 2023)')
    
    parser.add_argument('--month', type=int, default=None,
                        help='分析月份 (可选)')
    
    parser.add_argument('--min-days', type=int, default=2,
                        help='连续涨停最小天数 (默认: 2)')
    
    parser.add_argument('--max-days', type=int, default=5,
                        help='连续涨停最大天数 (默认: 5)')
    
    parser.add_argument('--date', type=str, default=None,
                        help='特定日期，格式YYYYMMDD (可选)')
    
    parser.add_argument('--cache-dir', type=str, default='./cache/limit_up',
                        help='缓存目录 (默认: ./cache/limit_up)')
    
    return parser

def main():
    """主函数"""
    # 解析命令行参数
    parser = get_parser()
    args = parser.parse_args()
    
    try:
        # 初始化数据获取器
        fetcher = VipLimitUpFetcher(args.token, cache_dir=args.cache_dir)
        
        if args.date:
            # 获取特定日期的涨停数据
            logger.info(f"获取 {args.date} 的涨停数据")
            limit_data = fetcher.get_limit_list(args.date)
            
            if not limit_data.empty:
                print(f"\n{args.date} 涨停股票清单:")
                print(limit_data)
            else:
                print(f"\n{args.date} 未找到涨停股票")
        
        elif args.month:
            # 获取特定月份的涨停数据
            logger.info(f"获取 {args.year}年{args.month}月 的涨停数据")
            monthly_data = fetcher.get_monthly_limit_up(args.year, args.month)
            
            if not monthly_data.empty:
                print(f"\n{args.year}年{args.month}月 共有 {len(monthly_data)} 条涨停记录")
                # 按日期统计
                date_count = monthly_data.groupby('trade_date').size()
                print("\n日期统计:")
                print(date_count)
            else:
                print(f"\n{args.year}年{args.month}月 未找到涨停数据")
        
        else:
            # 获取全年涨停数据
            logger.info(f"分析 {args.year} 年涨停数据")
            
            # 1. 获取全年涨停统计
            stats = fetcher.analyze_limit_up_stats(args.year)
            if stats:
                fetcher.print_stats_summary(stats)
            
            # 2. 分析连续涨停
            continuous_result = fetcher.get_continuous_limit_up(
                args.year, min_days=args.min_days, max_days=args.max_days
            )
            
            # 打印连续涨停统计
            print("\n连续涨停统计:")
            for days in range(args.min_days, args.max_days + 1):
                if days in continuous_result and continuous_result[days]:
                    count = len(continuous_result[days])
                    print(f"连续{days}天涨停: {count}只股票")
                else:
                    print(f"连续{days}天涨停: 0只股票")
        
        logger.info("涨停数据分析完成!")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()