#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
涨停板交易策略模块
基于涨停板数据生成买入推荐
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tushare as ts
import argparse
from final_limit_up import VipLimitUpFetcher

logger = logging.getLogger("TradingStrategy")

class LimitUpStrategy:
    """涨停板交易策略"""
    
    def __init__(self, token, cache_dir='./cache/trading'):
        """初始化
        
        Args:
            token: Tushare API Token
            cache_dir: 缓存目录
        """
        self.token = token
        self.cache_dir = cache_dir
        self.results_dir = os.path.join(cache_dir, 'results')
        
        # 创建缓存和结果目录
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 初始化数据获取器
        self.fetcher = VipLimitUpFetcher(token, cache_dir=os.path.join(cache_dir, 'limit_up'))
        
        # 初始化tushare直接接口
        ts.set_token(token)
        self.pro = ts.pro_api()
    
    def get_trading_date(self, benchmark_date=None, offset=0):
        """获取交易日期
        
        Args:
            benchmark_date: 基准日期，默认为当天
            offset: 日期偏移量，-1表示前一个交易日，1表示后一个交易日
        
        Returns:
            交易日期字符串，格式为YYYYMMDD
        """
        if benchmark_date is None:
            # 默认使用当天日期
            benchmark_date = datetime.now().strftime('%Y%m%d')
        else:
            # 格式化日期
            benchmark_date = benchmark_date.replace('-', '')
        
        # 获取日期前后的交易日历
        start_date = (datetime.strptime(benchmark_date, '%Y%m%d') - timedelta(days=20)).strftime('%Y%m%d')
        end_date = (datetime.strptime(benchmark_date, '%Y%m%d') + timedelta(days=20)).strftime('%Y%m%d')
        
        trade_cal = self.fetcher.api_call(
            self.pro.trade_cal,
            exchange='SSE',
            start_date=start_date,
            end_date=end_date,
            is_open='1'  # 只获取交易日
        )
        
        if trade_cal.empty:
            logger.error(f"获取交易日历失败")
            return None
        
        # 转换为日期列表
        dates = sorted(trade_cal['cal_date'].tolist())
        
        # 找到基准日期在列表中的位置
        try:
            idx = dates.index(benchmark_date)
        except ValueError:
            # 基准日期不是交易日，找到最近的交易日
            for i, date in enumerate(dates):
                if date > benchmark_date:
                    if offset >= 0:
                        idx = i
                    else:
                        idx = i - 1
                    break
            else:
                # 没找到合适的日期
                idx = len(dates) - 1
        
        # 计算目标日期的索引
        target_idx = idx + offset
        
        # 确保索引在有效范围内
        if target_idx < 0:
            target_idx = 0
        elif target_idx >= len(dates):
            target_idx = len(dates) - 1
        
        return dates[target_idx]
    
    def find_first_limit_up(self, start_date, end_date=None):
        """找出首次涨停的股票
        
        Args:
            start_date: 开始日期
            end_date: 结束日期，默认与开始日期相同
        
        Returns:
            首次涨停的股票列表
        """
        if end_date is None:
            end_date = start_date
        
        # 确保日期格式正确
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        
        logger.info(f"查找 {start_date} 至 {end_date} 首次涨停的股票")
        
        # 缓存文件
        cache_file = os.path.join(self.cache_dir, f"first_limit_up_{start_date}_{end_date}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file):
            logger.info("从缓存读取首次涨停股票")
            return pd.read_csv(cache_file)
        
        # 获取日期范围内的所有涨停股票
        if start_date[:6] == end_date[:6]:
            # 如果在同一个月内，直接获取月度数据
            year = int(start_date[:4])
            month = int(start_date[4:6])
            limit_up_data = self.fetcher.get_monthly_limit_up(year, month)
            
            # 筛选日期范围
            limit_up_data = limit_up_data[
                (limit_up_data['trade_date'].astype(str) >= start_date) & 
                (limit_up_data['trade_date'].astype(str) <= end_date)
            ]
        else:
            # 跨月，获取每一天的数据
            all_dates = []
            current_date = datetime.strptime(start_date, '%Y%m%d')
            end_date_obj = datetime.strptime(end_date, '%Y%m%d')
            
            while current_date <= end_date_obj:
                all_dates.append(current_date.strftime('%Y%m%d'))
                current_date += timedelta(days=1)
            
            # 获取每一天的涨停数据
            all_limit_up = []
            for date in all_dates:
                limit_data = self.fetcher.get_limit_list(date)
                if not limit_data.empty:
                    all_limit_up.append(limit_data)
            
            # 合并数据
            if all_limit_up:
                limit_up_data = pd.concat(all_limit_up, ignore_index=True)
            else:
                logger.warning(f"{start_date} 至 {end_date} 没有涨停数据")
                return pd.DataFrame()
        
        # 如果没有获取到数据
        if limit_up_data.empty:
            logger.warning(f"{start_date} 至 {end_date} 没有涨停数据")
            return pd.DataFrame()
        
        # 获取每只股票的历史涨停情况
        unique_stocks = limit_up_data['ts_code'].unique()
        logger.info(f"需要分析 {len(unique_stocks)} 只股票的历史涨停情况")
        
        # 获取前3个月的数据作为参考
        first_date_obj = datetime.strptime(start_date, '%Y%m%d')
        history_start = (first_date_obj - timedelta(days=90)).strftime('%Y%m%d')
        
        # 获取历史涨停数据
        history_end = (first_date_obj - timedelta(days=1)).strftime('%Y%m%d')
        
        # 历史涨停股票缓存
        history_cache_file = os.path.join(self.cache_dir, f"history_limit_up_{history_start}_{history_end}.csv")
        
        if os.path.exists(history_cache_file):
            history_limit_up = pd.read_csv(history_cache_file)
        else:
            # 获取历史每月数据并合并
            all_history = []
            current_date = datetime.strptime(history_start, '%Y%m%d')
            while current_date <= datetime.strptime(history_end, '%Y%m%d'):
                year = current_date.year
                month = current_date.month
                
                month_data = self.fetcher.get_monthly_limit_up(year, month)
                if not month_data.empty:
                    all_history.append(month_data)
                
                # 下一个月
                if month == 12:
                    current_date = datetime(year + 1, 1, 1)
                else:
                    current_date = datetime(year, month + 1, 1)
            
            # 合并历史数据
            if all_history:
                history_limit_up = pd.concat(all_history, ignore_index=True)
                history_limit_up.to_csv(history_cache_file, index=False)
            else:
                history_limit_up = pd.DataFrame()
        
        # 找出首次涨停的股票
        first_limit_up = []
        
        for stock in unique_stocks:
            # 当前日期范围内该股票的涨停记录
            stock_limit_up = limit_up_data[limit_up_data['ts_code'] == stock]
            
            # 该股票最早的涨停日期
            if 'trade_date' in stock_limit_up.columns:
                earliest_date = stock_limit_up['trade_date'].astype(str).min()
            else:
                # 如果没有trade_date列，跳过
                continue
            
            # 检查该股票在历史上是否有涨停记录
            if not history_limit_up.empty and 'ts_code' in history_limit_up.columns:
                history_stock = history_limit_up[history_limit_up['ts_code'] == stock]
                
                if history_stock.empty:
                    # 历史上没有涨停记录，是首次涨停
                    # 获取该股票在当前范围内最早那天的涨停数据
                    first_record = stock_limit_up[stock_limit_up['trade_date'].astype(str) == earliest_date]
                    first_limit_up.append(first_record.iloc[0].to_dict())
            else:
                # 没有历史数据，假设是首次涨停
                first_record = stock_limit_up[stock_limit_up['trade_date'].astype(str) == earliest_date]
                if not first_record.empty:
                    first_limit_up.append(first_record.iloc[0].to_dict())
        
        # 转换为DataFrame
        result = pd.DataFrame(first_limit_up)
        
        # 保存结果
        if not result.empty:
            result.to_csv(cache_file, index=False)
            logger.info(f"发现 {len(result)} 只首次涨停股票")
        else:
            logger.warning(f"未找到首次涨停的股票")
        
        return result
    
    def find_continuous_strong_stocks(self, date=None, lookback_days=5):
        """寻找持续走强的股票
        
        Args:
            date: 查询日期，默认为最近交易日
            lookback_days: 回溯天数
        
        Returns:
            持续走强的股票列表
        """
        if date is None:
            date = self.get_trading_date()
        else:
            date = date.replace('-', '')
        
        # 获取前N个交易日
        start_date = self.get_trading_date(date, -lookback_days)
        if start_date is None:
            logger.error("获取回溯日期失败")
            return pd.DataFrame()
        
        logger.info(f"分析 {start_date} 至 {date} 期间持续走强的股票")
        
        # 缓存文件
        cache_file = os.path.join(self.cache_dir, f"strong_stocks_{start_date}_{date}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file):
            logger.info("从缓存读取持续走强股票")
            return pd.read_csv(cache_file)
        
        # 获取股票列表
        stock_list = self.fetcher.api_call(
            self.pro.stock_basic,
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,market,list_date'
        )
        
        if stock_list.empty:
            logger.error("获取股票列表失败")
            return pd.DataFrame()
        
        # 随机选择一定数量的股票分析，避免API调用次数过多
        sample_size = min(1000, len(stock_list))
        sampled_stocks = stock_list.sample(sample_size)
        
        # 获取每只股票的K线数据
        strong_stocks = []
        
        for _, stock in sampled_stocks.iterrows():
            ts_code = stock['ts_code']
            
            try:
                # 获取日线数据
                daily_data = self.fetcher.api_call(
                    self.pro.daily,
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=date
                )
                
                if daily_data.empty or len(daily_data) < 3:
                    continue
                
                # 计算技术指标
                # 1. 日均涨幅
                daily_data['pct_chg'] = daily_data['pct_chg'].astype(float)
                avg_pct_chg = daily_data['pct_chg'].mean()
                
                # 2. 主力净流入
                # 这里需要额外获取资金流向数据，为简化处理，使用换手率作为指标
                if 'vol' in daily_data.columns and 'amount' in daily_data.columns:
                    daily_data['turnover'] = daily_data['amount'] / daily_data['vol']
                    avg_turnover = daily_data['turnover'].mean()
                else:
                    avg_turnover = 0
                
                # 3. 连续收盘价高于开盘价的天数
                daily_data['is_positive'] = (daily_data['close'] > daily_data['open']).astype(int)
                positive_days = daily_data['is_positive'].sum()
                
                # 4. 成交量变化趋势
                if len(daily_data) >= 3:
                    vol_trend = (daily_data['vol'].iloc[0] - daily_data['vol'].iloc[-1]) / daily_data['vol'].iloc[-1]
                else:
                    vol_trend = 0
                
                # 5. K线形态
                # 简单判断最近一天是否形成光头阳线
                recent_day = daily_data.iloc[0]
                if recent_day['close'] > recent_day['open'] and \
                   abs(recent_day['close'] - recent_day['high']) / recent_day['close'] < 0.005:
                    has_strong_pattern = True
                else:
                    has_strong_pattern = False
                
                # 综合评分
                score = 0
                # 平均涨幅 +10分（每1%涨幅加2分）
                score += min(10, avg_pct_chg * 2)
                # 换手率 +5分
                score += min(5, avg_turnover / 5)
                # 上涨天数 +5分（每天+1分）
                score += min(5, positive_days)
                # 成交量趋势 +5分（递增趋势加分）
                score += min(5, vol_trend * -5 if vol_trend < 0 else 0)
                # K线形态 +5分
                score += 5 if has_strong_pattern else 0
                
                # 总分25分，超过15分视为强势股
                if score >= 15:
                    strong_stocks.append({
                        'ts_code': ts_code,
                        'name': stock['name'],
                        'industry': stock.get('industry', ''),
                        'score': score,
                        'avg_pct_chg': avg_pct_chg,
                        'positive_days': positive_days,
                        'recent_close': recent_day['close'],
                        'recent_pct_chg': recent_day['pct_chg']
                    })
            
            except Exception as e:
                logger.warning(f"处理股票 {ts_code} 时出错: {str(e)}")
                continue
        
        # 转换为DataFrame并排序
        result = pd.DataFrame(strong_stocks).sort_values('score', ascending=False)
        
        # 保存结果
        if not result.empty:
            result.to_csv(cache_file, index=False)
            logger.info(f"发现 {len(result)} 只持续走强股票")
        else:
            logger.warning("未找到持续走强的股票")
        
        return result
    
    def get_recommended_stocks(self, date=None, strategy="limit_up"):
        """获取推荐买入的股票
        
        Args:
            date: 日期，默认为最近交易日
            strategy: 策略类型，可选值：
                - "limit_up": 首次涨停策略
                - "continuous": 连续涨停策略
                - "strong": 持续走强策略
                - "combined": 综合策略
        
        Returns:
            推荐买入的股票列表
        """
        if date is None:
            date = self.get_trading_date()
        else:
            date = date.replace('-', '')
        
        # 获取前一个交易日
        prev_date = self.get_trading_date(date, -1)
        if prev_date is None:
            logger.error("获取前一交易日失败")
            return pd.DataFrame()
        
        # 缓存文件
        cache_file = os.path.join(self.results_dir, f"recommended_{strategy}_{date}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file):
            logger.info(f"从缓存读取{strategy}推荐股票")
            return pd.read_csv(cache_file)
        
        recommended = pd.DataFrame()
        
        if strategy == "limit_up" or strategy == "combined":
            # 首次涨停策略：寻找前一个交易日首次涨停的股票
            first_limit_up = self.find_first_limit_up(prev_date)
            if not first_limit_up.empty:
                # 添加推荐理由
                first_limit_up['reason'] = "首次涨停，可能有较强后续表现"
                first_limit_up['strategy'] = "首次涨停策略"
                
                if strategy == "limit_up":
                    recommended = first_limit_up
                elif recommended.empty:
                    recommended = first_limit_up
                else:
                    recommended = pd.concat([recommended, first_limit_up], ignore_index=True)
        
        if strategy == "continuous" or strategy == "combined":
            # 连续涨停策略：获取连续两天及以上涨停的股票
            continuous_data = self.fetcher.get_continuous_limit_up(date[:4], min_days=2, max_days=3)
            
            if continuous_data and 2 in continuous_data and continuous_data[2]:
                continuous_df = pd.DataFrame(continuous_data[2])
                # 筛选出最近一个交易日结束涨停的股票
                if 'end_date' in continuous_df.columns:
                    recent_continuous = continuous_df[continuous_df['end_date'].astype(str) == prev_date]
                    if not recent_continuous.empty:
                        # 添加推荐理由
                        recent_continuous['reason'] = "连续涨停，强势明显"
                        recent_continuous['strategy'] = "连续涨停策略"
                        
                        if strategy == "continuous":
                            recommended = recent_continuous
                        elif recommended.empty:
                            recommended = recent_continuous
                        else:
                            recommended = pd.concat([recommended, recent_continuous], ignore_index=True)
        
        if strategy == "strong" or strategy == "combined":
            # 持续走强策略：寻找近期持续走强的股票
            strong_stocks = self.find_continuous_strong_stocks(date)
            if not strong_stocks.empty:
                # 添加推荐理由
                strong_stocks['reason'] = "持续走强，势头良好"
                strong_stocks['strategy'] = "持续走强策略"
                
                if strategy == "strong":
                    recommended = strong_stocks
                elif recommended.empty:
                    recommended = strong_stocks
                else:
                    recommended = pd.concat([recommended, strong_stocks], ignore_index=True)
        
        # 保存结果
        if not recommended.empty:
            # 确保没有重复的股票
            if 'ts_code' in recommended.columns:
                recommended = recommended.drop_duplicates(subset=['ts_code'])
            
            recommended.to_csv(cache_file, index=False)
            logger.info(f"共推荐 {len(recommended)} 只股票")
        else:
            logger.warning("没有找到符合条件的推荐股票")
        
        return recommended
    
    def print_recommendations(self, recommended):
        """打印推荐股票信息
        
        Args:
            recommended: 推荐股票DataFrame
        """
        if recommended.empty:
            print("\n没有找到符合条件的推荐股票")
            return
        
        print("\n" + "="*60)
        print("股票推荐清单")
        print("="*60)
        
        # 确定要显示的列
        display_columns = []
        for col in ['ts_code', 'name', 'industry', 'reason', 'strategy']:
            if col in recommended.columns:
                display_columns.append(col)
        
        # 添加评分列（如果有）
        if 'score' in recommended.columns:
            display_columns.append('score')
        
        # 打印推荐股票
        print(recommended[display_columns])
        print("\n" + "="*60)

def get_parser():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='涨停板交易策略工具')
    
    parser.add_argument('--token', type=str,
                        default="0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10",
                        help='Tushare API令牌')
    
    parser.add_argument('--date', type=str, default=None,
                        help='查询日期，格式为YYYYMMDD，默认为最近交易日')
    
    parser.add_argument('--strategy', type=str, default="combined",
                        choices=["limit_up", "continuous", "strong", "combined"],
                        help='策略类型：limit_up=首次涨停，continuous=连续涨停，strong=持续走强，combined=综合策略')
    
    parser.add_argument('--cache-dir', type=str, default='./cache/trading',
                        help='缓存目录，默认为./cache/trading')
    
    return parser

def main():
    """主函数"""
    # 解析命令行参数
    parser = get_parser()
    args = parser.parse_args()
    
    try:
        # 初始化策略
        strategy = LimitUpStrategy(args.token, cache_dir=args.cache_dir)
        
        # 获取推荐股票
        recommended = strategy.get_recommended_stocks(args.date, args.strategy)
        
        # 打印推荐
        strategy.print_recommendations(recommended)
        
        logger.info("交易推荐完成!")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("trading_strategy.log"),
            logging.StreamHandler()
        ]
    )
    
    main() 