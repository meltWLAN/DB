#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
涨停前买入策略模块
识别有涨停潜力的股票，在涨停前买入
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tushare as ts
import argparse
import concurrent.futures
from final_limit_up import VipLimitUpFetcher

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pre_limit_up.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PreLimitUp")

class PreLimitUpStrategy:
    """涨停前买入策略"""
    
    def __init__(self, token=None, cache_dir='./cache/pre_limit_up'):
        """初始化
        
        Args:
            token: Tushare API Token (可选，默认使用已配置的token)
            cache_dir: 缓存目录
        """
        # 使用提供的token或默认token
        self.token = token if token else "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
        self.cache_dir = cache_dir
        self.results_dir = os.path.join(cache_dir, 'results')
        
        # 创建缓存和结果目录
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 初始化数据获取器
        self.fetcher = VipLimitUpFetcher(self.token, cache_dir=os.path.join(cache_dir, 'limit_up_data'))
        
        # 初始化tushare直接接口
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        
        # 并发参数
        self.max_workers = 5  # 最大并发线程数
        
        # 东方财富板块成分和游资名录缓存目录
        self.dc_cache_dir = os.path.join(cache_dir, 'dc_data')
        self.hm_cache_dir = os.path.join(cache_dir, 'hm_data')
        os.makedirs(self.dc_cache_dir, exist_ok=True)
        os.makedirs(self.hm_cache_dir, exist_ok=True)
    
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
    
    def find_limit_up_boards(self, date=None, lookback_days=30):
        """找出最近活跃的涨停板块
        
        Args:
            date: 日期，默认为最近交易日
            lookback_days: 回溯天数
        
        Returns:
            活跃涨停板块列表
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
        
        logger.info(f"分析 {start_date} 至 {date} 期间活跃的涨停板块")
        
        # 缓存文件
        cache_file = os.path.join(self.cache_dir, f"active_boards_{start_date}_{date}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file):
            logger.info("从缓存读取活跃板块")
            return pd.read_csv(cache_file)
        
        # 获取期间的涨停股票
        all_limit_up_stocks = []
        
        # 按月获取数据
        current_date = datetime.strptime(start_date, '%Y%m%d')
        end_date_obj = datetime.strptime(date, '%Y%m%d')
        
        while current_date <= end_date_obj:
            year = current_date.year
            month = current_date.month
            
            # 获取月度涨停数据
            month_data = self.fetcher.get_monthly_limit_up(year, month)
            if not month_data.empty:
                # 筛选日期范围
                month_data = month_data[
                    (month_data['trade_date'].astype(str) >= start_date) & 
                    (month_data['trade_date'].astype(str) <= date)
                ]
                if not month_data.empty:
                    all_limit_up_stocks.append(month_data)
            
            # 下一个月
            if month == 12:
                current_date = datetime(year + 1, 1, 1)
            else:
                current_date = datetime(year, month + 1, 1)
        
        # 合并数据
        if not all_limit_up_stocks:
            logger.warning(f"{start_date} 至 {date} 期间没有涨停数据")
            return pd.DataFrame()
        
        limit_up_data = pd.concat(all_limit_up_stocks, ignore_index=True)
        
        # 获取行业信息
        if 'industry' not in limit_up_data.columns:
            # 获取股票行业信息
            stock_info = self.fetcher.api_call(
                self.pro.stock_basic,
                exchange='',
                list_status='L',
                fields='ts_code,name,industry'
            )
            
            if not stock_info.empty:
                # 合并行业信息
                limit_up_data = pd.merge(limit_up_data, stock_info[['ts_code', 'industry']], on='ts_code', how='left')
        
        # 统计每个行业的涨停次数
        industry_count = limit_up_data.groupby('industry').size()
        industry_count.name = 'limit_up_count'
        industry_count = industry_count.reset_index().sort_values('limit_up_count', ascending=False)
        
        # 计算最近的涨停趋势
        recent_days = min(7, lookback_days)  # 最近7天
        recent_date = self.get_trading_date(date, -recent_days)
        
        recent_limit_up = limit_up_data[limit_up_data['trade_date'].astype(str) >= recent_date]
        if not recent_limit_up.empty:
            recent_industry_count = recent_limit_up.groupby('industry').size()
            recent_industry_count.name = 'recent_count'
            recent_industry_count = recent_industry_count.reset_index()
            
            # 合并近期涨停数据
            industry_count = pd.merge(industry_count, recent_industry_count, on='industry', how='left')
            industry_count['recent_count'].fillna(0, inplace=True)
            
            # 计算近期活跃度占比
            industry_count['recent_ratio'] = industry_count['recent_count'] / industry_count['limit_up_count']
        else:
            industry_count['recent_count'] = 0
            industry_count['recent_ratio'] = 0
        
        # 计算综合活跃度得分
        industry_count['active_score'] = industry_count['limit_up_count'] * 0.6 + industry_count['recent_count'] * 0.4
        
        # 排序
        industry_count = industry_count.sort_values('active_score', ascending=False)
        
        # 保存结果
        industry_count.to_csv(cache_file, index=False)
        logger.info(f"找到 {len(industry_count)} 个活跃板块")
        
        return industry_count
    
    def analyze_stock_features(self, ts_code, date=None, lookback_days=10):
        """分析单只股票的特征，判断是否有涨停潜力
        
        Args:
            ts_code: 股票代码
            date: 日期，默认为最近交易日
            lookback_days: 回溯天数
        
        Returns:
            股票特征字典，包含涨停潜力评分
        """
        if date is None:
            date = self.get_trading_date()
        else:
            date = date.replace('-', '')
        
        # 获取前N个交易日
        start_date = self.get_trading_date(date, -lookback_days)
        if start_date is None:
            return None
        
        # 获取日线数据
        daily_data = self.fetcher.api_call(
            self.pro.daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=date
        )
        
        if daily_data.empty or len(daily_data) < 5:
            return None
        
        # 按日期降序排序
        daily_data = daily_data.sort_values('trade_date', ascending=False)
        
        # 获取行业信息
        stock_info = self.fetcher.api_call(
            self.pro.stock_basic,
            ts_code=ts_code,
            fields='ts_code,name,industry,market'
        )
        
        if stock_info.empty:
            return None
        
        # 计算特征
        features = {
            'ts_code': ts_code,
            'name': stock_info['name'].values[0],
            'industry': stock_info['industry'].values[0] if 'industry' in stock_info.columns else '',
            'market': stock_info['market'].values[0] if 'market' in stock_info.columns else '',
            'date': date,
            'current_price': daily_data.iloc[0]['close']
        }
        
        # 计算是否科创板或创业板
        is_tech_stock = ts_code.startswith('688') or ts_code.startswith('300')
        features['is_tech_stock'] = is_tech_stock
        
        # 计算近期涨幅
        features['recent_pct_chg'] = daily_data.iloc[0]['pct_chg']
        
        # 计算5日涨幅
        if len(daily_data) >= 5:
            five_day_change = (daily_data.iloc[0]['close'] / daily_data.iloc[4]['close'] - 1) * 100
            features['five_day_pct_chg'] = five_day_change
        else:
            features['five_day_pct_chg'] = 0
        
        # 计算10日涨幅
        if len(daily_data) >= 10:
            ten_day_change = (daily_data.iloc[0]['close'] / daily_data.iloc[9]['close'] - 1) * 100
            features['ten_day_pct_chg'] = ten_day_change
        else:
            features['ten_day_pct_chg'] = 0
        
        # 计算现价距离涨停价的距离
        limit_threshold = 20 if is_tech_stock else 10  # 科创板/创业板20%，主板10%
        limit_up_price = daily_data.iloc[0]['close'] * (1 + limit_threshold / 100)
        features['limit_up_price'] = limit_up_price
        features['pct_to_limit'] = limit_threshold - features['recent_pct_chg']
        
        # 计算5日和10日平均成交量比值
        if len(daily_data) >= 10:
            vol_5d_avg = daily_data.iloc[:5]['vol'].mean()
            vol_10d_avg = daily_data.iloc[:10]['vol'].mean()
            features['vol_ratio_5d_10d'] = vol_5d_avg / vol_10d_avg if vol_10d_avg > 0 else 1
        else:
            features['vol_ratio_5d_10d'] = 1
        
        # 计算近3天最高价与最低价的波动范围
        if len(daily_data) >= 3:
            max_high = daily_data.iloc[:3]['high'].max()
            min_low = daily_data.iloc[:3]['low'].min()
            features['price_range_3d'] = (max_high - min_low) / min_low * 100
        else:
            features['price_range_3d'] = 0
        
        # 计算最近交易日K线特征
        recent_day = daily_data.iloc[0]
        
        # 上影线比例
        if recent_day['high'] > recent_day['close']:
            features['upper_shadow_ratio'] = (recent_day['high'] - recent_day['close']) / recent_day['close'] * 100
        else:
            features['upper_shadow_ratio'] = 0
        
        # 下影线比例
        if recent_day['low'] < recent_day['open']:
            features['lower_shadow_ratio'] = (recent_day['open'] - recent_day['low']) / recent_day['open'] * 100
        else:
            features['lower_shadow_ratio'] = 0
        
        # 实体比例
        features['body_ratio'] = abs(recent_day['close'] - recent_day['open']) / recent_day['open'] * 100
        
        # 收盘价是否创近期新高
        if len(daily_data) >= 5:
            features['is_new_high_5d'] = 1 if recent_day['close'] >= daily_data.iloc[:5]['close'].max() else 0
        else:
            features['is_new_high_5d'] = 0
            
        # 尝试获取东方财富概念板块信息
        try:
            concept_analysis = self.analyze_stock_with_concept_boards(ts_code, date)
            if concept_analysis:
                features['concept_board_count'] = concept_analysis['concept_board_count']
                features['hot_concept_count'] = concept_analysis['hot_concept_count']
                features['max_concept_score'] = concept_analysis['max_board_score']
                features['avg_concept_score'] = concept_analysis['avg_board_score']
            else:
                features['concept_board_count'] = 0
                features['hot_concept_count'] = 0
                features['max_concept_score'] = 0
                features['avg_concept_score'] = 0
        except Exception as e:
            logger.warning(f"获取股票 {ts_code} 概念板块信息出错: {str(e)}")
            features['concept_board_count'] = 0
            features['hot_concept_count'] = 0
            features['max_concept_score'] = 0
            features['avg_concept_score'] = 0
            
        # 尝试获取游资活跃度信息
        try:
            hot_money_analysis = self.analyze_hot_money_activity(ts_code, date)
            if hot_money_analysis:
                features['hot_money_activity_score'] = hot_money_analysis['hot_money_activity_score']
                features['hot_money_count'] = hot_money_analysis['hot_money_count']
                features['recent_hot_money_activity'] = 1 if hot_money_analysis['recent_activity'] else 0
            else:
                features['hot_money_activity_score'] = 0
                features['hot_money_count'] = 0
                features['recent_hot_money_activity'] = 0
        except Exception as e:
            logger.warning(f"获取股票 {ts_code} 游资活跃度信息出错: {str(e)}")
            features['hot_money_activity_score'] = 0
            features['hot_money_count'] = 0
            features['recent_hot_money_activity'] = 0
        
        # 计算涨停潜力评分
        score = 0
        
        # 1. 近期涨幅得分 (0-20分)
        # 涨幅适中(3-7%)得高分，过低或过高都降分
        recent_pct = features['recent_pct_chg']
        if 3 <= recent_pct < 7:
            score += 20
        elif 1 <= recent_pct < 3 or 7 <= recent_pct < 9:
            score += 10
        elif recent_pct >= 9:
            score -= 10  # 已经接近涨停，风险较大
        
        # 2. 5日涨幅 (0-15分)
        five_day_pct = features['five_day_pct_chg']
        if 10 <= five_day_pct < 20:
            score += 15
        elif 5 <= five_day_pct < 10 or 20 <= five_day_pct < 30:
            score += 10
        elif five_day_pct >= 30:
            score += 5  # 涨幅过大，可能回调
        
        # 3. 距离涨停板距离 (0-15分，降低权重)
        pct_to_limit = features['pct_to_limit']
        if 3 <= pct_to_limit < 6:
            score += 15  # 适中距离，有冲击涨停可能
        elif 1.5 <= pct_to_limit < 3 or 6 <= pct_to_limit < 8:
            score += 8
        
        # 4. 成交量特征 (0-15分)
        vol_ratio = features['vol_ratio_5d_10d']
        if 1.2 <= vol_ratio < 2:
            score += 15  # 成交量适度放大
        elif 2 <= vol_ratio < 3:
            score += 10
        elif vol_ratio >= 3:
            score += 5  # 成交量剧增，可能短期过热
        
        # 5. K线形态 (0-15分)
        # 光头阳线特征
        if recent_day['close'] > recent_day['open'] and features['upper_shadow_ratio'] < 0.5:
            score += 10
        
        # 带下影线特征（抄底）
        if features['lower_shadow_ratio'] > 2 and recent_day['close'] > recent_day['open']:
            score += 5
        
        # 创新高特征
        if features['is_new_high_5d'] == 1:
            score += 5
        
        # 6. 价格波动范围 (0-10分，降低权重)
        price_range = features['price_range_3d']
        if 5 <= price_range < 10:
            score += 10  # 波动适中，活跃但不过度
        elif 10 <= price_range < 15:
            score += 7
        elif price_range >= 15:
            score += 3  # 波动过大，风险较高
            
        # 7. 概念板块特征 (0-15分)
        # 热门概念板块数量
        hot_concept_count = features['hot_concept_count']
        if hot_concept_count >= 3:
            score += 15  # 3个及以上热门概念
        elif hot_concept_count == 2:
            score += 10
        elif hot_concept_count == 1:
            score += 5
            
        # 最高概念板块得分
        max_concept_score = features.get('max_concept_score', 0)
        if max_concept_score >= 80:
            score += 10  # 有超高活跃度的概念
        elif max_concept_score >= 50:
            score += 5
            
        # 8. 游资活跃度特征 (0-10分)
        hot_money_score = features.get('hot_money_activity_score', 0)
        if hot_money_score >= 70:
            score += 10  # 游资高度活跃
        elif hot_money_score >= 40:
            score += 5
            
        # 最近游资活动
        if features.get('recent_hot_money_activity', 0) == 1:
            score += 5  # 最近有游资活动
        
        # 总分
        features['limit_up_potential_score'] = score
        
        # 风险等级
        if score >= 75:
            features['risk_level'] = '低'
        elif score >= 50:
            features['risk_level'] = '中'
        else:
            features['risk_level'] = '高'
        
        return features
    
    def get_dc_members(self, ts_code=None, trade_date=None):
        """获取东方财富板块成分数据
        
        Args:
            ts_code: 板块代码，如'BK1184.DC'，默认为None获取所有
            trade_date: 交易日期，默认为最近交易日
            
        Returns:
            板块成分数据DataFrame
        """
        if trade_date is None:
            trade_date = self.get_trading_date()
        else:
            trade_date = trade_date.replace('-', '')
            
        # 缓存文件名
        cache_key = f"dc_member_{trade_date}_{ts_code if ts_code else 'all'}.csv"
        cache_file = os.path.join(self.dc_cache_dir, cache_key)
        
        # 检查缓存
        if os.path.exists(cache_file):
            logger.info(f"从缓存读取东方财富板块成分数据: {cache_key}")
            return pd.read_csv(cache_file)
        
        try:
            # 调用东方财富板块成分接口
            params = {'trade_date': trade_date}
            if ts_code:
                params['ts_code'] = ts_code
                
            dc_members = self.fetcher.api_call(
                self.pro.dc_member,
                **params
            )
            
            if not dc_members.empty:
                # 保存到缓存
                dc_members.to_csv(cache_file, index=False)
                logger.info(f"获取到 {len(dc_members)} 条东方财富板块成分数据")
                return dc_members
            else:
                logger.warning(f"未获取到东方财富板块成分数据: {trade_date}, {ts_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取东方财富板块成分数据出错: {str(e)}")
            return pd.DataFrame()
    
    def get_hot_money_list(self, name=None):
        """获取游资名录信息
        
        Args:
            name: 游资名称，默认为None获取所有
            
        Returns:
            游资名录DataFrame
        """
        # 缓存文件名
        cache_key = f"hm_list_{name if name else 'all'}.csv"
        cache_file = os.path.join(self.hm_cache_dir, cache_key)
        
        # 检查缓存
        if os.path.exists(cache_file):
            logger.info(f"从缓存读取游资名录数据: {cache_key}")
            return pd.read_csv(cache_file)
        
        try:
            # 调用游资名录接口
            params = {}
            if name:
                params['name'] = name
                
            hm_list = self.fetcher.api_call(
                self.pro.hm_list,
                **params
            )
            
            if not hm_list.empty:
                # 保存到缓存
                hm_list.to_csv(cache_file, index=False)
                logger.info(f"获取到 {len(hm_list)} 条游资名录数据")
                return hm_list
            else:
                logger.warning(f"未获取到游资名录数据: {name}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取游资名录数据出错: {str(e)}")
            return pd.DataFrame()
    
    def get_concept_boards(self, date=None):
        """获取概念板块列表
        
        Args:
            date: 交易日期，默认为最近交易日
            
        Returns:
            概念板块列表DataFrame
        """
        if date is None:
            date = self.get_trading_date()
        else:
            date = date.replace('-', '')
            
        # 缓存文件名
        cache_file = os.path.join(self.dc_cache_dir, f"concept_boards_{date}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file):
            logger.info(f"从缓存读取概念板块列表: {date}")
            return pd.read_csv(cache_file)
        
        try:
            # 这里假设通过某种方式获取概念板块列表
            # 由于接口文档中没有直接提供获取板块列表的接口，这里可以通过其他方式获取
            # 例如，可以从已有的数据中提取，或者使用其他接口
            
            # 示例：从已有的dc_member数据中提取唯一的板块代码
            all_members = self.get_dc_members(trade_date=date)
            if not all_members.empty and 'ts_code' in all_members.columns:
                concept_boards = all_members[['ts_code']].drop_duplicates()
                concept_boards.to_csv(cache_file, index=False)
                logger.info(f"获取到 {len(concept_boards)} 个概念板块")
                return concept_boards
            else:
                logger.warning(f"未能获取概念板块列表: {date}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取概念板块列表出错: {str(e)}")
            return pd.DataFrame()
    
    def analyze_stock_with_concept_boards(self, ts_code, date=None):
        """分析股票所属的概念板块及其活跃度
        
        Args:
            ts_code: 股票代码
            date: 日期，默认为最近交易日
            
        Returns:
            概念板块分析结果字典
        """
        if date is None:
            date = self.get_trading_date()
        else:
            date = date.replace('-', '')
            
        # 获取所有板块成分
        all_members = self.get_dc_members(trade_date=date)
        if all_members.empty:
            return {}
            
        # 筛选该股票所属的所有板块
        stock_boards = all_members[all_members['con_code'] == ts_code]
        if stock_boards.empty:
            return {}
            
        # 获取板块活跃度数据
        active_boards = self.find_limit_up_boards(date)
        
        # 分析结果
        result = {
            'ts_code': ts_code,
            'concept_boards': [],
            'concept_board_count': len(stock_boards),
            'hot_concept_count': 0,
            'max_board_score': 0,
            'avg_board_score': 0
        }
        
        total_score = 0
        hot_concepts = 0
        
        # 分析每个板块
        for _, board in stock_boards.iterrows():
            board_code = board['ts_code']
            board_info = {
                'board_code': board_code,
                'board_name': board.get('name', '未知板块'),
                'active_score': 0
            }
            
            # 查找板块活跃度
            if not active_boards.empty and 'industry' in active_boards.columns:
                # 假设板块代码可以映射到industry字段
                # 实际使用时可能需要调整映射关系
                board_active = active_boards[active_boards['industry'] == board_code]
                if not board_active.empty:
                    score = board_active['active_score'].values[0]
                    board_info['active_score'] = score
                    total_score += score
                    
                    if score > result['max_board_score']:
                        result['max_board_score'] = score
                        
                    if score > 50:  # 假设50分以上为热门板块
                        hot_concepts += 1
            
            result['concept_boards'].append(board_info)
        
        # 计算平均分和热门板块数
        if result['concept_board_count'] > 0:
            result['avg_board_score'] = total_score / result['concept_board_count']
        result['hot_concept_count'] = hot_concepts
        
        return result
    
    def analyze_hot_money_activity(self, ts_code, date=None, lookback_days=30):
        """分析游资对该股票的活跃度
        
        Args:
            ts_code: 股票代码
            date: 日期，默认为最近交易日
            lookback_days: 回溯天数
            
        Returns:
            游资活跃度分析结果字典
        """
        # 这个函数需要龙虎榜数据支持，这里仅做示例
        # 实际实现时需要根据可用的数据接口调整
        
        if date is None:
            date = self.get_trading_date()
        else:
            date = date.replace('-', '')
            
        # 获取游资名录
        hot_money_list = self.get_hot_money_list()
        if hot_money_list.empty:
            return {}
            
        # 这里应该获取龙虎榜数据，分析该股票是否被游资关注
        # 由于没有直接的接口，这里仅做示例
        
        # 示例结果
        result = {
            'ts_code': ts_code,
            'hot_money_activity_score': 0,  # 游资活跃度得分
            'hot_money_count': 0,           # 关注该股的游资数量
            'recent_activity': False,        # 最近是否有游资活动
            'hot_money_names': []            # 关注该股的游资名称列表
        }
        
        # 实际实现时，应该根据龙虎榜数据分析游资活动
        # 这里仅返回示例结果
        return result
    
    def find_potential_limit_up_stocks(self, date=None, top_n=20):
        """找出有涨停潜力的股票
        
        Args:
            date: 日期，默认为最近交易日
            top_n: 返回的股票数量
        
        Returns:
            有涨停潜力的股票列表
        """
        if date is None:
            date = self.get_trading_date()
        else:
            date = date.replace('-', '')
        
        logger.info(f"查找 {date} 有涨停潜力的股票")
        
        # 缓存文件
        cache_file = os.path.join(self.results_dir, f"potential_limit_up_{date}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file):
            logger.info("从缓存读取有涨停潜力的股票")
            return pd.read_csv(cache_file)
        
        # 步骤1: 找出活跃的涨停板块
        active_boards = self.find_limit_up_boards(date)
        if active_boards.empty:
            logger.warning("未找到活跃的涨停板块")
            return pd.DataFrame()
        
        # 选择前10个最活跃的板块
        top_boards = active_boards.head(10)
        logger.info(f"选择了 {len(top_boards)} 个活跃板块进行分析")
        
        # 步骤2: 获取这些板块内的股票
        stock_info = self.fetcher.api_call(
            self.pro.stock_basic,
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,market,list_date'
        )
        
        if stock_info.empty:
            logger.error("获取股票列表失败")
            return pd.DataFrame()
        
        # 步骤3: 尝试获取东方财富概念板块数据
        try:
            # 获取东方财富概念板块成分
            dc_members = self.get_dc_members(trade_date=date)
            if not dc_members.empty:
                logger.info(f"成功获取东方财富概念板块成分数据: {len(dc_members)} 条记录")
                
                # 合并股票信息和概念板块信息
                dc_stocks = dc_members['con_code'].unique()
                logger.info(f"东方财富概念板块包含 {len(dc_stocks)} 只股票")
                
                # 将概念板块股票加入分析范围
                concept_stocks = stock_info[stock_info['ts_code'].isin(dc_stocks)]
                board_stocks = pd.concat([stock_info[stock_info['industry'].isin(top_boards['industry'])], concept_stocks]).drop_duplicates()
            else:
                # 如果没有获取到概念板块数据，仅使用行业板块
                board_stocks = stock_info[stock_info['industry'].isin(top_boards['industry'])]
        except Exception as e:
            logger.warning(f"获取东方财富概念板块数据出错: {str(e)}，仅使用行业板块")
            # 筛选出属于热门板块的股票
            board_stocks = stock_info[stock_info['industry'].isin(top_boards['industry'])]
        
        logger.info(f"找到 {len(board_stocks)} 只属于热门板块的股票")
        
        # 步骤4: 分析每只股票的特征，评估涨停潜力
        potential_stocks = []
        
        # 使用并发加速处理
        def process_stock(stock):
            try:
                ts_code = stock['ts_code']
                # 基础特征分析
                features = self.analyze_stock_features(ts_code, date)
                if features and features['limit_up_potential_score'] >= 50:  # 只选择评分50以上的
                    # 添加板块活跃度得分
                    industry = features['industry']
                    if industry in top_boards['industry'].values:
                        active_score = top_boards[top_boards['industry'] == industry]['active_score'].values[0]
                        features['board_active_score'] = active_score
                    else:
                        features['board_active_score'] = 0
                    
                    # 添加概念板块分析
                    concept_analysis = self.analyze_stock_with_concept_boards(ts_code, date)
                    if concept_analysis:
                        features['concept_board_count'] = concept_analysis['concept_board_count']
                        features['hot_concept_count'] = concept_analysis['hot_concept_count']
                        features['max_board_score'] = concept_analysis['max_board_score']
                        features['avg_board_score'] = concept_analysis['avg_board_score']
                    else:
                        features['concept_board_count'] = 0
                        features['hot_concept_count'] = 0
                        features['max_board_score'] = 0
                        features['avg_board_score'] = 0
                    
                    # 添加游资活跃度分析
                    hot_money_analysis = self.analyze_hot_money_activity(ts_code, date)
                    if hot_money_analysis:
                        features['hot_money_activity_score'] = hot_money_analysis['hot_money_activity_score']
                        features['hot_money_count'] = hot_money_analysis['hot_money_count']
                        features['recent_hot_money_activity'] = 1 if hot_money_analysis['recent_activity'] else 0
                    else:
                        features['hot_money_activity_score'] = 0
                        features['hot_money_count'] = 0
                        features['recent_hot_money_activity'] = 0
                    
                    return features
            except Exception as e:
                logger.warning(f"分析股票 {stock['ts_code']} 特征时出错: {str(e)}")
            return None
        
        # 限制处理的股票数量，避免API调用过多
        sample_size = min(500, len(board_stocks))
        sampled_stocks = board_stocks.sample(sample_size)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_stock, [stock for _, stock in sampled_stocks.iterrows()]))
        
        # 过滤掉None结果
        potential_stocks = [stock for stock in results if stock]
        
        # 转换为DataFrame并排序
        if potential_stocks:
            result = pd.DataFrame(potential_stocks)
            
            # 计算综合得分: 基础涨停潜力(50%) + 板块活跃度(20%) + 概念板块(20%) + 游资活跃度(10%)
            result['final_score'] = result['limit_up_potential_score'] * 0.5 + \
                                   result['board_active_score'] * 0.2
            
            # 如果有概念板块数据，加入评分
            if 'max_board_score' in result.columns:
                result['final_score'] += result['max_board_score'] * 0.1 + \
                                        result['hot_concept_count'] * 5 * 0.1  # 每个热门概念加5分，最多占10%
            
            # 如果有游资活跃度数据，加入评分
            if 'hot_money_activity_score' in result.columns:
                result['final_score'] += result['hot_money_activity_score'] * 0.1
            
            # 排序并选择前N名
            result = result.sort_values('final_score', ascending=False).head(top_n)
            
            # 保存结果
            result.to_csv(cache_file, index=False)
            logger.info(f"找到 {len(result)} 只有涨停潜力的股票")
            return result
        else:
            logger.warning("未找到有涨停潜力的股票")
            return pd.DataFrame()
    
    def print_potential_stocks(self, stocks):
        """打印有涨停潜力的股票
        
        Args:
            stocks: 有涨停潜力的股票DataFrame
        """
        if stocks.empty:
            print("\n未找到有涨停潜力的股票")
            return
        
        print("\n" + "="*80)
        print("涨停潜力股票推荐")
        print("="*80)
        
        # 选择要显示的列
        basic_columns = ['ts_code', 'name', 'industry', 'current_price', 
                         'recent_pct_chg', 'limit_up_potential_score', 'final_score', 'risk_level']
        
        # 概念板块相关列
        concept_columns = ['concept_board_count', 'hot_concept_count', 'max_concept_score']
        
        # 游资活跃度相关列
        hot_money_columns = ['hot_money_activity_score', 'hot_money_count', 'recent_hot_money_activity']
        
        # 检查是否有概念板块和游资活跃度数据
        has_concept_data = all(col in stocks.columns for col in concept_columns)
        has_hot_money_data = all(col in stocks.columns for col in hot_money_columns)
        
        # 构建显示列
        display_columns = basic_columns.copy()
        if has_concept_data:
            display_columns.extend(concept_columns)
        if has_hot_money_data:
            display_columns.extend(hot_money_columns)
        
        # 确保所有列都存在
        display_columns = [col for col in display_columns if col in stocks.columns]
        
        # 打印表头
        header = "排名  代码      名称          行业        现价    涨幅   潜力分  综合分  风险"
        if has_concept_data:
            header += "  概念数  热门概念  最高概念分"
        if has_hot_money_data:
            header += "  游资活跃度  游资关注  近期活动"
        print(header)
        print("-"*80)
        
        # 打印每只股票
        for i, (_, stock) in enumerate(stocks.iterrows(), 1):
            # 基本信息
            line = f"{i:2d}    {stock['ts_code'][:9]:<9} {stock['name'][:8]:<10} {stock['industry'][:8]:<10} "
            line += f"{stock['current_price']:6.2f} {stock['recent_pct_chg']:6.2f}% "
            line += f"{stock['limit_up_potential_score']:5.0f}  {stock['final_score']:6.1f}  {stock['risk_level']}"
            
            # 概念板块信息
            if has_concept_data:
                line += f"  {stock['concept_board_count']:4.0f}    {stock['hot_concept_count']:4.0f}     {stock['max_concept_score']:6.1f}"
            
            # 游资活跃度信息
            if has_hot_money_data:
                line += f"    {stock['hot_money_activity_score']:6.1f}    {stock['hot_money_count']:4.0f}     "
                line += "是" if stock['recent_hot_money_activity'] == 1 else "否"
            
            print(line)
        
        print("\n说明:")
        print("1. 潜力分: 股票涨停潜力评分，综合考虑技术指标")
        print("2. 综合分: 最终排名得分，结合板块活跃度、概念热度和游资活跃度")
        if has_concept_data:
            print("3. 概念数: 股票所属概念板块数量")
            print("4. 热门概念: 股票所属热门概念板块数量")
            print("5. 最高概念分: 股票所属概念中活跃度最高的分数")
        if has_hot_money_data:
            print("6. 游资活跃度: 游资对该股票的关注度评分")
            print("7. 游资关注: 关注该股票的游资数量")
            print("8. 近期活动: 最近是否有游资活动")
        print("="*80)

def get_parser():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='涨停前买入策略工具')
    
    parser.add_argument('--token', type=str,
                        help='Tushare API令牌')
    
    parser.add_argument('--date', type=str, default=None,
                        help='查询日期，格式为YYYYMMDD，默认为最近交易日')
    
    parser.add_argument('--top', type=int, default=20,
                        help='返回的股票数量，默认为20')
    
    parser.add_argument('--cache-dir', type=str, default='./cache/pre_limit_up',
                        help='缓存目录，默认为./cache/pre_limit_up')
    
    return parser

def main():
    """主函数"""
    # 解析命令行参数
    parser = get_parser()
    args = parser.parse_args()
    
    try:
        # 初始化策略
        strategy = PreLimitUpStrategy(args.token, cache_dir=args.cache_dir)
        
        # 获取有涨停潜力的股票
        potential_stocks = strategy.find_potential_limit_up_stocks(args.date, args.top)
        
        # 打印结果
        strategy.print_potential_stocks(potential_stocks)
        
        logger.info("涨停潜力股票分析完成!")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()