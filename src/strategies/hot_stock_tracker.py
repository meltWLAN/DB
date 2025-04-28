#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
暴涨股捕捉模块
专注于识别市场中的连续涨停股和潜在爆发股票
利用Tushare API实现高效的市场热点扫描
"""

import os
import time
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict

# 设置日志
logger = logging.getLogger(__name__)

class HotStockTracker:
    """
    暴涨股跟踪器
    专注于捕捉市场中连续涨停和即将爆发的股票
    """
    
    def __init__(self, tushare_fetcher=None, data_processor=None):
        """
        初始化暴涨股跟踪器
        
        Args:
            tushare_fetcher: TuShare数据获取器实例
            data_processor: 数据处理器实例
        """
        self.tushare_fetcher = tushare_fetcher
        self.data_processor = data_processor
        self.cache = {}
        self.cache_expires = {}
        self.cache_timeout = 3600  # 缓存有效期（秒）
        
        # 创建结果目录
        self.results_dir = os.path.join('results', 'hot_stocks')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 涨停板特征权重
        self.feature_weights = {
            'consecutive_limit_ups': 0.35,  # 连续涨停天数
            'turnover_rate': 0.15,          # 换手率
            'volume_ratio': 0.10,           # 量比
            'north_money_flow': 0.10,       # 北向资金流入
            'industry_heat': 0.15,          # 行业热度
            'institutional_participation': 0.15  # 机构参与度
        }
    
    def get_consecutive_limit_up_stocks(self, days=1, end_date=None) -> pd.DataFrame:
        """
        获取连续涨停股票
        
        Args:
            days: 连续涨停天数
            end_date: 结束日期，格式 YYYY-MM-DD，默认为今天
            
        Returns:
            DataFrame: 包含连续涨停股票及其详细信息
        """
        cache_key = f'limit_up_{days}_{end_date}'
        if cache_key in self.cache and time.time() - self.cache_expires.get(cache_key, 0) < self.cache_timeout:
            return self.cache[cache_key]
        
        if self.tushare_fetcher is None:
            logger.error("TuShare数据获取器未初始化")
            return pd.DataFrame()
        
        try:
            # 使用TuShare获取连续涨停股票
            limit_up_stocks = self.tushare_fetcher.get_continuous_limit_up_stocks(days, end_date)
            
            if limit_up_stocks is None or limit_up_stocks.empty:
                logger.warning(f"未找到连续{days}天涨停的股票")
                return pd.DataFrame()
            
            # 增强数据：获取更多信息
            enriched_data = []
            for _, stock in limit_up_stocks.iterrows():
                stock_code = stock['code']
                
                # 获取基本信息
                stock_info = {
                    'code': stock_code,
                    'name': stock['name'],
                    'industry': stock.get('industry', ''),
                    'consecutive_days': days,
                    'last_date': stock.get('last_date', '')
                }
                
                # 获取股票最近交易数据
                daily_data = self.tushare_fetcher.get_daily_data(
                    stock_code, 
                    start_date=(datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d'),
                    end_date=end_date
                )
                
                if daily_data is not None and not daily_data.empty:
                    # 计算涨停前交易特征
                    recent_data = daily_data.tail(days + 5)  # 获取涨停前后的数据
                    
                    # 计算量比（当日成交量/5日平均成交量）
                    if len(recent_data) >= 6:
                        vol_5d_avg = recent_data['volume'].iloc[:-days].mean()
                        if vol_5d_avg > 0:
                            stock_info['volume_ratio'] = recent_data['volume'].iloc[-days:].mean() / vol_5d_avg
                        else:
                            stock_info['volume_ratio'] = 1.0
                    else:
                        stock_info['volume_ratio'] = 1.0
                    
                    # 计算涨停前换手率
                    if 'turnover' in recent_data.columns:
                        stock_info['turnover_rate'] = recent_data['turnover'].iloc[-days:].mean()
                    else:
                        stock_info['turnover_rate'] = 0
                    
                    # 计算涨停前价格波动
                    if len(recent_data) >= 10:
                        pre_limit_data = recent_data.iloc[:-days]
                        if len(pre_limit_data) >= 5:
                            stock_info['pre_limit_volatility'] = pre_limit_data['close'].pct_change().std() * 100
                        else:
                            stock_info['pre_limit_volatility'] = 0
                    else:
                        stock_info['pre_limit_volatility'] = 0
                
                # 获取资金流向数据
                flow_data = self.tushare_fetcher.get_stock_fund_flow(
                    stock_code,
                    start_date=(datetime.datetime.now() - datetime.timedelta(days=10)).strftime('%Y-%m-%d'),
                    end_date=end_date
                )
                
                if flow_data is not None and not flow_data.empty:
                    recent_flow = flow_data.tail(days)
                    # 计算主力资金净流入
                    if 'net_amount' in recent_flow.columns:
                        stock_info['net_inflow'] = recent_flow['net_amount'].sum()
                    
                    # 计算大单占比
                    if all(col in recent_flow.columns for col in ['large_buy_volume', 'large_sell_volume', 'small_buy_volume', 'small_sell_volume']):
                        large_vol = recent_flow['large_buy_volume'].sum() + recent_flow['large_sell_volume'].sum()
                        small_vol = recent_flow['small_buy_volume'].sum() + recent_flow['small_sell_volume'].sum()
                        total_vol = large_vol + small_vol
                        stock_info['large_order_ratio'] = large_vol / total_vol if total_vol > 0 else 0
                
                enriched_data.append(stock_info)
            
            # 转换为DataFrame
            result_df = pd.DataFrame(enriched_data)
            
            # 缓存结果
            self.cache[cache_key] = result_df
            self.cache_expires[cache_key] = time.time()
            
            return result_df
            
        except Exception as e:
            logger.error(f"获取连续涨停股票时出错: {str(e)}")
            return pd.DataFrame()
    
    def identify_potential_breakout_stocks(self, end_date=None, threshold_score=70) -> pd.DataFrame:
        """
        识别潜在的爆发股
        基于技术指标和资金流向，发现可能即将暴涨的股票
        
        Args:
            end_date: 结束日期，格式 YYYY-MM-DD，默认为今天
            threshold_score: 最低爆发潜力得分
            
        Returns:
            DataFrame: 包含潜在爆发股及其详细信息
        """
        cache_key = f'potential_breakout_{end_date}_{threshold_score}'
        if cache_key in self.cache and time.time() - self.cache_expires.get(cache_key, 0) < self.cache_timeout:
            return self.cache[cache_key]
        
        if self.tushare_fetcher is None:
            logger.error("TuShare数据获取器未初始化")
            return pd.DataFrame()
        
        try:
            # 获取股票列表
            stocks = self.tushare_fetcher.get_stock_list()
            if stocks is None or stocks.empty:
                logger.error("获取股票列表失败")
                return pd.DataFrame()
            
            # 随机抽样以减少处理时间（实际应用中可能需要处理全部股票）
            sample_size = min(300, len(stocks))
            sampled_stocks = stocks.sample(sample_size)
            
            # 分析每只股票
            potential_stocks = []
            
            for _, stock in sampled_stocks.iterrows():
                stock_code = stock['code']
                
                # 获取股票最近交易数据
                daily_data = self.tushare_fetcher.get_daily_data(
                    stock_code, 
                    start_date=(datetime.datetime.now() - datetime.timedelta(days=60)).strftime('%Y-%m-%d'),
                    end_date=end_date
                )
                
                if daily_data is None or len(daily_data) < 30:
                    continue
                
                # 使用数据处理器计算技术指标
                if self.data_processor:
                    technical_data = self.data_processor.calculate_technical_indicators(daily_data)
                else:
                    # 简单计算必要的指标
                    daily_data['ma5'] = daily_data['close'].rolling(window=5).mean()
                    daily_data['ma10'] = daily_data['close'].rolling(window=10).mean()
                    daily_data['ma20'] = daily_data['close'].rolling(window=20).mean()
                    daily_data['ma30'] = daily_data['close'].rolling(window=30).mean()
                    daily_data['vol_ratio'] = daily_data['volume'] / daily_data['volume'].rolling(window=5).mean()
                    daily_data['price_change'] = daily_data['close'].pct_change(periods=1) * 100
                    technical_data = daily_data
                
                # 只保留最近的数据
                recent_data = technical_data.dropna().tail(10)
                if len(recent_data) < 5:
                    continue
                
                # 计算爆发潜力得分
                score = self._calculate_breakout_potential(recent_data)
                
                if score >= threshold_score:
                    # 获取资金流向
                    fund_flow = self.tushare_fetcher.get_stock_fund_flow(
                        stock_code,
                        start_date=(datetime.datetime.now() - datetime.timedelta(days=10)).strftime('%Y-%m-%d'),
                        end_date=end_date
                    )
                    
                    net_inflow = 0
                    if fund_flow is not None and not fund_flow.empty and 'net_amount' in fund_flow.columns:
                        net_inflow = fund_flow['net_amount'].sum()
                    
                    # 记录潜在爆发股
                    potential_stocks.append({
                        'code': stock_code,
                        'name': stock['name'],
                        'industry': stock.get('industry', ''),
                        'close': recent_data['close'].iloc[-1],
                        'breakout_score': score,
                        'price_change_5d': (recent_data['close'].iloc[-1] / recent_data['close'].iloc[-5] - 1) * 100 if len(recent_data) >= 5 else 0,
                        'volume_ratio': recent_data['vol_ratio'].iloc[-1] if 'vol_ratio' in recent_data.columns else 1.0,
                        'ma_trend': 'bullish' if recent_data['ma5'].iloc[-1] > recent_data['ma20'].iloc[-1] else 'bearish',
                        'net_inflow': net_inflow
                    })
            
            # 转换为DataFrame
            result_df = pd.DataFrame(potential_stocks) if potential_stocks else pd.DataFrame()
            
            # 按爆发潜力得分排序
            if not result_df.empty:
                result_df = result_df.sort_values(by='breakout_score', ascending=False)
            
            # 缓存结果
            self.cache[cache_key] = result_df
            self.cache_expires[cache_key] = time.time()
            
            return result_df
            
        except Exception as e:
            logger.error(f"识别潜在爆发股时出错: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_breakout_potential(self, data: pd.DataFrame) -> float:
        """
        计算股票的爆发潜力得分
        
        Args:
            data: 包含技术指标的DataFrame
            
        Returns:
            float: 爆发潜力得分 (0-100)
        """
        score = 0
        
        try:
            # 1. 均线多头排列得分 (0-25分)
            ma_score = 0
            if len(data) > 0 and all(col in data.columns for col in ['ma5', 'ma10', 'ma20', 'ma30']):
                last_row = data.iloc[-1]
                if last_row['ma5'] > last_row['ma10'] > last_row['ma20']:
                    ma_score += 15
                    if last_row['ma20'] > last_row['ma30']:
                        ma_score += 10
                elif last_row['ma5'] > last_row['ma10']:
                    ma_score += 8
                elif last_row['ma5'] > last_row['ma20']:
                    ma_score += 5
            
            # 2. 成交量放大得分 (0-25分)
            volume_score = 0
            if 'vol_ratio' in data.columns and len(data) >= 3:
                recent_vol_ratio = data['vol_ratio'].iloc[-3:].mean()
                if recent_vol_ratio > 2.0:
                    volume_score = 25
                elif recent_vol_ratio > 1.5:
                    volume_score = 20
                elif recent_vol_ratio > 1.2:
                    volume_score = 15
                elif recent_vol_ratio > 1.0:
                    volume_score = 10
            
            # 3. 价格动能得分 (0-25分)
            momentum_score = 0
            if 'price_change' in data.columns and len(data) >= 5:
                recent_changes = data['price_change'].iloc[-5:]
                # 计算近期涨幅
                positive_days = sum(1 for change in recent_changes if change > 0)
                # 连续上涨天数
                consecutive_up = 0
                for change in reversed(recent_changes.tolist()):
                    if change > 0:
                        consecutive_up += 1
                    else:
                        break
                
                # 基于连续上涨天数和总上涨天数评分
                if consecutive_up >= 3:
                    momentum_score += 15
                elif consecutive_up >= 2:
                    momentum_score += 10
                elif consecutive_up >= 1:
                    momentum_score += 5
                
                momentum_score += positive_days * 2  # 每个上涨日加2分
            
            # 4. 突破关键阻力位得分 (0-25分)
            breakout_score = 0
            if all(col in data.columns for col in ['close', 'high', 'ma20', 'ma30']):
                last_close = data['close'].iloc[-1]
                ma20 = data['ma20'].iloc[-1]
                ma30 = data['ma30'].iloc[-1]
                
                # 计算近期高点
                if len(data) >= 20:
                    recent_high = data['high'].iloc[-20:-1].max()
                    
                    # 突破近期高点
                    if last_close > recent_high:
                        breakout_score += 15
                    elif last_close > 0.95 * recent_high:
                        breakout_score += 8
                
                # 突破重要均线
                if last_close > ma20:
                    breakout_score += 5
                if last_close > ma30:
                    breakout_score += 5
            
            # 计算总分
            score = ma_score + volume_score + momentum_score + breakout_score
            
            # 标准化为0-100
            score = min(100, max(0, score))
            
        except Exception as e:
            logger.error(f"计算爆发潜力得分时出错: {str(e)}")
            score = 0
        
        return score
    
    def analyze_hot_sectors(self, top_n=10, consecutive_days=3, end_date=None) -> Dict:
        """
        分析热门板块
        识别近期涨停股集中的行业板块
        
        Args:
            top_n: 返回前N个热门板块
            consecutive_days: 分析的天数
            end_date: 结束日期，格式 YYYY-MM-DD，默认为今天
            
        Returns:
            Dict: 包含热门板块及相关股票的字典
        """
        cache_key = f'hot_sectors_{top_n}_{consecutive_days}_{end_date}'
        if cache_key in self.cache and time.time() - self.cache_expires.get(cache_key, 0) < self.cache_timeout:
            return self.cache[cache_key]
        
        try:
            # 获取最近N天的涨停股
            hot_stocks = []
            for day in range(1, consecutive_days + 1):
                # 获取1天、2天、3天连续涨停的股票
                limit_up_stocks = self.get_consecutive_limit_up_stocks(day, end_date)
                if not limit_up_stocks.empty:
                    hot_stocks.extend(limit_up_stocks.to_dict('records'))
            
            # 按行业分组
            industry_stats = defaultdict(list)
            for stock in hot_stocks:
                industry = stock.get('industry', '其他')
                if industry:
                    industry_stats[industry].append(stock)
            
            # 计算每个行业的热度分数
            industry_scores = []
            for industry, stocks in industry_stats.items():
                # 计算该行业的连续涨停天数总和
                total_consecutive_days = sum(stock.get('consecutive_days', 1) for stock in stocks)
                # 计算平均换手率
                avg_turnover = np.mean([stock.get('turnover_rate', 0) for stock in stocks if 'turnover_rate' in stock])
                # 计算热度分数
                heat_score = (len(stocks) * 50 + total_consecutive_days * 100) * (1 + avg_turnover * 0.5)
                
                industry_scores.append({
                    'industry': industry,
                    'stock_count': len(stocks),
                    'heat_score': heat_score,
                    'avg_consecutive_days': total_consecutive_days / len(stocks),
                    'avg_turnover': avg_turnover,
                    'stocks': stocks[:5]  # 只保留前5只股票作为示例
                })
            
            # 按热度分数排序
            industry_scores = sorted(industry_scores, key=lambda x: x['heat_score'], reverse=True)
            
            # 只返回前N个热门板块
            top_industries = industry_scores[:top_n]
            
            # 创建结果
            result = {
                'date': end_date or datetime.datetime.now().strftime('%Y-%m-%d'),
                'hot_sectors': top_industries,
                'total_sectors': len(industry_scores),
                'analysis_period': consecutive_days
            }
            
            # 缓存结果
            self.cache[cache_key] = result
            self.cache_expires[cache_key] = time.time()
            
            return result
            
        except Exception as e:
            logger.error(f"分析热门板块时出错: {str(e)}")
            return {
                'date': end_date or datetime.datetime.now().strftime('%Y-%m-%d'),
                'hot_sectors': [],
                'total_sectors': 0,
                'analysis_period': consecutive_days,
                'error': str(e)
            }
    
    def predict_limit_up_continuation(self, ts_code: str, end_date=None) -> Dict:
        """
        预测涨停板是否会继续
        分析个股的涨停特征，预测明日是否会继续涨停
        
        Args:
            ts_code: 股票代码
            end_date: 结束日期，格式 YYYY-MM-DD，默认为今天
            
        Returns:
            Dict: 包含预测结果和相关特征
        """
        if self.tushare_fetcher is None:
            logger.error("TuShare数据获取器未初始化")
            return {'continuation_probability': 0, 'features': {}}
        
        try:
            # 获取股票最近交易数据
            daily_data = self.tushare_fetcher.get_daily_data(
                ts_code, 
                start_date=(datetime.datetime.now() - datetime.timedelta(days=60)).strftime('%Y-%m-%d'),
                end_date=end_date
            )
            
            if daily_data is None or daily_data.empty or len(daily_data) < 30:
                logger.warning(f"获取 {ts_code} 的交易数据失败或数据不足")
                return {'continuation_probability': 0, 'features': {}}
            
            # 判断最新交易日是否涨停
            daily_data['pct_change'] = daily_data['close'].pct_change() * 100
            daily_data['is_limit_up'] = daily_data['pct_change'] > 9.5
            
            if not daily_data['is_limit_up'].iloc[-1]:
                logger.info(f"{ts_code} 最新交易日未涨停，无需预测")
                return {'continuation_probability': 0, 'features': {}}
            
            # 获取资金流向数据
            flow_data = self.tushare_fetcher.get_stock_fund_flow(
                ts_code,
                start_date=(datetime.datetime.now() - datetime.timedelta(days=10)).strftime('%Y-%m-%d'),
                end_date=end_date
            )
            
            # 计算特征
            features = {}
            
            # 1. 计算连续涨停天数
            consecutive_days = 0
            for i in range(len(daily_data) - 1, -1, -1):
                if daily_data['is_limit_up'].iloc[i]:
                    consecutive_days += 1
                else:
                    break
            features['consecutive_days'] = consecutive_days
            
            # 2. 计算换手率特征
            if 'turnover' in daily_data.columns:
                features['turnover_rate'] = daily_data['turnover'].iloc[-1]
                features['avg_turnover_5d'] = daily_data['turnover'].iloc[-5:].mean() if len(daily_data) >= 5 else 0
            else:
                features['turnover_rate'] = 0
                features['avg_turnover_5d'] = 0
            
            # 3. 计算量比
            features['volume_ratio'] = daily_data['volume'].iloc[-1] / daily_data['volume'].iloc[-6:-1].mean() if len(daily_data) >= 6 else 1
            
            # 4. 计算北向资金
            features['north_money_flow'] = 0  # 默认值
            
            # 5. 计算行业热度 (简化版)
            features['industry_heat'] = 0  # 默认值
            
            # 6. 计算机构参与度
            if flow_data is not None and not flow_data.empty:
                if all(col in flow_data.columns for col in ['large_buy_volume', 'large_sell_volume', 'small_buy_volume', 'small_sell_volume']):
                    large_vol = flow_data['large_buy_volume'].iloc[-1] + flow_data['large_sell_volume'].iloc[-1]
                    small_vol = flow_data['small_buy_volume'].iloc[-1] + flow_data['small_sell_volume'].iloc[-1]
                    total_vol = large_vol + small_vol
                    features['institutional_participation'] = large_vol / total_vol if total_vol > 0 else 0
                else:
                    features['institutional_participation'] = 0
            else:
                features['institutional_participation'] = 0
            
            # 计算综合得分
            score = 0
            for feature, value in features.items():
                if feature in self.feature_weights:
                    normalized_value = min(1.0, value / 10.0) if feature == 'consecutive_days' else min(1.0, value)
                    score += normalized_value * self.feature_weights[feature]
            
            # 转换为继续涨停的概率
            continuation_probability = score * 100
            
            return {
                'code': ts_code,
                'date': daily_data['date'].iloc[-1] if 'date' in daily_data.columns else end_date,
                'continuation_probability': continuation_probability,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"预测 {ts_code} 涨停延续性时出错: {str(e)}")
            return {'continuation_probability': 0, 'features': {}, 'error': str(e)}
    
    def generate_hot_stock_report(self, top_n=20, end_date=None) -> Dict:
        """
        生成暴涨股分析报告
        综合分析连续涨停股、潜在爆发股和热门板块
        
        Args:
            top_n: 每类返回的股票数量
            end_date: 结束日期，格式 YYYY-MM-DD，默认为今天
            
        Returns:
            Dict: 包含完整分析报告的字典
        """
        try:
            report = {
                'date': end_date or datetime.datetime.now().strftime('%Y-%m-%d'),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'report_type': '暴涨股捕捉分析报告'
            }
            
            # 1. 获取连续涨停股
            consecutive_1d = self.get_consecutive_limit_up_stocks(1, end_date)
            consecutive_2d = self.get_consecutive_limit_up_stocks(2, end_date)
            consecutive_3d = self.get_consecutive_limit_up_stocks(3, end_date)
            
            report['consecutive_limit_up'] = {
                '1day': consecutive_1d.head(top_n).to_dict('records') if not consecutive_1d.empty else [],
                '2days': consecutive_2d.head(top_n).to_dict('records') if not consecutive_2d.empty else [],
                '3days': consecutive_3d.head(top_n).to_dict('records') if not consecutive_3d.empty else [],
                'summary': {
                    'total_1day': len(consecutive_1d),
                    'total_2days': len(consecutive_2d),
                    'total_3days': len(consecutive_3d)
                }
            }
            
            # 2. 获取潜在爆发股
            potential_breakout = self.identify_potential_breakout_stocks(end_date)
            report['potential_breakout'] = {
                'stocks': potential_breakout.head(top_n).to_dict('records') if not potential_breakout.empty else [],
                'total': len(potential_breakout)
            }
            
            # 3. 获取热门板块
            hot_sectors = self.analyze_hot_sectors(10, 3, end_date)
            report['hot_sectors'] = hot_sectors
            
            # 4. 为每个连续2天及以上涨停的股票预测续板概率
            continuation_predictions = []
            for stock in (report['consecutive_limit_up']['2days'] + report['consecutive_limit_up']['3days']):
                prediction = self.predict_limit_up_continuation(stock['code'], end_date)
                if 'continuation_probability' in prediction and prediction['continuation_probability'] > 0:
                    continuation_predictions.append({
                        'code': stock['code'],
                        'name': stock['name'],
                        'industry': stock.get('industry', ''),
                        'consecutive_days': stock.get('consecutive_days', 0),
                        'continuation_probability': prediction['continuation_probability'],
                        'key_features': {k: v for k, v in prediction.get('features', {}).items() 
                                        if k in ['turnover_rate', 'volume_ratio', 'institutional_participation']}
                    })
            
            # 按续板概率排序
            continuation_predictions = sorted(continuation_predictions, 
                                             key=lambda x: x['continuation_probability'], 
                                             reverse=True)
            
            report['continuation_predictions'] = continuation_predictions[:top_n]
            
            # 5. 生成市场热度指标
            market_heat = 0
            if report['consecutive_limit_up']['summary']['total_1day'] > 0:
                # 基于涨停股数量的市场热度
                limit_up_ratio = min(1.0, report['consecutive_limit_up']['summary']['total_1day'] / 100)
                consecutive_ratio = (report['consecutive_limit_up']['summary']['total_2days'] * 2 + 
                                   report['consecutive_limit_up']['summary']['total_3days'] * 3) / 50
                
                market_heat = (limit_up_ratio * 40 + consecutive_ratio * 60) * 100
                market_heat = min(100, max(0, market_heat))
            
            report['market_indicators'] = {
                'market_heat': market_heat,
                'heat_level': '火爆' if market_heat >= 80 else 
                              '热' if market_heat >= 60 else 
                              '温和' if market_heat >= 40 else 
                              '冷' if market_heat >= 20 else '极冷',
                'consecutive_stocks_count': {
                    '1day': report['consecutive_limit_up']['summary']['total_1day'],
                    '2days': report['consecutive_limit_up']['summary']['total_2days'],
                    '3days': report['consecutive_limit_up']['summary']['total_3days']
                }
            }
            
            # 6. 保存报告
            self._save_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"生成暴涨股分析报告时出错: {str(e)}")
            return {
                'date': end_date or datetime.datetime.now().strftime('%Y-%m-%d'),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'report_type': '暴涨股捕捉分析报告',
                'error': str(e)
            }
    
    def _save_report(self, report: Dict) -> None:
        """
        保存分析报告
        
        Args:
            report: 分析报告字典
        """
        try:
            # 创建时间戳文件名
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(self.results_dir, f'hot_stock_report_{timestamp}.json')
            
            # 保存为JSON
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"暴涨股分析报告已保存到 {file_path}")
            
        except Exception as e:
            logger.error(f"保存暴涨股分析报告时出错: {str(e)}") 