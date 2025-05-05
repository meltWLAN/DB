#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tushare as ts
import concurrent.futures
import json
from typing import Dict, Any, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置Tushare API Key
try:
    # 设置用户提供的Tushare Token作为备选数据源
    TUSHARE_TOKEN = '0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10'
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    logger.info("Tushare初始化成功")
except Exception as e:
    logger.error(f"初始化Tushare API失败: {e}")
    pro = None

# 初始化JoinQuant (优先使用)
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.config import DATA_SOURCE_CONFIG
    import jqdatasdk as jq
    
    # 获取JoinQuant配置
    jq_config = DATA_SOURCE_CONFIG.get('joinquant', {})
    jq_username = jq_config.get('username', '')
    jq_password = jq_config.get('password', '')
    
    if jq_username and jq_password:
        jq.auth(jq_username, jq_password)
        jq_available = True
        logger.info("JoinQuant初始化成功")
    else:
        jq_available = False
        logger.warning("JoinQuant账号未配置，将使用Tushare作为数据源")
except Exception as e:
    logger.error(f"初始化JoinQuant失败: {e}")
    jq_available = False

# 创建结果目录
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'limit_up_capture')
os.makedirs(RESULTS_DIR, exist_ok=True)

class LimitUpCaptureStrategy:
    """涨停和大幅上涨股票捕捉策略"""
    
    def __init__(self, start_date=None, end_date=None, results_dir=None):
        """初始化策略
        
        Args:
            start_date (str, optional): 分析起始日期，默认为None，使用当前日期
            end_date (str, optional): 分析结束日期，默认为None，使用当前日期
            results_dir (str, optional): 结果保存目录，默认为None，使用当前目录下的results/limit_up_capture
        """
        # 设置日期范围
        today = datetime.now().strftime('%Y%m%d')
        self.start_date = start_date or (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        self.end_date = end_date or today
        self.today = today
        
        # 初始化数据结构
        self.consecutive_limit_up_stocks = {
            '2days': [],
            '3days': [],
            '5days': []
        }
        self.high_momentum_stocks = []
        self.breakout_stocks = []
        
        # 用于存储历史趋势分析结果
        self.trend_continuation_probability = {}
        
        # 新增机构资金流向数据
        self.institution_capital_flow = {}
        
        # 新增信号质量评分
        self.signal_quality_scores = {}
        
        # 新增实时行情数据质量检查结果
        self.data_quality_check = {}
        
        # 设置结果保存目录
        if results_dir:
            self.results_dir = results_dir
        else:
            self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/limit_up_capture")
            os.makedirs(self.results_dir, exist_ok=True)
            
        logger.info(f"策略初始化完成，分析日期范围: {self.start_date} - {self.end_date}, 结果将保存至: {self.results_dir}")
        
        # 尝试初始化数据源
        self.init_data_source()
        
        # 添加数据缓存
        self.data_cache = {}
        self.cache_timeout = 300  # 缓存超时时间（秒）
        self.cache_timestamps = {}
        
        # 优化线程池配置
        self.max_workers = min(32, (os.cpu_count() or 1) * 4)  # 根据CPU核心数优化线程数
        self.chunk_size = 100  # 数据分块处理大小
        
        # 添加性能监控
        self.performance_metrics = {
            'data_fetch_time': [],
            'analysis_time': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def init_data_source(self):
        """初始化数据源"""
        # 优先使用JoinQuant
        if jq_available:
            try:
                stocks = jq.get_all_securities(['stock'])
                # 转换为与Tushare相似的格式
                stocks_df = pd.DataFrame({
                    'ts_code': stocks.index,
                    'symbol': stocks.index.str.split('.').str[0],
                    'name': stocks['display_name'],
                    'list_date': stocks['start_date'].astype(str)
                })
                logger.info(f"使用JoinQuant获取到 {len(stocks_df)} 只股票")
                self.stock_list = stocks_df
            except Exception as e:
                logger.error(f"使用JoinQuant获取股票列表失败: {e}")
        
        # 备选使用Tushare
        try:
            stocks = pro.stock_basic(exchange='', list_status='L')
            logger.info(f"使用Tushare获取到 {len(stocks)} 只股票")
            self.stock_list = stocks
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            self.stock_list = None
            
    def _get_cached_data(self, key):
        """从缓存获取数据"""
        if key in self.data_cache:
            timestamp = self.cache_timestamps.get(key)
            if timestamp and (datetime.now() - timestamp).seconds < self.cache_timeout:
                self.performance_metrics['cache_hits'] += 1
                return self.data_cache[key]
        self.performance_metrics['cache_misses'] += 1
        return None
        
    def _set_cached_data(self, key, data):
        """设置缓存数据"""
        self.data_cache[key] = data
        self.cache_timestamps[key] = datetime.now()
        
    def fetch_daily_data(self, ts_code, start_date, end_date):
        """获取单个股票的日线数据（优化版）"""
        cache_key = f"{ts_code}_{start_date}_{end_date}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
            
        start_time = datetime.now()
        try:
            df = super().fetch_daily_data(ts_code, start_date, end_date)
            if df is not None:
                self._set_cached_data(cache_key, df)
            
            self.performance_metrics['data_fetch_time'].append(
                (datetime.now() - start_time).total_seconds()
            )
            return df
        except Exception as e:
            logger.error(f"获取股票 {ts_code} 数据失败: {e}")
            return None
            
    def fetch_limit_list(self, trade_date):
        """获取某日涨停股票列表"""
        # JoinQuant没有直接获取涨停列表的接口，需要间接计算
        if jq_available:
            try:
                # 获取所有A股代码
                stocks = jq.get_all_securities(['stock'])
                stock_codes = list(stocks.index)
                
                # 获取当日行情
                df = jq.get_price(stock_codes, count=2, end_date=trade_date, 
                               fields=['open', 'close', 'high', 'low', 'volume', 'money'])
                
                # 计算涨跌幅
                df = df.reset_index()
                df = df.pivot(index='time', columns='code', values='close')
                
                # 计算涨跌幅
                daily_returns = df.pct_change().iloc[-1] * 100
                
                # 筛选涨停股票 (涨幅>=9.5%)
                limit_up_series = daily_returns[daily_returns >= 9.5]
                
                # 转换为与Tushare相似的格式
                limit_up_list = pd.DataFrame({
                    'ts_code': limit_up_series.index,
                    'trade_date': trade_date,
                    'pct_chg': limit_up_series.values
                })
                
                logger.info(f"使用JoinQuant分析得到 {len(limit_up_list)} 只涨停股票")
                return limit_up_list
            except Exception as e:
                logger.error(f"使用JoinQuant获取涨停股票列表失败: {e}")
        
        # 备选使用Tushare
        try:
            # 获取涨停股票数据
            limit_list = pro.limit_list(trade_date=trade_date, limit_type='U')
            logger.info(f"使用Tushare获取到 {len(limit_list)} 只涨停股票")
            return limit_list
        except Exception as e:
            logger.error(f"获取涨停股票列表失败: {e}")
            return None
            
    def _monitor_real_time_signals(self, stock_data: pd.DataFrame, window: int = 5) -> Dict[str, Any]:
        """监控实时信号变化
        
        Args:
            stock_data (pd.DataFrame): 股票数据
            window (int): 监控窗口大小，默认为5分钟

        Returns:
            Dict[str, Any]: 实时信号监控结果
        """
        monitoring_result = {
            'signal_change': False,
            'price_momentum': 0,
            'volume_surge': False,
            'breakout_alert': False,
            'alert_level': 0  # 0: 无警报, 1: 低级警报, 2: 中级警报, 3: 高级警报
        }
        
        try:
            # 获取最近window个周期的数据
            recent_data = stock_data.tail(window)
            
            # 计算价格动量
            price_change = recent_data['close'].pct_change()
            monitoring_result['price_momentum'] = price_change.mean() * 100
            
            # 检测成交量突变
            volume_mean = stock_data['volume'].rolling(window=20).mean().iloc[-1]
            recent_volume = recent_data['volume'].mean()
            if recent_volume > volume_mean * 2:
                monitoring_result['volume_surge'] = True
            
            # 检测突破信号
            upper_band = recent_data['bb_upper'].iloc[-1]
            lower_band = recent_data['bb_lower'].iloc[-1]
            last_close = recent_data['close'].iloc[-1]
            
            if last_close > upper_band:
                monitoring_result['breakout_alert'] = True
                monitoring_result['alert_level'] = 2
            elif last_close < lower_band:
                monitoring_result['breakout_alert'] = True
                monitoring_result['alert_level'] = 1
            
            # 综合判断信号变化
            if (abs(monitoring_result['price_momentum']) > 2 and  # 价格变动超过2%
                monitoring_result['volume_surge'] and  # 成交量突变
                monitoring_result['breakout_alert']):  # 出现突破信号
                monitoring_result['signal_change'] = True
                monitoring_result['alert_level'] = 3
            
            # 添加DMI指标确认
            if 'adx' in recent_data.columns:
                last_adx = recent_data['adx'].iloc[-1]
                last_di_plus = recent_data['di_plus'].iloc[-1]
                last_di_minus = recent_data['di_minus'].iloc[-1]
                
                if last_adx > 25:  # 趋势强度确认
                    if last_di_plus > last_di_minus and monitoring_result['price_momentum'] > 0:
                        monitoring_result['alert_level'] = min(3, monitoring_result['alert_level'] + 1)
                    elif last_di_minus > last_di_plus and monitoring_result['price_momentum'] < 0:
                        monitoring_result['alert_level'] = min(3, monitoring_result['alert_level'] + 1)
            
            # 添加MACD确认
            if all(x in recent_data.columns for x in ['macd', 'signal']):
                if (recent_data['macd'].iloc[-2] <= recent_data['signal'].iloc[-2] and
                    recent_data['macd'].iloc[-1] > recent_data['signal'].iloc[-1]):
                    # MACD金叉确认
                    monitoring_result['alert_level'] = min(3, monitoring_result['alert_level'] + 1)
            
            return monitoring_result
            
        except Exception as e:
            logger.error(f"监控实时信号时发生错误: {e}")
            return monitoring_result

    def analyze_stock(self, ts_code: str, days: int = 20) -> Dict[str, Any]:
        """分析单只股票
        
        Args:
            ts_code (str): 股票代码
            days (int, optional): 分析天数. 默认为 20.

        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            # 获取股票数据
            df = self.fetch_daily_data(ts_code, self.start_date, self.end_date)
            if df is None or len(df) < days:
                return None
            
            # 检查数据质量
            quality_check = self._check_data_quality(df)
            if quality_check['quality_score'] < 0.7:
                logger.warning(f"{ts_code} 数据质量不佳，跳过分析")
                return None
            
            # 添加技术指标
            df = TechnicalIndicators.add_all_indicators(df)
            
            # 获取技术指标信号
            technical_signals = TechnicalIndicators.get_indicator_signals(df)
            
            # 计算信号可信度
            signal_confidence = self._calculate_signal_confidence(df, technical_signals)
            
            # 获取最新的机构资金流向
            inst_flow = self.fetch_institution_flow(ts_code)
            
            # 监控实时信号变化
            real_time_signals = self._monitor_real_time_signals(df)
            
            # 如果出现高级别实时信号，提高信号可信度
            if real_time_signals['alert_level'] >= 2:
                signal_confidence = min(1.0, signal_confidence * 1.2)
            
            analysis_result = {
                'technical_signals': technical_signals,
                'signal_confidence': signal_confidence,
                'data_quality': quality_check,
                'institution_flow': inst_flow,
                'last_price': df['close'].iloc[-1],
                'price_change': df['close'].pct_change().iloc[-1] * 100,
                'volume_ratio': df['volume'].iloc[-1] / df['volume'].mean(),
                'momentum_score': technical_signals.get('signal_strength', 0),
                'real_time_signals': real_time_signals
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"分析股票 {ts_code} 时发生错误: {e}")
            return None

    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查数据质量
        
        Args:
            df (pd.DataFrame): 股票数据

        Returns:
            Dict[str, Any]: 数据质量检查结果
        """
        quality_check = {
            'missing_data': False,
            'abnormal_price': False,
            'abnormal_volume': False,
            'data_delay': False,
            'quality_score': 1.0
        }
        
        # 检查缺失数据
        if df.isnull().any().any():
            quality_check['missing_data'] = True
            quality_check['quality_score'] *= 0.8
        
        # 检查异常价格
        price_std = df['close'].std()
        price_mean = df['close'].mean()
        if any(abs(df['close'] - price_mean) > 3 * price_std):
            quality_check['abnormal_price'] = True
            quality_check['quality_score'] *= 0.9
        
        # 检查异常成交量
        volume_std = df['volume'].std()
        volume_mean = df['volume'].mean()
        if any(abs(df['volume'] - volume_mean) > 3 * volume_std):
            quality_check['abnormal_volume'] = True
            quality_check['quality_score'] *= 0.9
        
        # 检查数据延迟
        if datetime.now().strftime('%Y%m%d') == self.today:
            last_time = pd.to_datetime(df.index[-1])
            if (datetime.now() - last_time).seconds > 60:
                quality_check['data_delay'] = True
                quality_check['quality_score'] *= 0.7
        
        return quality_check

    def _calculate_signal_confidence(self, stock_data: pd.DataFrame, technical_signals: Dict[str, Any]) -> float:
        """计算信号可信度
        
        Args:
            stock_data (pd.DataFrame): 股票数据
            technical_signals (Dict[str, Any]): 技术指标信号

        Returns:
            float: 信号可信度得分 (0-1)
        """
        confidence_score = 1.0
        
        # 考虑技术指标的综合信号强度
        signal_strength = abs(technical_signals.get('signal_strength', 0))
        confidence_score *= (0.5 + 0.5 * signal_strength)
        
        # 考虑成交量确认
        volume_signal = technical_signals.get('volume_signals', {}).get('volume_trend', '')
        if volume_signal == '放量':
            confidence_score *= 1.2
        elif volume_signal == '缩量':
            confidence_score *= 0.8
        
        # 考虑趋势确认
        trend_prob = self.trend_continuation_probability.get(stock_data.index[-1], 0.5)
        confidence_score *= (0.5 + 0.5 * trend_prob)
        
        # 考虑机构资金流向
        inst_flow = self.institution_capital_flow.get(stock_data.index[-1], 0)
        if inst_flow > 0:
            confidence_score *= 1.2
        elif inst_flow < 0:
            confidence_score *= 0.8
        
        # 考虑数据质量
        quality_score = self._check_data_quality(stock_data)['quality_score']
        confidence_score *= quality_score
        
        return min(confidence_score, 1.0)

    def analyze_consecutive_limit_up(self):
        """分析连续涨停股票"""
        # 获取今日涨停股票
        limit_list = self.fetch_limit_list(self.today)
        if limit_list is None or limit_list.empty:
            logger.warning("获取涨停股票列表失败或今日无涨停股票")
            return
            
        logger.info(f"今日共有 {len(limit_list)} 只涨停股票")
        self.limit_up_stocks = limit_list
        
        # 保存今日涨停股票列表
        self.limit_up_stocks.to_csv(
            os.path.join(self.results_dir, f"limit_up_stocks_{self.today}.csv"), 
            index=False
        )
        
        # 并行处理多只股票
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            stock_analysis_futures = {}
            for _, row in limit_list.iterrows():
                ts_code = row['ts_code']
                future = executor.submit(self.analyze_stock, ts_code)
                stock_analysis_futures[future] = ts_code
                
            # 处理结果
            for future in concurrent.futures.as_completed(stock_analysis_futures):
                ts_code = stock_analysis_futures[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        # 获取最近一天数据
                        latest_data = data.iloc[-1]
                        
                        # 判断连续涨停天数
                        consecutive_days = latest_data['consecutive_limit_up_days']
                        
                        if consecutive_days >= 2 and consecutive_days < 3:
                            self.consecutive_limit_up_stocks['2days'].append({
                                'ts_code': ts_code,
                                'consecutive_days': consecutive_days,
                                'close': latest_data['close'],
                                'vol_ratio': latest_data['vol_ratio'],
                                'momentum_5d': latest_data['momentum_5d']
                            })
                        elif consecutive_days >= 3 and consecutive_days < 4:
                            self.consecutive_limit_up_stocks['3days'].append({
                                'ts_code': ts_code,
                                'consecutive_days': consecutive_days,
                                'close': latest_data['close'],
                                'vol_ratio': latest_data['vol_ratio'],
                                'momentum_5d': latest_data['momentum_5d']
                            })
                        elif consecutive_days >= 4:
                            self.consecutive_limit_up_stocks['5days'].append({
                                'ts_code': ts_code,
                                'consecutive_days': consecutive_days,
                                'close': latest_data['close'],
                                'vol_ratio': latest_data['vol_ratio'],
                                'momentum_5d': latest_data['momentum_5d']
                            })
                except Exception as e:
                    logger.error(f"处理股票 {ts_code} 时发生错误: {e}")
                    
        # 保存连续涨停股票结果
        for days, stocks in self.consecutive_limit_up_stocks.items():
            if stocks:
                df = pd.DataFrame(stocks)
                df.to_csv(
                    os.path.join(self.results_dir, f"consecutive_limit_up_{days}_{self.today}.csv"),
                    index=False
                )
                logger.info(f"找到 {len(stocks)} 只{days}连续涨停股票")
                
    def analyze_high_momentum_stocks(self, days=20, top_n=50):
        """分析高动量股票（近期涨幅较大但未必涨停的股票）"""
        # 获取所有股票列表
        stocks = self.stock_list
        if stocks is None:
            logger.error("无法获取股票列表")
            return
            
        # 计算起止日期
        end_date = self.today
        start_date = (datetime.now() - timedelta(days=days+10)).strftime('%Y%m%d')
        
        # 获取所有股票的日线数据
        all_stock_data = []
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.fetch_daily_data, code, start_date, end_date): code 
                      for code in stocks['ts_code'].tolist()[:300]}  # 为了加快处理，可以先取前300只
            
            for future in concurrent.futures.as_completed(futures):
                code = futures[future]
                try:
                    data = future.result()
                    if data is not None and len(data) >= days:
                        # 计算近期涨幅
                        data = data.sort_values('trade_date')
                        latest_close = data.iloc[-1]['close']
                        days_ago_close = data.iloc[-days]['close']
                        momentum = latest_close / days_ago_close - 1
                        
                        # 计算5日、10日均线多头排列
                        data['ma5'] = data['close'].rolling(5).mean()
                        data['ma10'] = data['close'].rolling(10).mean()
                        data['ma20'] = data['close'].rolling(20).mean()
                        latest = data.iloc[-1]
                        
                        # 判断均线多头排列
                        ma_alignment = (latest['ma5'] > latest['ma10']) and (latest['ma10'] > latest['ma20'])
                        
                        # 计算成交量变化
                        data['vol_ratio'] = data['vol'] / data['vol'].rolling(5).mean()
                        vol_change = data.iloc[-1]['vol_ratio']
                        
                        all_stock_data.append({
                            'ts_code': code,
                            'momentum': momentum * 100,  # 转为百分比
                            'ma_alignment': ma_alignment,
                            'vol_ratio': vol_change,
                            'close': latest_close
                        })
                except Exception as e:
                    logger.error(f"处理股票 {code} 动量时发生错误: {e}")
        
        # 创建数据框并排序
        if all_stock_data:
            df = pd.DataFrame(all_stock_data)
            df = df.sort_values('momentum', ascending=False)
            
            # 选择前N只高动量股票
            high_momentum = df.head(top_n)
            self.high_momentum_stocks = high_momentum
            
            # 保存结果
            high_momentum.to_csv(
                os.path.join(self.results_dir, f"high_momentum_stocks_{days}d_{self.today}.csv"),
                index=False
            )
            logger.info(f"找到 {len(high_momentum)} 只高动量股票 (近{days}日)")
            
    def analyze_breakout_stocks(self, top_n=50):
        """分析突破重要阻力位的股票"""
        # 获取所有股票列表
        stocks = self.stock_list
        if stocks is None:
            logger.error("无法获取股票列表")
            return
            
        # 计算起止日期
        end_date = self.today
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
        
        # 获取所有股票的日线数据
        breakout_stock_data = []
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.fetch_daily_data, code, start_date, end_date): code 
                      for code in stocks['ts_code'].tolist()[:300]}  # 为了加快处理，可以先取前300只
            
            for future in concurrent.futures.as_completed(futures):
                code = futures[future]
                try:
                    data = future.result()
                    if data is not None and len(data) >= 60:
                        # 计算技术指标
                        # 1. 新高突破
                        data['break_high_20d'] = data['close'] > data['close'].rolling(20).max().shift(1)
                        data['break_high_60d'] = data['close'] > data['close'].rolling(60).max().shift(1)
                        
                        # 2. 成交量突破
                        data['vol_ratio'] = data['vol'] / data['vol'].rolling(5).mean()
                        
                        # 获取最新一天数据
                        latest = data.iloc[-1]
                        
                        # 判断是否为突破股
                        is_breakout_20d = latest['break_high_20d']
                        is_breakout_60d = latest['break_high_60d']
                        
                        # 判断是否为强势突破（同时放量）
                        is_strong_breakout = (is_breakout_20d or is_breakout_60d) and (latest['vol_ratio'] > 1.5)
                        
                        if is_breakout_20d or is_breakout_60d:
                            breakout_stock_data.append({
                                'ts_code': code,
                                'break_high_20d': is_breakout_20d,
                                'break_high_60d': is_breakout_60d,
                                'vol_ratio': latest['vol_ratio'],
                                'is_strong_breakout': is_strong_breakout,
                                'close': latest['close']
                            })
                except Exception as e:
                    logger.error(f"处理股票 {code} 突破时发生错误: {e}")
        
        # 创建数据框并排序
        if breakout_stock_data:
            df = pd.DataFrame(breakout_stock_data)
            
            # 优先选择60日突破 + 强势突破的股票
            df['score'] = (df['break_high_60d'] * 5 + 
                          df['break_high_20d'] * 3 + 
                          df['is_strong_breakout'] * 4)
            
            df = df.sort_values('score', ascending=False)
            
            # 选择前N只突破股票
            breakout_stocks = df.head(top_n)
            self.breakout_stocks = breakout_stocks
            
            # 保存结果
            breakout_stocks.to_csv(
                os.path.join(self.results_dir, f"breakout_stocks_{self.today}.csv"),
                index=False
            )
            logger.info(f"找到 {len(breakout_stocks)} 只突破股票")
            
    def analyze_trend_continuation(self, lookback_days=180):
        """分析历史涨停后的趋势延续概率
        
        Args:
            lookback_days (int): 回溯历史数据的天数
        """
        logger.info(f"开始分析历史涨停趋势延续概率 (回溯{lookback_days}天)")
        
        # 计算历史日期范围
        end_date = self.today
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y%m%d')
        
        # 获取历史涨停数据
        all_limit_up_stocks = []
        
        # 按日期逐步获取历史涨停数据
        current_date = datetime.strptime(start_date, '%Y%m%d')
        end_date_obj = datetime.strptime(end_date, '%Y%m%d')
        
        sample_dates = []
        # 为了提高效率，每隔5天采样一次
        while current_date <= end_date_obj:
            # 跳过周末
            if current_date.weekday() < 5:  # 0-4 表示周一至周五
                sample_dates.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=5)
        
        # 初始化统计数据
        continuation_stats = {
            'single_limit_up': {'continued': 0, 'total': 0},
            'double_limit_up': {'continued': 0, 'total': 0},
            'triple_limit_up': {'continued': 0, 'total': 0}
        }
        
        # 过滤出交易日
        for date in sample_dates:
            try:
                # 获取当天涨停股票
                limit_list = self.fetch_limit_list(date)
                if limit_list is None or limit_list.empty:
                    continue
                    
                # 记录这一天的涨停股
                for _, row in limit_list.iterrows():
                    ts_code = row['ts_code']
                    all_limit_up_stocks.append({
                        'date': date,
                        'ts_code': ts_code
                    })
                    
                    # 获取该股票后续5天的表现
                    next_date = (datetime.strptime(date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
                    end_check_date = (datetime.strptime(date, '%Y%m%d') + timedelta(days=5)).strftime('%Y%m%d')
                    
                    # 获取后续行情数据
                    future_data = self.fetch_daily_data(ts_code, next_date, end_check_date)
                    
                    if future_data is None or len(future_data) < 1:
                        continue
                    
                    # 分析是否第二天也涨停
                    if len(future_data) >= 1:
                        pct_chg = future_data.iloc[0]['pct_chg']
                        if pct_chg >= 9.5:  # 如果第二天也涨停
                            continuation_stats['single_limit_up']['continued'] += 1
                        continuation_stats['single_limit_up']['total'] += 1
                        
                        # 分析连续两天涨停后第三天是否涨停
                        if pct_chg >= 9.5 and len(future_data) >= 2:
                            pct_chg_day3 = future_data.iloc[1]['pct_chg']
                            if pct_chg_day3 >= 9.5:  # 如果第三天也涨停
                                continuation_stats['double_limit_up']['continued'] += 1
                            continuation_stats['double_limit_up']['total'] += 1
                            
                            # 分析连续三天涨停后第四天是否涨停
                            if pct_chg_day3 >= 9.5 and len(future_data) >= 3:
                                pct_chg_day4 = future_data.iloc[2]['pct_chg']
                                if pct_chg_day4 >= 9.5:  # 如果第四天也涨停
                                    continuation_stats['triple_limit_up']['continued'] += 1
                                continuation_stats['triple_limit_up']['total'] += 1
            except Exception as e:
                logger.error(f"处理日期 {date} 的涨停趋势分析时出错: {e}")
        
        # 计算各种情况下的趋势延续概率
        for key, stats in continuation_stats.items():
            if stats['total'] > 0:
                probability = stats['continued'] / stats['total'] * 100
                self.trend_continuation_probability[key] = probability
                logger.info(f"{key} 趋势延续概率: {probability:.2f}% ({stats['continued']}/{stats['total']})")
            else:
                self.trend_continuation_probability[key] = 0
                
        return self.trend_continuation_probability
        
    def fetch_institution_flow(self, ts_code):
        """获取机构资金流向数据
        
        Args:
            ts_code (str): 股票代码
            
        Returns:
            float: 机构资金净流入比例，正值表示净流入，负值表示净流出
        """
        try:
            # 如果使用JoinQuant
            if jq_available:
                # 转换代码格式
                if '.' not in ts_code:
                    if ts_code.startswith('6'):
                        jq_code = ts_code + '.XSHG'
                    else:
                        jq_code = ts_code + '.XSHE'
                else:
                    jq_code = ts_code
                
                # 获取最近5个交易日的资金流向数据
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
                
                try:
                    # 获取资金流数据
                    df = jq.get_money_flow([jq_code], start_date=start_date, end_date=end_date)
                    
                    if df is not None and not df.empty:
                        # 计算机构资金流向
                        # 大单资金净流入 + 超大单资金净流入
                        df['net_amount_main'] = df['net_amount_l'] + df['net_amount_xl']
                        # 总成交额
                        df['total_amount'] = df['money']
                        # 机构资金净流入比例
                        df['main_net_pct'] = df['net_amount_main'] / df['total_amount']
                        
                        # 计算最近5个交易日的平均值
                        avg_flow = df['main_net_pct'].mean()
                        
                        return avg_flow
                except Exception as e:
                    logger.error(f"获取 {ts_code} 的JoinQuant资金流向数据失败: {e}")
            
            # 如果使用Tushare
            if pro is not None:
                # 获取最近5个交易日的资金流向数据
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
                
                try:
                    # 获取资金流数据
                    df = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
                    
                    if df is not None and not df.empty:
                        # 计算机构资金流向 (买入额 - 卖出额) / 成交总额
                        df['net_mf_amount'] = df['buy_elg_amount'] + df['buy_lg_amount'] - df['sell_elg_amount'] - df['sell_lg_amount']
                        df['net_pct'] = df['net_mf_amount'] / df['amount']
                        
                        # 计算最近交易日的平均值
                        avg_flow = df['net_pct'].mean()
                        
                        return avg_flow
                except Exception as e:
                    logger.error(f"获取 {ts_code} 的Tushare资金流向数据失败: {e}")
            
            return 0.0
        except Exception as e:
            logger.error(f"获取机构资金流向数据异常: {e}")
            return 0.0
            
    def calculate_prediction_score(self, stock_data):
        """计算股票后续涨停或大幅上涨的预测得分
        
        Args:
            stock_data (dict): 股票数据，包含各种技术指标
            
        Returns:
            float: 预测得分，越高表示越有可能后续涨停或大幅上涨
        """
        score = 0
        
        # 基础分 - 已有的连续涨停天数
        consecutive_days = stock_data.get('consecutive_days', 1)
        if consecutive_days >= 2:
            # 根据历史统计计算加分
            if consecutive_days == 2 and 'double_limit_up' in self.trend_continuation_probability:
                # 二连板后的涨停概率
                continuation_prob = self.trend_continuation_probability['double_limit_up']
                score += 20 + continuation_prob / 2  # 基础分20，加上概率加成
            elif consecutive_days >= 3 and 'triple_limit_up' in self.trend_continuation_probability:
                # 三连板后的涨停概率
                continuation_prob = self.trend_continuation_probability['triple_limit_up']
                score += 30 + continuation_prob  # 基础分30，加上概率加成
        else:
            # 单日涨停
            if 'single_limit_up' in self.trend_continuation_probability:
                continuation_prob = self.trend_continuation_probability['single_limit_up']
                score += 10 + continuation_prob / 3  # 基础分10，加上概率加成
        
        # 成交量因素 - 放量上涨更有可能持续
        vol_ratio = stock_data.get('vol_ratio', 1.0)
        if vol_ratio > 3.0:
            score += 15  # 成交量是5日均量的3倍以上，强势放量
        elif vol_ratio > 2.0:
            score += 10  # 成交量是5日均量的2倍以上
        elif vol_ratio > 1.5:
            score += 5   # 成交量是5日均量的1.5倍以上
            
        # 价格因素 - 股价较低的股票更容易连续涨停
        price = stock_data.get('close', 100)
        if price < 15:
            score += 10  # 低价股加分
        elif price < 30:
            score += 5   # 中低价股加分
            
        # 动量因素
        momentum = stock_data.get('momentum_5d', 0)
        if momentum > 0.15:  # 5日涨幅超过15%
            score += 10
        elif momentum > 0.10:  # 5日涨幅超过10%
            score += 7
        elif momentum > 0.05:  # 5日涨幅超过5%
            score += 3
            
        # 机构资金流向因素
        ts_code = stock_data.get('ts_code')
        if ts_code:
            # 如果没有缓存，则获取资金流向数据
            if ts_code not in self.institution_capital_flow:
                self.institution_capital_flow[ts_code] = self.fetch_institution_flow(ts_code)
                
            flow_pct = self.institution_capital_flow[ts_code]
            if flow_pct > 0.05:  # 机构资金大幅净流入
                score += 15
            elif flow_pct > 0.02:  # 机构资金明显净流入
                score += 10
            elif flow_pct > 0:  # 机构资金小幅净流入
                score += 5
        
        # MACD指标 - MACD金叉
        macd_cross = stock_data.get('macd_cross', False)
        if macd_cross:
            score += 8
            
        # 返回最终得分
        return score
        
    def process_stock_batch(self, stock_batch):
        """批量处理股票数据"""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_stock = {
                executor.submit(self.analyze_stock, stock['ts_code']): stock['ts_code']
                for stock in stock_batch
            }
            
            for future in concurrent.futures.as_completed(future_to_stock):
                stock_code = future_to_stock[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"处理股票 {stock_code} 时发生错误: {e}")
        return results
        
    def run(self):
        """运行策略（优化版）"""
        try:
            if self.stock_list is None:
                logger.error("未能获取股票列表，策略无法运行")
                return
            
            start_time = datetime.now()
            results = []
            
            # 将股票列表分块处理
            stock_batches = [
                self.stock_list[i:i + self.chunk_size].to_dict('records')
                for i in range(0, len(self.stock_list), self.chunk_size)
            ]
            
            # 使用进程池处理数据块
            with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as process_executor:
                batch_results = list(process_executor.map(self.process_stock_batch, stock_batches))
                
            # 合并结果
            for batch in batch_results:
                results.extend(batch)
            
            # 按信号可信度排序
            results.sort(key=lambda x: x['signal_confidence'], reverse=True)
            
            # 记录性能指标
            total_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['total_processing_time'] = total_time
            self.performance_metrics['average_stock_processing_time'] = total_time / len(self.stock_list)
            
            # 生成性能报告
            self._generate_performance_report()
            
            # 保存分析结果
            self.save_analysis_results(results)
            
            # 生成分析报告
            self.generate_analysis_report(results)
            
            return results
            
        except Exception as e:
            logger.error(f"运行策略时发生错误: {e}")
            return None
            
    def _generate_performance_report(self):
        """生成性能报告"""
        try:
            report = {
                'total_processing_time': self.performance_metrics['total_processing_time'],
                'average_stock_processing_time': self.performance_metrics['average_stock_processing_time'],
                'cache_efficiency': {
                    'hits': self.performance_metrics['cache_hits'],
                    'misses': self.performance_metrics['cache_misses'],
                    'hit_rate': self.performance_metrics['cache_hits'] / 
                               (self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])
                },
                'data_fetch_stats': {
                    'average_time': np.mean(self.performance_metrics['data_fetch_time']),
                    'max_time': np.max(self.performance_metrics['data_fetch_time']),
                    'min_time': np.min(self.performance_metrics['data_fetch_time'])
                }
            }
            
            # 保存性能报告
            report_file = os.path.join(self.results_dir, f"performance_report_{self.today}.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                
            logger.info(f"性能报告已生成: {report_file}")
            
        except Exception as e:
            logger.error(f"生成性能报告时发生错误: {e}")

    def save_analysis_results(self, results: List[Dict[str, Any]]):
        """保存分析结果"""
        try:
            # 保存为CSV文件
            df = pd.DataFrame(results)
            csv_file = os.path.join(self.results_dir, f"analysis_results_{self.today}.csv")
            df.to_csv(csv_file, index=False)
            
            # 保存为JSON文件（包含更详细的信息）
            json_file = os.path.join(self.results_dir, f"analysis_results_{self.today}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            logger.info(f"分析结果已保存至: {self.results_dir}")
            
        except Exception as e:
            logger.error(f"保存分析结果时发生错误: {e}")

    def generate_analysis_report(self, results: List[Dict[str, Any]]):
        """生成分析报告"""
        try:
            report_file = os.path.join(self.results_dir, f"analysis_report_{self.today}.html")
            
            # 使用pandas生成HTML报告
            df = pd.DataFrame(results)
            
            # 添加样式
            styled_df = df.style.background_gradient(subset=['signal_confidence', 'momentum_score'])
            
            # 生成HTML报告
            html_content = f"""
            <html>
            <head>
                <title>股票分析报告 - {self.today}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .summary {{ margin-bottom: 20px; }}
                    .table {{ width: 100%; border-collapse: collapse; }}
                    .table th, .table td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                    .table th {{ background-color: #f5f5f5; }}
                    .high-confidence {{ background-color: #e6ffe6; }}
                    .low-confidence {{ background-color: #ffe6e6; }}
                </style>
            </head>
            <body>
                <h1>股票分析报告 - {self.today}</h1>
                <div class="summary">
                    <h2>分析摘要</h2>
                    <p>分析股票数量: {len(results)}</p>
                    <p>高可信度信号数量: {len([r for r in results if r['signal_confidence'] > 0.8])}</p>
                    <p>数据质量良好的股票比例: {len([r for r in results if r['data_quality']['quality_score'] > 0.9]) / len(results):.2%}</p>
                </div>
                {styled_df.to_html(classes='table', escape=False)}
            </body>
            </html>
            """
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"分析报告已生成: {report_file}")
            
        except Exception as e:
            logger.error(f"生成分析报告时发生错误: {e}")


# 主函数
if __name__ == "__main__":
    try:
        # 执行策略
        strategy = LimitUpCaptureStrategy()
        high_potential_stocks = strategy.run()
        
        # 输出结果汇总
        print("\n===== 连续涨停和大幅上涨股票捕捉结果 =====")
        print(f"分析日期: {strategy.today}")
        
        # 输出连续涨停股票
        for days, stocks in strategy.consecutive_limit_up_stocks.items():
            if stocks:
                print(f"\n=== {days}连续涨停股票 (共{len(stocks)}只) ===")
                for stock in stocks:
                    print(f"股票代码: {stock['ts_code']}, 连续涨停天数: {stock['consecutive_days']}, 成交量比: {stock['vol_ratio']:.2f}")
        
        # 输出高动量股票 (前10只)
        if not strategy.high_momentum_stocks.empty:
            print(f"\n=== 高动量股票 (前10只) ===")
            for _, row in strategy.high_momentum_stocks.head(10).iterrows():
                print(f"股票代码: {row['ts_code']}, 20日涨幅: {row['momentum']:.2f}%, 均线多头: {row['ma_alignment']}, 成交量比: {row['vol_ratio']:.2f}")
        
        # 输出突破股票 (前10只)
        if not strategy.breakout_stocks.empty:
            print(f"\n=== 突破重要阻力位股票 (前10只) ===")
            for _, row in strategy.breakout_stocks.head(10).iterrows():
                print(f"股票代码: {row['ts_code']}, 突破20日新高: {row['break_high_20d']}, 突破60日新高: {row['break_high_60d']}, 强势突破: {row['is_strong_breakout']}")
        
        # 输出综合评分最高的股票
        if not high_potential_stocks.empty:
            print(f"\n=== 综合评分最高的高潜力股票 (前20只) ===")
            for _, row in high_potential_stocks.iterrows():
                print(f"股票代码: {row['ts_code']}, 综合评分: {row['score']}")
                
        print("\n提示: 所有详细结果已保存在 results/limit_up_capture/ 目录下")
        
    except Exception as e:
        logger.error(f"策略执行过程中发生错误: {e}")
        sys.exit(1) 