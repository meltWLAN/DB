#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化的数据处理器
使用向量化操作和并行处理提高性能
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import concurrent.futures
import os
from typing import Dict, List, Optional, Union, Any, Callable
from functools import partial

from ...config.settings import PROCESSING_CONFIG

# 设置日志
logger = logging.getLogger(__name__)

class OptimizedDataProcessor:
    """
    优化的数据处理器，使用向量化操作和并行处理提高性能
    """
    
    def __init__(self):
        """初始化处理器"""
        self.use_vectorization = PROCESSING_CONFIG.get("use_vectorization", True)
        self.num_workers = PROCESSING_CONFIG.get("num_workers", 4)
        self.chunk_size = PROCESSING_CONFIG.get("chunk_size", 5000)
        
        logger.info(f"优化数据处理器初始化，向量化处理: {self.use_vectorization}, 并行工作线程: {self.num_workers}")
    
    def clean_daily_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗日线数据，使用向量化操作
        
        Args:
            data: 包含日线数据的DataFrame
            
        Returns:
            DataFrame: 清洗后的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 删除重复行
        df = df.drop_duplicates(subset=['date'])
        
        # 按日期排序
        df = df.sort_values('date')
        
        # 处理缺失值 - 向量化处理
        price_columns = ['open', 'high', 'low', 'close']
        
        # 检查是否有缺失值
        missing_counts = df[price_columns].isnull().sum()
        
        if missing_counts.sum() > 0:
            logger.warning(f"发现缺失值: {missing_counts.to_dict()}")
            
            # 使用前值填充close的缺失值
            if missing_counts['close'] > 0:
                df['close'] = df['close'].ffill()
            
            # 用close填充open的缺失值
            if missing_counts['open'] > 0:
                df['open'] = df['open'].fillna(df['close'])
            
            # 一次性处理high和low
            if missing_counts['high'] > 0 or missing_counts['low'] > 0:
                # 创建一个临时的max值列
                df['temp_max'] = df[['open', 'close']].max(axis=1)
                df['high'] = df['high'].fillna(df['temp_max'])
                
                # 创建一个临时的min值列
                df['temp_min'] = df[['open', 'close']].min(axis=1)
                df['low'] = df['low'].fillna(df['temp_min'])
                
                # 删除临时列
                df = df.drop(['temp_max', 'temp_min'], axis=1)
            
            # 成交量缺失填充为0
            if 'volume' in df.columns and df['volume'].isnull().any():
                df['volume'] = df['volume'].fillna(0)
        
        return df
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        向量化计算各种收益率
        
        Args:
            data: 包含日线数据的DataFrame
            
        Returns:
            DataFrame: 添加了收益率的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 一次性计算多个时间窗口的收益率
        for period in [1, 5, 10, 20, 60]:
            column_name = f"{period}d_return" if period > 1 else "daily_return"
            df[column_name] = df['close'].pct_change(periods=period)
        
        # 计算对数收益率
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 计算累积收益率
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        
        # 计算相对于N日最高价的回撤
        for window in [20, 60, 120]:
            # 计算N日滚动最高价
            df[f'high_{window}d'] = df['close'].rolling(window=window).max()
            # 计算回撤
            df[f'drawdown_{window}d'] = (df[f'high_{window}d'] - df['close']) / df[f'high_{window}d']
        
        return df
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        向量化计算技术指标
        
        Args:
            data: 包含日线数据的DataFrame
            
        Returns:
            DataFrame: 添加了技术指标的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 计算移动平均
        for window in [5, 10, 20, 30, 60, 120]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        
        # 计算指数移动平均
        for window in [5, 10, 20, 30, 60]:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # 计算MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算不同周期的RSI
        for window in [6, 12, 14, 24]:
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            # 避免除零
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        for window in [20, 30]:
            df[f'boll_mid_{window}'] = df['close'].rolling(window=window).mean()
            df[f'boll_std_{window}'] = df['close'].rolling(window=window).std()
            
            # 上下轨
            df[f'boll_upper_{window}'] = df[f'boll_mid_{window}'] + 2 * df[f'boll_std_{window}']
            df[f'boll_lower_{window}'] = df[f'boll_mid_{window}'] - 2 * df[f'boll_std_{window}']
        
        # 计算KDJ指标
        low_9 = df['low'].rolling(window=9).min()
        high_9 = df['high'].rolling(window=9).max()
        df['k_value'] = 100 * ((df['close'] - low_9) / (high_9 - low_9))
        df['d_value'] = df['k_value'].ewm(alpha=1/3, adjust=False).mean()
        df['j_value'] = 3 * df['k_value'] - 2 * df['d_value']
        
        # 计算ATR (Average True Range)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(window=14).mean()
        
        return df
    
    def detect_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        检测K线形态
        
        Args:
            data: 包含日线数据的DataFrame
            
        Returns:
            DataFrame: 添加了形态标记的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 计算实体长度和上下影线
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # 十字星
        df['is_doji'] = ((df['body'] / df['close']) < 0.001) & (df['upper_shadow'] > 0) & (df['lower_shadow'] > 0)
        
        # 锤子线
        df['is_hammer'] = (df['lower_shadow'] > 2 * df['body']) & (df['upper_shadow'] < 0.5 * df['body']) & (df['body'] > 0)
        
        # 吊颈线
        df['is_hanging_man'] = (df['lower_shadow'] > 2 * df['body']) & (df['upper_shadow'] < 0.5 * df['body']) & (df['body'] > 0) & (df['close'] < df['close'].shift(1))
        
        # 穿刺线
        df['is_piercing'] = (df['open'].shift(1) > df['close'].shift(1)) & (df['open'] < df['low'].shift(1)) & (df['close'] > (df['open'].shift(1) + df['close'].shift(1)) / 2)
        
        # 暗云盖顶
        df['is_dark_cloud'] = (df['close'].shift(1) > df['open'].shift(1)) & (df['open'] > df['high'].shift(1)) & (df['close'] < (df['open'].shift(1) + df['close'].shift(1)) / 2)
        
        # 早晨之星
        df['is_morning_star'] = (df['close'].shift(2) > df['open'].shift(2)) & (((df['open'].shift(1) + df['close'].shift(1)) / 2) < df['close'].shift(2)) & (((df['open'].shift(1) + df['close'].shift(1)) / 2) < df['open']) & (df['close'] > df['open'])
        
        # 黄昏之星
        df['is_evening_star'] = (df['close'].shift(2) < df['open'].shift(2)) & (((df['open'].shift(1) + df['close'].shift(1)) / 2) > df['close'].shift(2)) & (((df['open'].shift(1) + df['close'].shift(1)) / 2) > df['open']) & (df['close'] < df['open'])
        
        return df
    
    def parallel_process(self, df: pd.DataFrame, func: Callable, **kwargs) -> pd.DataFrame:
        """
        并行处理大数据集
        
        Args:
            df: 要处理的DataFrame
            func: 处理函数，接收DataFrame片段，返回处理后的DataFrame
            **kwargs: 传递给func的其他参数
            
        Returns:
            DataFrame: 处理后的数据
        """
        if df is None or df.empty or df.shape[0] < self.chunk_size:
            # 数据量小，直接处理
            return func(df, **kwargs)
        
        # 将数据分割成块
        chunks = [df.iloc[i:i+self.chunk_size] for i in range(0, len(df), self.chunk_size)]
        
        # 创建部分函数
        process_func = partial(func, **kwargs)
        
        # 并行处理
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_func, chunks))
        
        # 合并结果
        result_df = pd.concat(results, ignore_index=True)
        
        return result_df
    
    def process_data_pipeline(self, data: pd.DataFrame, steps: List[str] = None) -> pd.DataFrame:
        """
        数据处理流水线，按顺序执行多个处理步骤
        
        Args:
            data: 输入DataFrame
            steps: 要执行的处理步骤列表，None表示执行所有步骤
            
        Returns:
            DataFrame: 处理后的数据
        """
        if data is None or data.empty:
            return data
        
        # 可用的处理步骤
        available_steps = {
            'clean': self.clean_daily_data,
            'returns': self.calculate_returns,
            'technical': self.calculate_technical_indicators,
            'patterns': self.detect_patterns
        }
        
        # 如果没有指定步骤，默认执行所有步骤
        if steps is None:
            steps = list(available_steps.keys())
        
        result = data.copy()
        executed_steps = []
        
        # 按顺序执行步骤
        for step in steps:
            if step in available_steps:
                logger.info(f"执行处理步骤: {step}")
                start_time = datetime.now()
                
                # 判断是否使用并行处理
                if self.use_vectorization and data.shape[0] >= self.chunk_size:
                    result = self.parallel_process(result, available_steps[step])
                else:
                    result = available_steps[step](result)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                executed_steps.append({
                    'step': step,
                    'duration': duration,
                    'success': True
                })
                
                logger.info(f"完成处理步骤: {step}，耗时: {duration:.2f}秒")
            else:
                logger.warning(f"未知的处理步骤: {step}")
        
        return result
    
    def detect_limit_up(self, data: pd.DataFrame, limit_pct: float = 0.1) -> pd.DataFrame:
        """
        检测涨停股票，使用向量化操作
        
        Args:
            data: 包含日线数据的DataFrame
            limit_pct: 涨停幅度，默认为0.1 (10%)
            
        Returns:
            DataFrame: 添加了涨停标记的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 确保pct_chg字段存在
        if 'pct_chg' not in df.columns:
            if 'change' in df.columns:
                df['pct_chg'] = df['change']
            elif 'daily_return' in df.columns:
                df['pct_chg'] = df['daily_return'] * 100  # 转换为百分比
            else:
                df['pct_chg'] = df['close'].pct_change() * 100  # 计算并转换为百分比
        
        # 使用pct_chg字段判断涨停（向量化操作）
        df['is_limit_up'] = (df['pct_chg'] >= 9.5)
        
        # 高效计算连续涨停天数
        # 1. 创建一个辅助列，标识涨停状态变化
        df['limit_change'] = df['is_limit_up'] != df['is_limit_up'].shift(1)
        
        # 2. 对状态变化进行累计求和，得到分组ID
        df['limit_group'] = df['limit_change'].cumsum()
        
        # 3. 对每个分组进行累计计数
        df['consecutive_limit_up'] = df.groupby(['limit_group'])['is_limit_up'].cumcount() + 1
        
        # 4. 非涨停日期的连续天数设为0
        df.loc[~df['is_limit_up'], 'consecutive_limit_up'] = 0
        
        # 删除辅助列
        df = df.drop(['limit_change', 'limit_group'], axis=1)
        
        return df 