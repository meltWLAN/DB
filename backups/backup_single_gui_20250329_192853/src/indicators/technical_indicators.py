#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, Any

class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """添加所有技术指标到数据框（优化版）"""
        df = df.copy()
        
        # 预计算常用值以避免重复计算
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_change'] = df['close'].diff()
        df['volume_change'] = df['volume'].diff()
        
        # 使用向量化操作替代循环
        periods = np.array([5, 10, 20, 30, 60])
        for period in periods:
            ma_key = f'ma{period}'
            ema_key = f'ema{period}'
            df[ma_key] = df['close'].rolling(window=period, min_periods=1).mean()
            df[ema_key] = df['close'].ewm(span=period, adjust=False, min_periods=1).mean()
        
        # 优化布林带计算
        df = TechnicalIndicators.add_bollinger_bands(df)
        
        # 优化MACD计算
        df = TechnicalIndicators.add_macd(df)
        
        # 优化RSI计算
        df = TechnicalIndicators.add_rsi(df)
        
        # 优化KDJ计算
        df = TechnicalIndicators.add_kdj(df)
        
        # 优化成交量指标计算
        df = TechnicalIndicators.add_volume_indicators(df)
        
        # 优化DMI指标计算
        df = TechnicalIndicators.add_dmi(df)
        
        # 优化CCI指标计算
        df = TechnicalIndicators.add_cci(df)
        
        # 优化TRIX指标计算
        df = TechnicalIndicators.add_trix(df)
        
        # 优化自适应布林带计算
        df = TechnicalIndicators.add_adaptive_bollinger(df)
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """添加布林带指标（优化版）"""
        # 使用向量化操作
        df['bb_middle'] = df['close'].rolling(window=period, min_periods=1).mean()
        rolling_std = df['close'].rolling(window=period, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (rolling_std * 2)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * 2)
        
        # 添加带宽和百分比B指标
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame) -> pd.DataFrame:
        """添加MACD指标（优化版）"""
        # 使用向量化操作
        fast_ema = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
        slow_ema = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
        df['macd'] = fast_ema - slow_ema
        df['signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
        df['macd_hist'] = df['macd'] - df['signal']
        
        # 添加MACD趋势强度
        df['macd_trend_strength'] = abs(df['macd_hist']) / df['close'] * 100
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """添加RSI指标"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def add_kdj(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
        """添加KDJ指标"""
        low_list = df['low'].rolling(window=n, min_periods=1).min()
        high_list = df['high'].rolling(window=n, min_periods=1).max()
        rsv = (df['close'] - low_list) / (high_list - low_list) * 100
        df['k'] = rsv.ewm(com=2, adjust=False).mean()
        df['d'] = df['k'].ewm(com=2, adjust=False).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量相关指标（优化版）"""
        # 使用向量化操作计算成交量均线
        periods = [5, 10, 20]
        for period in periods:
            df[f'volume_ma{period}'] = df['volume'].rolling(window=period, min_periods=1).mean()
        
        # 计算成交量变化率
        df['volume_change_rate'] = df['volume'].pct_change()
        
        # 计算价格成交量趋势指标(PVT)
        df['pvt'] = (df['close'].pct_change() * df['volume']).cumsum()
        
        # 添加成交量强度指标
        df['volume_intensity'] = df['volume'] * abs(df['close'].pct_change())
        
        # 添加资金流向指标
        df['money_flow'] = df['typical_price'] * df['volume']
        df['money_flow_positive'] = np.where(df['close'] > df['close'].shift(1), df['money_flow'], 0)
        df['money_flow_negative'] = np.where(df['close'] < df['close'].shift(1), df['money_flow'], 0)
        
        return df
    
    @staticmethod
    def add_dmi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """添加DMI（动向指标）"""
        df['tr'] = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        
        df['dm_plus'] = df['high'].diff()
        df['dm_minus'] = -df['low'].diff()
        
        df['dm_plus'] = df['dm_plus'].where(
            (df['dm_plus'] > df['dm_minus']) & (df['dm_plus'] > 0),
            0.0
        )
        df['dm_minus'] = df['dm_minus'].where(
            (df['dm_minus'] > df['dm_plus']) & (df['dm_minus'] > 0),
            0.0
        )
        
        df['tr_ma'] = df['tr'].rolling(window=period).mean()
        df['dm_plus_ma'] = df['dm_plus'].rolling(window=period).mean()
        df['dm_minus_ma'] = df['dm_minus'].rolling(window=period).mean()
        
        df['di_plus'] = (df['dm_plus_ma'] / df['tr_ma']) * 100
        df['di_minus'] = (df['dm_minus_ma'] / df['tr_ma']) * 100
        df['dx'] = abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus']) * 100
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """添加CCI（顺势指标）"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_ma = tp.rolling(window=period).mean()
        md = abs(tp - tp_ma).rolling(window=period).mean()
        df['cci'] = (tp - tp_ma) / (0.015 * md)
        return df
    
    @staticmethod
    def add_trix(df: pd.DataFrame, period: int = 15) -> pd.DataFrame:
        """添加TRIX（三重指数平滑移动平均指标）"""
        tr = df['close'].ewm(span=period, adjust=False).mean()
        tr2 = tr.ewm(span=period, adjust=False).mean()
        tr3 = tr2.ewm(span=period, adjust=False).mean()
        df['trix'] = tr3.pct_change() * 100
        df['trix_signal'] = df['trix'].rolling(window=9).mean()
        return df
    
    @staticmethod
    def add_adaptive_bollinger(df: pd.DataFrame) -> pd.DataFrame:
        """添加自适应布林带
        使用ATR来动态调整布林带宽度"""
        # 计算ATR
        df['tr'] = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # 计算自适应布林带
        df['adaptive_bb_middle'] = df['close'].rolling(window=20).mean()
        multiplier = df['atr'] / df['close'].rolling(window=20).std()
        df['adaptive_bb_upper'] = df['adaptive_bb_middle'] + (2 * multiplier * df['close'].rolling(window=20).std())
        df['adaptive_bb_lower'] = df['adaptive_bb_middle'] - (2 * multiplier * df['close'].rolling(window=20).std())
        
        return df
    
    @staticmethod
    def get_indicator_signals(df: pd.DataFrame) -> Dict[str, Any]:
        """获取技术指标信号（优化版）"""
        signals = {
            'ma_signals': TechnicalIndicators._get_ma_signals(df),
            'bollinger_signals': TechnicalIndicators._get_bollinger_signals(df),
            'macd_signals': TechnicalIndicators._get_macd_signals(df),
            'rsi_signals': TechnicalIndicators._get_rsi_signals(df),
            'kdj_signals': TechnicalIndicators._get_kdj_signals(df),
            'volume_signals': TechnicalIndicators._get_volume_signals(df),
            'dmi_signals': TechnicalIndicators._get_dmi_signals(df),
            'cci_signals': TechnicalIndicators._get_cci_signals(df),
            'trix_signals': TechnicalIndicators._get_trix_signals(df),
            'adaptive_bb_signals': TechnicalIndicators._get_adaptive_bb_signals(df)
        }
        
        # 计算综合信号强度
        signals['signal_strength'] = TechnicalIndicators._calculate_signal_strength(signals)
        
        # 添加信号可信度评分
        signals['signal_reliability'] = TechnicalIndicators._calculate_signal_reliability(df, signals)
        
        return signals
    
    @staticmethod
    def _get_ma_signals(df: pd.DataFrame) -> Dict[str, str]:
        """获取移动平均线信号"""
        last_row = df.iloc[-1]
        signals = {}
        
        # MA5与MA10交叉信号
        if df['ma5'].iloc[-2] <= df['ma10'].iloc[-2] and last_row['ma5'] > last_row['ma10']:
            signals['ma5_ma10'] = '金叉'
        elif df['ma5'].iloc[-2] >= df['ma10'].iloc[-2] and last_row['ma5'] < last_row['ma10']:
            signals['ma5_ma10'] = '死叉'
        else:
            signals['ma5_ma10'] = '无信号'
            
        # 价格与MA20关系
        if last_row['close'] > last_row['ma20']:
            signals['price_ma20'] = '多头'
        else:
            signals['price_ma20'] = '空头'
            
        return signals
    
    @staticmethod
    def _get_bollinger_signals(df: pd.DataFrame) -> Dict[str, str]:
        """获取布林带信号"""
        last_row = df.iloc[-1]
        
        if last_row['close'] > last_row['bb_upper']:
            return {'signal': '超买'}
        elif last_row['close'] < last_row['bb_lower']:
            return {'signal': '超卖'}
        else:
            return {'signal': '区间震荡'}
    
    @staticmethod
    def _get_macd_signals(df: pd.DataFrame) -> Dict[str, str]:
        """获取MACD信号"""
        if df['macd'].iloc[-2] <= df['signal'].iloc[-2] and df['macd'].iloc[-1] > df['signal'].iloc[-1]:
            return {'signal': '金叉'}
        elif df['macd'].iloc[-2] >= df['signal'].iloc[-2] and df['macd'].iloc[-1] < df['signal'].iloc[-1]:
            return {'signal': '死叉'}
        else:
            return {'signal': '无信号'}
    
    @staticmethod
    def _get_rsi_signals(df: pd.DataFrame) -> Dict[str, str]:
        """获取RSI信号"""
        last_rsi = df['rsi'].iloc[-1]
        
        if last_rsi > 70:
            return {'signal': '超买'}
        elif last_rsi < 30:
            return {'signal': '超卖'}
        else:
            return {'signal': '区间震荡'}
    
    @staticmethod
    def _get_kdj_signals(df: pd.DataFrame) -> Dict[str, str]:
        """获取KDJ信号"""
        if df['k'].iloc[-2] <= df['d'].iloc[-2] and df['k'].iloc[-1] > df['d'].iloc[-1]:
            return {'signal': '金叉'}
        elif df['k'].iloc[-2] >= df['d'].iloc[-2] and df['k'].iloc[-1] < df['d'].iloc[-1]:
            return {'signal': '死叉'}
        else:
            return {'signal': '无信号'}
    
    @staticmethod
    def _get_volume_signals(df: pd.DataFrame) -> Dict[str, str]:
        """获取成交量信号"""
        signals = {}
        
        # 成交量趋势
        if df['volume'].iloc[-1] > df['volume_ma5'].iloc[-1]:
            signals['volume_trend'] = '放量'
        else:
            signals['volume_trend'] = '缩量'
            
        # 价格成交量配合
        if df['close'].iloc[-1] > df['close'].iloc[-2] and df['volume'].iloc[-1] > df['volume'].iloc[-2]:
            signals['price_volume'] = '量价齐升'
        elif df['close'].iloc[-1] < df['close'].iloc[-2] and df['volume'].iloc[-1] > df['volume'].iloc[-2]:
            signals['price_volume'] = '量增价跌'
        else:
            signals['price_volume'] = '背离'
            
        return signals
    
    @staticmethod
    def _get_dmi_signals(df: pd.DataFrame) -> Dict[str, str]:
        """获取DMI信号"""
        last_row = df.iloc[-1]
        
        if last_row['di_plus'] > last_row['di_minus'] and last_row['adx'] > 25:
            return {'signal': '强势多头', 'strength': 2}
        elif last_row['di_minus'] > last_row['di_plus'] and last_row['adx'] > 25:
            return {'signal': '强势空头', 'strength': -2}
        elif last_row['di_plus'] > last_row['di_minus']:
            return {'signal': '弱势多头', 'strength': 1}
        else:
            return {'signal': '弱势空头', 'strength': -1}
    
    @staticmethod
    def _get_cci_signals(df: pd.DataFrame) -> Dict[str, str]:
        """获取CCI信号"""
        last_cci = df['cci'].iloc[-1]
        
        if last_cci > 100:
            return {'signal': '超买', 'strength': -1}
        elif last_cci < -100:
            return {'signal': '超卖', 'strength': 1}
        else:
            return {'signal': '中性', 'strength': 0}
    
    @staticmethod
    def _get_trix_signals(df: pd.DataFrame) -> Dict[str, str]:
        """获取TRIX信号"""
        if df['trix'].iloc[-2] <= df['trix_signal'].iloc[-2] and df['trix'].iloc[-1] > df['trix_signal'].iloc[-1]:
            return {'signal': '金叉', 'strength': 2}
        elif df['trix'].iloc[-2] >= df['trix_signal'].iloc[-2] and df['trix'].iloc[-1] < df['trix_signal'].iloc[-1]:
            return {'signal': '死叉', 'strength': -2}
        else:
            return {'signal': '无信号', 'strength': 0}
    
    @staticmethod
    def _get_adaptive_bb_signals(df: pd.DataFrame) -> Dict[str, str]:
        """获取自适应布林带信号"""
        last_row = df.iloc[-1]
        
        if last_row['close'] > last_row['adaptive_bb_upper']:
            return {'signal': '超买', 'strength': -2}
        elif last_row['close'] < last_row['adaptive_bb_lower']:
            return {'signal': '超卖', 'strength': 2}
        else:
            return {'signal': '区间震荡', 'strength': 0}
    
    @staticmethod
    def _calculate_signal_strength(signals: Dict[str, Any]) -> float:
        """计算综合信号强度
        返回值在-1到1之间，绝对值越大表示信号越强"""
        strength = 0
        weights = {
            'ma_signals': 0.15,
            'bollinger_signals': 0.1,
            'macd_signals': 0.15,
            'rsi_signals': 0.1,
            'kdj_signals': 0.1,
            'volume_signals': 0.1,
            'dmi_signals': 0.1,
            'cci_signals': 0.05,
            'trix_signals': 0.1,
            'adaptive_bb_signals': 0.05
        }
        
        for indicator, signal in signals.items():
            if indicator in weights and 'strength' in signal:
                strength += signal['strength'] * weights[indicator]
        
        return np.clip(strength, -1, 1)
    
    @staticmethod
    def _calculate_signal_reliability(df: pd.DataFrame, signals: Dict[str, Any]) -> float:
        """计算信号可信度评分"""
        reliability = 1.0
        
        # 检查数据质量
        if df.isnull().any().any():
            reliability *= 0.8
        
        # 检查成交量支撑
        volume_signal = signals['volume_signals']
        if volume_signal.get('volume_trend') == '放量':
            reliability *= 1.2
        
        # 检查趋势确认
        if signals['ma_signals'].get('price_ma20') == '多头' and df['macd_trend_strength'].iloc[-1] > 0:
            reliability *= 1.1
        
        # 检查信号一致性
        consistent_signals = 0
        total_signals = 0
        
        for signal_type, signal in signals.items():
            if isinstance(signal, dict) and 'signal' in signal:
                total_signals += 1
                if signal['signal'] in ['多头', '金叉', '超卖']:
                    consistent_signals += 1
                elif signal['signal'] in ['空头', '死叉', '超买']:
                    consistent_signals -= 1
        
        if total_signals > 0:
            consistency = abs(consistent_signals) / total_signals
            reliability *= (0.5 + 0.5 * consistency)
        
        return min(reliability, 1.0) 