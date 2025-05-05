#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级技术指标计算模块，提供更多专业技术分析指标
"""

import pandas as pd
import numpy as np
import logging
from ..config import INDICATOR_PARAMS

class AdvancedIndicators:
    """高级技术指标计算类"""
    
    def __init__(self, params=None):
        """初始化
        
        Args:
            params: 指标参数，默认使用配置文件中的参数
        """
        self.params = params or INDICATOR_PARAMS
        self.logger = logging.getLogger(__name__)
    
    def calculate_ichimoku(self, data):
        """计算一目均衡表指标（云图）
        
        Args:
            data: DataFrame，包含OHLC数据
            
        Returns:
            DataFrame: 添加了一目均衡表指标的数据
        """
        df = data.copy()
        
        # 转换线 (Conversion Line): (9日高点 + 9日低点) / 2
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['ichimoku_conversion'] = (high_9 + low_9) / 2
        
        # 基准线 (Base Line): (26日高点 + 26日低点) / 2
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['ichimoku_base'] = (high_26 + low_26) / 2
        
        # 先行带A (Leading Span A): (转换线 + 基准线) / 2，向前推移26天
        df['ichimoku_span_a'] = ((df['ichimoku_conversion'] + df['ichimoku_base']) / 2).shift(26)
        
        # 先行带B (Leading Span B): (52日高点 + 52日低点) / 2，向前推移26天
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['ichimoku_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
        # 延迟线 (Lagging Span): 收盘价向后推移26天
        df['ichimoku_lagging'] = df['close'].shift(-26)
        
        # 计算信号
        # 转换线上穿基准线 (买入信号)
        df['ichimoku_bullish'] = (df['ichimoku_conversion'] > df['ichimoku_base']) & (df['ichimoku_conversion'].shift(1) <= df['ichimoku_base'].shift(1))
        
        # 转换线下穿基准线 (卖出信号)
        df['ichimoku_bearish'] = (df['ichimoku_conversion'] < df['ichimoku_base']) & (df['ichimoku_conversion'].shift(1) >= df['ichimoku_base'].shift(1))
        
        # 价格上穿云层上方 (强力买入信号)
        df['ichimoku_strong_bullish'] = (df['close'] > df['ichimoku_span_a']) & (df['close'] > df['ichimoku_span_b']) & (df['close'].shift(1) <= df[['ichimoku_span_a', 'ichimoku_span_b']].max(axis=1).shift(1))
        
        # 价格下穿云层下方 (强力卖出信号)
        df['ichimoku_strong_bearish'] = (df['close'] < df['ichimoku_span_a']) & (df['close'] < df['ichimoku_span_b']) & (df['close'].shift(1) >= df[['ichimoku_span_a', 'ichimoku_span_b']].min(axis=1).shift(1))
        
        return df
    
    def calculate_atr(self, data, window=14):
        """计算平均真实范围(ATR)
        
        Args:
            data: DataFrame，包含OHLC数据
            window: 窗口大小，默认14天
            
        Returns:
            DataFrame: 添加了ATR指标的数据
        """
        df = data.copy()
        
        # 计算真实范围 (True Range)
        df['tr1'] = df['high'] - df['low']  # 当日最高点和最低点的差
        df['tr2'] = abs(df['high'] - df['close'].shift(1))  # 当日最高点和前一日收盘价的差的绝对值
        df['tr3'] = abs(df['low'] - df['close'].shift(1))  # 当日最低点和前一日收盘价的差的绝对值
        
        # 真实范围取以上三者的最大值
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # 计算ATR (平均真实范围)
        df['atr'] = df['tr'].rolling(window=window).mean()
        
        # 计算ATR百分比 (ATR/收盘价)，表示价格波动率
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        # 删除中间计算列
        df = df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)
        
        return df
    
    def calculate_keltner_channel(self, data, ema_window=20, atr_window=10, multiplier=2):
        """计算肯特纳通道
        
        Args:
            data: DataFrame，包含OHLC数据
            ema_window: EMA窗口大小，默认20天
            atr_window: ATR窗口大小，默认10天
            multiplier: ATR乘数，默认2
            
        Returns:
            DataFrame: 添加了肯特纳通道指标的数据
        """
        df = data.copy()
        
        # 计算EMA (指数移动平均线)
        df['ema'] = df['close'].ewm(span=ema_window, adjust=False).mean()
        
        # 计算ATR
        df = self.calculate_atr(df, window=atr_window)
        
        # 计算上通道线
        df['keltner_upper'] = df['ema'] + (multiplier * df['atr'])
        
        # 计算下通道线
        df['keltner_lower'] = df['ema'] - (multiplier * df['atr'])
        
        # 计算通道宽度
        df['keltner_width'] = (df['keltner_upper'] - df['keltner_lower']) / df['ema'] * 100
        
        # 计算价格位置 (0-1之间，从下通道到上通道)
        df['keltner_position'] = (df['close'] - df['keltner_lower']) / (df['keltner_upper'] - df['keltner_lower'])
        
        # 计算价格突破上通道
        df['keltner_break_upper'] = (df['close'] > df['keltner_upper']) & (df['close'].shift(1) <= df['keltner_upper'].shift(1))
        
        # 计算价格突破下通道
        df['keltner_break_lower'] = (df['close'] < df['keltner_lower']) & (df['close'].shift(1) >= df['keltner_lower'].shift(1))
        
        return df
    
    def calculate_stochastic(self, data, k_window=14, d_window=3, smooth_window=3):
        """计算随机指标
        
        Args:
            data: DataFrame，包含OHLC数据
            k_window: %K窗口大小，默认14天
            d_window: %D窗口大小，默认3天
            smooth_window: 平滑窗口大小，默认3天
            
        Returns:
            DataFrame: 添加了随机指标的数据
        """
        df = data.copy()
        
        # 计算最近k_window天的最低价和最高价
        low_min = df['low'].rolling(window=k_window).min()
        high_max = df['high'].rolling(window=k_window).max()
        
        # 计算原始%K (当前价格在最近k_window天价格范围内的位置)
        df['stoch_k_raw'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        
        # 计算平滑后的%K
        df['stoch_k'] = df['stoch_k_raw'].rolling(window=smooth_window).mean()
        
        # 计算%D (对%K再次平滑)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_window).mean()
        
        # 计算超买状态 (>80)
        df['stoch_overbought'] = df['stoch_k'] > 80
        
        # 计算超卖状态 (<20)
        df['stoch_oversold'] = df['stoch_k'] < 20
        
        # 计算%K上穿%D (买入信号)
        df['stoch_bullish_cross'] = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        
        # 计算%K下穿%D (卖出信号)
        df['stoch_bearish_cross'] = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
        
        # 计算超卖区域的看涨交叉 (更强的买入信号)
        df['stoch_bullish_cross_oversold'] = df['stoch_bullish_cross'] & df['stoch_oversold'].shift(1)
        
        # 计算超买区域的看跌交叉 (更强的卖出信号)
        df['stoch_bearish_cross_overbought'] = df['stoch_bearish_cross'] & df['stoch_overbought'].shift(1)
        
        # 删除中间计算列
        df = df.drop(['stoch_k_raw'], axis=1)
        
        return df
    
    def calculate_adx(self, data, window=14):
        """计算平均趋向指数(ADX)
        
        Args:
            data: DataFrame，包含OHLC数据
            window: 窗口大小，默认14天
            
        Returns:
            DataFrame: 添加了ADX指标的数据
        """
        df = data.copy()
        
        # 计算方向运动指标
        # +DM: 如果今天的最高价大于昨天的最高价，则+DM = 今天最高价 - 昨天最高价，否则+DM = 0
        df['+dm'] = np.where(
            (df['high'] > df['high'].shift(1)) & 
            ((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low'])),
            df['high'] - df['high'].shift(1),
            0
        )
        
        # -DM: 如果今天的最低价小于昨天的最低价，则-DM = 昨天最低价 - 今天最低价，否则-DM = 0
        df['-dm'] = np.where(
            (df['low'] < df['low'].shift(1)) & 
            ((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1))),
            df['low'].shift(1) - df['low'],
            0
        )
        
        # 计算真实波幅 (TR)
        df = self.calculate_atr(df, window=window)
        
        # 计算方向指数
        # +DI = 100 * (+DM的移动平均 / TR的移动平均)
        df['+di'] = 100 * df['+dm'].rolling(window=window).mean() / df['atr']
        
        # -DI = 100 * (-DM的移动平均 / TR的移动平均)
        df['-di'] = 100 * df['-dm'].rolling(window=window).mean() / df['atr']
        
        # 计算方向指数差值
        df['di_diff'] = abs(df['+di'] - df['-di'])
        
        # 计算方向指数之和
        df['di_sum'] = df['+di'] + df['-di']
        
        # 计算方向运动指数 (DX)
        df['dx'] = 100 * df['di_diff'] / df['di_sum']
        
        # 计算平均方向指数 (ADX)
        df['adx'] = df['dx'].rolling(window=window).mean()
        
        # 计算ADX趋势强度
        # ADX > 25 表示强趋势
        df['adx_strong_trend'] = df['adx'] > 25
        
        # ADX > 20 且 +DI > -DI 表示强劲上涨趋势
        df['adx_strong_uptrend'] = (df['adx'] > 20) & (df['+di'] > df['-di'])
        
        # ADX > 20 且 -DI > +DI 表示强劲下跌趋势
        df['adx_strong_downtrend'] = (df['adx'] > 20) & (df['-di'] > df['+di'])
        
        # 计算+DI上穿-DI (买入信号)
        df['adx_bullish_cross'] = (df['+di'] > df['-di']) & (df['+di'].shift(1) <= df['-di'].shift(1))
        
        # 计算+DI下穿-DI (卖出信号)
        df['adx_bearish_cross'] = (df['+di'] < df['-di']) & (df['+di'].shift(1) >= df['-di'].shift(1))
        
        # 删除中间计算列
        df = df.drop(['+dm', '-dm', 'di_diff', 'di_sum', 'dx'], axis=1)
        
        return df
    
    def calculate_parabolic_sar(self, data, af_start=0.02, af_step=0.02, af_max=0.2):
        """计算抛物线转向指标(SAR)
        
        Args:
            data: DataFrame，包含OHLC数据
            af_start: 初始加速因子，默认0.02
            af_step: 加速因子步长，默认0.02
            af_max: 最大加速因子，默认0.2
            
        Returns:
            DataFrame: 添加了SAR指标的数据
        """
        df = data.copy()
        
        # 初始化SAR列
        df['sar'] = np.nan
        
        # 初始化趋势方向列 (1表示上涨，-1表示下跌)
        df['sar_trend'] = np.nan
        
        # 初始化极值点
        df['sar_ep'] = np.nan
        
        # 初始化加速因子
        df['sar_af'] = np.nan
        
        # 从第二个交易日开始计算
        for i in range(1, len(df)):
            if i == 1:
                # 第一个有效数据点，设置初始值
                if df['close'].iloc[1] > df['close'].iloc[0]:
                    # 初始趋势为上涨
                    df['sar_trend'].iloc[1] = 1
                    df['sar'].iloc[1] = df['low'].iloc[0]  # SAR设置为前一天最低点
                    df['sar_ep'].iloc[1] = df['high'].iloc[1]  # 极值点设置为当前最高点
                else:
                    # 初始趋势为下跌
                    df['sar_trend'].iloc[1] = -1
                    df['sar'].iloc[1] = df['high'].iloc[0]  # SAR设置为前一天最高点
                    df['sar_ep'].iloc[1] = df['low'].iloc[1]  # 极值点设置为当前最低点
                
                df['sar_af'].iloc[1] = af_start  # 设置初始加速因子
            else:
                # 获取前一天的值
                prev_sar = df['sar'].iloc[i-1]
                prev_trend = df['sar_trend'].iloc[i-1]
                prev_ep = df['sar_ep'].iloc[i-1]
                prev_af = df['sar_af'].iloc[i-1]
                
                # 延续前一天的趋势
                df['sar_trend'].iloc[i] = prev_trend
                
                # 计算当天的SAR
                df['sar'].iloc[i] = prev_sar + prev_af * (prev_ep - prev_sar)
                
                # 处理趋势反转
                if prev_trend == 1:  # 前一天是上涨趋势
                    # 如果当天最低点低于SAR，则趋势反转为下跌
                    if df['low'].iloc[i] < df['sar'].iloc[i]:
                        df['sar_trend'].iloc[i] = -1  # 转为下跌趋势
                        df['sar'].iloc[i] = prev_ep  # SAR设置为前一个极值点（最高点）
                        df['sar_ep'].iloc[i] = df['low'].iloc[i]  # 极值点设置为当前最低点
                        df['sar_af'].iloc[i] = af_start  # 重置加速因子
                    else:
                        # 继续上涨趋势
                        if df['high'].iloc[i] > prev_ep:
                            # 如果创新高，更新极值点和加速因子
                            df['sar_ep'].iloc[i] = df['high'].iloc[i]
                            df['sar_af'].iloc[i] = min(prev_af + af_step, af_max)
                        else:
                            # 没有创新高，保持极值点和加速因子不变
                            df['sar_ep'].iloc[i] = prev_ep
                            df['sar_af'].iloc[i] = prev_af
                        
                        # 确保SAR不高于前两天的最低点
                        df['sar'].iloc[i] = min(df['sar'].iloc[i], df['low'].iloc[i-1], df['low'].iloc[i-2] if i > 2 else float('inf'))
                else:  # 前一天是下跌趋势
                    # 如果当天最高点高于SAR，则趋势反转为上涨
                    if df['high'].iloc[i] > df['sar'].iloc[i]:
                        df['sar_trend'].iloc[i] = 1  # 转为上涨趋势
                        df['sar'].iloc[i] = prev_ep  # SAR设置为前一个极值点（最低点）
                        df['sar_ep'].iloc[i] = df['high'].iloc[i]  # 极值点设置为当前最高点
                        df['sar_af'].iloc[i] = af_start  # 重置加速因子
                    else:
                        # 继续下跌趋势
                        if df['low'].iloc[i] < prev_ep:
                            # 如果创新低，更新极值点和加速因子
                            df['sar_ep'].iloc[i] = df['low'].iloc[i]
                            df['sar_af'].iloc[i] = min(prev_af + af_step, af_max)
                        else:
                            # 没有创新低，保持极值点和加速因子不变
                            df['sar_ep'].iloc[i] = prev_ep
                            df['sar_af'].iloc[i] = prev_af
                        
                        # 确保SAR不低于前两天的最高点
                        df['sar'].iloc[i] = max(df['sar'].iloc[i], df['high'].iloc[i-1], df['high'].iloc[i-2] if i > 2 else float('-inf'))
        
        # 计算SAR反转信号
        df['sar_bullish'] = (df['sar_trend'] == 1) & (df['sar_trend'].shift(1) == -1)  # 由下跌转为上涨
        df['sar_bearish'] = (df['sar_trend'] == -1) & (df['sar_trend'].shift(1) == 1)  # 由上涨转为下跌
        
        # 删除中间计算列
        df = df.drop(['sar_ep', 'sar_af'], axis=1)
        
        return df
    
    def calculate_all_advanced_indicators(self, data):
        """计算所有高级技术指标
        
        Args:
            data: DataFrame，包含OHLC数据
            
        Returns:
            DataFrame: 添加了所有高级技术指标的数据
        """
        df = data.copy()
        
        try:
            # 计算一目均衡表
            df = self.calculate_ichimoku(df)
            
            # 计算ATR
            df = self.calculate_atr(df)
            
            # 计算肯特纳通道
            df = self.calculate_keltner_channel(df)
            
            # 计算随机指标
            df = self.calculate_stochastic(df)
            
            # 计算ADX
            df = self.calculate_adx(df)
            
            # 计算抛物线SAR
            df = self.calculate_parabolic_sar(df)
            
            self.logger.info("所有高级技术指标计算完成")
        except Exception as e:
            self.logger.error(f"计算高级技术指标时出错: {str(e)}")
        
        return df 