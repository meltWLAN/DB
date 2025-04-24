import pandas as pd
import numpy as np
import logging
from ..config import INDICATOR_PARAMS

class TechnicalIndicators:
    """技术指标计算类"""
    
    def __init__(self, params=None):
        """初始化
        
        Args:
            params: 指标参数，默认使用配置文件中的参数
        """
        self.params = params or INDICATOR_PARAMS
        self.logger = logging.getLogger(__name__)
    
    def calculate_bollinger_bands(self, data, window=None, window_dev=None):
        """计算布林带指标
        
        Args:
            data: DataFrame，包含股价数据
            window: 移动平均窗口，默认使用配置参数
            window_dev: 标准差倍数，默认使用配置参数
            
        Returns:
            DataFrame: 添加了布林带指标的数据
        """
        window = window or self.params['bollinger_bands']['window']
        window_dev = window_dev or self.params['bollinger_bands']['window_dev']
        
        df = data.copy()
        
        # 计算中轨线 (移动平均线)
        df['bb_middle'] = df['close'].rolling(window=window).mean()
        
        # 计算标准差
        df['bb_std'] = df['close'].rolling(window=window).std()
        
        # 计算上轨线
        df['bb_upper'] = df['bb_middle'] + window_dev * df['bb_std']
        
        # 计算下轨线
        df['bb_lower'] = df['bb_middle'] - window_dev * df['bb_std']
        
        # 计算价格相对位置 (0-1，从下轨到上轨)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 计算带宽
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 计算带宽的变化率
        df['bb_bandwidth_change'] = df['bb_bandwidth'].pct_change()
        
        # 计算压缩区间 (带宽收缩)
        df['bb_squeeze'] = df['bb_bandwidth'] < df['bb_bandwidth'].shift(1)
        
        # 计算价格突破上轨
        df['bb_break_upper'] = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
        
        # 计算价格突破下轨
        df['bb_break_lower'] = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
        
        # 计算带宽突破 (带宽扩张)
        df['bb_bandwidth_breakout'] = (df['bb_bandwidth'] > df['bb_bandwidth'].shift(1) * 1.1) & df['bb_squeeze'].shift(1)
        
        return df
    
    def calculate_macd(self, data, fast=None, slow=None, signal=None):
        """计算MACD指标
        
        Args:
            data: DataFrame，包含股价数据
            fast: 快线周期，默认使用配置参数
            slow: 慢线周期，默认使用配置参数
            signal: 信号线周期，默认使用配置参数
            
        Returns:
            DataFrame: 添加了MACD指标的数据
        """
        fast = fast or self.params['macd']['fast']
        slow = slow or self.params['macd']['slow']
        signal = signal or self.params['macd']['signal']
        
        df = data.copy()
        
        # 计算快线EMA
        df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
        
        # 计算慢线EMA
        df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
        
        # 计算MACD线 (快线 - 慢线)
        df['macd'] = df['ema_fast'] - df['ema_slow']
        
        # 计算信号线 (MACD的EMA)
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        
        # 计算柱状图 (MACD - 信号线)
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 计算MACD金叉 (MACD上穿信号线)
        df['macd_golden_cross'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        
        # 计算MACD死叉 (MACD下穿信号线)
        df['macd_death_cross'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # 计算MACD零轴上穿
        df['macd_zero_cross_up'] = (df['macd'] > 0) & (df['macd'].shift(1) <= 0)
        
        # 计算MACD零轴下穿
        df['macd_zero_cross_down'] = (df['macd'] < 0) & (df['macd'].shift(1) >= 0)
        
        # 计算柱状图由负转正
        df['macd_hist_turn_positive'] = (df['macd_histogram'] > 0) & (df['macd_histogram'].shift(1) <= 0)
        
        # 计算柱状图由正转负
        df['macd_hist_turn_negative'] = (df['macd_histogram'] < 0) & (df['macd_histogram'].shift(1) >= 0)
        
        # 计算MACD背离
        # 上升趋势中的看跌背离 (价格创新高，但MACD未创新高)
        df['macd_bearish_divergence'] = (
            (df['close'] > df['close'].shift(1)) &
            (df['close'] > df['close'].shift(2)) &
            (df['macd'] < df['macd'].shift(1)) &
            (df['macd'] < df['macd'].shift(2))
        )
        
        # 下降趋势中的看涨背离 (价格创新低，但MACD未创新低)
        df['macd_bullish_divergence'] = (
            (df['close'] < df['close'].shift(1)) &
            (df['close'] < df['close'].shift(2)) &
            (df['macd'] > df['macd'].shift(1)) &
            (df['macd'] > df['macd'].shift(2))
        )
        
        # 删除中间计算列
        df = df.drop(['ema_fast', 'ema_slow'], axis=1)
        
        return df
    
    def calculate_rsi(self, data, window=None, overbought=None, oversold=None, uptrend=None):
        """计算RSI指标
        
        Args:
            data: DataFrame，包含股价数据
            window: 窗口大小，默认使用配置参数
            overbought: 超买阈值，默认使用配置参数
            oversold: 超卖阈值，默认使用配置参数
            uptrend: 上涨趋势阈值，默认使用配置参数
            
        Returns:
            DataFrame: 添加了RSI指标的数据
        """
        window = window or self.params['rsi']['window']
        overbought = overbought or self.params['rsi']['threshold_overbought']
        oversold = oversold or self.params['rsi']['threshold_oversold']
        uptrend = uptrend or self.params['rsi']['threshold_uptrend']
        
        df = data.copy()
        
        # 计算价格变化
        df['price_change'] = df['close'].diff()
        
        # 分离上涨和下跌
        df['gain'] = df['price_change'].clip(lower=0)
        df['loss'] = -df['price_change'].clip(upper=0)
        
        # 计算平均上涨和下跌
        df['avg_gain'] = df['gain'].rolling(window=window).mean()
        df['avg_loss'] = df['loss'].rolling(window=window).mean()
        
        # 计算相对强度 (RS = 平均上涨 / 平均下跌)
        df['rs'] = df['avg_gain'] / df['avg_loss'].replace(0, 1e-9)  # 避免除以零
        
        # 计算RSI (RSI = 100 - 100 / (1 + RS))
        df['rsi'] = 100 - (100 / (1 + df['rs']))
        
        # 计算超买状态
        df['rsi_overbought'] = df['rsi'] > overbought
        
        # 计算超卖状态
        df['rsi_oversold'] = df['rsi'] < oversold
        
        # 计算RSI上穿超卖线
        df['rsi_cross_above_oversold'] = (df['rsi'] > oversold) & (df['rsi'].shift(1) <= oversold)
        
        # 计算RSI下穿超买线
        df['rsi_cross_below_overbought'] = (df['rsi'] < overbought) & (df['rsi'].shift(1) >= overbought)
        
        # 计算RSI上穿50，表明趋势向上
        df['rsi_cross_above_50'] = (df['rsi'] > uptrend) & (df['rsi'].shift(1) <= uptrend)
        
        # 计算RSI下穿50，表明趋势向下
        df['rsi_cross_below_50'] = (df['rsi'] < uptrend) & (df['rsi'].shift(1) >= uptrend)
        
        # 计算RSI背离
        # 上升趋势中的看跌背离 (价格创新高，但RSI未创新高)
        df['rsi_bearish_divergence'] = (
            (df['close'] > df['close'].shift(1)) &
            (df['close'] > df['close'].shift(2)) &
            (df['rsi'] < df['rsi'].shift(1)) &
            (df['rsi'] < df['rsi'].shift(2))
        )
        
        # 下降趋势中的看涨背离 (价格创新低，但RSI未创新低)
        df['rsi_bullish_divergence'] = (
            (df['close'] < df['close'].shift(1)) &
            (df['close'] < df['close'].shift(2)) &
            (df['rsi'] > df['rsi'].shift(1)) &
            (df['rsi'] > df['rsi'].shift(2))
        )
        
        # 删除中间计算列
        df = df.drop(['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], axis=1)
        
        return df
    
    def calculate_moving_averages(self, data, ma_windows=None):
        """计算移动平均线
        
        Args:
            data: DataFrame，包含股价数据
            ma_windows: 移动平均窗口列表，默认使用配置参数
            
        Returns:
            DataFrame: 添加了移动平均线的数据
        """
        df = data.copy()
        
        # 准备移动平均窗口列表
        if ma_windows is None:
            ma_windows = []
            for key in ['short_windows', 'medium_windows', 'long_windows']:
                if key in self.params['moving_averages']:
                    ma_windows.extend(self.params['moving_averages'][key])
        
        # 计算各种周期的移动平均线
        for window in ma_windows:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        
        # 计算均线多头排列和空头排列
        if all(f'ma_{w}' in df.columns for w in [5, 10, 20, 60]):
            # 多头排列: 短期均线在上，长期均线在下
            df['ma_bull_alignment'] = (
                (df['ma_5'] > df['ma_10']) & 
                (df['ma_10'] > df['ma_20']) & 
                (df['ma_20'] > df['ma_60'])
            )
            
            # 空头排列: 短期均线在下，长期均线在上
            df['ma_bear_alignment'] = (
                (df['ma_5'] < df['ma_10']) & 
                (df['ma_10'] < df['ma_20']) & 
                (df['ma_20'] < df['ma_60'])
            )
        
        # 计算均线交叉信号
        for i, short_window in enumerate(ma_windows[:-1]):
            for long_window in ma_windows[i+1:]:
                # 计算两条均线的交叉
                short_ma = f'ma_{short_window}'
                long_ma = f'ma_{long_window}'
                
                # 金叉 (短期上穿长期)
                df[f'{short_ma}_cross_above_{long_ma}'] = (
                    (df[short_ma] > df[long_ma]) & 
                    (df[short_ma].shift(1) <= df[long_ma].shift(1))
                )
                
                # 死叉 (短期下穿长期)
                df[f'{short_ma}_cross_below_{long_ma}'] = (
                    (df[short_ma] < df[long_ma]) & 
                    (df[short_ma].shift(1) >= df[long_ma].shift(1))
                )
        
        # 计算均线扩张和收缩
        if 'ma_5' in df.columns and 'ma_20' in df.columns:
            # 均线间距
            df['ma_5_20_distance'] = (df['ma_5'] - df['ma_20']) / df['ma_20']
            
            # 均线扩张 (间距增大)
            df['ma_expansion'] = df['ma_5_20_distance'] > df['ma_5_20_distance'].shift(1)
            
            # 均线收缩 (间距减小)
            df['ma_contraction'] = df['ma_5_20_distance'] < df['ma_5_20_distance'].shift(1)
        
        return df
    
    def calculate_volume_indicators(self, data, short_window=None, long_window=None):
        """计算成交量指标
        
        Args:
            data: DataFrame，包含股价和成交量数据
            short_window: 短期窗口，默认使用配置参数
            long_window: 长期窗口，默认使用配置参数
            
        Returns:
            DataFrame: 添加了成交量指标的数据
        """
        short_window = short_window or self.params['volume']['short_window']
        long_window = long_window or self.params['volume']['long_window']
        
        df = data.copy()
        
        # 计算成交量移动平均
        df[f'volume_ma_{short_window}'] = df['volume'].rolling(window=short_window).mean()
        df[f'volume_ma_{long_window}'] = df['volume'].rolling(window=long_window).mean()
        
        # 计算相对成交量 (当前成交量 / 成交量移动平均)
        df[f'relative_volume_{short_window}'] = df['volume'] / df[f'volume_ma_{short_window}']
        df[f'relative_volume_{long_window}'] = df['volume'] / df[f'volume_ma_{long_window}']
        
        # 计算成交量放大信号 (成交量超过短期均量的倍数)
        df['volume_surge'] = df[f'relative_volume_{short_window}'] > 2.0
        
        # 计算量价关系
        # 股价上涨且成交量放大
        df['price_up_volume_up'] = (
            (df['close'] > df['close'].shift(1)) & 
            (df['volume'] > df['volume'].shift(1))
        )
        
        # 股价下跌且成交量放大
        df['price_down_volume_up'] = (
            (df['close'] < df['close'].shift(1)) & 
            (df['volume'] > df['volume'].shift(1))
        )
        
        # 股价上涨且成交量萎缩
        df['price_up_volume_down'] = (
            (df['close'] > df['close'].shift(1)) & 
            (df['volume'] < df['volume'].shift(1))
        )
        
        # 股价下跌且成交量萎缩
        df['price_down_volume_down'] = (
            (df['close'] < df['close'].shift(1)) & 
            (df['volume'] < df['volume'].shift(1))
        )
        
        # 计算量能突破 (成交量持续放大)
        df['volume_breakout'] = (
            (df['volume'] > df['volume'].shift(1)) &
            (df['volume'].shift(1) > df['volume'].shift(2)) &
            (df['close'] > df['close'].shift(1))
        )
        
        # 计算高位放量
        df['high_volume_at_high'] = (
            (df['close'] > df['close'].rolling(window=20).mean()) &
            (df['volume'] > df['volume'].rolling(window=20).mean() * 1.5)
        )
        
        # 计算低位放量
        df['high_volume_at_low'] = (
            (df['close'] < df['close'].rolling(window=20).mean()) &
            (df['volume'] > df['volume'].rolling(window=20).mean() * 1.5)
        )
        
        return df
    
    def calculate_price_momentum(self, data):
        """计算价格动量指标
        
        Args:
            data: DataFrame，包含股价数据
            
        Returns:
            DataFrame: 添加了价格动量指标的数据
        """
        df = data.copy()
        
        # 计算不同周期的收益率
        for period in [1, 3, 5, 10, 20, 60]:
            df[f'return_{period}d'] = df['close'].pct_change(periods=period)
        
        # 计算价格相对于N日最高价的位置 (0-1)
        for period in [20, 60, 120]:
            # N日最高价
            df[f'high_{period}d'] = df['high'].rolling(window=period).max()
            
            # 相对位置
            df[f'price_position_{period}d'] = df['close'] / df[f'high_{period}d']
            
            # 突破N日最高价
            df[f'break_high_{period}d'] = (df['close'] > df[f'high_{period}d'].shift(1))
        
        # 计算价格动量
        # 5日动量 (当前价格 / 5日前价格 - 1)
        df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
        
        # 10日动量
        df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
        
        # 20日动量
        df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
        
        # 计算价格加速度 (当前动量 / 前一日动量 - 1)
        df['momentum_acceleration'] = df['momentum_5d'] / df['momentum_5d'].shift(1) - 1
        
        # 计算突破回踩
        # 突破前高后回踩
        df['breakout_pullback'] = (
            df['break_high_20d'].shift(1) &  # 昨日突破20日最高价
            (df['close'] < df['close'].shift(1)) &  # 今日收盘价低于昨日
            (df['close'] > df['high_20d'].shift(2))  # 但仍高于前日的20日最高价
        )
        
        return df
    
    def calculate_all_indicators(self, data):
        """计算所有技术指标
        
        Args:
            data: DataFrame，包含股价和成交量数据
            
        Returns:
            DataFrame: 添加了所有技术指标的数据
        """
        if data is None or data.empty:
            self.logger.warning("Empty data provided for indicator calculation")
            return data
        
        df = data.copy()
        
        # 计算布林带
        df = self.calculate_bollinger_bands(df)
        self.logger.info("Calculated Bollinger Bands indicators")
        
        # 计算MACD
        df = self.calculate_macd(df)
        self.logger.info("Calculated MACD indicators")
        
        # 计算RSI
        df = self.calculate_rsi(df)
        self.logger.info("Calculated RSI indicators")
        
        # 计算移动平均线
        df = self.calculate_moving_averages(df)
        self.logger.info("Calculated Moving Averages indicators")
        
        # 计算成交量指标
        df = self.calculate_volume_indicators(df)
        self.logger.info("Calculated Volume indicators")
        
        # 计算价格动量
        df = self.calculate_price_momentum(df)
        self.logger.info("Calculated Price Momentum indicators")
        
        return df 