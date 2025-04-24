#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级技术指标模块
提供复杂的技术分析指标和市场情绪指标
"""

import pandas as pd
import numpy as np
import logging
import warnings
import traceback

# 设置日志
# 从utils导入logger使用get_logger函数
try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    # 如果导入失败，使用默认logger
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class AdvancedIndicators:
    """高级技术指标类"""
    
    @staticmethod
    def add_advanced_indicators(data):
        """
        为输入数据添加高级技术指标
        
        Args:
            data: 包含价格和成交量的DataFrame
                必须包含 'open', 'high', 'low', 'close', 'volume' 列
            
        Returns:
            添加了技术指标的DataFrame
        """
        # 确保数据不为空
        if data is None or len(data) < 20:
            logger.warning(f"数据样本不足: {len(data) if data is not None else 0} 行")
            return data
            
        # 检查并转换数据类型
        df = AdvancedIndicators._ensure_dataframe(data)
        if df is None:
            logger.error("无法转换数据为DataFrame")
            return data
            
        # 检查必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"数据缺少必要的列: {col}")
                return data
                
        try:
            # 添加移动平均线
            df = AdvancedIndicators.add_moving_averages(df)
            
            # 添加动量指标
            df = AdvancedIndicators.add_momentum_indicators(df)
            
            # 添加波动率指标
            df = AdvancedIndicators.add_volatility_indicators(df)
            
            # 添加情绪指标
            df = AdvancedIndicators.add_sentiment_indicators(df)
            
            # 添加反转信号指标
            df = AdvancedIndicators.add_reversal_indicators(df)
            
            # 添加交易量指标
            df = AdvancedIndicators.add_volume_indicators(df)
            
            # 添加波段指标
            df = AdvancedIndicators.add_cycle_indicators(df)
            
            # 添加布林带指标
            df = AdvancedIndicators._add_bollinger_bands(df)
            
            # 添加威廉指标
            df = AdvancedIndicators._add_williams_r(df)
            
            # 计算综合评分
            df = AdvancedIndicators.calculate_combined_score(df)
            
            return df
            
        except Exception as e:
            logger.error(f"添加高级指标时出错: {str(e)}")
            return data
            
    @staticmethod
    def _add_bollinger_bands(df, window=20, num_std=2):
        """添加布林带指标"""
        try:
            # 计算移动平均线
            if 'ma20' not in df.columns:
                df['ma20'] = df['close'].rolling(window=window).mean()
                
            # 计算标准差
            rolling_std = df['close'].rolling(window=window).std()
            
            # 计算上中下轨
            df['boll_upper'] = df['ma20'] + (rolling_std * num_std)
            df['boll_middle'] = df['ma20']
            df['boll_lower'] = df['ma20'] - (rolling_std * num_std)
            
            # 计算带宽和百分比B
            df['boll_bandwidth'] = (df['boll_upper'] - df['boll_lower']) / df['boll_middle']
            df['boll_b'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'])
            
            # 向后填充NaN值
            df['boll_upper'] = df['boll_upper'].fillna(method='bfill')
            df['boll_middle'] = df['boll_middle'].fillna(method='bfill')
            df['boll_lower'] = df['boll_lower'].fillna(method='bfill')
            df['boll_bandwidth'] = df['boll_bandwidth'].fillna(method='bfill')
            df['boll_b'] = df['boll_b'].fillna(method='bfill')
            
        except Exception as e:
            logger.error(f"计算布林带时出错: {e}")
            
        return df
        
    @staticmethod
    def _add_williams_r(df, period=14):
        """添加威廉指标 %R"""
        try:
            # 计算周期内的最高价和最低价
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            
            # 计算 Williams %R: (最高价 - 收盘价) / (最高价 - 最低价) * -100
            df['williams_r'] = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
            
            # 向后填充NaN值
            df['williams_r'] = df['williams_r'].fillna(method='bfill')
            
        except Exception as e:
            logger.error(f"计算威廉指标时出错: {e}")
            
        return df

    @staticmethod
    def _ensure_dataframe(data):
        """确保数据是pandas DataFrame格式"""
        if isinstance(data, pd.DataFrame):
            return data.copy()
            
        try:
            if isinstance(data, np.ndarray):
                # 假设数组的列顺序为: open, high, low, close, volume
                column_names = ['open', 'high', 'low', 'close', 'volume']
                if data.shape[1] >= 5:
                    df = pd.DataFrame(data, columns=column_names[:data.shape[1]])
                    
                    # 创建一个日期索引
                    # Try setting index, log warning if fails
                    try:
                        import datetime
                        end_date = datetime.datetime.now()
                        dates = pd.date_range(end=end_date, periods=len(df), freq='B')
                        df.index = dates
                    except Exception as idx_e:
                         logger.warning(f"无法为从numpy数组创建的DataFrame设置日期索引: {idx_e}")

                    return df
                else:
                     # Handle case where array doesn't have enough columns
                     logger.error(f"NumPy array passed to _ensure_dataframe has shape {data.shape}, insufficient columns for OHLCV.")
                     return None # Explicitly return None if shape is wrong
            else:
                # Handle case where input is neither DataFrame nor ndarray
                logger.warning(f"Data passed to _ensure_dataframe is of unexpected type: {type(data)}. Returning None.")
                return None
        except Exception as e:
             logger.error(f"Error converting data to DataFrame in _ensure_dataframe: {e}")
             return None

        # Fallback if conversion failed or type was unexpected
        # This part might not be reached if the exceptions above are hit, but good practice
        # logger.warning("Returning None from _ensure_dataframe due to conversion failure or unexpected type.")
        # return None # Ensure function always returns something, though the above paths should cover it
            
    @staticmethod
    def add_moving_averages(df):
        """添加各种移动平均线"""
        # 确保是DataFrame
        df = AdvancedIndicators._ensure_dataframe(df)
        if df is None:
            return df
            
        try:
            # 简单移动平均线
            for period in [5, 10, 20, 60]:
                df[f'ma{period}'] = df['close'].rolling(window=period).mean()
                
            # 指数移动平均线
            for period in [5, 10, 20, 60]:
                df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                
            # 成交量移动平均线
            for period in [5, 10, 20]:
                df[f'volume_ma{period}'] = df['volume'].rolling(window=period).mean()
                
            # 布林带 (20日移动平均线，2倍标准差)
            if 'ma20' in df.columns:
                rolling_std = df['close'].rolling(window=20).std()
                df['boll_upper'] = df['ma20'] + (rolling_std * 2)
                df['boll_middle'] = df['ma20']
                df['boll_lower'] = df['ma20'] - (rolling_std * 2)
        except Exception as e:
            logger.error(f"添加移动平均线时出错: {e}")
            logger.error(traceback.format_exc())
        
        return df
        
    @staticmethod
    def add_momentum_indicators(df):
        """添加动量指标，如RSI, MACD, CCI"""
        # 确保是DataFrame
        df = AdvancedIndicators._ensure_dataframe(df)
        if df is None:
            return df
            
        try:
            # RSI
            for period in [6, 12, 24]:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()
                
                rs = avg_gain / avg_loss
                df[f'rsi{period}'] = 100 - (100 / (1 + rs))
                
            # MACD (12日EMA, 26日EMA, 9日信号线)
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # 价格动量
            for period in [1, 5, 10, 20]:
                df[f'return_{period}d'] = df['close'].pct_change(periods=period)
                
            # 相对表现
            # 假设我们有一个市场指数列，如不存在则跳过
            if 'index_close' in df.columns:
                for period in [5, 20]:
                    df[f'rel_strength_{period}d'] = (df['close'].pct_change(periods=period) - 
                                                df['index_close'].pct_change(periods=period))
        except Exception as e:
            logger.error(f"添加动量指标时出错: {e}")
            logger.error(traceback.format_exc())
        
        return df
    
    @staticmethod
    def add_volatility_indicators(df):
        """添加波动率指标，如ATR, Bollinger Bands"""
        # 确保是DataFrame
        df = AdvancedIndicators._ensure_dataframe(df)
        if df is None:
            return df
            
        try:
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # 历史波动率
            for period in [5, 10, 20]:
                df[f'volatility_{period}d'] = df['close'].pct_change().rolling(period).std() * np.sqrt(252)
                
            # Bollinger带宽
            if all(col in df.columns for col in ['boll_upper', 'boll_lower', 'boll_middle']):
                df['boll_bandwidth'] = (df['boll_upper'] - df['boll_lower']) / df['boll_middle']
                
        except Exception as e:
            logger.error(f"添加波动率指标时出错: {e}")
            logger.error(traceback.format_exc())
            
        return df
        
    @staticmethod
    def add_sentiment_indicators(df):
        """添加情绪指标，如涨跌比率"""
        # 确保是DataFrame
        df = AdvancedIndicators._ensure_dataframe(df)
        if df is None:
            return df
            
        try:
            # 价格上涨日标记
            df['price_up'] = np.where(df['close'] > df['close'].shift(1), 1, 0)
            df['up_down_ratio'] = df['price_up'].rolling(window=10).sum() / 10
            
            # 收盘价在当日范围的位置 (0-100%)
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']) * 100
            
            # 收盘价相对于N日范围的位置
            for period in [10, 20]:
                period_high = df['high'].rolling(window=period).max()
                period_low = df['low'].rolling(window=period).min()
                df[f'close_position_{period}d'] = ((df['close'] - period_low) / 
                                              (period_high - period_low) * 100)
                
            # 多空力量指标 (Bulls Power 和 Bears Power)
            if 'ema13' not in df.columns:
                df['ema13'] = df['close'].ewm(span=13, adjust=False).mean()
                
            df['bulls_power'] = df['high'] - df['ema13']
            df['bears_power'] = df['low'] - df['ema13']
            
        except Exception as e:
            logger.error(f"添加情绪指标时出错: {e}")
            logger.error(traceback.format_exc())
        
        return df
        
    @staticmethod
    def add_reversal_indicators(df):
        """添加反转指标，如Stochastic Oscillator"""
        # 确保是DataFrame
        df = AdvancedIndicators._ensure_dataframe(df)
        if df is None:
            return df
            
        try:
            # Stochastic Oscillator (%K, %D) - ADDED CALCULATION
            # Set calculation period
            k_period = 14
            d_period = 3
            # Calculate lowest low and highest high over the K period
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            # Calculate %K
            df['stoch_k'] = ((df['close'] - low_min) / (high_max - low_min)) * 100
            # Calculate %D (SMA of %K)
            df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
            # Fill initial NaNs for %D - Use recommended assignment
            df['stoch_d'] = df['stoch_d'].fillna(method='bfill') # Backfill first
            df['stoch_k'] = df['stoch_k'].fillna(method='bfill') # Backfill %K as well

            # RSI超买超卖信号 (Existing)
            if 'rsi14' not in df.columns and 'rsi12' in df.columns:
                df['rsi14'] = df['rsi12']  # 使用最近似的RSI替代
                
            if 'rsi14' in df.columns:
                df['rsi_overbought'] = np.where(df['rsi14'] > 70, 1, 0)
                df['rsi_oversold'] = np.where(df['rsi14'] < 30, 1, 0)
                
            # MACD背离 (简化版)
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                df['macd_cross_above'] = np.where((df['macd'] > df['macd_signal']) & 
                                              (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 1, 0)
                df['macd_cross_below'] = np.where((df['macd'] < df['macd_signal']) & 
                                              (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 1, 0)
                
            # 价格触及布林带
            if all(col in df.columns for col in ['boll_upper', 'boll_lower']):
                df['touch_upper_band'] = np.where(df['high'] >= df['boll_upper'], 1, 0)
                df['touch_lower_band'] = np.where(df['low'] <= df['boll_lower'], 1, 0)
                
            # 蜡烛图形态 - 锤子线 (简化版)
            body_size = np.abs(df['close'] - df['open'])
            lower_shadow = np.minimum(df['open'], df['close']) - df['low']
            upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
            
            df['hammer'] = np.where((lower_shadow > 2 * body_size) & (upper_shadow < 0.2 * body_size) &
                                (body_size > 0), 1, 0)
                                
        except Exception as e:
            logger.error(f"添加反转信号指标时出错: {e}")
            logger.error(traceback.format_exc())
            
        return df
        
    @staticmethod
    def add_volume_indicators(df):
        """添加成交量相关指标，如OBV, CMF"""
        # 确保是DataFrame
        df = AdvancedIndicators._ensure_dataframe(df)
        if df is None:
            return df
            
        try:
            # OBV (On-Balance Volume)
            df['obv_direction'] = np.where(df['close'] > df['close'].shift(1), 
                                       df['volume'], 
                                       np.where(df['close'] < df['close'].shift(1), 
                                            -df['volume'], 0))
            df['obv'] = df['obv_direction'].cumsum()
            
            # 价格成交量趋势
            df['price_volume_trend'] = ((df['close'] - df['close'].shift(1)) / 
                                    df['close'].shift(1) * df['volume'])
            
            # Chaikin货币流量
            if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
                mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
                mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0)
                df['money_flow_volume'] = mf_multiplier * df['volume']
                df['cmf'] = df['money_flow_volume'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
                
        except Exception as e:
            logger.error(f"添加交易量指标时出错: {e}")
            logger.error(traceback.format_exc())
        
        return df
        
    @staticmethod
    def add_cycle_indicators(df):
        """添加周期指标，如CCI, ADX"""
        # 确保是DataFrame
        df = AdvancedIndicators._ensure_dataframe(df)
        if df is None:
            return df
            
        try:
            # CCI (Commodity Channel Index) - ADDED CALCULATION
            cci_period = 20
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            rolling_mean = typical_price.rolling(window=cci_period).mean()
            rolling_std = typical_price.rolling(window=cci_period).std()
            # Avoid division by zero in rolling_std
            rolling_std_safe = rolling_std.replace(0, np.nan) # Replace 0 with NaN before division
            df['cci'] = (typical_price - rolling_mean) / (0.015 * rolling_std_safe)
            # Handle potential infinities resulting from division, and fill NaNs - Use recommended assignment
            df['cci'] = df['cci'].replace([np.inf, -np.inf], np.nan)
            df['cci'] = df['cci'].fillna(method='bfill') # Backfill initial NaNs
            df['cci'] = df['cci'].fillna(0) # Fill remaining NaNs with 0 if any

            # ADX (Average Directional Index)
            if 'atr' in df.columns:
                plus_dm_series = df['high'].diff()
                minus_dm_series = df['low'].diff(-1).abs()

                # Convert result of np.where back to Pandas Series with correct index
                plus_dm_values = np.where((plus_dm_series > minus_dm_series) & (plus_dm_series > 0), plus_dm_series, 0)
                plus_dm = pd.Series(plus_dm_values, index=df.index)

                minus_dm_values = np.where((minus_dm_series > plus_dm_series) & (minus_dm_series > 0), minus_dm_series, 0)
                minus_dm = pd.Series(minus_dm_values, index=df.index)

                # Now .rolling() should work on the Pandas Series plus_dm and minus_dm
                plus_di = 100 * (plus_dm.rolling(window=14).mean() / df['atr'])
                minus_di = 100 * (minus_dm.rolling(window=14).mean() / df['atr'])

                # Handle potential division by zero if plus_di + minus_di is zero
                denominator = plus_di + minus_di
                dx = 100 * (np.abs(plus_di - minus_di) / denominator.replace(0, np.nan))
                dx = dx.replace([np.inf, -np.inf], 0) # Keep replacing inf
                dx = dx.fillna(0) # Fill NaNs resulting from division by zero
                df['adx'] = dx.rolling(window=14).mean()
                
        except Exception as e:
            logger.error(f"添加周期指标时出错: {e}")
            logger.error(traceback.format_exc())
        
        return df
        
    @staticmethod
    def calculate_combined_score(df):
        """计算综合评分"""
        # 确保是DataFrame
        df = AdvancedIndicators._ensure_dataframe(df)
        if df is None:
            return df
            
        # 初始化评分列
        df['trend_score'] = 50
        df['momentum_score'] = 50
        df['volatility_score'] = 50
        df['overall_score'] = 50
        
        try:
            # 趋势评分
            if all(col in df.columns for col in ['ma20', 'ma60']):
                # 价格与短期和中期均线的关系
                df['price_above_ma20'] = np.where(df['close'] > df['ma20'], 1, -1)
                df['price_above_ma60'] = np.where(df['close'] > df['ma60'], 1, -1)
                
                # 短期均线与中期均线的关系
                df['ma20_above_ma60'] = np.where(df['ma20'] > df['ma60'], 1, -1)
                
                # 计算趋势得分 (0-100)
                df['trend_score'] = 50 + (
                    df['price_above_ma20'] * 15 + 
                    df['price_above_ma60'] * 20 + 
                    df['ma20_above_ma60'] * 15
                )
                
            # 动量评分
            if all(col in df.columns for col in ['rsi14', 'macd']):
                # RSI评分 (RSI接近70为强，接近30为弱)
                df['rsi_score'] = (df['rsi14'] - 30) * 5/4
                df['rsi_score'] = df['rsi_score'].clip(0, 100)
                
                # MACD评分
                df['macd_score'] = 50 + df['macd'] * 10  # 乘数需要根据MACD的范围调整
                df['macd_score'] = df['macd_score'].clip(0, 100)
                
                # 综合动量评分
                df['momentum_score'] = (df['rsi_score'] * 0.5 + df['macd_score'] * 0.5)
                
            # 波动率评分
            if 'atr' in df.columns and 'volatility_20d' in df.columns:
                # 历史波动率 (低波动率得高分)
                max_vol = df['volatility_20d'].rolling(window=252).max()
                min_vol = df['volatility_20d'].rolling(window=252).min()
                range_vol = max_vol - min_vol
                
                vol_score = 100 - ((df['volatility_20d'] - min_vol) / range_vol * 100)
                vol_score = vol_score.replace([np.inf, -np.inf], 50)
                df['volatility_score'] = vol_score.clip(0, 100)
                
            # 综合评分
            df['overall_score'] = (
                df['trend_score'] * 0.4 + 
                df['momentum_score'] * 0.4 + 
                df['volatility_score'] * 0.2
            )
            
            # 评分清理，确保在0-100范围内
            for col in ['trend_score', 'momentum_score', 'volatility_score', 'overall_score']:
                df[col] = df[col].clip(0, 100)
                
        except Exception as e:
            logger.error(f"计算综合评分时出错: {str(e)}")
            logger.error(traceback.format_exc())
        
        return df 