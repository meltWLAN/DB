import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

class StockDataProcessor:
    """股票数据处理类，用于清洗、转换和特征工程"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def clean_daily_data(self, data):
        """清洗日线数据
        
        Args:
            data: DataFrame，包含日线数据
            
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
        
        # 处理缺失值
        if df['close'].isnull().any():
            self.logger.warning(f"Found {df['close'].isnull().sum()} missing close prices. Filling with forward fill.")
            df['close'] = df['close'].ffill()
            
        if df['open'].isnull().any():
            self.logger.warning(f"Found {df['open'].isnull().sum()} missing open prices. Filling with close price.")
            df['open'] = df['open'].fillna(df['close'])
            
        if df['high'].isnull().any():
            self.logger.warning(f"Found {df['high'].isnull().sum()} missing high prices. Filling with max of close and open.")
            df['high'] = df.apply(lambda row: max(row['close'], row['open']) if pd.isnull(row['high']) else row['high'], axis=1)
            
        if df['low'].isnull().any():
            self.logger.warning(f"Found {df['low'].isnull().sum()} missing low prices. Filling with min of close and open.")
            df['low'] = df.apply(lambda row: min(row['close'], row['open']) if pd.isnull(row['low']) else row['low'], axis=1)
            
        if df['volume'].isnull().any():
            self.logger.warning(f"Found {df['volume'].isnull().sum()} missing volumes. Filling with 0.")
            df['volume'] = df['volume'].fillna(0)
        
        return df
    
    def calculate_returns(self, data):
        """计算收益率
        
        Args:
            data: DataFrame，包含日线数据
            
        Returns:
            DataFrame: 添加了收益率列的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 日收益率
        df['daily_return'] = df['close'].pct_change()
        
        # 5日收益率
        df['5d_return'] = df['close'].pct_change(periods=5)
        
        # 10日收益率
        df['10d_return'] = df['close'].pct_change(periods=10)
        
        # 20日收益率
        df['20d_return'] = df['close'].pct_change(periods=20)
        
        # 计算相对于大盘的超额收益率（需要传入大盘数据）
        
        return df
    
    def detect_limit_up(self, data, limit_pct=0.1):
        """检测涨停
        
        Args:
            data: DataFrame，包含日线数据
            limit_pct: 涨停幅度，默认为0.1 (10%)
            
        Returns:
            DataFrame: 添加了涨停标记列的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 确保pct_chg字段存在
        if 'pct_chg' not in df.columns and 'change' in df.columns:
            df['pct_chg'] = df['change']
        elif 'pct_chg' not in df.columns and 'daily_return' in df.columns:
            df['pct_chg'] = df['daily_return'] * 100  # 转换为百分比
        elif 'pct_chg' not in df.columns:
            df['pct_chg'] = df['close'].pct_change() * 100  # 计算并转换为百分比
        
        # 使用pct_chg字段判断涨停（考虑不同市场的涨停幅度）
        # ST股票一般为5%，非ST为10%
        # 这里简化处理，统一使用9.5%作为判断标准
        df['is_limit_up'] = (df['pct_chg'] >= 9.5)
        
        # 计算连续涨停天数（使用更高效的方法）
        df['consecutive_limit_up'] = df['is_limit_up'].astype(int)
        
        # 创建一个临时列标识连续的涨停段
        temp = df['is_limit_up'] != df['is_limit_up'].shift()
        temp_cumsum = temp.cumsum()
        
        # 对每个连续涨停段进行累计计数
        df['consecutive_limit_up'] = df.groupby(temp_cumsum)['consecutive_limit_up'].cumsum()
        
        # 非涨停日期的连续天数设为0
        df.loc[~df['is_limit_up'], 'consecutive_limit_up'] = 0
        
        return df
    
    def calculate_price_volume_features(self, data):
        """计算价量特征
        
        Args:
            data: DataFrame，包含日线数据
            
        Returns:
            DataFrame: 添加了价量特征的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 计算均价
        df['avg_price'] = df['amount'] / df['volume']
        
        # 计算成交量变化率
        df['volume_change'] = df['volume'].pct_change()
        
        # 计算5日平均成交量
        df['volume_5d_avg'] = df['volume'].rolling(window=5).mean()
        
        # 计算相对于5日平均的成交量比
        df['volume_ratio_5d'] = df['volume'] / df['volume_5d_avg']
        
        # 计算成交额变化率
        df['amount_change'] = df['amount'].pct_change()
        
        # 计算振幅
        df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1)
        
        # 计算日内波动率
        df['intraday_volatility'] = (df['high'] - df['low']) / df['open']
        
        # 计算True Range
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # 计算20日平均真实波幅 (ATR)
        df['atr_20'] = df['true_range'].rolling(window=20).mean()
        
        return df
    
    def calculate_gap_features(self, data):
        """计算跳空缺口特征
        
        Args:
            data: DataFrame，包含日线数据
            
        Returns:
            DataFrame: 添加了跳空缺口特征的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 计算向上跳空缺口
        df['gap_up'] = (df['low'] > df['high'].shift(1))
        
        # 计算向下跳空缺口
        df['gap_down'] = (df['high'] < df['low'].shift(1))
        
        # 计算跳空缺口大小
        df['gap_size'] = np.where(
            df['gap_up'],
            (df['low'] - df['high'].shift(1)) / df['high'].shift(1),
            np.where(
                df['gap_down'],
                (df['high'] - df['low'].shift(1)) / df['low'].shift(1),
                0
            )
        )
        
        return df
    
    def add_price_momentum_features(self, data):
        """添加价格动量特征
        
        Args:
            data: DataFrame，包含日线数据
            
        Returns:
            DataFrame: 添加了价格动量特征的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 价格在n日内的相对位置
        for window in [5, 10, 20, 60]:
            # 相对于n日最高价的位置 (0-1)
            high_col = f'high_{window}d'
            df[high_col] = df['high'].rolling(window=window).max()
            df[f'price_position_{window}d'] = df['close'] / df[high_col]
            
            # n日内创新高
            df[f'new_high_{window}d'] = (df['close'] >= df[high_col].shift(1))
        
        # 计算价格突破前期高点
        df['break_prev_high'] = (df['close'] > df['high'].shift(1)) & (df['close'] > df['high'].shift(2))
        
        # 计算价格突破前期高点的幅度
        df['break_prev_high_pct'] = np.where(
            df['break_prev_high'],
            (df['close'] - df[['high'].shift(1), df['high'].shift(2)].max(axis=1)) / df[['high'].shift(1), df['high'].shift(2)].max(axis=1),
            0
        )
        
        # 距离前高的回撤百分比
        for window in [20, 60, 120]:
            high_col = f'high_{window}d'
            if high_col not in df.columns:
                df[high_col] = df['high'].rolling(window=window).max()
            df[f'drawdown_{window}d'] = (df[high_col] - df['close']) / df[high_col]
        
        return df
    
    def calculate_ma_features(self, data):
        """计算均线相关特征
        
        Args:
            data: DataFrame，包含日线数据
            
        Returns:
            DataFrame: 添加了均线特征的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 计算简单移动平均线 (SMA)
        ma_windows = [5, 10, 20, 30, 60, 120, 250]
        for window in ma_windows:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        
        # 计算价格相对于均线的比例
        for window in ma_windows:
            ma_col = f'ma_{window}'
            df[f'close_to_{ma_col}'] = df['close'] / df[ma_col]
        
        # 计算短期均线与长期均线的交叉情况
        for short_window, long_window in [(5, 10), (5, 20), (10, 20), (10, 30), (20, 60), (60, 120)]:
            short_ma = f'ma_{short_window}'
            long_ma = f'ma_{long_window}'
            
            # 当前短期均线相对于长期均线的比例
            df[f'{short_ma}_to_{long_ma}'] = df[short_ma] / df[long_ma]
            
            # 短期均线上穿长期均线
            df[f'{short_ma}_cross_above_{long_ma}'] = (df[short_ma] > df[long_ma]) & (df[short_ma].shift(1) <= df[long_ma].shift(1))
            
            # 短期均线下穿长期均线
            df[f'{short_ma}_cross_below_{long_ma}'] = (df[short_ma] < df[long_ma]) & (df[short_ma].shift(1) >= df[long_ma].shift(1))
        
        # 计算均线多头排列 (ma_5 > ma_10 > ma_20 > ma_60)
        df['ma_bull_alignment'] = (
            (df['ma_5'] > df['ma_10']) & 
            (df['ma_10'] > df['ma_20']) & 
            (df['ma_20'] > df['ma_60'])
        )
        
        # 计算均线空头排列 (ma_5 < ma_10 < ma_20 < ma_60)
        df['ma_bear_alignment'] = (
            (df['ma_5'] < df['ma_10']) & 
            (df['ma_10'] < df['ma_20']) & 
            (df['ma_20'] < df['ma_60'])
        )
        
        return df
    
    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """计算布林带指标
        
        Args:
            data: DataFrame，包含日线数据
            window: 窗口大小，默认为20
            num_std: 标准差倍数，默认为2
            
        Returns:
            DataFrame: 添加了布林带指标的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 计算中轨线 (简单移动平均线)
        df['bb_middle'] = df['close'].rolling(window=window).mean()
        
        # 计算标准差
        df['bb_std'] = df['close'].rolling(window=window).std()
        
        # 计算上轨线
        df['bb_upper'] = df['bb_middle'] + num_std * df['bb_std']
        
        # 计算下轨线
        df['bb_lower'] = df['bb_middle'] - num_std * df['bb_std']
        
        # 计算布林带宽度
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 计算价格相对于布林带的位置 (0表示在下轨，1表示在上轨)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 计算价格突破上轨
        df['bb_break_upper'] = df['close'] > df['bb_upper']
        
        # 计算价格突破下轨
        df['bb_break_lower'] = df['close'] < df['bb_lower']
        
        # 计算布林带收缩 (宽度减小)
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].shift(1)
        
        return df
    
    def calculate_rsi(self, data, window=14):
        """计算相对强弱指数 (RSI)
        
        Args:
            data: DataFrame，包含日线数据
            window: 窗口大小，默认为14
            
        Returns:
            DataFrame: 添加了RSI指标的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 计算价格变化
        df['price_change'] = df['close'].diff()
        
        # 分离上涨和下跌
        df['gain'] = np.where(df['price_change'] > 0, df['price_change'], 0)
        df['loss'] = np.where(df['price_change'] < 0, -df['price_change'], 0)
        
        # 计算平均上涨和下跌
        df['avg_gain'] = df['gain'].rolling(window=window).mean()
        df['avg_loss'] = df['loss'].rolling(window=window).mean()
        
        # 计算相对强度
        df['rs'] = df['avg_gain'] / df['avg_loss']
        
        # 计算RSI
        df['rsi'] = 100 - (100 / (1 + df['rs']))
        
        # 计算RSI突破50
        df['rsi_break_50'] = (df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)
        
        # 计算RSI突破70
        df['rsi_break_70'] = (df['rsi'] > 70) & (df['rsi'].shift(1) <= 70)
        
        # 计算RSI跌破30
        df['rsi_break_30'] = (df['rsi'] < 30) & (df['rsi'].shift(1) >= 30)
        
        # 删除中间计算列
        df = df.drop(['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], axis=1)
        
        return df
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """计算MACD指标
        
        Args:
            data: DataFrame，包含日线数据
            fast: 快线周期，默认为12
            slow: 慢线周期，默认为26
            signal: 信号线周期，默认为9
            
        Returns:
            DataFrame: 添加了MACD指标的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 计算快线EMA
        df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
        
        # 计算慢线EMA
        df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
        
        # 计算MACD线
        df['macd'] = df['ema_fast'] - df['ema_slow']
        
        # 计算信号线
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        
        # 计算MACD柱状图
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 计算MACD金叉 (MACD线上穿信号线)
        df['macd_golden_cross'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        
        # 计算MACD死叉 (MACD线下穿信号线)
        df['macd_death_cross'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # 计算MACD柱状图由负变正
        df['macd_hist_turn_positive'] = (df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0)
        
        # 计算MACD柱状图由正变负
        df['macd_hist_turn_negative'] = (df['macd_hist'] < 0) & (df['macd_hist'].shift(1) >= 0)
        
        # 删除中间计算列
        df = df.drop(['ema_fast', 'ema_slow'], axis=1)
        
        return df
    
    def calculate_kdj(self, data, window=9):
        """计算KDJ指标
        
        Args:
            data: DataFrame，包含日线数据
            window: 窗口大小，默认为9
            
        Returns:
            DataFrame: 添加了KDJ指标的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 计算最低价的最低值
        low_min = df['low'].rolling(window=window).min()
        
        # 计算最高价的最高值
        high_max = df['high'].rolling(window=window).max()
        
        # 计算RSV
        df['rsv'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        
        # 计算K值 (默认为9日RSV的3日移动平均)
        df['k'] = df['rsv'].ewm(com=2, adjust=False).mean()
        
        # 计算D值 (默认为9日K值的3日移动平均)
        df['d'] = df['k'].ewm(com=2, adjust=False).mean()
        
        # 计算J值 (3*K-2*D)
        df['j'] = 3 * df['k'] - 2 * df['d']
        
        # 计算KDJ金叉 (K线上穿D线)
        df['kdj_golden_cross'] = (df['k'] > df['d']) & (df['k'].shift(1) <= df['d'].shift(1))
        
        # 计算KDJ死叉 (K线下穿D线)
        df['kdj_death_cross'] = (df['k'] < df['d']) & (df['k'].shift(1) >= df['d'].shift(1))
        
        # 计算超买区 (K值和D值均大于80)
        df['kdj_overbought'] = (df['k'] > 80) & (df['d'] > 80)
        
        # 计算超卖区 (K值和D值均小于20)
        df['kdj_oversold'] = (df['k'] < 20) & (df['d'] < 20)
        
        # 删除中间计算列
        df = df.drop(['rsv'], axis=1)
        
        return df
    
    def calculate_money_flow_index(self, data, window=14):
        """计算资金流向指标 (MFI)
        
        Args:
            data: DataFrame，包含日线数据
            window: 窗口大小，默认为14
            
        Returns:
            DataFrame: 添加了MFI指标的数据
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 计算典型价格
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # 计算资金流量 (典型价格乘以成交量)
        df['money_flow'] = df['typical_price'] * df['volume']
        
        # 计算上涨的资金流量和下跌的资金流量
        df['positive_flow'] = np.where(df['typical_price'] > df['typical_price'].shift(1), df['money_flow'], 0)
        df['negative_flow'] = np.where(df['typical_price'] < df['typical_price'].shift(1), df['money_flow'], 0)
        
        # 计算n日内的正向资金流量和负向资金流量
        df['positive_flow_sum'] = df['positive_flow'].rolling(window=window).sum()
        df['negative_flow_sum'] = df['negative_flow'].rolling(window=window).sum()
        
        # 计算资金流量比率
        df['money_flow_ratio'] = df['positive_flow_sum'] / df['negative_flow_sum']
        
        # 计算资金流向指标 (MFI)
        df['mfi'] = 100 - (100 / (1 + df['money_flow_ratio']))
        
        # 计算MFI超买区 (MFI大于80)
        df['mfi_overbought'] = df['mfi'] > 80
        
        # 计算MFI超卖区 (MFI小于20)
        df['mfi_oversold'] = df['mfi'] < 20
        
        # 计算MFI与价格背离
        # 价格创新高但MFI未创新高 (看跌背离)
        df['mfi_bearish_divergence'] = (
            (df['close'] > df['close'].shift(1)) &
            (df['close'] > df['close'].shift(2)) &
            (df['mfi'] < df['mfi'].shift(1))
        )
        
        # 价格创新低但MFI未创新低 (看涨背离)
        df['mfi_bullish_divergence'] = (
            (df['close'] < df['close'].shift(1)) &
            (df['close'] < df['close'].shift(2)) &
            (df['mfi'] > df['mfi'].shift(1))
        )
        
        # 删除中间计算列
        df = df.drop(['typical_price', 'money_flow', 'positive_flow', 'negative_flow',
                      'positive_flow_sum', 'negative_flow_sum', 'money_flow_ratio'], axis=1)
        
        return df
    
    def process_stock_data(self, data, include_features=None):
        """处理股票数据，计算各种技术指标和特征
        
        Args:
            data: DataFrame，包含日线数据
            include_features: 要包含的特征列表，默认为None (全部特征)
            
        Returns:
            DataFrame: 处理后的数据，包含各种技术指标和特征
        """
        if data is None or data.empty:
            return data
        
        # 复制数据，避免修改原始数据
        df = data.copy()
        
        # 清洗数据
        df = self.clean_daily_data(df)
        
        # 如果没有指定特征，则计算所有特征
        all_features = {
            'returns': self.calculate_returns,
            'limit_up': self.detect_limit_up,
            'price_volume': self.calculate_price_volume_features,
            'gap': self.calculate_gap_features,
            'momentum': self.add_price_momentum_features,
            'ma': self.calculate_ma_features,
            'bollinger': self.calculate_bollinger_bands,
            'rsi': self.calculate_rsi,
            'macd': self.calculate_macd,
            'kdj': self.calculate_kdj,
            'mfi': self.calculate_money_flow_index
        }
        
        features_to_include = include_features or all_features.keys()
        
        # 对每个特征应用相应的处理函数
        for feature in features_to_include:
            if feature in all_features:
                df = all_features[feature](df)
                self.logger.info(f"Calculated {feature} features")
            else:
                self.logger.warning(f"Unknown feature: {feature}")
        
        return df
    
    def create_feature_matrix(self, data, feature_list, target_col=None, shift_periods=1):
        """创建特征矩阵，用于机器学习模型
        
        Args:
            data: DataFrame，包含处理后的数据
            feature_list: 特征列表
            target_col: 目标列名，默认为None
            shift_periods: 目标值的偏移期数，默认为1 (预测下一期)
            
        Returns:
            tuple: (X, y) 特征矩阵和目标值向量
        """
        if data is None or data.empty:
            return None, None
        
        df = data.copy()
        
        # 创建特征矩阵
        X = df[feature_list].copy()
        
        # 创建目标值向量
        y = None
        if target_col is not None:
            y = df[target_col].shift(-shift_periods)  # 预测未来的值
            
            # 删除末尾的NaN值
            X = X.iloc[:-shift_periods]
            y = y.iloc[:-shift_periods]
        
        return X, y
    
    def split_train_test(self, X, y, test_size=0.2, random_state=42):
        """分割训练集和测试集
        
        Args:
            X: 特征矩阵
            y: 目标值向量
            test_size: 测试集比例，默认为0.2
            random_state: 随机种子，默认为42
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        if X is None or y is None:
            return None, None, None, None
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test 