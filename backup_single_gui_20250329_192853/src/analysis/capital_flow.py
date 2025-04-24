import pandas as pd
import numpy as np
import logging
from ..config import MONEY_FLOW_PARAMS

class CapitalFlowAnalyzer:
    """资金流向分析类"""
    
    def __init__(self, params=None):
        """初始化
        
        Args:
            params: 资金流向参数，默认使用配置文件中的参数
        """
        self.params = params or MONEY_FLOW_PARAMS
        self.logger = logging.getLogger(__name__)
    
    def calculate_mfi(self, data, window=None):
        """计算资金流量指标 (Money Flow Index)
        
        Args:
            data: DataFrame，包含股价和成交量数据
            window: 窗口大小，默认使用配置参数
            
        Returns:
            DataFrame: 添加了MFI指标的数据
        """
        window = window or self.params['mfi']['window']
        
        df = data.copy()
        
        # 计算典型价格 (高低收三者的平均值)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # 计算资金流量 (典型价格 * 成交量)
        df['money_flow'] = df['typical_price'] * df['volume']
        
        # 计算上涨日和下跌日的资金流量
        df['up_day'] = df['typical_price'] > df['typical_price'].shift(1)
        df['positive_flow'] = np.where(df['up_day'], df['money_flow'], 0)
        df['negative_flow'] = np.where(~df['up_day'], df['money_flow'], 0)
        
        # 计算n日内的正向资金流和负向资金流
        df['positive_flow_sum'] = df['positive_flow'].rolling(window=window).sum()
        df['negative_flow_sum'] = df['negative_flow'].rolling(window=window).sum()
        
        # 计算资金比率
        df['money_ratio'] = np.where(
            df['negative_flow_sum'] != 0,
            df['positive_flow_sum'] / df['negative_flow_sum'],
            float('inf')  # 避免除零
        )
        
        # 计算MFI (类似于RSI，但考虑了成交量)
        df['mfi'] = 100 - (100 / (1 + df['money_ratio']))
        
        # 检测超买和超卖区域
        overbought = self.params['mfi']['threshold_overbought']
        oversold = self.params['mfi']['threshold_oversold']
        
        df['mfi_overbought'] = df['mfi'] > overbought
        df['mfi_oversold'] = df['mfi'] < oversold
        
        # MFI背离信号
        # 看跌背离：价格创新高，但MFI未创新高
        df['mfi_bearish_divergence'] = (
            (df['close'] > df['close'].shift(1)) &
            (df['close'] > df['close'].shift(2)) &
            (df['mfi'] < df['mfi'].shift(1)) &
            (df['mfi'] < df['mfi'].shift(2)) &
            df['mfi_overbought']
        )
        
        # 看涨背离：价格创新低，但MFI未创新低
        df['mfi_bullish_divergence'] = (
            (df['close'] < df['close'].shift(1)) &
            (df['close'] < df['close'].shift(2)) &
            (df['mfi'] > df['mfi'].shift(1)) &
            (df['mfi'] > df['mfi'].shift(2)) &
            df['mfi_oversold']
        )
        
        # 删除中间计算列
        df = df.drop(['typical_price', 'money_flow', 'up_day', 
                     'positive_flow', 'negative_flow', 
                     'positive_flow_sum', 'negative_flow_sum', 
                     'money_ratio'], axis=1)
        
        return df
    
    def analyze_fund_flow(self, fund_flow_data, n_days=None):
        """分析股票的资金流向数据
        
        Args:
            fund_flow_data: DataFrame，包含资金流向数据
            n_days: 分析的天数列表，默认使用配置参数
            
        Returns:
            DataFrame: 添加了资金流向分析指标的数据
        """
        if fund_flow_data is None or fund_flow_data.empty:
            self.logger.warning("Empty fund flow data provided")
            return fund_flow_data
        
        n_days = n_days or self.params['net_inflow']['days']
        
        df = fund_flow_data.copy()
        
        # 确保列名符合标准格式
        expected_columns = [
            'date', 'main_net_inflow', 'main_net_inflow_pct',
            'huge_net_inflow', 'large_net_inflow', 'medium_net_inflow', 'small_net_inflow'
        ]
        
        if not all(col in df.columns for col in expected_columns):
            self.logger.warning("Fund flow data does not contain all expected columns")
            return df
        
        # 计算各天数的主力资金净流入累计
        for days in n_days:
            df[f'main_net_inflow_sum_{days}d'] = df['main_net_inflow'].rolling(window=days).sum()
            df[f'main_net_inflow_pct_avg_{days}d'] = df['main_net_inflow_pct'].rolling(window=days).mean()
        
        # 计算超大单、大单、中单、小单的累计净流入
        for days in [3, 5]:
            df[f'huge_net_inflow_sum_{days}d'] = df['huge_net_inflow'].rolling(window=days).sum()
            df[f'large_net_inflow_sum_{days}d'] = df['large_net_inflow'].rolling(window=days).sum()
            df[f'medium_net_inflow_sum_{days}d'] = df['medium_net_inflow'].rolling(window=days).sum()
            df[f'small_net_inflow_sum_{days}d'] = df['small_net_inflow'].rolling(window=days).sum()
        
        # 计算主力流入连续天数
        df['main_inflow_days'] = 0
        inflow_days = 0
        
        for i in range(len(df)):
            if df['main_net_inflow'].iloc[i] > 0:
                inflow_days += 1
            else:
                inflow_days = 0
            df['main_inflow_days'].iloc[i] = inflow_days
        
        # 计算主力流入强度变化
        df['main_inflow_change'] = df['main_net_inflow'].pct_change()
        
        # 计算主力流入占比变化
        df['main_inflow_pct_change'] = df['main_net_inflow_pct'].diff()
        
        # 计算主力流入突增信号 (当日流入金额是前一日的1.5倍以上且为正)
        df['main_inflow_surge'] = (
            (df['main_net_inflow'] > 0) &
            (df['main_net_inflow'] > df['main_net_inflow'].shift(1) * 1.5) &
            (df['main_net_inflow'].shift(1) > 0)
        )
        
        # 计算主力流入趋势转变信号
        # 由流出转为流入
        df['main_inflow_trend_reversal_up'] = (
            (df['main_net_inflow'] > 0) &
            (df['main_net_inflow'].shift(1) <= 0) &
            (df['main_net_inflow'].shift(2) <= 0)
        )
        
        # 由流入转为流出
        df['main_inflow_trend_reversal_down'] = (
            (df['main_net_inflow'] < 0) &
            (df['main_net_inflow'].shift(1) >= 0) &
            (df['main_net_inflow'].shift(2) >= 0)
        )
        
        # 计算超大单与大单的流入比例 (反映机构参与度)
        df['institutional_participation'] = (
            (df['huge_net_inflow'] + df['large_net_inflow']) / 
            (df['huge_net_inflow'].abs() + df['large_net_inflow'].abs() + 
             df['medium_net_inflow'].abs() + df['small_net_inflow'].abs() + 1e-10)
        )
        
        # 计算散户与机构的流向背离
        # 机构看多散户看空 (超大单和大单流入，小单流出)
        df['inst_bullish_retail_bearish'] = (
            (df['huge_net_inflow'] > 0) &
            (df['large_net_inflow'] > 0) &
            (df['small_net_inflow'] < 0)
        )
        
        # 机构看空散户看多 (超大单和大单流出，小单流入)
        df['inst_bearish_retail_bullish'] = (
            (df['huge_net_inflow'] < 0) &
            (df['large_net_inflow'] < 0) &
            (df['small_net_inflow'] > 0)
        )
        
        return df
    
    def analyze_industry_fund_flow(self, industry_flow_data, n_days=None):
        """分析行业资金流向数据
        
        Args:
            industry_flow_data: DataFrame，包含行业资金流向数据
            n_days: 分析的天数列表，默认使用配置参数
            
        Returns:
            DataFrame: 添加了行业资金流向分析指标的数据
        """
        if industry_flow_data is None or industry_flow_data.empty:
            self.logger.warning("Empty industry fund flow data provided")
            return industry_flow_data
        
        n_days = n_days or self.params['net_inflow']['days']
        
        df = industry_flow_data.copy()
        
        # 计算行业资金净流入累计
        for days in n_days:
            if days == 1:
                continue  # 已经有当日数据
            df[f'net_inflow_sum_{days}d'] = df.groupby('industry')['net_inflow'].apply(
                lambda x: x.rolling(window=days).sum()
            )
        
        # 计算行业资金净流入排名
        for days in n_days:
            col_name = f'net_inflow_sum_{days}d' if days > 1 else 'net_inflow'
            df[f'industry_rank_{days}d'] = df.groupby('date')[col_name].rank(ascending=False)
        
        # 计算行业资金净流入强度变化
        df['net_inflow_change'] = df.groupby('industry')['net_inflow'].pct_change()
        
        # 计算行业资金流入连续天数
        df['industry_inflow_days'] = df.groupby('industry').apply(
            lambda group: (group['net_inflow'] > 0).astype(int).groupby(
                (group['net_inflow'] <= 0).astype(int).cumsum()
            ).cumsum()
        ).reset_index(level=0, drop=True)
        
        # 计算行业资金突增信号
        df['industry_inflow_surge'] = (
            (df['net_inflow'] > 0) &
            (df.groupby('industry')['net_inflow'].apply(
                lambda x: x > x.shift(1) * 1.5
            ))
        )
        
        # 计算行业资金趋势转变信号
        # 由流出转为流入
        df['industry_trend_reversal_up'] = df.groupby('industry').apply(
            lambda group: (
                (group['net_inflow'] > 0) &
                (group['net_inflow'].shift(1) <= 0) &
                (group['net_inflow'].shift(2) <= 0)
            )
        ).reset_index(level=0, drop=True)
        
        # 由流入转为流出
        df['industry_trend_reversal_down'] = df.groupby('industry').apply(
            lambda group: (
                (group['net_inflow'] < 0) &
                (group['net_inflow'].shift(1) >= 0) &
                (group['net_inflow'].shift(2) >= 0)
            )
        ).reset_index(level=0, drop=True)
        
        return df
    
    def combine_stock_and_industry_flow(self, stock_flow, industry_flow, stock_industry_map):
        """将个股和行业资金流向数据结合分析
        
        Args:
            stock_flow: DataFrame，个股资金流向数据
            industry_flow: DataFrame，行业资金流向数据
            stock_industry_map: DataFrame，股票和行业的对应关系
            
        Returns:
            DataFrame: 结合了个股和行业资金流向的分析数据
        """
        if stock_flow is None or stock_flow.empty:
            self.logger.warning("Empty stock fund flow data provided")
            return None
            
        if industry_flow is None or industry_flow.empty:
            self.logger.warning("Empty industry fund flow data provided")
            return stock_flow
            
        if stock_industry_map is None or stock_industry_map.empty:
            self.logger.warning("Empty stock-industry mapping provided")
            return stock_flow
        
        # 确保stock_industry_map包含必要的列
        required_cols = ['stock_code', 'industry']
        if not all(col in stock_industry_map.columns for col in required_cols):
            self.logger.warning("Stock-industry mapping does not contain required columns")
            return stock_flow
        
        # 将行业资金流向数据并入个股数据
        stock_flow_with_industry = stock_flow.merge(
            stock_industry_map[['stock_code', 'industry']],
            left_on='stock_code',
            right_on='stock_code',
            how='left'
        )
        
        # 对于每个日期和行业的组合，获取行业资金流向数据
        industry_flow_daily = industry_flow.pivot_table(
            index=['date', 'industry'],
            values=['net_inflow', 'net_inflow_sum_3d', 'net_inflow_sum_5d', 
                   'industry_rank_1d', 'industry_rank_3d', 'industry_rank_5d'],
            aggfunc='first'
        ).reset_index()
        
        # 将行业资金流向数据与个股数据合并
        result = stock_flow_with_industry.merge(
            industry_flow_daily,
            left_on=['date', 'industry'],
            right_on=['date', 'industry'],
            how='left',
            suffixes=('_stock', '_industry')
        )
        
        # 计算个股资金流向相对于行业的表现
        # 个股主力资金净流入占比 相对于 行业资金净流入占比
        result['stock_vs_industry_flow'] = result['main_net_inflow_pct'] / (result['net_inflow_industry'] / 100)
        
        # 个股资金流向优于行业
        result['stock_outperforms_industry'] = result['stock_vs_industry_flow'] > 1.2
        
        # 强势行业强势个股 (行业排名靠前且个股资金流入)
        result['strong_industry_strong_stock'] = (
            (result['industry_rank_1d'] <= 10) &  # 行业排名前10
            (result['main_net_inflow'] > 0)  # 个股主力资金流入
        )
        
        # 弱势行业强势个股 (行业排名靠后但个股资金流入强劲)
        result['weak_industry_strong_stock'] = (
            (result['industry_rank_1d'] > 20) &  # 行业排名靠后
            (result['main_net_inflow'] > 0) &  # 个股主力资金流入
            (result['main_net_inflow_pct'] > 0.5)  # 个股主力资金流入占比大
        )
        
        return result 