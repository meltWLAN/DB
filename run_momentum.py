#!/usr/bin/env python
"""
A wrapper script for momentum_analysis.py that adds the missing
calculate_momentum method and handles the use_tushare parameter
"""
import os
import sys
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta

# Define the calculate_momentum function
def calculate_momentum(self, data, period=None):
    """
    计算股票动量指标
    
    参数:
        data: 股票日线数据DataFrame
        period: 动量计算周期，默认为实例化时设置的周期
        
    返回:
        包含动量分析结果的字典
    """
    try:
        if data is None or data.empty or len(data) < 20:
            self.logger.warning("数据为空或不足，无法计算动量指标")
            return None
            
        # 使用实例变量或参数指定的值
        period = period if period is not None else self.momentum_period
        
        # 确保数据类型正确
        df = data.copy()
        
        # 确保有必要的列
        required_columns = ['close', 'open', 'high', 'low', 'vol']
        for col in required_columns:
            if col not in df.columns:
                self.logger.error(f"数据缺少必要的列: {col}")
                return None
        
        # 计算价格变化百分比
        price_change = (df['close'].iloc[-1] / df['close'].iloc[-period-1] - 1) * 100 if len(df) > period else 0
        
        # 计算均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean() if len(df) >= 60 else np.nan
        
        # 计算MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal']
        
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算成交量变化
        volume_change = (df['vol'].iloc[-period:].mean() / df['vol'].iloc[-period*2:-period].mean() - 1) * 100 if len(df) > period*2 else 0
        
        # 计算布林带
        df['boll_mid'] = df['close'].rolling(window=20).mean()
        df['boll_std'] = df['close'].rolling(window=20).std()
        df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
        
        # 获取最近的值
        last_values = df.iloc[-1]
        
        # 计算动量得分
        momentum_score = 0
        max_score = 100
        
        # 1. 价格相对于均线得分 (0-30分)
        ma_score = 0
        if not np.isnan(last_values['ma5']) and last_values['close'] > last_values['ma5']:
            ma_score += 6
        if not np.isnan(last_values['ma10']) and last_values['close'] > last_values['ma10']:
            ma_score += 8
        if not np.isnan(last_values['ma20']) and last_values['close'] > last_values['ma20']:
            ma_score += 8
        if not np.isnan(last_values['ma60']) and last_values['close'] > last_values['ma60']:
            ma_score += 8
        
        # 2. 价格变化得分 (0-20分)
        price_change_score = min(20, max(0, price_change))
        
        # 3. MACD信号得分 (0-20分)
        macd_score = 0
        if not np.isnan(last_values['macd']) and last_values['macd'] > 0:
            macd_score += 10
        if not np.isnan(last_values['macd']) and not np.isnan(last_values['signal']) and last_values['macd'] > last_values['signal']:
            macd_score += 10
            
        # 4. RSI得分 (0-15分)
        rsi_score = 0
        if not np.isnan(last_values['rsi']):
            if 40 <= last_values['rsi'] <= 70:
                rsi_score += 10
            if 50 <= last_values['rsi'] <= 65:
                rsi_score += 5
                
        # 5. 成交量得分 (0-15分)
        volume_score = min(15, max(0, volume_change))
        
        # 计算总得分
        momentum_score = ma_score + price_change_score + macd_score + rsi_score + volume_score
        momentum_score = min(max_score, max(0, momentum_score))  # 限制在0-100之间
        
        # 计算相对排名
        rank_score = momentum_score / max_score * 100
        
        # 确定MACD信号
        if not np.isnan(last_values['macd']) and not np.isnan(last_values['signal']):
            if last_values['macd'] > last_values['signal'] and last_values['macd'] > 0:
                macd_signal = "买入"
            elif last_values['macd'] < last_values['signal'] and last_values['macd'] < 0:
                macd_signal = "卖出"
            else:
                macd_signal = "中性"
        else:
            macd_signal = "未知"
            
        # 返回动量分析结果
        result = {
            'momentum_score': momentum_score,
            'rank_score': rank_score,
            'price_change': price_change,
            'volume_change': volume_change,
            'macd_signal': macd_signal,
            'ma_score': ma_score,
            'macd_score': macd_score,
            'rsi_score': rsi_score,
            'volume_score': volume_score
        }
        
        return result
        
    except Exception as e:
        self.logger.error(f"计算动量指标时出错: {str(e)}")
        return None

# Import the MomentumAnalyzer class
from momentum_analysis import MomentumAnalyzer

# Add the get_stock_pool function to the module where MomentumAnalyzer is defined
import momentum_analysis
momentum_analysis.get_stock_pool = get_stock_pool

# Monkey patch the class to add the calculate_momentum method and fix __init__
MomentumAnalyzer.calculate_momentum = calculate_momentum

# Store the original __init__ method
original_init = MomentumAnalyzer.__init__

# Create a new __init__ method that accepts use_tushare
def patched_init(self, stock_pool=None, start_date=None, end_date=None, 
                backtest_start_date=None, backtest_end_date=None, 
                momentum_period=20, use_parallel=True, max_processes=None, 
                use_cache=True, enable_optimization=True, use_tushare=True):
    # Call the original __init__ without use_tushare
    original_init(self, stock_pool, start_date, end_date, 
                 backtest_start_date, backtest_end_date,
                 momentum_period, use_parallel, max_processes, 
                 use_cache, enable_optimization)
    # Add the use_tushare attribute
    self.use_tushare = use_tushare

# Apply the patched __init__
MomentumAnalyzer.__init__ = patched_init

# Add the get_stock_name function if it's missing
def get_stock_name(stock_code):
    return f"Stock_{stock_code}"

if 'get_stock_name' not in globals():
    globals()['get_stock_name'] = get_stock_name

# Add the get_stock_pool function
def get_stock_pool():
    """返回一个简单的股票池"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '000063.SZ', '000333.SZ', '000651.SZ'],
        'name': ['平安银行', '万科A', '中兴通讯', '美的集团', '格力电器']
    })

# Make get_stock_pool available globally
globals()['get_stock_pool'] = get_stock_pool

# Run the same code as in momentum_analysis.py
if __name__ == "__main__":
    # ===== Tushare Direct Initialization Test Start =====
    print("\n===== Tushare 直接初始化功能测试 =====")
    USER_TOKEN = '0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10'
    direct_pro_api = None
    try:
        print(f"尝试使用 Token '{USER_TOKEN[:5]}...' 直接初始化 Pro API...")
        ts.set_token(USER_TOKEN)
        direct_pro_api = ts.pro_api()
        if direct_pro_api:
             print("直接初始化 Pro API 成功")
             # Test trade_cal
             print("\n尝试使用直接初始化的 API 调用 trade_cal...")
             try:
                 df_cal = direct_pro_api.trade_cal(exchange='', start_date='20240101', end_date='20240105')
                 print("直接调用 trade_cal 成功:")
                 print(df_cal)
             except Exception as e_cal:
                 print(f"直接调用 trade_cal 失败: {e_cal}")

             # Test pro_bar
             print("\n尝试使用直接初始化的 API 调用 pro_bar 获取 000001.SZ 数据...")
             try:
                 df_pro_bar = direct_pro_api.pro_bar(ts_code='000001.SZ', adj='qfq', start_date='20240101', end_date='20240110')
                 if df_pro_bar is not None and not df_pro_bar.empty:
                     print("直接调用 pro_bar 成功:")
                     print(df_pro_bar.head())
                 else:
                     print("直接调用 pro_bar 返回为空或 None")
             except Exception as e_pro_bar:
                 print(f"直接调用 pro_bar 失败: {e_pro_bar}")
        else:
            print("直接初始化 Pro API 失败 (返回 None)")

    except Exception as e_init:
         print(f"直接初始化 Pro API 或调用时发生错误: {e_init}")

    print("===== Tushare 直接初始化功能测试结束 =====\n")
    # ===== Tushare Direct Initialization Test End =====

    # 创建动量分析器实例
    analyzer = MomentumAnalyzer(use_tushare=True)
    
    # 获取股票列表 (使用一个简单的样例列表)
    stock_list = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '000063.SZ', '000333.SZ', '000651.SZ'],
        'name': ['平安银行', '万科A', '中兴通讯', '美的集团', '格力电器']
    })
    print(f"获取到 {len(stock_list)} 支股票")
    
    # 分析股票
    try:
        # 调用 analyze_stocks 方法
        results = analyzer.analyze_stocks(stock_list, momentum_period=20)
        
        # 输出结果
        print("分析结果:")
        for index, row in results.iterrows():
            print(f"{row.get('stock_name', 'Unknown')}({row.get('ts_code', 'Unknown')}): " 
                  f"得分={row.get('momentum_score', 0):.1f}, "
                  f"价格变化={row.get('price_change', 0):.2f}%")
    except Exception as e:
        print(f"分析股票时出错: {e}") 