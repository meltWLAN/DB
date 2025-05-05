#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多数据源管理模块，整合Tushare和AKShare接口
用于获取真实的历史股票数据
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, List, Any, Optional, Union

# 导入数据源
try:
    import tushare as ts
except ImportError:
    print("请安装Tushare: pip install tushare")

try:
    import akshare as ak
except ImportError:
    print("请安装AKShare: pip install akshare")

# 设置日志
logger = logging.getLogger(__name__)

class MultiDataSource:
    """多数据源管理类，整合不同的数据API"""
    
    def __init__(self, 
                 tushare_token: str = None, 
                 use_tushare: bool = True, 
                 use_akshare: bool = True,
                 cache_dir: str = "./data/cache"):
        """
        初始化数据源
        
        Args:
            tushare_token: Tushare的API token
            use_tushare: 是否使用Tushare
            use_akshare: 是否使用AKShare
            cache_dir: 数据缓存目录
        """
        self.use_tushare = use_tushare
        self.use_akshare = use_akshare
        self.cache_dir = cache_dir
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化Tushare
        if self.use_tushare and tushare_token:
            try:
                ts.set_token(tushare_token)
                self.tushare_pro = ts.pro_api()
                logger.info("Tushare初始化成功")
            except Exception as e:
                logger.error(f"Tushare初始化失败: {str(e)}")
                self.use_tushare = False
        else:
            self.use_tushare = False
            
        # 检查AKShare
        if self.use_akshare:
            try:
                # 测试AKShare是否可用
                ak.stock_zh_index_daily(symbol="sz399001")
                logger.info("AKShare初始化成功")
            except Exception as e:
                logger.error(f"AKShare初始化失败: {str(e)}")
                self.use_akshare = False
                
        if not self.use_tushare and not self.use_akshare:
            logger.warning("所有数据源都不可用，请检查配置")
    
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        
        Returns:
            股票列表DataFrame，包含代码、名称、行业等信息
        """
        cache_file = os.path.join(self.cache_dir, "stock_list.csv")
        
        # 检查缓存
        if os.path.exists(cache_file) and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days < 1:
            logger.info("从缓存加载股票列表")
            return pd.read_csv(cache_file)
        
        df = None
        
        # 尝试从Tushare获取
        if self.use_tushare:
            try:
                df = self.tushare_pro.stock_basic(
                    exchange='', 
                    list_status='L', 
                    fields='ts_code,symbol,name,area,industry,list_date'
                )
                logger.info(f"从Tushare获取股票列表成功，共 {len(df)} 只股票")
            except Exception as e:
                logger.error(f"从Tushare获取股票列表失败: {str(e)}")
        
        # 如果Tushare失败，尝试从AKShare获取
        if df is None and self.use_akshare:
            try:
                df = ak.stock_info_a_code_name()
                df.rename(columns={"code": "symbol", "name": "name"}, inplace=True)
                df["ts_code"] = df["symbol"].apply(
                    lambda x: x + ".SH" if x.startswith("6") else x + ".SZ"
                )
                logger.info(f"从AKShare获取股票列表成功，共 {len(df)} 只股票")
            except Exception as e:
                logger.error(f"从AKShare获取股票列表失败: {str(e)}")
        
        # 保存到缓存
        if df is not None:
            df.to_csv(cache_file, index=False)
            return df
        else:
            logger.error("获取股票列表失败，所有数据源都尝试失败")
            return pd.DataFrame()
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str, 
                       adjust: str = "qfq") -> pd.DataFrame:
        """
        获取股票日线数据
        
        Args:
            stock_code: 股票代码(带后缀，如000001.SZ)
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            adjust: 复权方式，None为不复权，qfq为前复权，hfq为后复权
            
        Returns:
            日线数据DataFrame，包含开盘价、收盘价、最高价、最低价、成交量等
        """
        # 处理股票代码格式
        code_with_suffix = stock_code
        if "." not in stock_code:
            # 添加后缀
            code_with_suffix = stock_code + (".SH" if stock_code.startswith("6") else ".SZ")
        
        # 去除后缀的代码
        code_without_suffix = code_with_suffix.split(".")[0]
        
        # 缓存文件名
        cache_key = f"{code_with_suffix}_{start_date}_{end_date}_{adjust}"
        cache_file = os.path.join(self.cache_dir, f"daily_{cache_key.replace('.', '_')}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file) and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days < 1:
            logger.info(f"从缓存加载 {stock_code} 的日线数据")
            df = pd.read_csv(cache_file)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        
        df = None
        
        # 尝试从Tushare获取
        if self.use_tushare:
            try:
                # Tushare日期格式转换
                start_date_fmt = start_date.replace("-", "")
                end_date_fmt = end_date.replace("-", "")
                
                # 获取日线数据
                df_daily = self.tushare_pro.daily(
                    ts_code=code_with_suffix, 
                    start_date=start_date_fmt, 
                    end_date=end_date_fmt
                )
                
                # 获取复权因子
                df_factor = None
                if adjust in ['qfq', 'hfq']:
                    df_factor = self.tushare_pro.adj_factor(
                        ts_code=code_with_suffix, 
                        start_date=start_date_fmt, 
                        end_date=end_date_fmt
                    )
                
                # 合并数据并处理复权
                if df_daily is not None and len(df_daily) > 0:
                    df_daily['date'] = pd.to_datetime(df_daily['trade_date'], format='%Y%m%d')
                    df_daily.sort_values('date', inplace=True)
                    
                    # 处理复权
                    if adjust in ['qfq', 'hfq'] and df_factor is not None and len(df_factor) > 0:
                        df_factor['date'] = pd.to_datetime(df_factor['trade_date'], format='%Y%m%d')
                        df = pd.merge(df_daily, df_factor, on='date')
                        
                        if adjust == 'qfq':
                            # 前复权
                            latest_factor = df['adj_factor'].iloc[-1]
                            df['adj_factor'] = df['adj_factor'] / latest_factor
                        # 复权价格计算
                        for col in ['open', 'high', 'low', 'close']:
                            df[col] = df[col] * df['adj_factor']
                    else:
                        df = df_daily
                    
                    # 重命名列
                    rename_map = {
                        'vol': 'volume',
                        'trade_date': 'date_str',
                        'pct_chg': 'pct_change'
                    }
                    df.rename(columns=rename_map, inplace=True)
                    
                    # 设置索引
                    df.set_index('date', inplace=True)
                    
                    logger.info(f"从Tushare获取 {stock_code} 的日线数据成功，共 {len(df)} 条记录")
            except Exception as e:
                logger.error(f"从Tushare获取 {stock_code} 日线数据失败: {str(e)}")
        
        # 如果Tushare失败，尝试从AKShare获取
        if (df is None or len(df) == 0) and self.use_akshare:
            try:
                # AKShare不需要转换日期格式
                suffix = stock_code.split('.')[-1].lower()
                symbol = f"sh{code_without_suffix}" if suffix == "sh" else f"sz{code_without_suffix}"
                
                # 获取日线数据
                if adjust == "qfq":
                    df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
                elif adjust == "hfq":
                    df = ak.stock_zh_a_daily(symbol=symbol, adjust="hfq")
                else:
                    df = ak.stock_zh_a_daily(symbol=symbol, adjust="")
                
                # 格式化日期
                df['date'] = pd.to_datetime(df['date'])
                
                # 筛选日期范围
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                # 重命名列
                rename_map = {
                    'volume': 'volume',
                    'outstanding_share': 'float_share',
                    'turnover': 'turnover_rate'
                }
                df.rename(columns=rename_map, inplace=True)
                
                # 设置索引
                df.set_index('date', inplace=True)
                
                logger.info(f"从AKShare获取 {stock_code} 的日线数据成功，共 {len(df)} 条记录")
            except Exception as e:
                logger.error(f"从AKShare获取 {stock_code} 日线数据失败: {str(e)}")
        
        # 保存到缓存
        if df is not None and len(df) > 0:
            df.reset_index().to_csv(cache_file, index=False)
            return df
        else:
            logger.error(f"获取 {stock_code} 日线数据失败，所有数据源都尝试失败")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算常用技术指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            添加技术指标后的DataFrame
        """
        if df is None or len(df) == 0:
            return df
            
        # 确保有必要的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning("数据缺少必要的列，无法计算技术指标")
            return df
            
        # 计算移动平均线
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_60'] = df['close'].rolling(window=60).mean()
        
        # 计算布林带
        df['boll_mid'] = df['close'].rolling(window=20).mean()
        df['boll_std'] = df['close'].rolling(window=20).std()
        df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
        
        # 计算MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['signal']
        
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(window=14).mean()
        
        # 计算日收益率
        df['daily_return'] = df['close'].pct_change()
        
        return df
    
    def get_industry_data(self) -> pd.DataFrame:
        """
        获取行业分类数据
        
        Returns:
            行业分类DataFrame，包含股票代码和所属行业
        """
        cache_file = os.path.join(self.cache_dir, "industry_data.csv")
        
        # 检查缓存
        if os.path.exists(cache_file) and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days < 7:
            logger.info("从缓存加载行业分类数据")
            return pd.read_csv(cache_file)
        
        df = None
        
        # 尝试从Tushare获取
        if self.use_tushare:
            try:
                df = self.tushare_pro.stock_basic(
                    exchange='', 
                    list_status='L', 
                    fields='ts_code,industry'
                )
                logger.info("从Tushare获取行业分类数据成功")
            except Exception as e:
                logger.error(f"从Tushare获取行业分类数据失败: {str(e)}")
        
        # 如果Tushare失败，尝试从AKShare获取
        if df is None and self.use_akshare:
            try:
                # AKShare没有直接提供行业分类数据，尝试从股票基本信息中提取
                df = ak.stock_individual_info_em(symbol="000001")
                # 提取行业信息
                industry_info = df[df['item'] == '所属行业']
                if not industry_info.empty:
                    logger.info("从AKShare获取行业分类数据成功")
                else:
                    logger.warning("AKShare未提供完整的行业分类数据")
                    df = None
            except Exception as e:
                logger.error(f"从AKShare获取行业分类数据失败: {str(e)}")
        
        # 保存到缓存
        if df is not None:
            df.to_csv(cache_file, index=False)
            return df
        else:
            logger.error("获取行业分类数据失败，所有数据源都尝试失败")
            return pd.DataFrame()
    
    def get_backtest_data(self, stock_codes: List[str], start_date: str, end_date: str, 
                         adjust: str = "qfq") -> Dict[str, pd.DataFrame]:
        """
        获取回测所需的历史数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权方式
            
        Returns:
            字典，键为股票代码，值为历史数据DataFrame
        """
        backtest_data = {}
        
        for stock_code in stock_codes:
            try:
                logger.info(f"获取 {stock_code} 的历史数据...")
                
                # 获取日线数据
                df = self.get_daily_data(stock_code, start_date, end_date, adjust)
                
                if df is not None and len(df) > 0:
                    # 计算技术指标
                    df = self.calculate_indicators(df)
                    
                    # 添加到回测数据字典
                    backtest_data[stock_code] = df
                    
                    logger.info(f"成功获取 {stock_code} 的历史数据，共 {len(df)} 条记录")
                else:
                    logger.warning(f"未能获取 {stock_code} 的有效历史数据")
                
                # 防止频繁请求
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"处理 {stock_code} 的历史数据时出错: {str(e)}")
        
        return backtest_data 