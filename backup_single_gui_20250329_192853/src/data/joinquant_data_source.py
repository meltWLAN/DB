#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JoinQuant数据源模块
用于从JoinQuant获取A股历史数据
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, List, Any, Optional, Union

# 设置日志
logger = logging.getLogger(__name__)

class JoinQuantDataSource:
    """JoinQuant数据源类，提供对JoinQuant数据的访问"""
    
    def __init__(self, 
                 username: str = None, 
                 password: str = None,
                 cache_dir: str = "./data/cache/joinquant"):
        """
        初始化JoinQuant数据源
        
        Args:
            username: JoinQuant用户名
            password: JoinQuant密码
            cache_dir: 数据缓存目录
        """
        self.username = username
        self.password = password
        self.cache_dir = cache_dir
        self.jq_client = None
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 尝试登录JoinQuant
        self._login()
    
    def _login(self):
        """登录JoinQuant"""
        try:
            # 尝试导入JoinQuant模块
            import jqdatasdk as jq
            
            # 登录
            if self.username and self.password:
                jq.auth(self.username, self.password)
                self.jq_client = jq
                
                # 验证登录状态
                is_auth = jq.is_auth()
                if is_auth:
                    logger.info("JoinQuant登录成功")
                    
                    # 查询剩余调用次数
                    quota_info = jq.get_query_count()
                    logger.info(f"当日剩余调用额度: 数据点 {quota_info['spare_count']}, 财务数据点 {quota_info['spare_financial_count']}")
                else:
                    logger.error("JoinQuant登录失败，请检查用户名和密码")
                    self.jq_client = None
            else:
                logger.warning("未提供JoinQuant用户名和密码，跳过登录")
                self.jq_client = None
                
        except ImportError:
            logger.error("未安装JoinQuant SDK，请使用 pip install jqdatasdk 安装")
            self.jq_client = None
        except Exception as e:
            logger.error(f"JoinQuant登录出错: {str(e)}")
            self.jq_client = None
    
    def is_available(self) -> bool:
        """
        检查JoinQuant数据源是否可用
        
        Returns:
            bool: 是否可用
        """
        if self.jq_client is None:
            return False
        
        try:
            # 验证登录状态
            import jqdatasdk as jq
            return jq.is_auth()
        except:
            return False
    
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        
        Returns:
            股票列表DataFrame，包含代码、名称等信息
        """
        cache_file = os.path.join(self.cache_dir, "stock_list.csv")
        
        # 检查缓存
        if os.path.exists(cache_file) and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days < 1:
            logger.info("从缓存加载股票列表")
            return pd.read_csv(cache_file)
        
        if not self.is_available():
            logger.error("JoinQuant数据源不可用")
            return pd.DataFrame()
        
        try:
            # 获取所有A股股票列表
            stocks = self.jq_client.get_all_securities(['stock'])
            
            # 只保留状态为"正常"的股票
            stocks = stocks[stocks['start_date'] <= datetime.now()]
            stocks = stocks[stocks['end_date'] >= datetime.now()]
            
            # 重命名列
            stocks = stocks.reset_index()
            stocks.rename(columns={
                'index': 'code',
                'display_name': 'name',
                'start_date': 'list_date',
                'type': 'security_type'
            }, inplace=True)
            
            # 添加ts_code字段，与tushare保持一致
            stocks['ts_code'] = stocks['code'].apply(
                lambda x: x.replace('.XSHE', '.SZ') if 'XSHE' in x else x.replace('.XSHG', '.SH')
            )
            
            # 获取行业信息
            industry_info = self.jq_client.get_industry(stocks['code'].tolist())
            
            # 提取申万一级行业
            industry_dict = {}
            for code, industries in industry_info.items():
                if 'sw_l1' in industries:
                    industry_dict[code] = industries['sw_l1']['industry_name']
                else:
                    industry_dict[code] = '其他'
            
            # 添加行业信息
            stocks['industry'] = stocks['code'].map(industry_dict)
            
            # 保存到缓存
            stocks.to_csv(cache_file, index=False)
            
            logger.info(f"从JoinQuant获取股票列表成功，共 {len(stocks)} 只股票")
            return stocks
            
        except Exception as e:
            logger.error(f"从JoinQuant获取股票列表失败: {str(e)}")
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
        # 转换代码格式为JoinQuant格式
        jq_code = self._convert_to_jq_code(stock_code)
        
        # 缓存文件名
        cache_key = f"{stock_code}_{start_date}_{end_date}_{adjust}"
        cache_file = os.path.join(self.cache_dir, f"daily_{cache_key.replace('.', '_')}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file) and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days < 1:
            logger.info(f"从缓存加载 {stock_code} 的日线数据")
            df = pd.read_csv(cache_file)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        
        if not self.is_available():
            logger.error("JoinQuant数据源不可用")
            return pd.DataFrame()
        
        try:
            # 设置复权类型
            if adjust == 'qfq':
                jq_adjust = 'pre'
            elif adjust == 'hfq':
                jq_adjust = 'post'
            else:
                jq_adjust = None
            
            # 获取日线数据
            df = self.jq_client.get_price(
                jq_code,
                start_date=start_date,
                end_date=end_date,
                frequency='daily',
                fields=['open', 'close', 'high', 'low', 'volume', 'money'],
                skip_paused=False,
                fq_ref_date=None,
                panel=False,
                fill_paused=True,
                adjust_type=jq_adjust
            )
            
            # 重命名列
            df.rename(columns={
                'money': 'amount'
            }, inplace=True)
            
            # 将索引改为date
            df = df.reset_index()
            df.rename(columns={'index': 'date'}, inplace=True)
            
            # 获取累计净值
            if adjust is not None:
                df['factor'] = self.jq_client.get_price_factors(
                    jq_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields=['factor'],
                    panel=False
                )['factor']
            
            # 设置索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 保存到缓存
            df.reset_index().to_csv(cache_file, index=False)
            
            logger.info(f"从JoinQuant获取 {stock_code} 的日线数据成功，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"从JoinQuant获取 {stock_code} 的日线数据失败: {str(e)}")
            return pd.DataFrame()
    
    def _convert_to_jq_code(self, stock_code: str) -> str:
        """
        将通用代码格式转换为JoinQuant格式
        
        Args:
            stock_code: 通用格式股票代码（如000001.SZ）
            
        Returns:
            JoinQuant格式股票代码（如000001.XSHE）
        """
        if "." not in stock_code:
            # 根据首位判断市场
            if stock_code.startswith("6"):
                return f"{stock_code}.XSHG"
            else:
                return f"{stock_code}.XSHE"
        
        # 已经包含后缀的情况
        code_parts = stock_code.split(".")
        code = code_parts[0]
        market = code_parts[1].upper()
        
        if market == "SH":
            return f"{code}.XSHG"
        elif market == "SZ":
            return f"{code}.XSHE"
        elif market in ["XSHG", "XSHE"]:
            return stock_code
        else:
            # 未知市场，尝试根据代码首位判断
            if code.startswith("6"):
                return f"{code}.XSHG"
            else:
                return f"{code}.XSHE"
    
    def _convert_from_jq_code(self, jq_code: str) -> str:
        """
        将JoinQuant格式转换为通用代码格式
        
        Args:
            jq_code: JoinQuant格式股票代码（如000001.XSHE）
            
        Returns:
            通用格式股票代码（如000001.SZ）
        """
        if "." not in jq_code:
            return jq_code
        
        code_parts = jq_code.split(".")
        code = code_parts[0]
        market = code_parts[1].upper()
        
        if market == "XSHG":
            return f"{code}.SH"
        elif market == "XSHE":
            return f"{code}.SZ"
        else:
            return jq_code
    
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
        
        if not self.is_available():
            logger.error("JoinQuant数据源不可用")
            return pd.DataFrame()
        
        try:
            # 获取股票列表
            stocks = self.get_stock_list()
            if stocks.empty:
                return pd.DataFrame()
            
            # 获取申万一级行业信息
            industry_info = self.jq_client.get_industry(stocks['code'].tolist(), date=datetime.now())
            
            # 构建行业数据
            data = []
            for code, industries in industry_info.items():
                if 'sw_l1' in industries:
                    data.append({
                        'code': code,
                        'ts_code': self._convert_from_jq_code(code),
                        'industry': industries['sw_l1']['industry_name'],
                        'industry_code': industries['sw_l1']['industry_code']
                    })
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 保存到缓存
            df.to_csv(cache_file, index=False)
            
            logger.info("从JoinQuant获取行业分类数据成功")
            return df
            
        except Exception as e:
            logger.error(f"从JoinQuant获取行业分类数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_fundamentals(self, stock_code: str, date: str = None) -> pd.DataFrame:
        """
        获取股票基本面数据
        
        Args:
            stock_code: 股票代码
            date: 日期，默认为最新日期
            
        Returns:
            基本面数据DataFrame
        """
        if not self.is_available():
            logger.error("JoinQuant数据源不可用")
            return pd.DataFrame()
        
        jq_code = self._convert_to_jq_code(stock_code)
        
        try:
            # 设置查询日期
            query_date = date if date else datetime.now().strftime('%Y-%m-%d')
            
            # 查询财务指标
            import jqdatasdk as jq
            q = jq.query(
                jq.valuation.code,
                jq.valuation.pe_ratio,
                jq.valuation.pe_ratio_lyr,
                jq.valuation.pb_ratio,
                jq.valuation.ps_ratio,
                jq.valuation.pcf_ratio,
                jq.valuation.market_cap,
                jq.valuation.circulating_market_cap
            ).filter(jq.valuation.code == jq_code)
            
            df = self.jq_client.get_fundamentals(q, date=query_date)
            
            if df.empty:
                logger.warning(f"未获取到 {stock_code} 的基本面数据")
                return pd.DataFrame()
            
            # 重命名列
            df.rename(columns={
                'pe_ratio': 'pe',
                'pe_ratio_lyr': 'pe_ttm',
                'pb_ratio': 'pb',
                'ps_ratio': 'ps',
                'pcf_ratio': 'pcf',
                'market_cap': 'total_mv',
                'circulating_market_cap': 'circ_mv'
            }, inplace=True)
            
            # 添加日期列
            df['date'] = query_date
            
            # 转换代码
            df['ts_code'] = df['code'].apply(self._convert_from_jq_code)
            
            logger.info(f"从JoinQuant获取 {stock_code} 的基本面数据成功")
            return df
            
        except Exception as e:
            logger.error(f"从JoinQuant获取 {stock_code} 的基本面数据失败: {str(e)}")
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