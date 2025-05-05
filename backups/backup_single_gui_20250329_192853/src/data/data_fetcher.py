#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import akshare as ak
import tushare as ts
import requests
from concurrent.futures import ThreadPoolExecutor
import time
import json
from technical_indicators import TechnicalIndicators
from data_quality import DataQuality

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataFetcher:
    def __init__(self):
        # 初始化Tushare
        self.ts_token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
        ts.set_token(self.ts_token)
        self.pro = ts.pro_api()
        
        # Alpha Vantage配置
        self.alpha_vantage_key = "H1GWJ27Z3CCWL5DP"
        self.alpha_vantage_base_url = "https://www.alphavantage.co/query"
        
        # 设置请求间隔（避免触发API限制）
        self.request_delay = 0.5
        
        # 初始化缓存目录
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_stock_data(self, stock_code, start_date, end_date, use_cache=True):
        """
        从多个数据源获取股票数据，并进行数据整合和清洗
        """
        try:
            # 验证输入参数
            if not DataQuality.validate_stock_code(stock_code):
                raise ValueError("无效的股票代码")
                
            valid, message = DataQuality.validate_date_range(start_date, end_date)
            if not valid:
                raise ValueError(message)
            
            # 检查缓存
            cache_file = os.path.join(self.cache_dir, f"{stock_code}_{start_date}_{end_date}.csv")
            if use_cache and os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col='date', parse_dates=True)
                logging.info(f"从缓存加载数据: {cache_file}")
            else:
                # 创建线程池
                with ThreadPoolExecutor(max_workers=3) as executor:
                    # 并行获取数据
                    future_akshare = executor.submit(self._get_akshare_data, stock_code, start_date, end_date)
                    future_tushare = executor.submit(self._get_tushare_data, stock_code, start_date, end_date)
                    future_alpha = executor.submit(self._get_alpha_vantage_data, stock_code)
                    
                    # 获取结果
                    df_akshare = future_akshare.result()
                    df_tushare = future_tushare.result()
                    df_alpha = future_alpha.result()
                
                # 数据整合和清洗
                df = self._merge_and_clean_data(df_akshare, df_tushare, df_alpha)
                
                # 保存到缓存
                if df is not None and use_cache:
                    df.to_csv(cache_file)
                    logging.info(f"数据已缓存: {cache_file}")
            
            if df is not None:
                # 添加技术指标
                df = TechnicalIndicators.add_all_indicators(df)
                
                # 检查数据质量
                quality_report = DataQuality.check_data_quality(df, stock_code)
                if quality_report['status'] == 'error':
                    logging.error(f"数据质量检查失败: {quality_report['message']}")
                elif quality_report['status'] == 'warning':
                    logging.warning(f"数据质量警告: {len(quality_report['warnings'])} 个问题")
                
                # 获取技术指标信号
                signals = TechnicalIndicators.get_indicator_signals(df)
                logging.info(f"技术指标信号: {json.dumps(signals, indent=2)}")
            
            return df, quality_report, signals
            
        except Exception as e:
            logging.error(f"获取股票数据失败: {str(e)}")
            return None, None, None
    
    def _get_akshare_data(self, stock_code, start_date, end_date):
        """从AKShare获取数据"""
        try:
            # 获取A股历史数据
            df = ak.stock_zh_a_hist(symbol=stock_code, 
                                  start_date=start_date,
                                  end_date=end_date,
                                  adjust="qfq")
            
            # 选择需要的列并重命名
            df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额']]
            df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']
            
            # 设置日期索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"AKShare数据获取失败: {str(e)}")
            return None
    
    def _get_tushare_data(self, stock_code, start_date, end_date):
        """从Tushare获取数据"""
        try:
            # 获取日线数据
            df = self.pro.daily(ts_code=stock_code, 
                              start_date=start_date,
                              end_date=end_date)
            
            # 选择需要的列并重命名
            df = df[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']]
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
            
            # 设置日期索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Tushare数据获取失败: {str(e)}")
            return None
    
    def _get_alpha_vantage_data(self, stock_code):
        """从Alpha Vantage获取数据"""
        try:
            # 构建请求URL
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': stock_code,
                'apikey': self.alpha_vantage_key
            }
            
            # 发送请求
            response = requests.get(self.alpha_vantage_base_url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                return None
                
            # 转换为DataFrame
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            
            # 重命名列
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # 转换数据类型
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            return df
            
        except Exception as e:
            logging.error(f"Alpha Vantage数据获取失败: {str(e)}")
            return None
    
    def _merge_and_clean_data(self, df_akshare, df_tushare, df_alpha):
        """合并和清洗数据"""
        try:
            # 选择最完整的数据源作为基础
            if df_akshare is not None and not df_akshare.empty:
                df_base = df_akshare.copy()
            elif df_tushare is not None and not df_tushare.empty:
                df_base = df_tushare.copy()
            elif df_alpha is not None and not df_alpha.empty:
                df_base = df_alpha.copy()
            else:
                return None
            
            # 数据清洗
            # 1. 删除重复数据
            df_base = df_base[~df_base.index.duplicated(keep='first')]
            
            # 2. 处理缺失值
            df_base = df_base.fillna(method='ffill').fillna(method='bfill')
            
            # 3. 异常值处理
            for col in ['open', 'high', 'low', 'close']:
                if col in df_base.columns:
                    # 使用3个标准差作为异常值判断标准
                    mean = df_base[col].mean()
                    std = df_base[col].std()
                    df_base[col] = df_base[col].clip(mean - 3*std, mean + 3*std)
            
            return df_base
            
        except Exception as e:
            logging.error(f"数据合并和清洗失败: {str(e)}")
            return None
    
    def get_stock_list(self):
        """获取股票列表"""
        try:
            # 从AKShare获取A股列表
            df = ak.stock_zh_a_spot_em()
            
            # 选择需要的列并重命名
            df = df[['代码', '名称', '最新价', '涨跌幅', '成交量', '成交额']]
            df.columns = ['code', 'name', 'price', 'change_pct', 'volume', 'amount']
            
            return df
            
        except Exception as e:
            logging.error(f"获取股票列表失败: {str(e)}")
            return None
    
    def get_sector_data(self):
        """获取行业板块数据"""
        try:
            # 从AKShare获取行业板块数据
            df = ak.stock_board_industry_name_em()
            
            # 选择需要的列并重命名
            df = df[['板块名称', '涨跌幅', '换手率', '总市值', '领涨股票']]
            df.columns = ['name', 'change_pct', 'turnover', 'market_value', 'leader']
            
            # 转换数据类型
            for col in ['change_pct', 'turnover']:
                df[col] = df[col].astype(str).str.strip('%').astype(float)
            
            return df
            
        except Exception as e:
            logging.error(f"获取行业板块数据失败: {str(e)}")
            return None

def main():
    # 测试数据获取
    fetcher = DataFetcher()
    
    # 测试获取股票数据
    stock_code = "000001"  # 平安银行
    start_date = "20240101"
    end_date = "20240325"
    
    logging.info(f"开始获取{stock_code}的数据...")
    df, quality_report, signals = fetcher.get_stock_data(stock_code, start_date, end_date)
    
    if df is not None:
        logging.info(f"成功获取数据，共{len(df)}条记录")
        logging.info(f"数据预览：\n{df.head()}")
        logging.info(f"数据质量报告：\n{json.dumps(quality_report, indent=2)}")
        logging.info(f"技术指标信号：\n{json.dumps(signals, indent=2)}")
    else:
        logging.error("获取数据失败")
    
    # 测试获取股票列表
    logging.info("\n获取股票列表...")
    stock_list = fetcher.get_stock_list()
    if stock_list is not None:
        logging.info(f"成功获取股票列表，共{len(stock_list)}只股票")
        logging.info(f"列表预览：\n{stock_list.head()}")
    else:
        logging.error("获取股票列表失败")
    
    # 测试获取行业板块数据
    logging.info("\n获取行业板块数据...")
    sector_data = fetcher.get_sector_data()
    if sector_data is not None:
        logging.info(f"成功获取行业板块数据，共{len(sector_data)}个板块")
        logging.info(f"数据预览：\n{sector_data.head()}")
    else:
        logging.error("获取行业板块数据失败")

if __name__ == "__main__":
    main() 