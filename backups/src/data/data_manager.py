#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据管理模块
负责数据的获取、存储和管理
"""

import os
import pandas as pd
import numpy as np
import json
import hashlib
import logging
from datetime import datetime, timedelta
import time
import random
import traceback
import glob
import shutil
from typing import Optional, List, Dict, Any
import tushare as ts
import akshare as ak
from pathlib import Path
from ..utils.logger import get_logger
from ..config.settings import (
    DATA_SOURCES,
    DATA_FETCH_CONFIG,
    DATA_DIR,
    CACHE_DIR
)

# 设置日志
logger = logging.getLogger(__name__)

# 导入配置
try:
    from ..config import CACHE_DIR, DATA_DIR
except ImportError:
    # 使用默认值
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    CACHE_DIR = os.path.join(DATA_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

class DataManager:
    """数据管理类"""
    
    def __init__(self):
        """初始化数据管理器"""
        self.logger = get_logger("DataManager")
        self._init_data_sources()
        self._setup_directories()
        
    def _init_data_sources(self) -> None:
        """初始化数据源"""
        # 初始化Tushare
        # 使用直接提供的token
        tushare_token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
        
        if tushare_token:
            ts.set_token(tushare_token)
            self.pro = ts.pro_api()
            self.logger.info("Tushare token配置成功")
        else:
            # 尝试从配置文件获取
            if DATA_SOURCES["tushare"]["token"]:
                ts.set_token(DATA_SOURCES["tushare"]["token"])
                self.pro = ts.pro_api()
            else:
                self.logger.warning("未配置Tushare token，部分功能可能无法使用")
            
    def _setup_directories(self) -> None:
        """设置数据目录"""
        # 创建数据目录
        self.stock_data_dir = DATA_DIR / "stock_data"
        self.stock_data_dir.mkdir(exist_ok=True)
        
        # 创建缓存目录
        self.cache_dir = CACHE_DIR / "data_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_stock_data(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source: str = "tushare"
    ) -> Optional[pd.DataFrame]:
        """
        获取股票数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            source: 数据源，可选：tushare, akshare
            
        Returns:
            pd.DataFrame: 股票数据，包含以下列：
                - date: 日期
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - volume: 成交量
                - amount: 成交额
        """
        try:
            # 检查缓存
            cache_file = self.cache_dir / f"{stock_code}_{start_date}_{end_date}.csv"
            if cache_file.exists():
                self.logger.info(f"从缓存加载数据: {stock_code}")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            
            # 获取数据
            if source == "tushare":
                df = self._get_stock_data_tushare(stock_code, start_date, end_date)
            elif source == "akshare":
                df = self._get_stock_data_akshare(stock_code, start_date, end_date)
            else:
                raise ValueError(f"不支持的数据源: {source}")
            
            if df is not None and not df.empty:
                # 保存到缓存
                df.to_csv(cache_file)
                self.logger.info(f"数据已缓存: {stock_code}")
                
            return df
            
        except Exception as e:
            self.logger.error(f"获取股票数据失败: {stock_code}", exc_info=e)
            return None
            
    def _get_stock_data_tushare(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """从Tushare获取股票数据"""
        try:
            if start_date is None:
                start_date = DATA_FETCH_CONFIG["default_start_date"]
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")
                
            # 获取日线数据
            df = self.pro.daily(
                ts_code=stock_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", "")
            )
            
            if df is not None and not df.empty:
                # 重命名列
                df = df.rename(columns={
                    "trade_date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "vol": "volume",
                    "amount": "amount"
                })
                
                # 设置日期索引
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                
                # 按日期排序
                df.sort_index(inplace=True)
                
                return df
                
            return None
            
        except Exception as e:
            self.logger.error(f"从Tushare获取数据失败: {stock_code}", exc_info=e)
            return None
            
    def _get_stock_data_akshare(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """从AKShare获取股票数据"""
        try:
            if start_date is None:
                start_date = DATA_FETCH_CONFIG["default_start_date"]
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")
                
            # 获取日线数据
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            
            if df is not None and not df.empty:
                # 重命名列
                df = df.rename(columns={
                    "日期": "date",
                    "开盘": "open",
                    "最高": "high",
                    "最低": "low",
                    "收盘": "close",
                    "成交量": "volume",
                    "成交额": "amount"
                })
                
                # 设置日期索引
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                
                # 按日期排序
                df.sort_index(inplace=True)
                
                return df
                
            return None
            
        except Exception as e:
            self.logger.error(f"从AKShare获取数据失败: {stock_code}", exc_info=e)
            return None
            
    def get_stock_list(self, source: str = "tushare") -> Optional[pd.DataFrame]:
        """
        获取股票列表
        
        Args:
            source: 数据源，可选：tushare, akshare
            
        Returns:
            pd.DataFrame: 股票列表，包含以下列：
                - code: 股票代码
                - name: 股票名称
                - industry: 所属行业
                - market: 市场类型
        """
        try:
            if source == "tushare":
                df = self._get_stock_list_tushare()
            elif source == "akshare":
                df = self._get_stock_list_akshare()
            else:
                raise ValueError(f"不支持的数据源: {source}")
                
            if df is not None and not df.empty:
                # 保存到文件
                output_file = self.stock_data_dir / "stock_list.csv"
                df.to_csv(output_file)
                self.logger.info("股票列表已更新")
                
            return df
            
        except Exception as e:
            self.logger.error("获取股票列表失败", exc_info=e)
            return None
            
    def _get_stock_list_tushare(self) -> Optional[pd.DataFrame]:
        """从Tushare获取股票列表"""
        try:
            df = self.pro.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,list_date"
            )
            
            if df is not None and not df.empty:
                # 重命名列
                df = df.rename(columns={
                    "ts_code": "code",
                    "symbol": "symbol",
                    "name": "name",
                    "area": "area",
                    "industry": "industry",
                    "list_date": "list_date"
                })
                
                return df
                
            return None
            
        except Exception as e:
            self.logger.error("从Tushare获取股票列表失败", exc_info=e)
            return None
            
    def _get_stock_list_akshare(self) -> Optional[pd.DataFrame]:
        """从AKShare获取股票列表"""
        try:
            df = ak.stock_zh_a_spot_em()
            
            if df is not None and not df.empty:
                # 重命名列
                df = df.rename(columns={
                    "代码": "code",
                    "名称": "name",
                    "行业": "industry",
                    "市场类型": "market"
                })
                
                return df
                
            return None
            
        except Exception as e:
            self.logger.error("从AKShare获取股票列表失败", exc_info=e)
            return None
            
    def refresh_data(self) -> None:
        """刷新数据"""
        try:
            self.logger.info("开始刷新数据")
            
            # 更新股票列表
            self.get_stock_list()
            
            # 获取股票列表
            stock_list_file = self.stock_data_dir / "stock_list.csv"
            if stock_list_file.exists():
                stock_list = pd.read_csv(stock_list_file)
                
                # 更新每只股票的数据
                for _, row in stock_list.iterrows():
                    stock_code = row["code"]
                    self.logger.info(f"更新股票数据: {stock_code}")
                    
                    # 获取数据
                    df = self.get_stock_data(stock_code)
                    
                    if df is not None and not df.empty:
                        # 保存数据
                        output_file = self.stock_data_dir / f"{stock_code}.csv"
                        df.to_csv(output_file)
                        
                    # 避免请求过于频繁
                    time.sleep(1)
                    
            self.logger.info("数据刷新完成")
            
        except Exception as e:
            self.logger.error("刷新数据失败", exc_info=e)
            
    def get_stock_info(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        获取股票基本信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 股票基本信息
        """
        try:
            # 获取股票列表
            stock_list_file = self.stock_data_dir / "stock_list.csv"
            if stock_list_file.exists():
                stock_list = pd.read_csv(stock_list_file)
                stock_info = stock_list[stock_list["code"] == stock_code].iloc[0]
                return stock_info.to_dict()
                
            return None
            
        except Exception as e:
            self.logger.error(f"获取股票信息失败: {stock_code}", exc_info=e)
            return None