#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票数据管理模块
负责股票数据的获取、存储、更新和验证，确保系统始终使用高质量的真实数据
消除对模拟数据的依赖
"""

import os
import sys
import time
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue
import requests
import json

# 导入数据源
import tushare as ts

# 尝试导入其他可选数据源
try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    BAOSTOCK_AVAILABLE = False

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 导入项目配置
try:
    from src.enhanced.config.settings import TUSHARE_TOKEN, LOG_DIR, DATA_DIR
except ImportError:
    # 设置默认配置
    TUSHARE_TOKEN = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    LOG_DIR = "./logs"
    DATA_DIR = "./data"

# 确保目录存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "stocks"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "indices"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "fundamentals"), exist_ok=True)

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 添加文件处理器
file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, f"stock_data_manager_{datetime.now().strftime('%Y%m%d')}.log"),
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 确保logger没有重复的处理器
if not logger.handlers:
    logger.addHandler(file_handler)

class StockDataManager:
    """股票数据管理类，负责数据获取、存储和更新"""
    
    def __init__(self, use_db=True, db_path=None):
        """初始化数据管理器
        
        Args:
            use_db: 是否使用本地数据库
            db_path: 数据库路径，默认为None，将使用默认路径
        """
        self.use_db = use_db
        self.db_path = db_path or os.path.join(DATA_DIR, "stock_data.db")
        
        # 初始化数据源
        self.data_sources = self._init_data_sources()
        
        # 初始化数据库
        if self.use_db:
            self._init_database()
        
        # 数据缓存
        self.cache = {
            'stock_list': None,
            'stock_data': {},
            'index_data': {}
        }
        
        # 缓存有效期（秒）
        self.cache_expiry = {
            'stock_list': 86400,  # 股票列表缓存1天
            'stock_data': 3600,   # 股票数据缓存1小时
            'index_data': 1800    # 指数数据缓存30分钟
        }
        
        # 缓存时间戳
        self.cache_timestamp = {
            'stock_list': 0,
            'stock_data': {},
            'index_data': {}
        }
        
        # 下载队列和线程
        self.download_queue = queue.Queue()
        self.is_downloading = False
        self.download_thread = None
        
        logger.info("股票数据管理器初始化完成")
    
    def _init_data_sources(self):
        """初始化数据源"""
        sources = {}
        
        # 初始化Tushare
        if TUSHARE_TOKEN:
            try:
                ts.set_token(TUSHARE_TOKEN)
                sources['tushare'] = ts.pro_api()
                logger.info(f"Tushare API初始化成功 (Token前5位: {TUSHARE_TOKEN[:5]}...)")
            except Exception as e:
                logger.error(f"Tushare API初始化失败: {str(e)}")
        
        # 初始化Baostock
        if BAOSTOCK_AVAILABLE:
            try:
                lg = bs.login()
                if lg.error_code == '0':
                    sources['baostock'] = bs
                    logger.info("Baostock API初始化成功")
                else:
                    logger.error(f"Baostock登录失败: {lg.error_msg}")
            except Exception as e:
                logger.error(f"Baostock API初始化失败: {str(e)}")
        
        # 初始化AKShare
        if AKSHARE_AVAILABLE:
            try:
                # AKShare不需要登录
                sources['akshare'] = ak
                logger.info("AKShare API初始化成功")
            except Exception as e:
                logger.error(f"AKShare API初始化失败: {str(e)}")
        
        return sources
    
    def _init_database(self):
        """初始化SQLite数据库"""
        try:
            # 连接数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建股票列表表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_list (
                ts_code TEXT PRIMARY KEY,
                symbol TEXT,
                name TEXT,
                area TEXT,
                industry TEXT,
                market TEXT,
                list_date TEXT,
                update_time TEXT
            )
            ''')
            
            # 创建股票日线数据表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_code TEXT,
                trade_date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                pre_close REAL,
                change REAL,
                pct_chg REAL,
                vol REAL,
                amount REAL,
                UNIQUE(ts_code, trade_date)
            )
            ''')
            
            # 创建指数数据表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_code TEXT,
                trade_date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                pre_close REAL,
                change REAL,
                pct_chg REAL,
                vol REAL,
                amount REAL,
                UNIQUE(index_code, trade_date)
            )
            ''')
            
            # 创建数据下载记录表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS download_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT,
                code TEXT,
                start_date TEXT,
                end_date TEXT,
                status TEXT,
                message TEXT,
                download_time TEXT
            )
            ''')
            
            # 创建指标
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_ts_code ON daily_data (ts_code)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_trade_date ON daily_data (trade_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_index_code ON index_data (index_code)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"数据库初始化成功: {self.db_path}")
        except Exception as e:
            logger.error(f"数据库初始化失败: {str(e)}")
            raise

    def get_stock_list(self, use_cache=True, force_update=False):
        """获取股票列表
        
        Args:
            use_cache: 是否使用缓存
            force_update: 是否强制更新
            
        Returns:
            DataFrame: 股票列表
        """
        # 检查缓存
        current_time = time.time()
        if use_cache and self.cache['stock_list'] is not None and not force_update:
            # 缓存未过期
            if current_time - self.cache_timestamp['stock_list'] < self.cache_expiry['stock_list']:
                logger.debug("使用缓存的股票列表")
                return self.cache['stock_list']
        
        # 从数据库获取
        if self.use_db and not force_update:
            try:
                conn = sqlite3.connect(self.db_path)
                query = "SELECT * FROM stock_list"
                df = pd.read_sql(query, conn)
                conn.close()
                
                if not df.empty:
                    # 更新缓存
                    self.cache['stock_list'] = df
                    self.cache_timestamp['stock_list'] = current_time
                    logger.info(f"从数据库获取股票列表成功，共 {len(df)} 支股票")
                    return df
            except Exception as e:
                logger.error(f"从数据库获取股票列表失败: {str(e)}")
        
        # 从API获取
        df = self._fetch_stock_list_from_api()
        
        if df is not None and not df.empty:
            # 保存到数据库
            if self.use_db:
                try:
                    conn = sqlite3.connect(self.db_path)
                    # 添加更新时间
                    df['update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # 删除旧数据
                    conn.execute("DELETE FROM stock_list")
                    # 写入新数据
                    df.to_sql('stock_list', conn, if_exists='append', index=False)
                    conn.close()
                    logger.info(f"股票列表已保存到数据库，共 {len(df)} 支股票")
                except Exception as e:
                    logger.error(f"保存股票列表到数据库失败: {str(e)}")
            
            # 更新缓存
            self.cache['stock_list'] = df
            self.cache_timestamp['stock_list'] = current_time
            
            return df
        else:
            logger.error("获取股票列表失败")
            # 返回空DataFrame
            return pd.DataFrame()
    
    def _fetch_stock_list_from_api(self):
        """从API获取股票列表"""
        # 尝试所有可用的数据源
        if 'tushare' in self.data_sources:
            try:
                logger.info("从Tushare获取股票列表")
                pro = self.data_sources['tushare']
                df = pro.stock_basic(exchange='', list_status='L',
                                     fields='ts_code,symbol,name,area,industry,list_date,market')
                if not df.empty:
                    logger.info(f"从Tushare获取股票列表成功，共 {len(df)} 支股票")
                    return df
            except Exception as e:
                logger.error(f"从Tushare获取股票列表失败: {str(e)}")
        
        if 'baostock' in self.data_sources and BAOSTOCK_AVAILABLE:
            try:
                logger.info("从Baostock获取股票列表")
                rs = bs.query_stock_basic()
                if rs.error_code == '0':
                    data_list = []
                    while (rs.error_code == '0') & rs.next():
                        data_list.append(rs.get_row_data())
                    df = pd.DataFrame(data_list, columns=rs.fields)
                    # 转换为与Tushare相似的格式
                    df.rename(columns={
                        'code': 'ts_code',
                        'code_name': 'name',
                        'ipoDate': 'list_date'
                    }, inplace=True)
                    # 添加额外列
                    df['symbol'] = df['ts_code'].apply(lambda x: x.split('.')[0])
                    df['area'] = ''
                    df['industry'] = ''
                    df['market'] = df['ts_code'].apply(lambda x: 'SH' if x.endswith('.SH') else 'SZ')
                    
                    if not df.empty:
                        logger.info(f"从Baostock获取股票列表成功，共 {len(df)} 支股票")
                        return df
            except Exception as e:
                logger.error(f"从Baostock获取股票列表失败: {str(e)}")
        
        if 'akshare' in self.data_sources and AKSHARE_AVAILABLE:
            try:
                logger.info("从AKShare获取股票列表")
                # 获取A股股票列表
                df = ak.stock_info_a_code_name()
                if not df.empty:
                    # 转换为与Tushare相似的格式
                    df.rename(columns={
                        '证券代码': 'symbol',
                        '证券简称': 'name'
                    }, inplace=True)
                    # 添加额外列
                    df['ts_code'] = df['symbol'].apply(
                        lambda x: x + '.SH' if x.startswith('6') else x + '.SZ'
                    )
                    df['area'] = ''
                    df['industry'] = ''
                    df['list_date'] = ''
                    df['market'] = df['symbol'].apply(
                        lambda x: 'SH' if x.startswith('6') else 'SZ'
                    )
                    
                    logger.info(f"从AKShare获取股票列表成功，共 {len(df)} 支股票")
                    return df
            except Exception as e:
                logger.error(f"从AKShare获取股票列表失败: {str(e)}")
        
        logger.error("所有API获取股票列表均失败")
        return None
    
    def get_stock_data(self, ts_code, start_date=None, end_date=None, use_cache=True):
        """获取股票日线数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期，默认为None，将获取尽可能多的历史数据
            end_date: 结束日期，默认为None，将使用当前日期
            use_cache: 是否使用缓存
            
        Returns:
            DataFrame: 股票日线数据
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        if start_date is None:
            # 默认获取2年数据
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')
        
        # 检查缓存
        cache_key = f"{ts_code}_{start_date}_{end_date}"
        current_time = time.time()
        
        if use_cache and cache_key in self.cache['stock_data']:
            # 缓存未过期
            if current_time - self.cache_timestamp['stock_data'].get(cache_key, 0) < self.cache_expiry['stock_data']:
                logger.debug(f"使用缓存的股票数据: {ts_code}")
                return self.cache['stock_data'][cache_key]
        
        # 从数据库获取
        if self.use_db:
            try:
                conn = sqlite3.connect(self.db_path)
                query = f"""
                SELECT * FROM daily_data 
                WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date DESC
                """
                df = pd.read_sql(query, conn, params=(ts_code, start_date, end_date))
                conn.close()
                
                if not df.empty:
                    # 设置索引
                    if 'trade_date' in df.columns:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        df.set_index('trade_date', inplace=True)
                    
                    # 更新缓存
                    self.cache['stock_data'][cache_key] = df
                    self.cache_timestamp['stock_data'][cache_key] = current_time
                    
                    logger.info(f"从数据库获取 {ts_code} 数据成功，共 {len(df)} 条记录")
                    
                    # 检查数据是否最新
                    latest_date_str = df.index[0].strftime('%Y%m%d') if not df.empty else '19900101'
                    latest_date = datetime.strptime(latest_date_str, '%Y%m%d')
                    today = datetime.now()
                    
                    # 如果最新数据不是今天或昨天，尝试更新数据
                    if (today - latest_date).days > 1 and today.weekday() < 5:  # 工作日才更新
                        logger.info(f"{ts_code} 的数据不是最新的，尝试更新...")
                        self._update_stock_data(ts_code, latest_date_str, end_date)
                        # 重新获取数据
                        return self.get_stock_data(ts_code, start_date, end_date, use_cache=False)
                    
                    return df
            except Exception as e:
                logger.error(f"从数据库获取 {ts_code} 数据失败: {str(e)}")
        
        # 从API获取
        df = self._fetch_stock_data_from_api(ts_code, start_date, end_date)
        
        if df is not None and not df.empty:
            # 保存到数据库
            if self.use_db:
                try:
                    conn = sqlite3.connect(self.db_path)
                    # 转换日期格式以便存储
                    df_to_save = df.reset_index()
                    if 'trade_date' in df_to_save.columns:
                        df_to_save['trade_date'] = df_to_save['trade_date'].dt.strftime('%Y%m%d')
                    elif df_to_save.index.name == 'trade_date':
                        df_to_save['trade_date'] = df_to_save.index.strftime('%Y%m%d')
                    
                    # 使用REPLACE防止重复
                    df_to_save.to_sql('daily_data', conn, if_exists='replace', index=False)
                    conn.close()
                    logger.info(f"{ts_code} 数据已保存到数据库，共 {len(df)} 条记录")
                except Exception as e:
                    logger.error(f"保存 {ts_code} 数据到数据库失败: {str(e)}")
            
            # 更新缓存
            self.cache['stock_data'][cache_key] = df
            self.cache_timestamp['stock_data'][cache_key] = current_time
            
            return df
        else:
            logger.error(f"获取 {ts_code} 数据失败")
            # 返回空DataFrame
            return pd.DataFrame()
    
    def _fetch_stock_data_from_api(self, ts_code, start_date, end_date):
        """从API获取股票数据"""
        # 尝试所有可用的数据源
        if 'tushare' in self.data_sources:
            try:
                logger.info(f"从Tushare获取 {ts_code} 从 {start_date} 到 {end_date} 的数据")
                pro = self.data_sources['tushare']
                
                # 使用pro.daily获取日线数据
                df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                
                if not df.empty:
                    # 处理日期
                    if 'trade_date' in df.columns:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        df.sort_values('trade_date', inplace=True, ascending=False)
                        df.set_index('trade_date', inplace=True)
                    
                    logger.info(f"从Tushare获取 {ts_code} 数据成功，共 {len(df)} 条记录")
                    return df
            except Exception as e:
                logger.error(f"从Tushare获取 {ts_code} 数据失败: {str(e)}")
        
        if 'baostock' in self.data_sources and BAOSTOCK_AVAILABLE:
            try:
                logger.info(f"从Baostock获取 {ts_code} 从 {start_date} 到 {end_date} 的数据")
                # 转换日期格式
                start_date_fmt = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
                end_date_fmt = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
                
                # 调整代码格式 (Tushare: 000001.SZ, Baostock: sz.000001)
                if ts_code.endswith('.SH'):
                    bs_code = f"sh.{ts_code.split('.')[0]}"
                else:
                    bs_code = f"sz.{ts_code.split('.')[0]}"
                
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,open,high,low,close,preclose,volume,amount,turn,pctChg",
                    start_date=start_date_fmt,
                    end_date=end_date_fmt,
                    frequency="d",
                    adjustflag="3"  # 前复权
                )
                
                if rs.error_code == '0':
                    data_list = []
                    while (rs.error_code == '0') & rs.next():
                        data_list.append(rs.get_row_data())
                    df = pd.DataFrame(data_list, columns=rs.fields)
                    
                    # 转换为与Tushare相似的格式
                    df.rename(columns={
                        'date': 'trade_date',
                        'volume': 'vol',
                        'pctChg': 'pct_chg',
                        'preclose': 'pre_close'
                    }, inplace=True)
                    
                    # 转换数据类型
                    for col in ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount', 'pct_chg']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # 处理日期
                    if 'trade_date' in df.columns:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        df.sort_values('trade_date', inplace=True, ascending=False)
                        df.set_index('trade_date', inplace=True)
                    
                    # 添加ts_code列
                    df['ts_code'] = ts_code
                    
                    if not df.empty:
                        logger.info(f"从Baostock获取 {ts_code} 数据成功，共 {len(df)} 条记录")
                        return df
            except Exception as e:
                logger.error(f"从Baostock获取 {ts_code} 数据失败: {str(e)}")
        
        if 'akshare' in self.data_sources and AKSHARE_AVAILABLE:
            try:
                logger.info(f"从AKShare获取 {ts_code} 数据")
                
                # 转换代码格式 (Tushare: 000001.SZ, AKShare: 000001)
                symbol = ts_code.split('.')[0]
                
                # 获取日线数据
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                         start_date=start_date, end_date=end_date,
                                         adjust="qfq")  # 前复权
                
                if not df.empty:
                    # 转换为与Tushare相似的格式
                    df.rename(columns={
                        '日期': 'trade_date',
                        '开盘': 'open',
                        '收盘': 'close',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'vol',
                        '成交额': 'amount',
                        '涨跌幅': 'pct_chg',
                        '涨跌额': 'change',
                        '换手率': 'turnover'
                    }, inplace=True)
                    
                    # 计算pre_close
                    df['pre_close'] = df['close'] / (1 + df['pct_chg'] / 100)
                    
                    # 处理日期
                    if 'trade_date' in df.columns:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        df.sort_values('trade_date', inplace=True, ascending=False)
                        df.set_index('trade_date', inplace=True)
                    
                    # 添加ts_code列
                    df['ts_code'] = ts_code
                    
                    logger.info(f"从AKShare获取 {ts_code} 数据成功，共 {len(df)} 条记录")
                    return df
            except Exception as e:
                logger.error(f"从AKShare获取 {ts_code} 数据失败: {str(e)}")
        
        logger.error(f"所有API获取 {ts_code} 数据均失败")
        return None
    
    def _update_stock_data(self, ts_code, start_date, end_date):
        """更新单个股票数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            bool: 是否更新成功
        """
        logger.info(f"更新 {ts_code} 从 {start_date} 到 {end_date} 的数据")
        
        # 从API获取新数据
        df = self._fetch_stock_data_from_api(ts_code, start_date, end_date)
        
        if df is not None and not df.empty:
            # 保存到数据库
            if self.use_db:
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # 删除已有的重复数据
                    cursor.execute(
                        "DELETE FROM daily_data WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?",
                        (ts_code, start_date, end_date)
                    )
                    
                    # 转换日期格式以便存储
                    df_to_save = df.reset_index()
                    if 'trade_date' in df_to_save.columns:
                        df_to_save['trade_date'] = df_to_save['trade_date'].dt.strftime('%Y%m%d')
                    
                    # 保存新数据
                    df_to_save.to_sql('daily_data', conn, if_exists='append', index=False)
                    
                    # 记录下载历史
                    cursor.execute(
                        "INSERT INTO download_history (data_type, code, start_date, end_date, status, message, download_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        ('daily', ts_code, start_date, end_date, 'success', f'获取了{len(df)}条记录', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    )
                    
                    conn.commit()
                    conn.close()
                    
                    logger.info(f"{ts_code} 数据更新成功，共 {len(df)} 条记录")
                    return True
                except Exception as e:
                    logger.error(f"保存 {ts_code} 更新数据到数据库失败: {str(e)}")
                    return False
        else:
            logger.error(f"更新 {ts_code} 数据失败，无法获取新数据")
            
            # 记录失败历史
            if self.use_db:
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO download_history (data_type, code, start_date, end_date, status, message, download_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        ('daily', ts_code, start_date, end_date, 'fail', '无法获取新数据', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    logger.error(f"记录下载历史失败: {str(e)}")
            
            return False
    
    def update_all_stock_data(self):
        """更新所有股票数据
        
        从最近一次更新日期开始，更新所有股票的最新数据
        
        Returns:
            dict: 更新结果统计
        """
        logger.info("开始更新所有股票数据")
        
        # 获取股票列表
        stock_list = self.get_stock_list()
        
        if stock_list.empty:
            logger.error("股票列表为空，无法更新数据")
            return {'status': 'error', 'message': '股票列表为空'}
        
        total = len(stock_list)
        success_count = 0
        fail_count = 0
        skip_count = 0
        
        # 获取当前日期
        today = datetime.now().strftime('%Y%m%d')
        
        # 启动后台下载线程
        self._start_download_thread()
        
        # 对每支股票，查找最新数据日期，然后更新至今
        for idx, (_, stock) in enumerate(stock_list.iterrows()):
            ts_code = stock['ts_code']
            
            # 查询最新数据日期
            if self.use_db:
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT MAX(trade_date) FROM daily_data WHERE ts_code = ?",
                        (ts_code,)
                    )
                    result = cursor.fetchone()
                    conn.close()
                    
                    if result[0]:
                        last_date = result[0]
                        # 转换为日期对象
                        last_date_obj = datetime.strptime(last_date, '%Y%m%d')
                        # 计算与今天的间隔
                        days_diff = (datetime.now() - last_date_obj).days
                        
                        # 如果最后更新日期是今天或昨天，则跳过
                        if days_diff <= 1:
                            logger.info(f"[{idx+1}/{total}] {ts_code} 数据已是最新，跳过更新")
                            skip_count += 1
                            continue
                        
                        # 从最后一天的下一天开始更新
                        start_date = (last_date_obj + timedelta(days=1)).strftime('%Y%m%d')
                    else:
                        # 无历史数据，获取1年数据
                        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                    
                    # 将更新任务添加到队列
                    logger.info(f"[{idx+1}/{total}] 将 {ts_code} 数据更新任务添加到队列 (从 {start_date} 到 {today})")
                    self.download_queue.put((ts_code, start_date, today))
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"查询 {ts_code} 最新数据日期失败: {str(e)}")
                    fail_count += 1
            else:
                # 无数据库，直接获取近期数据
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                
                # 将更新任务添加到队列
                logger.info(f"[{idx+1}/{total}] 将 {ts_code} 数据更新任务添加到队列 (从 {start_date} 到 {today})")
                self.download_queue.put((ts_code, start_date, today))
                success_count += 1
        
        logger.info(f"所有股票数据更新任务已添加到队列，共 {total} 支，成功 {success_count}，失败 {fail_count}，跳过 {skip_count}")
        
        return {
            'status': 'success',
            'total': total,
            'success': success_count,
            'fail': fail_count,
            'skip': skip_count,
            'message': f'已将{success_count}支股票的更新任务添加到队列'
        }
    
    def _start_download_thread(self):
        """启动后台下载线程"""
        if not self.is_downloading:
            self.is_downloading = True
            self.download_thread = threading.Thread(target=self._download_worker)
            self.download_thread.daemon = True
            self.download_thread.start()
            logger.info("后台下载线程已启动")
    
    def _download_worker(self):
        """后台下载工作线程"""
        logger.info("下载工作线程开始运行")
        
        try:
            while True:
                try:
                    # 从队列获取任务，设置超时以便检查队列是否为空
                    ts_code, start_date, end_date = self.download_queue.get(timeout=1)
                    
                    # 更新数据
                    success = self._update_stock_data(ts_code, start_date, end_date)
                    
                    # 标记任务完成
                    self.download_queue.task_done()
                    
                    # 添加间隔以避免API限制
                    time.sleep(0.5)
                    
                except queue.Empty:
                    # 队列为空，检查是否应该退出
                    if self.download_queue.qsize() == 0:
                        logger.info("下载队列为空，下载工作线程退出")
                        break
                except Exception as e:
                    logger.error(f"下载工作线程处理任务时出错: {str(e)}")
                    # 继续处理下一个任务
                    continue
        finally:
            self.is_downloading = False
            logger.info("下载工作线程已退出")
    
    def check_data_integrity(self, ts_code=None):
        """检查数据完整性
        
        Args:
            ts_code: 股票代码，默认为None，将检查所有股票
            
        Returns:
            dict: 检查结果
        """
        logger.info(f"开始检查数据完整性 {ts_code if ts_code else '所有股票'}")
        
        if not self.use_db:
            logger.error("未使用数据库，无法检查数据完整性")
            return {'status': 'error', 'message': '未使用数据库，无法检查数据完整性'}
        
        # 获取交易日历
        trading_days = self._get_trading_calendar()
        if not trading_days:
            logger.error("获取交易日历失败，无法检查数据完整性")
            return {'status': 'error', 'message': '获取交易日历失败'}
        
        # 确定要检查的股票
        if ts_code:
            stock_codes = [ts_code]
        else:
            # 获取数据库中所有股票代码
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT ts_code FROM daily_data")
                stock_codes = [row[0] for row in cursor.fetchall()]
                conn.close()
            except Exception as e:
                logger.error(f"获取股票代码列表失败: {str(e)}")
                return {'status': 'error', 'message': f'获取股票代码列表失败: {str(e)}'}
        
        total = len(stock_codes)
        results = []
        
        for idx, ts_code in enumerate(stock_codes):
            logger.info(f"[{idx+1}/{total}] 检查 {ts_code} 数据完整性")
            
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 获取该股票的所有交易日
                cursor.execute(
                    "SELECT trade_date FROM daily_data WHERE ts_code = ? ORDER BY trade_date",
                    (ts_code,)
                )
                stock_days = [row[0] for row in cursor.fetchall()]
                
                # 计算应有的交易日数量
                # 获取该股票的上市日期
                cursor.execute(
                    "SELECT list_date FROM stock_list WHERE ts_code = ?",
                    (ts_code,)
                )
                result = cursor.fetchone()
                
                if result and result[0]:
                    list_date = result[0]
                    # 转换为日期对象
                    list_date_obj = datetime.strptime(list_date, '%Y%m%d')
                    # 找出大于上市日期的交易日
                    expected_days = [d for d in trading_days if datetime.strptime(d, '%Y%m%d') >= list_date_obj]
                else:
                    # 无上市日期信息，使用第一个交易日
                    if stock_days:
                        first_day = stock_days[0]
                        expected_days = [d for d in trading_days if d >= first_day]
                    else:
                        expected_days = []
                
                # 计算缺失的交易日
                missing_days = set(expected_days) - set(stock_days)
                
                conn.close()
                
                result = {
                    'ts_code': ts_code,
                    'total_days': len(stock_days),
                    'expected_days': len(expected_days),
                    'missing_days': len(missing_days),
                    'completeness': round(len(stock_days) / max(1, len(expected_days)) * 100, 2),
                    'missing_dates': sorted(list(missing_days))[:10]  # 只返回前10个缺失日期
                }
                
                results.append(result)
                
                logger.info(f"{ts_code} 数据完整性: {result['completeness']}%，共有{result['missing_days']}个缺失交易日")
                
                # 如果缺失数据较多且未指定特定股票，添加修复任务
                if result['missing_days'] > 0 and not ts_code:
                    # 按年份分组修复，避免一次性请求过多数据
                    missing_dates_by_year = {}
                    for date in missing_days:
                        year = date[:4]
                        if year not in missing_dates_by_year:
                            missing_dates_by_year[year] = []
                        missing_dates_by_year[year].append(date)
                    
                    for year, dates in missing_dates_by_year.items():
                        if dates:
                            # 将每年的缺失日期按区间合并
                            start_date = min(dates)
                            end_date = max(dates)
                            
                            logger.info(f"添加 {ts_code} {year}年缺失数据修复任务 (从 {start_date} 到 {end_date})")
                            self.download_queue.put((ts_code, start_date, end_date))
                
            except Exception as e:
                logger.error(f"检查 {ts_code} 数据完整性失败: {str(e)}")
                results.append({
                    'ts_code': ts_code,
                    'error': str(e)
                })
        
        # 启动下载线程处理修复任务
        if self.download_queue.qsize() > 0:
            self._start_download_thread()
        
        return {
            'status': 'success',
            'total_stocks': total,
            'results': results,
            'summary': {
                'avg_completeness': round(sum(r['completeness'] for r in results if 'completeness' in r) / max(1, len(results)), 2),
                'total_missing': sum(r['missing_days'] for r in results if 'missing_days' in r),
                'repair_tasks': self.download_queue.qsize()
            }
        }
    
    def _get_trading_calendar(self):
        """获取交易日历"""
        logger.info("获取交易日历")
        
        # 首先尝试从数据库获取
        if self.use_db:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 检查是否存在交易日历表
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trading_calendar'")
                if cursor.fetchone():
                    # 获取交易日历
                    cursor.execute("SELECT trade_date FROM trading_calendar WHERE is_open=1 ORDER BY trade_date")
                    trading_days = [row[0] for row in cursor.fetchall()]
                    
                    if trading_days:
                        conn.close()
                        logger.info(f"从数据库获取交易日历成功，共 {len(trading_days)} 个交易日")
                        return trading_days
                
                conn.close()
            except Exception as e:
                logger.error(f"从数据库获取交易日历失败: {str(e)}")
        
        # 从API获取
        trading_days = []
        
        if 'tushare' in self.data_sources:
            try:
                logger.info("从Tushare获取交易日历")
                pro = self.data_sources['tushare']
                
                # 获取2000年至今的交易日历
                start_year = '2000'
                end_year = datetime.now().strftime('%Y')
                
                # 按年获取，避免一次性获取过多数据
                all_days = []
                
                for year in range(int(start_year), int(end_year) + 1):
                    df = pro.trade_cal(exchange='SSE', start_date=f'{year}0101', end_date=f'{year}1231')
                    if not df.empty:
                        # 只保留交易日
                        df = df[df['is_open'] == 1]
                        all_days.extend(df['cal_date'].tolist())
                
                trading_days = sorted(all_days)
                
                # 保存到数据库
                if self.use_db:
                    try:
                        conn = sqlite3.connect(self.db_path)
                        cursor = conn.cursor()
                        
                        # 创建交易日历表
                        cursor.execute('''
                        CREATE TABLE IF NOT EXISTS trading_calendar (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            trade_date TEXT,
                            is_open INTEGER,
                            UNIQUE(trade_date)
                        )
                        ''')
                        
                        # 插入数据
                        for date in trading_days:
                            cursor.execute(
                                "INSERT OR REPLACE INTO trading_calendar (trade_date, is_open) VALUES (?, ?)",
                                (date, 1)
                            )
                        
                        conn.commit()
                        conn.close()
                        
                        logger.info(f"交易日历已保存到数据库，共 {len(trading_days)} 个交易日")
                    except Exception as e:
                        logger.error(f"保存交易日历到数据库失败: {str(e)}")
                
                return trading_days
            except Exception as e:
                logger.error(f"从Tushare获取交易日历失败: {str(e)}")
        
        # 如果无法获取交易日历，创建一个模拟交易日历
        if not trading_days:
            logger.warning("无法获取真实交易日历，创建模拟交易日历")
            
            # 从2000年至今，生成所有工作日
            start_date = datetime(2000, 1, 1)
            end_date = datetime.now()
            
            current_date = start_date
            while current_date <= end_date:
                # 如果是工作日(0-4对应周一至周五)
                if current_date.weekday() < 5:
                    trading_days.append(current_date.strftime('%Y%m%d'))
                current_date += timedelta(days=1)
        
        return trading_days

# 创建单例实例
stock_data_manager = StockDataManager()

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据管理器实例
    manager = StockDataManager()
    
    # 测试获取股票列表
    stock_list = manager.get_stock_list()
    print(f"获取到 {len(stock_list)} 支股票")
    
    # 测试获取股票数据
    if not stock_list.empty:
        test_stock = stock_list.iloc[0]['ts_code']
        print(f"测试获取 {test_stock} 的数据:")
        data = manager.get_stock_data(test_stock)
        print(f"获取到 {len(data)} 条数据") 