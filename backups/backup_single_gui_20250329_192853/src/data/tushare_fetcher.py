#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tushare数据获取模块
专门处理Tushare数据源的数据获取
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tushare as ts
from ..config import DATA_SOURCE_CONFIG, CACHE_DIR

# 设置日志
logger = logging.getLogger(__name__)

class TushareFetcher:
    """Tushare数据获取类"""
    
    def __init__(self, config=None):
        """初始化Tushare数据获取器
        
        Args:
            config: 配置字典
        """
        # 导入依赖
        import tushare as ts
        import logging
        import os
        import time
        from datetime import datetime
        
        # 设置logger
        self.logger = logging.getLogger(__name__)
        
        # 读取配置
        if config is None:
            from src.config import DATA_SOURCE_CONFIG
            if 'tushare' in DATA_SOURCE_CONFIG:
                config = DATA_SOURCE_CONFIG['tushare']
            else:
                config = {}
        
        self.config = config
        self.token = config.get('token', '')
        self.timeout = config.get('timeout', 60)
        self.max_retry = config.get('max_retry', 3)
        self.retry_delay = config.get('retry_delay', 5)
        self.rate_limit = config.get('rate_limit', 300)  # 降低API调用频率
        self.concurrent_calls = config.get('concurrent_calls', 5)  # 降低并发调用数
        self.use_cache = config.get('use_cache', True)  # 添加缓存开关属性
        
        # API调用计数和时间记录
        self.api_call_count = 0
        self.api_call_times = []
        
        # 创建缓存目录
        from src.config import CACHE_DIR
        self.cache_dir = os.path.join(CACHE_DIR, 'tushare')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 初始化pro_api
        if self.token:
            try:
                ts.set_token(self.token)
                self.pro = ts.pro_api()
                self.logger.info("Tushare数据获取器初始化完成，token前5位: %s", self.token[:5])
            except Exception as e:
                self.pro = None
                self.logger.error("Tushare初始化失败: %s", str(e))
        else:
            self.pro = None
            self.logger.error("未提供Tushare token，Tushare数据源不可用")
        
    def api_call(self, func, **kwargs):
        """封装API调用，包含重试和错误处理
        
        Args:
            func: API函数
            **kwargs: 函数参数
            
        Returns:
            pandas.DataFrame: API调用结果
        """
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retry:
            try:
                # 检查API调用频率
                self._check_rate_limit()
                
                # 记录API调用
                self.api_call_count += 1
                
                # 调用API并记录时间
                start_time = time.time()
                result = func(**kwargs)
                elapsed_time = time.time() - start_time
                
                # 记录调用成功
                self.logger.debug(f"API调用成功，耗时 {elapsed_time:.2f} 秒")
                
                # 如果是空DataFrame，也视为失败
                if isinstance(result, pd.DataFrame) and result.empty:
                    self.logger.warning(f"API返回空DataFrame")
                    time.sleep(self.retry_delay)  # 在重试之前等待
                    retry_count += 1
                    continue
                
                # 添加随机延迟，避免连续请求过快
                import random
                delay = random.uniform(0.5, 1.5)
                time.sleep(delay)
                
                return result
                
            except Exception as e:
                retry_count += 1
                last_error = e
                
                if retry_count < self.max_retry:
                    # 指数增加重试延迟
                    delay = self.retry_delay * (2 ** (retry_count - 1))
                    self.logger.warning(f"API调用失败，{delay}秒后重试: {str(e)}")
                    time.sleep(delay)  # 重试前延迟
                else:
                    self.logger.error(f"达到最大重试次数，API调用失败: {str(e)}")
        
        # 所有重试都失败了
        self.logger.error(f"API调用最终失败: {str(last_error)}")
        return None
    
    def _check_rate_limit(self):
        """检查API调用频率，如果超过限制则等待"""
        # 如果没有设置限制，直接返回
        if not hasattr(self, 'rate_limit') or not self.rate_limit:
            return
        
        # 检查并计算需要等待的时间
        now = time.time()
        
        # 计算当前时间窗口内的调用次数
        window_size = 60  # 每分钟的API调用次数
        window_start = now - window_size
        
        # 删除旧的调用记录
        self.api_call_times = [t for t in self.api_call_times if t > window_start]
        
        # 如果当前时间窗口内的调用次数超过限制，等待
        if len(self.api_call_times) >= self.rate_limit:
            # 计算需要等待的时间
            oldest_call = self.api_call_times[0]
            wait_time = window_size - (now - oldest_call) + 1  # 额外等待1秒，以确保安全
            
            if wait_time > 0:
                self.logger.warning(f"API调用频率超过限制，等待 {wait_time:.2f} 秒")
                time.sleep(wait_time)
        
        # 记录当前调用时间
        self.api_call_times.append(time.time())
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表
        
        Returns:
            DataFrame: 股票列表，包含股票代码、名称、所属行业等信息
        """
        logger.info("获取股票列表")
        
        # 从缓存读取
        cache_file = os.path.join(self.cache_dir, 'stock_list.csv')
        if os.path.exists(cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(days=1):  # 缓存1天
                logger.info("从缓存读取股票列表")
                return pd.read_csv(cache_file)
        
        try:
            # 获取股票基本信息
            df = self.api_call(
                self.pro.stock_basic,
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date'
            )
            
            # 保存到缓存
            df.to_csv(cache_file, index=False)
            logger.info(f"获取到 {len(df)} 只股票信息")
            
            return df
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return None
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """获取日线数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD，默认为当前日期
            
        Returns:
            DataFrame: 日线数据
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"获取股票 {stock_code} 的日线数据")
        
        try:
            # 获取日线数据
            df = self.api_call(
                self.pro.daily,
                ts_code=stock_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                fields='ts_code,trade_date,open,high,low,close,vol,amount,pct_chg'
            )
            
            if df is not None and not df.empty:
                # 转换日期格式
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                # 按日期升序排序
                df = df.sort_values('trade_date')
                # 重命名列
                df = df.rename(columns={
                    'trade_date': 'date',
                    'vol': 'volume',
                    'pct_chg': 'pct_chg'  # 保留pct_chg字段
                })
                
                # 添加change字段作为pct_chg的副本，以保持兼容性
                df['change'] = df['pct_chg']
                
                logger.info(f"获取到 {len(df)} 条日线数据")
                return df
            
        except Exception as e:
            logger.error(f"获取日线数据失败: {e}")
            
        return None
    
    def get_industry_list(self) -> pd.DataFrame:
        """获取行业列表
        
        Returns:
            DataFrame: 行业列表
        """
        logger.info("获取行业列表")
        
        try:
            # 获取行业分类数据
            df = self.api_call(
                self.pro.stock_basic,
                fields='industry'
            )
            
            if df is not None and not df.empty:
                # 获取唯一的行业列表
                industries = df['industry'].unique()
                industry_df = pd.DataFrame({'industry': industries})
                logger.info(f"获取到 {len(industry_df)} 个行业")
                return industry_df
                
        except Exception as e:
            logger.error(f"获取行业列表失败: {e}")
            
        return None
    
    def get_stock_fund_flow(self, stock_code: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """获取个股资金流向数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD，默认为当前日期
            
        Returns:
            DataFrame: 资金流向数据
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"获取股票 {stock_code} 的资金流向数据")
        
        try:
            # 获取资金流向数据
            df = self.api_call(
                self.pro.moneyflow,
                ts_code=stock_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                fields='ts_code,trade_date,buy_sm_vol,buy_sm_amount,sell_sm_vol,sell_sm_amount,buy_md_vol,buy_md_amount,sell_md_vol,sell_md_amount,buy_lg_vol,buy_lg_amount,sell_lg_vol,sell_lg_amount,buy_elg_vol,buy_elg_amount,sell_elg_vol,sell_elg_amount,net_mf_vol,net_mf_amount'
            )
            
            if df is not None and not df.empty:
                # 转换日期格式
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                # 按日期升序排序
                df = df.sort_values('trade_date')
                # 重命名列
                df = df.rename(columns={
                    'trade_date': 'date'
                })
                
                logger.info(f"获取到 {len(df)} 条资金流向数据")
                return df
                
        except Exception as e:
            logger.error(f"获取资金流向数据失败: {e}")
            
        return None
    
    
    def get_today(self):
        """获取今天的日期，格式为YYYYMMDD"""
        return datetime.now().strftime('%Y%m%d')
    def get_continuous_limit_up_stocks(self, days=1, end_date=None):
        """
        获取截至指定日期连续N天涨停的股票
        
        Args:
            days (int): 连续涨停天数，默认为1
            end_date (str): 结束日期，格式为YYYYMMDD，默认为None，表示最新交易日
            
        Returns:
            pandas.DataFrame: 连续涨停股票数据，包含股票代码、名称等信息
        """
        self.logger.info(f"获取连续{days}天涨停的股票，截至 {end_date}")
        
        # 参数处理
        if end_date is None:
            end_date = self.get_today()
        elif isinstance(end_date, str):
            # 转换日期格式为YYYYMMDD
            if '-' in end_date:
                end_date = end_date.replace('-', '')

        # 创建缓存目录
        cache_dir = os.path.join(self.cache_dir, 'limit_up')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # 缓存文件名
        cache_file = os.path.join(cache_dir, f"limit_up_{days}d_{end_date}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file) and self.use_cache:
            self.logger.info(f"从缓存加载连续{days}天涨停数据: {cache_file}")
            try:
                return pd.read_csv(cache_file)
            except Exception as e:
                self.logger.error(f"读取缓存文件失败: {e}")
                # 缓存读取失败，继续执行获取数据的逻辑
        
        try:
            # 获取全部股票列表
            stock_list = self.get_stock_list()
            if stock_list is None or stock_list.empty:
                self.logger.error("获取股票列表失败")
                return pd.DataFrame()
                
            self.logger.info(f"分析 {len(stock_list)} 只股票的涨停情况")
            
            # 计算开始日期（往前多取几天的数据）
            start_date_obj = datetime.strptime(end_date, '%Y%m%d') - timedelta(days=days*3)
            start_date = start_date_obj.strftime('%Y%m%d')
            
            # 为提高效率，随机抽取部分股票测试
            # 在生产环境中应该扫描所有股票
            sample_size = min(300, len(stock_list))
            sample_stocks = stock_list.sample(sample_size)
            self.logger.info(f"抽样分析 {sample_size} 只股票")
            
            # 连续涨停股票列表
            continuous_limit_up_stocks = []
            
            # 遍历股票
            for idx, stock in sample_stocks.iterrows():
                ts_code = stock['ts_code']
                name = stock.get('name', '')
                
                # 定义连续重试机制
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    try:
                        # 获取股票历史数据
                        hist_data = self.api_call(
                            self.pro.daily,
                            ts_code=ts_code,
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        if hist_data is None or hist_data.empty or len(hist_data) < days:
                            self.logger.debug(f"股票 {ts_code} ({name}) 历史数据不足")
                            break
                            
                        # 按日期降序排序
                        hist_data = hist_data.sort_values('trade_date', ascending=False)
                        
                        # 根据股票类型确定涨停幅度
                        # 简化判断：
                        # - 创业板(300开头)、科创板(688开头): 阈值19.5%
                        # - 其他股票: 阈值9.5%
                        is_tech_stock = ts_code.startswith('300') or ts_code.startswith('688')
                        threshold = 19.5 if is_tech_stock else 9.5
                        
                        # 添加涨停标志
                        hist_data['is_limit_up'] = hist_data['pct_chg'] >= threshold
                        
                        # 检查连续涨停
                        is_continuous = True
                        limit_dates = []
                        
                        for i in range(days):
                            if i < len(hist_data):
                                if not hist_data.iloc[i]['is_limit_up']:
                                    is_continuous = False
                                    break
                                limit_dates.append(hist_data.iloc[i]['trade_date'])
                            else:
                                is_continuous = False
                                break
                        
                        if is_continuous:
                            self.logger.info(f"发现连续{days}天涨停股票: {ts_code} ({name})")
                            
                            # 获取行业信息
                            industry = stock.get('industry', '')
                            if not industry and 'industry' not in stock:
                                try:
                                    stock_info = self.api_call(
                                        self.pro.stock_basic,
                                        ts_code=ts_code,
                                        fields='industry'
                                    )
                                    if stock_info is not None and not stock_info.empty:
                                        industry = stock_info.iloc[0].get('industry', '')
                                except Exception as e:
                                    self.logger.warning(f"获取股票 {ts_code} 行业信息失败: {e}")
                            
                            # 添加到结果列表
                            stock_info = {
                                'ts_code': ts_code,
                                'name': name,
                                'industry': industry,
                                'close': hist_data.iloc[0]['close'],
                                'pct_chg': hist_data.iloc[0]['pct_chg'],
                                'vol': hist_data.iloc[0].get('vol', 0),
                                'amount': hist_data.iloc[0].get('amount', 0),
                                'threshold': threshold,
                                'limit_dates': ','.join(limit_dates)
                            }
                            continuous_limit_up_stocks.append(stock_info)
                        
                        # 成功获取数据，跳出重试循环
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        self.logger.warning(f"获取 {ts_code} 历史数据第 {retry_count} 次失败: {e}")
                        time.sleep(1)  # 等待1秒后重试
                        
                        if retry_count >= max_retries:
                            self.logger.error(f"获取 {ts_code} 历史数据失败，已达到最大重试次数")
                
            # 转换为DataFrame
            result_df = pd.DataFrame(continuous_limit_up_stocks)
            
            # 保存到缓存
            if not result_df.empty and self.use_cache:
                result_df.to_csv(cache_file, index=False)
                self.logger.info(f"已缓存连续{days}天涨停数据: {cache_file}")
            
            self.logger.info(f"找到 {len(result_df)} 只连续{days}天涨停的股票")
            return result_df
            
        except Exception as e:
            self.logger.error(f"获取连续涨停股票失败: {e}")
            return pd.DataFrame()
    
    def get_realtime_quotes(self, stock_codes: List[str]) -> pd.DataFrame:
        """获取实时行情数据
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            DataFrame: 实时行情数据
        """
        logger.info(f"获取 {len(stock_codes)} 只股票的实时行情")
        
        try:
            # 获取实时行情
            df = self.api_call(
                self.pro.daily,
                ts_code=','.join(stock_codes),
                trade_date=datetime.now().strftime('%Y%m%d'),
                fields='ts_code,trade_date,open,high,low,close,vol,amount,pct_chg'
            )
            
            if df is not None and not df.empty:
                logger.info(f"获取到 {len(df)} 只股票的实时行情")
                return df
                
        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")
            
        return None
    
    def get_stock_indicators(self, stock_code: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """获取股票技术指标数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD，默认为当前日期
            
        Returns:
            DataFrame: 技术指标数据
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"获取股票 {stock_code} 的技术指标数据")
        
        try:
            # 获取技术指标数据
            df = self.api_call(
                self.pro.daily_basic,
                ts_code=stock_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                fields='ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv'
            )
            
            if df is not None and not df.empty:
                # 转换日期格式
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                # 按日期升序排序
                df = df.sort_values('trade_date')
                # 重命名列
                df = df.rename(columns={
                    'trade_date': 'date',
                    'turnover_rate': 'turnover',
                    'volume_ratio': 'vol_ratio',
                    'total_mv': 'market_cap',
                    'circ_mv': 'float_market_cap'
                })
                
                logger.info(f"获取到 {len(df)} 条技术指标数据")
                return df
                
        except Exception as e:
            logger.error(f"获取技术指标数据失败: {e}")
            
        return None 