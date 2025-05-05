#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import pandas as pd
import time
from datetime import datetime, timedelta
import akshare as ak

from src.config import CACHE_DIR

logger = logging.getLogger(__name__)

class AKShareFetcher:
    """AKShare数据获取类"""
    
    def __init__(self, config=None):
        """初始化AKShare数据获取器
        
        Args:
            config: 配置字典
        """
        # 读取配置
        if config is None:
            from src.config import DATA_SOURCE_CONFIG
            if 'akshare' in DATA_SOURCE_CONFIG:
                config = DATA_SOURCE_CONFIG['akshare']
            else:
                config = {}
        
        self.config = config
        self.timeout = config.get('timeout', 60)
        self.max_retry = config.get('max_retry', 3)
        self.retry_delay = config.get('retry_delay', 5)
        
        # API调用计数和时间记录
        self.api_call_count = 0
        self.api_call_times = []
        
        # 设置logger
        self.logger = logging.getLogger(__name__)
        
        # 创建缓存目录
        from src.config import CACHE_DIR
        self.cache_dir = os.path.join(CACHE_DIR, 'akshare')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.logger.info("AKShare数据获取器初始化完成")
    
    def is_enabled(self):
        """检查数据源是否可用"""
        return self.config.get('enable', True)
    
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
                
                return result
                
            except Exception as e:
                retry_count += 1
                last_error = e
                
                if retry_count < self.max_retry:
                    self.logger.warning(f"API调用失败，{self.retry_delay}秒后重试: {str(e)}")
                    time.sleep(self.retry_delay)  # 重试前延迟
                else:
                    self.logger.error(f"达到最大重试次数，API调用失败: {str(e)}")
        
        # 所有重试都失败了
        self.logger.error(f"API调用最终失败: {str(last_error)}")
        return None
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表
        
        Returns:
            DataFrame: 股票列表
        """
        self.logger.info("获取股票列表")
        
        # 检查缓存
        cache_file = os.path.join(self.cache_dir, "stock_list.csv")
        if os.path.exists(cache_file):
            # 如果缓存文件存在且不超过1天，直接读取缓存
            file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_mtime).days < 1:
                try:
                    self.logger.info("从缓存读取股票列表")
                    df = pd.read_csv(cache_file)
                    if not df.empty:
                        return df
                except Exception as e:
                    self.logger.warning(f"读取缓存文件失败: {e}")
        
        # 使用akshare获取A股股票列表
        try:
            df = self.api_call(ak.stock_info_a_code_name)
            
            # 处理列名，使其与Tushare一致
            if not df.empty:
                df = df.rename(columns={
                    '代码': 'symbol',
                    '名称': 'name'
                })
                # 添加ts_code字段
                df['ts_code'] = df['symbol'].apply(
                    lambda x: f"{x}.SH" if x.startswith('6') else f"{x}.SZ"
                )
                # 保存到缓存
                df.to_csv(cache_file, index=False)
                self.logger.info(f"获取到 {len(df)} 只股票")
                return df
            else:
                self.logger.warning("AKShare返回空的股票列表")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def get_daily_data(self, stock_code, start_date, end_date=None):
        """获取日线数据
        
        Args:
            stock_code: 股票代码，格式：000001.SZ
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD，默认为今天
            
        Returns:
            DataFrame: 日线数据
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        self.logger.info(f"获取股票 {stock_code} 的日线数据")
        
        # 检查缓存
        cache_file = os.path.join(self.cache_dir, f"{stock_code}_{start_date}_{end_date}.csv")
        if os.path.exists(cache_file):
            # 如果缓存文件存在且不超过1天，直接读取缓存
            file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_mtime).days < 1:
                try:
                    self.logger.info("从缓存读取日线数据")
                    df = pd.read_csv(cache_file)
                    if not df.empty:
                        return df
                except Exception as e:
                    self.logger.warning(f"读取缓存文件失败: {e}")
        
        # 转换股票代码格式为akshare格式
        if '.' in stock_code:
            symbol = stock_code.split('.')[0]
        else:
            symbol = stock_code
            
        try:
            # 使用akshare获取股票日线数据
            df = self.api_call(
                ak.stock_zh_a_hist,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            if df is not None and not df.empty:
                # 处理列名，使其与Tushare一致
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '涨跌幅': 'pct_chg',
                    '涨跌额': 'change'
                })
                
                # 添加ts_code字段
                df['ts_code'] = stock_code
                
                # 确保日期格式正确
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                
                # 保存到缓存
                df.to_csv(cache_file, index=False)
                
                self.logger.info(f"获取到 {len(df)} 条日线数据")
                return df
            else:
                self.logger.warning(f"获取股票 {stock_code} 日线数据返回空结果")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"获取日线数据失败: {e}")
            return pd.DataFrame()
    
    def get_industry_list(self):
        """获取行业列表
        
        Returns:
            DataFrame: 行业列表
        """
        self.logger.info("获取行业列表")
        
        # 检查缓存
        cache_file = os.path.join(self.cache_dir, "industry_list.csv")
        if os.path.exists(cache_file):
            # 如果缓存文件存在且不超过7天，直接读取缓存
            file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_mtime).days < 7:
                try:
                    self.logger.info("从缓存读取行业列表")
                    df = pd.read_csv(cache_file)
                    if not df.empty:
                        return df
                except Exception as e:
                    self.logger.warning(f"读取缓存文件失败: {e}")
        
        try:
            # 使用akshare获取行业列表
            stock_list = self.get_stock_list()
            if stock_list is None or stock_list.empty:
                self.logger.warning("无法获取股票列表，无法生成行业列表")
                return pd.DataFrame()
                
            # 从同花顺行业分类获取行业数据
            industries = self.api_call(ak.stock_board_industry_name_ths)
            
            if industries is not None and not industries.empty:
                # 提取行业名称列
                df = pd.DataFrame({'industry': industries['name'].unique()})
                
                # 保存到缓存
                df.to_csv(cache_file, index=False)
                
                self.logger.info(f"获取到 {len(df)} 个行业")
                return df
            else:
                self.logger.warning("获取行业列表返回空结果")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"获取行业列表失败: {e}")
            return pd.DataFrame()
    
    def get_continuous_limit_up_stocks(self, days=1, end_date=None):
        """获取连续涨停股票
        
        Args:
            days: 连续涨停天数
            end_date: 结束日期，默认为最新交易日
            
        Returns:
            DataFrame: 连续涨停股票数据
        """
        self.logger.info(f"获取连续{days}天涨停的股票")
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # 计算开始日期（往前多取几天的数据）
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days*3)).strftime('%Y-%m-%d')
        
        # 检查缓存
        cache_file = os.path.join(self.cache_dir, f"limit_up_{days}d_{end_date}.csv")
        if os.path.exists(cache_file):
            # 如果缓存文件存在且不超过1天，直接读取缓存
            file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_mtime).days < 1:
                try:
                    self.logger.info("从缓存读取涨停股票数据")
                    df = pd.read_csv(cache_file)
                    if not df.empty:
                        return df
                except Exception as e:
                    self.logger.warning(f"读取缓存文件失败: {e}")
        
        try:
            # 获取当天涨停股票
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
            trade_date = end_date_dt.strftime('%Y%m%d')
            
            # 使用akshare获取涨停股票
            limit_up_df = self.api_call(
                ak.stock_em_zt_pool,
                date=trade_date
            )
            
            if limit_up_df is None or limit_up_df.empty:
                self.logger.warning(f"{end_date} 没有涨停股票数据")
                return pd.DataFrame()
                
            # 如果只需要一天的涨停股票，直接返回
            if days == 1:
                # 处理列名，使其与Tushare一致
                result_df = limit_up_df.rename(columns={
                    '代码': 'symbol',
                    '名称': 'name',
                    '涨跌幅': 'pct_chg',
                    '最新价': 'close'
                })
                
                # 添加ts_code字段
                result_df['ts_code'] = result_df['symbol'].apply(
                    lambda x: f"{x}.SH" if x.startswith('6') else f"{x}.SZ"
                )
                
                # 保存到缓存
                result_df.to_csv(cache_file, index=False)
                
                self.logger.info(f"获取到 {len(result_df)} 只连续1天涨停的股票")
                return result_df
            
            # 如果需要连续多天涨停，需要获取历史数据进行筛选
            # 获取股票列表
            stock_list = self.get_stock_list()
            if stock_list is None or stock_list.empty:
                self.logger.warning("无法获取股票列表，无法筛选连续涨停股票")
                return pd.DataFrame()
                
            # 在涨停股票中筛选出连续涨停的
            continuous_limit_up_stocks = []
            
            for _, stock in limit_up_df.iterrows():
                symbol = stock['代码']
                # 获取该股票的历史数据
                if '.' not in symbol:
                    if symbol.startswith('6'):
                        ts_code = f"{symbol}.SH"
                    else:
                        ts_code = f"{symbol}.SZ"
                else:
                    ts_code = symbol
                
                # 获取日线数据
                hist_data = self.get_daily_data(ts_code, start_date, end_date)
                
                if hist_data is None or hist_data.empty or len(hist_data) < days:
                    continue
                
                # 按日期降序排序
                hist_data = hist_data.sort_values('date', ascending=False)
                
                # 检查是否连续涨停
                is_continuous = True
                for i in range(days):
                    if i < len(hist_data):
                        # 涨停标准：涨幅大于等于9.5%
                        if hist_data.iloc[i]['pct_chg'] < 9.5:
                            is_continuous = False
                            break
                    else:
                        is_continuous = False
                        break
                
                if is_continuous:
                    # 获取股票名称等信息
                    stock_info = stock_list[stock_list['symbol'] == symbol]
                    if not stock_info.empty:
                        name = stock_info.iloc[0]['name']
                        industry = stock_info.iloc[0].get('industry', '未知')
                    else:
                        name = stock['名称'] if '名称' in stock else '未知'
                        industry = '未知'
                        
                    continuous_limit_up_stocks.append({
                        'ts_code': ts_code,
                        'symbol': symbol,
                        'name': name,
                        'industry': industry,
                        'close': stock['最新价'] if '最新价' in stock else hist_data.iloc[0]['close'],
                        'pct_chg': stock['涨跌幅'] if '涨跌幅' in stock else hist_data.iloc[0]['pct_chg']
                    })
            
            # 转换为DataFrame
            result_df = pd.DataFrame(continuous_limit_up_stocks)
            
            # 保存到缓存
            if not result_df.empty:
                result_df.to_csv(cache_file, index=False)
                
            self.logger.info(f"获取到 {len(result_df)} 只连续{days}天涨停的股票")
            return result_df
                
        except Exception as e:
            self.logger.error(f"获取连续涨停股票失败: {e}")
            return pd.DataFrame()
    
    def get_stock_fund_flow(self, stock_code, start_date, end_date=None):
        """获取个股资金流向数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期，默认为今天
            
        Returns:
            DataFrame: 资金流向数据
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        self.logger.info(f"获取股票 {stock_code} 的资金流向数据")
        
        # 检查缓存
        cache_file = os.path.join(self.cache_dir, f"{stock_code}_fund_flow_{start_date}_{end_date}.csv")
        if os.path.exists(cache_file):
            # 如果缓存文件存在且不超过1天，直接读取缓存
            file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_mtime).days < 1:
                try:
                    self.logger.info("从缓存读取资金流向数据")
                    df = pd.read_csv(cache_file)
                    if not df.empty:
                        return df
                except Exception as e:
                    self.logger.warning(f"读取缓存文件失败: {e}")
        
        # 转换股票代码格式为akshare格式
        if '.' in stock_code:
            symbol = stock_code.split('.')[0]
        else:
            symbol = stock_code
            
        try:
            # 使用akshare获取个股资金流向
            df = self.api_call(
                ak.stock_individual_fund_flow,
                stock=symbol
            )
            
            if df is not None and not df.empty:
                # 处理列名，使其与Tushare一致
                df = df.rename(columns={
                    '日期': 'date',
                    '收盘价': 'close',
                    '涨跌幅': 'pct_chg',
                    '主力净流入': 'net_inflow',
                    '主力净流入占比': 'net_inflow_rate',
                    '超大单净流入': 'super_inflow',
                    '超大单净流入占比': 'super_inflow_rate',
                    '大单净流入': 'big_inflow',
                    '大单净流入占比': 'big_inflow_rate',
                    '中单净流入': 'medium_inflow',
                    '中单净流入占比': 'medium_inflow_rate',
                    '小单净流入': 'small_inflow',
                    '小单净流入占比': 'small_inflow_rate'
                })
                
                # 添加ts_code字段
                df['ts_code'] = stock_code
                
                # 确保日期格式正确
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                
                # 过滤日期范围
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                # 保存到缓存
                df.to_csv(cache_file, index=False)
                
                self.logger.info(f"获取到 {len(df)} 条资金流向数据")
                return df
            else:
                self.logger.warning(f"获取股票 {stock_code} 资金流向数据返回空结果")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"获取资金流向数据失败: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 测试
    fetcher = AKShareFetcher()
    stock_list = fetcher.get_stock_list()
    print(f"获取到 {len(stock_list)} 只股票")
    print(stock_list.head()) 