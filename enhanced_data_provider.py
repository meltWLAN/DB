#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强数据提供者模块
整合多数据源支持，提供统一的数据获取接口，并确保数据质量
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import traceback
import json

# 自定义异常类
class DataNotAvailableError(Exception):
    """数据不可用异常"""
    pass

class DataQualityError(Exception):
    """数据质量异常"""
    pass

class EnhancedDataProvider:
    """增强数据提供者类"""
    
    def __init__(self, config=None, data_manager=None, quality_checker=None):
        """初始化增强数据提供者
        
        Args:
            config: 配置字典，为None时使用默认配置
            data_manager: 数据管理器实例，为None时尝试导入并创建
            quality_checker: 数据质量检查器实例，为None时尝试导入并创建
        """
        # 配置日志
        self.logger = logging.getLogger(__name__)
        
        # 默认配置
        self.default_config = {
            'use_mock_data': False,  # 不使用模拟数据
            'data_quality_threshold': 60,  # 数据质量分数阈值
            'max_retry_count': 3,  # 最大重试次数
            'retry_delay': 2,  # 重试延迟秒数
            'stock_data_path': './data/stock_data',  # 本地数据路径
            'cache_enabled': True,  # 启用缓存
            'cache_ttl': 86400,  # 缓存生存时间（秒）
            'check_data_quality': True,  # 检查数据质量
        }
        
        # 使用传入配置覆盖默认配置
        self.config = self.default_config.copy()
        if config:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
        
        # 初始化数据管理器
        self.data_manager = data_manager
        if not self.data_manager:
            try:
                # 尝试导入数据管理器
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from stock_data_manager import StockDataManager
                self.data_manager = StockDataManager()
                self.logger.info("已初始化StockDataManager")
            except ImportError:
                self.logger.warning("无法导入StockDataManager，将使用本地文件作为数据源")
                self.data_manager = None
        
        # 初始化数据质量检查器
        self.quality_checker = quality_checker
        if not self.quality_checker and self.config['check_data_quality']:
            try:
                # 尝试导入数据质量检查器
                from data_quality_checker import DataQualityChecker
                self.quality_checker = DataQualityChecker()
                self.logger.info("已初始化DataQualityChecker")
            except ImportError:
                self.logger.warning("无法导入DataQualityChecker，将关闭数据质量检查")
                self.quality_checker = None
                self.config['check_data_quality'] = False
        
        self.logger.info(f"EnhancedDataProvider初始化完成，配置：{json.dumps(self.config, indent=2)}")
    
    def get_stock_list(self, include_delisted=False, update=False):
        """获取股票列表
        
        Args:
            include_delisted: 是否包含已退市股票
            update: 是否强制更新数据
            
        Returns:
            DataFrame: 股票代码列表DataFrame
            
        Raises:
            DataNotAvailableError: 当无法获取股票列表时抛出
        """
        self.logger.info(f"获取股票列表 include_delisted={include_delisted}, update={update}")
        
        stock_list_df = None
        error_msg = ""
        
        # 1. 尝试从数据管理器获取
        if self.data_manager:
            try:
                self.logger.debug("尝试从StockDataManager获取股票列表")
                stock_list_df = self.data_manager.get_stock_list(include_delisted=include_delisted, update=update)
                if stock_list_df is not None and not stock_list_df.empty:
                    self.logger.info(f"从StockDataManager获取了{len(stock_list_df)}只股票")
                    return stock_list_df
            except Exception as e:
                self.logger.error(f"从StockDataManager获取股票列表失败: {str(e)}")
                error_msg += f"StockDataManager: {str(e)}; "
        
        # 2. 尝试从本地文件读取
        local_file = os.path.join(self.config['stock_data_path'], 'stock_list.csv')
        if os.path.exists(local_file):
            try:
                self.logger.debug(f"尝试从本地文件{local_file}读取股票列表")
                stock_list_df = pd.read_csv(local_file)
                
                # 如果不包含已退市股票，过滤数据
                if not include_delisted and 'status' in stock_list_df.columns:
                    stock_list_df = stock_list_df[stock_list_df['status'] != 'delisted']
                
                if stock_list_df is not None and not stock_list_df.empty:
                    self.logger.info(f"从本地文件读取了{len(stock_list_df)}只股票")
                    return stock_list_df
            except Exception as e:
                self.logger.error(f"从本地文件读取股票列表失败: {str(e)}")
                error_msg += f"Local file: {str(e)}; "
        
        # 如果所有方法都失败，抛出异常
        error_message = f"无法获取股票列表: {error_msg}"
        self.logger.error(error_message)
        raise DataNotAvailableError(error_message)
    
    def get_stock_data(self, code, start_date=None, end_date=None, adjust='qfq', fields=None, update=False):
        """获取单个股票的历史数据
        
        Args:
            code: 股票代码
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            adjust: 价格复权类型，'qfq'前复权, 'hfq'后复权，None不复权
            fields: 需要的字段列表，None返回所有字段
            update: 是否强制更新数据
            
        Returns:
            DataFrame: 股票历史数据
            
        Raises:
            DataNotAvailableError: 当无法获取股票数据时抛出
            DataQualityError: 当数据质量不符合要求时抛出
        """
        if not code:
            raise ValueError("股票代码不能为空")
        
        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        
        self.logger.info(f"获取股票{code}从{start_date}到{end_date}的数据")
        
        stock_data = None
        error_msg = ""
        retry_count = 0
        
        # 最多重试指定次数
        while retry_count < self.config['max_retry_count'] and stock_data is None:
            if retry_count > 0:
                self.logger.info(f"第{retry_count}次重试获取股票{code}数据")
                time.sleep(self.config['retry_delay'])
            
            # 1. 首先尝试从数据管理器获取
            if self.data_manager:
                try:
                    self.logger.debug(f"尝试从StockDataManager获取股票{code}数据")
                    stock_data = self.data_manager.get_stock_data(
                        code, 
                        start_date=start_date, 
                        end_date=end_date, 
                        adjust=adjust,
                        update=update
                    )
                    
                    if stock_data is not None and not stock_data.empty:
                        self.logger.info(f"从StockDataManager获取了股票{code}的{len(stock_data)}条数据")
                        break
                except Exception as e:
                    self.logger.error(f"从StockDataManager获取股票{code}数据失败: {str(e)}")
                    error_msg += f"StockDataManager: {str(e)}; "
            
            # 2. 尝试从本地文件读取
            try:
                local_file = os.path.join(self.config['stock_data_path'], f"{code}.csv")
                if os.path.exists(local_file):
                    self.logger.debug(f"尝试从本地文件{local_file}读取股票{code}数据")
                    stock_data = pd.read_csv(local_file)
                    
                    # 如果有日期列，转换为日期类型并过滤日期范围
                    if 'trade_date' in stock_data.columns:
                        stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
                        stock_data = stock_data[
                            (stock_data['trade_date'] >= pd.to_datetime(start_date)) &
                            (stock_data['trade_date'] <= pd.to_datetime(end_date))
                        ]
                    
                    if stock_data is not None and not stock_data.empty:
                        self.logger.info(f"从本地文件读取了股票{code}的{len(stock_data)}条数据")
                        break
            except Exception as e:
                self.logger.error(f"从本地文件读取股票{code}数据失败: {str(e)}")
                error_msg += f"Local file: {str(e)}; "
            
            retry_count += 1
        
        # 检查是否获取到数据
        if stock_data is None or stock_data.empty:
            error_message = f"无法获取股票{code}数据: {error_msg}"
            self.logger.error(error_message)
            raise DataNotAvailableError(error_message)
        
        # 检查数据质量
        if self.config['check_data_quality'] and self.quality_checker:
            try:
                quality_result = self.quality_checker.check_data_quality(stock_data, code)
                if not quality_result['pass'] and quality_result['score'] < self.config['data_quality_threshold']:
                    error_message = f"股票{code}数据质量不符合要求: 得分{quality_result['score']}, 原因: {quality_result['reason']}"
                    self.logger.warning(error_message)
                    raise DataQualityError(error_message)
                else:
                    self.logger.info(f"股票{code}数据质量检查通过，得分: {quality_result['score']}")
            except Exception as e:
                self.logger.error(f"检查股票{code}数据质量时出错: {str(e)}")
        
        # 过滤需要的字段
        if fields and isinstance(fields, list):
            available_fields = [f for f in fields if f in stock_data.columns]
            if available_fields:
                stock_data = stock_data[available_fields]
        
        return stock_data
    
    def get_index_data(self, code, start_date=None, end_date=None, update=False):
        """获取指数数据
        
        Args:
            code: 指数代码
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            update: 是否强制更新数据
            
        Returns:
            DataFrame: 指数历史数据
            
        Raises:
            DataNotAvailableError: 当无法获取指数数据时抛出
        """
        if not code:
            raise ValueError("指数代码不能为空")
        
        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        
        self.logger.info(f"获取指数{code}从{start_date}到{end_date}的数据")
        
        index_data = None
        error_msg = ""
        
        # 1. 尝试从数据管理器获取
        if self.data_manager:
            try:
                self.logger.debug(f"尝试从StockDataManager获取指数{code}数据")
                index_data = self.data_manager.get_index_data(
                    code, 
                    start_date=start_date, 
                    end_date=end_date,
                    update=update
                )
                
                if index_data is not None and not index_data.empty:
                    self.logger.info(f"从StockDataManager获取了指数{code}的{len(index_data)}条数据")
                    return index_data
            except Exception as e:
                self.logger.error(f"从StockDataManager获取指数{code}数据失败: {str(e)}")
                error_msg += f"StockDataManager: {str(e)}; "
        
        # 2. 尝试从本地文件读取
        try:
            local_file = os.path.join(self.config['stock_data_path'], f"index_{code}.csv")
            if os.path.exists(local_file):
                self.logger.debug(f"尝试从本地文件{local_file}读取指数{code}数据")
                index_data = pd.read_csv(local_file)
                
                # 如果有日期列，转换为日期类型并过滤日期范围
                if 'trade_date' in index_data.columns:
                    index_data['trade_date'] = pd.to_datetime(index_data['trade_date'])
                    index_data = index_data[
                        (index_data['trade_date'] >= pd.to_datetime(start_date)) &
                        (index_data['trade_date'] <= pd.to_datetime(end_date))
                    ]
                
                if index_data is not None and not index_data.empty:
                    self.logger.info(f"从本地文件读取了指数{code}的{len(index_data)}条数据")
                    return index_data
        except Exception as e:
            self.logger.error(f"从本地文件读取指数{code}数据失败: {str(e)}")
            error_msg += f"Local file: {str(e)}; "
        
        # 检查是否获取到数据
        if index_data is None or index_data.empty:
            error_message = f"无法获取指数{code}数据: {error_msg}"
            self.logger.error(error_message)
            raise DataNotAvailableError(error_message)
        
        return index_data
    
    def get_industry_stocks(self, industry_code, date=None):
        """获取行业成分股
        
        Args:
            industry_code: 行业代码
            date: 查询日期，格式YYYY-MM-DD，默认为当前日期
            
        Returns:
            DataFrame: 行业成分股列表
            
        Raises:
            DataNotAvailableError: 当无法获取行业成分股时抛出
        """
        if not industry_code:
            raise ValueError("行业代码不能为空")
        
        # 设置默认日期
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        self.logger.info(f"获取行业{industry_code}在{date}的成分股")
        
        industry_stocks = None
        error_msg = ""
        
        # 1. 尝试从数据管理器获取
        if self.data_manager:
            try:
                self.logger.debug(f"尝试从StockDataManager获取行业{industry_code}成分股")
                industry_stocks = self.data_manager.get_industry_stocks(industry_code, date=date)
                
                if industry_stocks is not None and not industry_stocks.empty:
                    self.logger.info(f"从StockDataManager获取了行业{industry_code}的{len(industry_stocks)}只成分股")
                    return industry_stocks
            except Exception as e:
                self.logger.error(f"从StockDataManager获取行业{industry_code}成分股失败: {str(e)}")
                error_msg += f"StockDataManager: {str(e)}; "
        
        # 2. 尝试从本地文件读取
        try:
            local_file = os.path.join(self.config['stock_data_path'], f"industry_{industry_code}.csv")
            if os.path.exists(local_file):
                self.logger.debug(f"尝试从本地文件{local_file}读取行业{industry_code}成分股")
                industry_stocks = pd.read_csv(local_file)
                
                if industry_stocks is not None and not industry_stocks.empty:
                    self.logger.info(f"从本地文件读取了行业{industry_code}的{len(industry_stocks)}只成分股")
                    return industry_stocks
        except Exception as e:
            self.logger.error(f"从本地文件读取行业{industry_code}成分股失败: {str(e)}")
            error_msg += f"Local file: {str(e)}; "
        
        # 检查是否获取到数据
        if industry_stocks is None or industry_stocks.empty:
            error_message = f"无法获取行业{industry_code}成分股: {error_msg}"
            self.logger.error(error_message)
            raise DataNotAvailableError(error_message)
        
        return industry_stocks
    
    def get_financial_data(self, code, report_type='income', period='annual', update=False):
        """获取财务数据
        
        Args:
            code: 股票代码
            report_type: 报表类型，income(利润表)、balance(资产负债表)、cash_flow(现金流量表)
            period: 报告期，annual(年报)、quarterly(季报)
            update: 是否强制更新数据
            
        Returns:
            DataFrame: 财务数据
            
        Raises:
            DataNotAvailableError: 当无法获取财务数据时抛出
        """
        if not code:
            raise ValueError("股票代码不能为空")
        
        self.logger.info(f"获取股票{code}的{report_type}{period}财务数据")
        
        financial_data = None
        error_msg = ""
        
        # 1. 尝试从数据管理器获取
        if self.data_manager:
            try:
                self.logger.debug(f"尝试从StockDataManager获取股票{code}财务数据")
                financial_data = self.data_manager.get_financial_data(
                    code, 
                    report_type=report_type,
                    period=period,
                    update=update
                )
                
                if financial_data is not None and not financial_data.empty:
                    self.logger.info(f"从StockDataManager获取了股票{code}的{len(financial_data)}条财务数据")
                    return financial_data
            except Exception as e:
                self.logger.error(f"从StockDataManager获取股票{code}财务数据失败: {str(e)}")
                error_msg += f"StockDataManager: {str(e)}; "
        
        # 2. 尝试从本地文件读取
        try:
            local_file = os.path.join(self.config['stock_data_path'], f"financial_{code}_{report_type}_{period}.csv")
            if os.path.exists(local_file):
                self.logger.debug(f"尝试从本地文件{local_file}读取股票{code}财务数据")
                financial_data = pd.read_csv(local_file)
                
                if financial_data is not None and not financial_data.empty:
                    self.logger.info(f"从本地文件读取了股票{code}的{len(financial_data)}条财务数据")
                    return financial_data
        except Exception as e:
            self.logger.error(f"从本地文件读取股票{code}财务数据失败: {str(e)}")
            error_msg += f"Local file: {str(e)}; "
        
        # 检查是否获取到数据
        if financial_data is None or financial_data.empty:
            error_message = f"无法获取股票{code}财务数据: {error_msg}"
            self.logger.error(error_message)
            raise DataNotAvailableError(error_message)
        
        return financial_data

# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建增强数据提供者实例
    provider = EnhancedDataProvider()
    
    # 测试获取股票列表
    try:
        logging.info("测试获取股票列表")
        stock_list = provider.get_stock_list()
        logging.info(f"获取到{len(stock_list)}只股票")
        logging.info(f"前5只股票: \n{stock_list.head()}")
    except Exception as e:
        logging.error(f"获取股票列表失败: {str(e)}")
    
    # 测试获取股票数据
    try:
        logging.info("\n测试获取股票数据")
        # 以上证50ETF为例
        stock_data = provider.get_stock_data('sh.000001', start_date='2023-01-01', end_date='2023-12-31')
        logging.info(f"获取到{len(stock_data)}条数据")
        logging.info(f"数据示例: \n{stock_data.head()}")
    except Exception as e:
        logging.error(f"获取股票数据失败: {str(e)}")
    
    # 测试获取指数数据
    try:
        logging.info("\n测试获取指数数据")
        index_data = provider.get_index_data('000300', start_date='2023-01-01', end_date='2023-12-31')
        logging.info(f"获取到{len(index_data)}条数据")
        logging.info(f"数据示例: \n{index_data.head()}")
    except Exception as e:
        logging.error(f"获取指数数据失败: {str(e)}")
    
    # 测试数据质量检查
    try:
        logging.info("\n测试数据质量检查")
        from data_quality_checker import DataQualityChecker
        checker = DataQualityChecker()
        
        if 'stock_data' in locals() and stock_data is not None and not stock_data.empty:
            result = checker.check_data_quality(stock_data, 'sh.000001')
            logging.info(f"数据质量检查结果: {'通过' if result['pass'] else '不通过'}")
            logging.info(f"质量得分: {result['score']:.2f}")
            logging.info(f"原因: {result['reason']}")
            
            # 获取质量报告
            report = checker.get_quality_report(stock_data, 'sh.000001')
            logging.info(f"数据点数量: {report['data_stats']['row_count']}")
            if 'date_stats' in report:
                logging.info(f"数据开始日期: {report['date_stats']['start_date']}")
                logging.info(f"数据结束日期: {report['date_stats']['end_date']}")
    except Exception as e:
        logging.error(f"测试数据质量检查失败: {str(e)}") 