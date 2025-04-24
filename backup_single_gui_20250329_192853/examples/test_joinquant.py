#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JoinQuant API连接测试脚本
用于测试JoinQuant账号是否正确配置，API是否能正常使用
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd

# 添加当前目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from src.config import DATA_SOURCE_CONFIG
from src.data.joinquant_data_source import JoinQuantDataSource

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_joinquant_connection():
    """
    测试JoinQuant API连接
    """
    logger.info("=== 测试JoinQuant API连接 ===")
    
    # 获取JoinQuant配置
    joinquant_config = DATA_SOURCE_CONFIG.get('joinquant', {})
    username = joinquant_config.get('username', '')
    password = joinquant_config.get('password', '')
    
    if not username or not password:
        logger.error("JoinQuant用户名或密码未配置，请在src/config/__init__.py中设置")
        return False
    
    logger.info(f"使用配置的JoinQuant账号: {username}")
    
    # 初始化JoinQuant数据源
    jq_source = JoinQuantDataSource(
        username=username,
        password=password,
        cache_dir="./data/cache/joinquant_test"
    )
    
    # 测试是否可用
    if jq_source.is_available():
        logger.info("JoinQuant API连接成功，账号验证通过！")
        
        # 测试获取股票列表
        logger.info("测试获取股票列表...")
        start_time = datetime.now()
        stocks = jq_source.get_stock_list()
        end_time = datetime.now()
        if not stocks.empty:
            logger.info(f"成功获取到 {len(stocks)} 只股票信息，耗时 {(end_time - start_time).total_seconds():.2f} 秒")
            logger.info(f"股票列表前5只: {stocks['name'].head(5).tolist()}")
        else:
            logger.error("获取股票列表失败")
            return False
        
        # 测试获取单只股票数据
        sample_stock = "000001.SZ"  # 平安银行
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        logger.info(f"测试获取单只股票 {sample_stock} 最近30天数据...")
        start_time = datetime.now()
        stock_data = jq_source.get_daily_data(sample_stock, start_date, end_date)
        end_time = datetime.now()
        
        if not stock_data.empty:
            logger.info(f"成功获取 {sample_stock} 数据，包含 {len(stock_data)} 条记录，耗时 {(end_time - start_time).total_seconds():.2f} 秒")
            logger.info(f"最新收盘价: {stock_data['close'].iloc[-1]:.2f}")
            
            # 测试计算技术指标
            logger.info("测试计算技术指标...")
            with_indicators = jq_source.calculate_indicators(stock_data)
            indicator_cols = [col for col in with_indicators.columns if col not in stock_data.columns]
            logger.info(f"成功计算 {len(indicator_cols)} 个技术指标: {', '.join(indicator_cols[:5])}")
            
            # 尝试获取基本面数据
            logger.info("测试获取基本面数据...")
            fundamental_data = jq_source.get_fundamentals(sample_stock)
            if not fundamental_data.empty:
                logger.info(f"成功获取基本面数据，包含列: {', '.join(fundamental_data.columns[:5])}")
            else:
                logger.warning("基本面数据获取失败")
            
            # 查询剩余额度
            try:
                import jqdatasdk as jq
                if jq.is_auth():
                    quote = jq.get_query_count()
                    logger.info(f"当前剩余数据调用额度: 数据点 {quote['spare_count']}, 财务数据点 {quote['spare_financial_count']}")
            except:
                logger.warning("无法查询剩余额度")
            
            return True
        else:
            logger.error(f"获取 {sample_stock} 数据失败")
            return False
    else:
        logger.error("JoinQuant API连接失败，请检查账号和密码是否正确")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print(" JoinQuant API 测试程序 ".center(80, "="))
    print("=" * 80)
    
    # 运行测试
    success = test_joinquant_connection()
    
    print("=" * 80)
    print(f" 测试结果: {'成功' if success else '失败'} ".center(80, "="))
    print("=" * 80) 