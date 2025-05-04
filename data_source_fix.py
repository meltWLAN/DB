#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据源修复脚本
解决市场概览刷新慢的问题并修复数据获取接口
"""

import logging
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_source_fix.log')
    ]
)
logger = logging.getLogger()

def fix_data_source_manager():
    """修复DataSourceManager相关问题"""
    logger.info("开始执行数据源修复...")
    
    try:
        # 导入DataSourceManager
        from src.enhanced.data.fetchers.data_source_manager import DataSourceManager
        logger.info("成功导入DataSourceManager")
        
        # 检查DataSourceManager实例是否能正确初始化
        logger.info("测试DataSourceManager初始化...")
        manager = DataSourceManager()
        logger.info(f"成功初始化DataSourceManager，主数据源：{manager.primary_source}")
        logger.info(f"可用数据源：{manager.available_sources}")
        
        # 测试获取指数数据
        logger.info("\n测试获取指数数据...")
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 测试多个指数
        indices = [
            ("上证指数", "000001.SH"),
            ("深证成指", "399001.SZ"),
            ("中证500", "000905.SH"),
        ]
        
        for name, code in indices:
            try:
                logger.info(f"获取{name}({code})数据...")
                index_data = manager.get_stock_index_data(code, start_date, end_date)
                
                if index_data is not None and not index_data.empty:
                    logger.info(f"成功获取{name}数据，共{len(index_data)}条记录")
                else:
                    logger.warning(f"未获取到{name}数据")
            except Exception as e:
                logger.error(f"获取{name}数据出错：{str(e)}")
        
        # 检查get_stock_data方法是否可用
        logger.info("\n检查get_stock_data方法...")
        try:
            if hasattr(manager, 'get_stock_data'):
                logger.info("找到get_stock_data方法，进行测试...")
                stock_data = manager.get_stock_data('000001.SZ', start_date, end_date)
                
                if stock_data is not None and not stock_data.empty:
                    logger.info(f"成功获取股票数据，共{len(stock_data)}条记录")
                else:
                    logger.warning("未获取到股票数据")
            else:
                logger.error("DataSourceManager对象没有get_stock_data方法")
                
                # 获取DataSourceManager类属性
                methods = [name for name in dir(manager) if callable(getattr(manager, name)) and not name.startswith('_')]
                logger.info(f"可用方法：{methods}")
                
                # 查看get_daily_data是否可用
                if hasattr(manager, 'get_daily_data'):
                    logger.info("找到get_daily_data方法，进行测试...")
                    stock_data = manager.get_daily_data('000001.SZ', start_date, end_date)
                    
                    if stock_data is not None and not stock_data.empty:
                        logger.info(f"成功通过get_daily_data获取股票数据，共{len(stock_data)}条记录")
                    else:
                        logger.warning("未通过get_daily_data获取到股票数据")
                else:
                    logger.error("DataSourceManager对象也没有get_daily_data方法")
        except Exception as e:
            logger.error(f"测试get_stock_data方法出错：{str(e)}")
        
        # 测试获取市场概览功能
        logger.info("\n测试获取市场概览...")
        try:
            # 使用最新交易日
            latest_date = manager.get_latest_trading_date()
            logger.info(f"最新交易日：{latest_date}")
            
            market_overview = manager.get_market_overview(latest_date)
            
            if market_overview and isinstance(market_overview, dict) and len(market_overview) > 0:
                logger.info(f"成功获取市场概览数据，包含{len(market_overview)}个键")
                logger.info(f"市场概览关键数据：")
                if 'up_count' in market_overview and 'down_count' in market_overview:
                    logger.info(f"涨跌家数：{market_overview.get('up_count', 0)}涨 / {market_overview.get('down_count', 0)}跌")
                if 'limit_up_count' in market_overview and 'limit_down_count' in market_overview:
                    logger.info(f"涨停/跌停：{market_overview.get('limit_up_count', 0)}涨停 / {market_overview.get('limit_down_count', 0)}跌停")
            else:
                logger.warning("未获取到市场概览数据")
        except Exception as e:
            logger.error(f"测试市场概览出错：{str(e)}")
            
        # 测试get_market_overview的性能
        logger.info("\n测试市场概览性能...")
        try:
            import time
            start_time = time.time()
            market_overview = manager.get_market_overview()
            elapsed = time.time() - start_time
            
            logger.info(f"获取市场概览耗时：{elapsed:.2f}秒")
            
            # 分析耗时较长的原因
            if elapsed > 10:
                logger.warning("市场概览获取耗时较长，建议检查以下几点：")
                logger.warning("1. 缓存机制是否正常工作")
                logger.warning("2. API调用是否有不必要的重试")
                logger.warning("3. 数据处理逻辑是否高效")
        except Exception as e:
            logger.error(f"测试市场概览性能出错：{str(e)}")
            
        logger.info("\n数据源修复检查完成")
        
    except Exception as e:
        logger.error(f"数据源修复脚本执行过程中发生错误：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_data_source_manager() 