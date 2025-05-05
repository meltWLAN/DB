#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版股票数据系统启动脚本
使用单一数据源直接获取数据
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# 确保src包可以被导入
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 导入配置
from src.enhanced.config.settings import LOG_DIR, DATA_DIR, DATA_SOURCE_CONFIG

# 创建必要目录
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 配置日志
log_file = os.path.join(LOG_DIR, f"simple_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """主函数"""
    try:
        logger.info("==== 简化版股票数据系统启动 ====")
        
        # 导入数据源 - 这里直接使用TuShare作为主要数据源
        from src.enhanced.data.fetchers.tushare_fetcher import EnhancedTushareFetcher
        
        # 获取TuShare配置
        tushare_config = DATA_SOURCE_CONFIG.get('tushare', {})
        if not tushare_config:
            logger.error("找不到TuShare的配置信息")
            return None
            
        if not tushare_config.get('enabled', False):
            logger.error("TuShare数据源未启用")
            return None
            
        # 初始化TuShare数据源
        logger.info("初始化TuShare数据源...")
        
        # 处理rate_limit配置
        if 'rate_limit' in tushare_config:
            rate_limit_config = tushare_config.get('rate_limit', {})
            rate_limit = rate_limit_config.get('calls_per_minute', 500) / 60  # 转换为每秒请求数
        else:
            rate_limit = 500 / 60
            
        # 处理retry配置
        if 'retry' in tushare_config:
            retry_config = tushare_config.get('retry', {})
            max_retries = retry_config.get('max_retries', 3)
            retry_delay = retry_config.get('retry_interval', 5)
        else:
            max_retries = 3
            retry_delay = 5
        
        # 创建配置字典
        fetcher_config = {
            'token': tushare_config.get('token', ''),
            'rate_limit': rate_limit,
            'connection_retries': max_retries,
            'retry_delay': retry_delay
        }
        
        # 初始化数据获取器
        tushare_fetcher = EnhancedTushareFetcher(fetcher_config)
        
        # 测试获取股票列表
        logger.info("测试获取股票列表...")
        stock_list = tushare_fetcher.get_stock_list()
        if stock_list is not None and not stock_list.empty:
            logger.info(f"成功获取股票列表，共 {len(stock_list)} 只股票")
            logger.info("\n示例数据:\n" + str(stock_list.head()))
        else:
            logger.warning("获取股票列表失败")
            
        # 测试获取行业列表
        logger.info("测试获取行业列表...")
        industry_list = tushare_fetcher.get_industry_list()
        if industry_list is not None and not industry_list.empty:
            logger.info(f"成功获取行业列表，共 {len(industry_list)} 个行业")
            logger.info("\n示例数据:\n" + str(industry_list.head()))
        else:
            logger.warning("获取行业列表失败")
            
        # 测试获取单只股票的日线数据
        stock_code = "000001.SZ"  # 平安银行
        start_date = "2024-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"测试获取股票 {stock_code} 的日线数据...")
        daily_data = tushare_fetcher.get_daily_data(stock_code, start_date, end_date)
        if daily_data is not None and not daily_data.empty:
            logger.info(f"成功获取股票 {stock_code} 的日线数据，共 {len(daily_data)} 条记录")
            logger.info("\n示例数据:\n" + str(daily_data.head()))
        else:
            logger.warning(f"获取股票 {stock_code} 的日线数据失败")
            
        logger.info("==== 简化版股票数据系统启动完成 ====")
        
        # 返回数据源实例以便可以在交互式环境中使用
        return tushare_fetcher
        
    except Exception as e:
        logger.error(f"系统启动出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    data_source = main() 