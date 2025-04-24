#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版数据处理系统启动脚本
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from src.enhanced.config.settings import LOG_DIR
from src.enhanced.data.data_manager import EnhancedDataManager
from src.enhanced.data.processors.optimized_processor import OptimizedDataProcessor
from src.enhanced.data.quality.data_quality_checker import DataQualityChecker
from src.enhanced.data.cache.cache_manager import CacheManager

# 配置日志
log_file = os.path.join(LOG_DIR, f"enhanced_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
        logger.info("正在启动增强版数据处理系统...")
        
        # 初始化缓存管理器
        logger.info("初始化缓存管理器...")
        cache_manager = CacheManager()
        
        # 初始化数据管理器
        logger.info("初始化数据管理器...")
        data_manager = EnhancedDataManager()
        
        # 初始化数据处理器
        logger.info("初始化数据处理器...")
        processor = OptimizedDataProcessor()
        
        # 初始化数据质量检查器
        logger.info("初始化数据质量检查器...")
        quality_checker = DataQualityChecker()
        
        logger.info("系统初始化完成，开始运行...")
        
        # 获取股票列表
        stock_list = data_manager.get_stock_list()
        if stock_list is not None:
            logger.info(f"成功获取股票列表，共 {len(stock_list)} 只股票")
            logger.info("\n示例数据:\n" + str(stock_list.head()))
        else:
            logger.warning("获取股票列表失败")
        
        # 获取示例股票数据
        stock_code = "000001.SZ"  # 平安银行
        start_date = "2024-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        stock_data = data_manager.get_stock_data(stock_code, start_date, end_date)
        if stock_data is not None:
            logger.info(f"成功获取股票 {stock_code} 的数据，共 {len(stock_data)} 条记录")
            logger.info("\n示例数据:\n" + str(stock_data.head()))
        else:
            logger.warning(f"获取股票 {stock_code} 的数据失败")
        
        logger.info("系统运行完成")
        
    except Exception as e:
        logger.error(f"系统运行出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 