#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理和管理流程测试脚本
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.enhanced.data.data_manager import EnhancedDataManager
from src.enhanced.data.processors.optimized_processor import OptimizedDataProcessor
from src.enhanced.data.cache.cache_manager import CacheManager
from src.enhanced.data.quality.data_quality_checker import DataQualityChecker
from src.enhanced.config.settings import (
    ROOT_DIR, DATA_DIR, RESULTS_DIR, LOG_DIR
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'workflow_test.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# @with_cache(method_name="test_data_fetching")  # 注释掉这一行，因为它引用了未定义的装饰器
def test_data_fetching():
    """测试数据获取功能"""
    logger.info("=" * 50)
    logger.info("测试数据获取功能")
    
    data_manager = EnhancedDataManager()
    
    # 测试获取股票列表
    logger.info("测试获取股票列表")
    stock_list = data_manager.get_stock_list()
    if stock_list is not None and not stock_list.empty:
        logger.info(f"成功获取股票列表，共 {len(stock_list)} 只股票")
        logger.info(f"股票列表示例:\n{stock_list.head()}")
    else:
        logger.warning("获取股票列表失败")
    
    # 测试获取单只股票数据
    test_stock = '000001.SZ'
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"测试获取股票 {test_stock} 的数据")
    stock_data = data_manager.get_stock_data(test_stock, start_date, end_date)
    
    if stock_data is not None and not stock_data.empty:
        logger.info(f"成功获取股票数据，共 {len(stock_data)} 条记录")
        logger.info(f"数据示例:\n{stock_data.head()}")
        return stock_data
    else:
        logger.warning("获取股票数据失败，创建模拟数据")
        return create_mock_data()

def test_data_processing(data):
    """测试数据处理功能"""
    logger.info("=" * 50)
    logger.info("测试数据处理功能")
    
    processor = OptimizedDataProcessor()
    
    # 测试数据清洗
    logger.info("测试数据清洗")
    cleaned_data = processor.clean_daily_data(data)
    logger.info(f"数据清洗完成，处理后数据形状: {cleaned_data.shape}")
    
    # 测试技术指标计算
    logger.info("测试技术指标计算")
    data_with_indicators = processor.calculate_technical_indicators(cleaned_data)
    logger.info(f"技术指标计算完成，新增指标: {[col for col in data_with_indicators.columns if col not in cleaned_data.columns]}")
    
    # 测试完整处理流程
    logger.info("测试完整处理流程")
    processed_data = processor.process_data_pipeline(data)
    logger.info(f"完整处理流程完成，最终数据形状: {processed_data.shape}")
    
    return processed_data

def test_data_quality(data):
    """测试数据质量检查功能"""
    logger.info("=" * 50)
    logger.info("测试数据质量检查功能")
    
    quality_checker = DataQualityChecker()
    
    # 测试数据质量验证
    logger.info("测试数据质量验证")
    quality_report = quality_checker.validate_data(data)
    
    if quality_report:
        logger.info(f"数据质量分数: {quality_report.get('quality_score', 0):.2f}")
        if 'missing_values' in quality_report:
            logger.info(f"缺失值检查结果: {quality_report['missing_values']}")
        if 'validation_errors' in quality_report:
            logger.info(f"验证错误: {quality_report['validation_errors']}")
    
    # 测试数据修复
    logger.info("测试数据修复")
    fixed_data, fix_report = quality_checker.fix_data_issues(data)
    
    if fix_report:
        logger.info(f"修复的问题数量: {len(fix_report.get('fixed_issues', []))}")
        logger.info(f"修改的行数: {fix_report.get('rows_modified', 0)}")
    
    return fixed_data

def test_caching():
    """测试缓存管理功能"""
    logger.info("=" * 50)
    logger.info("测试缓存管理功能")
    
    cache_dir = os.path.join(DATA_DIR, 'test_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_manager = CacheManager(
        cache_dir=cache_dir,
        memory_cache_size=64,  # 64MB
        max_age_hours=1
    )
    
    # 测试缓存DataFrame
    test_df = pd.DataFrame({
        'A': np.random.randn(1000),
        'B': np.random.randn(1000),
        'C': pd.date_range('2023-01-01', periods=1000)
    })
    
    logger.info("测试缓存DataFrame")
    cache_manager.set('test_dataframe', test_df, format_type='parquet')
    
    # 测试读取缓存
    logger.info("测试读取缓存")
    cached_df = cache_manager.get('test_dataframe')
    if cached_df is not None:
        logger.info(f"成功读取缓存的DataFrame，形状: {cached_df.shape}")
    
    # 测试缓存键列表
    logger.info("测试缓存键列表")
    cache_keys = cache_manager.list_keys()
    logger.info(f"当前缓存键: {cache_keys}")
    
    # 测试缓存清理
    logger.info("测试缓存清理")
    cache_manager.clear_expired()
    logger.info("缓存清理完成")

def create_mock_data():
    """创建模拟数据用于测试"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(10, 100, 100),
        'high': np.random.uniform(10, 100, 100),
        'low': np.random.uniform(10, 100, 100),
        'close': np.random.uniform(10, 100, 100),
        'volume': np.random.uniform(1000000, 10000000, 100),
        'amount': np.random.uniform(10000000, 100000000, 100)
    })
    
    # 确保high >= open >= low和high >= close >= low
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data

def main():
    """主测试流程"""
    logger.info("开始测试数据处理和管理流程")
    
    try:
        # 1. 测试数据获取
        stock_data = test_data_fetching()
        
        # 2. 测试数据处理
        if stock_data is not None:
            processed_data = test_data_processing(stock_data)
            
            # 3. 测试数据质量
            if processed_data is not None:
                quality_checked_data = test_data_quality(processed_data)
                
                # 4. 测试缓存管理
                test_caching()
            
        logger.info("数据处理和管理流程测试完成")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 