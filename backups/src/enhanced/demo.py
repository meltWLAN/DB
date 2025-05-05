#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版数据处理系统演示
展示系统各组件的功能和性能
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import argparse
import json

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入增强版模块
from src.enhanced.data.data_manager import EnhancedDataManager
from src.enhanced.data.processors.optimized_processor import OptimizedDataProcessor
from src.enhanced.data.cache.cache_manager import CacheManager
from src.enhanced.data.quality.data_quality_checker import DataQualityChecker
from src.enhanced.config.settings import (
    ROOT_DIR, DATA_DIR, RESULTS_DIR, LOG_DIR, 
    PROCESSING_CONFIG, DATA_QUALITY_CONFIG
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'demo.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def demo_data_fetching():
    """演示数据获取功能"""
    logger.info("=" * 50)
    logger.info("演示数据获取功能")
    
    # 创建数据管理器
    data_manager = EnhancedDataManager()
    
    # 股票代码列表
    stock_codes = ['000001.SZ', '600000.SH', '000858.SZ', '601318.SH']
    
    # 设置时间范围
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    # 获取单只股票数据
    logger.info(f"获取股票 {stock_codes[0]} 的数据")
    start_time = time.time()
    stock_data = data_manager.get_stock_data(stock_codes[0], start_date, end_date)
    end_time = time.time()
    
    if stock_data is not None and not stock_data.empty:
        logger.info(f"成功获取 {stock_codes[0]} 的数据，共 {len(stock_data)} 条记录, 耗时: {end_time - start_time:.2f}秒")
        logger.info(f"数据列: {list(stock_data.columns)}")
        
        # 展示部分数据
        if len(stock_data) > 0:
            logger.info(f"数据样例:\n{stock_data.head(3)}")
    else:
        logger.warning(f"未能获取 {stock_codes[0]} 的数据")
    
    # 并行获取多只股票数据
    logger.info(f"并行获取多只股票数据: {stock_codes}")
    start_time = time.time()
    stocks_data = data_manager.process_stocks_in_parallel(stock_codes, start_date, end_date)
    end_time = time.time()
    
    success_count = sum(1 for data in stocks_data.values() if data is not None and not data.empty)
    logger.info(f"成功获取 {success_count}/{len(stock_codes)} 只股票的数据, 耗时: {end_time - start_time:.2f}秒")
    
    # 返回获取的数据用于后续演示
    if stock_data is not None and not stock_data.empty:
        return stock_data
    elif success_count > 0:
        for data in stocks_data.values():
            if data is not None and not data.empty:
                return data
    
    # 如果没有获取到数据，创建模拟数据
    logger.warning("未能获取真实数据，创建模拟数据用于演示")
    return create_mock_data()

def demo_data_processing(data):
    """演示数据处理功能"""
    logger.info("=" * 50)
    logger.info("演示数据处理功能")
    
    # 创建处理器
    processor = OptimizedDataProcessor()
    
    # 数据清洗
    logger.info("执行数据清洗")
    start_time = time.time()
    cleaned_data = processor.clean_daily_data(data)
    end_time = time.time()
    logger.info(f"数据清洗完成，耗时: {end_time - start_time:.2f}秒")
    
    # 计算收益率
    logger.info("计算收益率指标")
    start_time = time.time()
    data_with_returns = processor.calculate_returns(cleaned_data)
    end_time = time.time()
    logger.info(f"收益率计算完成，耗时: {end_time - start_time:.2f}秒")
    
    # 计算技术指标
    logger.info("计算技术指标")
    start_time = time.time()
    data_with_indicators = processor.calculate_technical_indicators(data_with_returns)
    end_time = time.time()
    logger.info(f"技术指标计算完成，耗时: {end_time - start_time:.2f}秒")
    
    # 检测K线形态
    logger.info("检测K线形态")
    start_time = time.time()
    data_with_patterns = processor.detect_patterns(data_with_indicators)
    end_time = time.time()
    logger.info(f"K线形态检测完成，耗时: {end_time - start_time:.2f}秒")
    
    # 使用处理流水线
    logger.info("使用处理流水线执行全部步骤")
    start_time = time.time()
    processed_data = processor.process_data_pipeline(data)
    end_time = time.time()
    logger.info(f"流水线处理完成，耗时: {end_time - start_time:.2f}秒")
    
    # 展示部分结果
    technical_columns = [col for col in processed_data.columns if col.startswith('ma_') or col.startswith('rsi_')]
    if technical_columns:
        logger.info(f"计算出的技术指标列: {technical_columns[:10]}...")
    
    pattern_columns = [col for col in processed_data.columns if col.startswith('is_')]
    if pattern_columns:
        logger.info(f"检测出的形态列: {pattern_columns}")
    
    return processed_data

def demo_data_quality(data):
    """演示数据质量检查功能"""
    logger.info("=" * 50)
    logger.info("演示数据质量检查功能")
    
    # 创建质量检查器
    quality_checker = DataQualityChecker()
    
    # 添加一些问题数据用于演示
    demo_data = introduce_quality_issues(data.copy())
    
    # 验证数据质量
    logger.info("验证数据质量")
    start_time = time.time()
    quality_report = quality_checker.validate_data(demo_data)
    end_time = time.time()
    logger.info(f"数据质量验证完成，耗时: {end_time - start_time:.2f}秒")
    
    # 打印验证结果概要
    logger.info(f"数据质量分数: {quality_report.get('quality_score', 0):.2f}")
    if 'missing_values' in quality_report and quality_report['missing_values']:
        logger.info(f"发现缺失值: {len(quality_report['missing_values'])} 个列")
    
    if 'validation_errors' in quality_report:
        error_count = sum(len(errors) for errors in quality_report['validation_errors'].values())
        logger.info(f"发现验证错误: {error_count} 个")
    
    # 修复数据问题
    logger.info("修复数据问题")
    start_time = time.time()
    fixed_data, fix_report = quality_checker.fix_data_issues(demo_data)
    end_time = time.time()
    logger.info(f"数据问题修复完成，耗时: {end_time - start_time:.2f}秒")
    
    # 打印修复结果概要
    if 'fixed_issues' in fix_report:
        logger.info(f"修复的问题: {len(fix_report['fixed_issues'])} 类型")
        for issue_type, issue_info in fix_report['fixed_issues'].items():
            logger.info(f"  - {issue_type}: {issue_info['count']} 个, 操作: {issue_info['action']}")
    
    logger.info(f"修改的行数: {fix_report.get('rows_modified', 0)}")
    logger.info(f"删除的行数: {fix_report.get('rows_dropped', 0)}")
    
    # 完整的质量处理流程
    logger.info("执行完整的质量处理流程")
    start_time = time.time()
    processed_data, full_report = quality_checker.process_data_quality(demo_data)
    end_time = time.time()
    logger.info(f"完整质量处理完成，耗时: {end_time - start_time:.2f}秒")
    
    # 导出质量报告
    report_path = os.path.join(RESULTS_DIR, 'quality_report.json')
    quality_checker.export_report(full_report, report_path)
    logger.info(f"质量报告已保存到: {report_path}")
    
    return processed_data

def demo_caching():
    """演示缓存管理功能"""
    logger.info("=" * 50)
    logger.info("演示缓存管理功能")
    
    # 创建缓存管理器
    cache_dir = os.path.join(DATA_DIR, 'demo_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_manager = CacheManager(
        cache_dir=cache_dir,
        memory_cache_size=64,  # 64MB
        max_age_hours=1
    )
    
    # 缓存不同类型的数据
    # 1. DataFrame
    logger.info("缓存DataFrame")
    df = pd.DataFrame({
        'A': np.random.randn(10000),
        'B': np.random.randn(10000),
        'C': pd.date_range('2023-01-01', periods=10000)
    })
    
    start_time = time.time()
    cache_manager.set('demo_dataframe', df, format_type='parquet')
    end_time = time.time()
    logger.info(f"DataFrame缓存完成，耗时: {end_time - start_time:.2f}秒")
    
    # 2. 字典
    logger.info("缓存字典")
    sample_dict = {
        'name': 'Demo Dict',
        'values': list(range(1000)),
        'metadata': {
            'created': datetime.now().isoformat(),
            'version': '1.0.0'
        }
    }
    
    start_time = time.time()
    cache_manager.set('demo_dict', sample_dict, format_type='json')
    end_time = time.time()
    logger.info(f"字典缓存完成，耗时: {end_time - start_time:.2f}秒")
    
    # 从缓存获取数据
    logger.info("从缓存获取DataFrame")
    start_time = time.time()
    cached_df = cache_manager.get('demo_dataframe')
    end_time = time.time()
    
    if cached_df is not None:
        logger.info(f"从缓存获取DataFrame成功，耗时: {end_time - start_time:.2f}秒")
        logger.info(f"缓存的DataFrame形状: {cached_df.shape}")
    
    logger.info("从缓存获取字典")
    start_time = time.time()
    cached_dict = cache_manager.get('demo_dict')
    end_time = time.time()
    
    if cached_dict is not None:
        logger.info(f"从缓存获取字典成功，耗时: {end_time - start_time:.2f}秒")
        logger.info(f"缓存的字典键: {list(cached_dict.keys())}")
    
    # 获取缓存统计信息
    stats = cache_manager.get_stats()
    logger.info(f"缓存统计信息: {stats}")
    
    # 清除部分缓存
    logger.info("清除部分缓存")
    cache_manager.clear('demo_dict')
    
    # 再次获取统计信息
    stats = cache_manager.get_stats()
    logger.info(f"清除后缓存统计信息: {stats}")
    
    # 最后清除所有缓存
    logger.info("清除所有缓存")
    cache_manager.clear()

def create_mock_data():
    """创建模拟股票数据用于演示"""
    # 设定日期范围
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # 初始价格
    base_price = 50.0
    
    # 创建价格数据
    np.random.seed(42)  # 设置随机种子以便结果可重现
    
    # 随机游走生成价格
    returns = np.random.normal(0.0005, 0.015, size=len(dates))
    price_series = base_price * (1 + returns).cumprod()
    
    # 创建OHLC数据
    data = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': price_series * (1 + np.random.normal(0, 0.005, size=len(dates))),
        'high': price_series * (1 + abs(np.random.normal(0, 0.01, size=len(dates)))),
        'low': price_series * (1 - abs(np.random.normal(0, 0.01, size=len(dates)))),
        'close': price_series,
        'volume': np.random.randint(1000000, 10000000, size=len(dates)),
        'amount': np.random.randint(50000000, 500000000, size=len(dates))
    })
    
    # 确保high >= open, close和low <= open, close
    for i in range(len(data)):
        row = data.iloc[i]
        max_price = max(row['open'], row['close'])
        min_price = min(row['open'], row['close'])
        
        if row['high'] < max_price:
            data.loc[data.index[i], 'high'] = max_price * (1 + np.random.uniform(0.001, 0.01))
        
        if row['low'] > min_price:
            data.loc[data.index[i], 'low'] = min_price * (1 - np.random.uniform(0.001, 0.01))
    
    return data

def introduce_quality_issues(data):
    """向数据中引入一些质量问题用于演示"""
    # 复制数据
    df = data.copy()
    
    # 1. 添加重复行
    duplicate_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    duplicates = df.loc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 2. 添加缺失值
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            missing_indices = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
            df.loc[missing_indices, col] = np.nan
    
    # 3. 添加无效的价格数据（负值或零）
    if 'close' in df.columns:
        invalid_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
        df.loc[invalid_indices, 'close'] = -df.loc[invalid_indices, 'close']
    
    # 4. 添加逻辑错误（如最高价低于收盘价）
    if all(col in df.columns for col in ['high', 'close']):
        invalid_indices = np.random.choice(df.index, size=int(len(df) * 0.04), replace=False)
        df.loc[invalid_indices, 'high'] = df.loc[invalid_indices, 'close'] * 0.9
    
    # 5. 添加异常值
    if 'volume' in df.columns:
        outlier_indices = np.random.choice(df.index, size=5, replace=False)
        df.loc[outlier_indices, 'volume'] = df['volume'].max() * 10
    
    return df

def main():
    """主函数：运行完整演示"""
    logger.info("=" * 50)
    logger.info("增强版数据处理系统演示开始")
    logger.info("=" * 50)
    
    # 1. 数据获取演示
    data = demo_data_fetching()
    
    # 2. 数据处理演示
    processed_data = demo_data_processing(data)
    
    # 3. 数据质量检查演示
    quality_processed_data = demo_data_quality(data)
    
    # 4. 缓存管理演示
    demo_caching()
    
    # 保存处理后的数据到结果目录
    results_path = os.path.join(RESULTS_DIR, 'processed_data.parquet')
    if processed_data is not None and not processed_data.empty:
        processed_data.to_parquet(results_path)
        logger.info(f"处理后的数据已保存到: {results_path}")
    
    logger.info("=" * 50)
    logger.info("增强版数据处理系统演示完成")
    logger.info("=" * 50)

if __name__ == "__main__":
    main() 