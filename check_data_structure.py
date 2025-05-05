#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票数据结构检查脚本
专门用于检查股票数据的结构，以便诊断技术指标计算问题
"""

import os
import sys
import logging
import pandas as pd
import json
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入动量分析模块
try:
    from momentum_analysis_enhanced_performance import EnhancedMomentumAnalyzer
    modules_loaded = True
except ImportError as e:
    logger.error(f"导入动量分析模块失败: {str(e)}")
    modules_loaded = False

def inspect_data_structure():
    """检查实际股票数据的结构"""
    if not modules_loaded:
        logger.error("无法加载必要的模块，检查终止")
        return
    
    # 初始化分析器
    logger.info("初始化增强版动量分析器...")
    analyzer = EnhancedMomentumAnalyzer(use_tushare=True, cache_size=50)
    
    # 获取股票列表
    logger.info("获取股票列表...")
    stock_list = analyzer.get_stock_list()
    
    if stock_list is None or stock_list.empty:
        logger.error("获取股票列表失败")
        return
    
    logger.info(f"成功获取股票列表，共 {len(stock_list)} 支股票")
    
    # 获取一些样本股票
    sample_stocks = []
    for i in range(min(5, len(stock_list))):
        sample_stocks.append({
            'ts_code': stock_list.iloc[i]['ts_code'],
            'name': stock_list.iloc[i]['name']
        })
    
    logger.info(f"选取 {len(sample_stocks)} 支样本股票进行检查")
    
    # 详细检查每支股票的数据
    for stock in sample_stocks:
        ts_code = stock['ts_code']
        name = stock['name']
        
        logger.info(f"\n检查 {name}({ts_code}) 的数据...")
        
        # 获取股票数据
        data = analyzer.get_stock_daily_data(ts_code)
        
        if data is None or data.empty:
            logger.warning(f"无法获取 {ts_code} 的数据")
            continue
        
        logger.info(f"1. 数据基本信息:")
        logger.info(f"   - 记录数量: {len(data)}")
        logger.info(f"   - 日期范围: {data.index.min()} 至 {data.index.max()}")
        logger.info(f"   - 数据类型: {type(data)}")
        
        logger.info(f"2. 数据列信息:")
        for col in data.columns:
            logger.info(f"   - {col}: {data[col].dtype}")
        
        logger.info(f"3. 索引信息:")
        logger.info(f"   - 索引类型: {type(data.index)}")
        logger.info(f"   - 索引dtype: {data.index.dtype}")
        
        # 检查是否有空值
        null_counts = data.isnull().sum()
        non_zero_nulls = null_counts[null_counts > 0]
        if len(non_zero_nulls) > 0:
            logger.warning(f"4. 空值检查 - 存在空值:")
            for col, count in non_zero_nulls.items():
                logger.warning(f"   - {col}: {count} 个空值")
        else:
            logger.info(f"4. 空值检查 - 无空值")
        
        # 检查交易量列
        if 'vol' in data.columns:
            logger.info(f"5. 交易量检查:")
            logger.info(f"   - 最小值: {data['vol'].min()}")
            logger.info(f"   - 最大值: {data['vol'].max()}")
            logger.info(f"   - 平均值: {data['vol'].mean()}")
        
        # 数据示例
        logger.info(f"6. 数据示例(前3行):")
        logger.info(f"{data.head(3)}")
        
        # 尝试计算技术指标
        logger.info(f"7. 尝试计算技术指标...")
        try:
            start_time = datetime.now()
            processed_data = analyzer.calculate_momentum_vectorized(data.copy())
            duration = (datetime.now() - start_time).total_seconds()
            
            if processed_data is not None and not processed_data.empty:
                # 检查新增的指标列
                new_columns = [col for col in processed_data.columns if col not in data.columns]
                
                logger.info(f"   - 计算耗时: {duration:.4f}秒")
                logger.info(f"   - 计算结果: 成功")
                logger.info(f"   - 新增指标: {new_columns}")
                
                # 显示计算后的示例数据
                if new_columns:
                    logger.info(f"   - 指标示例(尾部3行):")
                    sample_cols = ['close'] + new_columns
                    logger.info(f"{processed_data[sample_cols].tail(3)}")
            else:
                logger.warning(f"   - 计算结果: 空或None")
                logger.warning(f"   - 原始数据类型: {type(data)}")
                logger.warning(f"   - 原始数据形状: {data.shape}")
                logger.warning(f"   - 处理后数据: {type(processed_data)}")
                
                # 尝试手动计算一些基本指标
                logger.info(f"8. 尝试手动计算基本指标...")
                
                try:
                    # 计算简单的移动平均线
                    if 'close' in data.columns:
                        ma20 = data['close'].rolling(window=20).mean()
                        logger.info(f"   - 手动计算MA20(尾部3个值): {ma20.tail(3).to_list()}")
                        
                        # 计算动量
                        momentum_20 = data['close'] / data['close'].shift(20) - 1
                        logger.info(f"   - 手动计算动量20(尾部3个值): {momentum_20.tail(3).to_list()}")
                except Exception as e:
                    logger.error(f"   - 手动计算指标失败: {str(e)}")
        except Exception as e:
            logger.error(f"   - 计算技术指标时出错: {str(e)}")
        
        # 写入样本数据到文件
        try:
            sample_file = f"{ts_code}_sample_data.csv"
            data.head(50).to_csv(sample_file)
            logger.info(f"已将 {ts_code} 的样本数据保存到 {sample_file}")
        except Exception as e:
            logger.error(f"保存样本数据失败: {str(e)}")

if __name__ == "__main__":
    logger.info("开始检查股票数据结构...")
    inspect_data_structure()
    logger.info("数据结构检查完成") 