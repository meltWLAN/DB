#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试增强版的股票名称获取函数
"""

import sys
import os
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入增强的动量分析器
try:
    from enhanced_momentum_analysis import get_stock_name, get_stock_names_batch
except ImportError:
    logger.error("无法导入enhanced_momentum_analysis模块，请确保该文件存在")
    sys.exit(1)

def test_get_stock_name():
    """测试get_stock_name函数的功能"""
    
    # 测试常见股票
    test_codes = [
        '601318.SH',  # 中国平安
        '000651.SZ',  # 格力电器
        '000333.SZ',  # 美的集团
        '600519.SH',  # 贵州茅台
        '000002.SZ',  # 万科A
        '600036.SH',  # 招商银行
        '000999.SZ',  # 不存在于映射中的股票
        '123456.SZ',  # 假股票代码
    ]
    
    logger.info("开始测试get_stock_name函数...")
    
    for code in test_codes:
        name = get_stock_name(code)
        logger.info(f"股票代码: {code} -> 股票名称: {name}")
    
    logger.info("测试完成")

def test_get_stock_names_batch():
    """测试get_stock_names_batch函数的功能"""
    
    # 测试常见股票
    test_codes = [
        '601318.SH',  # 中国平安
        '000651.SZ',  # 格力电器
        '000333.SZ',  # 美的集团
        '600519.SH',  # 贵州茅台
        '000002.SZ',  # 万科A
        '600036.SH',  # 招商银行
        '000999.SZ',  # 不存在于映射中的股票
        '123456.SZ',  # 假股票代码
    ]
    
    logger.info("开始测试get_stock_names_batch函数...")
    
    result = get_stock_names_batch(test_codes)
    logger.info(f"批量查询结果包含 {len(result)} 个股票名称")
    
    # 输出结果
    for code, name in result.items():
        logger.info(f"批量查询 - 股票代码: {code} -> 股票名称: {name}")
    
    logger.info("批量测试完成")

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("测试增强版的股票名称获取函数")
    logger.info("=" * 50)
    
    # 测试单个获取函数
    test_get_stock_name()
    
    logger.info("-" * 50)
    
    # 测试批量获取函数
    test_get_stock_names_batch() 