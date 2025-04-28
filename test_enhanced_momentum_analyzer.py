#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试增强版的动量分析器
"""

import sys
import os
import logging
import time

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入增强的动量分析器
try:
    from enhanced_momentum_analysis import EnhancedMomentumAnalyzer
except ImportError:
    logger.error("无法导入enhanced_momentum_analysis模块，请确保该文件存在")
    sys.exit(1)

def test_enhanced_momentum_analyzer():
    """测试增强版动量分析器的功能"""
    
    # 测试常见股票
    test_stocks = [
        '601318.SH',  # 中国平安
        '000651.SZ',  # 格力电器
        '000333.SZ',  # 美的集团
        '600519.SH',  # 贵州茅台
        '000002.SZ',  # 万科A
        '600036.SH',  # 招商银行
        '000999.SZ',  # 华润三九
        '600276.SH',  # 恒瑞医药
    ]
    
    logger.info("创建增强版动量分析器实例...")
    analyzer = EnhancedMomentumAnalyzer(use_tushare=True, cache_timeout=3600)
    
    # 测试函数 1: 股票名称获取
    logger.info("测试批量获取股票名称功能...")
    start_time = time.time()
    
    # 预先运行一次批量获取股票名称（这应该已在analyze_stocks_enhanced中自动完成）
    for ts_code in test_stocks:
        # 获取并显示股票名称
        name = analyzer.get_stock_name(ts_code)
        logger.info(f"股票代码: {ts_code} -> 股票名称: {name}")
    
    name_time = time.time() - start_time
    logger.info(f"获取股票名称耗时: {name_time:.2f}秒")
    
    # 测试函数 2: 获取股票行业
    logger.info("测试获取股票行业功能...")
    start_time = time.time()
    
    for ts_code in test_stocks:
        # 获取并显示股票行业
        industry = analyzer.get_stock_industry(ts_code)
        logger.info(f"股票代码: {ts_code} -> 行业: {industry}")
    
    industry_time = time.time() - start_time
    logger.info(f"获取股票行业耗时: {industry_time:.2f}秒")
    
    # 测试函数 3: 分析股票
    logger.info("测试分析股票功能...")
    
    # 直接使用analyze_stocks_enhanced，这将测试我们的批量优化
    start_time = time.time()
    
    results = analyzer.analyze_stocks_enhanced(
        test_stocks, 
        min_score=0,  # 设置为0以便查看所有股票的分析结果
        sample_size=None  # 不限制样本大小
    )
    
    analysis_time = time.time() - start_time
    
    # 展示分析结果
    logger.info(f"分析股票耗时: {analysis_time:.2f}秒")
    logger.info(f"分析结果数量: {len(results)}")
    
    for result in results:
        logger.info("-" * 50)
        logger.info(f"股票: {result['name']}({result['ts_code']})")
        logger.info(f"得分: {result['score']}")
        logger.info(f"信号: {result['signals']}")
        logger.info(f"最新价格: {result['last_price']}")
        logger.info(f"涨跌幅: {result['change_pct']}%")
        logger.info(f"分析时间: {result['analysis_time']}")
        if 'chart_path' in result:
            logger.info(f"图表路径: {result['chart_path']}")

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("测试增强版动量分析器")
    logger.info("=" * 50)
    
    test_enhanced_momentum_analyzer() 