#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试股票数据管理器和动量分析器的集成
验证使用真实数据而非模拟数据
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入数据管理器和动量分析器
from stock_data_manager import StockDataManager
from momentum_analysis import MomentumAnalyzer

def test_stock_data_manager():
    """测试数据管理器基本功能"""
    logger.info("=" * 50)
    logger.info("测试数据管理器基本功能")
    logger.info("=" * 50)
    
    # 创建数据管理器实例
    manager = StockDataManager()
    
    # 测试获取股票列表
    logger.info("\n1. 测试获取股票列表")
    stock_list = manager.get_stock_list()
    logger.info(f"获取到 {len(stock_list)} 支股票")
    logger.info(f"前5支股票:\n{stock_list.head()}")
    
    # 测试获取单个股票数据
    logger.info("\n2. 测试获取单个股票数据")
    # 选择第一支股票进行测试
    test_stock = stock_list.iloc[0]['ts_code']
    logger.info(f"获取 {test_stock} 的历史数据")
    
    # 获取2年数据
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y%m%d')
    
    # 获取数据
    stock_data = manager.get_stock_data(test_stock, start_date, end_date)
    logger.info(f"获取到 {len(stock_data)} 条数据")
    logger.info(f"数据样例:\n{stock_data.head()}")
    
    # 测试更新股票数据
    logger.info("\n3. 测试更新股票数据")
    # 仅更新指定的几支股票，避免更新太多
    test_stocks = ['000001.SZ', '600036.SH', '601318.SH']
    for ts_code in test_stocks:
        # 查找最新数据日期
        logger.info(f"更新 {ts_code} 的最新数据")
        latest_data = manager.get_stock_data(ts_code, 
                                           (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                                           datetime.now().strftime('%Y%m%d'))
        if not latest_data.empty:
            latest_date = latest_data.index[0]
            logger.info(f"{ts_code} 最新数据日期: {latest_date.strftime('%Y-%m-%d')}")
    
    # 测试数据完整性检查
    logger.info("\n4. 测试数据完整性检查")
    # 仅检查几支股票
    for ts_code in test_stocks:
        logger.info(f"检查 {ts_code} 数据完整性")
        result = manager.check_data_integrity(ts_code)
        if 'results' in result and result['results']:
            for r in result['results']:
                if 'completeness' in r:
                    logger.info(f"{ts_code} 数据完整性: {r['completeness']}%, 缺失 {r['missing_days']} 天")
    
    logger.info("\n数据管理器基本功能测试完成")

def test_integration_with_momentum_analyzer():
    """测试动量分析器与数据管理器的集成"""
    logger.info("=" * 50)
    logger.info("测试动量分析器与数据管理器的集成")
    logger.info("=" * 50)
    
    # 创建动量分析器实例
    analyzer = MomentumAnalyzer()
    
    # 测试获取股票列表
    logger.info("\n1. 测试通过动量分析器获取股票列表")
    stocks = analyzer.get_stock_list()
    logger.info(f"获取到 {len(stocks)} 支股票")
    
    # 测试获取并分析单个股票
    logger.info("\n2. 测试获取并分析单个股票")
    # 选择一些优质蓝筹股测试
    test_stocks = [
        {'code': '601318.SH', 'name': '中国平安'},
        {'code': '600036.SH', 'name': '招商银行'},
        {'code': '000651.SZ', 'name': '格力电器'},
        {'code': '000333.SZ', 'name': '美的集团'},
        {'code': '600519.SH', 'name': '贵州茅台'}
    ]
    
    results = []
    
    for stock in test_stocks:
        ts_code = stock['code']
        name = stock['name']
        logger.info(f"获取 {name}({ts_code}) 的数据")
        
        # 获取股票数据
        data = analyzer.get_stock_daily_data(ts_code)
        if data is None or data.empty:
            logger.warning(f"获取 {name}({ts_code}) 数据失败")
            continue
        
        logger.info(f"成功获取 {name}({ts_code}) 数据，共 {len(data)} 条记录")
        logger.info(f"时间范围: {data.index.min().strftime('%Y-%m-%d')} 至 {data.index.max().strftime('%Y-%m-%d')}")
        
        # 计算技术指标
        data_with_indicators = analyzer.calculate_momentum(data)
        if data_with_indicators is None or data_with_indicators.empty:
            logger.warning(f"计算 {name}({ts_code}) 技术指标失败")
            continue
        
        #
        score, score_details = analyzer.calculate_momentum_score(data_with_indicators)
        logger.info(f"{name}({ts_code}) 动量得分: {score}")
        logger.info(f"得分详情: {score_details}")
        
        # 添加到结果
        results.append({
            'ts_code': ts_code,
            'name': name,
            'score': score,
            'details': score_details,
            'data': data_with_indicators
        })
        
        # 绘制图表
        try:
            chart_path = f"./results/charts/{ts_code}_momentum_test.png"
            analyzer.plot_stock_chart(data_with_indicators, ts_code, name, score_details, chart_path)
            logger.info(f"已绘制 {name}({ts_code}) 的动量分析图表: {chart_path}")
        except Exception as e:
            logger.error(f"绘制图表失败: {str(e)}")
    
    # 测试批量分析
    logger.info("\n3. 测试批量分析股票")
    # 从股票列表中选择10支进行分析
    if not stocks.empty and len(stocks) > 10:
        sample = stocks.head(10)
        logger.info(f"对样本 {len(sample)} 支股票进行动量分析")
        
        # 分析股票
        analysis_results = analyzer.analyze_stocks(sample, sample_size=10, min_score=0)
        
        # 显示结果
        logger.info(f"分析完成，结果数量: {len(analysis_results)}")
        for result in analysis_results:
            logger.info(f"{result['name']}({result['ts_code']}): 得分={result['total_score']}, 等级={result.get('grade', 'N/A')}")
    
    logger.info("\n动量分析器与数据管理器集成测试完成")

def main():
    """主函数"""
    logger.info("开始测试数据管理器和动量分析器")
    
    # 测试数据管理器基本功能
    test_stock_data_manager()
    
    # 测试与动量分析器的集成
    test_integration_with_momentum_analyzer()
    
    logger.info("所有测试完成")

if __name__ == "__main__":
    main() 