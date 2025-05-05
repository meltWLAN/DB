#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础动量分析测试脚本
测试动量分析模块的基本功能，不依赖分布式计算框架
"""

import os
import sys
import time
import logging
import pandas as pd
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

def test_enhanced_momentum(sample_size=3):
    """
    测试增强版动量分析模块的基本功能
    
    Args:
        sample_size: 测试的股票数量
    """
    if not modules_loaded:
        logger.error("无法加载必要的模块，测试终止")
        return False
    
    logger.info(f"开始增强版动量分析测试: 样本大小={sample_size}")
    
    try:
        # 初始化增强版分析器
        logger.info("初始化 EnhancedMomentumAnalyzer")
        analyzer = EnhancedMomentumAnalyzer(cache_size=50)
        
        # 测试获取股票列表
        stock_list = analyzer.get_stock_list()
        if stock_list is not None and not stock_list.empty:
            logger.info(f"成功获取股票列表，共 {len(stock_list)} 支股票")
            # 显示前几支股票信息
            logger.info(f"股票列表示例: \n{stock_list.head(3)}")
        else:
            logger.error("获取股票列表失败")
            return False
        
        # 测试缓存预热
        logger.info("\n测试缓存预热功能")
        try:
            start_time = time.time()
            analyzer.warm_up_cache(stock_list.head(sample_size))
            logger.info(f"缓存预热完成，耗时: {time.time() - start_time:.2f}秒")
        except Exception as e:
            logger.error(f"缓存预热失败: {str(e)}")
        
        # 测试技术指标计算
        logger.info("\n测试技术指标计算")
        try:
            # 选取一支股票进行测试
            test_code = stock_list.iloc[0]['ts_code']
            test_name = stock_list.iloc[0]['name']
            logger.info(f"选择测试股票: {test_name}({test_code})")
            
            # 获取股票数据
            stock_data = analyzer.get_stock_daily_data(test_code)
            
            if stock_data is not None and not stock_data.empty:
                logger.info(f"成功获取 {test_code} 的日线数据，共 {len(stock_data)} 条记录")
                logger.info(f"数据示例: \n{stock_data.head(3)}")
                
                # 计算向量化技术指标
                vectorized_data = analyzer.calculate_momentum_vectorized(stock_data.copy())
                
                if not vectorized_data.empty:
                    logger.info("向量化技术指标计算成功")
                    # 检查是否包含关键指标
                    key_indicators = ['ma20', 'ma60', 'momentum_20', 'rsi', 'macd']
                    available = [col for col in key_indicators if col in vectorized_data.columns]
                    logger.info(f"计算的指标: {available}")
                    
                    # 显示部分计算结果
                    if available:
                        logger.info(f"技术指标计算结果示例: \n{vectorized_data[['close'] + available].tail(3)}")
                else:
                    logger.warning("向量化技术指标计算结果为空")
            else:
                logger.warning(f"获取 {test_code} 的数据失败，尝试获取其他股票")
                
                # 尝试其他股票
                for i in range(1, min(5, len(stock_list))):
                    test_code = stock_list.iloc[i]['ts_code']
                    test_name = stock_list.iloc[i]['name']
                    logger.info(f"尝试股票: {test_name}({test_code})")
                    
                    stock_data = analyzer.get_stock_daily_data(test_code)
                    if stock_data is not None and not stock_data.empty:
                        logger.info(f"成功获取 {test_code} 的日线数据，共 {len(stock_data)} 条记录")
                        
                        # 计算技术指标
                        vectorized_data = analyzer.calculate_momentum_vectorized(stock_data.copy())
                        if not vectorized_data.empty:
                            logger.info("向量化技术指标计算成功")
                            break
                
                if stock_data is None or stock_data.empty:
                    logger.error("无法获取任何股票数据，技术指标测试失败")
        except Exception as e:
            logger.error(f"技术指标计算测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 测试单只股票分析
        logger.info("\n测试单只股票分析")
        try:
            if stock_list is not None and not stock_list.empty:
                test_stock = stock_list.iloc[0].to_dict()
                result = analyzer.analyze_single_stock_optimized(test_stock)
                
                if result:
                    logger.info(f"单只股票分析成功: {result['name']}({result['ts_code']}), 得分: {result.get('score', 'N/A')}")
                    if 'score_details' in result:
                        logger.info(f"得分详情: {result['score_details']}")
                else:
                    logger.warning("单只股票分析未返回结果，可能是因为得分不符合标准或者数据获取失败")
                    
                    # 尝试其他股票
                    for i in range(1, min(5, len(stock_list))):
                        test_stock = stock_list.iloc[i].to_dict()
                        result = analyzer.analyze_single_stock_optimized(test_stock)
                        if result:
                            logger.info(f"单只股票分析成功: {result['name']}({result['ts_code']}), 得分: {result.get('score', 'N/A')}")
                            break
        except Exception as e:
            logger.error(f"单只股票分析测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 测试批量分析
        logger.info("\n测试批量分析功能")
        try:
            start_time = time.time()
            results = analyzer.analyze_stocks(stock_list.head(sample_size), min_score=50)
            duration = time.time() - start_time
            
            logger.info(f"批量分析完成，耗时: {duration:.2f}秒，平均每只股票 {duration/sample_size:.2f}秒")
            logger.info(f"分析结果数量: {len(results)}")
            
            if results:
                # 显示前几个结果
                for i, r in enumerate(results[:min(3, len(results))]):
                    logger.info(f"结果 {i+1}: {r['name']}({r['ts_code']}), 得分: {r.get('score', 'N/A')}")
            else:
                logger.info("没有股票符合筛选条件 (分数 >= 50)")
        except Exception as e:
            logger.error(f"批量分析测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 测试缓存机制
        logger.info("\n测试缓存机制")
        try:
            # 连续获取两次相同股票数据，应该有缓存命中
            test_code = stock_list.iloc[0]['ts_code']
            
            # 第一次获取
            start_time = time.time()
            analyzer.get_stock_daily_data(test_code)
            first_time = time.time() - start_time
            
            # 第二次获取（应该命中缓存）
            start_time = time.time()
            analyzer.get_stock_daily_data(test_code)
            second_time = time.time() - start_time
            
            logger.info(f"第一次获取耗时: {first_time:.4f}秒")
            logger.info(f"第二次获取耗时: {second_time:.4f}秒")
            
            if second_time < first_time:
                logger.info("缓存机制工作正常，第二次获取速度更快")
            else:
                logger.warning("缓存可能未生效，第二次获取速度没有提升")
        except Exception as e:
            logger.error(f"缓存机制测试失败: {str(e)}")
        
        logger.info("\n增强版动量分析测试完成")
        return True
    
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="增强版动量分析模块测试")
    parser.add_argument('-s', '--sample', type=int, default=3, help="测试样本大小")
    
    args = parser.parse_args()
    
    test_enhanced_momentum(args.sample)

if __name__ == "__main__":
    main() 