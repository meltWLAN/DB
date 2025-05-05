#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动量分析性能测试脚本
对比优化前后的性能差异
"""

import os
import sys
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入动量分析模块
try:
    from momentum_analysis import MomentumAnalyzer
    from momentum_analysis_optimized import MomentumAnalyzer as OptimizedAnalyzer
    from momentum_analysis_enhanced_performance import EnhancedMomentumAnalyzer
    modules_loaded = True
except ImportError as e:
    logger.error(f"导入动量分析模块失败: {str(e)}")
    modules_loaded = False

def run_performance_test(sample_size=20, use_multiprocessing=True):
    """
    运行性能测试，对比不同版本的动量分析模块
    
    Args:
        sample_size: 测试的股票数量
        use_multiprocessing: 是否使用多进程
    
    Returns:
        dict: 包含性能测试结果的字典
    """
    if not modules_loaded:
        return {"error": "无法导入必要的模块"}
    
    results = {
        "sample_size": sample_size,
        "use_multiprocessing": use_multiprocessing,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "performance": {}
    }
    
    try:
        # 1. 测试原始版本
        logger.info(f"测试原始版本 (样本: {sample_size}, 多进程: {use_multiprocessing})")
        analyzer = MomentumAnalyzer(use_tushare=True, use_multiprocessing=use_multiprocessing)
        stocks = analyzer.get_stock_list()
        
        if stocks.empty or len(stocks) == 0:
            return {"error": "无法获取股票列表"}
        
        start_time = time.time()
        original_results = analyzer.analyze_stocks(stocks.head(sample_size), min_score=50)
        end_time = time.time()
        original_duration = end_time - start_time
        
        results["performance"]["original"] = {
            "duration": original_duration,
            "result_count": len(original_results)
        }
        
        logger.info(f"原始版本耗时: {original_duration:.2f}秒, 结果数: {len(original_results)}")
        
        # 2. 测试优化版本
        logger.info(f"测试优化版本 (样本: {sample_size}, 多进程: {use_multiprocessing})")
        opt_analyzer = OptimizedAnalyzer(use_tushare=True, use_multiprocessing=use_multiprocessing)
        
        start_time = time.time()
        opt_results = opt_analyzer.analyze_stocks(stocks.head(sample_size), min_score=50)
        end_time = time.time()
        optimized_duration = end_time - start_time
        
        results["performance"]["optimized"] = {
            "duration": optimized_duration,
            "result_count": len(opt_results),
            "improvement": (original_duration - optimized_duration) / original_duration * 100 if original_duration > 0 else 0
        }
        
        logger.info(f"优化版本耗时: {optimized_duration:.2f}秒, 结果数: {len(opt_results)}")
        logger.info(f"性能提升: {results['performance']['optimized']['improvement']:.2f}%")
        
        # 3. 测试增强性能版本
        logger.info(f"测试增强性能版本 (样本: {sample_size}, 多进程: {use_multiprocessing})")
        enh_analyzer = EnhancedMomentumAnalyzer(use_tushare=True, use_multiprocessing=use_multiprocessing)
        
        # 预热缓存
        enh_analyzer.warm_up_cache(stocks.head(sample_size))
        
        start_time = time.time()
        enh_results = enh_analyzer.analyze_stocks(stocks.head(sample_size), min_score=50)
        end_time = time.time()
        enhanced_duration = end_time - start_time
        
        results["performance"]["enhanced"] = {
            "duration": enhanced_duration,
            "result_count": len(enh_results),
            "improvement": (original_duration - enhanced_duration) / original_duration * 100 if original_duration > 0 else 0,
            "vs_optimized": (optimized_duration - enhanced_duration) / optimized_duration * 100 if optimized_duration > 0 else 0
        }
        
        logger.info(f"增强性能版本耗时: {enhanced_duration:.2f}秒, 结果数: {len(enh_results)}")
        logger.info(f"相比原始版本提升: {results['performance']['enhanced']['improvement']:.2f}%")
        logger.info(f"相比优化版本提升: {results['performance']['enhanced']['vs_optimized']:.2f}%")
        
        # 生成性能对比图表
        generate_performance_chart(results)
        
        return results
    
    except Exception as e:
        logger.error(f"性能测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        
        results["error"] = str(e)
        return results

def generate_performance_chart(results):
    """
    生成性能对比图表
    
    Args:
        results: 性能测试结果
    """
    try:
        performance = results.get("performance", {})
        if not performance:
            logger.warning("没有性能数据，无法生成图表")
            return
            
        # 提取数据
        labels = []
        durations = []
        improvements = []
        
        if "original" in performance:
            labels.append("原始版本")
            durations.append(performance["original"]["duration"])
            improvements.append(0)  # 原始版本作为基准，提升为0%
            
        if "optimized" in performance:
            labels.append("优化版本")
            durations.append(performance["optimized"]["duration"])
            improvements.append(performance["optimized"]["improvement"])
            
        if "enhanced" in performance:
            labels.append("增强性能版本")
            durations.append(performance["enhanced"]["duration"])
            improvements.append(performance["enhanced"]["improvement"])
        
        # 创建图表
        plt.figure(figsize=(14, 8))
        
        # 绘制执行时间对比
        plt.subplot(1, 2, 1)
        bars = plt.bar(labels, durations, color=['#3498db', '#2ecc71', '#9b59b6'])
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f"{height:.2f}秒", ha='center', va='bottom')
        
        plt.title('执行时间对比')
        plt.ylabel('耗时(秒)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 绘制性能提升百分比
        plt.subplot(1, 2, 2)
        bars = plt.bar(labels, improvements, color=['#3498db', '#2ecc71', '#9b59b6'])
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f"{height:.2f}%", ha='center', va='bottom')
        
        plt.title('性能提升百分比(相比原始版本)')
        plt.ylabel('提升百分比(%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加整体标题
        plt.suptitle(f"动量分析模块性能对比 (样本大小: {results['sample_size']})", fontsize=16)
        
        # 调整布局并保存
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        chart_path = os.path.join(output_dir, f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(chart_path)
        
        logger.info(f"性能对比图表已保存至: {chart_path}")
        
    except Exception as e:
        logger.error(f"生成性能图表失败: {str(e)}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="动量分析模块性能测试")
    parser.add_argument('-s', '--sample', type=int, default=20, help="测试样本大小")
    parser.add_argument('-m', '--multiprocessing', action='store_true', help="使用多进程")
    
    args = parser.parse_args()
    
    logger.info(f"开始性能测试: 样本={args.sample}, 多进程={args.multiprocessing}")
    results = run_performance_test(args.sample, args.multiprocessing)
    
    # 显示结果摘要
    if "error" in results:
        logger.error(f"测试失败: {results['error']}")
    else:
        logger.info("性能测试完成")
        
        original = results["performance"].get("original", {}).get("duration", 0)
        optimized = results["performance"].get("optimized", {}).get("duration", 0)
        enhanced = results["performance"].get("enhanced", {}).get("duration", 0)
        
        logger.info("\n性能测试结果摘要:")
        logger.info(f"- 原始版本耗时: {original:.2f}秒")
        logger.info(f"- 优化版本耗时: {optimized:.2f}秒 (提升: {(original-optimized)/original*100:.2f}%)")
        logger.info(f"- 增强版本耗时: {enhanced:.2f}秒 (提升: {(original-enhanced)/original*100:.2f}%)")

if __name__ == "__main__":
    main() 