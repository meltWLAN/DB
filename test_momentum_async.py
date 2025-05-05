#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动量分析异步优化测试脚本
对比各版本之间的性能差异
"""

import os
import sys
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import psutil
import gc
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入动量分析模块
try:
    from momentum_analysis import MomentumAnalyzer
    from momentum_analysis_enhanced_performance import EnhancedMomentumAnalyzer
    from momentum_analysis_async import AsyncMomentumAnalyzer
    modules_loaded = True
except ImportError as e:
    logger.error(f"导入动量分析模块失败: {str(e)}")
    modules_loaded = False

def run_comparison_test(sample_size=20, use_multiprocessing=True):
    """
    运行性能对比测试，比较三个版本的动量分析模块
    
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
        "performance": {},
        "memory_usage": {}
    }
    
    try:
        # 先获取股票列表备用
        logger.info("获取股票列表供测试使用")
        original_analyzer = MomentumAnalyzer(use_tushare=True, use_multiprocessing=False)
        stocks = original_analyzer.get_stock_list()
        
        if stocks.empty or len(stocks) == 0:
            return {"error": "无法获取股票列表"}
        
        # 限制样本大小
        stocks = stocks.head(sample_size)
        
        # 运行垃圾回收，确保公平起点
        gc.collect()
        
        # 1. 测试原始版本
        logger.info(f"测试原始版本 (样本: {sample_size}, 多进程: {use_multiprocessing})")
        analyzer1 = MomentumAnalyzer(use_tushare=True, use_multiprocessing=use_multiprocessing)
        
        # 记录开始时的内存使用
        process = psutil.Process(os.getpid())
        mem_before_1 = process.memory_info().rss / (1024 * 1024)
        
        # 执行分析
        start_time = time.time()
        original_results = analyzer1.analyze_stocks(stocks, min_score=50)
        end_time = time.time()
        original_duration = end_time - start_time
        
        # 记录结束时的内存使用
        mem_after_1 = process.memory_info().rss / (1024 * 1024)
        mem_increase_1 = mem_after_1 - mem_before_1
        
        results["performance"]["original"] = {
            "duration": original_duration,
            "result_count": len(original_results),
            "per_stock_time": original_duration / sample_size
        }
        
        results["memory_usage"]["original"] = {
            "before": mem_before_1,
            "after": mem_after_1,
            "increase": mem_increase_1
        }
        
        logger.info(f"原始版本耗时: {original_duration:.2f}秒, 结果数: {len(original_results)}")
        logger.info(f"内存增加: {mem_increase_1:.2f}MB")
        
        # 清理内存，准备下一个测试
        del analyzer1, original_results
        gc.collect()
        time.sleep(1)  # 短暂暂停，确保资源释放
        
        # 2. 测试第一阶段优化版本
        logger.info(f"测试第一阶段优化版本 (样本: {sample_size}, 多进程: {use_multiprocessing})")
        analyzer2 = EnhancedMomentumAnalyzer(use_tushare=True, use_multiprocessing=use_multiprocessing)
        
        # 记录开始时的内存使用
        mem_before_2 = process.memory_info().rss / (1024 * 1024)
        
        # 预热缓存
        analyzer2.warm_up_cache(stocks, top_n=sample_size)
        
        # 执行分析
        start_time = time.time()
        enhanced_results = analyzer2.analyze_stocks(stocks, min_score=50)
        end_time = time.time()
        enhanced_duration = end_time - start_time
        
        # 记录结束时的内存使用
        mem_after_2 = process.memory_info().rss / (1024 * 1024)
        mem_increase_2 = mem_after_2 - mem_before_2
        
        results["performance"]["enhanced"] = {
            "duration": enhanced_duration,
            "result_count": len(enhanced_results),
            "per_stock_time": enhanced_duration / sample_size,
            "vs_original": (original_duration - enhanced_duration) / original_duration * 100
        }
        
        results["memory_usage"]["enhanced"] = {
            "before": mem_before_2,
            "after": mem_after_2,
            "increase": mem_increase_2
        }
        
        logger.info(f"第一阶段优化版本耗时: {enhanced_duration:.2f}秒, 结果数: {len(enhanced_results)}")
        logger.info(f"相比原始版本提升: {results['performance']['enhanced']['vs_original']:.2f}%")
        logger.info(f"内存增加: {mem_increase_2:.2f}MB")
        
        # 清理内存，准备下一个测试
        del analyzer2, enhanced_results
        gc.collect()
        time.sleep(1)  # 短暂暂停，确保资源释放
        
        # 3. 测试第二阶段异步优化版本
        logger.info(f"测试第二阶段异步优化版本 (样本: {sample_size})")
        analyzer3 = AsyncMomentumAnalyzer(use_tushare=True, use_multiprocessing=use_multiprocessing)
        
        # 记录开始时的内存使用
        mem_before_3 = process.memory_info().rss / (1024 * 1024)
        
        # 执行分析 (注意异步版本已包含预热)
        start_time = time.time()
        async_results = analyzer3.analyze_stocks(stocks, min_score=50)
        end_time = time.time()
        async_duration = end_time - start_time
        
        # 记录结束时的内存使用
        mem_after_3 = process.memory_info().rss / (1024 * 1024)
        mem_increase_3 = mem_after_3 - mem_before_3
        
        # 获取缓存命中统计
        cache_hits = analyzer3.cache_hit_stats["hits"]
        cache_misses = analyzer3.cache_hit_stats["misses"]
        cache_hit_ratio = cache_hits / (cache_hits + cache_misses) * 100 if (cache_hits + cache_misses) > 0 else 0
        
        results["performance"]["async"] = {
            "duration": async_duration,
            "result_count": len(async_results),
            "per_stock_time": async_duration / sample_size,
            "vs_original": (original_duration - async_duration) / original_duration * 100,
            "vs_enhanced": (enhanced_duration - async_duration) / enhanced_duration * 100,
            "cache_hit_ratio": cache_hit_ratio
        }
        
        results["memory_usage"]["async"] = {
            "before": mem_before_3,
            "after": mem_after_3,
            "increase": mem_increase_3
        }
        
        logger.info(f"第二阶段异步优化版本耗时: {async_duration:.2f}秒, 结果数: {len(async_results)}")
        logger.info(f"相比原始版本提升: {results['performance']['async']['vs_original']:.2f}%")
        logger.info(f"相比第一阶段提升: {results['performance']['async']['vs_enhanced']:.2f}%")
        logger.info(f"缓存命中率: {cache_hit_ratio:.2f}%")
        logger.info(f"内存增加: {mem_increase_3:.2f}MB")
        
        # 生成性能对比图表
        generate_performance_chart(results)
        generate_memory_chart(results)
        
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
        per_stock_times = []
        
        if "original" in performance:
            labels.append("原始版本")
            durations.append(performance["original"]["duration"])
            improvements.append(0)  # 原始版本作为基准，提升为0%
            per_stock_times.append(performance["original"]["per_stock_time"])
            
        if "enhanced" in performance:
            labels.append("第一阶段优化")
            durations.append(performance["enhanced"]["duration"])
            improvements.append(performance["enhanced"]["vs_original"])
            per_stock_times.append(performance["enhanced"]["per_stock_time"])
            
        if "async" in performance:
            labels.append("第二阶段异步优化")
            durations.append(performance["async"]["duration"])
            improvements.append(performance["async"]["vs_original"])
            per_stock_times.append(performance["async"]["per_stock_time"])
        
        # 创建图表
        plt.figure(figsize=(15, 12))
        
        # 1. 绘制总执行时间对比
        plt.subplot(2, 2, 1)
        bars = plt.bar(labels, durations, color=['#3498db', '#2ecc71', '#9b59b6'])
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f"{height:.2f}秒", ha='center', va='bottom')
        
        plt.title('总执行时间对比')
        plt.ylabel('耗时(秒)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. 绘制性能提升百分比
        plt.subplot(2, 2, 2)
        bars = plt.bar(labels, improvements, color=['#3498db', '#2ecc71', '#9b59b6'])
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f"{height:.2f}%", ha='center', va='bottom')
        
        plt.title('性能提升百分比(相比原始版本)')
        plt.ylabel('提升百分比(%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. 绘制每股票平均时间
        plt.subplot(2, 2, 3)
        bars = plt.bar(labels, per_stock_times, color=['#3498db', '#2ecc71', '#9b59b6'])
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f"{height:.2f}秒", ha='center', va='bottom')
        
        plt.title('每股票平均处理时间')
        plt.ylabel('平均耗时(秒/股)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 4. 绘制结果数量对比
        result_counts = [performance[ver].get("result_count", 0) for ver in performance.keys()]
        plt.subplot(2, 2, 4)
        bars = plt.bar(labels, result_counts, color=['#3498db', '#2ecc71', '#9b59b6'])
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f"{height}", ha='center', va='bottom')
        
        plt.title('结果数量对比')
        plt.ylabel('筛选出的股票数量')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加整体标题
        plt.suptitle(f"动量分析模块优化性能对比 (样本大小: {results['sample_size']})", fontsize=16)
        
        # 调整布局并保存
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        chart_path = os.path.join(output_dir, f"performance_comparison_async_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(chart_path)
        
        logger.info(f"性能对比图表已保存至: {chart_path}")
        
    except Exception as e:
        logger.error(f"生成性能图表失败: {str(e)}")

def generate_memory_chart(results):
    """
    生成内存使用对比图表
    
    Args:
        results: 性能测试结果
    """
    try:
        memory_usage = results.get("memory_usage", {})
        if not memory_usage:
            logger.warning("没有内存使用数据，无法生成图表")
            return
        
        # 提取数据
        labels = []
        memory_before = []
        memory_after = []
        memory_increase = []
        
        for version, data in memory_usage.items():
            if version == "original":
                labels.append("原始版本")
            elif version == "enhanced":
                labels.append("第一阶段优化")
            elif version == "async":
                labels.append("第二阶段异步优化")
            else:
                labels.append(version)
                
            memory_before.append(data.get("before", 0))
            memory_after.append(data.get("after", 0))
            memory_increase.append(data.get("increase", 0))
        
        # 创建图表
        plt.figure(figsize=(14, 8))
        
        # 1. 绘制内存使用变化
        plt.subplot(1, 2, 1)
        x = range(len(labels))
        width = 0.35
        
        plt.bar(x, memory_before, width, label='执行前', color='#3498db')
        plt.bar([i + width for i in x], memory_after, width, label='执行后', color='#e74c3c')
        
        plt.xlabel('版本')
        plt.ylabel('内存使用 (MB)')
        plt.title('内存使用对比')
        plt.xticks([i + width/2 for i in x], labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. 绘制内存增加量
        plt.subplot(1, 2, 2)
        bars = plt.bar(labels, memory_increase, color=['#3498db', '#2ecc71', '#9b59b6'])
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f"{height:.2f} MB", ha='center', va='bottom')
        
        plt.title('内存增加量')
        plt.ylabel('内存增加 (MB)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加整体标题
        plt.suptitle(f"动量分析模块内存使用对比 (样本大小: {results['sample_size']})", fontsize=16)
        
        # 调整布局并保存
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        chart_path = os.path.join(output_dir, f"memory_comparison_async_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(chart_path)
        
        logger.info(f"内存使用对比图表已保存至: {chart_path}")
        
    except Exception as e:
        logger.error(f"生成内存图表失败: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="动量分析模块异步优化性能测试")
    parser.add_argument('-s', '--sample', type=int, default=10, help="测试样本大小")
    parser.add_argument('-m', '--multiprocessing', action='store_true', help="使用多进程")
    
    args = parser.parse_args()
    
    logger.info(f"开始性能测试: 样本={args.sample}, 多进程={args.multiprocessing}")
    results = run_comparison_test(args.sample, args.multiprocessing)
    
    # 显示结果摘要
    if "error" in results:
        logger.error(f"测试失败: {results['error']}")
    else:
        logger.info("性能测试完成")
        
        performance = results.get("performance", {})
        original = performance.get("original", {}).get("duration", 0)
        enhanced = performance.get("enhanced", {}).get("duration", 0)
        async_time = performance.get("async", {}).get("duration", 0)
        
        # 计算提升百分比
        enh_improvement = (original - enhanced) / original * 100 if original > 0 else 0
        async_improvement = (original - async_time) / original * 100 if original > 0 else 0
        async_vs_enh = (enhanced - async_time) / enhanced * 100 if enhanced > 0 else 0
        
        logger.info("\n性能测试结果摘要:")
        logger.info(f"- 原始版本耗时: {original:.2f}秒")
        logger.info(f"- 第一阶段优化版本耗时: {enhanced:.2f}秒 (提升: {enh_improvement:.2f}%)")
        logger.info(f"- 第二阶段异步优化版本耗时: {async_time:.2f}秒 (提升: {async_improvement:.2f}%)")
        logger.info(f"- 第二阶段相比第一阶段提升: {async_vs_enh:.2f}%")
        
        # 检查缓存命中率
        if "async" in performance:
            cache_hit_ratio = performance["async"].get("cache_hit_ratio", 0)
            logger.info(f"- 缓存命中率: {cache_hit_ratio:.2f}%")

if __name__ == "__main__":
    main() 