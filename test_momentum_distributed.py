#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动量分析分布式优化测试脚本
对比三个阶段的性能差异
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import psutil
import gc
import argparse
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入动量分析模块
try:
    from momentum_analysis_enhanced_performance import EnhancedMomentumAnalyzer
    from momentum_analysis_async import AsyncMomentumAnalyzer
    from momentum_analysis_distributed import DistributedMomentumAnalyzer
    modules_loaded = True
except ImportError as e:
    logger.error(f"导入动量分析模块失败: {str(e)}")
    modules_loaded = False

def run_comprehensive_test(sample_size=20, use_multiprocessing=True, use_distributed=True):
    """
    运行全面性能对比测试，比较三个阶段的优化效果
    
    Args:
        sample_size: 测试的股票数量
        use_multiprocessing: 是否使用多进程
        use_distributed: 是否使用分布式计算
    
    Returns:
        dict: 包含性能测试结果的字典
    """
    if not modules_loaded:
        return {"error": "无法导入必要的模块"}
    
    results = {
        "sample_size": sample_size,
        "use_multiprocessing": use_multiprocessing,
        "use_distributed": use_distributed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "performance": {},
        "memory_usage": {}
    }
    
    try:
        # 先获取股票列表备用
        logger.info("获取股票列表供测试使用")
        phase1_analyzer = EnhancedMomentumAnalyzer(use_tushare=True, use_multiprocessing=False)
        stocks = phase1_analyzer.get_stock_list()
        
        if stocks.empty or len(stocks) == 0:
            return {"error": "无法获取股票列表"}
        
        # 限制样本大小
        stocks = stocks.head(sample_size)
        
        # 运行垃圾回收，确保公平起点
        gc.collect()
        
        # 1. 测试第一阶段优化版本
        logger.info(f"测试第一阶段优化版本 (样本: {sample_size}, 多进程: {use_multiprocessing})")
        analyzer1 = EnhancedMomentumAnalyzer(use_tushare=True, use_multiprocessing=use_multiprocessing)
        
        # 记录开始时的内存使用
        process = psutil.Process(os.getpid())
        mem_before_1 = process.memory_info().rss / (1024 * 1024)
        
        # 预热缓存
        analyzer1.warm_up_cache(stocks)
        
        # 执行分析
        start_time = time.time()
        phase1_results = analyzer1.analyze_stocks(stocks, min_score=50)
        end_time = time.time()
        phase1_duration = end_time - start_time
        
        # 记录结束时的内存使用
        mem_after_1 = process.memory_info().rss / (1024 * 1024)
        mem_increase_1 = mem_after_1 - mem_before_1
        
        results["performance"]["phase1"] = {
            "name": "第一阶段优化",
            "duration": phase1_duration,
            "result_count": len(phase1_results),
            "per_stock_time": phase1_duration / sample_size
        }
        
        results["memory_usage"]["phase1"] = {
            "before": mem_before_1,
            "after": mem_after_1,
            "increase": mem_increase_1
        }
        
        logger.info(f"第一阶段优化版本耗时: {phase1_duration:.2f}秒, 结果数: {len(phase1_results)}")
        logger.info(f"内存增加: {mem_increase_1:.2f}MB")
        
        # 清理内存，准备下一个测试
        del analyzer1, phase1_results
        gc.collect()
        time.sleep(1)  # 短暂暂停，确保资源释放
        
        # 2. 测试第二阶段异步优化版本
        logger.info(f"测试第二阶段异步优化版本 (样本: {sample_size}, 多进程: {use_multiprocessing})")
        analyzer2 = AsyncMomentumAnalyzer(use_tushare=True, use_multiprocessing=use_multiprocessing)
        
        # 记录开始时的内存使用
        mem_before_2 = process.memory_info().rss / (1024 * 1024)
        
        # 执行分析 (异步版本已包含预热)
        start_time = time.time()
        phase2_results = analyzer2.analyze_stocks(stocks, min_score=50)
        end_time = time.time()
        phase2_duration = end_time - start_time
        
        # 记录结束时的内存使用
        mem_after_2 = process.memory_info().rss / (1024 * 1024)
        mem_increase_2 = mem_after_2 - mem_before_2
        
        results["performance"]["phase2"] = {
            "name": "第二阶段异步优化",
            "duration": phase2_duration,
            "result_count": len(phase2_results),
            "per_stock_time": phase2_duration / sample_size,
            "vs_phase1": (phase1_duration - phase2_duration) / phase1_duration * 100
        }
        
        results["memory_usage"]["phase2"] = {
            "before": mem_before_2,
            "after": mem_after_2,
            "increase": mem_increase_2
        }
        
        logger.info(f"第二阶段异步优化版本耗时: {phase2_duration:.2f}秒, 结果数: {len(phase2_results)}")
        logger.info(f"相比第一阶段提升: {results['performance']['phase2']['vs_phase1']:.2f}%")
        logger.info(f"内存增加: {mem_increase_2:.2f}MB")
        
        # 清理内存，准备下一个测试
        del analyzer2, phase2_results
        gc.collect()
        time.sleep(1)  # 短暂暂停，确保资源释放
        
        # 3. 测试第三阶段分布式优化版本
        logger.info(f"测试第三阶段分布式优化版本 (样本: {sample_size}, 分布式: {use_distributed})")
        analyzer3 = DistributedMomentumAnalyzer(
            use_tushare=True, 
            use_multiprocessing=use_multiprocessing,
            use_distributed=use_distributed,
            storage_format='parquet'
        )
        
        # 记录开始时的内存使用
        mem_before_3 = process.memory_info().rss / (1024 * 1024)
        
        # 执行分析 (分布式版本已包含预热)
        start_time = time.time()
        phase3_results = analyzer3.analyze_stocks(stocks, min_score=50)
        end_time = time.time()
        phase3_duration = end_time - start_time
        
        # 记录结束时的内存使用
        mem_after_3 = process.memory_info().rss / (1024 * 1024)
        mem_increase_3 = mem_after_3 - mem_before_3
        
        results["performance"]["phase3"] = {
            "name": "第三阶段分布式优化",
            "duration": phase3_duration,
            "result_count": len(phase3_results),
            "per_stock_time": phase3_duration / sample_size,
            "vs_phase1": (phase1_duration - phase3_duration) / phase1_duration * 100,
            "vs_phase2": (phase2_duration - phase3_duration) / phase2_duration * 100
        }
        
        results["memory_usage"]["phase3"] = {
            "before": mem_before_3,
            "after": mem_after_3,
            "increase": mem_increase_3
        }
        
        logger.info(f"第三阶段分布式优化版本耗时: {phase3_duration:.2f}秒, 结果数: {len(phase3_results)}")
        logger.info(f"相比第一阶段提升: {results['performance']['phase3']['vs_phase1']:.2f}%")
        logger.info(f"相比第二阶段提升: {results['performance']['phase3']['vs_phase2']:.2f}%")
        logger.info(f"内存增加: {mem_increase_3:.2f}MB")
        
        # 尝试关闭分布式客户端
        if hasattr(analyzer3, 'dask_client') and analyzer3.dask_client:
            try:
                analyzer3.dask_client.close()
            except:
                pass
        
        # 生成性能对比图表
        generate_performance_chart(results)
        generate_memory_chart(results)
        
        # 保存测试结果
        save_test_results(results)
        
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
        phases = []
        durations = []
        improvements = []
        per_stock_times = []
        
        # 按阶段顺序整理数据
        for phase in ["phase1", "phase2", "phase3"]:
            if phase in performance:
                phases.append(performance[phase]["name"])
                durations.append(performance[phase]["duration"])
                per_stock_times.append(performance[phase]["per_stock_time"])
                
                # 计算相对于第一阶段的提升百分比
                if phase == "phase1":
                    improvements.append(0)  # 第一阶段作为基准
                else:
                    imp = performance[phase].get("vs_phase1", 0)
                    improvements.append(imp)
        
        # 创建图表
        plt.figure(figsize=(18, 12))
        
        # 绘制执行时间对比
        plt.subplot(2, 2, 1)
        bars = plt.bar(phases, durations, color=['#3498db', '#2ecc71', '#9b59b6'])
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f"{height:.2f}秒", ha='center', va='bottom')
        
        plt.title('总执行时间对比')
        plt.ylabel('耗时(秒)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 绘制每支股票平均处理时间
        plt.subplot(2, 2, 2)
        bars = plt.bar(phases, per_stock_times, color=['#3498db', '#2ecc71', '#9b59b6'])
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f"{height:.2f}秒/股", ha='center', va='bottom')
        
        plt.title('每支股票平均处理时间')
        plt.ylabel('耗时(秒/股)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 绘制性能提升百分比
        plt.subplot(2, 2, 3)
        bars = plt.bar(phases, improvements, color=['#3498db', '#2ecc71', '#9b59b6'])
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f"{height:.2f}%", ha='center', va='bottom')
        
        plt.title('性能提升百分比(相比第一阶段)')
        plt.ylabel('提升百分比(%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 绘制结果统计
        plt.subplot(2, 2, 4)
        result_counts = [performance[p].get("result_count", 0) for p in ["phase1", "phase2", "phase3"] if p in performance]
        
        if len(phases) == len(result_counts):
            bars = plt.bar(phases, result_counts, color=['#3498db', '#2ecc71', '#9b59b6'])
            
            # 添加数据标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f"{int(height)}", ha='center', va='bottom')
            
            plt.title('筛选出的股票数量')
            plt.ylabel('数量')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加整体标题
        plt.suptitle(f"动量分析模块各阶段性能对比 (样本大小: {results['sample_size']})", fontsize=16)
        
        # 调整布局并保存
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        chart_path = os.path.join(output_dir, f"phase_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(chart_path)
        
        logger.info(f"性能对比图表已保存至: {chart_path}")
        
    except Exception as e:
        logger.error(f"生成性能图表失败: {str(e)}")
        import traceback
        traceback.print_exc()

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
        phases = []
        memory_before = []
        memory_after = []
        memory_increase = []
        
        # 按阶段顺序整理数据
        for phase in ["phase1", "phase2", "phase3"]:
            if phase in memory_usage:
                if phase == "phase1":
                    phases.append("第一阶段优化")
                elif phase == "phase2":
                    phases.append("第二阶段异步优化")
                else:
                    phases.append("第三阶段分布式优化")
                    
                memory_before.append(memory_usage[phase]["before"])
                memory_after.append(memory_usage[phase]["after"])
                memory_increase.append(memory_usage[phase]["increase"])
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 绘制内存使用对比
        plt.subplot(2, 1, 1)
        
        x = np.arange(len(phases))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, memory_before, width, label='处理前内存', color='#3498db')
        bars2 = plt.bar(x + width/2, memory_after, width, label='处理后内存', color='#e74c3c')
        
        plt.xlabel('优化阶段')
        plt.ylabel('内存使用 (MB)')
        plt.title('各阶段内存使用对比')
        plt.xticks(x, phases)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 为每个条添加数据标签
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f"{height:.1f}", ha='center', va='bottom')
                     
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f"{height:.1f}", ha='center', va='bottom')
        
        # 绘制内存增长对比
        plt.subplot(2, 1, 2)
        bars = plt.bar(phases, memory_increase, color=['#3498db', '#2ecc71', '#9b59b6'])
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f"{height:.1f} MB", ha='center', va='bottom')
        
        plt.title('各阶段内存增长对比')
        plt.ylabel('内存增长 (MB)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加整体标题
        plt.suptitle(f"动量分析模块各阶段内存使用对比 (样本大小: {results['sample_size']})", fontsize=16)
        
        # 调整布局并保存
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        chart_path = os.path.join(output_dir, f"memory_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(chart_path)
        
        logger.info(f"内存使用对比图表已保存至: {chart_path}")
        
    except Exception as e:
        logger.error(f"生成内存图表失败: {str(e)}")
        import traceback
        traceback.print_exc()

def save_test_results(results):
    """
    保存测试结果为JSON文件
    
    Args:
        results: 测试结果字典
    
    Returns:
        str: 保存的文件路径
    """
    try:
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 为结果添加一个唯一标识符
        results["id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为JSON文件
        file_path = os.path.join(output_dir, f"performance_test_{results['id']}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试结果已保存至: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"保存测试结果失败: {str(e)}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="动量分析模块全面性能测试")
    parser.add_argument('-s', '--sample', type=int, default=20, help="测试样本大小")
    parser.add_argument('-m', '--multiprocessing', action='store_true', help="使用多进程")
    parser.add_argument('-d', '--distributed', action='store_true', help="使用分布式计算")
    
    args = parser.parse_args()
    
    logger.info(f"开始全面性能测试: 样本={args.sample}, 多进程={args.multiprocessing}, 分布式={args.distributed}")
    results = run_comprehensive_test(args.sample, args.multiprocessing, args.distributed)
    
    # 显示结果摘要
    if "error" in results:
        logger.error(f"测试失败: {results['error']}")
    else:
        logger.info("全面性能测试完成")
        
        # 获取各阶段性能数据
        phase1 = results["performance"].get("phase1", {}).get("duration", 0)
        phase2 = results["performance"].get("phase2", {}).get("duration", 0)
        phase3 = results["performance"].get("phase3", {}).get("duration", 0)
        
        # 计算各阶段提升
        p1_to_p2 = (phase1 - phase2) / phase1 * 100 if phase1 > 0 else 0
        p2_to_p3 = (phase2 - phase3) / phase2 * 100 if phase2 > 0 else 0
        p1_to_p3 = (phase1 - phase3) / phase1 * 100 if phase1 > 0 else 0
        
        logger.info("\n性能测试结果摘要:")
        logger.info(f"- 第一阶段优化版本耗时: {phase1:.2f}秒")
        logger.info(f"- 第二阶段异步优化版本耗时: {phase2:.2f}秒 (vs 第一阶段: {p1_to_p2:.2f}%)")
        logger.info(f"- 第三阶段分布式优化版本耗时: {phase3:.2f}秒 (vs 第二阶段: {p2_to_p3:.2f}%, vs 第一阶段: {p1_to_p3:.2f}%)")
        
        # 内存使用摘要
        logger.info("\n内存使用摘要:")
        for phase in ["phase1", "phase2", "phase3"]:
            if phase in results["memory_usage"]:
                phase_name = results["performance"][phase]["name"]
                mem_increase = results["memory_usage"][phase]["increase"]
                logger.info(f"- {phase_name}内存增长: {mem_increase:.2f}MB")

if __name__ == "__main__":
    main() 