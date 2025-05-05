#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动量分析模块超级优化配置
提供优化后的参数配置，立即提升性能
"""

import os
import sys
import logging
import multiprocessing
import psutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 确保当前目录在Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def get_optimal_workers():
    """
    根据系统情况确定最优的工作进程数
    """
    # 获取CPU核心数
    cpu_cores = multiprocessing.cpu_count()
    
    # 获取系统内存情况
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024 ** 3)  # 转换为GB
    
    # 基于CPU和内存计算最优工作进程数
    # 每个进程大约需要0.5GB内存(根据实际情况调整)
    memory_based = max(1, int(memory_gb / 0.5) - 2)  # 预留更多内存给系统
    cpu_based = max(1, cpu_cores - 1)  # 预留1个CPU核心给系统
    
    # 取较小值，避免资源耗尽
    workers = min(memory_based, cpu_based)
    logger.info(f"系统资源: CPU核心={cpu_cores}, 内存={memory_gb:.1f}GB, 最优工作进程数={workers}")
    
    return workers

def create_hyper_optimized_analyzer():
    """
    创建超级优化版本的动量分析器
    
    Returns:
        EnhancedMomentumAnalyzer: 优化后的动量分析器实例
    """
    try:
        from momentum_analysis_enhanced_performance import EnhancedMomentumAnalyzer
        
        # 获取最优工作进程数
        workers = get_optimal_workers()
        
        # 计算最优批处理大小 - 根据系统内存动态调整
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        
        # 根据可用内存调整批处理大小和缓存大小
        if available_gb > 8:  # 大内存系统
            batch_size = 200
            cache_size = 512
        elif available_gb > 4:  # 中等内存系统
            batch_size = 100
            cache_size = 256
        else:  # 小内存系统
            batch_size = 50
            cache_size = 128
        
        # 创建优化后的动量分析器
        analyzer = EnhancedMomentumAnalyzer(
            use_tushare=True,
            use_multiprocessing=True,  
            workers=workers,  # 动态计算最优工作进程数
            cache_size=cache_size,  # 根据内存动态调整缓存大小
            cache_timeout=43200,  # 12小时缓存过期，保持数据新鲜度
            batch_size=batch_size,  # 动态调整批处理大小
            memory_limit=0.7  # 内存使用限制，避免系统内存压力
        )
        
        logger.info(f"创建超级优化版动量分析器成功: 工作进程数={workers}, 缓存大小={cache_size}, 批处理大小={batch_size}")
        return analyzer
    except ImportError as e:
        logger.error(f"导入EnhancedMomentumAnalyzer失败: {e}")
        logger.info("尝试导入原始动量分析器...")
        
        try:
            from momentum_analysis import MomentumAnalyzer
            analyzer = MomentumAnalyzer(
                use_tushare=True,
                use_multiprocessing=True
            )
            logger.info("使用原始动量分析器 (未优化)")
            return analyzer
        except ImportError:
            logger.error("导入动量分析器失败，无法创建分析器")
            return None

# 为简化使用提供一个便捷函数
get_optimized_analyzer = create_hyper_optimized_analyzer 