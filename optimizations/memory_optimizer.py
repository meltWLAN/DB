#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
内存优化模块
提供DataFrame内存优化和智能内存管理功能
"""

import os
import sys
import logging
import gc
import psutil
import pandas as pd
import numpy as np

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

def optimize_dataframe(df, inplace=False, category_threshold=50):
    """
    优化DataFrame内存使用
    
    Args:
        df: 需要优化的DataFrame
        inplace: 是否直接修改原DataFrame
        category_threshold: 唯一值数量阈值，低于此值的string类型列将转换为category类型
    
    Returns:
        DataFrame: 优化后的DataFrame
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    
    if not inplace:
        df = df.copy()
    
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.debug(f"DataFrame原始内存使用: {start_mem:.2f} MB")
    
    # 优化数值列
    for col in df.select_dtypes(include=['int']).columns:
        # 先检查列是否包含NA值
        if df[col].isna().any():
            # 如果有NA值，转换为浮点类型的downcast
            df[col] = pd.to_numeric(df[col], downcast='float')
        else:
            # 整数列降低为最小所需类型
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # 优化浮点列
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # 优化文本列 - 转换为category类型
    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        
        # 如果唯一值数量小于阈值且低于总数的50%，转换为category
        if num_unique_values < category_threshold and num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype('category')
    
    # 对日期列使用专用类型
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception as e:
                    logger.debug(f"无法将列 {col} 转换为datetime类型: {str(e)}")
    
    # 计算优化后的内存使用
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    logger.debug(f"DataFrame优化后内存使用: {end_mem:.2f} MB, 减少: {reduction:.1f}%")
    
    return df

def optimize_in_chunks(df, chunk_size=100000, **kwargs):
    """
    分块优化大型DataFrame
    
    Args:
        df: 需要优化的DataFrame
        chunk_size: 每块的大小
        **kwargs: 传递给optimize_dataframe的参数
    
    Returns:
        DataFrame: 优化后的DataFrame
    """
    if len(df) <= chunk_size:
        return optimize_dataframe(df, **kwargs)
    
    # 计算块数
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    logger.info(f"将DataFrame分为{num_chunks}块进行优化")
    
    # 分块优化并连接结果
    result = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx]
        
        # 优化当前块
        optimized_chunk = optimize_dataframe(chunk, **kwargs)
        result.append(optimized_chunk)
        
        # 手动触发垃圾回收
        del chunk
        gc.collect()
    
    # 合并所有块
    return pd.concat(result, ignore_index=df.index.name is None)

def get_memory_usage():
    """
    获取当前程序的内存使用情况
    
    Returns:
        dict: 内存使用信息
    """
    # 获取系统内存信息
    sys_mem = psutil.virtual_memory()
    
    # 获取当前进程内存使用
    process = psutil.Process(os.getpid())
    process_mem = process.memory_info().rss / (1024 * 1024)  # MB
    
    return {
        'system_total': sys_mem.total / (1024 * 1024),  # MB
        'system_used': (sys_mem.total - sys_mem.available) / (1024 * 1024),  # MB
        'system_percent': sys_mem.percent,
        'process_used_mb': process_mem,
        'process_percent': (process_mem * 1024 * 1024 / sys_mem.total) * 100
    }

def cleanup_memory(threshold_percent=80, force=False):
    """
    清理内存，当系统内存使用超过阈值时触发
    
    Args:
        threshold_percent: 触发清理的内存使用百分比阈值
        force: 是否强制清理，不考虑阈值
    
    Returns:
        dict: 清理前后的内存使用信息
    """
    # 获取清理前的内存使用
    before = get_memory_usage()
    
    # 检查是否需要清理
    if not force and before['system_percent'] < threshold_percent:
        logger.debug(f"内存使用 ({before['system_percent']:.1f}%) 低于阈值 ({threshold_percent}%), 跳过清理")
        return {'before': before, 'after': before, 'cleanup_performed': False}
    
    logger.info(f"开始清理内存, 当前使用率: {before['system_percent']:.1f}%")
    
    # 执行清理操作
    # 1. 调用Python垃圾回收器
    collected = gc.collect(generation=2)
    
    # 2. 清空不再使用的缓存
    # 这里可以添加特定于应用程序的缓存清理代码
    
    # 获取清理后的内存使用
    after = get_memory_usage()
    
    # 计算节省的内存
    saved_mb = before['process_used_mb'] - after['process_used_mb']
    
    if saved_mb > 0:
        logger.info(f"内存清理完成, 释放了 {saved_mb:.2f} MB, 当前使用率: {after['system_percent']:.1f}%")
    else:
        logger.info(f"内存清理完成, 但没有释放内存, 当前使用率: {after['system_percent']:.1f}%")
    
    return {
        'before': before,
        'after': after,
        'cleanup_performed': True,
        'objects_collected': collected,
        'memory_saved_mb': saved_mb
    }

def get_dataframe_info(df):
    """
    获取DataFrame的详细信息，包括内存使用和优化建议
    
    Args:
        df: 要分析的DataFrame
    
    Returns:
        dict: DataFrame信息
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return {"error": "输入不是有效的DataFrame"}
    
    # 基本信息
    info = {
        "shape": df.shape,
        "columns": len(df.columns),
        "rows": len(df),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "dtypes": {},
        "null_counts": {},
        "optimization_suggestions": []
    }
    
    # 分析数据类型
    for col in df.columns:
        info["dtypes"][col] = str(df[col].dtype)
        info["null_counts"][col] = df[col].isna().sum()
        
        # 生成优化建议
        if df[col].dtype == 'int64' and df[col].max() < 2147483647:
            info["optimization_suggestions"].append(f"列 '{col}' 可以从int64降级为int32")
        elif df[col].dtype == 'float64':
            info["optimization_suggestions"].append(f"列 '{col}' 可以从float64降级为float32")
        elif df[col].dtype == 'object':
            n_unique = df[col].nunique()
            if n_unique < len(df) * 0.5 and n_unique < 100:
                info["optimization_suggestions"].append(f"列 '{col}' 可以转换为category类型，有{n_unique}个唯一值")
    
    return info

# 导出主要函数
__all__ = [
    'optimize_dataframe',
    'optimize_in_chunks',
    'get_memory_usage',
    'cleanup_memory',
    'get_dataframe_info'
]

if __name__ == "__main__":
    # 测试内存优化功能
    print("内存使用情况:")
    mem_info = get_memory_usage()
    for key, value in mem_info.items():
        if 'percent' in key:
            print(f"  {key}: {value:.1f}%")
        else:
            print(f"  {key}: {value:.1f} MB")
    
    # 执行内存清理
    print("\n执行内存清理:")
    cleanup_result = cleanup_memory(force=True)
    saved = cleanup_result.get('memory_saved_mb', 0)
    print(f"释放了 {saved:.2f} MB 内存") 