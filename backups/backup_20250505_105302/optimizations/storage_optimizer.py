#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据存储优化模块
提供高效的数据存储和压缩功能，减少磁盘使用并加快读取速度
"""

import os
import sys
import logging
import glob
import shutil
import pickle
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

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

# 定义常量
CACHE_DIR = "./cache"
OPTIMIZED_CACHE_DIR = "./optimized_cache"
COMPRESSION = "snappy"  # 使用snappy压缩算法，平衡压缩率和速度

def setup_optimized_storage():
    """
    设置优化的存储环境
    """
    # 创建优化后的存储目录
    os.makedirs(OPTIMIZED_CACHE_DIR, exist_ok=True)
    logger.info(f"优化存储环境设置完成，存储目录: {OPTIMIZED_CACHE_DIR}")
    return OPTIMIZED_CACHE_DIR

def convert_pickle_to_parquet(pickle_file, delete_original=False):
    """
    将pickle格式文件转换为parquet格式
    
    Args:
        pickle_file: pickle文件路径
        delete_original: 是否删除原始文件
    
    Returns:
        str: 新的parquet文件路径，如果转换失败则返回None
    """
    try:
        # 解析文件名
        basename = os.path.basename(pickle_file)
        root_name = os.path.splitext(basename)[0]
        output_file = os.path.join(OPTIMIZED_CACHE_DIR, f"{root_name}.parquet")
        
        # 检查目标文件是否已存在且更新
        if os.path.exists(output_file):
            output_mtime = os.path.getmtime(output_file)
            input_mtime = os.path.getmtime(pickle_file)
            if output_mtime >= input_mtime:
                logger.debug(f"文件 {output_file} 已是最新，跳过转换")
                return output_file
        
        # 加载pickle文件
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        # 根据数据类型进行转换
        if isinstance(data, pd.DataFrame):
            # 优化DataFrame类型
            for col in data.select_dtypes(include=['float']).columns:
                # 尝试降低精度
                data[col] = pd.to_numeric(data[col], downcast='float')
            
            for col in data.select_dtypes(include=['int']).columns:
                # 整数列降低为最小所需类型
                data[col] = pd.to_numeric(data[col], downcast='integer')
            
            # 保存为parquet格式
            data.to_parquet(output_file, compression=COMPRESSION, index=True)
            
            # 记录转换信息
            original_size = os.path.getsize(pickle_file)
            new_size = os.path.getsize(output_file)
            savings = (1 - new_size/original_size) * 100
            logger.info(f"转换 {pickle_file} 成功, 节省空间: {savings:.1f}%, {original_size/1024:.1f}KB -> {new_size/1024:.1f}KB")
            
            # 如果指定，删除原始文件
            if delete_original:
                os.remove(pickle_file)
                logger.debug(f"删除原始文件 {pickle_file}")
            
            return output_file
        else:
            # 非DataFrame类型，仍然使用pickle，但优化存储
            optimized_pickle = os.path.join(OPTIMIZED_CACHE_DIR, basename)
            with open(optimized_pickle, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"非DataFrame数据，使用优化的pickle格式: {pickle_file} -> {optimized_pickle}")
            
            if delete_original and pickle_file != optimized_pickle:
                os.remove(pickle_file)
            
            return optimized_pickle
    except Exception as e:
        logger.error(f"转换文件 {pickle_file} 失败: {str(e)}")
        return None

def optimize_all_cache_files():
    """
    优化所有缓存文件
    
    Returns:
        int: 成功优化的文件数量
    """
    setup_optimized_storage()
    
    cache_files = glob.glob(os.path.join(CACHE_DIR, "*.pkl"))
    logger.info(f"找到 {len(cache_files)} 个缓存文件需要优化")
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    total_space_saved = 0
    
    start_time = time.time()
    
    for file in cache_files:
        try:
            result = convert_pickle_to_parquet(file, delete_original=False)
            if result:
                success_count += 1
                # 计算节省空间
                old_size = os.path.getsize(file)
                new_size = os.path.getsize(result)
                total_space_saved += (old_size - new_size)
            else:
                failed_count += 1
        except Exception as e:
            logger.error(f"处理文件 {file} 时出错: {str(e)}")
            failed_count += 1
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # 转换缓存元数据文件
    try:
        meta_files = glob.glob(os.path.join(CACHE_DIR, "*.meta"))
        for meta_file in meta_files:
            target_meta = os.path.join(OPTIMIZED_CACHE_DIR, os.path.basename(meta_file))
            shutil.copy2(meta_file, target_meta)
            skipped_count += 1
    except Exception as e:
        logger.error(f"复制元数据文件时出错: {str(e)}")
    
    total_mb_saved = total_space_saved / (1024 * 1024)
    logger.info(f"缓存优化完成: 成功={success_count}, 失败={failed_count}, 跳过={skipped_count}, 耗时={elapsed:.2f}秒")
    logger.info(f"总共节省空间: {total_mb_saved:.2f}MB")
    
    return success_count

def read_optimized_cache(cache_key):
    """
    读取优化后的缓存
    
    Args:
        cache_key: 缓存键
    
    Returns:
        object: 缓存的数据，如果不存在则返回None
    """
    # 首先尝试读取优化版本
    parquet_file = os.path.join(OPTIMIZED_CACHE_DIR, f"{cache_key}.parquet")
    if os.path.exists(parquet_file):
        try:
            return pd.read_parquet(parquet_file)
        except Exception as e:
            logger.warning(f"从优化缓存读取 {cache_key} 失败: {str(e)}")
    
    # 尝试读取pickle版本
    pickle_file = os.path.join(OPTIMIZED_CACHE_DIR, f"{cache_key}.pkl")
    if os.path.exists(pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"从优化pickle缓存读取 {cache_key} 失败: {str(e)}")
    
    # 返回到原始缓存
    original_pickle = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    if os.path.exists(original_pickle):
        try:
            with open(original_pickle, 'rb') as f:
                data = pickle.load(f)
                # 自动转换并存储为优化格式以备将来使用
                if isinstance(data, pd.DataFrame):
                    convert_pickle_to_parquet(original_pickle)
                return data
        except Exception as e:
            logger.warning(f"从原始缓存读取 {cache_key} 失败: {str(e)}")
    
    return None

def write_optimized_cache(cache_key, data):
    """
    写入优化后的缓存
    
    Args:
        cache_key: 缓存键
        data: 要缓存的数据
    
    Returns:
        bool: 是否成功写入
    """
    try:
        os.makedirs(OPTIMIZED_CACHE_DIR, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            # DataFrame写入为parquet
            parquet_file = os.path.join(OPTIMIZED_CACHE_DIR, f"{cache_key}.parquet")
            data.to_parquet(parquet_file, compression=COMPRESSION, index=True)
            logger.debug(f"写入优化缓存 {cache_key} 成功 (parquet)")
            return True
        else:
            # 其他类型写入为pickle
            pickle_file = os.path.join(OPTIMIZED_CACHE_DIR, f"{cache_key}.pkl")
            with open(pickle_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"写入优化缓存 {cache_key} 成功 (pickle)")
            return True
    except Exception as e:
        logger.error(f"写入优化缓存 {cache_key} 失败: {str(e)}")
        return False

# 导出主要函数
__all__ = [
    'setup_optimized_storage',
    'optimize_all_cache_files',
    'read_optimized_cache',
    'write_optimized_cache'
]

if __name__ == "__main__":
    # 执行优化
    print("开始优化数据存储...")
    optimize_all_cache_files()
    print("数据存储优化完成！") 