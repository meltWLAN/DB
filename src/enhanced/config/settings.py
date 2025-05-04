#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版数据处理系统配置
包含路径配置、数据源配置和处理参数
"""

import os
from pathlib import Path
import multiprocessing

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
ENHANCED_CACHE_DIR = os.path.join(DATA_DIR, "enhanced_cache")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# 确保目录存在
for directory in [DATA_DIR, CACHE_DIR, ENHANCED_CACHE_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# 数据处理配置
PROCESSING_CONFIG = {
    "parallel": True,
    "num_workers": multiprocessing.cpu_count() - 1,  # 保留一个核心给系统
    "chunk_size": 5000,  # 并行处理时的块大小
    "use_vectorization": True,  # 启用向量化操作
    "cache_enabled": True,  # 启用缓存
    "cache_format": "parquet",  # 缓存格式: parquet, hdf5, csv
    "cache_compression": "snappy",  # parquet的压缩方式
    "verbose": True,  # 详细输出
}

# 数据质量配置
DATA_QUALITY_CONFIG = {
    "validation_enabled": True,  # 启用数据验证
    "auto_fix_enabled": True,  # 启用自动修复
    "validation_threshold": 0.95,  # 数据质量阈值，低于此值发出警告
}

# 数据源配置
DATA_SOURCE_CONFIG = {
    # Tushare配置
    "tushare": {
        "token": "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10",  # Tushare API token
        "enabled": True,
        "is_primary": True,
        "connection_retries": 3,
        "retry_delay": 2,
        "rate_limit": 3,  # 每秒请求数
        "default_start_date": "2020-01-01"
    },
    # AKShare配置
    "akshare": {
        "enabled": True,
        "is_primary": False,
        "rate_limit": {
            "calls_per_minute": 120,
            "pause_on_limit": True,
        },
        "retry": {
            "max_retries": 3,
            "retry_interval": 5,
        }
    },
    # JoinQuant配置(备用)
    "joinquant": {
        "username": "",
        "password": "",
        "enabled": False,
        "is_primary": False,
    },
}

# 缓存配置
CACHE_CONFIG = {
    "memory_cache_size": 128,  # 内存缓存大小(MB)
    "disk_cache_max_age": 24,  # 磁盘缓存最大存活时间(小时)
    "prefetch_enabled": True,  # 启用预取
    "prefetch_days": 2,        # 预取未来多少天的数据(如有)
}

# 增量更新配置
INCREMENTAL_UPDATE_CONFIG = {
    "enabled": True,
    "batch_size": 1000,        # 每批处理的记录数
    "timeout": 300,            # 单个批次的超时时间(秒)
}

# 健康检查配置
HEALTH_CHECK_CONFIG = {
    "enabled": True,
    "check_interval": 300,     # 健康检查间隔(秒)
    "retry_interval": 1800,    # 故障恢复检查间隔(秒)
}

# Tushare 数据源配置
TUSHARE_TOKEN = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
TUSHARE_API_URL = "http://api.tushare.pro"
TUSHARE_RATE_LIMIT = 3  # 每秒请求次数限制 