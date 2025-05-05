"""
性能优化模块
提供多项性能优化功能，用于提升系统运行效率
"""

# 导出主要功能函数
from .hyper_optimize import get_optimized_analyzer

# 导出数据存储优化功能
from .storage_optimizer import (
    setup_optimized_storage,
    optimize_all_cache_files,
    read_optimized_cache,
    write_optimized_cache
)

# 导出内存优化功能
from .memory_optimizer import (
    optimize_dataframe,
    optimize_in_chunks,
    get_memory_usage,
    cleanup_memory,
    get_dataframe_info
)

# 导出异步预加载功能
from .async_prefetch import (
    get_prefetcher,
    prefetch_data,
    get_prefetched_data,
    prefetch_decorator,
    get_prefetch_stats,
    clear_prefetch_cache
) 