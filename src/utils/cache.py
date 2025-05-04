#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强缓存模块

提供多级缓存系统，支持内存缓存、磁盘缓存，带有智能过期策略
"""

import os
import time
import pickle
import hashlib
import logging
import threading
import pandas as pd
import numpy as np
import zlib
import lzma
import json
from enum import Enum
from typing import Dict, Any, Optional, Union, Callable, Tuple, List, Set
from datetime import datetime, timedelta
from functools import wraps
from collections import Counter

# 设置日志
logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """缓存级别枚举"""
    MEMORY = "memory"      # 仅内存缓存
    DISK = "disk"          # 仅磁盘缓存
    BOTH = "both"          # 内存+磁盘缓存
    NONE = "none"          # 不使用缓存

class CompressionMethod(Enum):
    """压缩方法枚举"""
    NONE = "none"          # 不压缩
    ZLIB = "zlib"          # zlib压缩
    LZMA = "lzma"          # lzma压缩 (更高压缩比)

class EvictionPolicy(Enum):
    """缓存淘汰策略枚举"""
    LRU = "lru"            # 最近最少使用
    LFU = "lfu"            # 最不经常使用
    FIFO = "fifo"          # 先进先出
    TTL = "ttl"            # 仅基于过期时间
    SMART = "smart"        # 智能策略（基于频率和时间的综合加权）

class AccessPattern(Enum):
    """访问模式枚举"""
    RANDOM = "random"      # 随机访问
    SEQUENTIAL = "sequential"  # 顺序访问
    TEMPORAL = "temporal"  # 时间局部性
    UNKNOWN = "unknown"    # 未知模式

class CacheStats:
    """缓存统计类"""
    
    def __init__(self):
        """初始化缓存统计"""
        self.hit_count = {"memory": 0, "disk": 0}
        self.miss_count = {"memory": 0, "disk": 0}
        self.write_count = {"memory": 0, "disk": 0}
        self.eviction_count = {"memory": 0, "disk": 0}
        self.byte_saved = {"memory": 0, "disk": 0}
        self.access_patterns = Counter()
        self.access_history = []  # 最近的访问记录
        self.last_analysis = 0  # 上次分析时间戳
        
    def record_hit(self, level: str, size: int = 0):
        """记录缓存命中"""
        self.hit_count[level] += 1
        self.byte_saved[level] += size
        
    def record_miss(self, level: str):
        """记录缓存未命中"""
        self.miss_count[level] += 1
        
    def record_write(self, level: str, size: int = 0):
        """记录缓存写入"""
        self.write_count[level] += 1
        
    def record_eviction(self, level: str, count: int = 1):
        """记录缓存淘汰"""
        self.eviction_count[level] += count
        
    def record_access(self, key: str, hit: bool = True):
        """记录访问模式"""
        timestamp = time.time()
        self.access_history.append((timestamp, key, hit))
        
        # 保持历史记录在合理大小，只保留最近1000条
        if len(self.access_history) > 1000:
            self.access_history = self.access_history[-1000:]
            
        # 每100次访问分析一次模式
        if len(self.access_history) % 100 == 0 and timestamp - self.last_analysis > 60:
            self._analyze_access_pattern()
            self.last_analysis = timestamp
            
    def _analyze_access_pattern(self):
        """分析访问模式"""
        if len(self.access_history) < 50:
            return
            
        # 提取最近的键
        recent_keys = [item[1] for item in self.access_history[-50:]]
        
        # 检查是否为顺序访问
        sequential_count = 0
        for i in range(1, len(recent_keys)):
            if recent_keys[i] == recent_keys[i-1] + 1 or recent_keys[i] == recent_keys[i-1] - 1:
                sequential_count += 1
                
        # 检查是否为时间局部性访问
        unique_keys = set(recent_keys)
        repeat_ratio = 1 - len(unique_keys) / len(recent_keys)
        
        # 确定访问模式
        if sequential_count > len(recent_keys) * 0.7:
            self.access_patterns[AccessPattern.SEQUENTIAL.value] += 1
        elif repeat_ratio > 0.5:
            self.access_patterns[AccessPattern.TEMPORAL.value] += 1
        else:
            self.access_patterns[AccessPattern.RANDOM.value] += 1
            
    def get_dominant_pattern(self) -> AccessPattern:
        """获取主要访问模式"""
        if not self.access_patterns:
            return AccessPattern.UNKNOWN
            
        # 返回出现次数最多的模式
        return AccessPattern(max(self.access_patterns.items(), key=lambda x: x[1])[0])
        
    def get_hit_ratio(self, level: str = "memory") -> float:
        """获取缓存命中率"""
        total = self.hit_count[level] + self.miss_count[level]
        if total == 0:
            return 0.0
        return self.hit_count[level] / total
        
    def get_stats_dict(self) -> Dict:
        """获取统计信息字典"""
        stats = {
            "hits": dict(self.hit_count),
            "misses": dict(self.miss_count),
            "writes": dict(self.write_count),
            "evictions": dict(self.eviction_count),
            "bytes_saved": dict(self.byte_saved),
            "hit_ratio": {
                "memory": self.get_hit_ratio("memory"),
                "disk": self.get_hit_ratio("disk")
            },
            "access_patterns": dict(self.access_patterns),
            "dominant_pattern": self.get_dominant_pattern().value
        }
        return stats
        
    def reset(self):
        """重置统计信息"""
        self.__init__()

class MemoryCacheManager:
    """内存缓存管理器"""
    
    def __init__(self, max_size: int, 
                default_ttl: int = 3600,
                eviction_policy: EvictionPolicy = EvictionPolicy.SMART,
                compression: CompressionMethod = CompressionMethod.NONE,
                compression_threshold: int = 10240):  # 默认10KB以上对象才压缩
        """初始化内存缓存管理器
        
        Args:
            max_size: 最大内存大小(字节)
            default_ttl: 默认过期时间(秒)
            eviction_policy: 淘汰策略
            compression: 压缩方法
            compression_threshold: 压缩阈值(字节)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self.compression = compression
        self.compression_threshold = compression_threshold
        
        self.cache = {}             # 缓存数据 {key: value}
        self.metadata = {}          # 缓存元数据 {key: metadata}
        self.current_size = 0       # 当前缓存大小(字节)
        
        # 频率计数器(用于LFU)
        self.frequency = {}         # {key: access_count}
        
        # 插入顺序(用于FIFO)
        self.insert_order = []      # [(timestamp, key)]
        
        # 线程安全锁
        self.lock = threading.RLock()
        
        # 热点键检测
        self.hotspot_detection = True
        self.hotspot_threshold = 100  # 单位时间内访问次数阈值
        self.hotspot_time_window = 60  # 时间窗口(秒)
        self.access_tracker = {}    # {key: [(timestamp, count)]}
        
    def get(self, key: str) -> Tuple[bool, Any]:
        """从内存缓存获取值
        
        Args:
            key: 缓存键
            
        Returns:
            Tuple[bool, Any]: (是否命中, 缓存值或None)
        """
        with self.lock:
            if key not in self.cache:
                return False, None
                
            # 检查是否过期
            if self._is_expired(key):
                self._remove(key)
                return False, None
                
            # 更新访问统计
            self._update_access_stats(key)
            
            # 获取值
            value = self.cache[key]
            
            # 解压缩(如果需要)
            if self.metadata[key].get("compressed", False):
                value = self._decompress(value, self.metadata[key]["compression_method"])
                
            return True, value
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
           skip_compression: bool = False) -> bool:
        """设置内存缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间(秒)，None使用默认值
            skip_compression: 是否跳过压缩
            
        Returns:
            bool: 是否成功设置
        """
        with self.lock:
            # 计算对象大小
            obj_size = self._estimate_size(value)
            
            # 如果对象超过缓存总大小的一半，不缓存
            if obj_size > self.max_size / 2:
                logger.debug(f"对象过大 ({obj_size/1024/1024:.1f}MB)，不加入内存缓存")
                return False
                
            # 如果键已存在，先删除旧值
            if key in self.cache:
                self._remove(key)
                
            # 确保有足够空间
            self._ensure_space(obj_size)
            
            # 尝试压缩大对象
            compressed = False
            compression_method = CompressionMethod.NONE
            compressed_value = value
            
            if (not skip_compression and 
                self.compression != CompressionMethod.NONE and 
                obj_size >= self.compression_threshold):
                try:
                    compressed_value, compression_ratio = self._compress(value, self.compression)
                    # 只有当压缩比超过25%才使用压缩值
                    if compression_ratio > 0.25:
                        compressed = True
                        compression_method = self.compression
                        # 更新对象大小为压缩后的大小
                        obj_size = self._estimate_size(compressed_value)
                except Exception as e:
                    logger.debug(f"压缩对象失败: {str(e)}")
            
            # 设置过期时间
            ttl = ttl if ttl is not None else self.default_ttl
            expires_at = time.time() + ttl
            
            # 保存到缓存
            self.cache[key] = compressed_value
            
            # 更新元数据
            self.metadata[key] = {
                "size": obj_size,
                "created_at": time.time(),
                "expires_at": expires_at,
                "last_access": time.time(),
                "access_count": 0,
                "compressed": compressed,
                "compression_method": compression_method.value if compressed else None
            }
            
            # 更新缓存大小
            self.current_size += obj_size
            
            # 更新频率计数和插入顺序
            self.frequency[key] = 0
            self.insert_order.append((time.time(), key))
            
            return True
            
    def remove(self, key: str) -> bool:
        """从内存缓存删除键
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否成功删除
        """
        with self.lock:
            return self._remove(key)
            
    def clear(self) -> None:
        """清空内存缓存"""
        with self.lock:
            self.cache.clear()
            self.metadata.clear()
            self.frequency.clear()
            self.insert_order.clear()
            self.access_tracker.clear()
            self.current_size = 0
            
    def get_size(self) -> int:
        """获取当前缓存大小(字节)"""
        with self.lock:
            return self.current_size
            
    def get_keys(self) -> List[str]:
        """获取所有缓存键"""
        with self.lock:
            return list(self.cache.keys())
            
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            # 计算过期项目数
            now = time.time()
            expired_count = sum(1 for meta in self.metadata.values() 
                               if meta["expires_at"] <= now)
            
            # 计算压缩项目数和节省空间
            compressed_count = sum(1 for meta in self.metadata.values() 
                                 if meta.get("compressed", False))
            
            # 平均访问次数和标准差
            access_counts = [meta["access_count"] for meta in self.metadata.values()]
            avg_access = sum(access_counts) / max(1, len(access_counts))
            std_access = (sum((x - avg_access) ** 2 for x in access_counts) / max(1, len(access_counts))) ** 0.5
            
            # 热点键
            hotspots = self._detect_hotspots()
            
            return {
                "current_size": self.current_size,
                "max_size": self.max_size,
                "usage_ratio": self.current_size / max(1, self.max_size),
                "items_count": len(self.cache),
                "expired_count": expired_count,
                "compressed_count": compressed_count,
                "avg_access_count": avg_access,
                "std_access_count": std_access,
                "hotspot_keys": hotspots,
                "eviction_policy": self.eviction_policy.value,
                "compression": self.compression.value
            }
    
    # 内部方法
    
    def _remove(self, key: str) -> bool:
        """内部方法：删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否成功删除
        """
        if key not in self.cache:
            return False
            
        # 减少缓存大小
        self.current_size -= self.metadata[key]["size"]
        
        # 删除缓存项和元数据
        del self.cache[key]
        del self.metadata[key]
        
        # 更新频率计数和插入顺序
        if key in self.frequency:
            del self.frequency[key]
            
        # 从插入顺序列表中移除
        self.insert_order = [(ts, k) for ts, k in self.insert_order if k != key]
        
        # 从访问追踪中移除
        if key in self.access_tracker:
            del self.access_tracker[key]
            
        return True
        
    def _ensure_space(self, needed_size: int) -> None:
        """确保有足够空间存储新对象
        
        Args:
            needed_size: 需要的空间大小(字节)
        """
        # 如果有足够空间，直接返回
        if self.current_size + needed_size <= self.max_size:
            return
            
        # 首先清理已过期的项目
        self._cleanup_expired()
        
        # 如果仍然空间不足，根据淘汰策略删除项目
        while self.current_size + needed_size > self.max_size and self.cache:
            key_to_evict = self._select_eviction_candidate()
            if key_to_evict:
                self._remove(key_to_evict)
            else:
                break
                
    def _cleanup_expired(self) -> int:
        """清理过期的缓存项
        
        Returns:
            int: 清理的项目数
        """
        now = time.time()
        expired_keys = [key for key, meta in self.metadata.items() 
                      if meta["expires_at"] <= now]
                      
        for key in expired_keys:
            self._remove(key)
            
        return len(expired_keys)
        
    def _select_eviction_candidate(self) -> Optional[str]:
        """根据淘汰策略选择要淘汰的缓存项
        
        Returns:
            Optional[str]: 要淘汰的键或None
        """
        if not self.cache:
            return None
            
        if self.eviction_policy == EvictionPolicy.LRU:
            # 最近最少使用：选择最久未访问的项目
            return min(self.metadata.items(), key=lambda x: x[1]["last_access"])[0]
            
        elif self.eviction_policy == EvictionPolicy.LFU:
            # 最不经常使用：选择访问次数最少的项目
            return min(self.frequency.items(), key=lambda x: x[1])[0]
            
        elif self.eviction_policy == EvictionPolicy.FIFO:
            # 先进先出：选择最早插入的项目
            if self.insert_order:
                return self.insert_order[0][1]
            return list(self.cache.keys())[0]
            
        elif self.eviction_policy == EvictionPolicy.TTL:
            # 基于过期时间：选择最快过期的项目
            return min(self.metadata.items(), key=lambda x: x[1]["expires_at"])[0]
            
        elif self.eviction_policy == EvictionPolicy.SMART:
            # 智能策略：综合考虑访问频率、时间和大小
            scores = {}
            now = time.time()
            
            for key, meta in self.metadata.items():
                # 访问频率因子(0-1)，越小越可能被淘汰
                freq_factor = min(1.0, self.frequency.get(key, 0) / 100.0)
                
                # 时间因子(0-1)，越小越可能被淘汰
                time_factor = min(1.0, (now - meta["last_access"]) / 3600.0)
                
                # 大小因子(0-1)，越大越可能被淘汰
                size_factor = min(1.0, meta["size"] / (self.max_size / 10.0))
                
                # TTL因子(0-1)，越小越可能被淘汰
                ttl_factor = min(1.0, (meta["expires_at"] - now) / self.default_ttl)
                
                # 综合得分，越低越可能被淘汰
                scores[key] = (0.4 * (1 - freq_factor) + 
                              0.3 * time_factor + 
                              0.2 * size_factor + 
                              0.1 * (1 - ttl_factor))
            
            # 返回得分最高的键
            return max(scores.items(), key=lambda x: x[1])[0]
            
        # 默认使用LRU
        return min(self.metadata.items(), key=lambda x: x[1]["last_access"])[0]
        
    def _is_expired(self, key: str) -> bool:
        """检查缓存项是否过期
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否已过期
        """
        if key not in self.metadata:
            return True
            
        return time.time() > self.metadata[key]["expires_at"]
        
    def _update_access_stats(self, key: str) -> None:
        """更新访问统计信息
        
        Args:
            key: 缓存键
        """
        now = time.time()
        
        # 更新最后访问时间和访问计数
        if key in self.metadata:
            self.metadata[key]["last_access"] = now
            self.metadata[key]["access_count"] += 1
            
        # 更新频率计数
        if key in self.frequency:
            self.frequency[key] += 1
            
        # 更新热点访问追踪
        if self.hotspot_detection:
            if key not in self.access_tracker:
                self.access_tracker[key] = []
                
            self.access_tracker[key].append(now)
            
            # 只保留时间窗口内的访问记录
            window_start = now - self.hotspot_time_window
            self.access_tracker[key] = [t for t in self.access_tracker[key] if t >= window_start]
            
    def _detect_hotspots(self) -> List[str]:
        """检测热点键
        
        Returns:
            List[str]: 热点键列表
        """
        if not self.hotspot_detection:
            return []
            
        now = time.time()
        window_start = now - self.hotspot_time_window
        
        hotspots = []
        for key, timestamps in self.access_tracker.items():
            # 计算时间窗口内的访问次数
            recent_count = sum(1 for t in timestamps if t >= window_start)
            
            if recent_count >= self.hotspot_threshold:
                hotspots.append(key)
                
        return hotspots
        
    def _estimate_size(self, obj: Any) -> int:
        """估算对象大小(字节)
        
        Args:
            obj: 要估算大小的对象
            
        Returns:
            int: 对象大小(字节)
        """
        # 对于numpy和pandas对象有专门的方法
        if isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
            return obj.nbytes
            
        # 其他类型简单估计
        try:
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            # 返回一个合理的默认值
            return 1024  # 1KB
            
    def _compress(self, value: Any, method: CompressionMethod) -> Tuple[bytes, float]:
        """压缩对象
        
        Args:
            value: 要压缩的对象
            method: 压缩方法
            
        Returns:
            Tuple[bytes, float]: (压缩后的数据, 压缩比)
        """
        try:
            # 序列化对象
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            original_size = len(data)
            
            # 根据方法压缩数据
            if method == CompressionMethod.ZLIB:
                compressed = zlib.compress(data, level=6)
            elif method == CompressionMethod.LZMA:
                compressed = lzma.compress(data, preset=6)
            else:
                return data, 0.0
                
            # 计算压缩比
            compressed_size = len(compressed)
            compression_ratio = 1.0 - (compressed_size / original_size)
            
            return compressed, compression_ratio
        except Exception as e:
            logger.error(f"压缩对象失败: {str(e)}")
            return pickle.dumps(value), 0.0
            
    def _decompress(self, value: bytes, method: str) -> Any:
        """解压对象
        
        Args:
            value: 压缩后的数据
            method: 压缩方法
            
        Returns:
            Any: 解压后的对象
        """
        try:
            # 根据方法解压数据
            if method == CompressionMethod.ZLIB.value:
                decompressed = zlib.decompress(value)
            elif method == CompressionMethod.LZMA.value:
                decompressed = lzma.decompress(value)
            else:
                decompressed = value
                
            # 反序列化对象
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"解压对象失败: {str(e)}")
            # 尝试直接反序列化
            try:
                return pickle.loads(value)
            except:
                return None

class CacheManager:
    """
    增强版缓存管理器
    
    支持内存和磁盘两级缓存，具有自动清理和过期策略
    """
    
    def __init__(self, cache_dir: str = "./cache", 
                 memory_limit_mb: int = 256,
                 disk_limit_mb: int = 1024,
                 default_memory_ttl: int = 3600,  # 1小时
                 default_disk_ttl: int = 86400,   # 1天
                 auto_cleanup_interval: int = 600):  # 10分钟清理一次
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 磁盘缓存目录
            memory_limit_mb: 内存缓存限制(MB)
            disk_limit_mb: 磁盘缓存限制(MB)
            default_memory_ttl: 默认内存缓存TTL(秒)
            default_disk_ttl: 默认磁盘缓存TTL(秒) 
            auto_cleanup_interval: 自动清理间隔(秒)
        """
        # 缓存配置
        self.cache_dir = cache_dir
        self.memory_limit = memory_limit_mb * 1024 * 1024  # 转换为字节
        self.disk_limit = disk_limit_mb * 1024 * 1024      # 转换为字节
        self.memory_ttl = default_memory_ttl
        self.disk_ttl = default_disk_ttl
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 内存缓存
        self.memory_cache = {}             # 缓存数据
        self.memory_metadata = {}          # 缓存元数据
        self.memory_size = 0               # 当前内存缓存大小
        
        # 线程安全锁
        self.memory_lock = threading.RLock()
        self.disk_lock = threading.RLock()
        
        # 缓存统计
        self.hit_count = {"memory": 0, "disk": 0}
        self.miss_count = {"memory": 0, "disk": 0}
        
        # 启动自动清理线程
        if auto_cleanup_interval > 0:
            self._start_auto_cleanup(auto_cleanup_interval)
    
    def _start_auto_cleanup(self, interval: int):
        """
        启动自动清理线程
        
        Args:
            interval: 清理间隔(秒)
        """
        def cleanup_thread():
            while True:
                time.sleep(interval)
                try:
                    self.cleanup()
                except Exception as e:
                    logger.error(f"自动清理缓存出错: {str(e)}")
        
        # 创建守护线程
        thread = threading.Thread(target=cleanup_thread, daemon=True)
        thread.start()
        logger.debug(f"已启动缓存自动清理线程，间隔: {interval}秒")
    
    def get(self, key: str, default: Any = None, 
            level: CacheLevel = CacheLevel.BOTH) -> Any:
        """
        获取缓存数据
        
        Args:
            key: 缓存键
            default: 默认值
            level: 缓存级别，默认同时检查内存和磁盘
            
        Returns:
            Any: 缓存的数据或默认值
        """
        # 检查内存缓存
        if level in (CacheLevel.MEMORY, CacheLevel.BOTH):
            with self.memory_lock:
                if key in self.memory_cache:
                    # 检查是否过期
                    if not self._is_memory_expired(key):
                        # 更新访问时间和计数
                        self.memory_metadata[key]["last_access"] = time.time()
                        self.memory_metadata[key]["access_count"] += 1
                        self.hit_count["memory"] += 1
                        return self.memory_cache[key]
                    else:
                        # 已过期，从内存缓存中删除
                        self._remove_from_memory(key)
                else:
                    self.miss_count["memory"] += 1
        
        # 检查磁盘缓存
        if level in (CacheLevel.DISK, CacheLevel.BOTH):
            with self.disk_lock:
                cache_path = self._get_disk_path(key)
                meta_path = cache_path + ".meta"
                
                if os.path.exists(cache_path) and os.path.exists(meta_path):
                    # 检查是否过期
                    if not self._is_disk_expired(meta_path):
                        try:
                            # 加载缓存数据
                            data = self._load_from_disk(cache_path)
                            
                            # 更新元数据
                            self._update_disk_metadata(meta_path)
                            
                            # 如果使用双级缓存，同时加入内存缓存
                            if level == CacheLevel.BOTH:
                                self.set(key, data, level=CacheLevel.MEMORY)
                                
                            self.hit_count["disk"] += 1
                            return data
                        except Exception as e:
                            logger.error(f"读取磁盘缓存出错 ({key}): {str(e)}")
                    else:
                        # 已过期，删除磁盘缓存
                        self._remove_from_disk(key)
                else:
                    self.miss_count["disk"] += 1
        
        return default
    
    def set(self, key: str, value: Any, 
            level: CacheLevel = CacheLevel.BOTH,
            memory_ttl: Optional[int] = None,
            disk_ttl: Optional[int] = None) -> bool:
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            value: 要缓存的值
            level: 缓存级别，默认同时写入内存和磁盘
            memory_ttl: 内存缓存TTL(秒)，None表示使用默认值
            disk_ttl: 磁盘缓存TTL(秒)，None表示使用默认值
            
        Returns:
            bool: 是否成功写入缓存
        """
        success = True
        
        # 写入内存缓存
        if level in (CacheLevel.MEMORY, CacheLevel.BOTH):
            try:
                self._set_memory(key, value, memory_ttl)
            except Exception as e:
                logger.error(f"写入内存缓存失败 ({key}): {str(e)}")
                success = False
        
        # 写入磁盘缓存
        if level in (CacheLevel.DISK, CacheLevel.BOTH):
            try:
                self._set_disk(key, value, disk_ttl)
            except Exception as e:
                logger.error(f"写入磁盘缓存失败 ({key}): {str(e)}")
                success = False
                
        return success
    
    def remove(self, key: str, level: CacheLevel = CacheLevel.BOTH) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            level: 缓存级别，默认同时从内存和磁盘删除
            
        Returns:
            bool: 是否成功删除
        """
        success = True
        
        # 从内存中删除
        if level in (CacheLevel.MEMORY, CacheLevel.BOTH):
            with self.memory_lock:
                success &= self._remove_from_memory(key)
                
        # 从磁盘删除
        if level in (CacheLevel.DISK, CacheLevel.BOTH):
            with self.disk_lock:
                success &= self._remove_from_disk(key)
                
        return success
    
    def clear(self, level: CacheLevel = CacheLevel.BOTH) -> bool:
        """
        清空缓存
        
        Args:
            level: 缓存级别，默认同时清空内存和磁盘
            
        Returns:
            bool: 是否成功清空
        """
        success = True
        
        # 清空内存缓存
        if level in (CacheLevel.MEMORY, CacheLevel.BOTH):
            with self.memory_lock:
                self.memory_cache.clear()
                self.memory_metadata.clear()
                self.memory_size = 0
                logger.info("已清空内存缓存")
                
        # 清空磁盘缓存
        if level in (CacheLevel.DISK, CacheLevel.BOTH):
            with self.disk_lock:
                try:
                    # 删除所有缓存文件
                    for filename in os.listdir(self.cache_dir):
                        file_path = os.path.join(self.cache_dir, filename)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    logger.info("已清空磁盘缓存")
                except Exception as e:
                    logger.error(f"清空磁盘缓存失败: {str(e)}")
                    success = False
                    
        return success
    
    def cleanup(self) -> Tuple[int, int]:
        """
        清理过期缓存
        
        Returns:
            Tuple[int, int]: 清理的(内存项数, 磁盘项数)
        """
        memory_cleaned = 0
        disk_cleaned = 0
        
        # 清理内存缓存
        with self.memory_lock:
            now = time.time()
            expired_keys = []
            
            # 找出过期的键
            for key, meta in self.memory_metadata.items():
                if now > meta["expires"]:
                    expired_keys.append(key)
                    
            # 删除过期项
            for key in expired_keys:
                self._remove_from_memory(key)
            
            memory_cleaned = len(expired_keys)
                
            # 如果内存使用超过限制，清理最不常用项
            if self.memory_size > self.memory_limit:
                logger.info(f"内存缓存超出限制 ({self.memory_size/1024/1024:.1f}MB/{self.memory_limit/1024/1024:.1f}MB)，清理中...")
                
                # 按最后访问时间排序
                items = list(self.memory_metadata.items())
                items.sort(key=lambda x: (x[1]["access_count"], x[1]["last_access"]))
                
                # 清理到限制的80%以下
                target_size = int(self.memory_limit * 0.8)
                for key, _ in items:
                    if self.memory_size <= target_size:
                        break
                        
                    if self._remove_from_memory(key):
                        memory_cleaned += 1
        
        # 清理磁盘缓存
        with self.disk_lock:
            # 遍历缓存目录中的所有文件
            total_size = 0
            cache_files = []
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".meta"):
                    continue
                    
                file_path = os.path.join(self.cache_dir, filename)
                meta_path = file_path + ".meta"
                
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    
                    # 检查是否过期
                    if os.path.exists(meta_path) and self._is_disk_expired(meta_path):
                        try:
                            os.unlink(file_path)
                            os.unlink(meta_path)
                            disk_cleaned += 1
                        except Exception as e:
                            logger.error(f"删除过期磁盘缓存文件失败 ({file_path}): {str(e)}")
                    else:
                        # 收集缓存文件信息
                        last_access = os.path.getatime(file_path)
                        cache_files.append((file_path, meta_path, file_size, last_access))
            
            # 如果磁盘缓存超过限制，清理最旧的文件
            if total_size > self.disk_limit and cache_files:
                logger.info(f"磁盘缓存超出限制 ({total_size/1024/1024:.1f}MB/{self.disk_limit/1024/1024:.1f}MB)，清理中...")
                
                # 按最后访问时间排序
                cache_files.sort(key=lambda x: x[3])
                
                # 清理到限制的80%以下
                target_size = int(self.disk_limit * 0.8)
                current_size = total_size
                
                for cache_path, meta_path, file_size, _ in cache_files:
                    if current_size <= target_size:
                        break
                        
                    try:
                        os.unlink(cache_path)
                        if os.path.exists(meta_path):
                            os.unlink(meta_path)
                            
                        current_size -= file_size
                        disk_cleaned += 1
                    except Exception as e:
                        logger.error(f"删除磁盘缓存文件失败 ({cache_path}): {str(e)}")
        
        if memory_cleaned > 0 or disk_cleaned > 0:
            logger.info(f"缓存清理完成: 内存 {memory_cleaned} 项, 磁盘 {disk_cleaned} 项")
            
        return memory_cleaned, disk_cleaned
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict: 缓存统计信息
        """
        # 获取磁盘缓存大小
        disk_size = 0
        disk_files = 0
        
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    disk_size += os.path.getsize(file_path)
                    disk_files += 1
        except Exception as e:
            logger.error(f"获取磁盘缓存统计信息失败: {str(e)}")
        
        return {
            "memory_items": len(self.memory_cache),
            "memory_size": self.memory_size,
            "memory_limit": self.memory_limit,
            "disk_files": disk_files // 2,  # 除以2，因为每个缓存项有.meta文件
            "disk_size": disk_size,
            "disk_limit": self.disk_limit,
            "hit_rate": {
                "memory": self.hit_count["memory"] / max(1, self.hit_count["memory"] + self.miss_count["memory"]),
                "disk": self.hit_count["disk"] / max(1, self.hit_count["disk"] + self.miss_count["disk"])
            },
            "hits": dict(self.hit_count),
            "misses": dict(self.miss_count)
        }
        
    # 内部工具方法
        
    def _set_memory(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """内部方法：写入内存缓存"""
        with self.memory_lock:
            # 估算对象大小
            try:
                if isinstance(value, pd.DataFrame):
                    obj_size = value.memory_usage(deep=True).sum()
                else:
                    obj_size = self._estimate_size(value)
            except Exception:
                # 保守估计
                obj_size = 1024 * 8  # 8KB
            
            # 如果单个对象大于内存限制的一半，则不缓存
            if obj_size > self.memory_limit / 2:
                logger.debug(f"对象过大 ({obj_size/1024/1024:.1f}MB)，跳过内存缓存")
                return False
                
            # 如果已存在，先移除旧值
            if key in self.memory_cache:
                old_size = self.memory_metadata[key]["size"]
                self.memory_size -= old_size
                
            # 计算过期时间
            ttl = ttl if ttl is not None else self.memory_ttl
            expires = time.time() + ttl
            
            # 更新内存缓存
            self.memory_cache[key] = value
            self.memory_metadata[key] = {
                "size": obj_size,
                "created": time.time(),
                "expires": expires,
                "last_access": time.time(),
                "access_count": 0
            }
            
            # 更新缓存大小
            self.memory_size += obj_size
            
            return True
    
    def _set_disk(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """内部方法：写入磁盘缓存"""
        with self.disk_lock:
            cache_path = self._get_disk_path(key)
            meta_path = cache_path + ".meta"
            
            # 计算过期时间
            ttl = ttl if ttl is not None else self.disk_ttl
            expires = time.time() + ttl
            
            try:
                # 将数据写入文件
                self._save_to_disk(value, cache_path)
                
                # 写入元数据
                metadata = {
                    "key": key,
                    "created": time.time(),
                    "expires": expires,
                    "last_access": time.time(),
                    "access_count": 0
                }
                
                with open(meta_path, 'wb') as f:
                    pickle.dump(metadata, f)
                    
                return True
            except Exception as e:
                logger.error(f"写入磁盘缓存失败 ({key}): {str(e)}")
                return False
    
    def _remove_from_memory(self, key: str) -> bool:
        """内部方法：从内存缓存删除项"""
        if key in self.memory_cache:
            # 更新内存使用量
            obj_size = self.memory_metadata[key]["size"]
            self.memory_size -= obj_size
            
            # 删除缓存项
            del self.memory_cache[key]
            del self.memory_metadata[key]
            return True
        return False
    
    def _remove_from_disk(self, key: str) -> bool:
        """内部方法：从磁盘缓存删除项"""
        cache_path = self._get_disk_path(key)
        meta_path = cache_path + ".meta"
        success = True
        
        # 删除缓存文件
        if os.path.exists(cache_path):
            try:
                os.unlink(cache_path)
            except Exception as e:
                logger.error(f"删除磁盘缓存文件失败 ({cache_path}): {str(e)}")
                success = False
                
        # 删除元数据文件
        if os.path.exists(meta_path):
            try:
                os.unlink(meta_path)
            except Exception as e:
                logger.error(f"删除磁盘缓存元数据文件失败 ({meta_path}): {str(e)}")
                success = False
                
        return success
    
    def _get_disk_path(self, key: str) -> str:
        """内部方法：获取磁盘缓存路径"""
        # 将键转换为文件名
        cache_key = hashlib.md5(key.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, cache_key)
    
    def _save_to_disk(self, value: Any, path: str) -> None:
        """内部方法：保存数据到磁盘"""
        try:
            # 针对不同类型使用不同的保存方法
            if isinstance(value, pd.DataFrame):
                value.to_pickle(path)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(value, f)
        except Exception as e:
            logger.error(f"保存数据到磁盘失败 ({path}): {str(e)}")
            raise
    
    def _load_from_disk(self, path: str) -> Any:
        """内部方法：从磁盘加载数据"""
        try:
            # 尝试作为DataFrame加载
            try:
                return pd.read_pickle(path)
            except:
                # 尝试作为普通对象加载
                with open(path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"从磁盘加载数据失败 ({path}): {str(e)}")
            raise
    
    def _update_disk_metadata(self, meta_path: str) -> None:
        """内部方法：更新磁盘缓存元数据"""
        try:
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
                
            # 更新访问时间和计数
            metadata["last_access"] = time.time()
            metadata["access_count"] += 1
            
            with open(meta_path, 'wb') as f:
                pickle.dump(metadata, f)
        except Exception as e:
            logger.error(f"更新磁盘缓存元数据失败 ({meta_path}): {str(e)}")
    
    def _is_memory_expired(self, key: str) -> bool:
        """内部方法：检查内存缓存是否过期"""
        return time.time() > self.memory_metadata[key]["expires"]
    
    def _is_disk_expired(self, meta_path: str) -> bool:
        """内部方法：检查磁盘缓存是否过期"""
        try:
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
                
            return time.time() > metadata["expires"]
        except Exception as e:
            logger.error(f"检查磁盘缓存是否过期失败 ({meta_path}): {str(e)}")
            return True  # 发生错误时视为过期
    
    def _estimate_size(self, obj: Any) -> int:
        """内部方法：估算对象大小(字节)"""
        # 对于numpy和pandas对象有专门的方法
        if isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
            return obj.nbytes
            
        # 其他类型简单估计
        try:
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            # 返回一个合理的默认值
            return 1024  # 1KB

# 缓存装饰器
def cached(key_pattern: str = "{module}.{function}:{args}",
          level: CacheLevel = CacheLevel.BOTH,
          ttl: Optional[int] = None,
          cache_manager: Optional[CacheManager] = None):
    """
    缓存装饰器
    
    Args:
        key_pattern: 缓存键模式，支持的变量：{module}, {function}, {args}, {kwargs}
        level: 缓存级别
        ttl: 过期时间(秒)
        cache_manager: 指定缓存管理器，如果为None则使用默认管理器
        
    Returns:
        装饰后的函数
    """
    # 使用默认缓存管理器
    if cache_manager is None:
        cache_dir = os.environ.get("CACHE_DIR", "./cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_manager = CacheManager(cache_dir=cache_dir)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 如果明确禁用缓存，直接调用原函数
            if level == CacheLevel.NONE:
                return func(*args, **kwargs)
                
            # 生成缓存键
            func_module = func.__module__
            func_name = func.__name__
            args_str = "_".join([str(arg) for arg in args])
            kwargs_str = "_".join([f"{k}_{v}" for k, v in sorted(kwargs.items())])
            
            cache_key = key_pattern.format(
                module=func_module,
                function=func_name,
                args=args_str,
                kwargs=kwargs_str
            )
            
            # 对长键名进行哈希处理
            if len(cache_key) > 250:
                args_hash = hashlib.md5(args_str.encode()).hexdigest()
                kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()
                cache_key = key_pattern.format(
                    module=func_module,
                    function=func_name,
                    args=args_hash,
                    kwargs=kwargs_hash
                )
            
            # 尝试获取缓存值
            cached_value = cache_manager.get(cache_key, level=level)
            if cached_value is not None:
                return cached_value
                
            # 执行函数
            result = func(*args, **kwargs)
            
            # 如果结果不为None，则缓存
            if result is not None:
                cache_manager.set(cache_key, result, level=level, memory_ttl=ttl, disk_ttl=ttl)
                
            return result
            
        return wrapper
    return decorator


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建缓存管理器
    cache = CacheManager(cache_dir="./cache", memory_limit_mb=100, disk_limit_mb=500)
    
    # 测试缓存
    cache.set("test_key", {"data": "测试数据", "created": datetime.now().isoformat()})
    
    result = cache.get("test_key")
    print(f"缓存数据: {result}")
    
    # 测试缓存装饰器
    @cached(ttl=60, cache_manager=cache)
    def slow_function(param):
        print(f"执行耗时函数 param={param}")
        time.sleep(1)  # 模拟耗时操作
        return f"处理结果: {param}"
    
    # 第一次调用，会执行函数
    print(slow_function("test1"))
    
    # 第二次调用，从缓存获取
    print(slow_function("test1"))
    
    # 不同参数，会执行函数
    print(slow_function("test2"))
    
    # 获取缓存统计
    stats = cache.get_stats()
    print(f"缓存统计: {stats}")
    
    # 清理缓存
    cache.cleanup() 