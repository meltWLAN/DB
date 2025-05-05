"""
缓存管理模块
实现高效的数据缓存机制
"""

import os
import json
import pickle
import hashlib
import time
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from ..utils.logger import get_logger
from ..config.settings import CACHE_DIR

class CacheManager:
    """缓存管理器类"""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_size_mb: int = 500, 
                 default_expiry_days: int = 1):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
            max_size_mb: 最大缓存大小（MB）
            default_expiry_days: 默认过期时间（天）
        """
        self.logger = get_logger("CacheManager")
        
        # 设置缓存目录
        if cache_dir is None:
            self.cache_dir = CACHE_DIR / "enhanced_cache"
        else:
            self.cache_dir = cache_dir
            
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置缓存配置
        self.max_size_bytes = max_size_mb * 1024 * 1024  # 转换为字节
        self.default_expiry = timedelta(days=default_expiry_days)
        
        # 缓存索引文件
        self.index_file = self.cache_dir / "cache_index.json"
        
        # 加载缓存索引
        self.cache_index = self._load_index()
        
        # 缓存统计
        self.hits = 0
        self.misses = 0
        
        self.logger.info(f"缓存管理器初始化完成，缓存目录：{self.cache_dir}，最大大小：{max_size_mb}MB")
        
        # 启动时清理过期缓存和保持大小限制
        self._clean_expired_cache()
        self._enforce_size_limit()
        
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """加载缓存索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"加载缓存索引失败: {str(e)}")
                return {}
        return {}
        
    def _save_index(self) -> None:
        """保存缓存索引"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存缓存索引失败: {str(e)}")
            
    def _generate_key(self, key_data: Any) -> str:
        """
        生成缓存键
        
        Args:
            key_data: 用于生成键的数据
            
        Returns:
            str: 缓存键
        """
        if isinstance(key_data, str):
            key_str = key_data
        else:
            try:
                key_str = json.dumps(key_data, sort_keys=True)
            except:
                key_str = str(key_data)
                
        # 计算哈希值作为键
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
        
    def _get_file_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"
        
    def _is_expired(self, cache_key: str) -> bool:
        """检查缓存是否过期"""
        if cache_key not in self.cache_index:
            return True
            
        expire_time = datetime.fromisoformat(self.cache_index[cache_key]["expire_time"])
        return datetime.now() > expire_time
        
    def _update_access_time(self, cache_key: str) -> None:
        """更新缓存访问时间"""
        if cache_key in self.cache_index:
            self.cache_index[cache_key]["last_access"] = datetime.now().isoformat()
            self._save_index()
            
    def get(self, key: Any, default: Any = None) -> Any:
        """
        获取缓存数据
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            Any: 缓存数据或默认值
        """
        cache_key = self._generate_key(key)
        file_path = self._get_file_path(cache_key)
        
        # 检查缓存是否存在且未过期
        if not file_path.exists() or self._is_expired(cache_key):
            self.misses += 1
            return default
            
        try:
            # 读取缓存数据
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            # 更新访问时间
            self._update_access_time(cache_key)
            
            self.hits += 1
            return data
            
        except Exception as e:
            self.logger.error(f"读取缓存失败 [{cache_key}]: {str(e)}")
            self.misses += 1
            return default
            
    def set(self, key: Any, value: Any, expiry: Optional[timedelta] = None) -> bool:
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            value: 缓存值
            expiry: 过期时间，None表示使用默认过期时间
            
        Returns:
            bool: 是否成功设置缓存
        """
        cache_key = self._generate_key(key)
        file_path = self._get_file_path(cache_key)
        
        if expiry is None:
            expiry = self.default_expiry
            
        expire_time = datetime.now() + expiry
        
        try:
            # 保存数据到缓存文件
            with open(file_path, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            # 更新缓存索引
            self.cache_index[cache_key] = {
                "key": str(key),
                "created": datetime.now().isoformat(),
                "last_access": datetime.now().isoformat(),
                "expire_time": expire_time.isoformat(),
                "size": file_path.stat().st_size
            }
            
            self._save_index()
            
            # 检查是否需要清理缓存
            self._enforce_size_limit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"设置缓存失败 [{cache_key}]: {str(e)}")
            return False
            
    def delete(self, key: Any) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否成功删除
        """
        cache_key = self._generate_key(key)
        file_path = self._get_file_path(cache_key)
        
        try:
            if file_path.exists():
                file_path.unlink()
                
            if cache_key in self.cache_index:
                del self.cache_index[cache_key]
                self._save_index()
                
            return True
            
        except Exception as e:
            self.logger.error(f"删除缓存失败 [{cache_key}]: {str(e)}")
            return False
            
    def clear(self) -> bool:
        """
        清空所有缓存
        
        Returns:
            bool: 是否成功清空
        """
        try:
            for file_path in self.cache_dir.glob("*.pkl"):
                file_path.unlink()
                
            self.cache_index = {}
            self._save_index()
            
            self.logger.info("缓存已清空")
            return True
            
        except Exception as e:
            self.logger.error(f"清空缓存失败: {str(e)}")
            return False
            
    def _clean_expired_cache(self) -> None:
        """清理过期缓存"""
        current_time = datetime.now()
        keys_to_delete = []
        
        for cache_key, info in self.cache_index.items():
            expire_time = datetime.fromisoformat(info["expire_time"])
            if current_time > expire_time:
                keys_to_delete.append(cache_key)
                
        for cache_key in keys_to_delete:
            file_path = self._get_file_path(cache_key)
            if file_path.exists():
                file_path.unlink()
                
            del self.cache_index[cache_key]
            
        if keys_to_delete:
            self._save_index()
            self.logger.info(f"已清理{len(keys_to_delete)}个过期缓存")
            
    def _enforce_size_limit(self) -> None:
        """
        强制缓存大小限制
        使用LRU策略删除最久未访问的缓存
        """
        # 计算当前缓存大小
        current_size = sum(info.get("size", 0) for info in self.cache_index.values())
        
        if current_size <= self.max_size_bytes:
            return
            
        # 按最后访问时间排序
        entries = [(k, datetime.fromisoformat(v["last_access"]), v.get("size", 0)) 
                   for k, v in self.cache_index.items()]
        entries.sort(key=lambda x: x[1])  # 按访问时间升序排序
        
        # 删除旧缓存直到大小符合限制
        removed_size = 0
        removed_count = 0
        
        for cache_key, _, size in entries:
            file_path = self._get_file_path(cache_key)
            if file_path.exists():
                file_path.unlink()
                
            del self.cache_index[cache_key]
            removed_size += size
            removed_count += 1
            
            if current_size - removed_size <= self.max_size_bytes:
                break
                
        if removed_count > 0:
            self._save_index()
            self.logger.info(f"缓存空间不足，已清理{removed_count}个最久未使用的缓存")
            
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        # 重新计算当前缓存大小
        current_size = sum(info.get("size", 0) for info in self.cache_index.values())
        current_size_mb = current_size / (1024 * 1024)
        
        stats = {
            "total_entries": len(self.cache_index),
            "current_size_mb": round(current_size_mb, 2),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "usage_percent": round(current_size * 100 / self.max_size_bytes, 2) if self.max_size_bytes > 0 else 0,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": round(self.hits * 100 / (self.hits + self.misses), 2) if (self.hits + self.misses) > 0 else 0
        }
        
        return stats
        
    def get_dataframe(self, key: Any, default: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """
        获取DataFrame类型的缓存数据
        针对Pandas DataFrame优化的方法
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            Optional[pd.DataFrame]: 缓存的DataFrame或默认值
        """
        cache_key = self._generate_key(key)
        file_path = self._get_file_path(cache_key)
        
        # 检查缓存是否存在且未过期
        if not file_path.exists() or self._is_expired(cache_key):
            self.misses += 1
            return default
            
        try:
            # 读取DataFrame缓存
            df = pd.read_pickle(file_path)
            
            # 更新访问时间
            self._update_access_time(cache_key)
            
            self.hits += 1
            return df
            
        except Exception as e:
            self.logger.error(f"读取DataFrame缓存失败 [{cache_key}]: {str(e)}")
            self.misses += 1
            return default
            
    def set_dataframe(self, key: Any, df: pd.DataFrame, expiry: Optional[timedelta] = None) -> bool:
        """
        设置DataFrame类型的缓存数据
        
        Args:
            key: 缓存键
            df: 要缓存的DataFrame
            expiry: 过期时间，None表示使用默认过期时间
            
        Returns:
            bool: 是否成功设置缓存
        """
        cache_key = self._generate_key(key)
        file_path = self._get_file_path(cache_key)
        
        if expiry is None:
            expiry = self.default_expiry
            
        expire_time = datetime.now() + expiry
        
        try:
            # 保存DataFrame到缓存文件
            df.to_pickle(file_path)
            
            # 更新缓存索引
            self.cache_index[cache_key] = {
                "key": str(key),
                "created": datetime.now().isoformat(),
                "last_access": datetime.now().isoformat(),
                "expire_time": expire_time.isoformat(),
                "size": file_path.stat().st_size,
                "type": "dataframe"
            }
            
            self._save_index()
            
            # 检查是否需要清理缓存
            self._enforce_size_limit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"设置DataFrame缓存失败 [{cache_key}]: {str(e)}")
            return False 