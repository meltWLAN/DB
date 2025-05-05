"""
动量分析模块（分布式优化版）- 第三阶段优化
提供股票动量分析的相关功能，包括技术指标计算、筛选和评分
实现了多项高级性能优化：
- 优化的数据存储（含压缩和索引）
- 分布式计算框架
- 高效的数据序列化
- 自适应资源管理
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from pathlib import Path
import tushare as ts
import warnings
import asyncio
import aiohttp
import time
import pickle
import hashlib
import functools
from collections import OrderedDict
import psutil
import gc
import zlib
import json
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool, cpu_count, Manager, shared_memory, Lock
import h5py
import redis
import dask
import dask.dataframe as dd
from distributed import Client, LocalCluster, wait
import pyarrow as pa
import pyarrow.parquet as pq
import lz4.frame
import joblib

# 导入第二阶段优化版本作为基础
try:
    from momentum_analysis_async import AsyncMomentumAnalyzer
except ImportError:
    print("找不到第二阶段优化版本，请确保文件存在")
    sys.exit(1)

warnings.filterwarnings('ignore')

# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 导入项目配置
try:
    from src.enhanced.config.settings import TUSHARE_TOKEN, LOG_DIR, DATA_DIR, RESULTS_DIR
except ImportError:
    # 设置默认配置
    TUSHARE_TOKEN = ""
    LOG_DIR = "./logs"
    DATA_DIR = "./data"
    RESULTS_DIR = "./results"

# 确保目录存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "charts"), exist_ok=True)

# 添加高级存储目录
CACHE_DIR = "./cache"
PARQUET_STORE = os.path.join(DATA_DIR, "parquet_store")
HDF_STORE = os.path.join(DATA_DIR, "hdf_store")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PARQUET_STORE, exist_ok=True)
os.makedirs(HDF_STORE, exist_ok=True)

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # 添加文件处理器
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "momentum_analysis_distributed.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# 设置Tushare
if not TUSHARE_TOKEN:
    # 直接在代码中设置Token（如果配置文件中没有设置）
    TUSHARE_TOKEN = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
else:
    pro = None

class DistributedMomentumAnalyzer(AsyncMomentumAnalyzer):
    """分布式优化的动量分析器类，提供高级数据存储和分布式计算功能"""
    
    def __init__(self, use_tushare=True, use_multiprocessing=True, workers=None, 
                 cache_size=512, cache_timeout=86400, batch_size=100, memory_limit=0.8,
                 thread_pool_size=20, use_distributed=True, storage_format='parquet',
                 compression_level=6, redis_config=None, use_shared_memory=True):
        """初始化分布式动量分析器
        
        Args:
            use_tushare: 是否使用Tushare数据
            use_multiprocessing: 是否使用多进程
            workers: 工作进程数，None则自动确定
            cache_size: 内存缓存项数量上限
            cache_timeout: 缓存过期时间(秒)
            batch_size: 批处理大小
            memory_limit: 内存使用限制百分比(0-1)
            thread_pool_size: 线程池大小，用于异步IO
            use_distributed: 是否使用分布式计算
            storage_format: 数据存储格式('parquet'或'hdf')
            compression_level: 压缩级别(0-9)
            redis_config: Redis配置(dict)，用于分布式缓存
            use_shared_memory: 是否使用共享内存
        """
        # 调用父类初始化方法
        super().__init__(use_tushare, use_multiprocessing, workers, 
                        cache_size, cache_timeout, batch_size, memory_limit,
                        thread_pool_size)
        
        # 分布式和存储相关设置
        self.use_distributed = use_distributed
        self.storage_format = storage_format
        self.compression_level = compression_level
        self.use_shared_memory = use_shared_memory
        
        # 共享内存对象
        self.shared_mem = {}
        self.shared_mem_lock = Lock()
        
        # 分布式计算客户端
        self.dask_client = None
        self.dask_cluster = None
        
        # Redis缓存客户端
        self.redis_client = None
        if redis_config:
            try:
                self.redis_client = redis.Redis(**redis_config)
                self.redis_client.ping()  # 测试连接
                logger.info("Redis连接成功，使用分布式缓存")
            except Exception as e:
                logger.warning(f"Redis连接失败: {str(e)}，将使用本地缓存")
                self.redis_client = None
                
        # 初始化分布式计算环境
        if self.use_distributed:
            self._init_distributed_env()
        
        logger.info(f"初始化分布式动量分析器: 分布式计算={use_distributed}, "
                   f"存储格式={storage_format}, 共享内存={use_shared_memory}")
    
    def _init_distributed_env(self):
        """初始化分布式计算环境"""
        try:
            # 确定工作进程数
            n_workers = self.workers if self.workers else max(1, cpu_count() - 1)
            
            # 创建本地Dask集群
            self.dask_cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=2,
                memory_limit='auto',
                processes=True,
                silence_logs=logging.WARNING
            )
            
            # 创建客户端
            self.dask_client = Client(self.dask_cluster)
            
            logger.info(f"初始化分布式计算环境成功：{n_workers}个工作进程")
            logger.info(f"Dask仪表盘地址: {self.dask_client.dashboard_link}")
        except Exception as e:
            logger.error(f"初始化分布式计算环境失败: {str(e)}")
            self.use_distributed = False
            
    def __del__(self):
        """析构函数，清理资源"""
        # 关闭分布式客户端和集群
        if self.dask_client:
            try:
                self.dask_client.close()
            except:
                pass
        
        if self.dask_cluster:
            try:
                self.dask_cluster.close()
            except:
                pass
        
        # 清理共享内存
        for name, shm in self.shared_mem.items():
            try:
                shm.close()
                shm.unlink()
            except:
                pass
        
        # 调用父类析构函数
        super().__del__() if hasattr(super(), '__del__') else None 
    
    # ===== 高级数据存储方法 =====
    
    def _get_storage_path(self, ts_code, storage_type=None):
        """获取股票数据的存储路径
        
        Args:
            ts_code: 股票代码
            storage_type: 存储类型，默认使用当前设置的格式
            
        Returns:
            str: 文件路径
        """
        if storage_type is None:
            storage_type = self.storage_format
        
        # 使用目录层级存储，避免单一目录文件过多
        # 使用股票代码的前三位作为子目录名
        subdir = ts_code[:3] if len(ts_code) >= 3 else "misc"
        
        if storage_type == 'parquet':
            storage_dir = os.path.join(PARQUET_STORE, subdir)
            os.makedirs(storage_dir, exist_ok=True)
            return os.path.join(storage_dir, f"{ts_code}.parquet")
        elif storage_type == 'hdf':
            storage_dir = os.path.join(HDF_STORE, subdir)
            os.makedirs(storage_dir, exist_ok=True)
            return os.path.join(storage_dir, f"{ts_code}.h5")
        else:
            # 默认使用CSV格式
            storage_dir = os.path.join(DATA_DIR, subdir)
            os.makedirs(storage_dir, exist_ok=True)
            return os.path.join(storage_dir, f"{ts_code}.csv")
    
    def _store_data_parquet(self, ts_code, data):
        """将数据存储为Parquet格式
        
        Args:
            ts_code: 股票代码
            data: 股票数据DataFrame
            
        Returns:
            bool: 成功与否
        """
        try:
            if data.empty:
                return False
            
            # 复制数据防止修改原始数据
            df = data.copy()
            
            # 确保索引被保存
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            
            # 使用PyArrow保存为Parquet格式，支持多种压缩
            path = self._get_storage_path(ts_code, 'parquet')
            
            # 使用PyArrow表格格式
            table = pa.Table.from_pandas(df)
            
            # 压缩选项
            compression = 'snappy'  # 选择一种高效的压缩方式
            pq.write_table(table, path, compression=compression)
            
            logger.debug(f"将{ts_code}数据存储为Parquet格式：{path}")
            return True
        except Exception as e:
            logger.error(f"存储{ts_code}数据为Parquet格式失败: {str(e)}")
            return False
    
    def _read_data_parquet(self, ts_code, start_date=None, end_date=None):
        """从Parquet文件读取数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 股票数据
        """
        try:
            path = self._get_storage_path(ts_code, 'parquet')
            if not os.path.exists(path):
                logger.debug(f"找不到{ts_code}的Parquet数据：{path}")
                return pd.DataFrame()
            
            # 使用过滤器筛选日期范围，提高效率
            filters = []
            if start_date or end_date:
                # 需要构建过滤条件
                if 'trade_date' in pq.read_schema(path).names:
                    if start_date:
                        filters.append(('trade_date', '>=', pd.Timestamp(start_date)))
                    if end_date:
                        filters.append(('trade_date', '<=', pd.Timestamp(end_date)))
            
            # 如果有过滤条件且PyArrow支持，使用过滤
            if filters:
                try:
                    df = pq.read_table(path, filters=filters).to_pandas()
                except:
                    # 如果过滤失败，读取全部数据再过滤
                    df = pq.read_table(path).to_pandas()
                    
                    # 手动过滤日期
                    if 'trade_date' in df.columns:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        if start_date:
                            df = df[df['trade_date'] >= pd.Timestamp(start_date)]
                        if end_date:
                            df = df[df['trade_date'] <= pd.Timestamp(end_date)]
            else:
                # 读取全部数据
                df = pq.read_table(path).to_pandas()
            
            # 设置日期索引
            if 'trade_date' in df.columns:
                df.set_index('trade_date', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"从Parquet读取{ts_code}数据失败: {str(e)}")
            return pd.DataFrame()
    
    def _store_data_hdf(self, ts_code, data):
        """将数据存储为HDF5格式
        
        Args:
            ts_code: 股票代码
            data: 股票数据DataFrame
            
        Returns:
            bool: 成功与否
        """
        try:
            if data.empty:
                return False
            
            # 复制数据防止修改原始数据
            df = data.copy()
            
            # 确保日期列
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            
            path = self._get_storage_path(ts_code, 'hdf')
            
            # 使用pandas HDFStore存储
            with pd.HDFStore(path, mode='w', complevel=self.compression_level, complib='blosc') as store:
                store.put('data', df, format='table')
            
            logger.debug(f"将{ts_code}数据存储为HDF5格式：{path}")
            return True
        except Exception as e:
            logger.error(f"存储{ts_code}数据为HDF5格式失败: {str(e)}")
            return False
    
    def _read_data_hdf(self, ts_code, start_date=None, end_date=None):
        """从HDF5文件读取数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 股票数据
        """
        try:
            path = self._get_storage_path(ts_code, 'hdf')
            if not os.path.exists(path):
                logger.debug(f"找不到{ts_code}的HDF5数据：{path}")
                return pd.DataFrame()
            
            # 使用查询语句筛选日期范围
            query = None
            if start_date or end_date:
                query_parts = []
                if start_date:
                    query_parts.append(f"trade_date >= '{pd.Timestamp(start_date)}'")
                if end_date:
                    query_parts.append(f"trade_date <= '{pd.Timestamp(end_date)}'")
                if query_parts:
                    query = " & ".join(query_parts)
            
            # 打开HDFStore
            with pd.HDFStore(path, mode='r') as store:
                if query:
                    try:
                        df = store.select('data', where=query)
                    except:
                        # 如果查询失败，读取全部数据再过滤
                        df = store.select('data')
                        # 手动过滤
                        if 'trade_date' in df.columns:
                            df['trade_date'] = pd.to_datetime(df['trade_date'])
                            if start_date:
                                df = df[df['trade_date'] >= pd.Timestamp(start_date)]
                            if end_date:
                                df = df[df['trade_date'] <= pd.Timestamp(end_date)]
                else:
                    df = store.select('data')
            
            # 设置日期索引
            if 'trade_date' in df.columns:
                df.set_index('trade_date', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"从HDF5读取{ts_code}数据失败: {str(e)}")
            return pd.DataFrame()
    
    def store_stock_data(self, ts_code, data):
        """存储股票数据（自动选择存储格式）
        
        Args:
            ts_code: 股票代码
            data: 股票数据DataFrame
            
        Returns:
            bool: 成功与否
        """
        if self.storage_format == 'parquet':
            return self._store_data_parquet(ts_code, data)
        elif self.storage_format == 'hdf':
            return self._store_data_hdf(ts_code, data)
        else:
            # 默认CSV格式
            try:
                path = self._get_storage_path(ts_code, 'csv')
                data.to_csv(path)
                return True
            except Exception as e:
                logger.error(f"存储{ts_code}数据为CSV格式失败: {str(e)}")
                return False
    
    def read_stock_data(self, ts_code, start_date=None, end_date=None):
        """读取股票数据（自动选择存储格式）
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 股票数据
        """
        if self.storage_format == 'parquet':
            return self._read_data_parquet(ts_code, start_date, end_date)
        elif self.storage_format == 'hdf':
            return self._read_data_hdf(ts_code, start_date, end_date)
        else:
            # 默认CSV格式
            try:
                path = self._get_storage_path(ts_code, 'csv')
                if not os.path.exists(path):
                    return pd.DataFrame()
                
                df = pd.read_csv(path)
                
                # 过滤日期
                if 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    if start_date:
                        df = df[df['trade_date'] >= pd.Timestamp(start_date)]
                    if end_date:
                        df = df[df['trade_date'] <= pd.Timestamp(end_date)]
                    df.set_index('trade_date', inplace=True)
                
                return df
            except Exception as e:
                logger.error(f"从CSV读取{ts_code}数据失败: {str(e)}")
                return pd.DataFrame()
    
    # ===== 分布式计算和共享内存方法 =====
    
    def _create_shared_memory_for_dataframe(self, df, name=None):
        """为DataFrame创建共享内存
        
        Args:
            df: 数据DataFrame
            name: 共享内存名称，默认自动生成
            
        Returns:
            tuple: (共享内存名称, 形状信息)
        """
        try:
            if df.empty:
                return None, None
            
            # 生成唯一名称
            if name is None:
                name = f"shm_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"
            
            # 序列化DataFrame为字节
            data_bytes = df.to_pickle(compression='xz')
            data_size = len(data_bytes)
            
            # 创建共享内存
            shm = shared_memory.SharedMemory(create=True, size=data_size, name=name)
            
            # 写入数据
            shm.buf[:data_size] = data_bytes
            
            # 保存共享内存对象引用
            with self.shared_mem_lock:
                self.shared_mem[name] = shm
            
            # 返回名称和大小信息
            metadata = {'size': data_size}
            
            return name, metadata
        except Exception as e:
            logger.error(f"创建共享内存失败: {str(e)}")
            return None, None
    
    def _read_dataframe_from_shared_memory(self, name, metadata):
        """从共享内存读取DataFrame
        
        Args:
            name: 共享内存名称
            metadata: 元数据（包含大小信息）
            
        Returns:
            DataFrame: 恢复的DataFrame
        """
        try:
            # 尝试从已有对象获取共享内存
            shm = None
            with self.shared_mem_lock:
                if name in self.shared_mem:
                    shm = self.shared_mem[name]
            
            # 如果没有找到，则连接到现有共享内存
            if shm is None:
                try:
                    shm = shared_memory.SharedMemory(name=name)
                    # 保存引用
                    with self.shared_mem_lock:
                        self.shared_mem[name] = shm
                except Exception as e:
                    logger.error(f"连接到共享内存失败: {str(e)}")
                    return pd.DataFrame()
            
            # 读取数据
            data_size = metadata['size']
            data_bytes = bytes(shm.buf[:data_size])
            
            # 反序列化为DataFrame
            df = pd.read_pickle(data_bytes, compression='xz')
            
            return df
        except Exception as e:
            logger.error(f"从共享内存读取DataFrame失败: {str(e)}")
            return pd.DataFrame()
    
    def _release_shared_memory(self, name):
        """释放共享内存
        
        Args:
            name: 共享内存名称
        """
        with self.shared_mem_lock:
            if name in self.shared_mem:
                try:
                    self.shared_mem[name].close()
                    self.shared_mem[name].unlink()
                    del self.shared_mem[name]
                except Exception as e:
                    logger.error(f"释放共享内存失败: {str(e)}")
    
    def _distribute_stocks_to_workers(self, stock_list, n_partitions=None):
        """将股票列表分配给多个工作节点
        
        Args:
            stock_list: 股票列表DataFrame
            n_partitions: 分区数，默认为工作节点数
            
        Returns:
            list: 股票列表分区
        """
        if stock_list.empty:
            return []
        
        # 确定分区数
        if n_partitions is None:
            if self.dask_client:
                n_partitions = len(self.dask_client.ncores())
            else:
                n_partitions = max(1, cpu_count() - 1)
        
        # 确保分区数不超过股票数
        n_partitions = min(n_partitions, len(stock_list))
        
        # 划分股票列表
        partitions = np.array_split(stock_list, n_partitions)
        
        return [df for df in partitions if not df.empty]
    
    def _analyze_partition_distributed(self, partition_df, min_score=60):
        """分析一个分区的股票（分布式工作函数）
        
        Args:
            partition_df: 分区股票列表
            min_score: 最低分数阈值
            
        Returns:
            list: 分析结果
        """
        results = []
        
        for _, stock in partition_df.iterrows():
            try:
                # 准备分析参数
                stock_data = stock.to_dict()
                stock_data['min_score'] = min_score
                
                # 分析单只股票
                result = self.analyze_single_stock_optimized(stock_data)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.error(f"分析股票 {stock.get('ts_code', 'unknown')} 失败: {str(e)}")
        
        return results
    
    def analyze_stocks_distributed(self, stock_list, sample_size=100, min_score=60):
        """使用分布式计算分析股票列表
        
        Args:
            stock_list: 股票列表DataFrame
            sample_size: 样本大小
            min_score: 最低分数阈值
            
        Returns:
            list: 分析结果
        """
        # 检查分布式环境
        if not self.use_distributed or not self.dask_client:
            logger.warning("分布式环境未初始化，使用普通多进程分析")
            return super().analyze_stocks(stock_list, sample_size, min_score)
        
        try:
            # 记录原始股票数量
            original_count = len(stock_list)
            logger.info(f"准备分布式分析 {original_count} 支股票")
            
            # 限制样本大小
            if sample_size < len(stock_list):
                stock_list = stock_list.sample(sample_size)
                logger.info(f"从 {original_count} 支股票中随机选择 {sample_size} 支进行分析")
                
            # 无股票则返回空结果
            if stock_list.empty:
                return []
                
            # 分区股票列表
            partitions = self._distribute_stocks_to_workers(stock_list)
            logger.info(f"将 {len(stock_list)} 支股票分为 {len(partitions)} 个分区")
            
            # 将分区转换为Dask DataFrame
            dask_partitions = []
            for i, part in enumerate(partitions):
                dask_partitions.append(dask.delayed(self._analyze_partition_distributed)(part, min_score))
            
            # 启动分布式计算
            logger.info("启动分布式计算")
            start_time = time.time()
            results_partitions = dask.compute(*dask_partitions)
            
            # 合并结果
            all_results = []
            for part_result in results_partitions:
                all_results.extend(part_result)
            
            end_time = time.time()
            
            # 按得分排序
            all_results.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"分布式分析完成，耗时: {end_time - start_time:.2f}秒, "
                       f"找到 {len(all_results)} 支符合条件的股票")
            
            # 将结果保存为CSV
            if all_results:
                result_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'data' and k != 'score_details'}
                                        for r in all_results])
                csv_path = os.path.join(RESULTS_DIR, 
                                      f"momentum_distributed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                logger.info(f"已将分析结果保存至: {csv_path}")
            
            return all_results
        except Exception as e:
            logger.error(f"分布式分析失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 回退到普通分析
            logger.info("回退到普通多进程分析")
            return super().analyze_stocks(stock_list, sample_size, min_score)
    
    def _process_data_partition(self, codes_partition, start_date, end_date):
        """处理一批股票代码的数据获取（分布式工作函数）
        
        Args:
            codes_partition: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            dict: 股票数据字典 {ts_code: dataframe}
        """
        results = {}
        
        for ts_code in codes_partition:
            try:
                # 尝试从存储读取
                df = self.read_stock_data(ts_code, start_date, end_date)
                
                if df.empty and self.use_tushare:
                    # 从API获取并保存
                    df = self.get_stock_daily_data(ts_code, start_date, end_date)
                    if not df.empty:
                        self.store_stock_data(ts_code, df)
                
                if not df.empty:
                    # 计算技术指标
                    df = self.calculate_momentum_vectorized(df)
                    results[ts_code] = df
            except Exception as e:
                logger.error(f"处理 {ts_code} 的数据失败: {str(e)}")
        
        return results
    
    def prefetch_data_distributed(self, stock_list, start_date=None, end_date=None):
        """使用分布式计算预取并处理多只股票数据
        
        Args:
            stock_list: 股票列表DataFrame或列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            dict: 股票数据字典 {ts_code: dataframe}
        """
        if not self.use_distributed or not self.dask_client:
            logger.warning("分布式环境未初始化，使用普通方式预取数据")
            return {}
        
        try:
            # 设置默认日期
            if not end_date:
                end_date = datetime.now().strftime('%Y%m%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            
            # 提取股票代码
            if isinstance(stock_list, pd.DataFrame):
                codes = stock_list['ts_code'].tolist()
            else:
                codes = list(stock_list)
            
            # 分区
            partitions = np.array_split(codes, len(self.dask_client.ncores()))
            
            # 创建分布式任务
            logger.info(f"分布式预取 {len(codes)} 支股票的数据")
            tasks = []
            for part in partitions:
                if len(part) > 0:
                    task = dask.delayed(self._process_data_partition)(part, start_date, end_date)
                    tasks.append(task)
            
            # 执行分布式计算
            results_partitions = dask.compute(*tasks)
            
            # 合并结果
            all_data = {}
            for part_result in results_partitions:
                all_data.update(part_result)
            
            logger.info(f"分布式预取完成，共获取 {len(all_data)} 支股票的数据")
            
            return all_data
        except Exception as e:
            logger.error(f"分布式预取数据失败: {str(e)}")
            return {}
    
    # ===== 重写父类方法 =====
    
    def get_stock_daily_data(self, ts_code, start_date=None, end_date=None):
        """获取股票日线数据（增强版，带分布式缓存支持）
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 股票数据
        """
        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            
        # 检查内存缓存
        cache_key = f"{ts_code}_{start_date}_{end_date}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            self.cache_hit_stats["hits"] += 1
            return cached_data
        
        # 检查Redis分布式缓存
        redis_key = f"stock_data:{cache_key}"
        if self.redis_client:
            try:
                redis_data = self.redis_client.get(redis_key)
                if redis_data:
                    # 解压并反序列化
                    decompressed = lz4.frame.decompress(redis_data)
                    df = pd.read_pickle(decompressed)
                    # 加入内存缓存
                    self._add_to_cache(cache_key, df)
                    logger.debug(f"Redis缓存命中: {ts_code}")
                    self.cache_hit_stats["hits"] += 1
                    return df
            except Exception as e:
                logger.warning(f"读取Redis缓存失败: {str(e)}")
        
        # 缓存未命中
        self.cache_hit_stats["misses"] += 1
        
        # 尝试从优化存储读取
        df = self.read_stock_data(ts_code, start_date, end_date)
        
        if not df.empty:
            # 加入各级缓存
            self._add_to_cache(cache_key, df)
            
            # 加入Redis缓存
            if self.redis_client:
                try:
                    # 使用LZ4压缩序列化
                    pickled = df.to_pickle(compression=None)
                    compressed = lz4.frame.compress(pickled)
                    # 设置过期时间
                    self.redis_client.setex(redis_key, self.cache_timeout, compressed)
                except Exception as e:
                    logger.warning(f"写入Redis缓存失败: {str(e)}")
            
            return df
        
        # 存储中没有，使用API获取
        df = super().get_stock_daily_data(ts_code, start_date, end_date)
        
        # 如果获取成功，保存到存储
        if not df.empty:
            self.store_stock_data(ts_code, df)
        
        return df
    
    def analyze_stocks(self, stock_list, sample_size=100, min_score=60):
        """分析股票列表的主接口方法，自动使用分布式计算
        
        Args:
            stock_list: 股票列表DataFrame
            sample_size: 样本大小
            min_score: 最低分数阈值
            
        Returns:
            list: 分析结果
        """
        if self.use_distributed and self.dask_client:
            return self.analyze_stocks_distributed(stock_list, sample_size, min_score)
        else:
            return super().analyze_stocks(stock_list, sample_size, min_score)
    
    def warm_up_cache(self, stock_list, top_n=20):
        """预热缓存，使用分布式计算提高效率
        
        Args:
            stock_list: 股票列表DataFrame
            top_n: 预热的股票数量上限
        """
        if stock_list.empty or len(stock_list) == 0:
            logger.warning("股票列表为空，无法预热缓存")
            return
            
        # 如果列表太大，只预热一部分热门股票
        if top_n > 0 and len(stock_list) > top_n:
            stock_subset = stock_list.head(top_n)
        else:
            stock_subset = stock_list
            
        logger.info(f"开始预热缓存，加载 {len(stock_subset)} 支股票的数据")
        
        # 获取日期范围
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        # 使用分布式预取
        if self.use_distributed and self.dask_client:
            self.prefetch_data_distributed(stock_subset, start_date, end_date)
        else:
            # 使用父类的预热方法
            super().warm_up_cache(stock_subset, top_n)
    
    def analyze_single_stock_distributed(self, stock_data):
        """分布式环境下的单个股票分析函数
        
        Args:
            stock_data: 股票信息字典
            
        Returns:
            dict: 分析结果或None
        """
        # 这个方法主要为了在分布式环境中调用，内部逻辑与父类相同
        return self.analyze_single_stock_optimized(stock_data)
    
    def calculate_batch_statistics(self, results):
        """计算批处理统计信息
        
        Args:
            results: 分析结果列表
            
        Returns:
            dict: 统计信息
        """
        if not results:
            return {}
        
        scores = [r['score'] for r in results]
        momentum_values = [r.get('momentum_20', 0) for r in results]
        rsi_values = [r.get('rsi', 0) for r in results]
        
        stats = {
            'result_count': len(results),
            'avg_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'avg_momentum': sum(momentum_values) / len(momentum_values),
            'avg_rsi': sum(rsi_values) / len(rsi_values),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 分析行业分布
        industries = {}
        for r in results:
            ind = r.get('industry', '未知')
            if ind in industries:
                industries[ind] += 1
            else:
                industries[ind] = 1
        
        # 排序行业并添加到统计中
        top_industries = sorted(industries.items(), key=lambda x: x[1], reverse=True)[:5]
        stats['top_industries'] = {k: v for k, v in top_industries}
        
        return stats
    
    def export_results_as_json(self, results, output_path=None):
        """将分析结果导出为JSON格式
        
        Args:
            results: 分析结果列表
            output_path: 输出路径，默认自动生成
            
        Returns:
            str: 输出文件路径
        """
        if not results:
            return None
        
        # 处理结果，移除不可序列化的部分
        export_results = []
        for r in results:
            export_item = {k: v for k, v in r.items() if k != 'data' and k != 'score_details'}
            # 处理score_details
            if 'score_details' in r:
                export_item['score_details'] = r['score_details']
            export_results.append(export_item)
        
        # 添加统计信息
        stats = self.calculate_batch_statistics(results)
        export_data = {
            'results': export_results,
            'statistics': stats
        }
        
        # 确定输出路径
        if not output_path:
            output_path = os.path.join(
                RESULTS_DIR, 
                f"momentum_distributed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        # 写入文件
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            logger.info(f"分析结果已导出为JSON: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"导出JSON失败: {str(e)}")
            return None

# 提供便捷测试函数
def test_distributed_performance(sample_size=20, use_distributed=True):
    """测试分布式优化版的性能
    
    Args:
        sample_size: 测试样本大小
        use_distributed: 是否使用分布式计算
        
    Returns:
        tuple: (分析结果, 耗时)
    """
    # 初始化分析器
    analyzer = DistributedMomentumAnalyzer(
        use_tushare=True, 
        use_multiprocessing=True,
        use_distributed=use_distributed,
        storage_format='parquet'
    )
    
    # 获取股票列表
    stocks = analyzer.get_stock_list()
    print(f"获取到 {len(stocks)} 支股票")
    
    # 预热缓存
    analyzer.warm_up_cache(stocks.head(sample_size))
    
    # 分析样本股票
    start_time = time.time()
    results = analyzer.analyze_stocks(stocks.head(sample_size), min_score=50)
    end_time = time.time()
    
    # 输出结果和性能数据
    duration = end_time - start_time
    print(f"分析完成，耗时: {duration:.2f}秒")
    print(f"符合条件的股票数量: {len(results)}")
    
    # 输出股票分析结果
    for r in results[:5]:  # 只显示前5个结果
        print(f"{r['name']}({r['ts_code']}): 得分={r['score']}, 动量20={r['momentum_20']:.2%}, RSI={r['rsi']:.2f}")
    
    # 导出结果
    analyzer.export_results_as_json(results)
    
    # 关闭分布式客户端
    if analyzer.dask_client:
        analyzer.dask_client.close()
    
    return results, duration

if __name__ == "__main__":
    # 运行测试
    import argparse
    
    parser = argparse.ArgumentParser(description="测试分布式优化的动量分析器")
    parser.add_argument("-s", "--sample", type=int, default=20, help="测试样本大小")
    parser.add_argument("-d", "--distributed", action="store_true", help="使用分布式计算")
    args = parser.parse_args()
    
    print(f"开始测试: 样本大小={args.sample}, 分布式计算={args.distributed}")
    test_distributed_performance(args.sample, args.distributed) 