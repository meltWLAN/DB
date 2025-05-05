"""
动量分析模块（异步优化版）
提供股票动量分析的相关功能，包括技术指标计算、筛选和评分
实现了多项性能优化：
- 异步数据获取
- 增强的内存管理
- 高级缓存预热功能
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import threading
from multiprocessing import Pool, cpu_count, Manager
from queue import Queue

# 导入第一阶段优化版本作为基础
try:
    from momentum_analysis_enhanced_performance import EnhancedMomentumAnalyzer, enhanced_disk_cache
except ImportError:
    print("找不到第一阶段优化版本，请确保文件存在")
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

# 添加缓存目录
CACHE_DIR = "./cache"
MEMORY_PROFILE_DIR = os.path.join(LOG_DIR, "memory_profiles")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MEMORY_PROFILE_DIR, exist_ok=True)

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # 添加文件处理器
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "momentum_analysis_async.log"))
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

class AsyncMomentumAnalyzer(EnhancedMomentumAnalyzer):
    """异步优化的动量分析器类，提供异步数据获取和高级内存管理"""
    
    def __init__(self, use_tushare=True, use_multiprocessing=True, workers=None, 
                 cache_size=256, cache_timeout=86400, batch_size=50, memory_limit=0.8,
                 thread_pool_size=10):
        """初始化异步动量分析器
        
        Args:
            use_tushare: 是否使用Tushare数据
            use_multiprocessing: 是否使用多进程
            workers: 工作进程数，None则自动确定
            cache_size: 内存缓存项数量上限
            cache_timeout: 缓存过期时间(秒)
            batch_size: 批处理大小
            memory_limit: 内存使用限制百分比(0-1)
            thread_pool_size: 线程池大小，用于异步IO
        """
        # 调用父类初始化方法
        super().__init__(use_tushare, use_multiprocessing, workers, 
                        cache_size, cache_timeout, batch_size, memory_limit)
        
        # 异步相关设置
        self.thread_pool_size = thread_pool_size
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        self.async_loop = None  # 将在第一次需要时创建
        
        # 高级内存管理
        self.memory_usage_snapshots = []  # 记录内存使用情况
        self.cache_hit_stats = {"hits": 0, "misses": 0}  # 缓存命中统计
        self.last_cleanup_time = time.time()  # 上次清理时间
        self.cleanup_interval = 60  # 默认1分钟检查一次内存
        
        # 创建共享同步队列，用于跨进程传递数据
        self.data_queue = Queue()
        
        # 预热状态
        self.prewarmed = False
        self.prewarmed_codes = set()
        
        logger.info(f"初始化异步优化动量分析器: 线程池大小={thread_pool_size}, "
                   f"缓存大小={cache_size}, 内存限制={memory_limit*100}%")
    
    def _get_event_loop(self):
        """获取或创建事件循环"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # 如果不在主线程，创建新循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop
    
    def _run_async(self, coro):
        """运行异步协程并返回结果"""
        loop = self._get_event_loop()
        return loop.run_until_complete(coro)
    
    async def _fetch_stock_data_async(self, ts_code, start_date, end_date):
        """异步获取股票数据"""
        logger.debug(f"异步获取{ts_code}数据")
        
        # 先检查缓存
        cache_key = f"{ts_code}_{start_date}_{end_date}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            self.cache_hit_stats["hits"] += 1
            return cached_data
        
        self.cache_hit_stats["misses"] += 1
        
        # 使用Tushare API获取数据
        # 注意：Tushare API不支持原生异步，所以在线程池中运行
        if self.use_tushare:
            try:
                loop = asyncio.get_event_loop()
                # 在线程池中运行阻塞操作
                df = await loop.run_in_executor(
                    self.thread_pool, 
                    lambda: pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                )
                
                if df.empty:
                    # 尝试备用API
                    df = await loop.run_in_executor(
                        self.thread_pool,
                        lambda: ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date)
                    )
                
                if not df.empty:
                    # 数据处理
                    if 'trade_date' in df.columns:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        df.sort_values('trade_date', inplace=True)
                        df.set_index('trade_date', inplace=True)
                    
                    # 缓存数据
                    self._add_to_cache(cache_key, df)
                    
                    return df
                else:
                    logger.warning(f"异步获取{ts_code}的日线数据为空")
                    # 使用本地备用数据
                    return await self._get_local_stock_data_async(ts_code, start_date, end_date)
            except Exception as e:
                logger.error(f"异步获取{ts_code}的日线数据失败: {str(e)}")
                return await self._get_local_stock_data_async(ts_code, start_date, end_date)
        else:
            return await self._get_local_stock_data_async(ts_code, start_date, end_date)
    
    async def _get_local_stock_data_async(self, ts_code, start_date=None, end_date=None):
        """异步从本地获取股票数据"""
        # 由于文件IO可能阻塞，使用线程池执行
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            lambda: self._get_local_stock_data(ts_code, start_date, end_date)
        )
    
    async def _fetch_multiple_stocks_async(self, stock_codes, start_date, end_date):
        """并发获取多只股票数据"""
        tasks = []
        for code in stock_codes:
            task = asyncio.create_task(self._fetch_stock_data_async(code, start_date, end_date))
            tasks.append(task)
        
        # 使用asyncio.gather并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        stock_data_dict = {}
        for code, result in zip(stock_codes, results):
            if isinstance(result, Exception):
                logger.error(f"获取{code}数据时出错: {str(result)}")
                continue
            if not isinstance(result, pd.DataFrame) or result.empty:
                logger.warning(f"获取{code}的数据为空或无效")
                continue
            stock_data_dict[code] = result
        
        return stock_data_dict
    
    def get_stock_daily_data_async(self, ts_code, start_date=None, end_date=None):
        """异步获取股票日线数据的公共接口"""
        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        # 运行异步协程
        return self._run_async(self._fetch_stock_data_async(ts_code, start_date, end_date))
    
    def get_multiple_stocks_data(self, stock_codes, start_date=None, end_date=None):
        """一次获取多只股票数据"""
        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        # 运行异步协程
        return self._run_async(self._fetch_multiple_stocks_async(stock_codes, start_date, end_date))
    
    def take_memory_snapshot(self):
        """记录当前内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'rss': memory_info.rss / (1024 * 1024),  # RSS in MB
            'vms': memory_info.vms / (1024 * 1024),  # VMS in MB
            'cache_size': len(self.data_cache),
            'system_used_percent': virtual_memory.percent,
            'cache_hits': self.cache_hit_stats['hits'],
            'cache_misses': self.cache_hit_stats['misses']
        }
        
        self.memory_usage_snapshots.append(snapshot)
        
        # 如果快照数量太多，只保留最近100个
        if len(self.memory_usage_snapshots) > 100:
            self.memory_usage_snapshots = self.memory_usage_snapshots[-100:]
        
        # 定期保存内存快照到文件
        if len(self.memory_usage_snapshots) % 10 == 0:
            self._save_memory_snapshots()
        
        # 根据内存使用情况决定是否需要清理
        if (virtual_memory.percent > self.memory_limit * 100 or 
            time.time() - self.last_cleanup_time > self.cleanup_interval):
            self.adaptive_cleanup()
    
    def _save_memory_snapshots(self):
        """保存内存使用快照到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(MEMORY_PROFILE_DIR, f'memory_profile_{timestamp}.json')
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.memory_usage_snapshots, f, indent=2)
            
            logger.debug(f"内存使用快照已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存内存快照失败: {str(e)}")
    
    def adaptive_cleanup(self):
        """自适应内存清理，根据系统负载动态调整清理力度"""
        logger.info("执行自适应内存清理")
        
        # 获取当前内存使用情况
        virtual_memory = psutil.virtual_memory()
        usage_percent = virtual_memory.percent
        
        # 根据内存使用比例调整清理力度
        if usage_percent > 90:  # 内存紧张
            # 保留非常少的缓存
            keep_ratio = 0.1  # 只保留10%热点数据
            logger.warning(f"内存使用率达到{usage_percent}%，进行紧急清理")
        elif usage_percent > 80:  # 内存偏紧
            keep_ratio = 0.3  # 保留30%热点数据
            logger.warning(f"内存使用率达到{usage_percent}%，进行中度清理")
        elif usage_percent > self.memory_limit * 100:  # 超过设定限制
            keep_ratio = 0.5  # 保留50%热点数据
            logger.info(f"内存使用率超过设定限制{self.memory_limit*100}%，进行轻度清理")
        else:  # 例行清理
            keep_ratio = 0.7  # 保留70%热点数据
            logger.debug("执行例行内存清理")
        
        # 清理LRU缓存
        cache_size_before = len(self.data_cache)
        items_to_keep = max(1, int(self.cache_size * keep_ratio))
        
        while len(self.data_cache) > items_to_keep:
            self.data_cache.popitem(last=False)
        
        # 清理过期的时间戳缓存
        expired_keys = []
        for key, (timestamp, _) in self.timestamp_cache.items():
            if (datetime.now() - timestamp).total_seconds() > self.cache_timeout:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.timestamp_cache[key]
        
        # 强制垃圾回收
        gc.collect()
        
        # 更新最后清理时间
        self.last_cleanup_time = time.time()
        
        # 记录清理结果
        logger.info(f"内存清理完成: 缓存从 {cache_size_before} 减少到 {len(self.data_cache)} 项, "
                   f"清理了 {len(expired_keys)} 个过期缓存项")
        
        # 清理后再次检查内存
        new_usage = psutil.virtual_memory().percent
        logger.info(f"清理前内存使用率: {usage_percent}%, 清理后: {new_usage}%, 减少了 {usage_percent - new_usage}%")
    
    async def prewarm_cache_async(self, stock_list, top_n=20):
        """异步预热缓存，提前并发加载常用股票数据"""
        if stock_list.empty or len(stock_list) == 0:
            logger.warning("股票列表为空，无法预热缓存")
            return
        
        # 如果列表太大，只预热一部分热门股票
        if top_n > 0 and len(stock_list) > top_n:
            # 这里可以根据交易量或其他指标选择最活跃的股票
            stock_subset = stock_list.head(top_n)
        else:
            stock_subset = stock_list
        
        # 记录预热状态
        self.prewarmed = True
        
        # 获取日期范围
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        # 获取要预热的股票代码列表
        stock_codes = [stock['ts_code'] for _, stock in stock_subset.iterrows()]
        self.prewarmed_codes.update(stock_codes)
        
        logger.info(f"开始异步预热缓存，加载 {len(stock_codes)} 支股票的数据")
        
        # 分批次并发获取数据，避免一次创建太多协程
        batch_size = min(10, len(stock_codes))  # 每批最多10只股票
        
        # 统计计时
        start_time = time.time()
        total_loaded = 0
        
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i+batch_size]
            
            logger.debug(f"预热批次 {i//batch_size + 1}: 加载 {len(batch_codes)} 只股票")
            
            # 并发获取当前批次的股票数据
            batch_data = await self._fetch_multiple_stocks_async(batch_codes, start_date, end_date)
            
            # 预计算技术指标 (可以在线程池中运行，避免阻塞事件循环)
            for code, data in batch_data.items():
                if not data.empty:
                    self.thread_pool.submit(self.calculate_momentum_vectorized, data)
                    total_loaded += 1
        
        end_time = time.time()
        logger.info(f"缓存预热完成，成功加载 {total_loaded}/{len(stock_codes)} 只股票的数据，耗时 {end_time-start_time:.2f} 秒")
    
    def prewarm_cache(self, stock_list, top_n=20):
        """预热缓存的公共接口"""
        return self._run_async(self.prewarm_cache_async(stock_list, top_n))
    
    def analyze_single_stock_async(self, stock_data):
        """异步版本的单只股票分析函数"""
        ts_code = stock_data['ts_code']
        name = stock_data['name']
        industry = stock_data.get('industry', '')
        min_score = stock_data.get('min_score', 60)
        
        # 自动检查内存使用情况
        self.take_memory_snapshot()
        
        try:
            logger.info(f"正在分析: {name}({ts_code})")
            
            # 检查是否预热过
            if self.prewarmed and ts_code in self.prewarmed_codes:
                logger.debug(f"{ts_code} 已预热，使用缓存数据")
            
            # 获取日线数据
            data = self.get_stock_daily_data_async(ts_code)
            if data.empty:
                logger.warning(f"无法获取{ts_code}的数据，跳过分析")
                return None
            
            # 使用向量化方法计算技术指标
            data = self.calculate_momentum_vectorized(data)
            if data.empty:
                logger.warning(f"计算{ts_code}的技术指标失败，跳过分析")
                return None
            
            # 使用优化的评分方法
            score, score_details = self.calculate_momentum_score_optimized(data)
            if score >= min_score:
                # 获取最新数据
                latest = data.iloc[-1]
                
                # 保存分析结果
                result = {
                    'ts_code': ts_code,
                    'name': name,
                    'industry': industry,
                    'close': latest['close'],
                    'momentum_20': latest.get('momentum_20', 0),
                    'momentum_20d': latest.get('momentum_20', 0),
                    'rsi': latest.get('rsi', 0),
                    'macd': latest.get('macd', 0),
                    'macd_hist': latest.get('macd_hist', 0),
                    'volume_ratio': latest.get('vol_ratio_20', 1),
                    'score': score,
                    'score_details': score_details,
                    'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'data': data
                }
                
                # 图表生成可选（可注释掉以提高性能）
                # chart_path = os.path.join(RESULTS_DIR, "charts", f"{ts_code}_momentum_async.png")
                # self.plot_stock_chart(data, ts_code, name, score_details, save_path=chart_path)
                
                return result
        except Exception as e:
            logger.error(f"分析{name}({ts_code})时出错: {str(e)}")
        
        return None
    
    async def analyze_stocks_async(self, stock_list, sample_size=100, min_score=60):
        """异步分析股票列表"""
        results = []
        
        # 确保股票列表不为空
        if stock_list.empty:
            logger.error("股票列表为空，无法进行分析")
            return results
        
        # 记录原始股票数量
        original_count = len(stock_list)
        logger.info(f"准备分析 {original_count} 支股票")
        
        # 限制样本大小
        if sample_size < len(stock_list):
            stock_list = stock_list.sample(sample_size)
            logger.info(f"从 {original_count} 支股票中随机选择 {sample_size} 支进行分析")
        else:
            logger.info(f"分析全部 {original_count} 支股票")
        
        total = len(stock_list)
        
        # 先进行缓存预热
        await self.prewarm_cache_async(stock_list)
        
        logger.info(f"开始异步分析 {total} 支股票")
        
        # 准备任务列表
        tasks = []
        for _, stock in stock_list.iterrows():
            stock_data = stock.to_dict()
            stock_data['min_score'] = min_score
            
            # 使用线程池处理分析任务
            future = self.thread_pool.submit(self.analyze_single_stock_async, stock_data)
            tasks.append(future)
        
        # 处理结果
        for future in tasks:
            result = future.result()
            if result is not None:
                results.append(result)
        
        # 按得分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"分析完成，符合条件的股票数量: {len(results)}")
        
        # 将结果保存为CSV
        if results:
            result_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'data' and k != 'score_details'}
                                     for r in results])
            csv_path = os.path.join(RESULTS_DIR, f"momentum_async_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"已将分析结果保存至: {csv_path}")
        
        return results
    
    def analyze_stocks(self, stock_list, sample_size=100, min_score=60):
        """分析股票列表的公共接口"""
        return self._run_async(self.analyze_stocks_async(stock_list, sample_size, min_score))

# 添加便捷的测试函数
def test_async_performance(sample_size=20):
    """测试异步性能"""
    print(f"开始异步性能测试，样本大小: {sample_size}")
    
    # 创建异步分析器
    analyzer = AsyncMomentumAnalyzer(use_tushare=True, use_multiprocessing=True)
    
    # 获取股票列表
    start_time = time.time()
    stocks = analyzer.get_stock_list()
    
    if stocks.empty or len(stocks) == 0:
        print("无法获取股票列表，测试终止")
        return None
    
    print(f"获取到 {len(stocks)} 支股票，耗时: {time.time() - start_time:.2f}秒")
    
    # 分析股票
    start_time = time.time()
    results = analyzer.analyze_stocks(stocks.head(sample_size), min_score=50)
    end_time = time.time()
    
    # 打印结果
    duration = end_time - start_time
    print(f"异步分析完成，耗时: {duration:.2f}秒，每只股票平均: {duration/sample_size:.2f}秒")
    print(f"符合条件的股票数量: {len(results)}")
    
    # 打印缓存统计
    print(f"缓存命中统计: 命中 {analyzer.cache_hit_stats['hits']}次，未命中 {analyzer.cache_hit_stats['misses']}次")
    
    # 输出内存使用情况
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"内存使用: RSS={memory_info.rss/(1024*1024):.2f}MB, VMS={memory_info.vms/(1024*1024):.2f}MB")
    
    return results, duration

if __name__ == "__main__":
    # 运行测试
    test_async_performance(sample_size=20) 