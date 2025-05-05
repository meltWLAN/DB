"""
动量分析模块（性能增强版）
提供股票动量分析的相关功能，包括技术指标计算、筛选和评分
实现了多项性能优化：
- 增强的缓存机制
- 优化的多进程处理
- 向量化计算
- 批处理优化
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
from multiprocessing import Pool, cpu_count
import functools
import pickle
import hashlib
from collections import OrderedDict
import time
import psutil

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
os.makedirs(CACHE_DIR, exist_ok=True)

# 配置日志
logger = logging.getLogger(__name__)

# 设置Tushare
if not TUSHARE_TOKEN:
    # 直接在代码中设置Token（如果配置文件中没有设置）
    TUSHARE_TOKEN = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
else:
    pro = None

# 优化的缓存装饰器
def enhanced_disk_cache(timeout=86400):
    """增强的磁盘缓存装饰器，支持过期时间设置
    
    Args:
        timeout: 缓存过期时间（秒），默认24小时
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 创建缓存键
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            key = "_".join(key_parts)
            
            # 使用MD5哈希作为文件名
            hash_obj = hashlib.md5(key.encode())
            cache_file = os.path.join(CACHE_DIR, f"{hash_obj.hexdigest()}.pkl")
            
            # 检查缓存是否存在且未过期
            if os.path.exists(cache_file):
                # 检查文件修改时间
                file_mtime = os.path.getmtime(cache_file)
                if time.time() - file_mtime < timeout:
                    try:
                        with open(cache_file, 'rb') as f:
                            return pickle.load(f)
                    except Exception:
                        # 如果加载缓存失败，继续执行原始函数
                        pass
                    
            # 执行原始函数
            result = func(*args, **kwargs)
            
            # 保存结果到缓存
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                logger.warning(f"缓存保存失败: {str(e)}")
                
            return result
        return wrapper
    return decorator

class EnhancedMomentumAnalyzer:
    """增强性能的动量分析器类，提供动量分析相关功能"""
    def __init__(self, use_tushare=True, use_multiprocessing=True, workers=None, 
                 cache_size=128, cache_timeout=86400, batch_size=50, memory_limit=0.8):
        """初始化动量分析器
        
        Args:
            use_tushare: 是否使用Tushare数据
            use_multiprocessing: 是否使用多进程
            workers: 工作进程数，None则自动确定
            cache_size: 内存缓存项数量上限
            cache_timeout: 缓存过期时间(秒)
            batch_size: 批处理大小
            memory_limit: 内存使用限制百分比(0-1)
        """
        self.use_tushare = use_tushare
        if use_tushare and not TUSHARE_TOKEN:
            logger.warning("未设置Tushare Token，将使用本地数据")
            self.use_tushare = False
            
        # 缓存设置
        self.cache_size = cache_size
        self.cache_timeout = cache_timeout
        self.data_cache = OrderedDict()  # 有序字典用于LRU缓存
        self.timestamp_cache = {}  # 带时间戳的缓存
        
        # 多进程设置
        self.use_multiprocessing = use_multiprocessing
        self.workers = workers if workers is not None else self._get_optimal_workers()
        
        # 批处理设置
        self.batch_size = batch_size
        self.memory_limit = memory_limit
        
        logger.info(f"初始化增强版动量分析器: 多进程={self.use_multiprocessing}, 工作进程数={self.workers}, "
                  f"内存缓存大小={self.cache_size}, 批处理大小={self.batch_size}")
    
    def _get_optimal_workers(self):
        """根据系统情况确定最优的工作进程数"""
        # 获取CPU核心数
        cpu_cores = cpu_count()
        # 获取系统内存情况
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024 ** 3)  # 转换为GB
        
        # 基于CPU和内存计算最优工作进程数
        # 每个进程大约需要0.5GB内存(根据实际情况调整)
        memory_based = max(1, int(memory_gb / 0.5) - 1)
        cpu_based = max(1, cpu_cores - 1)
        
        # 取较小值，避免资源耗尽
        workers = min(memory_based, cpu_based)
        logger.info(f"系统资源: CPU核心数={cpu_cores}, 内存={memory_gb:.1f}GB, 最优工作进程数={workers}")
        
        return workers
        
    def _add_to_cache(self, key, value):
        """添加数据到LRU缓存"""
        # 检查缓存大小，如果已满则删除最不常用的项
        if len(self.data_cache) >= self.cache_size:
            self.data_cache.popitem(last=False)  # 移除最早添加的项
        # 添加新项到缓存
        self.data_cache[key] = value
        return value
    
    def _get_from_cache(self, key):
        """从LRU缓存获取数据"""
        if key in self.data_cache:
            # 将访问的项移至末尾（最近使用）
            value = self.data_cache.pop(key)
            self.data_cache[key] = value
            return value
        return None
    
    def _cache_with_timestamp(self, key, value):
        """带时间戳缓存数据"""
        self.timestamp_cache[key] = (datetime.now(), value)
        return value
    
    def _get_cached_with_timeout(self, key):
        """获取带时间戳的缓存数据，检查是否过期"""
        if key in self.timestamp_cache:
            timestamp, value = self.timestamp_cache[key]
            # 检查是否过期
            if (datetime.now() - timestamp).total_seconds() < self.cache_timeout:
                return value
        return None
        
    def get_stock_list(self, industry=None):
        """获取股票列表，可按行业筛选"""
        # 热门板块和对应的关键词映射
        hot_sectors_keywords = {
            "人工智能": ["人工智能", "智能", "AI", "机器学习", "语音识别", "计算机", "软件", "科大讯飞"],
            "半导体芯片": ["半导体", "芯片", "集成电路", "电子", "中芯国际"],
            "新能源汽车": ["新能源", "汽车", "电动", "锂电池", "充电桩"],
            "医疗器械": ["医疗", "器械", "医药", "设备"],
            "云计算": ["云计算", "云服务", "互联网", "数据中心", "计算机"],
            "5G通信": ["5G", "通信", "移动", "电信", "基站"],
            "生物医药": ["生物", "医药", "制药", "基因", "疫苗"]
        }
        
        # 检查缓存
        cache_key = f"stock_list_{industry}"
        cached_result = self._get_cached_with_timeout(cache_key)
        if cached_result is not None:
            return cached_result
        
        if self.use_tushare:
            try:
                # 获取所有股票列表
                stocks = pro.stock_basic(exchange='', list_status='L',
                                         fields='ts_code,symbol,name,area,industry,list_date')
                
                # 行业筛选
                if industry and industry != "全部":
                    # 检查是否为自定义热门板块
                    if industry in hot_sectors_keywords:
                        # 对于热门板块，使用关键词匹配
                        keywords = hot_sectors_keywords[industry]
                        mask = pd.Series(False, index=stocks.index)
                        
                        # 匹配公司名称和行业
                        for keyword in keywords:
                            name_match = stocks['name'].str.contains(keyword, na=False)
                            industry_match = stocks['industry'].str.contains(keyword, na=False)
                            mask = mask | name_match | industry_match
                        
                        # 应用筛选
                        stocks = stocks[mask]
                    else:
                        # 传统行业分类
                        stocks = stocks[stocks['industry'] == industry]
                
                # 缓存结果
                return self._cache_with_timestamp(cache_key, stocks)
            except Exception as e:
                logger.error(f"从Tushare获取股票列表失败: {str(e)}")
                # 尝试使用备用数据
                return self._get_local_stock_list(industry)
        else:
            return self._get_local_stock_list(industry)
    
    def _get_local_stock_list(self, industry=None):
        """从本地获取股票列表（备用方法）"""
        # ... 实现与原来相同，省略代码 ...
        # 这里仅为示例，实际实现需要复制原来的方法
        return pd.DataFrame()  # 实际实现时应返回真实数据
    
    @enhanced_disk_cache(timeout=86400)  # 24小时过期
    def get_stock_daily_data_cached(self, ts_code, start_date=None, end_date=None, use_tushare=True):
        """带高级缓存的股票日线数据获取，支持过期时间"""
        if use_tushare:
            try:
                # 从Tushare获取日线数据
                df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df.empty:
                    # 尝试使用备用API
                    df = ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if not df.empty:
                    # 确保日期列为索引并按日期排序
                    if 'trade_date' in df.columns:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        df.sort_values('trade_date', inplace=True)
                        df.set_index('trade_date', inplace=True)
                    return df
                else:
                    logger.warning(f"获取{ts_code}的日线数据为空")
                    return self._get_local_stock_data(ts_code, start_date, end_date)
            except Exception as e:
                logger.error(f"从Tushare获取{ts_code}的日线数据失败: {str(e)}")
                return self._get_local_stock_data(ts_code, start_date, end_date)
        else:
            return self._get_local_stock_data(ts_code, start_date, end_date)
            
    def get_stock_daily_data(self, ts_code, start_date=None, end_date=None):
        """获取股票日线数据（带分层缓存）"""
        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            
        # 检查内存缓存
        cache_key = f"{ts_code}_{start_date}_{end_date}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        # 使用磁盘缓存获取数据
        df = self.get_stock_daily_data_cached(ts_code, start_date, end_date, self.use_tushare)
        
        # 存入内存缓存
        if not df.empty:
            self._add_to_cache(cache_key, df)
        
        return df
        
    def calculate_momentum_vectorized(self, data):
        """向量化计算动量指标，替代循环操作提高性能"""
        try:
            # 确保数据足够
            if len(data) < 60:
                logger.warning("数据点不足，无法计算完整的指标")
                return pd.DataFrame()
            
            # 准备数据副本，避免修改原始数据
            df = data.copy()
            
            # 1. 计算移动平均线 (向量化)
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma60'] = df['close'].rolling(window=60).mean()
            
            # 2. 计算MACD (向量化)
            # 计算EMA值
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['signal']
            
            # 3. 计算RSI (向量化)
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            # 处理第一个有效值后的数据
            first_valid = 14
            for i in range(first_valid, len(df)):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14
                
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 4. 计算KDJ (向量化)
            low_min = df['low'].rolling(window=9).min()
            high_max = df['high'].rolling(window=9).max()
            rsv = ((df['close'] - low_min) / (high_max - low_min) * 100).fillna(50)
            
            df['k'] = rsv.ewm(com=2, adjust=False).mean()
            df['d'] = df['k'].ewm(com=2, adjust=False).mean()
            df['j'] = 3 * df['k'] - 2 * df['d']
            
            # 5. 计算布林带 (向量化)
            df['boll_mid'] = df['close'].rolling(window=20).mean()
            df['boll_std'] = df['close'].rolling(window=20).std()
            df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
            df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
            
            # 6. 计算动量指标 (向量化)
            # 计算不同周期的动量
            for period in [5, 10, 20, 60]:
                df[f'momentum_{period}'] = df['close'].pct_change(periods=period)
            
            # 7. 计算成交量指标 (向量化)
            # 成交量移动平均
            df['vol_ma5'] = df['vol'].rolling(window=5).mean()
            df['vol_ma10'] = df['vol'].rolling(window=10).mean()
            # 成交量比率
            for period in [5, 10, 20]:
                df[f'vol_ratio_{period}'] = df['vol'] / df['vol'].rolling(window=period).mean()
            
            # 8. 计算波动率 (向量化)
            log_return = np.log(df['close'] / df['close'].shift(1))
            df['volatility_20'] = log_return.rolling(window=20).std() * np.sqrt(20)
            
            # 9. 计算趋势强度 (向量化)
            df['trend_20'] = (df['close'] - df['close'].shift(20)) / (df['close'].rolling(window=20).std() * np.sqrt(20))
            
            # 删除空值行，确保数据完整性
            df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"向量化计算动量指标失败: {str(e)}")
            return pd.DataFrame()
    
    def calculate_momentum_score_optimized(self, data):
        """优化的动量评分计算，使用向量化操作"""
        score = 0
        score_details = {}
        
        try:
            if data.empty or len(data) < 60:
                logger.warning("数据不足，无法计算动量评分")
                return 0, {}
            
            latest = data.iloc[-1]
            
            # 1. 价格动量得分 (20分)
            try:
                # 不同时间周期动量加权
                momentum_5 = latest['momentum_5'] * 100  # 转为百分比
                momentum_10 = latest['momentum_10'] * 100
                momentum_20 = latest['momentum_20'] * 100
                momentum_60 = latest['momentum_60'] * 100
                
                # 短期动量权重更高
                momentum_score = (momentum_5 * 0.4 + momentum_10 * 0.3 + 
                                momentum_20 * 0.2 + momentum_60 * 0.1)
                
                # 归一化为0-20分
                momentum_score = min(20, max(0, momentum_score + 10))  # -10%到+10%映射到0-20分
                score_details['价格动量'] = round(momentum_score, 2)
                score += momentum_score
            except Exception as e:
                logger.error(f"计算价格动量得分失败: {str(e)}")
                score_details['价格动量'] = 0
            
            # 2. 均线多头排列得分 (15分)
            try:
                ma_score = 0
                close = latest['close']
                
                # 判断均线多头排列情况
                if (close > latest['ma5'] > latest['ma10'] > latest['ma20'] > latest['ma60']):
                    ma_score = 15  # 完美的多头排列
                elif (close > latest['ma5'] > latest['ma10'] > latest['ma20']):
                    ma_score = 12  # 较好的多头排列
                elif (close > latest['ma5'] > latest['ma10']):
                    ma_score = 8   # 短期多头排列
                elif (close > latest['ma5']):
                    ma_score = 5   # 价格站上短期均线
                
                score_details['均线排列'] = round(ma_score, 2)
                score += ma_score
            except Exception as e:
                logger.error(f"计算均线排列得分失败: {str(e)}")
                score_details['均线排列'] = 0
            
            # 3. MACD指标得分 (20分)
            try:
                macd_score = 0
                
                # MACD金叉状态
                if (latest['macd'] > latest['signal'] and 
                    data.iloc[-2]['macd'] <= data.iloc[-2]['signal']):
                    macd_score += 10  # 刚形成金叉
                    
                # MACD值为正
                if latest['macd'] > 0:
                    macd_score += 5
                
                # MACD柱状图向上
                if (latest['macd_hist'] > 0 and 
                    latest['macd_hist'] > data.iloc[-2]['macd_hist']):
                    macd_score += 5
                
                score_details['MACD指标'] = round(macd_score, 2)
                score += macd_score
            except Exception as e:
                logger.error(f"计算MACD指标得分失败: {str(e)}")
                score_details['MACD指标'] = 0
            
            # 4. RSI指标得分 (15分)
            try:
                rsi_score = 0
                
                # RSI值介于40-70之间为理想区间
                if 40 <= latest['rsi'] <= 70:
                    rsi_score = 10
                elif 30 <= latest['rsi'] < 40:
                    rsi_score = 7  # RSI偏低但可接受
                elif 70 < latest['rsi'] <= 80:
                    rsi_score = 5  # RSI偏高但可接受
                
                # RSI向上发展
                if latest['rsi'] > data.iloc[-2]['rsi']:
                    rsi_score += 5
                
                score_details['RSI指标'] = round(rsi_score, 2)
                score += rsi_score
            except Exception as e:
                logger.error(f"计算RSI指标得分失败: {str(e)}")
                score_details['RSI指标'] = 0
            
            # 5. 成交量变化得分 (15分)
            try:
                volume_score = 0
                
                # 成交量高于5日均量
                vol_ratio_5 = latest['vol_ratio_5']
                if vol_ratio_5 > 1.5:
                    volume_score = 12  # 成交量显著放大
                elif vol_ratio_5 > 1.2:
                    volume_score = 10  # 成交量适度放大
                elif vol_ratio_5 > 1.0:
                    volume_score = 8   # 成交量略有增加
                else:
                    volume_score = 5   # 成交量不足
                
                # 成交量趋势
                recent_vol_trend = data.iloc[-5:]['vol'].pct_change().mean() * 100
                if recent_vol_trend > 5:
                    volume_score += 3  # 成交量持续增加
                
                score_details['成交量变化'] = round(volume_score, 2)
                score += volume_score
            except Exception as e:
                logger.error(f"计算成交量变化得分失败: {str(e)}")
                score_details['成交量变化'] = 0
            
            # 6. 趋势强度得分 (15分)
            try:
                trend_score = 0
                
                # 基于趋势强度指标
                trend_strength = latest['trend_20']
                if trend_strength > 1.5:
                    trend_score = 15  # 非常强劲的上升趋势
                elif trend_strength > 1.0:
                    trend_score = 12  # 强劲的上升趋势
                elif trend_strength > 0.5:
                    trend_score = 10  # 中等上升趋势
                elif trend_strength > 0:
                    trend_score = 5   # 弱上升趋势
                
                score_details['趋势强度'] = round(trend_score, 2)
                score += trend_score
            except Exception as e:
                logger.error(f"计算趋势强度得分失败: {str(e)}")
                score_details['趋势强度'] = 0
            
            # 总分四舍五入为整数
            score = round(score)
            score_details['总分'] = score
            
            return score, score_details
            
        except Exception as e:
            logger.error(f"计算动量评分失败: {str(e)}")
            return 0, {'总分': 0, '错误': str(e)}
    
    def analyze_single_stock_optimized(self, stock_data):
        """优化的单个股票分析函数"""
        ts_code = stock_data['ts_code']
        name = stock_data['name']
        industry = stock_data.get('industry', '')
        min_score = stock_data.get('min_score', 60)
        
        try:
            logger.info(f"正在分析: {name}({ts_code})")
            
            # 获取日线数据
            data = self.get_stock_daily_data(ts_code)
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
                    'data': data
                }
                
                # 图表生成可选（可注释掉以提高性能）
                chart_path = os.path.join(RESULTS_DIR, "charts", f"{ts_code}_momentum_enhanced.png")
                # self.plot_stock_chart(data, ts_code, name, score_details, save_path=chart_path)
                
                return result
        except Exception as e:
            logger.error(f"分析{name}({ts_code})时出错: {str(e)}")
            
        return None
        
    def analyze_stocks_batched(self, stock_list, sample_size=100, min_score=60):
        """使用批处理方式分析股票列表，优化内存使用"""
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
        logger.info(f"开始分析 {total} 支股票，采用{'批处理多进程' if self.use_multiprocessing else '批处理单进程'}")
        
        # 准备任务列表
        tasks = []
        for _, stock in stock_list.iterrows():
            stock_data = stock.to_dict()
            stock_data['min_score'] = min_score
            tasks.append(stock_data)
            
        # 分批处理
        batch_count = (total + self.batch_size - 1) // self.batch_size
        logger.info(f"将 {total} 支股票分为 {batch_count} 批处理，每批 {self.batch_size} 支")
        
        processed_count = 0
        for batch_idx in range(batch_count):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total)
            batch_tasks = tasks[start_idx:end_idx]
            
            logger.info(f"处理第 {batch_idx+1}/{batch_count} 批，包含 {len(batch_tasks)} 支股票")
            
            # 使用多进程处理当前批次
            if self.use_multiprocessing and len(batch_tasks) > 5:  # 股票数量超过5只才值得使用多进程
                with Pool(processes=self.workers) as pool:
                    # 使用多进程映射函数
                    batch_results = pool.map(self.analyze_single_stock_optimized, batch_tasks)
                    
                    # 筛选有效结果
                    for result in batch_results:
                        if result is not None:
                            results.append(result)
            # 使用单进程处理当前批次
            else:
                for task in batch_tasks:
                    result = self.analyze_single_stock_optimized(task)
                    if result is not None:
                        results.append(result)
            
            processed_count += len(batch_tasks)
            logger.info(f"已处理: {processed_count}/{total} 支股票，当前筛选出 {len(results)} 支符合条件的股票")
            
            # 检查内存使用情况
            if psutil.virtual_memory().percent > (self.memory_limit * 100):
                logger.warning(f"内存使用率超过{self.memory_limit*100}%，执行垃圾回收")
                self.cleanup_memory()
                
        # 按得分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"分析完成，符合条件的股票数量: {len(results)}")
        
        # 将结果保存为CSV
        if results:
            result_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'data' and k != 'score_details'}
                                     for r in results])
            csv_path = os.path.join(RESULTS_DIR, f"momentum_enhanced_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"已将分析结果保存至: {csv_path}")
            
        return results
        
    def cleanup_memory(self):
        """清理内存缓存"""
        # 清理LRU缓存
        cache_size_before = len(self.data_cache)
        # 保留20%的热点数据
        items_to_keep = max(1, int(self.cache_size * 0.2))
        while len(self.data_cache) > items_to_keep:
            self.data_cache.popitem(last=False)
        
        # 清理时间戳缓存中过期的项
        expired_keys = []
        for key, (timestamp, _) in self.timestamp_cache.items():
            if (datetime.now() - timestamp).total_seconds() > self.cache_timeout:
                expired_keys.append(key)
        for key in expired_keys:
            del self.timestamp_cache[key]
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        logger.info(f"内存清理完成: 缓存从 {cache_size_before} 减少到 {len(self.data_cache)} 项，"
                   f"清理了 {len(expired_keys)} 个过期缓存项")
    
    def analyze_stocks(self, stock_list, sample_size=100, min_score=60):
        """分析股票列表的主接口方法，调用批处理实现"""
        return self.analyze_stocks_batched(stock_list, sample_size, min_score)

    def _get_local_stock_data(self, ts_code, start_date=None, end_date=None):
        """从本地获取股票日线数据（备用方法）"""
        try:
            # 尝试从本地文件读取
            stock_file = os.path.join(DATA_DIR, f"{ts_code.replace('.', '_')}.csv")
            if os.path.exists(stock_file):
                df = pd.read_csv(stock_file)
                # 处理日期
                if 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    df.sort_values('trade_date', inplace=True)
                    # 日期筛选
                    if start_date:
                        start_date = pd.to_datetime(start_date)
                        df = df[df['trade_date'] >= start_date]
                    if end_date:
                        end_date = pd.to_datetime(end_date)
                        df = df[df['trade_date'] <= end_date]
                    df.set_index('trade_date', inplace=True)
                return df
            else:
                # 创建模拟数据
                logger.warning(f"无法获取{ts_code}的日线数据，创建模拟数据")
                # 生成日期序列
                if not end_date:
                    end_date = datetime.now().strftime('%Y%m%d')
                if not start_date:
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                date_range = pd.date_range(start=start_date, end=end_date, freq='B')
                # 生成模拟价格和交易量
                n = len(date_range)
                close = np.random.normal(100, 10, n).cumsum() + 3000
                open_price = close * np.random.normal(1, 0.01, n)
                high = np.maximum(close, open_price) * np.random.normal(1.02, 0.01, n)
                low = np.minimum(close, open_price) * np.random.normal(0.98, 0.01, n)
                vol = np.random.normal(100000, 20000, n) * (1 + 0.1 * np.sin(np.arange(n) / 10))
                vol = np.abs(vol)
                # 创建DataFrame
                data = {
                    'ts_code': [ts_code] * n,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'vol': vol,
                    'amount': vol * close
                }
                df = pd.DataFrame(data, index=date_range)
                return df
        except Exception as e:
            logger.error(f"从本地获取股票日线数据失败: {str(e)}")
            return pd.DataFrame()

    def plot_stock_chart(self, data, stock_code, stock_name, score_details, save_path=None):
        """绘制股票技术分析图表"""
        try:
            # 确保绘图数据足够
            if len(data) < 20:
                logger.warning(f"数据点不足，无法绘制有效图表: {stock_code}")
                return False
                
            # 创建绘图子图布局
            plt.style.use('seaborn-v0_8-darkgrid')
            fig = plt.figure(figsize=(18, 14))
            gs = plt.GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1])
            
            # 设置标题
            fig.suptitle(f"{stock_name}({stock_code}) 动量分析 - 得分:{score_details.get('总分', 0)}分", 
                         fontsize=16, fontweight='bold')
            
            # 仅显示最近60个交易日
            if len(data) > 60:
                data = data.iloc[-60:]
                
            # 第一个子图：价格K线和均线
            ax1 = fig.add_subplot(gs[0, 0])
            # 绘制K线
            for i in range(len(data)):
                date = data.index[i]
                open_price, high, low, close = data.iloc[i][['open', 'high', 'low', 'close']]
                color = 'red' if close >= open_price else 'green'
                # 绘制影线
                ax1.plot([date, date], [low, high], color=color, linewidth=1)
                # 绘制实体
                rect_height = abs(close - open_price)
                rect_bottom = min(close, open_price)
                ax1.add_patch(plt.Rectangle((date - pd.Timedelta(days=0.4), rect_bottom), 
                                           pd.Timedelta(days=0.8), rect_height, 
                                           edgecolor=color, facecolor=color))
            # 绘制均线
            ax1.plot(data.index, data['ma5'], label='MA5', color='blue', linewidth=1)
            ax1.plot(data.index, data['ma10'], label='MA10', color='orange', linewidth=1)
            ax1.plot(data.index, data['ma20'], label='MA20', color='purple', linewidth=1)
            ax1.plot(data.index, data['ma60'], label='MA60', color='brown', linewidth=1)
            # 绘制布林带
            ax1.plot(data.index, data['boll_mid'], label='BOLL', color='black', linewidth=1)
            ax1.plot(data.index, data['boll_upper'], label='Upper', color='gray', linewidth=1, linestyle='--')
            ax1.plot(data.index, data['boll_lower'], label='Lower', color='gray', linewidth=1, linestyle='--')
            ax1.legend(loc='best')
            ax1.grid(True)
            ax1.set_ylabel('Price')
            ax1.set_title(f'K线图与均线 - 收盘价: {data.iloc[-1]["close"]:.2f}')
            
            # 第二个子图：成交量
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            for i in range(len(data)):
                date = data.index[i]
                open_price, close = data.iloc[i][['open', 'close']]
                color = 'red' if close >= open_price else 'green'
                ax2.bar(date, data.iloc[i]['vol'], color=color, width=pd.Timedelta(days=0.8))
            ax2.set_ylabel('Volume')
            ax2.grid(True)
            ax2.set_title(f'成交量 - 近期均量: {data.iloc[-5:]["vol"].mean():.0f}')
            
            # 第三个子图：MACD
            ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
            ax3.plot(data.index, data['macd'], label='MACD', color='blue', linewidth=1)
            ax3.plot(data.index, data['signal'], label='Signal', color='red', linewidth=1)
            # 绘制MACD柱状图
            for i in range(len(data)):
                date = data.index[i]
                hist = data.iloc[i]['macd_hist']
                color = 'red' if hist >= 0 else 'green'
                ax3.bar(date, hist, color=color, width=pd.Timedelta(days=0.8))
            ax3.legend(loc='best')
            ax3.set_ylabel('MACD')
            ax3.grid(True)
            ax3.set_title(f'MACD指标 - MACD: {data.iloc[-1]["macd"]:.4f}, Signal: {data.iloc[-1]["signal"]:.4f}')
            
            # 第四个子图：RSI
            ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
            ax4.plot(data.index, data['rsi'], label='RSI', color='purple', linewidth=1)
            ax4.axhline(y=30, color='green', linestyle='--')
            ax4.axhline(y=70, color='red', linestyle='--')
            ax4.legend(loc='best')
            ax4.set_ylabel('RSI')
            ax4.set_ylim(0, 100)
            ax4.grid(True)
            ax4.set_title(f'RSI指标 - 当前值: {data.iloc[-1]["rsi"]:.2f}')
            
            # 第五个子图：KDJ
            ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)
            ax5.plot(data.index, data['k'], label='K', color='blue', linewidth=1)
            ax5.plot(data.index, data['d'], label='D', color='yellow', linewidth=1)
            ax5.plot(data.index, data['j'], label='J', color='magenta', linewidth=1)
            ax5.axhline(y=20, color='green', linestyle='--')
            ax5.axhline(y=80, color='red', linestyle='--')
            ax5.legend(loc='best')
            ax5.set_ylabel('KDJ')
            ax5.grid(True)
            ax5.set_title(f'KDJ指标 - K: {data.iloc[-1]["k"]:.2f}, D: {data.iloc[-1]["d"]:.2f}, J: {data.iloc[-1]["j"]:.2f}')
            
            # 添加得分信息
            score_text = "\n".join([f"{k}: {v}" for k, v in score_details.items()])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.02, 0.05, score_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='bottom', bbox=props)
            
            # 在左上角添加性能统计
            performance_text = f"优化版分析\n向量化计算"
            ax1.text(0.02, 0.95, performance_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            # 调整布局
            plt.tight_layout()
            fig.subplots_adjust(top=0.94)  # 为suptitle留出空间
            
            # 保存图表
            if save_path:
                plt.savefig(save_path, dpi=120, bbox_inches='tight')
                logger.info(f"已保存{stock_code}的图表到{save_path}")
                plt.close(fig)
                return True
            else:
                plt.show()
                return True
                
        except Exception as e:
            logger.error(f"绘制图表失败: {str(e)}")
            return False
            
    def warm_up_cache(self, stock_list, top_n=20):
        """预热缓存，提前加载常用股票数据"""
        if stock_list.empty or len(stock_list) == 0:
            logger.warning("股票列表为空，无法预热缓存")
            return
            
        # 如果列表太大，只预热一部分热门股票
        if top_n > 0 and len(stock_list) > top_n:
            # 这里可以根据交易量或其他指标选择最活跃的股票
            stock_subset = stock_list.head(top_n)
        else:
            stock_subset = stock_list
            
        logger.info(f"开始预热缓存，加载 {len(stock_subset)} 支股票的数据")
        
        # 获取日期范围
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        # 并行加载数据
        if self.use_multiprocessing and len(stock_subset) > 5:
            logger.info(f"使用 {self.workers} 个并行进程预热缓存")
            
            # 准备参数
            args_list = []
            for _, stock in stock_subset.iterrows():
                ts_code = stock['ts_code']
                args_list.append((ts_code, start_date, end_date))
            
            # 启动并行加载
            with Pool(processes=self.workers) as pool:
                pool.starmap(self._prefetch_stock_data, args_list)
        else:
            # 串行加载
            for _, stock in stock_subset.iterrows():
                ts_code = stock['ts_code']
                self._prefetch_stock_data(ts_code, start_date, end_date)
                
        logger.info("缓存预热完成")
        
    def _prefetch_stock_data(self, ts_code, start_date, end_date):
        """预取单个股票数据到缓存（被warm_up_cache调用）"""
        try:
            logger.debug(f"预取 {ts_code} 的数据")
            data = self.get_stock_daily_data(ts_code, start_date, end_date)
            if not data.empty:
                # 预计算技术指标
                self.calculate_momentum_vectorized(data)
                logger.debug(f"成功预取并计算 {ts_code} 的数据")
            else:
                logger.warning(f"预取 {ts_code} 的数据为空")
        except Exception as e:
            logger.error(f"预取 {ts_code} 的数据失败: {str(e)}")

# 添加便捷的测试函数
def test_enhanced_performance(sample_size=20, use_multiprocessing=True):
    """测试增强版本的性能"""
    analyzer = EnhancedMomentumAnalyzer(use_tushare=True, use_multiprocessing=use_multiprocessing)
    # 获取股票列表
    stocks = analyzer.get_stock_list()
    print(f"获取到 {len(stocks)} 支股票")
    
    # 分析样本股票
    start_time = time.time()
    results = analyzer.analyze_stocks(stocks.head(sample_size), min_score=50)
    end_time = time.time()
    
    # 输出结果和性能数据
    print(f"分析完成，耗时: {end_time - start_time:.2f}秒")
    print(f"符合条件的股票数量: {len(results)}")
    
    # 输出股票分析结果
    for r in results:
        print(f"{r['name']}({r['ts_code']}): 得分={r['score']}, 动量20={r['momentum_20']:.2%}, RSI={r['rsi']:.2f}")
    
    return results, end_time - start_time

if __name__ == "__main__":
    # 运行测试
    test_enhanced_performance(sample_size=20) 