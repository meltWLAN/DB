"""
动量分析模块
提供股票动量分析的相关功能，包括技术指标计算、筛选和评分
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import logging
from datetime import datetime, timedelta
from pathlib import Path
import tushare as ts
import warnings
import random
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
# 配置日志
logger = logging.getLogger(__name__)
# 设置Tushare
if not TUSHARE_TOKEN:
    # 直接在代码中设置Token（如果配置文件中没有设置）
    TUSHARE_TOKEN = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
    # 直接使用token初始化Tushare API
    pro = ts.pro_api("0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10")
else:
    pro = None
class MomentumAnalyzer:
    """动量分析器类，提供动量分析相关功能"""
    def __init__(self, use_tushare=True, lookback_period=60, use_parallel=True, cache_limit=128):
        """初始化动量分析器
        
        Args:
            use_tushare: 是否使用Tushare API
            lookback_period: 回溯分析期限，默认60天
            use_parallel: 是否使用并行处理
            cache_limit: 缓存数据条目限制
        """
        # 基本设置
        self.use_tushare = use_tushare
        if use_tushare and not TUSHARE_TOKEN:
            logger.warning("未设置Tushare Token，将使用本地数据")
            self.use_tushare = False
        self.lookback_period = lookback_period
        self.use_parallel = use_parallel
        self.data_cache = {}  # 数据缓存
        self._cache_keys = []  # 缓存键列表，用于LRU缓存管理
        self._cache_limit = cache_limit  # 缓存大小限制
        
        # 设置日期范围
        end_date = datetime.now().strftime('%Y%m%d')
        days_ago = 365  # 默认获取一年的数据
        start_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y%m%d')
        self.start_date = start_date
        self.end_date = end_date
        
        # 设置logger
        self.logger = logger
        
        # 自定义分析阈值和参数
        self.momentum_thresholds = {
            'short': 5,     # 短期动量周期
            'medium': 20,   # 中期动量周期
            'long': 60      # 长期动量周期
        }
        
        # 配置不同指标的权重
        self.indicator_weights = {
            'ma_score': 0.15,       # 移动平均线权重
            'momentum_score': 0.25, # 价格动量权重
            'rsi_score': 0.15,      # RSI权重
            'macd_score': 0.15,     # MACD权重
            'kdj_score': 0.15,      # KDJ权重
            'volume_score': 0.10,   # 成交量权重
            'price_pattern': 0.05   # 价格形态权重
        }
        
        # 初始化并行处理池
        self._init_parallel_pool()
    
    def _init_parallel_pool(self):
        """初始化并行处理池"""
        if self.use_parallel:
            try:
                import multiprocessing as mp
                self.pool = mp.Pool(processes=max(1, mp.cpu_count() - 1))
            except ImportError:
                logger.warning("无法导入multiprocessing模块，将禁用并行处理")
                self.use_parallel = False
    
    def _get_cached_data(self, key):
        """从缓存获取数据"""
        if key in self.data_cache:
            # 更新缓存访问顺序
            self._cache_keys.remove(key)
            self._cache_keys.append(key)
            return self.data_cache[key]
        return None
    
    def _set_cached_data(self, key, data):
        """将数据存入缓存"""
        # 如果缓存满了，移除最久未使用的条目
        if len(self._cache_keys) >= self._cache_limit and self._cache_keys:
            oldest_key = self._cache_keys.pop(0)
            if oldest_key in self.data_cache:
                del self.data_cache[oldest_key]
        
        # 添加新条目
        if key not in self.data_cache:
            self._cache_keys.append(key)
        self.data_cache[key] = data
        return data

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
        
        logger.info(f"获取股票列表, 行业: {industry}, 使用Tushare: {self.use_tushare}")
        
        # 检查是否已有缓存
        cache_key = f"stock_list_{industry if industry else 'all'}"
        cache_data = self._get_cached_data(cache_key)
        if cache_data is not None:
            logger.info(f"使用缓存的股票列表, 数量: {len(cache_data)}")
            return cache_data
        
        if self.use_tushare:
            try:
                # 确保pro是全局变量
                global pro
                
                # 如果pro为None，尝试重新初始化
                if pro is None:
                    logger.warning("Tushare pro API未初始化，尝试重新初始化")
                    try:
                        ts.set_token(TUSHARE_TOKEN)
                        pro = ts.pro_api()
                        logger.info(f"Tushare API重新初始化成功 (Token前5位: {TUSHARE_TOKEN[:5]}...)")
                    except Exception as e:
                        logger.error(f"Tushare API重新初始化失败: {str(e)}")
                        pro = None
                
                if pro:
                    # 获取所有股票列表
                    logger.info("开始从Tushare获取股票列表")
                    stocks = pro.stock_basic(exchange='', list_status='L',
                                            fields='ts_code,symbol,name,area,industry,list_date')
                    
                    if stocks is None or stocks.empty:
                        logger.warning("Tushare返回的股票列表为空，将使用本地数据")
                        return self._get_local_stock_list(industry)
                    
                    logger.info(f"成功从Tushare获取股票列表, 数量: {len(stocks)}")
                    
                    # 行业筛选
                    if industry and industry != "全部":
                        # 检查是否为自定义热门板块
                        if industry in hot_sectors_keywords:
                            # 对于热门板块，使用关键词匹配
                            logger.info(f"使用关键词匹配筛选热门板块: {industry}")
                            keywords = hot_sectors_keywords[industry]
                            mask = pd.Series(False, index=stocks.index)
                            
                            # 匹配公司名称和行业
                            for keyword in keywords:
                                name_match = stocks['name'].str.contains(keyword, na=False)
                                industry_match = stocks['industry'].str.contains(keyword, na=False)
                                mask = mask | name_match | industry_match
                            
                            # 应用筛选
                            filtered_stocks = stocks[mask]
                            logger.info(f"热门板块'{industry}'筛选后，股票数量: {len(filtered_stocks)}")
                            
                            # 如果筛选后没有股票，返回原始列表
                            if filtered_stocks.empty:
                                logger.warning(f"热门板块'{industry}'筛选后股票列表为空，返回全部股票")
                                filtered_stocks = stocks
                        else:
                            # 传统行业分类
                            logger.info(f"使用传统行业分类筛选: {industry}")
                            filtered_stocks = stocks[stocks['industry'] == industry]
                            logger.info(f"行业'{industry}'筛选后，股票数量: {len(filtered_stocks)}")
                            
                            # 如果筛选后没有股票，返回原始列表
                            if filtered_stocks.empty:
                                logger.warning(f"行业'{industry}'筛选后股票列表为空，返回全部股票")
                                filtered_stocks = stocks
                    else:
                        filtered_stocks = stocks
                    
                    # 缓存结果
                    self._set_cached_data(cache_key, filtered_stocks)
                    
                    return filtered_stocks
                else:
                    logger.warning("Tushare API未初始化，使用本地数据")
                    return self._get_local_stock_list(industry)
            except Exception as e:
                logger.error(f"从Tushare获取股票列表失败: {str(e)}")
                # 尝试使用备用数据
                logger.info("尝试使用本地备用数据")
                return self._get_local_stock_list(industry)
        else:
            logger.info("配置为不使用Tushare，使用本地数据")
            return self._get_local_stock_list(industry)
    def _get_local_stock_list(self, industry=None):
        """从本地获取股票列表（备用方法）"""
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
        
        logger.info("从本地获取股票列表")
        
        try:
            # 尝试从本地文件读取
            stock_file = os.path.join(DATA_DIR, "stock_list.csv")
            
            # 检查文件是否存在并且不为空
            file_exists = os.path.exists(stock_file)
            file_not_empty = False
            
            if file_exists:
                try:
                    file_not_empty = os.path.getsize(stock_file) > 100  # 检查文件至少有100字节
                except:
                    file_not_empty = False
            
            if file_exists and file_not_empty:
                logger.info(f"读取本地股票列表文件: {stock_file}")
                
                try:
                    stocks = pd.read_csv(stock_file)
                    
                    if stocks.empty:
                        logger.warning("本地股票列表文件为空，将生成模拟数据")
                        stocks = self._generate_mock_stock_list()
                    else:
                        logger.info(f"成功从本地文件加载股票列表，股票数量: {len(stocks)}")
                except Exception as e:
                    logger.error(f"读取本地股票列表文件错误: {str(e)}")
                    stocks = self._generate_mock_stock_list()
            else:
                # 如果文件不存在或为空，则生成并保存模拟数据
                logger.info("本地股票列表文件不存在或为空，生成模拟数据")
                stocks = self._generate_mock_stock_list()
                
                # 保存到本地文件
                try:
                    # 确保目录存在
                    os.makedirs(os.path.dirname(stock_file), exist_ok=True)
                    stocks.to_csv(stock_file, index=False)
                    logger.info(f"保存模拟股票列表到本地文件: {stock_file}")
                except Exception as e:
                    logger.error(f"保存模拟股票列表到本地文件失败: {str(e)}")
            
            # 行业筛选
            if industry and industry != "全部":
                # 检查是否为自定义热门板块
                if industry in hot_sectors_keywords:
                    # 对于热门板块，使用关键词匹配
                    logger.info(f"使用关键词匹配本地数据筛选热门板块: {industry}")
                    keywords = hot_sectors_keywords[industry]
                    mask = pd.Series(False, index=stocks.index)
                    
                    # 匹配公司名称和行业
                    for keyword in keywords:
                        name_match = stocks['name'].str.contains(keyword, na=False)
                        if 'industry' in stocks.columns:
                            industry_match = stocks['industry'].str.contains(keyword, na=False)
                            mask = mask | name_match | industry_match
                        else:
                            mask = mask | name_match
                    
                    # 应用筛选
                    filtered_stocks = stocks[mask]
                    logger.info(f"热门板块'{industry}'筛选后，本地股票数量: {len(filtered_stocks)}")
                    
                    # 如果筛选后没有股票，返回原始列表
                    if filtered_stocks.empty:
                        logger.warning(f"热门板块'{industry}'筛选后本地股票列表为空，返回全部股票")
                        filtered_stocks = stocks
                else:
                    # 传统行业分类
                    if 'industry' in stocks.columns:
                        logger.info(f"使用传统行业分类筛选本地数据: {industry}")
                        filtered_stocks = stocks[stocks['industry'] == industry]
                        logger.info(f"行业'{industry}'筛选后，本地股票数量: {len(filtered_stocks)}")
                        
                        # 如果筛选后没有股票，返回原始列表
                        if filtered_stocks.empty:
                            logger.warning(f"行业'{industry}'筛选后本地股票列表为空，返回全部股票")
                            filtered_stocks = stocks
                    else:
                        logger.warning("本地股票数据没有industry列，无法按传统行业筛选")
                        filtered_stocks = stocks
            else:
                filtered_stocks = stocks
            
            return filtered_stocks
            
        except Exception as e:
            logger.error(f"获取本地股票列表出错: {str(e)}")
            # 发生异常时生成模拟数据
            return self._generate_mock_stock_list()
    
    def _generate_mock_stock_list(self):
        """生成模拟股票列表数据"""
        logger.info("生成模拟股票列表")
        mock_stocks = []
        
        # 模拟股票代码前缀
        prefixes = ['000', '002', '300', '600', '601', '603', '688']
        
        # 模拟行业列表
        industries = [
            "银行", "保险", "证券", "房地产", "医药生物", "计算机", "通信", 
            "电子", "传媒", "汽车", "食品饮料", "家用电器", "建筑材料", 
            "电力设备", "机械设备", "钢铁", "煤炭", "石油石化", "有色金属", 
            "化工", "纺织服装", "农林牧渔", "商业贸易", "休闲服务"
        ]
        
        # 为每个前缀生成一些股票
        for prefix in prefixes:
            for i in range(1, 31):  # 每个前缀生成30只股票
                code = f"{prefix}{i:04d}"
                
                # 随机选择交易所后缀
                suffix = ".SZ" if prefix in ['000', '002', '300'] else ".SH"
                ts_code = f"{code}{suffix}"
                
                # 随机选择行业
                industry = random.choice(industries)
                
                # 股票名称
                name = f"模拟股票{code}"
                
                # 股票属性
                stock = {
                    'ts_code': ts_code,
                    'symbol': code,
                    'name': name,
                    'area': random.choice(['北京', '上海', '深圳', '广州', '杭州']),
                    'industry': industry,
                    'list_date': f'20{random.randint(10, 23)}{random.randint(1, 12):02d}{random.randint(1, 28):02d}'
                }
                
                mock_stocks.append(stock)
        
        # 转换为DataFrame
        df = pd.DataFrame(mock_stocks)
        logger.info(f"生成的模拟股票列表包含 {len(df)} 只股票")
        return df
    def get_stock_daily_data(self, ts_code, start_date=None, end_date=None):
        """
        获取股票日线数据
        
        参数:
            ts_code: 股票代码
            start_date: 开始日期，格式：YYYYMMDD，默认为None，将使用实例化时的开始日期
            end_date: 结束日期，格式：YYYYMMDD，默认为None，将使用实例化时的结束日期
            
        返回:
            包含股票日线数据的DataFrame
        """
        # 设置默认日期
        start_date = start_date if start_date is not None else self.start_date
        end_date = end_date if end_date is not None else self.end_date
        
        # 从Tushare获取数据
        try:
            self.logger.info(f"从Tushare获取{ts_code}日线数据")
            
            # 直接初始化Tushare API (和__main__中保持一致的方式)
            USER_TOKEN = '0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10'
            ts.set_token(USER_TOKEN)
            direct_pro_api = ts.pro_api()
            
            if direct_pro_api is None:
                self.logger.error("无法直接初始化Tushare Pro API实例")
                return self._get_local_stock_data(ts_code, start_date, end_date)
            
            # 尝试使用daily获取
            df = direct_pro_api.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            # 如果daily没有数据，尝试创建模拟数据
            if df is None or df.empty:
                self.logger.warning(f"通过daily获取{ts_code}数据为空，使用备用方法")
                return self._get_local_stock_data(ts_code, start_date, end_date)
            
            if df is not None and not df.empty:
                # 预处理数据
                if 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    df.sort_values('trade_date', inplace=True)
                    df.set_index('trade_date', inplace=True)
                
                return df
            else:
                self.logger.warning(f"从Tushare获取{ts_code}数据失败，尝试使用备用方法")
                return self._get_local_stock_data(ts_code, start_date, end_date)
                
        except Exception as e:
            self.logger.error(f"从Tushare获取{ts_code}数据时出错: {str(e)}")
            return self._get_local_stock_data(ts_code, start_date, end_date)
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
            logger.error(f"从本地获取{ts_code}的日线数据失败: {str(e)}")
            return pd.DataFrame()
    def calculate_momentum(self, data):
        """计算动量指标"""
        try:
            if data is None or data.empty:
                print("数据为空，无法计算技术指标")
                return pd.DataFrame()
                
            df = data.copy()
            
            print(f"calculate_momentum: 输入数据大小 {df.shape}, 第一行:\n{df.iloc[0]}")
            
            # 确保数据有足够的行
            if len(df) < 20:
                print(f"数据行数不足，仅有 {len(df)} 行，无法计算技术指标(至少需要20行)")
                return pd.DataFrame()
            
            # 基本指标 - 简化版
            # 计算简单移动平均线
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma60'] = df['close'].rolling(window=60).mean() if len(df) >= 60 else df['close']
            
            # 计算动量
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # 计算相对强弱指数 (RSI)
            delta = df['close'].diff()
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            # 避免除以0
            avg_loss = avg_loss.replace(0, 0.00001)
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算MACD
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['signal']
            
            # 计算成交量变化
            df['vol_ratio_5'] = df['vol'] / df['vol'].rolling(window=5).mean()
            
            # KDJ指标
            low_min = df['low'].rolling(window=9).min()
            high_max = df['high'].rolling(window=9).max()
            
            # 避免除以0
            denom = high_max - low_min
            denom = denom.replace(0, 0.00001)
            
            df['k'] = 100 * ((df['close'] - low_min) / denom)
            df['d'] = df['k'].rolling(window=3).mean()
            df['j'] = 3 * df['k'] - 2 * df['d']
            
            # 去除NaN值
            df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            print(f"指标计算完成，结果数据大小 {df.shape}")
            
            return df
        except Exception as e:
            print(f"计算技术指标时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    def calculate_momentum_score(self, data, custom_weights=None):
        """计算动量综合评分
        
        Args:
            data: 股票数据
            custom_weights: 自定义指标权重
            
        Returns:
            tuple: (总分, 得分详情字典)
        """
        try:
            if data is None or data.empty:
                print("数据为空，无法计算动量得分")
                return 0, {}
                
            # 获取最新数据
            latest = data.iloc[-1]
            
            # 初始化得分
            scores = {}
            
            # 1. 价格动量得分 (0-40分)
            momentum_score = 0
            if 'momentum_5' in latest and not pd.isna(latest['momentum_5']):
                momentum_5 = latest['momentum_5'] * 100  # 转为百分比
                if momentum_5 > 2:  # 5日涨幅超过2%
                    momentum_score += 10
                elif momentum_5 > 0:  # 5日为正
                    momentum_score += 5
            
            if 'momentum_20' in latest and not pd.isna(latest['momentum_20']):
                momentum_20 = latest['momentum_20'] * 100  # 转为百分比
                if momentum_20 > 10:  # 20日涨幅超过10%
                    momentum_score += 30
                elif momentum_20 > 5:  # 20日涨幅超过5%
                    momentum_score += 20
                elif momentum_20 > 0:  # 20日为正
                    momentum_score += 10
            
            scores['momentum_score'] = momentum_score
            scores['momentum_5'] = latest.get('momentum_5', 0) * 100
            scores['momentum_20'] = latest.get('momentum_20', 0) * 100
            
            # 2. 均线得分 (0-20分)
            ma_score = 0
            close = latest['close']
            
            if 'ma5' in latest and not pd.isna(latest['ma5']) and close > latest['ma5']:
                ma_score += 5
            if 'ma10' in latest and not pd.isna(latest['ma10']) and close > latest['ma10']:
                ma_score += 5
            if 'ma20' in latest and not pd.isna(latest['ma20']) and close > latest['ma20']:
                ma_score += 5
            if 'ma60' in latest and not pd.isna(latest['ma60']) and close > latest['ma60']:
                ma_score += 5
            
            scores['ma_score'] = ma_score
            
            # 3. RSI得分 (0-20分)
            rsi_score = 0
            if 'rsi' in latest and not pd.isna(latest['rsi']):
                rsi = latest['rsi']
                if 40 <= rsi <= 80:  # 适中且偏强的RSI
                    rsi_score += 10
                    if 50 <= rsi <= 70:  # 理想区间
                        rsi_score += 10
            
            scores['rsi_score'] = rsi_score
            scores['rsi'] = latest.get('rsi', 0)
            
            # 4. MACD得分 (0-20分)
            macd_score = 0
            if all(x in latest and not pd.isna(latest[x]) for x in ['macd', 'signal', 'macd_hist']):
                macd = latest['macd']
                signal = latest['signal']
                hist = latest['macd_hist']
                
                if macd > 0 and signal > 0:  # MACD和信号线都为正
                    macd_score += 10
                if macd > signal:  # MACD在信号线之上
                    macd_score += 5
                if hist > 0:  # 柱状图为正
                    macd_score += 5
            
            scores['macd_score'] = macd_score
            scores['macd'] = latest.get('macd', 0)
            
            # 计算总分
            total_score = momentum_score + ma_score + rsi_score + macd_score
            
            # 总分控制在100以内
            total_score = min(100, total_score)
            
            print(f"动量得分: {total_score} (动量: {momentum_score}, 均线: {ma_score}, RSI: {rsi_score}, MACD: {macd_score})")
            
            return total_score, scores
        except Exception as e:
            print(f"计算动量得分时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0, {}
    def plot_stock_chart(self, data, stock_code, stock_name, score_details, save_path=None, show_additional_indicators=True):
        """绘制股票K线和技术指标图
        
        Args:
            data: 股票数据
            stock_code: 股票代码
            stock_name: 股票名称
            score_details: 得分详情
            save_path: 保存路径
            show_additional_indicators: 是否显示额外指标
            
        Returns:
            bool: 是否成功
        """
        try:
            if data is None or data.empty:
                print(f"无法绘制{stock_code}的图表，数据为空")
                return False
                
            # 创建图表
            fig = plt.figure(figsize=(14, 14))
            
            # 设置网格
            gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 1, 1])
            
            # 第一个子图：K线和均线
            ax1 = fig.add_subplot(gs[0, 0])
            total_score = sum(v for k, v in score_details.items() if isinstance(v, (int, float)))
            ax1.set_title(f"{stock_name}({stock_code}) 动量分析 - 总分: {total_score}", fontsize=15)
            
            # 绘制K线
            for i in range(len(data)):
                date = data.index[i]
                open_price, close, high, low = data.iloc[i][['open', 'close', 'high', 'low']]
                # 确定颜色（红涨绿跌）
                color = 'red' if close >= open_price else 'green'
                # 绘制影线
                ax1.plot([date, date], [low, high], color=color, linewidth=1)
                # 绘制实体
                rect_height = abs(open_price - close)
                rect_bottom = min(open_price, close)
                ax1.add_patch(plt.Rectangle((date - pd.Timedelta(days=0.4), rect_bottom),
                                        pd.Timedelta(days=0.8), rect_height,
                                        edgecolor=color, facecolor=color))
                                        
            # 绘制均线
            for ma, color in [('ma5', 'blue'), ('ma10', 'orange'), ('ma20', 'purple'), ('ma60', 'brown')]:
                if ma in data.columns:
                    ax1.plot(data.index, data[ma], label=ma.upper(), color=color, linewidth=1)
            
            ax1.legend(loc='best')
            ax1.grid(True)
            
            # 第二个子图：成交量
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            for i in range(len(data)):
                date = data.index[i]
                open_price, close = data.iloc[i][['open', 'close']]
                color = 'red' if close >= open_price else 'green'
                ax2.bar(date, data.iloc[i]['vol'], color=color, width=pd.Timedelta(days=0.8))
            ax2.set_ylabel('Volume')
            ax2.grid(True)
            
            # 第三个子图：MACD
            if all(col in data.columns for col in ['macd', 'signal', 'macd_hist']):
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
            
            # 第四个子图：RSI
            if 'rsi' in data.columns:
                ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
                ax4.plot(data.index, data['rsi'], label='RSI', color='purple', linewidth=1)
                ax4.axhline(y=30, color='green', linestyle='--')
                ax4.axhline(y=70, color='red', linestyle='--')
                ax4.legend(loc='best')
                ax4.set_ylabel('RSI')
                ax4.set_ylim(0, 100)
                ax4.grid(True)
            
            # 第五个子图：KDJ
            if all(col in data.columns for col in ['k', 'd', 'j']):
                ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)
                ax5.plot(data.index, data['k'], label='K', color='blue', linewidth=1)
                ax5.plot(data.index, data['d'], label='D', color='yellow', linewidth=1)
                ax5.plot(data.index, data['j'], label='J', color='magenta', linewidth=1)
                ax5.axhline(y=20, color='green', linestyle='--')
                ax5.axhline(y=80, color='red', linestyle='--')
                ax5.legend(loc='best')
                ax5.set_ylabel('KDJ')
                ax5.grid(True)
            
            # 添加得分信息
            score_text = "\n".join([f"{k}: {v}" for k, v in score_details.items()])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.02, 0.05, score_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='bottom', bbox=props)
                    
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                plt.savefig(save_path)
                print(f"已保存{stock_code}的图表到{save_path}")
                plt.close(fig)
                return True
            else:
                plt.show()
                return True
        except Exception as e:
            print(f"绘制图表时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    def _analyze_single_stock(self, stock_data):
        """分析单个股票的动量情况
        
        Args:
            stock_data: 单个股票的信息元组 (idx, info)
            
        Returns:
            dict: 分析结果
        """
        idx, stock = stock_data
        
        try:
            stock_code = stock['ts_code']
            stock_name = stock['name']
            logger.debug(f"正在分析: {stock_name}({stock_code})")
            
            # 获取股票数据
            df = self.get_stock_daily_data(stock_code)
            
            # 检查数据是否足够分析
            if df is None or len(df) < self.lookback_period:
                logger.debug(f"股票 {stock_name}({stock_code}) 的历史数据不足 {self.lookback_period} 天，跳过分析")
                return None
                
            # 检查股票是否停牌或交易异常
            if pd.to_datetime(df.index[0]).date() != pd.to_datetime(self.end_date).date():
                trading_delay = (pd.to_datetime(self.end_date) - pd.to_datetime(df.index[0])).days
                if trading_delay > 3:  # 超过3天未交易，可能是停牌
                    logger.debug(f"股票 {stock_name}({stock_code}) 可能已停牌 {trading_delay} 天，跳过分析")
                    return None
            
            # 计算动量指标
            df = self.calculate_momentum(df)
            
            # 确保计算成功且数据有效
            if df is None or df.empty:
                logger.debug(f"股票 {stock_name}({stock_code}) 的指标计算失败，跳过分析")
                return None
                
            # 获取最新数据
            latest = df.iloc[0] if not df.empty else None
            
            if latest is None:
                logger.debug(f"股票 {stock_name}({stock_code}) 的最新数据获取失败，跳过分析")
                return None
                
            # 计算动量评分
            # 1. 价格相对于均线
            ma_signals = {}
            short_ma_score = 0
            mid_ma_score = 0
            long_ma_score = 0
            
            # 价格与均线关系评分（加权）
            if latest['close'] > latest['ma5']:
                short_ma_score += 8
                ma_signals['price_above_ma5'] = True
            if latest['close'] > latest['ma10']:
                short_ma_score += 7
                ma_signals['price_above_ma10'] = True
            if latest['close'] > latest['ma20']:
                mid_ma_score += 6
                ma_signals['price_above_ma20'] = True
            if latest['close'] > latest['ma30']:
                mid_ma_score += 5
                ma_signals['price_above_ma30'] = True
            if latest['close'] > latest['ma60']:
                long_ma_score += 4
                ma_signals['price_above_ma60'] = True
            if latest['close'] > latest['ma120']:
                long_ma_score += 3
                ma_signals['price_above_ma120'] = True
                
            # 2. 均线多头排列评分
            ma_alignment_score = 0
            # 短期均线多头排列
            if latest['ma5'] > latest['ma10'] > latest['ma20']:
                ma_alignment_score += 12
                ma_signals['short_ma_bull_alignment'] = True
            # 中期均线多头排列
            if latest['ma10'] > latest['ma30'] > latest['ma60']:
                ma_alignment_score += 10
                ma_signals['mid_ma_bull_alignment'] = True
            # 长期均线多头排列
            if latest['ma30'] > latest['ma60'] > latest['ma120']:
                ma_alignment_score += 8
                ma_signals['long_ma_bull_alignment'] = True
                
            # 3. 成交量评分
            volume_score = 0
            volume_signals = {}
            
            # 成交量连续增长
            vol_increase_days = 0
            for i in range(min(5, len(df)-1)):
                if df.iloc[i]['vol'] > df.iloc[i+1]['vol']:
                    vol_increase_days += 1
            
            # 成交量满足上升趋势（至少连续3天成交量上升）
            if vol_increase_days >= 3:
                volume_score += 10
                volume_signals['vol_increasing'] = True
            
            # 近期成交量明显高于平均水平
            vol_ratio = latest['vol'] / df['vol'].mean()
            if vol_ratio > 2.0:
                volume_score += 15
                volume_signals['vol_surge'] = True
            elif vol_ratio > 1.5:
                volume_score += 10
                volume_signals['vol_above_avg'] = True
            elif vol_ratio > 1.2:
                volume_score += 5
                volume_signals['vol_slightly_above_avg'] = True
            
            # 4. MACD评分
            macd_score = 0
            macd_signals = {}
            
            # MACD金叉
            if df.iloc[0]['macd_hist'] > 0 and df.iloc[1]['macd_hist'] <= 0:
                macd_score += 15
                macd_signals['macd_golden_cross'] = True
            
            # MACD柱状图连续增长
            macd_hist_increase_days = 0
            for i in range(min(3, len(df)-1)):
                if df.iloc[i]['macd_hist'] > df.iloc[i+1]['macd_hist']:
                    macd_hist_increase_days += 1
            
            if macd_hist_increase_days >= 2:
                macd_score += 10
                macd_signals['macd_hist_increasing'] = True
            
            # MACD在零轴上方
            if df.iloc[0]['macd'] > 0:
                macd_score += 5
                macd_signals['macd_above_zero'] = True
            
            # 5. RSI评分
            rsi_score = 0
            rsi_signals = {}
            
            # RSI在黄金区间（40-70）非超买也非超卖
            if 40 <= latest['rsi_14'] <= 70:
                rsi_score += 10
                rsi_signals['rsi_golden_range'] = True
            
            # RSI上升趋势
            rsi_increase_days = 0
            for i in range(min(3, len(df)-1)):
                if df.iloc[i]['rsi_14'] > df.iloc[i+1]['rsi_14']:
                    rsi_increase_days += 1
            
            if rsi_increase_days >= 2:
                rsi_score += 5
                rsi_signals['rsi_uptrend'] = True
            
            # 6. KDJ评分
            kdj_score = 0
            kdj_signals = {}
            
            # KDJ金叉
            if latest['k'] > latest['d'] and df.iloc[1]['k'] <= df.iloc[1]['d']:
                kdj_score += 15
                kdj_signals['kdj_golden_cross'] = True
            
            # KDJ值在20-80的良好区间
            if 20 <= latest['k'] <= 80 and 20 <= latest['d'] <= 80:
                kdj_score += 5
                kdj_signals['kdj_good_range'] = True
            
            # K线上穿D线且都在50以上
            if latest['k'] > latest['d'] and latest['k'] > 50 and latest['d'] > 50:
                kdj_score += 10
                kdj_signals['kdj_strong_momentum'] = True
            
            # 7. 价格趋势评分
            trend_score = 0
            trend_signals = {}
            
            # 计算过去N天的收盘价趋势
            close_trend_days = min(20, len(df))
            close_prices = df['close'].head(close_trend_days).values
            
            # 使用线性回归计算趋势强度
            import numpy as np
            from scipy import stats
            
            x = np.arange(close_trend_days)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, close_prices[::-1])
            
            # 趋势强度 = 斜率 * R方值 * 100 / 平均价格
            trend_strength = slope * (r_value ** 2) * 100 / np.mean(close_prices)
            
            # 根据趋势强度分配得分
            if trend_strength > 1.0:
                trend_score += 20
                trend_signals['strong_uptrend'] = True
            elif trend_strength > 0.5:
                trend_score += 15
                trend_signals['moderate_uptrend'] = True
            elif trend_strength > 0.2:
                trend_score += 10
                trend_signals['slight_uptrend'] = True
            elif trend_strength > 0:
                trend_score += 5
                trend_signals['flat_to_up'] = True
            
            # 8. 布林通道评分
            boll_score = 0
            boll_signals = {}
            
            # 价格突破上轨
            if latest['close'] > latest['boll_upper']:
                boll_score += 10
                boll_signals['price_above_upper_band'] = True
            
            # 价格在中轨和上轨之间，且接近上轨
            upper_mid_gap = latest['boll_upper'] - latest['boll_middle']
            price_mid_gap = latest['close'] - latest['boll_middle']
            if latest['close'] < latest['boll_upper'] and latest['close'] > latest['boll_middle'] and price_mid_gap > 0.5 * upper_mid_gap:
                boll_score += 15
                boll_signals['price_near_upper_band'] = True
            
            # 通道宽度扩大（通常意味着波动性增加，可能出现趋势）
            band_width = (latest['boll_upper'] - latest['boll_lower']) / latest['boll_middle']
            prev_band_width = (df.iloc[5]['boll_upper'] - df.iloc[5]['boll_lower']) / df.iloc[5]['boll_middle']
            
            if band_width > prev_band_width:
                boll_score += 5
                boll_signals['band_widening'] = True
            
            # 9. 突破信号评分
            breakout_score = 0
            breakout_signals = {}
            
            # 价格突破前期高点
            local_high = df['close'].iloc[1:21].max()
            if latest['close'] > local_high:
                breakout_score += 15
                breakout_signals['break_local_high'] = True
            
            # 计算20日的最高点与最低点
            high_20 = df['high'].iloc[1:21].max()
            low_20 = df['low'].iloc[1:21].min()
            
            # 突破盘整区间
            if abs(high_20 - low_20) / low_20 < 0.1 and latest['close'] > high_20:
                breakout_score += 10
                breakout_signals['break_consolidation'] = True
            
            # 10. 反转信号评分
            reversal_score = 0
            reversal_signals = {}
            
            # 检测W底形态
            if (df.iloc[0]['close'] > df.iloc[0]['ma5'] and 
                df.iloc[10]['close'] < df.iloc[10]['ma20'] and 
                df.iloc[20]['close'] > df.iloc[20]['ma10']):
                reversal_score += 10
                reversal_signals['w_bottom_pattern'] = True
            
            # 汇总各项评分
            ma_total_score = short_ma_score + mid_ma_score + long_ma_score + ma_alignment_score
            indicator_score = macd_score + rsi_score + kdj_score
            price_pattern_score = trend_score + boll_score + breakout_score + reversal_score
            
            # 总分 = 移动平均分数(35%) + 技术指标分数(30%) + 成交量分数(15%) + 价格形态分数(20%)
            total_score = (ma_total_score * 0.35) + (indicator_score * 0.30) + (volume_score * 0.15) + (price_pattern_score * 0.20)
            
            # 限制总分在0-100之间
            total_score = max(0, min(100, total_score))
            
            # 分析结果字典
            result = {
                'ts_code': stock_code,
                'name': stock_name,
                'industry': stock.get('industry', '未知'),
                'area': stock.get('area', '未知'),
                'market': stock.get('market', '未知'),
                'list_date': stock.get('list_date', '未知'),
                'close': round(latest['close'], 2),
                'change_pct': round(latest['pct_chg'], 2) if 'pct_chg' in latest else None,
                'ma_total_score': round(ma_total_score, 1),
                'indicator_score': round(indicator_score, 1),
                'volume_score': round(volume_score, 1),
                'price_pattern_score': round(price_pattern_score, 1),
                'total_score': round(total_score, 1),
                'signals': {
                    'ma': ma_signals,
                    'volume': volume_signals,
                    'macd': macd_signals,
                    'rsi': rsi_signals,
                    'kdj': kdj_signals,
                    'trend': trend_signals,
                    'boll': boll_signals,
                    'breakout': breakout_signals,
                    'reversal': reversal_signals
                },
                'data': df.head(30)  # 保存最近30天数据用于绘图
            }
            
            # 添加分析时间
            result['analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return result
            
        except Exception as e:
            logger.error(f"分析股票 {stock.get('name', '')}({stock.get('ts_code', '')}) 时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def analyze_stocks(self, sample_size=100, min_score=60):
        """
        分析股票动量
        
        Args:
            sample_size (int): 分析的股票数量，默认为100
            min_score (float): 最小动量得分，默认为60
            
        Returns:
            list: 动量分析结果列表，每个元素为字典，包含股票代码、名称、动量得分等信息
        """
        try:
            self.logger.info(f"开始分析股票动量，样本数量: {sample_size}，最小得分: {min_score}")
            
            # 获取股票列表
            stock_list = self.get_stock_list()
            if len(stock_list) > sample_size:
                stock_list = random.sample(stock_list, sample_size)
            
            momentum_stocks = []
            
            for stock_code in stock_list:
                try:
                    # 获取股票详情
                    stock_details = self.get_stock_daily_data(stock_code)
                    if not stock_details.empty:
                        # 计算动量得分
                        score = self.calculate_momentum_score(stock_details)
                        
                        if score >= min_score:
                            momentum_stocks.append({
                                'code': stock_code,
                                'name': stock_details.iloc[0]['name'],
                                'score': score,
                                'price': stock_details.iloc[0]['close'],
                                'volume': stock_details.iloc[0]['vol'],
                                'industry': stock_details.iloc[0]['industry']
                            })
                            
                except Exception as e:
                    self.logger.error(f"分析股票{stock_code}时出错: {str(e)}")
                    continue
            
            # 按得分排序
            momentum_stocks.sort(key=lambda x: -x['score'])
            
            self.logger.info(f"动量分析完成，找到{len(momentum_stocks)}只符合条件的股票")
            return momentum_stocks
            
        except Exception as e:
            self.logger.error(f"动量分析出错: {str(e)}")
            return []
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'pool') and self.pool:
            self.pool.close()
            self.pool.join()
# 运行测试
if __name__ == "__main__":
    print("开始测试Tushare API获取股票数据")
    # 测试
    analyzer = MomentumAnalyzer(use_tushare=True)
    analyzer.use_parallel = False  # 禁用并行处理
    
    # 设置日期范围
    analyzer.start_date = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
    analyzer.end_date = datetime.now().strftime('%Y%m%d')
    
    # 直接测试获取单个股票数据
    print("\n直接测试获取平安银行(000001.SZ)数据:")
    try:
        df_test = analyzer.get_stock_daily_data('000001.SZ')
        if df_test is not None and not df_test.empty:
            print(f"成功获取到股票数据，共{len(df_test)}条记录")
            print(df_test.head())
        else:
            print("获取数据失败，返回为空")
    except Exception as e:
        print(f"测试获取股票数据时出错: {e}")
    
    # 获取股票列表
    print("\n获取股票列表...")
    stocks = analyzer.get_stock_list()
    print(f"获取到 {len(stocks)} 支股票")
    
    # 分析前5支股票（小样本测试）
    sample_size = 5
    print(f"\n分析前{sample_size}支股票...")
    
    # 手动遍历股票列表进行分析，显示进度
    results = []
    for i, (idx, stock) in enumerate(stocks.head(sample_size).iterrows()):
        print(f"[{i+1}/{sample_size}] 正在分析: {stock['name']}({stock['ts_code']})")
        result = analyzer._analyze_single_stock((idx, stock))
        if result:
            results.append(result)
            print(f"✓ 分析完成: {stock['name']}({stock['ts_code']}) - 得分: {result['total_score']:.1f}")
        else:
            print(f"✗ {stock['name']}({stock['ts_code']}) 分析未通过筛选条件")
        print("-" * 50)  # 分隔线
    
    # 按得分排序
    results = sorted(results, key=lambda x: x['total_score'], reverse=True)
    
    # 输出结果
    print("\n分析结果:")
    if results:
        for r in results:
            print(f"{r['name']}({r['code']}): 得分={r['total_score']:.1f}, 动量20={r['price_momentum']:.2f}%")
    else:
        print("未找到符合条件的股票")