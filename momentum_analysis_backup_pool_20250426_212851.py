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
import functools
import multiprocessing as mp
import time
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

# 全局 Tushare Pro API 实例 (惰性加载)
_pro_api_instance = None

# 尝试在模块级别获取 TUSHARE_TOKEN
_MODULE_TUSHARE_TOKEN = None
try:
    from src.enhanced.config.settings import TUSHARE_TOKEN as _CONFIG_TOKEN
    _MODULE_TUSHARE_TOKEN = _CONFIG_TOKEN
except ImportError:
    logger.warning("无法从 src.enhanced.config.settings 导入 TUSHARE_TOKEN")
    # 尝试从环境变量获取
    _MODULE_TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
    if _MODULE_TUSHARE_TOKEN:
        logger.info("从环境变量加载 Tushare Token")
    else:
        # 尝试从文件获取
        try:
            home_dir = os.path.expanduser("~")
            token_file = os.path.join(home_dir, ".tushare", "token")
            if os.path.exists(token_file):
                 with open(token_file, 'r') as f:
                     _MODULE_TUSHARE_TOKEN = f.read().strip()
                     logger.info("从 ~/.tushare/token 文件加载 Tushare Token")
        except Exception as e:
             logger.warning(f"尝试读取 ~/.tushare/token 文件失败: {e}")

def get_pro_api():
    """获取 Tushare Pro API 实例 (惰性加载，确保只初始化一次)"""
    global _pro_api_instance
    global _MODULE_TUSHARE_TOKEN # 引用模块级别的 token

    if _pro_api_instance is None:
        token_to_use = _MODULE_TUSHARE_TOKEN # 使用在模块级别获取的token
        
        if token_to_use:
            try:
                ts.set_token(token_to_use)
                _pro_api_instance = ts.pro_api()
                logger.info(f"Tushare Pro API 初始化成功 (Token: {token_to_use[:5]}...)")
            except Exception as e:
                logger.error(f"Tushare Pro API 初始化失败: {e}")
                # 初始化失败，保持为 None
        else:
            logger.error("无法找到 Tushare Token，Tushare API 将不可用")
            
    return _pro_api_instance


def get_stock_name(ts_code):
    """
    根据股票代码获取股票名称（简单实现，实际使用时应从数据库或API获取）
    
    参数:
        ts_code: 股票代码(带后缀，如000001.SZ)
        
    返回:
        股票名称
    """
    # 常见股票名称映射
    stock_names = {
        '000001.SZ': '平安银行',
        '000002.SZ': '万科A',
        '000063.SZ': '中兴通讯',
        '000333.SZ': '美的集团',
        '000651.SZ': '格力电器',
        '000858.SZ': '五粮液',
        '002415.SZ': '海康威视',
        '600036.SH': '招商银行',
        '600276.SH': '恒瑞医药',
        '600887.SH': '伊利股份'
    }
    
    # 如果能找到，返回名称，否则返回代码
    return stock_names.get(ts_code, f"股票{ts_code}")


class MomentumAnalyzer:
    """动量分析器：用于分析股票的动量情况"""

    def __init__(self, stock_pool=None, start_date=None, end_date=None, 
                 backtest_start_date=None, backtest_end_date=None, 
                 momentum_period=20, use_parallel=True, max_processes=None, 
                 use_cache=True, enable_optimization=True, use_tushare=True):
        """
        初始化动量分析器
        
        参数:
            stock_pool: 股票池，默认为None，将使用默认股票池
            start_date: 开始日期，默认为None，将使用当前日期的前252个交易日
            end_date: 结束日期，默认为None，将使用当前日期
            backtest_start_date: 回测开始日期，默认为None，将使用当前日期的前252个交易日
            backtest_end_date: 回测结束日期，默认为None，将使用当前日期
            momentum_period: 动量计算周期，默认为20个交易日
            use_parallel: 是否使用并行计算，默认为True
            max_processes: 最大进程数，默认为None，将使用CPU核心数的2/3
            use_cache: 是否使用缓存，默认为True
            enable_optimization: 是否启用内存和CPU优化，默认为True
        """
        # 设置日志
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.logger = logging.getLogger("momentum_analysis")
        self.logger.setLevel(logging.INFO)
        
        # 如果没有处理器，则添加处理器
        if not self.logger.handlers:
            # 添加文件处理器
            file_handler = logging.FileHandler(
                os.path.join(log_dir, "momentum_analysis.log"), 
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)
            
            # 添加控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 设置格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加处理器
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
        # 禁用tushare日志
        logging.getLogger("tushare").setLevel(logging.ERROR)
        
        # 初始化变量
        self.stock_pool = stock_pool if stock_pool is not None else get_stock_pool()
        self.start_date = start_date
        self.end_date = end_date
        self.backtest_start_date = backtest_start_date
        self.backtest_end_date = backtest_end_date
        self.momentum_period = momentum_period
        self.use_parallel = use_parallel
        self.use_cache = use_cache
        self.enable_optimization = enable_optimization
        self.use_tushare = use_tushare
        
        # 限制最大进程数以避免资源耗尽
        if max_processes is None:
            # 使用CPU核心数的一半，最小为1
            self.max_processes = max(1, min(4, int(multiprocessing.cpu_count() / 2)))
        else:
            self.max_processes = max_processes
            
        self.logger.info(f"设置最大进程数为: {self.max_processes}")
        
        # 不在初始化时创建进程池
        self.pool = None
        
        # 初始化日期
        self._init_dates()
        
        # 初始化缓存目录
        if self.use_cache:
            self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
        
        # 启用内存优化
        if self.enable_optimization:
            self._enable_optimizations()
            
        # 记录初始化成功
        self.logger.info(f"动量分析器初始化成功，股票池大小：{len(self.stock_pool)}，并行计算：{self.use_parallel}，使用缓存：{self.use_cache}")
        
    def _enable_optimizations(self):
        """启用各种优化措施以减少内存和CPU使用"""
        try:
            # 设置pandas显示选项以减少内存使用
            pd.options.mode.chained_assignment = None
            
            # 减少进程数，如果股票池小于特定数量
            if len(self.stock_pool) < 100:
                self.max_processes = min(self.max_processes, 2)
                self.logger.info(f"由于股票池较小，减少最大进程数至: {self.max_processes}")
                
            # 如果股票池太大，限制大小以避免内存问题
            if len(self.stock_pool) > 1000:
                original_size = len(self.stock_pool)
                self.stock_pool = self.stock_pool[:1000]
                self.logger.warning(f"股票池过大，从 {original_size} 减少到 1000")
                
            self.logger.info("已启用内存和CPU优化")
        except Exception as e:
            self.logger.warning(f"启用优化时出错: {str(e)}")

    def _init_parallel_pool(self):
        """初始化并行计算的进程池"""
        if self.use_parallel and self.pool is None:
            try:
                self.logger.info(f"正在创建{self.max_processes}个进程的进程池...")
                self.pool = multiprocessing.Pool(processes=self.max_processes)
                self.logger.info("进程池创建成功")
            except Exception as e:
                self.logger.error(f"创建进程池失败: {str(e)}")
                self.use_parallel = False
                self.logger.warning("已禁用并行计算，将使用串行计算")

    
    def get_stock_list(self, industry=None):
        """
        获取股票列表
        
        参数:
            industry: 行业名称，默认为None，表示获取所有行业股票
            
        返回:
            包含股票信息的DataFrame
        """
        try:
            self.logger.info("获取股票列表")
            pro_api = get_pro_api()
            
            if pro_api is not None and self.use_tushare:
                # 从Tushare获取
                try:
                    # 获取股票基本信息
                    stocks = pro_api.stock_basic(
                        exchange='', 
                        list_status='L', 
                        fields='ts_code,symbol,name,area,industry,list_date'
                    )
                    
                    # 如果指定了行业，进行筛选
                    if industry and industry != "全部":
                        stocks = stocks[stocks['industry'] == industry]
                        
                    return stocks
                except Exception as e:
                    self.logger.error(f"从Tushare获取股票列表失败: {str(e)}")
                    return self._get_local_stock_list(industry)
            else:
                return self._get_local_stock_list(industry)
        except Exception as e:
            self.logger.error(f"获取股票列表时出错: {str(e)}")
            return pd.DataFrame()
    
    def _get_local_stock_list(self, industry=None):
        """从本地获取股票列表（备用方法）"""
        try:
            self.logger.info("从本地获取股票列表")
            
            # 生成一个简单的股票列表
            stock_data = {
                'ts_code': ['000001.SZ', '000002.SZ', '000063.SZ', '000333.SZ', '000651.SZ', 
                           '000858.SZ', '002415.SZ', '600036.SH', '600276.SH', '600887.SH'],
                'symbol': ['000001', '000002', '000063', '000333', '000651', 
                          '000858', '002415', '600036', '600276', '600887'],
                'name': ['平安银行', '万科A', '中兴通讯', '美的集团', '格力电器', 
                        '五粮液', '海康威视', '招商银行', '恒瑞医药', '伊利股份'],
                'area': ['深圳', '深圳', '深圳', '广东', '广东', 
                        '四川', '浙江', '上海', '江苏', '内蒙古'],
                'industry': ['银行', '房地产', '通信', '家电', '家电', 
                           '食品饮料', '电子', '银行', '医药', '食品饮料'],
                'list_date': ['19910403', '19910129', '19971118', '19970620', '19961118', 
                             '19980427', '20100528', '20021119', '20031216', '19960403']
            }
            
            stocks = pd.DataFrame(stock_data)
            
            # 如果指定了行业，进行筛选
            if industry and industry != "全部":
                stocks = stocks[stocks['industry'] == industry]
                
            return stocks
        except Exception as e:
            self.logger.error(f"创建本地股票列表失败: {str(e)}")
            # 返回一个空的DataFrame，具有正确的列
            return pd.DataFrame(columns=['ts_code', 'symbol', 'name', 'area', 'industry', 'list_date'])

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
        
        # 检查缓存
        if self.use_cache:
            cache_key = f"{ts_code}_{start_date}_{end_date}"
            cache_file = os.path.join(self.cache_dir, f"{ts_code.replace('.', '_')}.csv")
            
            # 尝试从内存缓存获取
            if hasattr(self, 'data_cache') and cache_key in self.data_cache:
                self.logger.debug(f"从内存缓存获取{ts_code}数据")
                return self.data_cache[cache_key]
            
            # 尝试从文件缓存获取
            if os.path.exists(cache_file):
                try:
                    df = pd.read_csv(cache_file)
                    if 'trade_date' in df.columns:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        df.sort_values('trade_date', inplace=True)
                        
                        # 日期筛选
                        if start_date:
                            start_date_dt = pd.to_datetime(start_date)
                            df = df[df['trade_date'] >= start_date_dt]
                        if end_date:
                            end_date_dt = pd.to_datetime(end_date)
                            df = df[df['trade_date'] <= end_date_dt]
                            
                        df.set_index('trade_date', inplace=True)
                    
                    # 添加到内存缓存
                    if not hasattr(self, 'data_cache'):
                        self.data_cache = {}
                    self.data_cache[cache_key] = df
                    
                    self.logger.debug(f"从文件缓存获取{ts_code}数据")
                    return df
                except Exception as e:
                    self.logger.warning(f"从文件缓存读取{ts_code}数据失败: {str(e)}")
        
        # 从Tushare获取数据
        try:
            self.logger.info(f"从Tushare获取{ts_code}日线数据")
            pro_api = get_pro_api()
            if pro_api is None:
                self.logger.error("无法获取Tushare Pro API实例")
                return self._get_local_stock_data(ts_code, start_date, end_date)
            
            # 尝试使用daily获取
            df = pro_api.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            # 如果daily没有数据，尝试使用pro_bar
            if df is None or df.empty:
                self.logger.info(f"通过daily获取{ts_code}数据为空，尝试使用pro_bar获取")
                df = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date=start_date, end_date=end_date)
            
            if df is not None and not df.empty:
                # 预处理数据
                if 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    df.sort_values('trade_date', inplace=True)
                    df.set_index('trade_date', inplace=True)
                
                # 添加到缓存
                if self.use_cache:
                    # 内存缓存
                    if not hasattr(self, 'data_cache'):
                        self.data_cache = {}
                    self.data_cache[cache_key] = df
                    
                    # 文件缓存
                    try:
                        os.makedirs(self.cache_dir, exist_ok=True)
                        df_to_save = df.reset_index()
                        df_to_save.to_csv(cache_file, index=False)
                        self.logger.debug(f"已将{ts_code}数据保存至缓存文件")
                    except Exception as e:
                        self.logger.warning(f"保存{ts_code}数据至缓存文件失败: {str(e)}")
                
                return df
            else:
                self.logger.warning(f"从Tushare获取{ts_code}数据失败，尝试使用备用方法")
                return self._get_local_stock_data(ts_code, start_date, end_date)
                
        except Exception as e:
            self.logger.error(f"从Tushare获取{ts_code}数据时出错: {str(e)}")
            return self._get_local_stock_data(ts_code, start_date, end_date)
    
    def _get_local_stock_data(self, ts_code, start_date=None, end_date=None):
        """
        从本地获取或生成股票日线数据（当Tushare数据获取失败时的备用方法）
        
        参数:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            包含股票日线数据的DataFrame
        """
        try:
            # 尝试从数据目录读取
            stock_file = os.path.join(DATA_DIR, f"{ts_code.replace('.', '_')}.csv")
            if os.path.exists(stock_file):
                self.logger.info(f"从本地文件读取{ts_code}数据")
                df = pd.read_csv(stock_file)
                
                # 处理日期
                if 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    df.sort_values('trade_date', inplace=True)
                    
                    # 日期筛选
                    if start_date:
                        start_date_dt = pd.to_datetime(start_date)
                        df = df[df['trade_date'] >= start_date_dt]
                    if end_date:
                        end_date_dt = pd.to_datetime(end_date)
                        df = df[df['trade_date'] <= end_date_dt]
                        
                    df.set_index('trade_date', inplace=True)
                
                return df
            else:
                # 创建模拟数据
                self.logger.warning(f"本地无{ts_code}数据，创建模拟数据")
                
                # 生成日期序列
                if not end_date:
                    end_date = datetime.now().strftime('%Y%m%d')
                if not start_date:
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                    
                start_date_dt = pd.to_datetime(start_date)
                end_date_dt = pd.to_datetime(end_date)
                date_range = pd.date_range(start=start_date_dt, end=end_date_dt, freq='B')
                
                # 生成模拟数据
                n = len(date_range)
                seed = sum(ord(c) for c in ts_code)  # 根据股票代码生成种子
                np.random.seed(seed)
                
                # 创建基础价格序列
                base_price = 50 + np.random.rand() * 50  # 基础价格50-100之间随机
                price_changes = np.random.normal(0, 1, n) * 0.5  # 每日价格变化
                
                # 添加趋势和周期性
                trend = np.linspace(0, 5, n) * np.random.choice([-1, 1]) * 0.2  # 添加轻微趋势
                cycle = np.sin(np.linspace(0, 4*np.pi, n)) * 5  # 添加周期波动
                
                # 合并所有因素
                cumulative_changes = np.cumsum(price_changes) + trend + cycle
                close = base_price + cumulative_changes
                close = np.maximum(close, base_price * 0.5)  # 确保价格不会太低
                
                # 根据收盘价生成其他价格
                daily_volatility = np.random.uniform(0.01, 0.03, n)  # 每日波动率
                high = close * (1 + daily_volatility)
                low = close * (1 - daily_volatility)
                
                # 随机开盘价，但确保在最高价和最低价之间
                rand_factor = np.random.rand(n)  # 0到1之间的随机数
                open_price = low + rand_factor * (high - low)
                
                # 确保价格关系合理
                high = np.maximum(high, np.maximum(close, open_price) * 1.001)
                low = np.minimum(low, np.minimum(close, open_price) * 0.999)
                
                # 确保所有价格都是正数
                high = np.maximum(high, 0.01)
                low = np.maximum(low, 0.01)
                close = np.maximum(close, 0.01)
                open_price = np.maximum(open_price, 0.01)
                
                # 成交量与价格变化正相关
                price_change = np.abs(np.diff(close, prepend=close[0]))
                vol_base = np.random.normal(1e6, 2e5, n)  # 基础成交量
                vol = vol_base * (1 + 3 * price_change / close)  # 价格变化大时，成交量增加
                vol = np.abs(vol)  # 确保成交量为正
                
                # 计算成交额和其他指标
                amount = vol * close  # 成交额
                change = np.diff(close, prepend=close[0])  # 价格变化
                pct_chg = change / np.roll(close, 1) * 100  # 涨跌幅
                pct_chg[0] = 0  # 第一天的涨跌幅设为0
                
                # 创建DataFrame
                data = {
                    'ts_code': [ts_code] * n,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'vol': vol,
                    'amount': amount,
                    'change': change,
                    'pct_chg': pct_chg
                }
                df = pd.DataFrame(data, index=date_range)
                
                # 恢复随机种子
                np.random.seed(None)
                
                # 保存到本地文件
                try:
                    os.makedirs(os.path.dirname(stock_file) if os.path.dirname(stock_file) else '.', exist_ok=True)
                    df_to_save = df.reset_index()
                    df_to_save.rename(columns={'index': 'trade_date'}, inplace=True)
                    df_to_save.to_csv(stock_file, index=False)
                    self.logger.info(f"已将{ts_code}模拟数据保存至: {stock_file}")
                except Exception as e:
                    self.logger.warning(f"保存{ts_code}模拟数据失败: {str(e)}")
                
                return df
                
        except Exception as e:
            self.logger.error(f"获取{ts_code}本地数据失败: {str(e)}")
            # 返回空DataFrame
            return pd.DataFrame()

    def calculate_momentum(self, data, period=None):
        """
        计算股票动量指标
        
        参数:
            data: 股票日线数据DataFrame
            period: 动量计算周期，默认为实例化时设置的周期
            
        返回:
            包含动量分析结果的字典
        """
        try:
            if data is None or data.empty or len(data) < 20:
                self.logger.warning("数据为空或不足，无法计算动量指标")
                return None
                
            # 使用实例变量或参数指定的值
            period = period if period is not None else self.momentum_period
            
            # 确保数据类型正确
            df = data.copy()
            
            # 确保有必要的列
            required_columns = ['close', 'open', 'high', 'low', 'vol']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"数据缺少必要的列: {col}")
                    return None
            
            # 计算价格变化百分比
            price_change = (df['close'].iloc[-1] / df['close'].iloc[-period-1] - 1) * 100 if len(df) > period else 0
            
            # 计算均线
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma60'] = df['close'].rolling(window=60).mean() if len(df) >= 60 else np.nan
            
            # 计算MACD
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['histogram'] = df['macd'] - df['signal']
            
            # 计算RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算成交量变化
            volume_change = (df['vol'].iloc[-period:].mean() / df['vol'].iloc[-period*2:-period].mean() - 1) * 100 if len(df) > period*2 else 0
            
            # 计算布林带
            df['boll_mid'] = df['close'].rolling(window=20).mean()
            df['boll_std'] = df['close'].rolling(window=20).std()
            df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
            df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
            
            # 获取最近的值
            last_values = df.iloc[-1]
            
            # 计算动量得分
            momentum_score = 0
            max_score = 100
            
            # 1. 价格相对于均线得分 (0-30分)
            ma_score = 0
            if not np.isnan(last_values['ma5']) and last_values['close'] > last_values['ma5']:
                ma_score += 6
            if not np.isnan(last_values['ma10']) and last_values['close'] > last_values['ma10']:
                ma_score += 8
            if not np.isnan(last_values['ma20']) and last_values['close'] > last_values['ma20']:
                ma_score += 8
            if not np.isnan(last_values['ma60']) and last_values['close'] > last_values['ma60']:
                ma_score += 8
            
            # 2. 价格变化得分 (0-20分)
            price_change_score = min(20, max(0, price_change))
            
            # 3. MACD信号得分 (0-20分)
            macd_score = 0
            if not np.isnan(last_values['macd']) and last_values['macd'] > 0:
                macd_score += 10
            if not np.isnan(last_values['macd']) and not np.isnan(last_values['signal']) and last_values['macd'] > last_values['signal']:
                macd_score += 10
                
            # 4. RSI得分 (0-15分)
            rsi_score = 0
            if not np.isnan(last_values['rsi']):
                if 40 <= last_values['rsi'] <= 70:
                    rsi_score += 10
                if 50 <= last_values['rsi'] <= 65:
                    rsi_score += 5
                    
            # 5. 成交量得分 (0-15分)
            volume_score = min(15, max(0, volume_change))
            
            # 计算总得分
            momentum_score = ma_score + price_change_score + macd_score + rsi_score + volume_score
            momentum_score = min(max_score, max(0, momentum_score))  # 限制在0-100之间
            
            # 计算相对排名
            rank_score = momentum_score / max_score * 100
            
            # 确定MACD信号
            if not np.isnan(last_values['macd']) and not np.isnan(last_values['signal']):
                if last_values['macd'] > last_values['signal'] and last_values['macd'] > 0:
                    macd_signal = "买入"
                elif last_values['macd'] < last_values['signal'] and last_values['macd'] < 0:
                    macd_signal = "卖出"
                else:
                    macd_signal = "中性"
            else:
                macd_signal = "未知"
                
            # 返回动量分析结果
            result = {
                'momentum_score': momentum_score,
                'rank_score': rank_score,
                'price_change': price_change,
                'volume_change': volume_change,
                'macd_signal': macd_signal,
                'ma_score': ma_score,
                'macd_score': macd_score,
                'rsi_score': rsi_score,
                'volume_score': volume_score
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算动量指标时出错: {str(e)}")
            return None

    def analyze_stocks(self, stocks=None, momentum_period=None, batch_size=50):
        """
        分析多个股票的动量情况
        
        参数:
            stocks: 股票代码列表，默认为None，将使用实例化时的股票池
            momentum_period: 动量计算周期，默认为None，将使用实例化时的周期
            batch_size: 批处理大小，用于分批获取数据，减少内存使用
            
        返回:
            包含各股票动量分析结果的DataFrame
        """
        start_time = time.time()
        
        # 使用实例变量或参数指定的值
        stocks = stocks if stocks is not None else self.stock_pool
        momentum_period = momentum_period if momentum_period is not None else self.momentum_period
        
        # 记录信息
        self.logger.info(f"开始分析{len(stocks)}只股票的动量情况，周期：{momentum_period}天")
        
        # 减少批大小以避免内存问题
        if self.enable_optimization and len(stocks) > 100:
            original_batch = batch_size
            batch_size = min(batch_size, 30)
            if original_batch != batch_size:
                self.logger.info(f"为优化内存使用，批处理大小从 {original_batch} 减少到 {batch_size}")
        
        # 准备分批处理股票
        stock_batches = [stocks[i:i+batch_size] for i in range(0, len(stocks), batch_size)]
        self.logger.info(f"将分{len(stock_batches)}批处理股票数据")
        
        all_results = []
        
        # 分批处理
        for batch_idx, batch_stocks in enumerate(stock_batches):
            self.logger.info(f"处理第{batch_idx+1}/{len(stock_batches)}批，包含{len(batch_stocks)}只股票")
            
            # 预先获取这批股票的数据
            stock_data_dict = {}
            failures = 0
            
            for stock in batch_stocks:
                try:
                    data = self.get_stock_daily_data(stock)
                    if data is not None and not data.empty:
                        stock_data_dict[stock] = data
                    else:
                        failures += 1
                except Exception as e:
                    self.logger.warning(f"获取{stock}数据失败: {str(e)}")
                    failures += 1
            
            self.logger.info(f"预获取数据完成: 成功{len(stock_data_dict)}只，失败{failures}只")
            
            batch_results = []
            
            if self.use_parallel:
                # 初始化进程池
                self._init_parallel_pool()
                
                if self.pool:
                    try:
                        # 准备并行任务参数
                        parallel_args = [(stock, momentum_period, stock_data_dict.get(stock, None)) 
                                       for stock in batch_stocks if stock in stock_data_dict]
                        
                        # 执行并行计算
                        batch_results = self.pool.starmap(self._analyze_single_stock, parallel_args)
                    except Exception as e:
                        self.logger.error(f"并行计算失败: {str(e)}")
                        self.logger.warning("切换到串行计算")
                        batch_results = []
                        
                        # 串行计算
                        for stock in batch_stocks:
                            if stock in stock_data_dict:
                                result = self._analyze_single_stock(stock, momentum_period, stock_data_dict[stock])
                                if result is not None:
                                    batch_results.append(result)
                else:
                    # 串行计算
                    for stock in batch_stocks:
                        if stock in stock_data_dict:
                            result = self._analyze_single_stock(stock, momentum_period, stock_data_dict[stock])
                            if result is not None:
                                batch_results.append(result)
            else:
                # 串行计算
                for stock in batch_stocks:
                    if stock in stock_data_dict:
                        result = self._analyze_single_stock(stock, momentum_period, stock_data_dict[stock])
                        if result is not None:
                            batch_results.append(result)
            
            # 将批次结果添加到所有结果中
            all_results.extend(batch_results)
            
            # 主动进行垃圾回收以释放内存
            if self.enable_optimization:
                import gc
                del stock_data_dict
                gc.collect()
        
        # 如果没有结果，返回空的DataFrame
        if not all_results:
            self.logger.warning("没有分析结果")
            return pd.DataFrame()
        
        # 将结果转换为DataFrame
        result_df = pd.DataFrame(all_results)
        
        # 按动量得分降序排序
        if 'momentum_score' in result_df.columns:
            result_df = result_df.sort_values(by='momentum_score', ascending=False)
        
        # 计算耗时
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        self.logger.info(f"动量分析完成，耗时：{elapsed_time:.2f}秒，分析了{len(result_df)}只股票")
        
        return result_df

    def _analyze_single_stock(self, stock, momentum_period, stock_data=None):
        """
        分析单个股票的动量情况
        
        参数:
            stock: 股票代码
            momentum_period: 动量计算周期
            stock_data: 预先获取的股票数据，如果为None则会获取
            
        返回:
            包含该股票动量分析结果的字典
        """
        try:
            # 如果没有提供股票数据，则获取数据
            if stock_data is None:
                stock_data = self.get_stock_daily_data(stock)
                
                # 如果数据为空，则返回None
                if stock_data is None or stock_data.empty:
                    return None
            
            # 计算动量
            momentum = self.calculate_momentum(stock_data, momentum_period)
            
            # 检查是否计算成功
            if momentum is None:
                return None
                
            # 获取最新的价格和名称
            latest_price = stock_data.iloc[-1]['close'] if not stock_data.empty else None
            stock_name = get_stock_name(stock)
            
            # 准备返回结果
            result = {
                'ts_code': stock,
                'stock_name': stock_name,
                'latest_price': latest_price,
                'momentum_score': momentum.get('momentum_score'),
                'rank_score': momentum.get('rank_score'),
                'price_change': momentum.get('price_change'),
                'volume_change': momentum.get('volume_change'),
                'macd_signal': momentum.get('macd_signal'),
                'momentum_period': momentum_period,
                'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
        except Exception as e:
            self.logger.error(f"分析{stock}时出错: {str(e)}")
            return None

    def __del__(self):
        """析构函数，用于释放资源"""
        try:
            # 关闭进程池
            if hasattr(self, 'pool') and self.pool is not None:
                self.logger.info("正在关闭进程池...")
                self.pool.close()
                self.pool.join()
                self.logger.info("进程池已关闭")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"关闭进程池时出错: {str(e)}")


# 运行测试
if __name__ == "__main__":
    # ===== Tushare Direct Initialization Test Start =====
    print("\n===== Tushare 直接初始化功能测试 =====")
    USER_TOKEN = '0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10'
    direct_pro_api = None
    try:
        print(f"尝试使用 Token '{USER_TOKEN[:5]}...' 直接初始化 Pro API...")
        ts.set_token(USER_TOKEN)
        direct_pro_api = ts.pro_api()
        if direct_pro_api:
             print("直接初始化 Pro API 成功")
             # Test trade_cal
             print("\n尝试使用直接初始化的 API 调用 trade_cal...")
             try:
                 df_cal = direct_pro_api.trade_cal(exchange='', start_date='20240101', end_date='20240105')
                 print("直接调用 trade_cal 成功:")
                 print(df_cal)
             except Exception as e_cal:
                 print(f"直接调用 trade_cal 失败: {e_cal}")

             # Test daily instead of pro_bar
             print("\n尝试使用直接初始化的 API 调用 daily 获取 000001.SZ 数据...")
             try:
                 # 尝试使用daily而不是pro_bar
                 df_daily = direct_pro_api.daily(ts_code='000001.SZ', start_date='20240101', end_date='20240110')
                 if df_daily is not None and not df_daily.empty:
                     print("直接调用 daily 成功:")
                     print(df_daily.head())
                 else:
                     print("直接调用 daily 返回为空或 None")
             except Exception as e_daily:
                 print(f"直接调用 daily 失败: {e_daily}")
                 
                 # 如果daily失败，尝试创建模拟数据
                 print("创建模拟数据用于测试:")
                 import pandas as pd
                 import numpy as np
                 from datetime import datetime, timedelta
                 
                 # 创建日期范围
                 start_date = datetime(2024, 1, 1)
                 end_date = datetime(2024, 1, 10)
                 date_range = pd.date_range(start=start_date, end=end_date, freq='B')
                 
                 # 生成模拟数据
                 n = len(date_range)
                 close_prices = np.linspace(10, 11, n) + np.random.normal(0, 0.1, n)
                 
                 # 创建DataFrame
                 df_dummy = pd.DataFrame({
                     'ts_code': ['000001.SZ'] * n,
                     'trade_date': [d.strftime('%Y%m%d') for d in date_range],
                     'open': close_prices * 0.99,
                     'high': close_prices * 1.02, 
                     'low': close_prices * 0.98,
                     'close': close_prices,
                     'vol': np.random.normal(1000000, 200000, n)
                 })
                 print("模拟数据样例:")
                 print(df_dummy.head())
        else:
            print("直接初始化 Pro API 失败 (返回 None)")

    except Exception as e_init:
         print(f"直接初始化 Pro API 或调用时发生错误: {e_init}")

    print("===== Tushare 直接初始化功能测试结束 =====\n")
    # ===== Tushare Direct Initialization Test End =====

    # 后续分析
    try:
        print("创建动量分析器实例...")
        analyzer = MomentumAnalyzer(use_tushare=True)
        
        # 获取股票列表
        print("获取股票列表...")
        stocks = analyzer.get_stock_list()
        print(f"获取到 {len(stocks)} 支股票")
        
        # 分析前5支股票
        sample_size = min(5, len(stocks))
        print(f"分析前{sample_size}支股票...")
        
        results = analyzer.analyze_stocks(stocks.head(sample_size))
        
        # 输出结果
        print("\n分析结果:")
        if isinstance(results, pd.DataFrame) and not results.empty:
            for index, row in results.iterrows():
                print(f"{row['stock_name']}({row['ts_code']}): " 
                      f"得分={row['momentum_score']:.1f}, "
                      f"价格变化={row['price_change']:.2f}%")
        else:
            print("没有有效的分析结果")
    except Exception as e:
        print(f"执行动量分析时出错: {e}")
        import traceback
        traceback.print_exc()
