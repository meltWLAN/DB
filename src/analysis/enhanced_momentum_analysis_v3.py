#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版动量分析模块 V3

提供高性能、智能化的股票动量分析功能，整合多种技术指标、机器学习模型和市场环境分析
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import json
import threading

# 抑制警告
warnings.filterwarnings('ignore')

# 确保当前目录在Python路径中
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 尝试导入项目配置
try:
    from src.enhanced.config.settings import (
        TUSHARE_TOKEN, LOG_DIR, DATA_DIR, RESULTS_DIR, 
        ENHANCED_CACHE_DIR, MARKET_DATA_CONFIG
    )
except ImportError:
    # 设置默认配置
    TUSHARE_TOKEN = ""
    LOG_DIR = "./logs"
    DATA_DIR = "./data"
    RESULTS_DIR = "./results"
    ENHANCED_CACHE_DIR = "./cache"
    MARKET_DATA_CONFIG = {
        "data_source": "tushare"
    }

# 尝试导入工具模块
try:
    from src.utils.parallel import ParallelExecutor, ParallelMode, parallelize
    from src.utils.cache import CacheManager, CacheLevel, cached
    from src.utils.preloader import DataPreloader, BatchDataFetcher
    from src.analysis.ml_models import MomentumMLModel
    
    # 导入原始动量分析模块
    from momentum_analysis import MomentumAnalyzer
    from enhanced_momentum_analysis import EnhancedMomentumAnalyzer
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"导入工具模块失败: {str(e)}")
    # 如果缺少关键模块，则终止
    if "momentum_analysis" in str(e):
        print("错误: 无法导入基础动量分析模块，请确保momentum_analysis.py存在")
        sys.exit(1)

# 确保目录存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "v3_charts"), exist_ok=True)
os.makedirs(ENHANCED_CACHE_DIR, exist_ok=True)

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'enhanced_momentum_v3.log')),
        logging.StreamHandler()
    ]
)

# 初始化全局缓存管理器
cache_manager = CacheManager(
    cache_dir=ENHANCED_CACHE_DIR,
    memory_limit_mb=512,
    disk_limit_mb=2048,
    default_memory_ttl=3600,  # 1小时
    default_disk_ttl=86400    # 1天
)

class MarketEnvironment:
    """市场环境分析类
    
    分析当前市场环境，判断是牛市、熊市还是震荡市
    提供自适应参数调整功能
    """
    
    # 市场环境类型
    BULL_MARKET = "bull"      # 牛市
    BEAR_MARKET = "bear"      # 熊市
    OSCILLATING = "oscillating"  # 震荡市
    UNDEFINED = "undefined"    # 未定义
    
    def __init__(self):
        """初始化市场环境分析器"""
        # 市场指数代码
        self.index_codes = ["000001.SH", "399001.SZ", "399006.SZ", "000300.SH"]
        self.environment_cache = {}
        self.lock = threading.RLock()
        
    @cached(key_pattern="market_env_{args}", ttl=43200, cache_manager=cache_manager)  # 12小时缓存
    def analyze_market_environment(self, lookback_days: int = 90) -> Dict[str, Any]:
        """
        分析当前市场环境
        
        Args:
            lookback_days: 回溯分析的天数
            
        Returns:
            Dict: 市场环境分析结果
        """
        try:
            # 计算重要指数的走势
            trend_scores = []
            
            # 使用原MomentumAnalyzer获取指数数据
            analyzer = MomentumAnalyzer()
            
            for index_code in self.index_codes:
                # 获取指数日线数据
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y%m%d')
                
                df = analyzer.get_stock_daily_data(index_code, start_date, end_date)
                if df is None or len(df) < 20:
                    continue
                
                # 计算技术指标
                df = self._calculate_market_indicators(df)
                
                # 判断市场趋势
                trend_score = self._calculate_trend_score(df)
                trend_scores.append(trend_score)
            
            # 如果没有获取到任何数据，返回未定义
            if not trend_scores:
                return {
                    "environment": self.UNDEFINED,
                    "confidence": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            
            # 计算平均趋势得分
            avg_trend_score = sum(trend_scores) / len(trend_scores)
            
            # 判断市场环境
            environment = self.UNDEFINED
            confidence = 0.0
            
            if avg_trend_score > 70:
                environment = self.BULL_MARKET
                confidence = min(1.0, (avg_trend_score - 70) / 30 + 0.5)
            elif avg_trend_score < 30:
                environment = self.BEAR_MARKET
                confidence = min(1.0, (30 - avg_trend_score) / 30 + 0.5)
            else:
                environment = self.OSCILLATING
                # 震荡市的置信度 - 接近50分时最高
                confidence = 1.0 - abs(avg_trend_score - 50) / 20
                confidence = max(0.4, confidence)  # 至少0.4的置信度
            
            result = {
                "environment": environment,
                "confidence": confidence,
                "trend_score": avg_trend_score,
                "lookback_days": lookback_days,
                "timestamp": datetime.now().isoformat()
            }
            
            # 缓存结果
            with self.lock:
                self.environment_cache = result
                
            return result
            
        except Exception as e:
            logger.error(f"分析市场环境失败: {str(e)}")
            return {
                "environment": self.UNDEFINED,
                "confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_market_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场技术指标
        
        Args:
            df: 指数日线数据DataFrame
            
        Returns:
            DataFrame: 添加了技术指标的数据
        """
        if df is None or len(df) < 20:
            return df
            
        try:
            # 确保日期索引并排序
            if 'trade_date' in df.columns and df.index.name != 'trade_date':
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
            df.sort_index(inplace=True)
            
            # 计算移动平均线
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma60'] = df['close'].rolling(window=60).mean()
            
            # 计算MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['hist'] = df['macd'] - df['signal']
            
            # 计算RSI
            delta = df['close'].diff()
            gain = delta.mask(delta < 0, 0)
            loss = -delta.mask(delta > 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算布林带
            df['boll_mid'] = df['close'].rolling(window=20).mean()
            df['boll_std'] = df['close'].rolling(window=20).std()
            df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
            df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
            
            # 计算成交量变化
            df['vol_change'] = df['vol'].pct_change()
            df['vol_ma10'] = df['vol'].rolling(window=10).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"计算市场指标失败: {str(e)}")
            return df
    
    def _calculate_trend_score(self, df: pd.DataFrame) -> float:
        """
        计算市场趋势得分 (0-100)
        
        Args:
            df: 带技术指标的指数DataFrame
            
        Returns:
            float: 趋势得分 (0-100)，0表示极度熊市，100表示极度牛市
        """
        if df is None or len(df) < 20:
            return 50.0  # 默认中性
            
        try:
            # 获取最近数据点
            recent = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 初始得分为50（中性）
            score = 50.0
            
            # 价格与移动平均线关系 (最多±15分)
            if recent['close'] > recent['ma60']:
                score += 10 * min(1, (recent['close'] / recent['ma60'] - 1) * 10)
            else:
                score -= 10 * min(1, (1 - recent['close'] / recent['ma60']) * 10)
                
            if recent['close'] > recent['ma20']:
                score += 5 * min(1, (recent['close'] / recent['ma20'] - 1) * 10)
            else:
                score -= 5 * min(1, (1 - recent['close'] / recent['ma20']) * 10)
                
            # 短期均线与长期均线关系 (最多±10分)
            if recent['ma5'] > recent['ma20']:
                score += 10 * min(1, (recent['ma5'] / recent['ma20'] - 1) * 10)
            else:
                score -= 10 * min(1, (1 - recent['ma5'] / recent['ma20']) * 10)
                
            # MACD指标 (最多±10分)
            if recent['hist'] > 0:
                score += 5
                if recent['hist'] > prev['hist']:
                    score += 5
            else:
                score -= 5
                if recent['hist'] < prev['hist']:
                    score -= 5
                    
            # RSI指标 (最多±10分)
            if recent['rsi'] > 50:
                # RSI强势
                score += 5
                if recent['rsi'] > 70:
                    # RSI超买
                    score += 5 * min(1, (recent['rsi'] - 70) / 10)
                else:
                    score += 5 * ((recent['rsi'] - 50) / 20)
            else:
                # RSI弱势
                score -= 5
                if recent['rsi'] < 30:
                    # RSI超卖
                    score -= 5 * min(1, (30 - recent['rsi']) / 10)
                else:
                    score -= 5 * ((50 - recent['rsi']) / 20)
            
            # 成交量变化 (最多±5分)
            if recent['vol'] > recent['vol_ma10']:
                score += 5 * min(1, (recent['vol'] / recent['vol_ma10'] - 1) * 2)
            else:
                score -= 5 * min(1, (1 - recent['vol'] / recent['vol_ma10']) * 2)
                
            # 布林带位置 (最多±10分)
            band_width = (recent['boll_upper'] - recent['boll_lower']) / recent['boll_mid']
            if recent['close'] > recent['boll_mid']:
                # 价格在布林带上方
                position = (recent['close'] - recent['boll_mid']) / (recent['boll_upper'] - recent['boll_mid'])
                score += 10 * min(1, position * 1.5)
            else:
                # 价格在布林带下方
                position = (recent['boll_mid'] - recent['close']) / (recent['boll_mid'] - recent['boll_lower'])
                score -= 10 * min(1, position * 1.5)
            
            # 确保得分在0-100之间
            score = max(0, min(100, score))
            
            return score
            
        except Exception as e:
            logger.error(f"计算趋势得分失败: {str(e)}")
            return 50.0
    
    def get_strategy_params(self, strategy_type: str) -> Dict[str, Any]:
        """
        根据市场环境返回策略参数
        
        Args:
            strategy_type: 策略类型
            
        Returns:
            Dict: 策略参数配置
        """
        # 获取当前市场环境
        env = self.environment_cache
        if not env or "environment" not in env:
            env = self.analyze_market_environment()
            
        market_type = env.get("environment", self.UNDEFINED)
        
        # 基于策略类型和市场环境返回合适的参数
        if strategy_type == "momentum":
            # 动量策略参数
            if market_type == self.BULL_MARKET:
                # 牛市参数 - 更激进
                return {
                    "lookback_period": 20,          # 回看天数
                    "threshold": 0.05,              # 动量阈值
                    "stop_loss": 0.08,              # 止损点
                    "take_profit": 0.15,            # 止盈点
                    "position_size": 0.1,           # 仓位大小
                    "trailing_stop": True,          # 启用追踪止损
                    "filter_threshold": 60          # 动量分数过滤阈值
                }
            elif market_type == self.BEAR_MARKET:
                # 熊市参数 - 更保守
                return {
                    "lookback_period": 10,          # 更短的回看期
                    "threshold": 0.08,              # 更高的动量阈值
                    "stop_loss": 0.05,              # 更紧的止损
                    "take_profit": 0.1,             # 更低的止盈目标
                    "position_size": 0.05,          # 更小的仓位 
                    "trailing_stop": True,          # 启用追踪止损
                    "filter_threshold": 75          # 更严格的过滤
                }
            else:
                # 震荡市参数 - 中性
                return {
                    "lookback_period": 15,          # 中等回看期
                    "threshold": 0.06,              # 中等动量阈值
                    "stop_loss": 0.07,              # 中等止损
                    "take_profit": 0.12,            # 中等止盈目标
                    "position_size": 0.07,          # 中等仓位
                    "trailing_stop": True,          # 启用追踪止损
                    "filter_threshold": 70          # 中等过滤
                }
        else:
            # 默认参数
            return {
                "lookback_period": 15,
                "threshold": 0.06,
                "stop_loss": 0.07,
                "take_profit": 0.12,
                "position_size": 0.07,
                "trailing_stop": True,
                "filter_threshold": 70
            }

class EnhancedMomentumAnalyzerV3:
    """增强版动量分析器 V3
    
    整合多项技术:
    1. 并行处理 - 提高性能
    2. 多级缓存 - 数据复用
    3. 机器学习模型 - 智能预测
    4. 市场环境分析 - 自适应策略
    5. 多技术指标 - 丰富分析维度
    """
    
    def __init__(self, max_workers: int = None, cache_dir: str = None,
                 load_ml_model: bool = True, ml_model_path: str = None):
        """
        初始化增强版动量分析器V3
        
        Args:
            max_workers: 最大工作线程/进程数，默认为CPU核心数+4
            cache_dir: 缓存目录，默认为ENHANCED_CACHE_DIR
            load_ml_model: 是否加载机器学习模型
            ml_model_path: 模型路径，None则使用默认模型
        """
        # 初始化配置
        self.max_workers = max_workers
        self.cache_dir = cache_dir or ENHANCED_CACHE_DIR
        self.data_cache = {}        # 数据缓存
        self.timestamp_cache = {}   # 带时间戳的缓存
        
        # 初始化基础分析器
        self.base_analyzer = MomentumAnalyzer(use_tushare=True)
        self.enhanced_analyzer = EnhancedMomentumAnalyzer(use_tushare=True)
        
        # 初始化市场环境分析器
        self.market_env = MarketEnvironment()
        
        # 初始化并行执行器
        self.parallel_executor = ParallelExecutor(
            mode=ParallelMode.THREAD,  # 默认使用线程模式，适合IO密集型任务
            max_workers=self.max_workers,
            timeout=300  # 5分钟超时
        )
        
        # 初始化机器学习模型
        self.ml_model = None
        self.ml_model_loaded = False
        
        if load_ml_model:
            try:
                # 如果提供了模型路径，尝试加载
                if ml_model_path and os.path.exists(ml_model_path):
                    self.ml_model = MomentumMLModel.load_model(ml_model_path)
                    if self.ml_model:
                        self.ml_model_loaded = True
                        logger.info(f"已加载动量预测模型: {ml_model_path}")
                else:
                    # 搜索默认目录下的模型
                    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
                    if os.path.exists(model_dir):
                        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'momentum' in f]
                        if model_files:
                            # 使用最新的模型
                            latest_model = sorted(model_files)[-1]
                            model_path = os.path.join(model_dir, latest_model)
                            self.ml_model = MomentumMLModel.load_model(model_path)
                            if self.ml_model:
                                self.ml_model_loaded = True
                                logger.info(f"已加载动量预测模型: {model_path}")
                            
                if not self.ml_model_loaded:
                    # 如果没有找到模型，创建一个新的随机森林模型
                    logger.warning("未找到预训练模型，创建新的随机森林模型")
                    self.ml_model = MomentumMLModel(model_type="random_forest")
            except Exception as e:
                logger.error(f"加载机器学习模型失败: {str(e)}")
                self.ml_model = None
    
    # 在这里只定义框架，具体实现将在下一步添加
    
    def train_ml_model(self, training_data: Optional[pd.DataFrame] = None, 
                      stock_codes: Optional[List[str]] = None, 
                      lookback_days: int = 365,
                      tune_hyperparams: bool = True) -> bool:
        """
        训练机器学习模型
        
        Args:
            training_data: 训练数据，如果为None则自动获取
            stock_codes: 用于训练的股票代码列表
            lookback_days: 回溯天数
            tune_hyperparams: 是否优化超参数
            
        Returns:
            bool: 训练是否成功
        """
        # 后续实现
        pass
    
    def analyze_stocks_v3(self, stock_list: pd.DataFrame, 
                         sample_size: int = 100, 
                         min_score: float = 60,
                         market_aware: bool = True) -> pd.DataFrame:
        """
        V3版本的股票分析方法
        
        Args:
            stock_list: 股票列表DataFrame
            sample_size: 样本大小，0表示分析全部
            min_score: 最低得分要求
            market_aware: 是否根据市场环境调整策略
            
        Returns:
            DataFrame: 分析结果
        """
        # 后续实现
        pass
    
    def calculate_multi_indicator_momentum(self, data: pd.DataFrame, 
                                          stock_code: str = None) -> Dict[str, Any]:
        """
        计算多指标动量得分
        
        Args:
            data: 股票数据DataFrame
            stock_code: 股票代码
            
        Returns:
            Dict: 动量分析结果
        """
        # 后续实现
        pass

# 全局缓存装饰器
def with_cache(ttl: int = 3600):
    """
    为方法添加缓存功能的装饰器
    
    Args:
        ttl: 缓存过期时间(秒)
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}_{args}_{kwargs}"
            
            # 检查缓存
            if hasattr(self, 'timestamp_cache') and cache_key in self.timestamp_cache:
                timestamp, data = self.timestamp_cache[cache_key]
                # 检查缓存是否过期
                if (datetime.now() - timestamp).total_seconds() < ttl:
                    return data
            
            # 执行原函数
            result = func(self, *args, **kwargs)
            
            # 更新缓存
            if hasattr(self, 'timestamp_cache'):
                self.timestamp_cache[cache_key] = (datetime.now(), result)
                
            return result
        return wrapper
    return decorator

if __name__ == "__main__":
    # 简单测试
    logger.info("初始化增强版动量分析器 V3")
    analyzer = EnhancedMomentumAnalyzerV3()
    
    # 获取市场环境
    market_env = analyzer.market_env.analyze_market_environment()
    logger.info(f"当前市场环境: {market_env['environment']}, 置信度: {market_env['confidence']:.2f}") 