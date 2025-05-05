import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import numba
from concurrent.futures import ThreadPoolExecutor
import logging

class Optimizer:
    """性能优化工具类"""
    
    def __init__(self, logger: logging.Logger):
        """
        初始化优化器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger
    
    @staticmethod
    @numba.jit(nopython=True)
    def calculate_momentum_numba(prices: np.ndarray, periods: int) -> np.ndarray:
        """
        使用Numba加速动量计算
        
        Args:
            prices: 价格数组
            periods: 周期
            
        Returns:
            动量数组
        """
        n = len(prices)
        momentum = np.zeros(n)
        for i in range(periods, n):
            momentum[i] = (prices[i] / prices[i-periods] - 1) * 100
        return momentum
    
    def parallel_feature_calculation(self, data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        并行计算特征
        
        Args:
            data: 输入数据
            features: 特征列表
            
        Returns:
            计算后的特征DataFrame
        """
        def calculate_feature(feature: str) -> pd.Series:
            if feature == 'price_momentum':
                return pd.Series(self.calculate_momentum_numba(data['close'].values, 5))
            elif feature == 'volume_momentum':
                return pd.Series(self.calculate_momentum_numba(data['volume'].values, 5))
            return data[feature]
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(calculate_feature, features))
        
        return pd.concat(results, axis=1)
    
    def optimize_memory_usage(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        优化内存使用
        
        Args:
            data: 输入数据
            
        Returns:
            优化后的DataFrame
        """
        # 降低数值精度
        for col in data.select_dtypes(include=['float64']).columns:
            data[col] = data[col].astype('float32')
        
        # 使用分类类型
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].nunique() / len(data) < 0.5:
                data[col] = data[col].astype('category')
        
        return data
    
    def batch_processing(self, data: pd.DataFrame, batch_size: int = 1000) -> List[pd.DataFrame]:
        """
        批量处理数据
        
        Args:
            data: 输入数据
            batch_size: 批次大小
            
        Returns:
            批次数据列表
        """
        return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    
    def cache_optimization(self, func):
        """
        优化缓存使用
        
        Args:
            func: 要优化的函数
            
        Returns:
            优化后的函数
        """
        cache = {}
        
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]
        
        return wrapper
    
    def feature_selection(self, data: pd.DataFrame, target: pd.Series, threshold: float = 0.01) -> List[str]:
        """
        特征选择优化
        
        Args:
            data: 特征数据
            target: 目标变量
            threshold: 相关性阈值
            
        Returns:
            选择的特征列表
        """
        correlations = data.corrwith(target)
        return correlations[abs(correlations) > threshold].index.tolist()
    
    def data_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理优化
        
        Args:
            data: 输入数据
            
        Returns:
            预处理后的数据
        """
        # 处理缺失值
        data = data.fillna(method='ffill').fillna(0)
        
        # 处理异常值
        for col in data.select_dtypes(include=['float32', 'float64']).columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            data[col] = data[col].clip(lower=q1-1.5*iqr, upper=q3+1.5*iqr)
        
        return data 