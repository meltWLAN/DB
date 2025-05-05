import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, List
import logging

class FeatureEngineer:
    """特征工程类,负责特征处理和标准化"""
    
    def __init__(self, feature_columns: List[str]):
        """
        初始化特征工程类
        
        Args:
            feature_columns: 特征列名列表
        """
        self.logger = logging.getLogger(__name__)
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def prepare_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        准备特征数据
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            处理后的特征数组,如果处理失败返回None
        """
        try:
            # 检查必要特征列
            missing_cols = [col for col in self.feature_columns if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"缺失特征列: {missing_cols}")
                return None
            
            # 提取基础特征
            features = data[self.feature_columns].copy()
            
            # 计算衍生特征
            features['price_momentum'] = features['close'].pct_change(5)
            features['volume_momentum'] = features['volume'].pct_change(5)
            
            # 处理缺失值
            features = features.fillna(method='ffill').fillna(0)
            
            # 标准化特征
            if not self._is_fitted:
                self.scaler.fit(features)
                self._is_fitted = True
            
            features_scaled = self.scaler.transform(features)
            
            return features_scaled
            
        except Exception as e:
            self.logger.error(f"特征处理失败: {str(e)}")
            return None
    
    def reset(self):
        """重置特征工程器状态"""
        self.scaler = StandardScaler()
        self._is_fitted = False 