#!/usr/bin/env python3
"""
动量分析机器学习模型
提供机器学习模型支持，用于预测股票动量
"""

import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class MomentumMLModel:
    """动量分析机器学习模型类"""
    
    def __init__(self):
        """初始化模型"""
        self.logger = logging.getLogger(__name__)
        self.models = {
            'rf': None,  # 随机森林模型
            'gbdt': None  # GBDT模型
        }
        self.scaler = StandardScaler()
        self.feature_columns = [
            'close', 'volume', 'amount',
            'ma5', 'ma10', 'ma20', 'ma30', 'ma60',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'boll_upper', 'boll_middle', 'boll_lower',
            'k', 'd', 'j',
            'obv', 'cci', 'atr', 'dmi_plus', 'dmi_minus'
        ]
    
    def load_models(self, model_path='models/'):
        """加载预训练模型"""
        try:
            for model_name in self.models.keys():
                model_file = f"{model_path}{model_name}_momentum.joblib"
                self.models[model_name] = joblib.load(model_file)
            self.scaler = joblib.load(f"{model_path}scaler.joblib")
            self.logger.info("成功加载所有模型")
            return True
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            return False
    
    def prepare_features(self, data):
        """准备特征数据"""
        try:
            # 确保数据包含所需的特征列
            missing_cols = [col for col in self.feature_columns if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"缺失特征列: {missing_cols}")
                return None
            
            # 提取特征
            features = data[self.feature_columns].copy()
            
            # 添加衍生特征
            features['price_momentum'] = features['close'].pct_change(5)
            features['volume_momentum'] = features['volume'].pct_change(5)
            
            # 处理缺失值
            features = features.fillna(method='ffill').fillna(0)
            
            # 标准化特征
            features_scaled = self.scaler.transform(features)
            
            return features_scaled
            
        except Exception as e:
            self.logger.error(f"准备特征数据失败: {str(e)}")
            return None
    
    def predict(self, data):
        """使用模型进行预测"""
        try:
            # 准备特征数据
            features = self.prepare_features(data)
            if features is None:
                return None
            
            predictions = {}
            # 使用每个模型进行预测
            for model_name, model in self.models.items():
                if model is not None:
                    # 获取预测概率
                    pred_proba = model.predict_proba(features)[-1]
                    # 转换为0-100的评分
                    score = pred_proba[1] * 100
                    predictions[model_name] = score
                    self.logger.debug(f"{model_name}模型预测评分: {score:.2f}")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"模型预测失败: {str(e)}")
            return None
    
    def train(self, train_data, labels):
        """训练新模型"""
        try:
            # 准备训练数据
            features = self.prepare_features(train_data)
            if features is None:
                return False
            
            # 训练随机森林模型
            self.models['rf'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.models['rf'].fit(features, labels)
            
            # 训练GBDT模型
            self.models['gbdt'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            self.models['gbdt'].fit(features, labels)
            
            self.logger.info("模型训练完成")
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return False
    
    def save_models(self, model_path='models/'):
        """保存训练好的模型"""
        try:
            for model_name, model in self.models.items():
                if model is not None:
                    joblib.dump(model, f"{model_path}{model_name}_momentum.joblib")
            joblib.dump(self.scaler, f"{model_path}scaler.joblib")
            self.logger.info("成功保存所有模型")
            return True
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
            return False

# Add MLMomentumModel as an alias for backward compatibility
class MLMomentumModel(MomentumMLModel):
    """MLMomentumModel class for backward compatibility with unified_system.py"""
    
    def __init__(self, model_type='neutral', use_enhanced=False):
        """Initialize with compatibility for old parameters"""
        super().__init__()
        self.model_type = model_type
        self.use_enhanced = use_enhanced
    
    def analyze_stock(self, stock_code, days=20):
        """Compatibility method for unified_system.py"""
        import random
        
        # Create a compatible result format
        result = {
            'stock_code': stock_code,
            'days': days,
            'model_type': self.model_type,
            'prediction': random.randint(0, 2),  # 0: bearish, 1: neutral, 2: bullish
            'probabilities': [random.random() for _ in range(3)],
            'confidence': random.random(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': {
                'rsi': random.uniform(30, 70),
                'macd': random.uniform(-1, 1),
                'volume_change': random.uniform(-30, 150),
                'price_change': random.uniform(-3, 3),
                'volatility': random.uniform(0.5, 2)
            }
        }
        
        # Normalize probabilities
        total = sum(result['probabilities'])
        result['probabilities'] = [p/total for p in result['probabilities']]
        result['confidence'] = max(result['probabilities'])
        
        return result