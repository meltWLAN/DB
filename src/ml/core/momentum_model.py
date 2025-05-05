import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import hashlib
import pickle

from ..config.model_config import ModelConfig
from ..features.feature_engineer import FeatureEngineer
from ..models.model_manager import ModelManager
from ..utils.logger import Logger

class MomentumMLModel:
    """动量分析机器学习模型主类"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        初始化动量分析模型
        
        Args:
            config: 模型配置,如果为None则使用默认配置
        """
        self.config = config or ModelConfig()
        
        # 设置日志
        self.logger = Logger.setup_logger(
            name=__name__,
            log_path=self.config.log_path,
            level=self.config.log_level
        )
        
        # 初始化组件
        self.feature_engineer = FeatureEngineer(self.config.feature_columns)
        self.model_manager = ModelManager(self.config.model_params)
        
        # 加载模型
        if not self.model_manager.load_models(self.config.model_path):
            self.logger.warning("模型加载失败,需要重新训练")
    
    def predict(self, data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        使用模型进行预测
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            各模型的预测结果字典,如果预测失败返回None
        """
        try:
            # 准备特征
            features = self.feature_engineer.prepare_features(data)
            if features is None:
                self.logger.error("特征准备失败")
                return None
            
            # 计算特征哈希值用于缓存
            features_hash = hashlib.md5(pickle.dumps(features)).hexdigest()
            
            # 使用模型进行预测
            predictions = self.model_manager.predict(features_hash, features)
            if not predictions:
                self.logger.error("模型预测失败")
                return None
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"预测过程出错: {str(e)}")
            return None
    
    def train(self, train_data: pd.DataFrame, labels: np.ndarray) -> bool:
        """
        训练新模型
        
        Args:
            train_data: 训练数据
            labels: 训练标签
            
        Returns:
            是否训练成功
        """
        try:
            # 准备特征
            features = self.feature_engineer.prepare_features(train_data)
            if features is None:
                self.logger.error("训练数据特征准备失败")
                return False
            
            # 训练模型
            for model_name, params in self.config.model_params.items():
                model = self.model_manager.models[model_name]
                if model is not None:
                    model.fit(features, labels)
            
            # 保存模型
            if not self.model_manager.save_models(self.config.model_path):
                self.logger.error("模型保存失败")
                return False
            
            self.logger.info("模型训练完成")
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return False 