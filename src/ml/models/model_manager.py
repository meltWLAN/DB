import joblib
import logging
from typing import Dict, Optional, Any
from sklearn.base import BaseEstimator
from functools import lru_cache

class ModelManager:
    """模型管理类,负责模型的加载、保存和预测"""
    
    def __init__(self, model_params: Dict[str, Dict[str, Any]]):
        """
        初始化模型管理器
        
        Args:
            model_params: 模型参数字典
        """
        self.logger = logging.getLogger(__name__)
        self.model_params = model_params
        self.models: Dict[str, Optional[BaseEstimator]] = {
            name: None for name in model_params.keys()
        }
    
    def load_models(self, model_path: str) -> bool:
        """
        加载预训练模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            是否成功加载
        """
        try:
            for model_name in self.models.keys():
                model_file = f"{model_path}{model_name}_momentum.joblib"
                self.models[model_name] = joblib.load(model_file)
            self.logger.info("成功加载所有模型")
            return True
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            return False
    
    def save_models(self, model_path: str) -> bool:
        """
        保存训练好的模型
        
        Args:
            model_path: 模型保存路径
            
        Returns:
            是否成功保存
        """
        try:
            for model_name, model in self.models.items():
                if model is not None:
                    joblib.dump(model, f"{model_path}{model_name}_momentum.joblib")
            self.logger.info("成功保存所有模型")
            return True
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
            return False
    
    @lru_cache(maxsize=1000)
    def predict(self, features_hash: str, features: Any) -> Dict[str, float]:
        """
        使用模型进行预测(带缓存)
        
        Args:
            features_hash: 特征数据的哈希值
            features: 特征数据
            
        Returns:
            各模型的预测结果字典
        """
        try:
            predictions = {}
            for model_name, model in self.models.items():
                if model is not None:
                    pred_proba = model.predict_proba(features)[-1]
                    score = pred_proba[1] * 100
                    predictions[model_name] = score
                    self.logger.debug(f"{model_name}模型预测评分: {score:.2f}")
            return predictions
        except Exception as e:
            self.logger.error(f"模型预测失败: {str(e)}")
            return {} 