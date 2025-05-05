import unittest
import numpy as np
import pandas as pd
from ..core.momentum_model import MomentumMLModel
from ..config.model_config import ModelConfig

class TestMomentumMLModel(unittest.TestCase):
    """动量模型测试类"""
    
    def setUp(self):
        """测试准备"""
        self.config = ModelConfig()
        self.model = MomentumMLModel(self.config)
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'close': np.random.rand(100),
            'volume': np.random.rand(100),
            'amount': np.random.rand(100),
            'ma5': np.random.rand(100),
            'ma10': np.random.rand(100),
            'ma20': np.random.rand(100),
            'ma30': np.random.rand(100),
            'ma60': np.random.rand(100),
            'rsi': np.random.rand(100),
            'macd': np.random.rand(100),
            'macd_signal': np.random.rand(100),
            'macd_hist': np.random.rand(100),
            'boll_upper': np.random.rand(100),
            'boll_middle': np.random.rand(100),
            'boll_lower': np.random.rand(100),
            'k': np.random.rand(100),
            'd': np.random.rand(100),
            'j': np.random.rand(100),
            'obv': np.random.rand(100),
            'cci': np.random.rand(100),
            'atr': np.random.rand(100),
            'dmi_plus': np.random.rand(100),
            'dmi_minus': np.random.rand(100)
        })
        
        self.test_labels = np.random.randint(0, 2, 100)
    
    def test_predict(self):
        """测试预测功能"""
        predictions = self.model.predict(self.test_data)
        self.assertIsNotNone(predictions)
        self.assertIn('rf', predictions)
        self.assertIn('gbdt', predictions)
    
    def test_train(self):
        """测试训练功能"""
        success = self.model.train(self.test_data, self.test_labels)
        self.assertTrue(success)
    
    def test_invalid_data(self):
        """测试无效数据处理"""
        invalid_data = pd.DataFrame()
        predictions = self.model.predict(invalid_data)
        self.assertIsNone(predictions)
    
    def test_missing_features(self):
        """测试缺失特征处理"""
        data_missing = self.test_data.drop(columns=['close'])
        predictions = self.model.predict(data_missing)
        self.assertIsNone(predictions)

if __name__ == '__main__':
    unittest.main() 