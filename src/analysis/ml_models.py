#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动量分析机器学习模型

提供机器学习模型来预测股票动量趋势，支持多种算法
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings

# 抑制scikit-learn的警告
warnings.filterwarnings("ignore")

# 设置日志
logger = logging.getLogger(__name__)

try:
    # 尝试导入机器学习相关包
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"机器学习依赖项导入失败: {str(e)}")
    logger.warning("请使用'pip install scikit-learn>=1.0.0'安装机器学习库")
    ML_AVAILABLE = False

# 模型保存目录
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class MomentumMLModel:
    """动量预测机器学习模型
    
    使用技术指标数据预测股票未来价格趋势
    """
    
    AVAILABLE_MODELS = {
        "random_forest": "随机森林模型",
        "xgboost": "梯度提升模型",
        "logistic": "逻辑回归模型",
        "svm": "支持向量机模型",
        "ensemble": "集成模型(多模型综合)"
    }
    
    def __init__(self, model_type: str = "random_forest", model_params: Optional[Dict] = None):
        """初始化动量ML模型
        
        Args:
            model_type: 模型类型，可选random_forest, xgboost, logistic, svm, ensemble
            model_params: 模型参数字典
        """
        if not ML_AVAILABLE:
            logger.error("机器学习功能不可用，请安装scikit-learn库")
            return
            
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.feature_scaler = None
        self.is_trained = False
        self.feature_importance = None
        self.model_metrics = {}
        self.best_params = {}
        
        # 为模型分配唯一ID
        self.model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 创建基础模型实例
        self._create_model()
        
    def _create_model(self):
        """创建机器学习模型实例"""
        if not ML_AVAILABLE:
            return None
            
        try:
            if self.model_type == "random_forest":
                # 随机森林默认参数
                defaults = {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 10,
                    "random_state": 42
                }
                params = {**defaults, **self.model_params}
                self.model = RandomForestClassifier(**params)
                
            elif self.model_type == "xgboost":
                # 梯度提升默认参数
                defaults = {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5, 
                    "random_state": 42
                }
                params = {**defaults, **self.model_params}
                self.model = GradientBoostingClassifier(**params)
                
            elif self.model_type == "logistic":
                # 逻辑回归默认参数
                defaults = {
                    "C": 1.0,
                    "penalty": "l2",
                    "solver": "liblinear",
                    "random_state": 42
                }
                params = {**defaults, **self.model_params}
                self.model = LogisticRegression(**params)
                
            elif self.model_type == "svm":
                # SVM默认参数
                defaults = {
                    "C": 1.0,
                    "kernel": "rbf",
                    "probability": True,
                    "random_state": 42
                }
                params = {**defaults, **self.model_params}
                self.model = SVC(**params)
                
            elif self.model_type == "ensemble":
                # 创建多个模型的集合
                self.models = {
                    "rf": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                    "gb": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
                    "lr": LogisticRegression(C=1.0, random_state=42)
                }
                # 主模型仍使用随机森林
                self.model = self.models["rf"]
                
            else:
                logger.warning(f"不支持的模型类型: {self.model_type}，使用默认的随机森林")
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                self.model_type = "random_forest"
                
            # 创建特征缩放器
            self.feature_scaler = StandardScaler()
            
            logger.info(f"已创建{self.AVAILABLE_MODELS.get(self.model_type, '未知')}实例")
            return self.model
        
        except Exception as e:
            logger.error(f"创建模型失败: {str(e)}")
            return None
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备特征和标签数据
        
        Args:
            df: 带有技术指标的股票数据DataFrame
            
        Returns:
            Tuple: (特征矩阵, 标签向量)
        """
        if not ML_AVAILABLE or df.empty:
            return np.array([]), np.array([])
            
        try:
            # 移除不需要的列
            feature_columns = [col for col in df.columns if col not in 
                             ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 
                              'vol', 'amount', 'future_return', 'future_price', 'adjustflag']]
            
            # 确保所有特征都是数值类型
            X = df[feature_columns].copy()
            
            # 创建标签：未来N天收益是否为正
            future_days = 5  # 默认预测5日后的涨跌
            if 'future_return' not in df.columns:
                # 如果没有未来收益列，创建一个
                df['future_close'] = df['close'].shift(-future_days)
                df['future_return'] = (df['future_close'] - df['close']) / df['close']
            
            # 创建二分类标签：1表示上涨，0表示不变或下跌
            y = (df['future_return'] > 0).astype(int)
            
            # 删除含有NaN的行
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_idx]
            y = y[valid_idx]
            
            # 特征缩放
            if len(X) > 0:
                X_values = X.values
                return X_values, y.values
            else:
                return np.array([]), np.array([])
                
        except Exception as e:
            logger.error(f"准备特征数据失败: {str(e)}")
            return np.array([]), np.array([])
    
    def train(self, train_data: pd.DataFrame, tune_hyperparams: bool = False, test_size: float = 0.2) -> bool:
        """
        训练动量预测模型
        
        Args:
            train_data: 训练数据，包含技术指标
            tune_hyperparams: 是否优化超参数
            test_size: 测试集比例
            
        Returns:
            bool: 训练是否成功
        """
        if not ML_AVAILABLE or self.model is None:
            logger.error("机器学习功能不可用或模型未初始化")
            return False
            
        try:
            # 准备特征和标签
            X, y = self.prepare_features(train_data)
            if len(X) == 0 or len(y) == 0:
                logger.error("训练数据为空或无效")
                return False
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=True
            )
            
            # 特征缩放
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # 如果需要，进行超参数优化
            if tune_hyperparams:
                if self.model_type == "random_forest":
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 15, None],
                        'min_samples_split': [2, 5, 10]
                    }
                elif self.model_type == "xgboost":
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                elif self.model_type == "logistic":
                    param_grid = {
                        'C': [0.1, 1, 10, 100],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear']
                    }
                elif self.model_type == "svm":
                    param_grid = {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale', 'auto', 0.1, 1]
                    }
                else:
                    param_grid = {}
                    
                if param_grid:
                    logger.info(f"开始{self.AVAILABLE_MODELS.get(self.model_type, '未知')}超参数优化...")
                    grid = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy')
                    grid.fit(X_train_scaled, y_train)
                    self.model = grid.best_estimator_
                    self.best_params = grid.best_params_
                    logger.info(f"超参数优化完成: {self.best_params}")
                
            # 训练模型
            logger.info(f"开始训练{self.AVAILABLE_MODELS.get(self.model_type, '未知')}...")
            self.model.fit(X_train_scaled, y_train)
            
            # 对于集成模型，训练所有子模型
            if self.model_type == "ensemble":
                for name, model in self.models.items():
                    if name != "rf":  # 随机森林已经训练过了
                        model.fit(X_train_scaled, y_train)
            
            # 评估模型性能
            train_preds = self.model.predict(X_train_scaled)
            test_preds = self.model.predict(X_test_scaled)
            
            # 计算各种指标
            self.model_metrics = {
                "train_accuracy": accuracy_score(y_train, train_preds),
                "test_accuracy": accuracy_score(y_test, test_preds),
                "test_precision": precision_score(y_test, test_preds, zero_division=0),
                "test_recall": recall_score(y_test, test_preds, zero_division=0),
                "test_f1": f1_score(y_test, test_preds, zero_division=0)
            }
            
            # 对于支持概率预测的模型，计算ROC AUC
            if hasattr(self.model, "predict_proba"):
                try:
                    test_probs = self.model.predict_proba(X_test_scaled)[:, 1]
                    self.model_metrics["test_roc_auc"] = roc_auc_score(y_test, test_probs)
                except Exception as e:
                    logger.warning(f"计算ROC AUC时出错: {str(e)}")
            
            # 记录特征重要性（如果模型支持）
            if hasattr(self.model, "feature_importances_"):
                self.feature_importance = self.model.feature_importances_
                
            self.is_trained = True
            
            logger.info(f"模型训练完成！测试集准确率: {self.model_metrics['test_accuracy']:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        使用模型预测股票未来趋势
        
        Args:
            data: 包含技术指标的股票数据
            
        Returns:
            Dict: 预测结果，包含预测分类和概率
        """
        if not ML_AVAILABLE or not self.is_trained:
            logger.error("模型未训练或机器学习功能不可用")
            return {"prediction": 0, "probability": 0.5, "error": "模型未训练"}
            
        try:
            # 准备特征
            X, _ = self.prepare_features(data)
            if len(X) == 0:
                return {"prediction": 0, "probability": 0.5, "error": "无有效特征数据"}
            
            # 获取最新的数据点
            last_record = X[-1].reshape(1, -1)
            
            # 数据缩放
            scaled_record = self.feature_scaler.transform(last_record)
            
            # 预测分类
            prediction = int(self.model.predict(scaled_record)[0])
            
            # 预测概率（如果模型支持）
            probability = 0.5
            if hasattr(self.model, "predict_proba"):
                probability = float(self.model.predict_proba(scaled_record)[0, 1])
                
            # 对于集成模型，综合所有子模型的预测
            if self.model_type == "ensemble":
                ensemble_probs = []
                for name, model in self.models.items():
                    if hasattr(model, "predict_proba"):
                        prob = float(model.predict_proba(scaled_record)[0, 1])
                        ensemble_probs.append(prob)
                        
                if ensemble_probs:
                    # 使用平均概率
                    probability = np.mean(ensemble_probs)
                    # 重新判断预测分类
                    prediction = 1 if probability > 0.5 else 0
            
            return {
                "prediction": prediction,  # 1=上涨, 0=下跌
                "probability": probability,  # 上涨概率
                "model_type": self.model_type,
                "confidence": abs(probability - 0.5) * 2  # 将概率转换为0-1的置信度
            }
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            return {"prediction": 0, "probability": 0.5, "error": str(e)}
    
    def get_top_features(self, feature_names: List[str] = None, top_n: int = 10) -> Dict[str, float]:
        """
        获取最重要的特征
        
        Args:
            feature_names: 特征名称列表
            top_n: 返回前N个重要特征
            
        Returns:
            Dict: 重要特征及其得分
        """
        if not self.is_trained or self.feature_importance is None:
            return {}
            
        try:
            if feature_names is None or len(feature_names) != len(self.feature_importance):
                # 如果没有提供特征名称，使用索引
                feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]
                
            # 创建特征重要性字典
            importance_dict = dict(zip(feature_names, self.feature_importance))
            
            # 按重要性排序并返回前N个
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_features[:top_n])
            
        except Exception as e:
            logger.error(f"获取重要特征失败: {str(e)}")
            return {}
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        保存模型到文件
        
        Args:
            filepath: 保存路径，默认使用模型ID创建文件
            
        Returns:
            str: 保存的文件路径
        """
        if not self.is_trained:
            logger.error("无法保存未训练的模型")
            return ""
            
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, f"momentum_model_{self.model_id}.pkl")
            
        try:
            # 创建保存对象
            save_data = {
                "model": self.model,
                "model_type": self.model_type,
                "scaler": self.feature_scaler,
                "is_trained": self.is_trained,
                "feature_importance": self.feature_importance,
                "metrics": self.model_metrics,
                "best_params": self.best_params,
                "model_id": self.model_id,
                "saved_at": datetime.now().isoformat()
            }
            
            # 如果是集成模型，保存所有子模型
            if self.model_type == "ensemble":
                save_data["sub_models"] = self.models
                
            # 保存到文件
            with open(filepath, "wb") as f:
                pickle.dump(save_data, f)
                
            logger.info(f"模型已保存至: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            return ""
    
    @classmethod
    def load_model(cls, filepath: str) -> "MomentumMLModel":
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            MomentumMLModel: 加载的模型实例
        """
        if not ML_AVAILABLE:
            logger.error("机器学习功能不可用，无法加载模型")
            return None
            
        try:
            # 从文件加载
            with open(filepath, "rb") as f:
                save_data = pickle.load(f)
                
            # 创建新实例
            instance = cls(model_type=save_data["model_type"])
            
            # 恢复模型属性
            instance.model = save_data["model"]
            instance.feature_scaler = save_data["scaler"]
            instance.is_trained = save_data["is_trained"]
            instance.feature_importance = save_data["feature_importance"]
            instance.model_metrics = save_data["metrics"]
            instance.best_params = save_data["best_params"]
            instance.model_id = save_data["model_id"]
            
            # 对于集成模型，恢复子模型
            if save_data["model_type"] == "ensemble" and "sub_models" in save_data:
                instance.models = save_data["sub_models"]
                
            logger.info(f"已从{filepath}加载模型")
            return instance
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return None
            
    def generate_model_report(self) -> Dict[str, Any]:
        """
        生成模型报告
        
        Returns:
            Dict: 包含模型信息的报告
        """
        if not self.is_trained:
            return {"error": "模型未训练，无法生成报告"}
            
        report = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "model_name": self.AVAILABLE_MODELS.get(self.model_type, "未知模型"),
            "is_trained": self.is_trained,
            "metrics": self.model_metrics,
            "best_params": self.best_params,
            "timestamp": datetime.now().isoformat()
        }
        
        # 添加特征重要性，如果可用
        if self.feature_importance is not None:
            # 根据重要性排序
            feature_indices = np.argsort(self.feature_importance)[::-1]
            top_importance = self.feature_importance[feature_indices[:10]]
            report["top_features"] = list(zip(
                [f"feature_{i}" for i in feature_indices[:10]], 
                top_importance.tolist()
            ))
            
        return report

# 简单测试
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 如果机器学习库不可用，退出测试
    if not ML_AVAILABLE:
        logger.warning("测试中止: 机器学习功能不可用")
        import sys
        sys.exit(0)
        
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    
    # 创建一些随机特征
    X1 = np.random.randn(n_samples) * 2  # 标准差为2的正态分布
    X2 = np.random.randn(n_samples)      # 标准差为1的正态分布
    X3 = np.random.rand(n_samples) * 10  # 0-10的均匀分布
    
    # 创建目标变量：简单的线性组合加噪声
    y = (X1 > 0.5) & (X2 > 0) & (X3 > 5)
    y = y.astype(int)
    
    # 创建DataFrame
    df = pd.DataFrame({
        "feature_1": X1,
        "feature_2": X2, 
        "feature_3": X3,
        "future_return": y
    })
    
    # 创建并训练模型
    model = MomentumMLModel(model_type="random_forest")
    model.train(df, tune_hyperparams=True)
    
    # 测试预测
    test_data = df.sample(5)
    predictions = model.predict(test_data)
    
    print(f"预测结果: {predictions}")
    print(f"模型报告: {model.generate_model_report()}")
    
    # 保存和加载测试
    save_path = model.save_model()
    if save_path:
        loaded_model = MomentumMLModel.load_model(save_path)
        if loaded_model:
            loaded_predictions = loaded_model.predict(test_data)
            print(f"加载模型预测结果: {loaded_predictions}") 