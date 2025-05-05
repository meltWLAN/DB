#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
机器学习预测模型
用于预测股票未来价格走势，支持多种机器学习算法
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings
import os
import pickle
from pathlib import Path

# 导入机器学习相关库
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    # Add imports for XGBoost and LightGBM
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GridSearchCV
except ImportError:
    warnings.warn("未安装scikit-learn, xgboost, 或 lightgbm库，部分功能可能不可用")

# 导入系统模块
try:
    # Adjust relative import path if necessary, assuming logger is in src/utils
    from src.utils.logger import get_logger
    logger = get_logger("ml_prediction_model")
except (ImportError, ValueError):
    # Fallback basic logger if import fails
    logger = logging.getLogger("ml_prediction_model")
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

class MLPredictionModel:
    """机器学习预测模型"""

    def __init__(self, prediction_horizon=5, model_type="ensemble",
                model_dir=None):
        """
        初始化预测模型

        Args:
            prediction_horizon: 预测天数，默认5天
            model_type: 模型类型，可选值: 'rf'（随机森林），'gbm'（梯度提升），
                      'svm'（支持向量机）, 'xgb' (XGBoost), 'lgb' (LightGBM)
                      或'ensemble'（集成所有模型）
            model_dir: 保存预训练模型的目录
        """
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type

        # 设置模型目录
        if model_dir is None:
            # Correct path assuming this file is in src/strategies
            current_dir = Path(__file__).parent
            self.model_dir = current_dir.parent / "models"
        else:
            self.model_dir = Path(model_dir)

        # 确保模型目录存在
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"模型目录设置为: {self.model_dir.resolve()}")


        # 初始化模型
        self.models = {}
        self.feature_importances = {}
        self.scaler = StandardScaler() # Scaler should be fit on training data

        # 初始化特征列表 (可以根据实际指标动态调整)
        # Keep a base list, can be updated later if needed
        self.feature_columns = [
            # Price & Volume related
            'open', 'high', 'low', 'close', 'volume',
            # Moving Averages (Price & Volume)
            'ma5', 'ma10', 'ma20', 'ma60',
            'volume_ma5', 'volume_ma10', 'volume_ma20',
            # Oscillators & Momentum
            'rsi6', 'rsi12', 'rsi24',
            'macd', 'macd_signal', 'macd_hist',
            'cci', 'stoch_k', 'stoch_d',
            # Volatility
            'atr', 'boll_bandwidth',
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            # Trend
            'adx',
            # Volume
            'obv', 'cmf',
            # Returns
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            # Sentiment/Positioning (if available from indicators)
            'close_position', 'bulls_power', 'bears_power'
            # Add other indicators calculated in AdvancedIndicators if needed
        ]
        # Ensure boll_upper/lower are present if needed for bandwidth calc
        # It's better calculated in indicators module directly.

        # 检查是否有预训练模型
        self._load_pretrained_models()

    def _get_model_path(self, model_key):
        """Helper to get model file path"""
        return self.model_dir / f"{model_key}_model_{self.prediction_horizon}d.pkl"

    def _load_pretrained_models(self):
        """加载预训练模型，如果存在"""
        # Include xgb and lgb models
        model_keys = ['rf', 'gbm', 'svm', 'xgb', 'lgb']
        loaded_models = []

        for key in model_keys:
            model_path = self._get_model_path(key)
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        self.models[key] = model_data['model']
                        # Load scaler if saved with model (recommended)
                        if 'scaler' in model_data:
                            self.scaler = model_data['scaler']
                        # Load features if saved with model (recommended)
                        if 'features' in model_data:
                            self.feature_columns = model_data['features']
                        # Load feature importances if available
                        if 'feature_importances' in model_data:
                             self.feature_importances[key] = model_data['feature_importances']
                        loaded_models.append(key)
                except Exception as e:
                    logger.error(f"加载模型 {model_path} 时出错: {e}")
        if loaded_models:
            logger.info(f"成功加载预训练模型: {', '.join(loaded_models)}")
        else:
             logger.warning("未找到任何预训练模型")


        # Load scaler separately if not saved with models (less ideal)
        scaler_path = self.model_dir / "scaler.pkl"
        if scaler_path.exists() and 'scaler' not in self.models.get(loaded_models[0] if loaded_models else '', {}):
             try:
                 with open(scaler_path, 'rb') as f:
                     self.scaler = pickle.load(f)
                 logger.info("成功加载独立的 scaler")
             except Exception as e:
                 logger.error(f"加载 scaler {scaler_path} 时出错: {e}")


    def _prepare_features(self, data):
        """
        准备预测所需的特征
        
        Args:
            data: 股票历史数据
            
        Returns:
            处理后的特征DataFrame
        """
        try:
            # 检查数据是否为空
            if data is None or len(data) < 10:
                logger.error("数据样本不足，无法准备特征")
                return pd.DataFrame()
                
            # 确保数据是DataFrame类型
            df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 提取最新的记录用于预测
            features = self._extract_features(df)
            
            # 检查特征是否包含模型所需的所有列
            if self.feature_columns is not None:
                missing_cols = [col for col in self.feature_columns if col not in features.columns]
                
                if missing_cols:
                    # 尝试生成缺失列，例如通过技术指标计算
                    logger.warning(f"缺少特征列: {missing_cols}，尝试自动生成")
                    features = self._generate_missing_features(df, missing_cols, features)
                    
                    # 再次检查
                    missing_cols = [col for col in self.feature_columns if col not in features.columns]
                    if missing_cols:
                        logger.error(f"无法生成全部缺失特征: {missing_cols}")
                        # 使用零填充缺失列
                        for col in missing_cols:
                            features[col] = 0
                
                # 确保特征仅包含预训练模型使用的列，且顺序一致
                if len(self.feature_columns) > 0:
                    # 首先检查是否有多余的列
                    extra_cols = [col for col in features.columns if col not in self.feature_columns]
                    if extra_cols:
                        logger.warning(f"移除多余特征列: {extra_cols}")
                        # 只保留模型训练时使用的特征列
                        features = features[self.feature_columns]
                    elif features.columns.tolist() != self.feature_columns:
                        # 如果列存在但顺序不同，则调整顺序
                        logger.info("调整特征列顺序以匹配训练模型")
                        features = features[self.feature_columns]
            
            return features
            
        except Exception as e:
            logger.error(f"准备特征时出错: {str(e)}")
            return pd.DataFrame()
            
    def _extract_features(self, df):
        """从原始数据中提取预测特征"""
        try:
            # 从完整历史数据中计算所需的特征
            # 复制数据以避免修改原始数据
            df_copy = df.copy()
            
            # 确保数据中包含基本的技术指标
            missing_indicators = False
            for col in ['rsi14', 'macd', 'ma20', 'ma50']:
                if col not in df_copy.columns:
                    missing_indicators = True
                    break
                    
            if missing_indicators:
                from src.indicators.advanced_indicators import AdvancedIndicators
                df_copy = AdvancedIndicators.add_advanced_indicators(df_copy)
            
            # 计算基础特征
            features = pd.DataFrame(index=df_copy.index)
            
            # 价格指标
            if 'close' in df_copy.columns:
                # 收盘价与移动平均线相对关系
                if 'ma20' in df_copy.columns:
                    features['close_to_ma20'] = df_copy['close'] / df_copy['ma20']
                if 'ma60' in df_copy.columns:
                    features['close_to_ma60'] = df_copy['close'] / df_copy['ma60']
                
                # 布林带相对位置
                if all(col in df_copy.columns for col in ['boll_upper', 'boll_lower', 'boll_middle']):
                    features['boll_position'] = (df_copy['close'] - df_copy['boll_lower']) / (df_copy['boll_upper'] - df_copy['boll_lower'])
                    features['boll_width'] = (df_copy['boll_upper'] - df_copy['boll_lower']) / df_copy['boll_middle']
            
            # 动量指标
            for col in ['rsi14', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'cci', 'williams_r']:
                if col in df_copy.columns:
                    features[col] = df_copy[col]
            
            # 交易量指标
            if 'volume' in df_copy.columns:
                # 归一化成交量
                if 'volume_ma20' in df_copy.columns:
                    features['volume_ratio'] = df_copy['volume'] / df_copy['volume_ma20']
                else:
                    # 计算20日成交量均值
                    vol_ma20 = df_copy['volume'].rolling(window=20).mean()
                    features['volume_ratio'] = df_copy['volume'] / vol_ma20
            
            # 价格动量指标
            if 'close' in df_copy.columns:
                # 5日、10日、20日价格变化率
                for days in [5, 10, 20]:
                    if len(df_copy) > days:
                        features[f'price_momentum_{days}d'] = df_copy['close'] / df_copy['close'].shift(days) - 1
            
            # 注意：这里不再使用删除NaN的方法，而是使用填充方法，保持索引不变
            # 使用向后填充方法填充NaN值
            features = features.fillna(method='bfill')
            # 任何剩余的NaN值用0填充
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"提取特征时出错: {str(e)}")
            return pd.DataFrame(index=df.index)
            
    def _generate_missing_features(self, df, missing_cols, features):
        """尝试生成缺失的特征列"""
        # 此方法可以根据业务需求自定义扩展
        return features

    def _create_target(self, df):
        """创建预测目标变量"""
        # 预测未来N天的价格是否上涨
        df['future_price'] = df['close'].shift(-self.prediction_horizon)
        df['target'] = np.where(df['future_price'] > df['close'], 1, 0)
        df.dropna(subset=['future_price'], inplace=True) # Drop rows where future price is unknown
        return df['target']

    def train(self, df):
        """
        使用提供的数据训练模型

        Args:
            df: 包含价格、成交量和技术指标的DataFrame
        """
        logger.info(f"开始训练模型，预测期: {self.prediction_horizon}天, 模型类型: {self.model_type}")
        if df is None or df.empty:
            logger.error("训练数据为空，无法训练模型")
            return

        # 1. 准备特征和目标
        try:
            target = self._create_target(df)
            features = self._prepare_features(df.loc[target.index]) # Align features with target index
        except Exception as e:
            logger.error(f"准备特征或目标时出错: {e}")
            logger.error(traceback.format_exc())
            return

        if features.empty or target.empty:
             logger.error("准备特征或目标后数据为空，无法训练")
             return

        # Align features and target after potential drops
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]

        if features.empty or target.empty:
            logger.error("特征和目标对齐后数据为空，无法训练")
            return

        logger.info(f"用于训练的数据大小: {features.shape}")

        # 2. 划分训练集和测试集 (Optional but recommended for evaluation)
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )

        # 3. 特征缩放 (Fit only on training data)
        try:
             logger.info("拟合特征缩放器 (StandardScaler)")
             self.scaler.fit(X_train)
             X_train_scaled = self.scaler.transform(X_train)
             X_test_scaled = self.scaler.transform(X_test)
             logger.info("特征缩放完成")
        except Exception as e:
             logger.error(f"特征缩放时出错: {e}")
             logger.error(traceback.format_exc())
             # Decide how to proceed: maybe train without scaling?
             X_train_scaled = X_train.values # Use numpy arrays
             X_test_scaled = X_test.values
             #return # Or exit if scaling is crucial

        # 4. 定义和训练模型
        models_to_train = []
        if self.model_type == 'ensemble':
            models_to_train = ['rf', 'gbm', 'xgb', 'lgb', 'svm'] # Add new models
        elif self.model_type in ['rf', 'gbm', 'svm', 'xgb', 'lgb']:
            models_to_train = [self.model_type]
        else:
            logger.error(f"不支持的模型类型: {self.model_type}，将训练集成模型")
            models_to_train = ['rf', 'gbm', 'xgb', 'lgb', 'svm']


        # Model definitions (consider adding hyperparameters)
        model_definitions = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'gbm': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42, class_weight='balanced'), # SVM needs probability=True for predict_proba
            'xgb': xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42),
            'lgb': lgb.LGBMClassifier(random_state=42, class_weight='balanced')
        }

        for key in models_to_train:
            if key not in model_definitions:
                logger.warning(f"模型 '{key}' 未定义，跳过训练。")
                continue

            try:
                logger.info(f"开始训练 {key.upper()} 模型...")
                model = model_definitions[key]
                model.fit(X_train_scaled, y_train)
                self.models[key] = model
                logger.info(f"成功训练 {key.upper()} 模型")

                # 评估模型 (可选)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                logger.info(f"{key.upper()} 模型评估: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

                 # 获取特征重要性 (if applicable)
                if hasattr(model, 'feature_importances_'):
                     # Match importances with original feature names
                     importances = pd.Series(model.feature_importances_, index=features.columns)
                     self.feature_importances[key] = importances.sort_values(ascending=False)
                     logger.info(f"{key.upper()} Top 5 Features:\n{self.feature_importances[key].head()}")
                elif key == 'svm' and hasattr(model, 'coef_'):
                     # For linear SVM, coef_ indicates importance
                     importances = pd.Series(np.abs(model.coef_[0]), index=features.columns)
                     self.feature_importances[key] = importances.sort_values(ascending=False)
                     logger.info(f"{key.upper()} Top 5 Feature Coefficients (abs):\n{self.feature_importances[key].head()}")


            except Exception as e:
                logger.error(f"训练 {key.upper()} 模型时出错: {e}")
                logger.error(traceback.format_exc())
                if key in self.models:
                    del self.models[key] # Remove partially trained model

        # 5. 保存模型和Scaler
        self.save_model(scaler_to_save=self.scaler, features_to_save=features.columns.tolist())


    def predict(self, data):
        """
        使用合适的模型对数据进行预测
        
        Args:
            data: 股票历史数据，包含必要的技术指标
            
        Returns:
            包含预测结果的DataFrame
        """
        try:
            logger.info(f"开始预测，使用模型: {self.model_type}, 数据大小: {data.shape}")
            
            # 准备特征
            X = self._prepare_features(data)
            
            # 检查预处理是否成功
            if X is None or X.empty:
                logger.error("特征准备失败，无法进行预测")
                return None
                
            # 执行预测
            if self.model_type == 'ensemble':
                # 对每个子模型分别进行预测
                return self._ensemble_predict(X, data)
            else:
                # 使用单个模型进行预测
                return self._single_model_predict(X, data, self.model_type)
        
        except Exception as e:
            logger.error(f"预测过程中出错: {str(e)}")
            return None
            
    def _ensemble_predict(self, X, data):
        """使用集成方法进行预测"""
        logger.info(f"将使用以下模型进行预测: {', '.join(self.models.keys())}")
        
        results = pd.DataFrame(index=data.index)
        all_predictions = {}
        all_probas = {}
        
        # 对每个子模型分别进行预测
        for model_name, model in self.models.items():
            try:
                # 跳过未加载的模型
                if model is None:
                    continue
                    
                # 使用模型进行预测
                pred = model.predict(X)
                
                # 获取预测概率
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[:, 1]  # 取正类的概率
                else:
                    # 使用决策函数的输出作为替代
                    decision = model.decision_function(X) if hasattr(model, 'decision_function') else None
                    proba = 1 / (1 + np.exp(-decision)) if decision is not None else pred
                
                all_predictions[model_name] = pred
                all_probas[model_name] = proba
                
            except Exception as e:
                logger.warning(f"使用模型 '{model_name}' 预测时出错: {str(e)}")
        
        # 如果所有模型都失败，则返回空结果
        if not all_predictions:
            logger.error("所有模型预测均失败")
            return None
            
        # 使用自定义投票权重
        model_weights = {
            'rf': 1.0,
            'gbm': 1.2,
            'svm': 0.8,
            'xgb': 1.3,
            'lgb': 1.2
        }
            
        # 计算加权平均概率
        weighted_proba = np.zeros(len(X))
        total_weight = 0
        
        for model_name, proba in all_probas.items():
            weight = model_weights.get(model_name, 1.0)  # 默认权重为1
            weighted_proba += proba * weight
            total_weight += weight
            
        if total_weight > 0:
            weighted_proba /= total_weight
        
        # 通过阈值转换为预测类别
        # 默认阈值为0.5，但可以根据上下文调整
        final_prediction = (weighted_proba >= 0.5).astype(int)
        
        # 创建结果DataFrame
        results['prediction'] = final_prediction
        results['confidence'] = weighted_proba
        
        # 记录完成
        logger.info("集成预测完成")
        
        return results

    def _single_model_predict(self, X, data, model_type):
        """使用单个模型进行预测"""
        model = self.models.get(model_type)
        if model is None:
            logger.error(f"模型 '{model_type}' 未训练或加载，无法进行预测")
            return None

        try:
            # 检查预测数据是否包含所有训练时使用的特征列
            missing_pred_cols = [col for col in self.feature_columns if col not in X.columns]
            if missing_pred_cols:
                logger.error(f"预测数据缺少训练时的关键特征列: {missing_pred_cols}. 无法进行可靠预测。")
                logger.error("这通常是因为上游指标计算失败。请检查 AdvancedIndicators 的日志。")
                # Fill missing columns with 0 just before scaling to avoid scaler error, 
                # but the prediction quality will be compromised.
                # Alternatively, return None here directly.
                # For now, let's try filling and proceed with caution:
                logger.warning(f"用0填充缺失的预测特征: {missing_pred_cols}")
                for col in missing_pred_cols:
                    X[col] = 0
                # Ensure the order is correct after filling
                X = X[self.feature_columns]
                # return None # Returning None might be safer depending on requirements
            else:
                # Ensure feature order matches training if all columns are present
                X = X[self.feature_columns] # Reorder/select based on training

            # Check if scaler is fitted
            if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
                logger.error("Scaler 未被拟合，无法进行预测。请先训练模型。")
                # Attempt to load scaler again as a fallback
                scaler_path = self.model_dir / "scaler.pkl"
                if scaler_path.exists():
                    try:
                        with open(scaler_path, 'rb') as f:
                            self.scaler = pickle.load(f)
                        logger.info("成功加载独立的 scaler 用于预测")
                        if not hasattr(self.scaler, 'mean_'): # Check again
                            raise ValueError("Loaded scaler is not fitted.")
                    except Exception as e:
                        logger.error(f"加载 scaler {scaler_path} 失败: {e}. 无法继续预测。")
                        return None
                else:
                    logger.error(f"未找到已拟合的 scaler 文件 {scaler_path}。无法继续预测。")
                    return None

            X_scaled = self.scaler.transform(X)
            logger.info(f"特征缩放完成，准备预测，特征数量: {X_scaled.shape[1]}")

            # 使用模型进行预测
            pred = model.predict(X_scaled)
            proba = model.predict_proba(X_scaled)[:, 1]

            # 创建结果DataFrame
            results = pd.DataFrame(index=data.index)
            results['prediction'] = pred
            results['confidence'] = proba

            logger.info(f"模型 '{model_type}' 预测完成")
            return results

        except Exception as e:
            logger.error(f"使用单个模型进行预测时出错: {e}")
            logger.error(traceback.format_exc())
            return None

    def save_model(self, scaler_to_save=None, features_to_save=None):
        """保存训练好的模型和Scaler"""
        if not self.models:
            logger.warning("没有训练好的模型可供保存。")
            return

        for key, model in self.models.items():
            model_path = self._get_model_path(key)
            try:
                model_data = {'model': model}
                if scaler_to_save:
                     model_data['scaler'] = scaler_to_save # Save scaler with each model
                if features_to_save:
                     model_data['features'] = features_to_save # Save feature list used
                if key in self.feature_importances:
                     model_data['feature_importances'] = self.feature_importances[key]

                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                logger.info(f"模型 {key} 已保存到 {model_path}")
            except Exception as e:
                logger.error(f"保存模型 {key} 到 {model_path} 时出错: {e}")

        # Save scaler separately as well (optional fallback)
        if scaler_to_save:
             scaler_path = self.model_dir / "scaler.pkl"
             try:
                 with open(scaler_path, 'wb') as f:
                     pickle.dump(scaler_to_save, f)
                 logger.info(f"Scaler 已保存到 {scaler_path}")
             except Exception as e:
                 logger.error(f"保存 Scaler 到 {scaler_path} 时出错: {e}")


    def get_feature_importance(self, model_key=None):
        """
        获取指定模型的特征重要性

        Args:
            model_key: 模型标识符 ('rf', 'gbm', 'xgb', 'lgb', 'svm').
                       如果为 None, 尝试返回集成模型的平均重要性（如果计算）。

        Returns:
            Pandas Series 包含特征重要性，降序排列。如果模型不支持或未计算则返回 None。
        """
        if model_key:
            if model_key in self.feature_importances:
                return self.feature_importances[model_key]
            else:
                logger.warning(f"模型 '{model_key}' 没有可用的特征重要性数据。")
                # Maybe try to calculate it on the fly if model exists? (Complex)
                return None
        elif self.model_type == 'ensemble':
             # Try to compute average importance across models that have it
             all_importances = [imp for imp in self.feature_importances.values() if isinstance(imp, pd.Series)]
             if len(all_importances) > 0:
                 # Ensure all series have the same index (features) before averaging
                 common_features = set(all_importances[0].index)
                 for imp in all_importances[1:]:
                     common_features.intersection_update(imp.index)

                 avg_importance = pd.DataFrame({i: imp.reindex(list(common_features)).fillna(0)
                                                for i, imp in enumerate(all_importances) if imp is not None}).mean(axis=1)
                 return avg_importance.sort_values(ascending=False)
             else:
                 logger.warning("集成模型中没有可用的特征重要性数据用于平均。")
                 return None
        else:
            # Try returning importance for the single specified model type
             if self.model_type in self.feature_importances:
                 return self.feature_importances[self.model_type]
             else:
                 logger.warning(f"当前模型类型 '{self.model_type}' 没有特征重要性数据。")
                 return None

# Helper function (can be moved to utils if needed)
import traceback

# Example Usage (for testing)
if __name__ == '__main__':
    # Create dummy data for testing
    dates = pd.date_range(start='2023-01-01', periods=200, freq='B')
    data = pd.DataFrame(index=dates)
    data['open'] = np.random.rand(200) * 100 + 50
    data['high'] = data['open'] * (1 + np.random.rand(200) * 0.05)
    data['low'] = data['open'] * (1 - np.random.rand(200) * 0.05)
    data['close'] = (data['high'] + data['low']) / 2 + np.random.randn(200)
    data['volume'] = np.random.randint(100000, 1000000, 200)

    # Add dummy indicators matching feature_columns
    def add_dummy_indicators(df, features):
        for col in features:
            if col not in df.columns:
                if 'ma' in col or 'rsi' in col or 'ema' in col or 'boll' in col or 'cci' in col or 'stoch' in col or 'adx' in col:
                    df[col] = np.random.rand(len(df)) * 100
                elif 'volume_ma' in col or 'obv' in col:
                     df[col] = np.random.rand(len(df)) * 10000
                elif 'return' in col or 'volatility' in col or 'atr' in col or 'cmf' in col:
                    df[col] = np.random.rand(len(df)) * 0.1 - 0.05
                elif 'power' in col:
                     df[col] = np.random.rand(len(df)) * 10 - 5
                else: # Default for any other missing
                     df[col] = np.random.rand(len(df))
        return df

    # --- Test Training ---
    print("--- Testing Model Training ---")
    # Use ensemble for testing all integrated models
    model = MLPredictionModel(prediction_horizon=5, model_type='ensemble')
    # Add dummy indicators required by the model's feature_columns
    test_features = model.feature_columns
    data_with_indicators = add_dummy_indicators(data.copy(), test_features)

    # Ensure necessary columns exist even if dummy ones were added
    if 'close' not in data_with_indicators.columns: data_with_indicators['close'] = 50

    model.train(data_with_indicators)

    # --- Test Prediction ---
    print("\n--- Testing Model Prediction ---")
    # Create new dummy data for prediction
    pred_dates = pd.date_range(start='2023-11-01', periods=10, freq='B')
    pred_data = pd.DataFrame(index=pred_dates)
    pred_data['open'] = np.random.rand(10) * 100 + 100
    pred_data['high'] = pred_data['open'] * (1 + np.random.rand(10) * 0.03)
    pred_data['low'] = pred_data['open'] * (1 - np.random.rand(10) * 0.03)
    pred_data['close'] = (pred_data['high'] + pred_data['low']) / 2 + np.random.randn(10)
    pred_data['volume'] = np.random.randint(100000, 1000000, 10)

    pred_data_with_indicators = add_dummy_indicators(pred_data.copy(), test_features)

    predictions = model.predict(pred_data_with_indicators)

    if predictions is not None:
        print("\nPrediction Results:")
        print(predictions)
    else:
        print("\nPrediction failed.")

    # --- Test Feature Importance ---
    print("\n--- Testing Feature Importance ---")
    importance_rf = model.get_feature_importance('rf')
    if importance_rf is not None:
         print("\nRandom Forest Feature Importance (Top 10):")
         print(importance_rf.head(10))

    importance_xgb = model.get_feature_importance('xgb')
    if importance_xgb is not None:
         print("\nXGBoost Feature Importance (Top 10):")
         print(importance_xgb.head(10))

    importance_ensemble = model.get_feature_importance() # Test ensemble importance
    if importance_ensemble is not None:
        print("\nEnsemble Average Feature Importance (Top 10):")
        print(importance_ensemble.head(10))