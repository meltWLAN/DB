"""
动量预测模型
使用机器学习技术预测股票未来动量和价格趋势
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import joblib
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MomentumPredictor:
    """动量预测模型类"""
    
    def __init__(self, prediction_horizon=5, use_classification=True, model_dir='./models'):
        """
        初始化动量预测模型
        
        Args:
            prediction_horizon: 预测未来天数
            use_classification: 是否使用分类模型(True)还是回归模型(False)
            model_dir: 模型保存目录
        """
        self.prediction_horizon = prediction_horizon
        self.use_classification = use_classification
        self.model_dir = model_dir
        
        # 创建模型目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 初始化模型和缩放器
        self.model = None
        self.scaler = StandardScaler()
        
        # 特征工程配置
        self.feature_columns = None
        self.target_column = None
        
        # 最后训练和预测时间
        self.last_train_time = None
        self.last_predict_time = None
        
        # 模型性能指标
        self.model_metrics = {}
    
    def _generate_target(self, data, column='close'):
        """
        生成目标变量
        
        Args:
            data: 股票数据DataFrame
            column: 用于计算目标的价格列名
            
        Returns:
            DataFrame: 添加目标列的数据
        """
        df = data.copy()
        
        # 计算未来n日收益率
        future_return = df[column].shift(-self.prediction_horizon) / df[column] - 1
        
        if self.use_classification:
            # 分类目标：1=上涨，0=下跌
            df['target'] = (future_return > 0).astype(int)
        else:
            # 回归目标：未来收益率
            df['target'] = future_return
        
        # 去除无效行
        df = df.dropna(subset=['target'])
        
        self.target_column = 'target'
        return df
    
    def _engineer_features(self, data):
        """
        特征工程
        
        Args:
            data: 股票数据DataFrame
            
        Returns:
            DataFrame: 处理后的特征数据
        """
        df = data.copy()
        
        # 清理缺失数据
        df = df.dropna()
        
        # 提取可能的特征列
        price_columns = ['open', 'high', 'low', 'close']
        available_price_cols = [col for col in price_columns if col in df.columns]
        
        # 技术指标列
        tech_indicators = []
        for col in df.columns:
            # 找出所有可能的技术指标列
            if any(indicator in col for indicator in ['ma', 'ema', 'rsi', 'macd', 'momentum', 'vol_ratio', 'boll']):
                tech_indicators.append(col)
        
        # 价格形态和趋势指标
        pattern_indicators = []
        for col in df.columns:
            if any(pattern in col for pattern in ['golden_cross', 'cross', 'is_local']):
                pattern_indicators.append(col)
        
        # 合并所有特征列
        feature_cols = available_price_cols + tech_indicators + pattern_indicators
        
        # 添加派生特征
        
        # 1. 价格与均线关系
        if 'close' in df.columns:
            for ma in [f'ma{n}' for n in [5, 10, 20, 60]]:
                if ma in df.columns:
                    df[f'close_to_{ma}_ratio'] = df['close'] / df[ma]
                    feature_cols.append(f'close_to_{ma}_ratio')
        
        # 2. 近期波动率
        if 'close' in df.columns:
            for window in [5, 10, 20]:
                df[f'volatility_{window}d'] = df['close'].pct_change().rolling(window=window).std()
                feature_cols.append(f'volatility_{window}d')
        
        # 3. 技术指标交叉特征
        if 'rsi_14' in df.columns and 'macd' in df.columns:
            df['rsi_macd_interaction'] = df['rsi_14'] * df['macd']
            feature_cols.append('rsi_macd_interaction')
        
        # 存储特征列
        self.feature_columns = feature_cols
        
        # 处理缺失值
        df = df.dropna(subset=feature_cols + [self.target_column])
        
        return df
    
    def prepare_data(self, data, column='close'):
        """
        准备训练和测试数据
        
        Args:
            data: 股票数据DataFrame
            column: 用于计算目标的价格列名
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # 生成目标变量
        data = self._generate_target(data, column)
        
        # 特征工程
        data = self._engineer_features(data)
        
        # 分离特征和目标
        X = data[self.feature_columns]
        y = data[self.target_column]
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def train(self, data, column='close', optimize_hyperparams=False):
        """
        训练模型
        
        Args:
            data: 股票数据DataFrame
            column: 用于计算目标的价格列名
            optimize_hyperparams: 是否优化超参数
            
        Returns:
            dict: 模型性能指标
        """
        logger.info(f"开始训练{'分类' if self.use_classification else '回归'}模型，预测周期：{self.prediction_horizon}天")
        
        try:
            # 准备数据
            X_train, X_test, y_train, y_test = self.prepare_data(data, column)
            
            # 创建模型
            if self.use_classification:
                # 分类模型
                if optimize_hyperparams:
                    # 超参数网格搜索
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                    model = GridSearchCV(RandomForestClassifier(random_state=42), 
                                        param_grid, cv=5, scoring='f1')
                    model.fit(X_train, y_train)
                    self.model = model.best_estimator_
                    logger.info(f"最佳超参数: {model.best_params_}")
                else:
                    # 默认参数
                    self.model = RandomForestClassifier(n_estimators=200, random_state=42)
                    self.model.fit(X_train, y_train)
                
                # 评估模型
                y_pred = self.model.predict(X_test)
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred)
                }
                
                logger.info(f"模型评估: 准确率={metrics['accuracy']:.4f}, 精确率={metrics['precision']:.4f}, "
                           f"召回率={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
            else:
                # 回归模型
                if optimize_hyperparams:
                    # 超参数网格搜索
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.05, 0.1]
                    }
                    model = GridSearchCV(GradientBoostingRegressor(random_state=42), 
                                        param_grid, cv=5, scoring='neg_mean_squared_error')
                    model.fit(X_train, y_train)
                    self.model = model.best_estimator_
                    logger.info(f"最佳超参数: {model.best_params_}")
                else:
                    # 默认参数
                    self.model = GradientBoostingRegressor(n_estimators=200, random_state=42)
                    self.model.fit(X_train, y_train)
                
                # 评估模型
                y_pred = self.model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'avg_target': y_test.mean(),  # 目标均值，用于比较RMSE
                    'rmse_ratio': rmse / abs(y_test.mean())  # RMSE与目标均值的比率
                }
                
                logger.info(f"模型评估: MSE={metrics['mse']:.6f}, RMSE={metrics['rmse']:.6f}, "
                           f"RMSE/平均目标={metrics['rmse_ratio']:.4f}")
            
            # 保存最后训练时间和指标
            self.last_train_time = datetime.now()
            self.model_metrics = metrics
            
            # 计算特征重要性
            try:
                feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
                # 按重要性排序
                sorted_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                          key=lambda x: x[1], reverse=True)}
                
                # 只保留前10个最重要的特征
                top_features = dict(list(sorted_importance.items())[:10])
                logger.info(f"前10个最重要特征: {top_features}")
                
                # 添加到指标中
                metrics['feature_importance'] = sorted_importance
            except AttributeError:
                logger.warning("模型不支持特征重要性计算")
            
            return metrics
            
        except Exception as e:
            logger.error(f"训练模型失败: {str(e)}", exc_info=True)
            return {}
    
    def predict(self, data):
        """
        使用模型进行预测
        
        Args:
            data: 用于预测的DataFrame，需包含模型训练时的所有特征
            
        Returns:
            numpy.ndarray: 预测结果
        """
        if self.model is None:
            logger.error("模型尚未训练，无法进行预测")
            return None
        
        try:
            # 特征工程
            if self.feature_columns is None:
                logger.error("特征列未定义，请先训练模型")
                return None
            
            # 确保数据包含所有必要的特征
            missing_features = [f for f in self.feature_columns if f not in data.columns]
            if missing_features:
                logger.error(f"数据缺少必要的特征: {missing_features}")
                return None
            
            # 提取特征
            X = data[self.feature_columns]
            
            # 处理缺失值
            if X.isna().any().any():
                logger.warning("数据包含缺失值，将进行填充")
                X = X.fillna(X.mean())
            
            # 标准化特征
            X_scaled = self.scaler.transform(X)
            
            # 预测
            if self.use_classification:
                # 分类预测及概率
                predictions = self.model.predict(X_scaled)
                probabilities = self.model.predict_proba(X_scaled)
                
                # 构建结果
                result = {
                    'prediction': predictions,
                    'probability': probabilities[:, 1],  # 上涨的概率
                    'prediction_type': 'classification',
                    'prediction_horizon': self.prediction_horizon
                }
            else:
                # 回归预测
                predictions = self.model.predict(X_scaled)
                
                # 构建结果
                result = {
                    'prediction': predictions,
                    'prediction_type': 'regression',
                    'prediction_horizon': self.prediction_horizon
                }
            
            # 更新最后预测时间
            self.last_predict_time = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}", exc_info=True)
            return None
    
    def predict_momentum_score(self, data, weights=None):
        """
        预测未来动量得分
        
        Args:
            data: 用于预测的DataFrame
            weights: 预测结果转化为动量得分的权重
            
        Returns:
            float: 预测的动量得分(0-100)
        """
        if weights is None:
            # 默认权重
            weights = {'probability': 1.0} if self.use_classification else {'prediction': 1.0}
        
        try:
            # 获取预测结果
            result = self.predict(data)
            
            if result is None:
                return 50  # 默认中性分数
            
            # 根据预测类型计算动量得分
            if self.use_classification:
                # 分类模型：根据上涨概率计算得分
                probability = result['probability'][-1]  # 最后一个数据点的上涨概率
                score = probability * 100
            else:
                # 回归模型：根据预测收益转化为得分
                prediction = result['prediction'][-1]  # 最后一个数据点的预测收益
                
                # 将预测收益映射到0-100的得分
                # 假设预测收益的合理范围是-10%到+10%
                score = 50 + prediction * 500  # 50 + 收益率*500
                score = min(100, max(0, score))  # 限制在0-100范围内
            
            return score
            
        except Exception as e:
            logger.error(f"计算预测动量得分失败: {str(e)}", exc_info=True)
            return 50  # 默认中性分数
    
    def save_model(self, filename=None):
        """
        保存模型到文件
        
        Args:
            filename: 文件名，默认使用日期和模型类型
            
        Returns:
            str: 保存的文件路径
        """
        if self.model is None:
            logger.error("模型尚未训练，无法保存")
            return None
        
        try:
            # 生成默认文件名
            if filename is None:
                model_type = "classification" if self.use_classification else "regression"
                date_str = datetime.now().strftime("%Y%m%d")
                filename = f"momentum_predictor_{model_type}_{self.prediction_horizon}d_{date_str}.joblib"
            
            # 完整路径
            filepath = os.path.join(self.model_dir, filename)
            
            # 保存模型和相关参数
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'use_classification': self.use_classification,
                'prediction_horizon': self.prediction_horizon,
                'last_train_time': self.last_train_time,
                'model_metrics': self.model_metrics
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"模型已保存到 {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}", exc_info=True)
            return None
    
    def load_model(self, filepath):
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            # 加载模型数据
            model_data = joblib.load(filepath)
            
            # 还原模型和参数
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            self.use_classification = model_data['use_classification']
            self.prediction_horizon = model_data['prediction_horizon']
            self.last_train_time = model_data['last_train_time']
            self.model_metrics = model_data['model_metrics']
            
            logger.info(f"已从 {filepath} 加载模型")
            
            # 记录模型类型和指标
            model_type = "分类" if self.use_classification else "回归"
            metrics_str = ""
            if self.use_classification:
                metrics_str = f", F1={self.model_metrics.get('f1', 0):.4f}"
            else:
                metrics_str = f", RMSE={self.model_metrics.get('rmse', 0):.6f}"
                
            logger.info(f"加载的模型 - 类型: {model_type}, 预测周期: {self.prediction_horizon}天" + metrics_str)
            
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}", exc_info=True)
            return False 