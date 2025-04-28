"""
自适应权重系统
根据不同市场环境和历史表现动态调整技术指标的权重
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AdaptiveWeightSystem:
    """自适应权重系统类"""
    
    def __init__(self, base_weights=None, learning_rate=0.05, history_length=90, market_regime_windows=[20, 60]):
        """
        初始化自适应权重系统
        
        Args:
            base_weights: 基础指标权重字典
            learning_rate: 权重调整学习率
            history_length: 用于学习的历史天数
            market_regime_windows: 市场状态判断窗口大小列表
        """
        # 设置默认基础权重
        self.base_weights = base_weights or {
            'ma_score': 0.15,       # 移动平均线权重
            'momentum_score': 0.25, # 价格动量权重
            'rsi_score': 0.15,      # RSI权重
            'macd_score': 0.15,     # MACD权重
            'kdj_score': 0.15,      # KDJ权重
            'volume_score': 0.10,   # 成交量权重
            'price_pattern': 0.05   # 价格形态权重
        }
        
        # 复制一份当前权重
        self.current_weights = self.base_weights.copy()
        
        # 设置学习参数
        self.learning_rate = learning_rate
        self.history_length = history_length
        self.market_regime_windows = market_regime_windows
        
        # 历史表现记录
        self.performance_history = []
        
        # 机器学习模型
        self.model = None
        
        # 状态变量
        self.last_update = None
        self.last_market_regime = "unknown"
    
    def _normalize_weights(self, weights):
        """
        归一化权重，确保总和为1
        
        Args:
            weights: 权重字典
            
        Returns:
            dict: 归一化后的权重字典
        """
        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        return weights
    
    def _determine_market_regime(self, market_data, column='close'):
        """
        确定当前市场状态
        
        Args:
            market_data: 市场数据DataFrame
            column: 用于分析的价格列名
            
        Returns:
            str: 市场状态描述字符串
        """
        if market_data.empty or len(market_data) < max(self.market_regime_windows):
            return "unknown"
        
        # 计算不同时间窗口的趋势
        trends = {}
        volatilities = {}
        
        for window in self.market_regime_windows:
            if len(market_data) < window:
                continue
                
            # 获取窗口数据
            window_data = market_data[column].iloc[-window:]
            
            # 计算线性回归斜率
            x = np.arange(len(window_data))
            y = window_data.values
            slope = np.polyfit(x, y, 1)[0]
            
            # 计算标准化趋势强度(相对于均值的变化率)
            trend_strength = slope * window / np.mean(window_data)
            
            # 计算波动率
            returns = window_data.pct_change().dropna()
            volatility = returns.std()
            
            trends[window] = trend_strength
            volatilities[window] = volatility
        
        # 综合不同窗口的趋势来确定市场状态
        short_trend = trends.get(min(self.market_regime_windows), 0)
        long_trend = trends.get(max(self.market_regime_windows), 0)
        avg_volatility = np.mean(list(volatilities.values()))
        
        # 判断趋势方向
        if short_trend > 0.01 and long_trend > 0.005:
            trend_direction = "uptrend"
        elif short_trend < -0.01 and long_trend < -0.005:
            trend_direction = "downtrend"
        else:
            trend_direction = "sideways"
        
        # 判断波动程度
        if avg_volatility > 0.02:  # 2%日波动率
            volatility_level = "high_volatility"
        elif avg_volatility > 0.01:  # 1%日波动率
            volatility_level = "medium_volatility"
        else:
            volatility_level = "low_volatility"
        
        # 判断趋势与短期趋势的关系
        if abs(short_trend - long_trend) > 0.01:
            if short_trend > long_trend:
                trend_change = "accelerating"
            else:
                trend_change = "decelerating"
        else:
            trend_change = "steady"
        
        # 组合成市场状态描述
        return f"{trend_direction}_{volatility_level}_{trend_change}"
    
    def _get_optimal_weights_for_regime(self, market_regime):
        """
        获取指定市场状态的最优权重
        
        Args:
            market_regime: 市场状态
            
        Returns:
            dict: 该市场状态下的优化权重
        """
        # 过滤该市场状态下的历史表现记录
        regime_history = [record for record in self.performance_history 
                        if record["market_regime"] == market_regime]
        
        if not regime_history:
            logger.info(f"没有市场状态 '{market_regime}' 的历史记录，使用基本权重")
            return self.base_weights.copy()
        
        # 根据历史表现调整权重
        optimal_weights = {}
        
        # 按照性能对记录排序
        sorted_history = sorted(regime_history, key=lambda x: x["performance"], reverse=True)
        
        # 获取表现最好的权重集合
        best_weights = sorted_history[0]["weights"]
        
        return best_weights
    
    def _train_ml_model(self):
        """
        训练机器学习模型来预测最优权重
        
        Returns:
            bool: 是否成功训练模型
        """
        if len(self.performance_history) < 10:  # 需要足够的历史数据
            logger.info("历史记录不足，无法训练模型")
            return False
        
        try:
            # 准备训练数据
            X = []  # 特征：市场指标
            y = []  # 目标：最优权重向量
            
            for record in self.performance_history:
                # 提取市场特征
                market_features = record.get("market_features", {})
                if not market_features:
                    continue
                    
                # 转换为特征向量
                feature_vector = [
                    market_features.get("trend_short", 0),
                    market_features.get("trend_long", 0),
                    market_features.get("volatility", 0),
                    market_features.get("volume_change", 0),
                    market_features.get("rsi_level", 50),
                    market_features.get("macd_signal", 0)
                ]
                
                # 提取权重向量
                weight_vector = [record["weights"].get(k, self.base_weights.get(k, 0)) 
                              for k in self.base_weights.keys()]
                
                X.append(feature_vector)
                y.append(weight_vector)
            
            # 确保数据充足
            if len(X) < 10:
                logger.info("有效数据记录不足，无法训练模型")
                return False
            
            # 转换为numpy数组
            X = np.array(X)
            y = np.array(y)
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 训练随机森林回归模型
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"权重预测模型训练完成 - MSE: {mse:.4f}, R²: {r2:.4f}")
            
            # 保存模型
            self.model = model
            return True
            
        except Exception as e:
            logger.error(f"训练权重预测模型失败: {str(e)}")
            return False
    
    def _predict_weights(self, market_features):
        """
        使用机器学习模型预测最优权重
        
        Args:
            market_features: 市场特征字典
            
        Returns:
            dict: 预测的权重字典，如果预测失败则返回当前权重
        """
        if self.model is None:
            return self.current_weights
        
        try:
            # 转换为特征向量
            feature_vector = np.array([
                market_features.get("trend_short", 0),
                market_features.get("trend_long", 0),
                market_features.get("volatility", 0),
                market_features.get("volume_change", 0),
                market_features.get("rsi_level", 50),
                market_features.get("macd_signal", 0)
            ]).reshape(1, -1)
            
            # 预测权重向量
            weight_vector = self.model.predict(feature_vector)[0]
            
            # 转换为权重字典并确保非负
            predicted_weights = {k: max(0, float(w)) for k, w in 
                               zip(self.base_weights.keys(), weight_vector)}
            
            # 归一化权重
            predicted_weights = self._normalize_weights(predicted_weights)
            
            return predicted_weights
            
        except Exception as e:
            logger.error(f"预测权重失败: {str(e)}")
            return self.current_weights
    
    def _extract_market_features(self, market_data, column='close'):
        """
        提取市场特征
        
        Args:
            market_data: 市场数据DataFrame
            column: 用于分析的价格列名
            
        Returns:
            dict: 市场特征字典
        """
        features = {}
        
        if market_data.empty or len(market_data) < max(self.market_regime_windows):
            return features
        
        try:
            # 提取短期和长期趋势
            short_window = min(self.market_regime_windows)
            long_window = max(self.market_regime_windows)
            
            # 短期趋势斜率
            short_data = market_data[column].iloc[-short_window:]
            x_short = np.arange(len(short_data))
            short_slope = np.polyfit(x_short, short_data.values, 1)[0]
            features["trend_short"] = short_slope * short_window / np.mean(short_data)
            
            # 长期趋势斜率
            long_data = market_data[column].iloc[-long_window:]
            x_long = np.arange(len(long_data))
            long_slope = np.polyfit(x_long, long_data.values, 1)[0]
            features["trend_long"] = long_slope * long_window / np.mean(long_data)
            
            # 波动率
            returns = market_data[column].pct_change().iloc[-20:].dropna()
            features["volatility"] = returns.std()
            
            # 成交量变化
            if 'vol' in market_data.columns:
                vol_data = market_data['vol']
                vol_ma = vol_data.rolling(20).mean()
                if not vol_ma.empty and not np.isnan(vol_ma.iloc[-1]):
                    vol_change = vol_data.iloc[-1] / vol_ma.iloc[-1] - 1
                    features["volume_change"] = vol_change
            
            # RSI水平
            if 'rsi' in market_data.columns:
                features["rsi_level"] = market_data['rsi'].iloc[-1]
            elif 'rsi_14' in market_data.columns:
                features["rsi_level"] = market_data['rsi_14'].iloc[-1]
            
            # MACD信号
            if all(col in market_data.columns for col in ['macd', 'signal']):
                macd_val = market_data['macd'].iloc[-1]
                signal_val = market_data['signal'].iloc[-1]
                features["macd_signal"] = macd_val - signal_val
            
            return features
            
        except Exception as e:
            logger.error(f"提取市场特征失败: {str(e)}")
            return {}
    
    def update_weights(self, market_data, recent_performance=None, column='close'):
        """
        根据当前市场状态和最近表现更新权重
        
        Args:
            market_data: 市场数据DataFrame
            recent_performance: 最近的策略性能 (0-1)
            column: 用于分析的价格列名
            
        Returns:
            dict: 更新后的权重字典
        """
        # 确定当前市场状态
        market_regime = self._determine_market_regime(market_data, column)
        
        # 提取市场特征
        market_features = self._extract_market_features(market_data, column)
        
        # 记录最近表现
        if recent_performance is not None:
            # 添加到历史记录
            record = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "market_regime": market_regime,
                "weights": self.current_weights.copy(),
                "performance": recent_performance,
                "market_features": market_features
            }
            self.performance_history.append(record)
            
            # 清理过旧的历史记录
            if len(self.performance_history) > self.history_length:
                self.performance_history = self.performance_history[-self.history_length:]
            
            # 尝试训练模型(每10次更新)
            if len(self.performance_history) % 10 == 0:
                self._train_ml_model()
        
        # 两种权重调整策略：
        # 1. 如果有ML模型，使用预测权重
        # 2. 否则使用该市场状态下的历史最优权重
        
        if self.model is not None and market_features:
            # 使用ML模型预测权重
            predicted_weights = self._predict_weights(market_features)
            
            # 逐步调整当前权重向预测权重靠近
            for key in self.current_weights:
                if key in predicted_weights:
                    self.current_weights[key] += self.learning_rate * (predicted_weights[key] - self.current_weights[key])
        else:
            # 使用该市场状态的历史最优权重
            optimal_weights = self._get_optimal_weights_for_regime(market_regime)
            
            # 逐步调整当前权重向最优权重靠近
            for key in self.current_weights:
                if key in optimal_weights:
                    self.current_weights[key] += self.learning_rate * (optimal_weights[key] - self.current_weights[key])
        
        # 确保权重归一化
        self.current_weights = self._normalize_weights(self.current_weights)
        
        # 记录本次更新
        self.last_update = datetime.now()
        self.last_market_regime = market_regime
        
        logger.info(f"权重已更新 - 市场状态: {market_regime}, 当前权重: {self.current_weights}")
        
        return self.current_weights.copy()
    
    def get_current_weights(self):
        """
        获取当前权重
        
        Returns:
            dict: 当前权重字典
        """
        return self.current_weights.copy()
    
    def get_market_regime_weights(self, market_data, column='close'):
        """
        根据当前市场状态获取优化的权重，但不更新当前权重
        
        Args:
            market_data: 市场数据DataFrame
            column: 用于分析的价格列
            
        Returns:
            dict: 该市场状态下的优化权重
        """
        # 确定当前市场状态
        market_regime = self._determine_market_regime(market_data, column)
        
        # 提取市场特征
        market_features = self._extract_market_features(market_data, column)
        
        # 使用ML模型预测(如果有)或者历史最优
        if self.model is not None and market_features:
            return self._predict_weights(market_features)
        else:
            return self._get_optimal_weights_for_regime(market_regime)
    
    def reset_weights(self):
        """
        重置权重为基础权重
        
        Returns:
            dict: 重置后的权重字典
        """
        self.current_weights = self.base_weights.copy()
        logger.info("权重已重置为基础权重")
        return self.current_weights.copy()
    
    def save_history(self, filepath):
        """
        保存权重历史记录到文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            # 转换为DataFrame
            history_data = pd.DataFrame(self.performance_history)
            
            # 存储为CSV文件
            history_data.to_csv(filepath, index=False)
            
            logger.info(f"权重历史记录已保存到 {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"保存权重历史记录失败: {str(e)}")
            return False
    
    def load_history(self, filepath):
        """
        从文件加载权重历史记录
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            # 读取CSV文件
            history_data = pd.read_csv(filepath)
            
            # 转换为列表字典
            self.performance_history = history_data.to_dict('records')
            
            # 尝试训练模型
            self._train_ml_model()
            
            logger.info(f"已从 {filepath} 加载 {len(self.performance_history)} 条权重历史记录")
            return True
            
        except Exception as e:
            logger.error(f"加载权重历史记录失败: {str(e)}")
            return False 