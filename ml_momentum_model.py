#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 确保momentum_analysis模块可以被导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from momentum_analysis import MomentumAnalyzer
from enhanced_momentum_analysis import EnhancedMomentumAnalyzer

# 创建日志记录器
logger = logging.getLogger("ml_momentum")
logger.setLevel(logging.INFO)

# 确保日志目录存在
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 文件处理器，按日期滚动
file_handler = logging.FileHandler(os.path.join(LOG_DIR, f"ml_momentum_{datetime.now().strftime('%Y%m%d')}.log"), encoding='utf-8')
file_handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 定义模型目录
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class MLMomentumModel:
    """基于机器学习的动量模型
    
    使用机器学习技术根据历史表现自动调整技术指标权重
    """
    
    def __init__(self, use_enhanced=True):
        """初始化
        
        Args:
            use_enhanced: 是否使用增强版动量分析（包括资金流向、北向资金等）
        """
        self.use_enhanced = use_enhanced
        self.analyzer = EnhancedMomentumAnalyzer() if use_enhanced else MomentumAnalyzer()
        self.model = None
        self.feature_names = [
            'rsi_score', 'macd_score', 'kdj_score', 'ma_score', 
            'volume_score', 'price_pattern_score', 'momentum_score'
        ]
        
        # 根据市场状态划分不同的模型
        self.market_models = {
            'bull': None,  # 牛市模型
            'bear': None,  # 熊市模型
            'volatile': None,  # 震荡市模型
            'neutral': None   # 中性市场模型
        }
        
        # 尝试加载已有模型
        self._load_models()
    
    def _load_models(self):
        """加载已训练的模型"""
        try:
            # 加载通用模型
            model_path = os.path.join(MODEL_DIR, 'momentum_model.joblib')
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("已加载通用动量模型")
            else:
                logger.warning("未找到通用动量模型，将使用默认预训练模型")
                self._create_default_model()
            
            # 加载市场状态特定模型
            for market_state in self.market_models.keys():
                model_path = os.path.join(MODEL_DIR, f'momentum_model_{market_state}.joblib')
                if os.path.exists(model_path):
                    self.market_models[market_state] = joblib.load(model_path)
                    logger.info(f"已加载{market_state}市场动量模型")
                else:
                    logger.info(f"未找到{market_state}市场模型，将使用默认预训练模型")
                    self._create_default_model(market_state)
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            logger.info("将使用默认预训练模型")
            self._create_default_model()
    
    def _create_default_model(self, market_state=None):
        """创建默认预训练模型
        
        当无法加载现有模型时，创建一个基于默认权重的预训练模型
        
        Args:
            market_state: 市场状态，如果指定则创建特定市场状态的模型
        """
        try:
            # 创建一个随机森林回归器作为默认模型
            default_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # 创建一个简单的特征矩阵和目标变量进行初始训练
            # 使用基础指标权重作为特征重要性的参考
            X = np.array([[1] * len(self.feature_names)])
            y = np.array([50])  # 中性预测值
            
            # 伪训练，只是为了初始化模型
            default_model.fit(X, y)
            
            # 手动设置特征重要性
            # 根据不同市场状态调整权重
            if market_state == 'bull':
                # 牛市更重视动量和均线
                importances = np.array([0.1, 0.25, 0.1, 0.2, 0.15, 0.15, 0.05])
            elif market_state == 'bear':
                # 熊市更重视RSI和KDJ等超买超卖指标
                importances = np.array([0.2, 0.1, 0.2, 0.1, 0.2, 0.15, 0.05])
            elif market_state == 'volatile':
                # 震荡市场重视成交量和价格形态
                importances = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.2, 0.05])
            else:  # neutral或其他
                # 中性市场平衡权重
                importances = np.array([0.15, 0.2, 0.15, 0.15, 0.15, 0.1, 0.1])
                
            # 如果使用增强版，添加增强特征的权重
            if self.use_enhanced and len(self.feature_names) > 7:
                # 为增强特征添加权重
                enhanced_weights = np.array([0.1, 0.1, 0.1, 0.05])  # 资金流、财务、北向资金、行业因子
                importances = np.concatenate([importances, enhanced_weights])
                
            # 归一化特征重要性使其总和为1
            importances = importances / np.sum(importances)
            
            # 设置模型的特征重要性
            # default_model.feature_importances_ = importances[:len(self.feature_names)] # <- 注释掉这一行
            
            # 保存模型
            if market_state:
                self.market_models[market_state] = default_model
                self._save_model(default_model, f'momentum_model_{market_state}')
                logger.info(f"已创建并保存{market_state}市场默认预训练模型")
            else:
                self.model = default_model
                self._save_model(default_model)
                logger.info("已创建并保存通用默认预训练模型")
                
            return default_model
        except Exception as e:
            logger.error(f"创建默认预训练模型失败: {str(e)}")
            # 创建失败时返回None
            return None
    
    def _save_model(self, model, name='momentum_model'):
        """保存模型
        
        Args:
            model: 训练好的模型
            name: 模型名称
        """
        try:
            model_path = os.path.join(MODEL_DIR, f'{name}.joblib')
            joblib.dump(model, model_path)
            logger.info(f"模型已保存: {model_path}")
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
    
    def collect_training_data(self, stock_list, lookback_days=180, forward_days=20, sample_size=100):
        """收集训练数据
        
        收集股票历史数据，计算技术指标得分，并标记未来收益率作为目标变量
        
        Args:
            stock_list: 股票列表
            lookback_days: 回溯天数
            forward_days: 未来预测天数
            sample_size: 样本大小
            
        Returns:
            tuple: (特征矩阵, 目标变量)
        """
        features = []
        targets = []
        
        # 限制样本大小
        if sample_size < len(stock_list):
            stock_list = stock_list.sample(sample_size)
            logger.info(f"从 {len(stock_list)} 支股票中随机选择 {sample_size} 支收集训练数据")
        else:
            logger.info(f"收集全部 {len(stock_list)} 支股票的训练数据")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days+forward_days)
        
        for idx, (_, stock) in enumerate(stock_list.iterrows()):
            try:
                ts_code = stock['ts_code']
                logger.info(f"收集训练数据进度: {idx+1}/{len(stock_list)} - 股票: {stock['name']}({ts_code})")
                
                # 获取足够长的历史数据
                data = self.analyzer.get_stock_daily_data(ts_code, start_date=start_date.strftime('%Y%m%d'))
                if data.empty or len(data) < lookback_days + forward_days:
                    logger.warning(f"股票{ts_code}历史数据不足，跳过")
                    continue
                
                # 计算技术指标
                data = self.analyzer.calculate_momentum(data)
                
                # 按日期切片，创建多个样本点
                for i in range(len(data) - lookback_days - forward_days + 1):
                    # 用于计算指标的数据段
                    analysis_data = data.iloc[i:i+lookback_days].copy()
                    
                    # 用于计算未来收益率的数据段
                    future_data = data.iloc[i+lookback_days:i+lookback_days+forward_days]
                    
                    if len(analysis_data) < lookback_days or len(future_data) < forward_days:
                        continue
                    
                    # 计算未来收益率作为目标变量
                    future_return = (future_data.iloc[-1]['close'] / analysis_data.iloc[-1]['close'] - 1) * 100
                    
                    # 计算各技术指标的分数
                    _, score_details = self.analyzer.calculate_momentum_score(analysis_data)
                    
                    # 提取分数明细作为特征
                    feature = [score_details.get(f, 0) for f in self.feature_names]
                    
                    # 如果使用增强版，添加增强指标
                    if self.use_enhanced and isinstance(self.analyzer, EnhancedMomentumAnalyzer):
                        # 获取行业因子
                        industry = self.analyzer.get_stock_industry(ts_code)
                        industry_factor = 1.0
                        if industry:
                            industry_factor = self.analyzer.analyze_industry_momentum(industry)
                        
                        # 获取资金流向、财务动量、北向资金得分
                        money_flow_score = self.analyzer.analyze_money_flow(ts_code)
                        finance_score = self.analyzer.calculate_finance_momentum(ts_code)
                        north_flow_score = self.analyzer.analyze_north_money_flow(ts_code)
                        
                        # 添加增强特征
                        feature.extend([money_flow_score, finance_score, north_flow_score, industry_factor])
                        
                        # 更新特征名称
                        if len(self.feature_names) < len(feature):
                            self.feature_names.extend(['money_flow_score', 'finance_score', 'north_flow_score', 'industry_factor'])
                    
                    features.append(feature)
                    targets.append(future_return)
                    
            except Exception as e:
                logger.error(f"收集{stock['name']}({ts_code})的训练数据时出错: {str(e)}")
                continue
        
        logger.info(f"共收集了 {len(features)} 个训练样本")
        return np.array(features), np.array(targets)
    
    def determine_market_state(self, index_data):
        """确定市场状态
        
        根据指数表现确定当前市场状态（牛市、熊市、震荡市或中性）
        
        Args:
            index_data: 指数数据
            
        Returns:
            str: 市场状态
        """
        try:
            # 确保数据按日期排序
            index_data = index_data.sort_index()
            
            # 计算20日和60日均线
            index_data['ma20'] = index_data['close'].rolling(window=20).mean()
            index_data['ma60'] = index_data['close'].rolling(window=60).mean()
            
            # 计算波动率（20日标准差）
            index_data['volatility'] = index_data['close'].rolling(window=20).std() / index_data['close'] * 100
            
            # 获取最近数据
            recent = index_data.iloc[-20:]
            
            # 计算趋势强度（20日收益率）
            trend_strength = (recent['close'].iloc[-1] / recent['close'].iloc[0] - 1) * 100
            
            # 计算均线关系和最近波动率
            ma_relation = recent['ma20'].iloc[-1] > recent['ma60'].iloc[-1]
            avg_volatility = recent['volatility'].mean()
            
            # 根据指标确定市场状态
            if trend_strength > 5 and ma_relation:
                return 'bull'  # 牛市：正趋势且短期均线在长期均线上方
            elif trend_strength < -5 and not ma_relation:
                return 'bear'  # 熊市：负趋势且短期均线在长期均线下方
            elif avg_volatility > 2:
                return 'volatile'  # 震荡市：高波动率
            else:
                return 'neutral'  # 中性市场
                
        except Exception as e:
            logger.error(f"确定市场状态时出错: {str(e)}")
            return 'neutral'  # 默认为中性
    
    def train_model(self, stock_list=None, lookback_days=90, forward_days=20, test_size=0.2):
        """
        训练ML模型来预测股票动量
        
        Args:
            stock_list: 用于训练的股票列表，默认为None（使用配置中的股票列表）
            lookback_days: 用于特征计算的历史数据天数
            forward_days: 用于计算未来收益的天数
            test_size: 测试集比例
        
        Returns:
            bool: 训练是否成功
        """
        try:
            logger.info(f"开始收集ML动量分析训练数据，从{self._get_stock_list_for_training(stock_list)}支股票中收集...")
            training_data = self._collect_training_data(stock_list, lookback_days, forward_days)
            
            if training_data is None or training_data.empty or len(training_data) < 20:
                logger.warning(f"收集到的训练样本不足，仅获取到{0 if training_data is None or training_data.empty else len(training_data)}个样本，需要至少20个")
                logger.info("将使用默认预训练模型继续提供服务")
                self._create_default_model()  # 创建默认通用模型
                self._determine_market_state()  # 创建各市场状态模型
                return False
                
            # 确保有足够的样本
            logger.info(f"收集了{len(training_data)}个训练样本")
            
            # 分离特征和目标变量
            X = training_data[self.feature_names]
            y = training_data['future_return']
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # 训练通用模型
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = self.model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            logger.info(f"通用模型评估 - R²: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
            
            # 保存模型
            self._save_model(self.model)
            
            # 按市场状态训练和保存特定模型
            market_states = training_data['market_state'].unique()
            
            for state in market_states:
                state_data = training_data[training_data['market_state'] == state]
                
                if len(state_data) > 20:  # 确保数据足够
                    X_state = state_data[self.feature_names]
                    y_state = state_data['future_return']
                    
                    # 划分训练集和测试集
                    X_train_state, X_test_state, y_train_state, y_test_state = train_test_split(
                        X_state, y_state, test_size=test_size, random_state=42)
                    
                    # 训练模型
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train_state, y_train_state)
                    
                    # 评估模型
                    y_pred_state = model.predict(X_test_state)
                    r2_state = r2_score(y_test_state, y_pred_state)
                    mae_state = mean_absolute_error(y_test_state, y_pred_state)
                    
                    logger.info(f"{state}市场模型评估 - R²: {r2_state:.4f}, MAE: {mae_state:.4f}")
                    
                    # 保存模型
                    self.market_models[state] = model
                    self._save_model(model, f'momentum_model_{state}')
                else:
                    logger.warning(f"{state}市场状态的训练样本不足({len(state_data)}个)，将使用默认预训练模型")
                    self._create_default_model(state)
            
            # 记录特征重要性
            feature_importances = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("特征重要性:")
            for _, row in feature_importances.iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.4f}")
                
            # 保存特征重要性
            self._save_feature_importances(feature_importances)
            
            return True
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"训练ML模型时发生错误: {error_message}")
            
            if "No numeric data to plot" in error_message or "empty 'DataFrame'" in error_message:
                logger.warning("训练数据集中没有足够的数值数据，可能是由于历史数据不足")
                self.ml_status = "训练失败: 历史数据不足，建议增加股票样本或减少需要的历史数据天数"
            elif "X has" in error_message and "features" in error_message:
                logger.warning("特征矩阵与模型期望不匹配，可能是特征计算错误")
                self.ml_status = "训练失败: 特征计算错误，请检查指标计算逻辑"
            elif "divisor" in error_message and "zero" in error_message:
                logger.warning("数据处理中遇到除零错误，可能是某些技术指标计算异常")
                self.ml_status = "训练失败: 技术指标计算异常，请检查数据完整性"
            else:
                self.ml_status = f"训练失败: {error_message}"
            
            logger.info("将使用默认预训练模型继续提供服务")
            self._create_default_model()  # 创建默认模型
            self._determine_market_state()  # 创建各市场状态模型
            
            return False
    
    def visualize_feature_importance(self, model=None, title='Feature Importance'):
        """可视化特征重要性
        
        Args:
            model: 模型实例，如果为None则使用当前模型
            title: 图表标题
        """
        if model is None:
            model = self.model
            
        if model is None:
            logger.error("无法可视化特征重要性：模型未训练")
            return
            
        try:
            # 获取特征重要性
            importances = model.feature_importances_
            indices = np.argsort(importances)
            
            feature_labels = [self.feature_names[i] if i < len(self.feature_names) else f'Feature {i}' 
                             for i in indices]
            
            # 创建图表
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), feature_labels)
            plt.xlabel('Relative Importance')
            plt.title(title)
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(os.path.join(MODEL_DIR, f"{title.replace(' ', '_').lower()}.png"))
            logger.info(f"特征重要性图表已保存")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"可视化特征重要性失败: {str(e)}")
    
    def get_optimal_weights(self, market_state=None):
        """获取最优权重
        
        根据模型学习的特征重要性得到各指标的最优权重
        
        Args:
            market_state: 市场状态，用于选择适当的模型
            
        Returns:
            dict: 权重字典
        """
        # 选择合适的模型
        if market_state and market_state in self.market_models and self.market_models[market_state]:
            model = self.market_models[market_state]
            logger.info(f"使用{market_state}市场专用模型获取权重")
        elif self.model:
            model = self.model
            logger.info("使用通用模型获取权重")
        else:
            logger.warning("没有训练好的模型，返回默认权重")
            return None
        
        # 获取特征重要性
        importances = model.feature_importances_
        
        # 创建权重字典
        weights = {}
        total_importance = sum(importances)
        
        for i, feature in enumerate(self.feature_names[:len(importances)]):
            # 归一化权重
            normalized_weight = importances[i] / total_importance
            weights[feature] = normalized_weight
            
        logger.info(f"获取到最优权重: {weights}")
        return weights
    
    def calculate_ml_momentum_score(self, data, ts_code=None, market_state=None):
        """使用ML优化的权重计算动量得分
        
        Args:
            data: 股票数据
            ts_code: 股票代码，用于获取增强指标
            market_state: 市场状态
            
        Returns:
            tuple: (总分, 详情分数)
        """
        try:
            # 获取最优权重
            weights = self.get_optimal_weights(market_state)
            
            # 如果没有训练好的模型，使用默认计算方法
            if weights is None:
                if self.use_enhanced and ts_code:
                    return self.analyzer.calculate_enhanced_momentum_score(data, ts_code)
                else:
                    return self.analyzer.calculate_momentum_score(data)
            
            # 计算各指标的分数
            _, score_details = self.analyzer.calculate_momentum_score(data)
            
            # 应用ML优化的权重
            total_score = 0
            ml_score_details = {}
            
            for feature, score in score_details.items():
                if feature in weights:
                    # 使用ML学习的权重
                    weight = weights[feature]
                    weighted_score = score * weight
                    ml_score_details[feature] = score
                    total_score += weighted_score
                else:
                    # 保留原始分数
                    ml_score_details[feature] = score
                    total_score += score
            
            # 处理增强指标
            if self.use_enhanced and ts_code and isinstance(self.analyzer, EnhancedMomentumAnalyzer):
                # 获取行业因子
                industry = self.analyzer.get_stock_industry(ts_code)
                industry_factor = 1.0
                if industry:
                    industry_factor = self.analyzer.analyze_industry_momentum(industry)
                
                # 获取增强指标得分
                money_flow_score = self.analyzer.analyze_money_flow(ts_code)
                finance_score = self.analyzer.calculate_finance_momentum(ts_code)
                north_flow_score = self.analyzer.analyze_north_money_flow(ts_code)
                
                # 更新分数详情
                ml_score_details['money_flow_score'] = money_flow_score
                ml_score_details['finance_score'] = finance_score
                ml_score_details['north_flow_score'] = north_flow_score
                ml_score_details['industry_factor'] = industry_factor
                
                # 应用ML权重（如果有）
                enhanced_total = total_score
                for feature in ['money_flow_score', 'finance_score', 'north_flow_score']:
                    if feature in weights:
                        weighted_score = ml_score_details[feature] * weights[feature]
                        enhanced_total += weighted_score
                    else:
                        enhanced_total += ml_score_details[feature]
                
                # 应用行业因子
                enhanced_total *= industry_factor
                ml_score_details['enhanced_total'] = enhanced_total
                
                return enhanced_total, ml_score_details
            
            # 确保分数在0-100之间
            total_score = max(0, min(100, total_score))
            ml_score_details['ml_total'] = total_score
            
            return total_score, ml_score_details
            
        except Exception as e:
            logger.error(f"使用ML计算动量得分失败: {str(e)}")
            # 回退到标准计算方法
            if self.use_enhanced and ts_code:
                return self.analyzer.calculate_enhanced_momentum_score(data, ts_code)
            else:
                return self.analyzer.calculate_momentum_score(data)
    
    def analyze_stocks_ml(self, stock_list, market_state=None, sample_size=100, min_score=60, gui_callback=None):
        """使用ML模型分析股票列表
        
        Args:
            stock_list: 股票列表
            market_state: 市场状态
            sample_size: 样本大小
            min_score: 最低得分
            gui_callback: GUI回调函数
            
        Returns:
            list: 分析结果
        """
        results = []
        
        # 确保股票列表不为空
        if stock_list.empty:
            logger.error("股票列表为空，无法进行分析")
            if gui_callback:
                gui_callback("progress", ("股票列表为空，无法进行分析", 100))
            return results
            
        # 记录原始股票数量
        original_count = len(stock_list)
        logger.info(f"准备使用ML模型分析 {original_count} 支股票")
        if gui_callback:
            gui_callback("progress", (f"准备使用ML模型分析 {original_count} 支股票", 5))
        
        # 限制样本大小
        if sample_size < len(stock_list):
            stock_list = stock_list.sample(sample_size)
            logger.info(f"从 {original_count} 支股票中随机选择 {sample_size} 支进行分析")
            if gui_callback:
                gui_callback("progress", (f"从 {original_count} 支股票中随机选择 {sample_size} 支进行分析", 10))
        else:
            logger.info(f"分析全部 {original_count} 支股票")
            if gui_callback:
                gui_callback("progress", (f"分析全部 {original_count} 支股票", 10))
            
        total = len(stock_list)
        logger.info(f"开始使用ML模型分析 {total} 支股票，市场状态: {market_state if market_state else '通用'}")
        if gui_callback:
            gui_callback("progress", (f"开始使用ML模型分析 {total} 支股票，市场状态: {market_state if market_state else '通用'}", 15))
        
        success_count = 0
        skip_count = 0
        
        for idx, (_, stock) in enumerate(stock_list.iterrows()):
            try:
                ts_code = stock['ts_code']
                name = stock['name']
                industry = stock.get('industry', '')
                
                # 计算进度百分比 (15-95%)
                progress = 15 + int(80 * (idx / total))
                progress_msg = f"ML分析进度 [{idx+1}/{total}]: {name}({ts_code})"
                logger.info(progress_msg)
                if gui_callback:
                    gui_callback("progress", (progress_msg, progress))
                
                # 获取日线数据
                data = self.analyzer.get_stock_daily_data(ts_code)
                if data.empty:
                    logger.warning(f"无法获取{ts_code}的数据，跳过分析")
                    skip_count += 1
                    if gui_callback:
                        gui_callback("progress", (f"跳过 [{idx+1}/{total}]: {ts_code} - 无法获取数据", progress))
                    continue
                
                # 计算技术指标
                data = self.analyzer.calculate_momentum(data)
                if data.empty:
                    logger.warning(f"计算{ts_code}的技术指标失败，跳过分析")
                    skip_count += 1
                    if gui_callback:
                        gui_callback("progress", (f"跳过 [{idx+1}/{total}]: {ts_code} - 计算指标失败", progress))
                    continue
                
                # 使用ML模型评分系统
                score, score_details = self.calculate_ml_momentum_score(data, ts_code, market_state)
                
                if score >= min_score:
                    # 获取最新数据
                    latest = data.iloc[-1]
                    
                    # 保存分析结果
                    result = {
                        'ts_code': ts_code,
                        'name': name,
                        'industry': industry,
                        'close': latest['close'],
                        'momentum_20': latest.get('momentum_20', 0),
                        'momentum_20d': latest.get('momentum_20', 0),  # 为兼容GUI
                        'rsi': latest.get('rsi', 0),
                        'macd': latest.get('macd', 0),
                        'macd_hist': latest.get('macd_hist', 0),  # 为兼容GUI
                        'volume_ratio': latest.get('vol_ratio_20', 1),
                        'score': score,  # ML总分
                        'score_details': score_details,
                        'data': data
                    }
                    
                    # 添加增强版指标（如果有）
                    if 'industry_factor' in score_details:
                        result['industry_factor'] = score_details['industry_factor']
                    
                    if 'enhanced_total' in score_details:
                        result['base_score'] = sum([v for k, v in score_details.items() 
                                                 if k not in ['money_flow_score', 'finance_score', 
                                                            'north_flow_score', 'industry_factor', 
                                                            'enhanced_total']])
                    
                    results.append(result)
                    success_count += 1
                    if gui_callback:
                        gui_callback("progress", (f"符合条件 [{idx+1}/{total}]: {name}({ts_code}) - ML得分: {score:.1f}", progress))
                else:
                    skip_count += 1
                    if gui_callback:
                        gui_callback("progress", (f"得分过低 [{idx+1}/{total}]: {name}({ts_code}) - ML得分: {score:.1f}", progress))
                    
            except Exception as e:
                logger.error(f"ML分析{stock['name']}({stock['ts_code']})时出错: {str(e)}")
                skip_count += 1
                if gui_callback:
                    gui_callback("progress", (f"分析出错 [{idx+1}/{total}]: {stock['name']}({stock['ts_code']}) - {str(e)}", progress))
                continue
                
        # 按得分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"ML分析完成，共找到 {len(results)} 支符合条件的股票，跳过 {skip_count} 支")
        
        # 将结果保存为CSV
        if results:
            result_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'data' and k != 'score_details'}
                                    for r in results])
            csv_path = os.path.join(self.analyzer.RESULTS_DIR, f"ml_momentum_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"已将ML分析结果保存至: {csv_path}")
            
        if gui_callback:
            gui_callback("progress", (f"ML分析完成，共找到 {len(results)} 支符合条件的股票，跳过 {skip_count} 支", 100))
            
        return results


# 如果直接运行本模块，则执行测试
if __name__ == "__main__":
    # 设置控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 初始化模型
    ml_model = MLMomentumModel(use_enhanced=True)
    
    # 获取股票列表
    stock_list = ml_model.analyzer.get_stock_list()
    
    # 收集训练数据
    X, y = ml_model.collect_training_data(stock_list, sample_size=30)
    
    if len(X) > 0:
        # 训练模型
        ml_model.train_model(stock_list, lookback_days=90, forward_days=20, test_size=0.2)
        
        # 可视化特征重要性
        ml_model.visualize_feature_importance()
        
        # 分析一些股票
        results = ml_model.analyze_stocks_ml(stock_list.head(10))
        for r in results:
            print(f"{r['name']}({r['ts_code']}): ML得分={r['score']:.1f}, RSI={r['rsi']:.1f}") 