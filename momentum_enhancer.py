"""
动量分析增强器
整合高级技术指标和机器学习增强的动量分析模块
"""
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 导入基础动量分析器
from momentum_analysis import MomentumAnalyzer

# 导入高级指标模块
from src.advanced_indicators import PricePatternRecognizer, SupportResistanceAnalyzer

# 导入机器学习增强模块
from src.ml_enhancements import AdaptiveWeightSystem, MomentumPredictor

logger = logging.getLogger(__name__)

class MomentumEnhancer:
    """动量分析增强器类，整合高级技术指标和机器学习能力"""
    
    def __init__(self, use_tushare=True, use_ml=True, use_adaptive_weights=True):
        """
        初始化动量分析增强器
        
        Args:
            use_tushare: 是否使用Tushare API
            use_ml: 是否使用机器学习预测
            use_adaptive_weights: 是否使用自适应权重系统
        """
        # 基础动量分析器
        self.base_analyzer = MomentumAnalyzer(use_tushare=use_tushare)
        
        # 高级指标分析器
        self.pattern_recognizer = PricePatternRecognizer()
        self.sr_analyzer = SupportResistanceAnalyzer()
        
        # 机器学习组件
        self.use_ml = use_ml
        self.use_adaptive_weights = use_adaptive_weights
        
        if use_ml:
            # 初始化预测模型 (5天和20天两个预测周期)
            self.predictors = {
                'short_term': MomentumPredictor(prediction_horizon=5),
                'medium_term': MomentumPredictor(prediction_horizon=20)
            }
            
            # 尝试加载预训练模型
            self._load_pretrained_models()
        
        if use_adaptive_weights:
            # 初始化自适应权重系统
            self.weight_system = AdaptiveWeightSystem()
            
            # 尝试加载历史权重数据
            weight_history_path = os.path.join('data', 'weight_history.csv')
            if os.path.exists(weight_history_path):
                self.weight_system.load_history(weight_history_path)
        
        # 缓存
        self.analysis_cache = {}
    
    def _load_pretrained_models(self):
        """加载预训练的预测模型"""
        model_dir = os.path.join('models')
        if not os.path.exists(model_dir):
            logger.info("模型目录不存在，将在训练后创建")
            return
        
        try:
            # 查找模型文件
            model_files = [f for f in os.listdir(model_dir) if f.startswith('momentum_predictor_') and f.endswith('.joblib')]
            
            if not model_files:
                logger.info("未找到预训练模型")
                return
            
            # 尝试加载模型
            for model_file in model_files:
                filepath = os.path.join(model_dir, model_file)
                
                # 确定模型类型
                if '5d' in model_file:
                    predictor = self.predictors['short_term']
                elif '20d' in model_file:
                    predictor = self.predictors['medium_term']
                else:
                    continue
                
                # 加载模型
                success = predictor.load_model(filepath)
                if success:
                    logger.info(f"已加载预训练模型: {model_file}")
        
        except Exception as e:
            logger.error(f"加载预训练模型时出错: {str(e)}")
    
    def _get_adaptive_weights(self, data):
        """
        获取当前市场环境下的自适应权重
        
        Args:
            data: 股票数据DataFrame
            
        Returns:
            dict: 权重字典
        """
        if not self.use_adaptive_weights:
            return self.base_analyzer.indicator_weights
        
        # 从自适应权重系统获取当前权重
        return self.weight_system.get_market_regime_weights(data)
    
    def _update_weights(self, data, performance):
        """
        根据最近表现更新权重
        
        Args:
            data: 股票数据DataFrame
            performance: 性能评分(0-1)
            
        Returns:
            dict: 更新后的权重
        """
        if not self.use_adaptive_weights:
            return self.base_analyzer.indicator_weights
        
        # 更新自适应权重系统
        updated_weights = self.weight_system.update_weights(data, performance)
        
        # 保存权重历史
        weight_history_path = os.path.join('data', 'weight_history.csv')
        self.weight_system.save_history(weight_history_path)
        
        return updated_weights
    
    def analyze_price_patterns(self, data):
        """
        分析价格形态
        
        Args:
            data: 股票数据DataFrame
            
        Returns:
            dict: 价格形态分析结果
        """
        # 检测所有支持的价格形态
        pattern_result = self.pattern_recognizer.detect_patterns(data)
        
        # 综合支撑位和阻力位分析
        sr_result = self.sr_analyzer.analyze_price_strength(data)
        
        # 合并结果
        result = {
            'price_pattern': pattern_result,
            'support_resistance': sr_result,
        }
        
        return result
    
    def predict_future_momentum(self, data):
        """
        预测未来动量
        
        Args:
            data: 股票数据DataFrame
            
        Returns:
            dict: 预测结果
        """
        if not self.use_ml:
            return {'prediction_available': False}
        
        result = {}
        
        # 短期预测
        short_term_predictor = self.predictors['short_term']
        if short_term_predictor.model is not None:
            short_term_score = short_term_predictor.predict_momentum_score(data)
            result['short_term'] = {
                'prediction_horizon': 5,
                'momentum_score': short_term_score
            }
        
        # 中期预测
        medium_term_predictor = self.predictors['medium_term']
        if medium_term_predictor.model is not None:
            medium_term_score = medium_term_predictor.predict_momentum_score(data)
            result['medium_term'] = {
                'prediction_horizon': 20,
                'momentum_score': medium_term_score
            }
        
        result['prediction_available'] = len(result) > 0
        
        return result
    
    def enhance_momentum_analysis(self, data, stock_code, stock_name=None):
        """
        增强版动量分析
        
        Args:
            data: 股票数据DataFrame
            stock_code: 股票代码
            stock_name: 股票名称(可选)
            
        Returns:
            dict: 增强的动量分析结果
        """
        try:
            # 检查缓存
            cache_key = f"{stock_code}_{data.index[-1].strftime('%Y%m%d')}"
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            
            # 基础动量分析
            data = self.base_analyzer.calculate_momentum(data)
            if data.empty:
                logger.warning(f"无法计算{stock_code}的技术指标")
                return None
            
            # 获取自适应权重
            adaptive_weights = self._get_adaptive_weights(data)
            
            # 计算基础动量得分
            base_score, base_score_details = self.base_analyzer.calculate_momentum_score(data, custom_weights=adaptive_weights)
            
            # 价格形态分析
            pattern_analysis = self.analyze_price_patterns(data)
            
            # 预测未来动量
            future_prediction = self.predict_future_momentum(data)
            
            # 综合评分
            enhanced_score = base_score
            
            # 根据价格形态调整得分
            pattern_result = pattern_analysis.get('price_pattern', {})
            if pattern_result.get('pattern') and pattern_result.get('confidence', 0) > 50:
                pattern_type = pattern_result['pattern']
                
                # 头肩底/双底/上升三角形等看涨形态
                bullish_patterns = ['head_and_shoulders_bottom', 'double_bottom', 'ascending_triangle',
                                   'bull_flag', 'bull_pennant']
                
                # 头肩顶/双顶/下降三角形等看跌形态
                bearish_patterns = ['head_and_shoulders_top', 'double_top', 'descending_triangle',
                                   'bear_flag', 'bear_pennant']
                
                # 形态得分调整
                if pattern_type in bullish_patterns:
                    # 看涨形态加分
                    pattern_bonus = pattern_result['confidence'] * 0.2
                    enhanced_score += pattern_bonus
                    logger.info(f"{stock_code} 发现看涨形态 {pattern_type}，得分+{pattern_bonus:.2f}")
                
                elif pattern_type in bearish_patterns:
                    # 看跌形态减分
                    pattern_penalty = pattern_result['confidence'] * 0.2
                    enhanced_score -= pattern_penalty
                    logger.info(f"{stock_code} 发现看跌形态 {pattern_type}，得分-{pattern_penalty:.2f}")
            
            # 根据支撑/阻力位分析调整得分
            sr_result = pattern_analysis.get('support_resistance', {})
            if sr_result.get('strength_score') is not None:
                # 价格接近支撑位时加分，接近阻力位时减分
                sr_score = sr_result['strength_score']
                sr_adjustment = (50 - sr_score) * 0.1  # 支撑位加分，阻力位减分
                enhanced_score += sr_adjustment
                logger.info(f"{stock_code} 支撑/阻力位分析，得分调整 {sr_adjustment:.2f}")
            
            # 根据预测结果调整得分
            if future_prediction.get('prediction_available', False):
                # 短期预测
                if 'short_term' in future_prediction:
                    short_term_score = future_prediction['short_term']['momentum_score']
                    short_term_adj = (short_term_score - 50) * 0.1
                    enhanced_score += short_term_adj
                    logger.info(f"{stock_code} 短期预测调整 {short_term_adj:.2f}")
                
                # 中期预测
                if 'medium_term' in future_prediction:
                    medium_term_score = future_prediction['medium_term']['momentum_score']
                    medium_term_adj = (medium_term_score - 50) * 0.15
                    enhanced_score += medium_term_adj
                    logger.info(f"{stock_code} 中期预测调整 {medium_term_adj:.2f}")
            
            # 确保最终得分在0-100之间
            enhanced_score = min(100, max(0, enhanced_score))
            
            # 构建增强分析结果
            result = {
                'code': stock_code,
                'name': stock_name or '',
                'base_score': base_score,
                'enhanced_score': enhanced_score,
                'price_patterns': pattern_analysis,
                'future_prediction': future_prediction,
                'data': data,
                'base_score_details': base_score_details,
                'adaptive_weights': adaptive_weights,
                'analysis_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            # 缓存结果
            self.analysis_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"增强动量分析失败: {str(e)}", exc_info=True)
            return None
    
    def analyze_stocks(self, stock_list, sample_size=100, min_score=60):
        """
        分析股票列表，找出具有强劲动量的股票
        
        Args:
            stock_list: 股票列表DataFrame
            sample_size: 分析样本大小
            min_score: 最低分数阈值
            
        Returns:
            list: 分析结果列表
        """
        results = []
        
        # 确保股票列表不为空
        if stock_list.empty:
            logger.error("股票列表为空，无法进行分析")
            return results
        
        # 限制样本大小
        if sample_size < len(stock_list):
            stock_list = stock_list.sample(sample_size)
        
        total = len(stock_list)
        logger.info(f"开始增强动量分析 {total} 支股票")
        
        # 分析每支股票
        for idx, (_, stock) in enumerate(stock_list.iterrows()):
            ts_code = stock['ts_code']
            name = stock['name']
            industry = stock.get('industry', '')
            
            logger.info(f"分析进度: {idx+1}/{total} - 正在分析: {name}({ts_code})")
            
            # 获取日线数据
            data = self.base_analyzer.get_stock_daily_data(ts_code)
            if data.empty:
                logger.warning(f"无法获取{ts_code}的数据，跳过分析")
                continue
            
            # 增强动量分析
            analysis_result = self.enhance_momentum_analysis(data, ts_code, name)
            if not analysis_result:
                logger.warning(f"增强分析{ts_code}失败，跳过")
                continue
            
            # 筛选高于最低分数的股票
            if analysis_result['enhanced_score'] >= min_score:
                # 添加行业信息
                analysis_result['industry'] = industry
                results.append(analysis_result)
                
                # 如果有机器学习模型，更新其性能评估
                if self.use_adaptive_weights:
                    # 使用今日与昨日收盘价比较作为简单性能评估
                    if len(data) >= 2:
                        yesterday_close = data['close'].iloc[-2]
                        today_close = data['close'].iloc[-1]
                        performance = (today_close / yesterday_close - 1) * 5 + 0.5  # 映射到0-1
                        performance = min(1, max(0, performance))
                        
                        # 更新权重
                        self._update_weights(data, performance)
        
        # 按增强得分排序
        results = sorted(results, key=lambda x: x['enhanced_score'], reverse=True)
        
        logger.info(f"增强动量分析完成，共找到 {len(results)} 支符合条件的股票")
        
        return results
    
    def plot_enhanced_stock_chart(self, data, stock_code, stock_name, analysis_result, save_path=None):
        """
        绘制增强版股票分析图表
        
        Args:
            data: 股票数据DataFrame
            stock_code: 股票代码
            stock_name: 股票名称
            analysis_result: 增强分析结果
            save_path: 保存路径
            
        Returns:
            bool: 是否成功
        """
        if data.empty:
            logger.warning(f"无法绘制{stock_code}的图表，数据为空")
            return False
            
        try:
            # 创建图表
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(6, 4, figure=fig)
            
            # 主图：K线和均线
            ax_main = fig.add_subplot(gs[0:2, 0:3])
            ax_main.set_title(f"{stock_name}({stock_code}) 增强动量分析", fontsize=15)
            
            # 绘制K线
            for i in range(len(data)):
                date = data.index[i]
                open_price, close, high, low = data.iloc[i][['open', 'close', 'high', 'low']]
                color = 'red' if close >= open_price else 'green'
                
                # 绘制实体
                ax_main.add_patch(plt.Rectangle((i-0.4, min(open_price, close)), 
                                               0.8, abs(open_price-close),
                                               fill=True, color=color))
                # 绘制影线
                ax_main.plot([i, i], [low, high], color=color, linewidth=1)
            
            # 绘制均线
            if 'ma5' in data.columns:
                ax_main.plot(range(len(data)), data['ma5'], label='MA5', color='blue', linewidth=1)
            if 'ma20' in data.columns:
                ax_main.plot(range(len(data)), data['ma20'], label='MA20', color='purple', linewidth=1)
            if 'ma60' in data.columns:
                ax_main.plot(range(len(data)), data['ma60'], label='MA60', color='brown', linewidth=1)
            
            # 绘制支撑位和阻力位
            sr_analysis = analysis_result.get('price_patterns', {}).get('support_resistance', {})
            current_price = data['close'].iloc[-1]
            
            for level, _, level_type in sr_analysis.get('support_levels', [])[:3]:
                color = 'green' if level_type == 'static' else 'lime' if level_type == 'dynamic' else 'darkgreen'
                ax_main.axhline(y=level, color=color, linestyle='--', alpha=0.7)
            
            for level, _, level_type in sr_analysis.get('resistance_levels', [])[:3]:
                color = 'red' if level_type == 'static' else 'pink' if level_type == 'dynamic' else 'darkred'
                ax_main.axhline(y=level, color=color, linestyle='--', alpha=0.7)
            
            # 标记价格形态
            pattern_result = analysis_result.get('price_patterns', {}).get('price_pattern', {})
            if pattern_result.get('pattern'):
                pattern_type = pattern_result['pattern']
                confidence = pattern_result['confidence']
                details = pattern_result['details']
                
                # 根据形态类型添加标注
                if 'head_and_shoulders' in pattern_type:
                    if 'left_shoulder_idx' in details and 'head_idx' in details and 'right_shoulder_idx' in details:
                        ls_idx = details['left_shoulder_idx']
                        h_idx = details['head_idx']
                        rs_idx = details['right_shoulder_idx']
                        
                        # 添加标注
                        ax_main.annotate('LS', xy=(ls_idx, data['close'].iloc[ls_idx]), 
                                        xytext=(ls_idx, data['close'].iloc[ls_idx]*1.05),
                                        arrowprops=dict(facecolor='yellow', shrink=0.05))
                        ax_main.annotate('H', xy=(h_idx, data['close'].iloc[h_idx]), 
                                        xytext=(h_idx, data['close'].iloc[h_idx]*1.05),
                                        arrowprops=dict(facecolor='yellow', shrink=0.05))
                        ax_main.annotate('RS', xy=(rs_idx, data['close'].iloc[rs_idx]), 
                                        xytext=(rs_idx, data['close'].iloc[rs_idx]*1.05),
                                        arrowprops=dict(facecolor='yellow', shrink=0.05))
                
                elif 'double_top' in pattern_type or 'double_bottom' in pattern_type:
                    if ('top1_idx' in details and 'top2_idx' in details) or \
                       ('bottom1_idx' in details and 'bottom2_idx' in details):
                        
                        idx1 = details.get('top1_idx', details.get('bottom1_idx'))
                        idx2 = details.get('top2_idx', details.get('bottom2_idx'))
                        
                        # 添加标注
                        ax_main.annotate('1', xy=(idx1, data['close'].iloc[idx1]), 
                                        xytext=(idx1, data['close'].iloc[idx1]*1.05),
                                        arrowprops=dict(facecolor='yellow', shrink=0.05))
                        ax_main.annotate('2', xy=(idx2, data['close'].iloc[idx2]), 
                                        xytext=(idx2, data['close'].iloc[idx2]*1.05),
                                        arrowprops=dict(facecolor='yellow', shrink=0.05))
            
            ax_main.set_ylabel('Price')
            ax_main.grid(True)
            ax_main.legend(loc='best')
            
            # 成交量图
            ax_vol = fig.add_subplot(gs[2, 0:3], sharex=ax_main)
            for i in range(len(data)):
                color = 'red' if data['close'].iloc[i] >= data['open'].iloc[i] else 'green'
                ax_vol.bar(i, data['vol'].iloc[i], color=color, width=0.8)
            ax_vol.set_ylabel('Volume')
            ax_vol.grid(True)
            
            # MACD图
            ax_macd = fig.add_subplot(gs[3, 0:3], sharex=ax_main)
            ax_macd.plot(range(len(data)), data['macd'], label='MACD', color='blue', linewidth=1)
            ax_macd.plot(range(len(data)), data['signal'], label='Signal', color='red', linewidth=1)
            
            # 绘制MACD柱状图
            for i in range(len(data)):
                hist = data['macd_hist'].iloc[i]
                color = 'red' if hist >= 0 else 'green'
                ax_macd.bar(i, hist, color=color, width=0.8)
            
            ax_macd.legend(loc='best')
            ax_macd.set_ylabel('MACD')
            ax_macd.grid(True)
            
            # RSI图
            ax_rsi = fig.add_subplot(gs[4, 0:3], sharex=ax_main)
            ax_rsi.plot(range(len(data)), data['rsi'], label='RSI', color='purple', linewidth=1)
            ax_rsi.axhline(y=30, color='green', linestyle='--')
            ax_rsi.axhline(y=70, color='red', linestyle='--')
            ax_rsi.legend(loc='best')
            ax_rsi.set_ylabel('RSI')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.grid(True)
            
            # KDJ图
            ax_kdj = fig.add_subplot(gs[5, 0:3], sharex=ax_main)
            ax_kdj.plot(range(len(data)), data['k'], label='K', color='blue', linewidth=1)
            ax_kdj.plot(range(len(data)), data['d'], label='D', color='yellow', linewidth=1)
            ax_kdj.plot(range(len(data)), data['j'], label='J', color='magenta', linewidth=1)
            ax_kdj.axhline(y=20, color='green', linestyle='--')
            ax_kdj.axhline(y=80, color='red', linestyle='--')
            ax_kdj.legend(loc='best')
            ax_kdj.set_ylabel('KDJ')
            ax_kdj.grid(True)
            
            # 设置X轴标签为日期
            date_indices = range(0, len(data), len(data)//5)
            date_labels = [data.index[i].strftime('%Y-%m-%d') for i in date_indices]
            ax_kdj.set_xticks(date_indices)
            ax_kdj.set_xticklabels(date_labels, rotation=45)
            
            # 侧面板：分析结果
            ax_score = fig.add_subplot(gs[0:2, 3])
            
            # 增强得分
            base_score = analysis_result.get('base_score', 0)
            enhanced_score = analysis_result.get('enhanced_score', 0)
            
            # 分数进度条
            ax_score.barh(['基础得分', '增强得分'], [base_score, enhanced_score], color=['blue', 'red'])
            ax_score.set_xlim(0, 100)
            ax_score.set_title('动量评分')
            
            # 添加文本说明
            score_text = f"""
            基础得分: {base_score:.1f}
            增强得分: {enhanced_score:.1f}
            
            技术指标:
            RSI: {data['rsi'].iloc[-1]:.1f}
            MACD: {data['macd'].iloc[-1]:.4f}
            KDJ(K): {data['k'].iloc[-1]:.1f}
            
            价格形态: {pattern_result.get('pattern', '无')}
            形态置信度: {pattern_result.get('confidence', 0):.1f}%
            """
            
            # 价格形态分析
            ax_pattern = fig.add_subplot(gs[2:4, 3])
            ax_pattern.axis('off')
            ax_pattern.text(0, 0.9, '价格形态分析', fontsize=12, fontweight='bold')
            
            pattern_info = f"""
            形态类型: {pattern_result.get('pattern', '无')}
            置信度: {pattern_result.get('confidence', 0):.1f}%
            
            支撑位: {sr_analysis.get('nearest_support', 0):.2f} ({sr_analysis.get('support_distance_pct', 0):.1f}%)
            阻力位: {sr_analysis.get('nearest_resistance', 0):.2f} ({sr_analysis.get('resistance_distance_pct', 0):.1f}%)
            
            位置: {sr_analysis.get('position', '未知')}
            强度得分: {sr_analysis.get('strength_score', 0):.1f}
            """
            ax_pattern.text(0, 0.7, pattern_info, fontsize=10)
            
            # 预测分析
            ax_predict = fig.add_subplot(gs[4:6, 3])
            ax_predict.axis('off')
            ax_predict.text(0, 0.9, '未来动量预测', fontsize=12, fontweight='bold')
            
            prediction_info = ""
            future_prediction = analysis_result.get('future_prediction', {})
            
            if future_prediction.get('prediction_available', False):
                short_term = future_prediction.get('short_term', {})
                medium_term = future_prediction.get('medium_term', {})
                
                prediction_info = f"""
                短期(5天)预测: {short_term.get('momentum_score', 0):.1f}
                中期(20天)预测: {medium_term.get('momentum_score', 0):.1f}
                
                当前市场环境: {analysis_result.get('market_regime', '未知')}
                """
            else:
                prediction_info = "预测模型未训练或不可用"
                
            ax_predict.text(0, 0.7, prediction_info, fontsize=10)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存或显示图表
            if save_path:
                plt.savefig(save_path)
                logger.info(f"已保存{stock_code}的图表到{save_path}")
                plt.close(fig)
                return True
            else:
                plt.show()
                return True
                
        except Exception as e:
            logger.error(f"绘制增强版股票图表失败: {str(e)}", exc_info=True)
            return False
    
    def train_prediction_models(self, stock_list, sample_size=50, test_size=10):
        """
        训练预测模型
        
        Args:
            stock_list: 股票列表DataFrame
            sample_size: 用于训练的股票数量
            test_size: 用于测试的股票数量
            
        Returns:
            dict: 训练结果统计
        """
        if not self.use_ml:
            logger.warning("机器学习功能未启用，无法训练模型")
            return {'status': 'disabled'}
        
        logger.info(f"开始训练预测模型，使用 {sample_size} 支股票训练，{test_size} 支股票测试")
        
        try:
            # 选择用于训练的股票
            train_stocks = stock_list.sample(sample_size + test_size)
            train_set = train_stocks.iloc[:sample_size]
            test_set = train_stocks.iloc[sample_size:sample_size+test_size]
            
            # 收集训练数据
            train_data = []
            
            # 处理训练集
            for _, stock in train_set.iterrows():
                ts_code = stock['ts_code']
                logger.info(f"收集 {ts_code} 的训练数据")
                
                # 获取股票数据
                data = self.base_analyzer.get_stock_daily_data(ts_code)
                if data.empty:
                    logger.warning(f"无法获取 {ts_code} 的数据，跳过")
                    continue
                
                # 计算技术指标
                data = self.base_analyzer.calculate_momentum(data)
                if data.empty:
                    logger.warning(f"计算 {ts_code} 的技术指标失败，跳过")
                    continue
                
                train_data.append(data)
            
            if not train_data:
                logger.error("没有收集到训练数据")
                return {'status': 'failed', 'error': 'No training data collected'}
            
            # 合并训练数据
            combined_data = pd.concat(train_data)
            logger.info(f"收集了 {len(combined_data)} 条数据点用于训练")
            
            # 训练短期预测模型
            short_term_model = self.predictors['short_term']
            short_term_metrics = short_term_model.train(combined_data, optimize_hyperparams=True)
            
            # 训练中期预测模型
            medium_term_model = self.predictors['medium_term']
            medium_term_metrics = medium_term_model.train(combined_data, optimize_hyperparams=True)
            
            # 评估测试集
            test_results = []
            
            for _, stock in test_set.iterrows():
                ts_code = stock['ts_code']
                logger.info(f"测试 {ts_code} 的预测模型")
                
                # 获取股票数据
                data = self.base_analyzer.get_stock_daily_data(ts_code)
                if data.empty:
                    continue
                
                # 计算技术指标
                data = self.base_analyzer.calculate_momentum(data)
                if data.empty:
                    continue
                
                # 短期预测
                short_term_prediction = short_term_model.predict(data)
                
                # 中期预测
                medium_term_prediction = medium_term_model.predict(data)
                
                # 保存测试结果
                test_result = {
                    'ts_code': ts_code,
                    'short_term_prediction': short_term_prediction,
                    'medium_term_prediction': medium_term_prediction
                }
                test_results.append(test_result)
            
            # 保存模型
            short_term_model_path = short_term_model.save_model()
            medium_term_model_path = medium_term_model.save_model()
            
            # 返回训练结果
            return {
                'status': 'success',
                'short_term_metrics': short_term_metrics,
                'medium_term_metrics': medium_term_metrics,
                'model_paths': {
                    'short_term': short_term_model_path,
                    'medium_term': medium_term_model_path
                },
                'test_results_count': len(test_results)
            }
            
        except Exception as e:
            logger.error(f"训练预测模型失败: {str(e)}", exc_info=True)
            return {'status': 'failed', 'error': str(e)}

# 当直接运行该模块时执行的测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建动量增强器
    enhancer = MomentumEnhancer(use_tushare=True, use_ml=True, use_adaptive_weights=True)
    
    # 获取股票列表
    stock_list = enhancer.base_analyzer.get_stock_list()
    print(f"获取到 {len(stock_list)} 支股票")
    
    # 训练预测模型
    if enhancer.use_ml:
        train_result = enhancer.train_prediction_models(stock_list, sample_size=10, test_size=3)
        print(f"模型训练结果: {train_result['status']}")
    
    # 分析示例股票
    sample_stock = stock_list.iloc[0]
    ts_code = sample_stock['ts_code']
    name = sample_stock['name']
    
    # 获取股票数据
    data = enhancer.base_analyzer.get_stock_daily_data(ts_code)
    
    # 增强动量分析
    result = enhancer.enhance_momentum_analysis(data, ts_code, name)
    
    if result:
        # 输出分析结果
        print(f"股票: {name}({ts_code})")
        print(f"基础得分: {result['base_score']:.1f}")
        print(f"增强得分: {result['enhanced_score']:.1f}")
        
        # 价格形态
        pattern = result['price_patterns']['price_pattern'].get('pattern', '无')
        confidence = result['price_patterns']['price_pattern'].get('confidence', 0)
        print(f"价格形态: {pattern} (置信度: {confidence:.1f}%)")
        
        # 支撑位和阻力位
        sr = result['price_patterns']['support_resistance']
        print(f"最近支撑位: {sr.get('nearest_support', 0):.2f}")
        print(f"最近阻力位: {sr.get('nearest_resistance', 0):.2f}")
        
        # 绘制图表
        enhancer.plot_enhanced_stock_chart(data, ts_code, name, result, 
                                         save_path=f"./results/charts/{ts_code}_enhanced.png") 