#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票分析策略模块，提供多种分析策略
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from ..indicators.technical import TechnicalIndicators
from ..indicators.advanced_indicators import AdvancedIndicators

class AnalysisStrategy:
    """股票分析策略基类"""
    
    def __init__(self, name="基础分析策略"):
        """初始化
        
        Args:
            name: 策略名称
        """
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.tech_indicators = TechnicalIndicators()
        self.adv_indicators = AdvancedIndicators()
    
    def analyze(self, data):
        """分析股票数据
        
        Args:
            data: DataFrame，包含股票数据
            
        Returns:
            dict: 分析结果
        """
        # 基类方法，子类应该重写
        return {
            "strategy_name": self.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "score": 0,
            "signals": []
        }
    
    def calculate_score(self, positive_signals, negative_signals, base_score=50):
        """计算综合评分
        
        Args:
            positive_signals: 正面信号列表，每项格式为 (信号名称, 权重)
            negative_signals: 负面信号列表，每项格式为 (信号名称, 权重)
            base_score: 基础分数，默认50分
            
        Returns:
            tuple: (总分, 信号列表)
        """
        total_score = base_score
        all_signals = []
        
        # 处理正面信号
        for signal_name, weight in positive_signals:
            total_score += weight
            all_signals.append({
                "name": signal_name,
                "type": "positive",
                "weight": weight
            })
        
        # 处理负面信号
        for signal_name, weight in negative_signals:
            total_score -= weight
            all_signals.append({
                "name": signal_name,
                "type": "negative",
                "weight": weight
            })
        
        # 确保分数在0-100范围内
        total_score = max(0, min(100, total_score))
        
        return total_score, all_signals


class TrendFollowingStrategy(AnalysisStrategy):
    """趋势跟踪策略"""
    
    def __init__(self):
        """初始化趋势跟踪策略"""
        super().__init__(name="趋势跟踪策略")
    
    def analyze(self, data):
        """分析股票数据，识别趋势并给出分数
        
        Args:
            data: DataFrame，包含股票数据
            
        Returns:
            dict: 分析结果
        """
        df = data.copy()
        
        # 计算技术指标
        df = self.tech_indicators.calculate_all_indicators(df)
        
        # 获取最新数据
        latest = df.iloc[-1]
        
        # 定义正面信号
        positive_signals = []
        
        # 均线多头排列 (MA5 > MA10 > MA20)
        if latest['MA5'] > latest['MA10'] > latest['MA20']:
            positive_signals.append(("均线多头排列", 10))
        
        # 金叉信号
        if latest['ma_golden_cross']:
            positive_signals.append(("MA5上穿MA10", 5))
        
        # MACD金叉
        if latest['macd_golden_cross']:
            positive_signals.append(("MACD金叉", 8))
        
        # MACD柱状图由负转正
        if latest['macd_hist_turn_positive']:
            positive_signals.append(("MACD柱状图转正", 5))
        
        # RSI上穿50
        if latest['rsi_cross_above_50']:
            positive_signals.append(("RSI上穿50", 5))
        
        # 股价突破上升趋势线
        if 'trend_breakout_up' in latest and latest['trend_breakout_up']:
            positive_signals.append(("突破上升趋势线", 8))
        
        # 上升突破布林带
        if latest['bb_break_upper']:
            positive_signals.append(("突破布林带上轨", 7))
        
        # 量价齐升
        if 'volume_increase' in latest and latest['volume_increase'] and latest['price_increase']:
            positive_signals.append(("量价齐升", 6))
        
        # 定义负面信号
        negative_signals = []
        
        # 均线空头排列 (MA5 < MA10 < MA20)
        if latest['MA5'] < latest['MA10'] < latest['MA20']:
            negative_signals.append(("均线空头排列", 10))
        
        # 死叉信号
        if latest['ma_death_cross']:
            negative_signals.append(("MA5下穿MA10", 5))
        
        # MACD死叉
        if latest['macd_death_cross']:
            negative_signals.append(("MACD死叉", 8))
        
        # MACD柱状图由正转负
        if latest['macd_hist_turn_negative']:
            negative_signals.append(("MACD柱状图转负", 5))
        
        # RSI下穿50
        if latest['rsi_cross_below_50']:
            negative_signals.append(("RSI下穿50", 5))
        
        # 股价跌破下降趋势线
        if 'trend_breakout_down' in latest and latest['trend_breakout_down']:
            negative_signals.append(("跌破下降趋势线", 8))
        
        # 下降突破布林带
        if latest['bb_break_lower']:
            negative_signals.append(("跌破布林带下轨", 7))
        
        # 量增价跌
        if 'volume_increase' in latest and latest['volume_increase'] and not latest['price_increase']:
            negative_signals.append(("量增价跌", 6))
        
        # 计算总分和信号
        score, signals = self.calculate_score(positive_signals, negative_signals)
        
        # 构建结果
        result = {
            "strategy_name": self.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "score": score,
            "signals": signals,
            "latest_data": {
                "date": str(df.iloc[-1]['date']),
                "close": float(df.iloc[-1]['close']),
                "change": float(df.iloc[-1]['daily_return'] * 100) if 'daily_return' in df.columns else None
            },
            "trend": "上升" if score > 60 else ("下降" if score < 40 else "中性")
        }
        
        return result


class ReversalStrategy(AnalysisStrategy):
    """反转策略"""
    
    def __init__(self):
        """初始化反转策略"""
        super().__init__(name="反转策略")
    
    def analyze(self, data):
        """分析股票数据，识别可能的价格反转点
        
        Args:
            data: DataFrame，包含股票数据
            
        Returns:
            dict: 分析结果
        """
        df = data.copy()
        
        # 计算技术指标
        df = self.tech_indicators.calculate_all_indicators(df)
        
        # 计算高级指标
        df = self.adv_indicators.calculate_all_advanced_indicators(df)
        
        # 获取最新数据
        latest = df.iloc[-1]
        
        # 定义正面反转信号（看涨反转）
        positive_signals = []
        
        # RSI超卖区域回升
        if latest['rsi_oversold'] and latest['rsi'] > latest['rsi'].shift(1):
            positive_signals.append(("RSI超卖回升", 8))
        
        # 布林带下轨支撑后反弹
        if latest['bb_position'] < 0.05 and latest['close'] > latest['close'].shift(1):
            positive_signals.append(("布林带下轨支撑反弹", 7))
        
        # MACD底背离
        if latest['macd_bullish_divergence']:
            positive_signals.append(("MACD底背离", 10))
        
        # 抛物线SAR看涨反转
        if 'sar_bullish' in latest and latest['sar_bullish']:
            positive_signals.append(("SAR看涨反转", 8))
        
        # 成交量暴增伴随阳线
        if 'volume_surge' in latest and latest['volume_surge'] and latest['close'] > latest['open']:
            positive_signals.append(("量增伴随阳线", 6))
        
        # KDJ超卖区金叉
        if 'stoch_bullish_cross_oversold' in latest and latest['stoch_bullish_cross_oversold']:
            positive_signals.append(("KDJ超卖区金叉", 9))
        
        # 定义负面反转信号（看跌反转）
        negative_signals = []
        
        # RSI超买区域回落
        if latest['rsi_overbought'] and latest['rsi'] < latest['rsi'].shift(1):
            negative_signals.append(("RSI超买回落", 8))
        
        # 布林带上轨压制后回落
        if latest['bb_position'] > 0.95 and latest['close'] < latest['close'].shift(1):
            negative_signals.append(("布林带上轨压制回落", 7))
        
        # MACD顶背离
        if latest['macd_bearish_divergence']:
            negative_signals.append(("MACD顶背离", 10))
        
        # 抛物线SAR看跌反转
        if 'sar_bearish' in latest and latest['sar_bearish']:
            negative_signals.append(("SAR看跌反转", 8))
        
        # 成交量暴增伴随阴线
        if 'volume_surge' in latest and latest['volume_surge'] and latest['close'] < latest['open']:
            negative_signals.append(("量增伴随阴线", 6))
        
        # KDJ超买区死叉
        if 'stoch_bearish_cross_overbought' in latest and latest['stoch_bearish_cross_overbought']:
            negative_signals.append(("KDJ超买区死叉", 9))
        
        # 计算总分和信号
        score, signals = self.calculate_score(positive_signals, negative_signals)
        
        # 构建结果
        result = {
            "strategy_name": self.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "score": score,
            "signals": signals,
            "latest_data": {
                "date": str(df.iloc[-1]['date']),
                "close": float(df.iloc[-1]['close']),
                "change": float(df.iloc[-1]['daily_return'] * 100) if 'daily_return' in df.columns else None
            },
            "reversal": "看涨反转" if score > 60 else ("看跌反转" if score < 40 else "无明显反转")
        }
        
        return result


class VolatilityBreakoutStrategy(AnalysisStrategy):
    """波动率突破策略"""
    
    def __init__(self):
        """初始化波动率突破策略"""
        super().__init__(name="波动率突破策略")
    
    def analyze(self, data):
        """分析股票数据，识别波动率突破形态
        
        Args:
            data: DataFrame，包含股票数据
            
        Returns:
            dict: 分析结果
        """
        df = data.copy()
        
        # 计算高级指标，特别是ATR和通道指标
        df = self.adv_indicators.calculate_all_advanced_indicators(df)
        
        # 获取最新数据及最近几天的数据
        latest = df.iloc[-1]
        recent = df.iloc[-5:]
        
        # 定义正面信号（适合做多的波动率突破）
        positive_signals = []
        
        # ATR数值增大（波动率增加）
        if 'atr' in latest and latest['atr'] > latest['atr'].shift(3) * 1.2:
            positive_signals.append(("波动率显著增加", 5))
        
        # 价格突破肯特纳通道上轨
        if 'keltner_break_upper' in latest and latest['keltner_break_upper']:
            positive_signals.append(("突破肯特纳通道上轨", 8))
        
        # 布林带宽度扩展（波动率增加）
        if 'bb_bandwidth_breakout' in latest and latest['bb_bandwidth_breakout']:
            positive_signals.append(("布林带宽度扩展", 7))
        
        # 连续上涨突破前期高点
        if latest['close'] > recent['high'].max() * 1.02:  # 突破前期最高点2%以上
            positive_signals.append(("突破前期高点", 10))
        
        # 在低波动率之后的突破（波动率压缩后的释放）
        if 'bb_squeeze' in df.iloc[-6:-1].any() and latest['bb_break_upper']:
            positive_signals.append(("低波动率后的上行突破", 12))
        
        # 价格突破近期横盘整理区间上沿
        if 'range_breakout_up' in latest and latest['range_breakout_up']:
            positive_signals.append(("突破盘整区间上沿", 9))
        
        # 定义负面信号（适合做空的波动率突破）
        negative_signals = []
        
        # 价格突破肯特纳通道下轨
        if 'keltner_break_lower' in latest and latest['keltner_break_lower']:
            negative_signals.append(("跌破肯特纳通道下轨", 8))
        
        # 连续下跌突破前期低点
        if latest['close'] < recent['low'].min() * 0.98:  # 跌破前期最低点2%以上
            negative_signals.append(("跌破前期低点", 10))
        
        # 在低波动率之后的跌破（波动率压缩后的释放）
        if 'bb_squeeze' in df.iloc[-6:-1].any() and latest['bb_break_lower']:
            negative_signals.append(("低波动率后的下行突破", 12))
        
        # 价格突破近期横盘整理区间下沿
        if 'range_breakout_down' in latest and latest['range_breakout_down']:
            negative_signals.append(("跌破盘整区间下沿", 9))
        
        # 计算总分和信号
        score, signals = self.calculate_score(positive_signals, negative_signals)
        
        # 构建结果
        result = {
            "strategy_name": self.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "score": score,
            "signals": signals,
            "latest_data": {
                "date": str(df.iloc[-1]['date']),
                "close": float(df.iloc[-1]['close']),
                "change": float(df.iloc[-1]['daily_return'] * 100) if 'daily_return' in df.columns else None,
                "atr": float(df.iloc[-1]['atr']) if 'atr' in df.columns else None,
                "atr_percent": float(df.iloc[-1]['atr_percent']) if 'atr_percent' in df.columns else None
            },
            "breakout": "上行突破" if score > 60 else ("下行突破" if score < 40 else "无明显突破")
        }
        
        return result


class MultiStrategyAnalyzer:
    """多策略分析器"""
    
    def __init__(self):
        """初始化多策略分析器"""
        self.strategies = [
            TrendFollowingStrategy(),
            ReversalStrategy(),
            VolatilityBreakoutStrategy()
        ]
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, data):
        """使用多种策略分析股票数据
        
        Args:
            data: DataFrame，包含股票数据
            
        Returns:
            dict: 综合分析结果
        """
        all_results = []
        
        # 运行每个策略
        for strategy in self.strategies:
            try:
                result = strategy.analyze(data)
                all_results.append(result)
                self.logger.info(f"策略 '{strategy.name}' 分析完成，得分: {result['score']}")
            except Exception as e:
                self.logger.error(f"策略 '{strategy.name}' 分析失败: {str(e)}")
        
        # 计算综合得分 (加权平均)
        weights = {
            "趋势跟踪策略": 0.4,
            "反转策略": 0.3,
            "波动率突破策略": 0.3
        }
        
        total_score = 0
        total_weight = 0
        
        for result in all_results:
            strategy_name = result["strategy_name"]
            if strategy_name in weights:
                total_score += result["score"] * weights[strategy_name]
                total_weight += weights[strategy_name]
        
        # 避免除零错误
        if total_weight > 0:
            combined_score = total_score / total_weight
        else:
            combined_score = 50  # 默认中性
        
        # 汇总所有信号
        all_signals = []
        for result in all_results:
            for signal in result["signals"]:
                # 添加策略来源
                signal_with_source = signal.copy()
                signal_with_source["strategy"] = result["strategy_name"]
                all_signals.append(signal_with_source)
        
        # 按权重排序信号
        all_signals.sort(key=lambda x: x["weight"], reverse=True)
        
        # 构建综合结果
        combined_result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "combined_score": round(combined_score, 2),
            "strategy_results": all_results,
            "top_signals": all_signals[:10],  # 取权重最高的前10个信号
            "stock_status": self._determine_stock_status(combined_score),
            "recommendation": self._generate_recommendation(combined_score, all_signals)
        }
        
        return combined_result
    
    def _determine_stock_status(self, score):
        """根据得分确定股票状态
        
        Args:
            score: 综合得分
            
        Returns:
            str: 股票状态描述
        """
        if score >= 80:
            return "强烈看涨"
        elif score >= 60:
            return "看涨"
        elif score >= 40:
            return "中性"
        elif score >= 20:
            return "看跌"
        else:
            return "强烈看跌"
    
    def _generate_recommendation(self, score, signals):
        """生成投资建议
        
        Args:
            score: 综合得分
            signals: 所有信号列表
            
        Returns:
            str: 投资建议
        """
        # 简单建议
        if score >= 80:
            return "可考虑建仓做多，股票呈现强劲上涨趋势"
        elif score >= 60:
            return "可考虑轻仓做多，关注技术指标确认"
        elif score >= 40:
            return "观望为主，等待更明确的市场信号"
        elif score >= 20:
            return "可考虑轻仓做空，密切关注反转信号"
        else:
            return "可考虑做空，股票呈现明显下跌趋势" 