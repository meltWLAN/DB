#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票选股策略
识别即将连续涨停和大幅上涨的股票
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

from src.enhanced.data.data_manager import EnhancedDataManager
from src.enhanced.data.quality.data_quality_checker import DataQualityChecker

# 配置日志
logger = logging.getLogger(__name__)

class StockPicker:
    """股票选股策略类"""
    
    def __init__(self):
        """初始化选股策略"""
        self.data_manager = EnhancedDataManager()
        self.quality_checker = DataQualityChecker()
        logger.info("股票选股策略初始化完成")
    
    def analyze_stock(self, stock_code: str, days: int = 20) -> Dict:
        """
        分析单个股票的走势
        
        Args:
            stock_code: 股票代码
            days: 分析的天数
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 获取历史数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            df = self.data_manager.get_stock_data(stock_code, start_date, end_date)
            if df is None or df.empty:
                return None
            
            # 计算技术指标
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA10'] = df['close'].rolling(window=10).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            
            # 计算涨跌幅
            df['change_pct'] = df['close'].pct_change()
            
            # 计算波动率
            df['volatility'] = df['change_pct'].rolling(window=5).std()
            
            # 获取最新数据
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 分析结果
            result = {
                'stock_code': stock_code,
                'current_price': latest['close'],
                'change_pct': latest['change_pct'],
                'volume': latest['volume'],
                'volatility': latest['volatility'],
                'ma5': latest['MA5'],
                'ma10': latest['MA10'],
                'ma20': latest['MA20'],
                'price_trend': 'up' if latest['close'] > latest['MA5'] > latest['MA10'] else 'down',
                'volume_trend': 'up' if latest['volume'] > df['volume'].mean() else 'down',
                'limit_up_count': self._count_limit_up(df),
                'continuous_up_days': self._count_continuous_up_days(df),
                'risk_score': self._calculate_risk_score(df)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"分析股票 {stock_code} 时出错: {str(e)}")
            return None
    
    def _count_limit_up(self, df: pd.DataFrame) -> int:
        """统计涨停次数"""
        limit_up_threshold = 0.099  # 涨停阈值
        return (df['change_pct'] >= limit_up_threshold).sum()
    
    def _count_continuous_up_days(self, df: pd.DataFrame) -> int:
        """统计连续上涨天数"""
        count = 0
        for change in reversed(df['change_pct']):
            if change > 0:
                count += 1
            else:
                break
        return count
    
    def _calculate_risk_score(self, df: pd.DataFrame) -> float:
        """计算风险分数"""
        # 基于波动率、涨跌幅和成交量计算风险分数
        volatility_score = df['volatility'].iloc[-1] * 100
        change_score = abs(df['change_pct'].iloc[-1]) * 100
        volume_score = (df['volume'].iloc[-1] / df['volume'].mean()) * 100
        
        return (volatility_score + change_score + volume_score) / 3
    
    def find_potential_stocks(self, min_limit_up: int = 1, min_continuous_up: int = 2,
                            max_risk_score: float = 80.0) -> List[Dict]:
        """
        寻找潜在的连续涨停和大幅上涨股票
        
        Args:
            min_limit_up: 最小涨停次数
            min_continuous_up: 最小连续上涨天数
            max_risk_score: 最大风险分数
            
        Returns:
            List[Dict]: 符合条件的股票列表
        """
        try:
            # 获取股票列表
            stock_list = self.data_manager.get_stock_list()
            if stock_list is None or stock_list.empty:
                logger.error("获取股票列表失败")
                return []
            
            potential_stocks = []
            
            # 分析每只股票
            for _, stock in stock_list.iterrows():
                stock_code = stock['code']
                analysis = self.analyze_stock(stock_code)
                
                if analysis is None:
                    continue
                
                # 检查是否符合条件
                if (analysis['limit_up_count'] >= min_limit_up and
                    analysis['continuous_up_days'] >= min_continuous_up and
                    analysis['risk_score'] <= max_risk_score and
                    analysis['price_trend'] == 'up' and
                    analysis['volume_trend'] == 'up'):
                    
                    potential_stocks.append(analysis)
            
            # 按风险分数排序
            potential_stocks.sort(key=lambda x: x['risk_score'])
            
            return potential_stocks
            
        except Exception as e:
            logger.error(f"寻找潜在股票时出错: {str(e)}")
            return []
    
    def get_stock_recommendations(self, top_n: int = 10) -> List[Dict]:
        """
        获取股票推荐列表
        
        Args:
            top_n: 返回前N只股票
            
        Returns:
            List[Dict]: 推荐股票列表
        """
        try:
            # 寻找潜在股票
            potential_stocks = self.find_potential_stocks()
            
            # 获取前N只股票
            recommendations = potential_stocks[:top_n]
            
            # 添加推荐理由
            for stock in recommendations:
                stock['recommendation_reason'] = self._generate_recommendation_reason(stock)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"获取股票推荐时出错: {str(e)}")
            return []
    
    def _generate_recommendation_reason(self, stock: Dict) -> str:
        """生成推荐理由"""
        reasons = []
        
        if stock['limit_up_count'] > 0:
            reasons.append(f"已出现{stock['limit_up_count']}次涨停")
        
        if stock['continuous_up_days'] >= 2:
            reasons.append(f"连续{stock['continuous_up_days']}天上涨")
        
        if stock['price_trend'] == 'up':
            reasons.append("价格处于上升趋势")
        
        if stock['volume_trend'] == 'up':
            reasons.append("成交量放大")
        
        if stock['risk_score'] < 50:
            reasons.append("风险较低")
        
        return "，".join(reasons) 