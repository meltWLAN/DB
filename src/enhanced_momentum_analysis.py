#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版动量分析模块
提供更全面的动量分析功能，包括技术、资金、基本面和行业因素
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import tushare as ts

# 设置日志
logger = logging.getLogger(__name__)

# 设置Tushare Token
TUSHARE_TOKEN = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

class EnhancedMomentumAnalyzer:
    """增强版动量分析器"""
    
    def __init__(self, use_tushare=True, cache_timeout=86400):
        """初始化分析器"""
        self.use_tushare = use_tushare
        self.cache_timeout = cache_timeout
        self.cache = {}
        self.cache_timestamps = {}
        
        # 配置参数
        self.params = {
            'ma_periods': [5, 10, 20, 30, 60],  # 移动平均周期
            'rsi_period': 14,  # RSI周期
            'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},  # MACD参数
            'boll_period': 20,  # 布林带周期
            'boll_std': 2,  # 布林带标准差
            'volume_ma': [5, 10, 20],  # 成交量均线
            'momentum_period': 10,  # 动量计算周期
            'min_data_points': 60,  # 最小数据点要求
        }
        
        # 评分权重配置
        self.score_weights = {
            'technical': {  # 技术面权重 (总计1.0)
                'price_momentum': 0.15,  # 价格动量
                'ma_trend': 0.15,       # 均线趋势
                'rsi': 0.10,            # RSI
                'macd': 0.15,           # MACD
                'kdj': 0.10,            # KDJ
                'boll': 0.10,           # 布林带
                'volume': 0.15,         # 成交量
                'obv': 0.05,            # OBV
                'cci': 0.05             # CCI
            },
            'composite': {  # 综合评分权重 (总计1.0)
                'technical': 0.30,      # 技术面
                'money_flow': 0.25,     # 资金面
                'fundamental': 0.20,    # 基本面
                'industry': 0.15,       # 行业面
                'ml_model': 0.10        # 机器学习模型
            }
        }
        
        # 初始化机器学习模型
        try:
            from ml_momentum_model import MomentumMLModel
            self.ml_model = MomentumMLModel()
            self.ml_model.load_models()
            logger.info("成功加载机器学习模型")
        except Exception as e:
            logger.warning(f"加载机器学习模型失败: {str(e)}")
            self.ml_model = None
        
    def analyze_stocks(self, stocks: List[str], min_score: float = 60.0) -> List[Dict]:
        """分析股票列表"""
        results = []
        for stock in stocks:
            try:
                score = self.calculate_momentum_score(stock)
                if score >= min_score:
                    results.append({
                        'ts_code': stock,
                        'name': self.get_stock_name(stock),
                        'score': score,
                        'industry': self.get_stock_industry(stock)
                    })
            except Exception as e:
                logger.error(f"分析股票 {stock} 时出错: {str(e)}")
                continue
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def calculate_momentum_score(self, ts_code: str) -> float:
        """计算动量评分"""
        try:
            # 获取数据
            data = self.get_stock_data(ts_code)
            if data is None or len(data) < self.params['min_data_points']:
                return 0.0
            
            # 计算各维度得分
            tech_score = self.calculate_technical_score(data)
            money_score = self.analyze_money_flow(ts_code)
            fundamental_score = self.calculate_fundamental_score(ts_code)
            industry_score = self.analyze_industry_momentum(ts_code)
            
            # 获取机器学习模型预测得分
            ml_score = 50.0  # 默认中性分数
            if self.ml_model is not None:
                predictions = self.ml_model.predict(data)
                if predictions:
                    ml_score = sum(predictions.values()) / len(predictions)
            
            # 计算综合得分
            weights = self.score_weights['composite']
            momentum_score = (
                tech_score * weights['technical'] +
                money_score * weights['money_flow'] +
                fundamental_score * weights['fundamental'] +
                industry_score * weights['industry'] +
                ml_score * weights['ml_model']
            )
            
            return min(max(momentum_score, 0), 100)
            
        except Exception as e:
            logger.error(f"计算动量评分出错: {str(e)}")
            return 0.0
    
    def get_stock_data(self, ts_code: str, days: int = 60) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        try:
            if not self.use_tushare:
                return None
            
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                df.sort_values('trade_date', inplace=True)
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"获取股票数据出错: {str(e)}")
            return None
    
    def get_stock_name(self, ts_code: str) -> str:
        """获取股票名称"""
        try:
            if not self.use_tushare:
                return ts_code
            
            df = pro.stock_basic(ts_code=ts_code, fields='name')
            if df is not None and not df.empty:
                return df.iloc[0]['name']
            
            return ts_code
            
        except Exception as e:
            logger.error(f"获取股票名称出错: {str(e)}")
            return ts_code
    
    def get_stock_industry(self, ts_code: str) -> str:
        """获取股票所属行业"""
        try:
            if not self.use_tushare:
                return "未知行业"
            
            df = pro.stock_basic(ts_code=ts_code, fields='industry')
            if df is not None and not df.empty:
                return df.iloc[0]['industry']
            
            return "未知行业"
            
        except Exception as e:
            logger.error(f"获取股票行业出错: {str(e)}")
            return "未知行业"
    
    def calculate_technical_score(self, data: pd.DataFrame) -> float:
        """计算技术面得分"""
        try:
            scores = {}
            
            # 1. 价格动量
            price_change = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
            scores['price_momentum'] = min(max(50 + price_change, 0), 100)
            
            # 2. 均线趋势
            ma_trends = []
            for period in self.params['ma_periods']:
                ma = data['close'].rolling(window=period).mean()
                trend = (ma.iloc[-1] / ma.iloc[-5] - 1) * 100
                ma_trends.append(min(max(50 + trend, 0), 100))
            scores['ma_trend'] = sum(ma_trends) / len(ma_trends)
            
            # 3. RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.params['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['rsi_period']).mean()
            rs = gain / loss
            scores['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
            
            # 4. MACD
            exp1 = data['close'].ewm(span=self.params['macd_params']['fast']).mean()
            exp2 = data['close'].ewm(span=self.params['macd_params']['slow']).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=self.params['macd_params']['signal']).mean()
            hist = macd - signal
            scores['macd'] = min(max(50 + hist.iloc[-1] * 100, 0), 100)
            
            # 计算加权技术得分
            tech_score = sum(
                scores[k] * v for k, v in self.score_weights['technical'].items()
            )
            
            return tech_score
            
        except Exception as e:
            logger.error(f"计算技术面得分出错: {str(e)}")
            return 50.0
    
    def analyze_money_flow(self, ts_code: str) -> float:
        """分析资金流向"""
        try:
            # 获取最近的资金流向数据
            df = pro.moneyflow(ts_code=ts_code, start_date=(datetime.now() - timedelta(days=10)).strftime('%Y%m%d'))
            if df is None or df.empty:
                return 50.0
            
            # 计算主力净流入
            df['net_mf_amount'] = df['buy_lg_amount'] - df['sell_lg_amount']
            net_flow = df['net_mf_amount'].sum()
            
            # 转换为0-100的评分
            if net_flow > 0:
                score = min(50 + net_flow / 1000000, 100)  # 每100万资金流入提高1分
            else:
                score = max(50 + net_flow / 1000000, 0)  # 每100万资金流出降低1分
            
            return score
            
        except Exception as e:
            logger.error(f"分析资金流向出错: {str(e)}")
            return 50.0
    
    def calculate_fundamental_score(self, ts_code: str) -> float:
        """计算基本面得分"""
        try:
            # 获取最新财务指标
            df = pro.fina_indicator(ts_code=ts_code)
            if df is None or df.empty:
                return 50.0
            
            latest = df.iloc[0]
            scores = []
            
            # ROE得分
            if 'roe' in latest:
                roe_score = min(max(50 + latest['roe'], 0), 100)
                scores.append(roe_score)
            
            # 毛利率得分
            if 'grossprofit_margin' in latest:
                gpm_score = min(max(50 + latest['grossprofit_margin'] / 2, 0), 100)
                scores.append(gpm_score)
            
            # 资产负债率得分（越低越好）
            if 'debt_to_assets' in latest:
                dar_score = min(max(100 - latest['debt_to_assets'], 0), 100)
                scores.append(dar_score)
            
            return sum(scores) / len(scores) if scores else 50.0
            
        except Exception as e:
            logger.error(f"计算基本面得分出错: {str(e)}")
            return 50.0
    
    def analyze_industry_momentum(self, ts_code: str) -> float:
        """分析行业动量"""
        try:
            industry = self.get_stock_industry(ts_code)
            if industry == "未知行业":
                return 50.0
            
            # 获取同行业股票
            df = pro.stock_basic(fields='ts_code,industry')
            industry_stocks = df[df['industry'] == industry]['ts_code'].tolist()[:10]
            
            if not industry_stocks:
                return 50.0
            
            # 计算行业整体涨幅
            industry_changes = []
            for stock in industry_stocks:
                data = self.get_stock_data(stock, days=20)
                if data is not None and not data.empty:
                    change = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
                    industry_changes.append(change)
            
            if not industry_changes:
                return 50.0
            
            # 计算行业平均涨幅
            avg_change = sum(industry_changes) / len(industry_changes)
            
            # 转换为0-100的评分
            return min(max(50 + avg_change, 0), 100)
            
        except Exception as e:
            logger.error(f"分析行业动量出错: {str(e)}")
            return 50.0 