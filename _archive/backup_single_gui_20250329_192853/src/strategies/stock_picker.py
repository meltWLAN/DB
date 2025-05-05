"""
股票选择器模块
负责股票推荐和筛选逻辑
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from ..utils.logger import get_logger
from ..config.settings import (
    RECOMMENDATION_CONFIG,
    ANALYSIS_CONFIG
)
from ..data.data_manager import DataManager

class StockPicker:
    """股票选择器类"""
    
    def __init__(self):
        """初始化股票选择器"""
        self.logger = get_logger("StockPicker")
        self.data_manager = DataManager()
        
    def get_stock_recommendations(
        self,
        top_n: int = 10,
        risk_level: str = "中等",
        industry: str = "全部"
    ) -> List[Dict[str, Any]]:
        """
        获取股票推荐
        
        Args:
            top_n: 推荐数量
            risk_level: 风险等级，可选：保守、中等、激进
            industry: 行业筛选
            
        Returns:
            List[Dict[str, Any]]: 推荐股票列表，每个股票包含：
                - stock_code: 股票代码
                - stock_name: 股票名称
                - current_price: 当前价格
                - change_pct: 涨跌幅
                - risk_score: 风险分数
                - recommendation_reason: 推荐理由
        """
        try:
            self.logger.info(f"开始获取股票推荐 - 数量: {top_n}, 风险等级: {risk_level}, 行业: {industry}")
            
            # 获取股票列表
            stock_list = self.data_manager.get_stock_list()
            if stock_list is None or stock_list.empty:
                self.logger.error("获取股票列表失败")
                return []
                
            # 行业筛选
            if industry != "全部":
                stock_list = stock_list[stock_list["industry"] == industry]
                
            # 获取推荐股票
            recommendations = []
            for _, row in stock_list.iterrows():
                stock_code = row["code"]
                stock_name = row["name"]
                
                # 获取股票数据
                df = self.data_manager.get_stock_data(stock_code)
                if df is None or df.empty:
                    continue
                    
                # 计算指标
                current_price = df["close"].iloc[-1]
                change_pct = (current_price - df["close"].iloc[-2]) / df["close"].iloc[-2]
                
                # 风险评分
                risk_score = self._calculate_risk_score(df, risk_level)
                
                # 推荐理由
                reason = self._generate_recommendation_reason(df, risk_score)
                
                # 添加到推荐列表
                recommendations.append({
                    "stock_code": stock_code,
                    "stock_name": stock_name,
                    "current_price": current_price,
                    "change_pct": change_pct,
                    "risk_score": risk_score,
                    "recommendation_reason": reason
                })
                
            # 排序并返回前N个推荐
            recommendations.sort(key=lambda x: x["risk_score"], reverse=True)
            return recommendations[:top_n]
            
        except Exception as e:
            self.logger.error("获取股票推荐失败", exc_info=e)
            return []
            
    def _calculate_risk_score(self, df: pd.DataFrame, risk_level: str) -> float:
        """
        计算风险分数
        
        Args:
            df: 股票数据
            risk_level: 风险等级
            
        Returns:
            float: 风险分数
        """
        try:
            # 计算波动率
            returns = df["close"].pct_change()
            volatility = returns.std() * np.sqrt(252)
            
            # 计算夏普比率
            risk_free_rate = 0.03  # 假设无风险利率为3%
            excess_returns = returns - risk_free_rate/252
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            
            # 计算最大回撤
            cummax = df["close"].cummax()
            drawdown = (df["close"] - cummax) / cummax
            max_drawdown = drawdown.min()
            
            # 根据风险等级调整权重
            if risk_level == "保守":
                weights = {"volatility": 0.4, "sharpe": 0.3, "drawdown": 0.3}
            elif risk_level == "激进":
                weights = {"volatility": 0.2, "sharpe": 0.5, "drawdown": 0.3}
            else:  # 中等
                weights = {"volatility": 0.3, "sharpe": 0.4, "drawdown": 0.3}
                
            # 计算综合得分
            score = (
                weights["volatility"] * (1 - volatility) +
                weights["sharpe"] * (sharpe_ratio + 2) / 4 +
                weights["drawdown"] * (1 + max_drawdown)
            )
            
            return max(0, min(1, score))
            
        except Exception as e:
            self.logger.error("计算风险分数失败", exc_info=e)
            return 0.0
            
    def _generate_recommendation_reason(self, df: pd.DataFrame, risk_score: float) -> str:
        """
        生成推荐理由
        
        Args:
            df: 股票数据
            risk_score: 风险分数
            
        Returns:
            str: 推荐理由
        """
        try:
            reasons = []
            
            # 计算技术指标
            ma5 = df["close"].rolling(window=5).mean()
            ma20 = df["close"].rolling(window=20).mean()
            
            # 趋势判断
            if df["close"].iloc[-1] > ma5.iloc[-1] > ma20.iloc[-1]:
                reasons.append("处于上升趋势")
            elif df["close"].iloc[-1] < ma5.iloc[-1] < ma20.iloc[-1]:
                reasons.append("处于下降趋势")
            else:
                reasons.append("处于震荡整理")
                
            # 成交量分析
            volume_ma5 = df["volume"].rolling(window=5).mean()
            if df["volume"].iloc[-1] > volume_ma5.iloc[-1] * 1.5:
                reasons.append("成交量显著放大")
            elif df["volume"].iloc[-1] < volume_ma5.iloc[-1] * 0.5:
                reasons.append("成交量明显萎缩")
                
            # 风险评分说明
            if risk_score >= 0.8:
                reasons.append("风险较低")
            elif risk_score >= 0.6:
                reasons.append("风险适中")
            else:
                reasons.append("风险较高")
                
            return "，".join(reasons)
            
        except Exception as e:
            self.logger.error("生成推荐理由失败", exc_info=e)
            return "数据不足，无法生成推荐理由"
            
    def get_stock_analysis(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        获取股票分析报告
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 分析报告，包含：
                - technical_analysis: 技术分析
                - fundamental_analysis: 基本面分析
                - risk_analysis: 风险分析
        """
        try:
            # 获取股票数据
            df = self.data_manager.get_stock_data(stock_code)
            if df is None or df.empty:
                return None
                
            # 获取股票信息
            stock_info = self.data_manager.get_stock_info(stock_code)
            if stock_info is None:
                return None
                
            # 技术分析
            technical = self._analyze_technical(df)
            
            # 基本面分析
            fundamental = self._analyze_fundamental(stock_info)
            
            # 风险分析
            risk = self._analyze_risk(df)
            
            return {
                "technical_analysis": technical,
                "fundamental_analysis": fundamental,
                "risk_analysis": risk
            }
            
        except Exception as e:
            self.logger.error(f"获取股票分析报告失败: {stock_code}", exc_info=e)
            return None
            
    def _analyze_technical(self, df: pd.DataFrame) -> Dict[str, Any]:
        """技术分析"""
        try:
            # 计算技术指标
            ma5 = df["close"].rolling(window=5).mean()
            ma20 = df["close"].rolling(window=20).mean()
            
            # 计算MACD
            exp1 = df["close"].ewm(span=12, adjust=False).mean()
            exp2 = df["close"].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            # 计算RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return {
                "ma5": ma5.iloc[-1],
                "ma20": ma20.iloc[-1],
                "macd": macd.iloc[-1],
                "macd_signal": signal.iloc[-1],
                "rsi": rsi.iloc[-1],
                "volume": df["volume"].iloc[-1]
            }
            
        except Exception as e:
            self.logger.error("技术分析失败", exc_info=e)
            return {}
            
    def _analyze_fundamental(self, stock_info: Dict[str, Any]) -> Dict[str, Any]:
        """基本面分析"""
        try:
            return {
                "industry": stock_info.get("industry", ""),
                "area": stock_info.get("area", ""),
                "list_date": stock_info.get("list_date", ""),
                "market": stock_info.get("market", "")
            }
            
        except Exception as e:
            self.logger.error("基本面分析失败", exc_info=e)
            return {}
            
    def _analyze_risk(self, df: pd.DataFrame) -> Dict[str, Any]:
        """风险分析"""
        try:
            # 计算波动率
            returns = df["close"].pct_change()
            volatility = returns.std() * np.sqrt(252)
            
            # 计算最大回撤
            cummax = df["close"].cummax()
            drawdown = (df["close"] - cummax) / cummax
            max_drawdown = drawdown.min()
            
            # 计算夏普比率
            risk_free_rate = 0.03
            excess_returns = returns - risk_free_rate/252
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            
            return {
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio
            }
            
        except Exception as e:
            self.logger.error("风险分析失败", exc_info=e)
            return {} 