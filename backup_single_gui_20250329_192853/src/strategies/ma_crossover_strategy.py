"""
移动平均线交叉策略
当短期均线上穿长期均线时买入，下穿时卖出
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from .strategy_base import StrategyBase
from ..indicators.technical_indicators import TechnicalIndicators

class MACrossoverStrategy(StrategyBase):
    """
    移动平均线交叉策略
    
    策略逻辑：
    1. 当短期均线上穿长期均线时买入
    2. 当短期均线下穿长期均线时卖出
    """
    
    def __init__(self, name: str = "MA_Crossover", **kwargs):
        """
        初始化移动平均线交叉策略
        
        Args:
            name: 策略名称
            **kwargs: 策略参数
                - short_period: 短期均线周期，默认5
                - long_period: 长期均线周期，默认20
                - ma_type: 均线类型，可选：simple, exponential, weighted，默认simple
                - use_volume_filter: 是否使用成交量过滤，默认False
                - volume_ratio: 成交量放大倍数，默认1.5
        """
        super().__init__(name)
        
        # 设置默认参数
        self.set_params(
            short_period=5,
            long_period=20,
            ma_type="simple",
            use_volume_filter=False,
            volume_ratio=1.5
        )
        
        # 更新用户参数
        if kwargs:
            self.set_params(**kwargs)
            
        self.indicators = TechnicalIndicators()
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 股票数据字典，key为股票代码，value为DataFrame
            
        Returns:
            pd.DataFrame: 交易信号DataFrame
        """
        self.logger.info("开始生成交易信号")
        
        signals = []
        
        # 获取策略参数
        short_period = self.params["short_period"]
        long_period = self.params["long_period"]
        ma_type = self.params["ma_type"]
        use_volume_filter = self.params["use_volume_filter"]
        volume_ratio = self.params["volume_ratio"]
        
        for stock_code, df in data.items():
            try:
                # 确保数据是按日期排序的
                df = df.sort_index()
                
                # 计算移动平均线
                if ma_type == "exponential":
                    ma_short = self.indicators.ema(df["close"], short_period)
                    ma_long = self.indicators.ema(df["close"], long_period)
                elif ma_type == "weighted":
                    ma_short = self.indicators.wma(df["close"], short_period)
                    ma_long = self.indicators.wma(df["close"], long_period)
                else:  # simple
                    ma_short = self.indicators.ma(df["close"], short_period)
                    ma_long = self.indicators.ma(df["close"], long_period)
                    
                # 计算金叉和死叉
                cross_above = (ma_short.shift(1) < ma_long.shift(1)) & (ma_short > ma_long)
                cross_below = (ma_short.shift(1) > ma_long.shift(1)) & (ma_short < ma_long)
                
                # 如果使用成交量过滤
                if use_volume_filter and "volume" in df.columns:
                    # 计算成交量的移动平均
                    volume_ma = df["volume"].rolling(window=5).mean()
                    
                    # 成交量放大条件
                    volume_expand = df["volume"] > volume_ma * volume_ratio
                    
                    # 应用成交量过滤
                    cross_above = cross_above & volume_expand
                    
                # 生成信号
                for date, row in df.iterrows():
                    if date in cross_above and cross_above[date]:
                        signals.append({
                            "date": date,
                            "stock_code": stock_code,
                            "signal": 1,  # 买入
                            "price": row["close"],
                            "reason": f"MA{short_period}上穿MA{long_period}"
                        })
                    elif date in cross_below and cross_below[date]:
                        signals.append({
                            "date": date,
                            "stock_code": stock_code,
                            "signal": -1,  # 卖出
                            "price": row["close"],
                            "reason": f"MA{short_period}下穿MA{long_period}"
                        })
                    else:
                        signals.append({
                            "date": date,
                            "stock_code": stock_code,
                            "signal": 0,  # 持仓不变
                            "price": row["close"],
                            "reason": "无交叉信号"
                        })
                        
            except Exception as e:
                self.logger.error(f"生成信号失败 [{stock_code}]: {str(e)}")
                
        # 转换为DataFrame
        if signals:
            signals_df = pd.DataFrame(signals)
            self.logger.info(f"信号生成完成，共{len(signals_df)}条信号")
            return signals_df
        else:
            self.logger.warning("没有生成任何信号")
            return pd.DataFrame() 