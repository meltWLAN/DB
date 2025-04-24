#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双动量策略 (Dual Momentum Strategy)
结合绝对动量和相对动量的高胜率策略

策略原理:
1. 绝对动量: 资产价格相对于自身历史表现的动量
2. 相对动量: 资产价格相对于其他资产表现的动量
3. 双动量结合: 先筛选出具有正绝对动量的资产，再从中选择相对动量最强的资产

参考文献:
- Gary Antonacci (2014) "Dual Momentum Investing: An Innovative Strategy for Higher Returns with Lower Risk"
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from ..backtest.strategy import BaseStrategy
from ..indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class DualMomentumStrategy(BaseStrategy):
    """
    双动量策略 (Dual Momentum Strategy)
    
    策略优势:
    1. 高胜率: 在强势市场中捕捉趋势，在弱势市场中规避风险
    2. 降低波动: 通过动量筛选和分散投资降低投资组合波动性
    3. 适应性强: 可适应不同市场环境
    
    典型参数设置:
    - 绝对动量判断周期: 3-12个月
    - 相对动量排名数量: 3-10只股票
    - 持有时间: 1-3个月
    """
    
    def __init__(self, 
                 absolute_lookback=60,  # 绝对动量回看期(天)
                 relative_lookback=120, # 相对动量回看期(天)
                 momentum_threshold=0,  # 绝对动量阈值
                 top_n=5,              # 选择相对动量排名前N的股票
                 holding_period=20,    # 持有期(天)
                 max_positions=5,      # 最大持仓数
                 weight_scheme='equal', # 权重分配方案: equal, momentum_weighted
                 profit_target=0.20,   # 止盈比例
                 stop_loss=0.08,       # 止损比例
                 risk_percent=0.02):   # 每笔交易风险比例
        """
        初始化双动量策略
        
        Args:
            absolute_lookback: 绝对动量回看期(天)
            relative_lookback: 相对动量回看期(天)
            momentum_threshold: 绝对动量阈值，默认为0(要求正动量)
            top_n: 选择相对动量排名前N的股票
            holding_period: 持有期(天)
            max_positions: 最大持仓数量
            weight_scheme: 权重分配方案
            profit_target: 止盈比例
            stop_loss: 止损比例
            risk_percent: 每笔交易风险比例
        """
        super().__init__()
        self.absolute_lookback = absolute_lookback
        self.relative_lookback = relative_lookback
        self.momentum_threshold = momentum_threshold
        self.top_n = top_n
        self.holding_period = holding_period
        self.max_positions = max_positions
        self.weight_scheme = weight_scheme
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.risk_percent = risk_percent
        
        # 持仓管理
        self.position_entry_dates = {}  # 记录入场日期
        self.tech_indicators = TechnicalIndicators()
        
        logger.info(f"初始化双动量策略: 绝对动量回看期={absolute_lookback}天, "
                   f"相对动量回看期={relative_lookback}天, 持有期={holding_period}天")
    
    def initialize(self):
        """初始化策略，在回测开始前调用"""
        if self.engine is None:
            return
            
        # 加载历史数据
        self.history = self.engine.data
        logger.info(f"加载历史数据，共{len(self.history)}只股票")
    
    def on_data(self, date, data):
        """
        处理每日数据
        
        Args:
            date: 当前日期
            data: 当日市场数据
            
        Returns:
            orders: 交易指令列表
        """
        # 更新持仓
        self.update_positions(date, data)
        
        # 检查是否应该退出已有仓位(基于持有时间)
        self._check_exit_positions(date, data)
        
        # 只有在仓位未满时才选择新股票
        current_positions = self._get_current_positions()
        if len(current_positions) >= self.max_positions:
            return []
        
        # 选择新的股票进行投资
        selected_stocks = self._select_stocks(date)
        available_positions = self.max_positions - len(current_positions)
        
        # 买入选定的股票
        for i, (stock_code, momentum_score) in enumerate(selected_stocks.items()):
            # 达到最大持仓数量，停止买入
            if i >= available_positions:
                break
                
            # 如果已持有该股票，跳过
            if stock_code in current_positions:
                continue
                
            # 获取当日数据
            if stock_code in data:
                stock_data = data[stock_code]
                
                # 获取当日收盘价
                if date in stock_data.index:
                    current_price = stock_data.loc[date, 'close']
                    
                    # 设置止损和止盈价格
                    stop_loss_price = current_price * (1 - self.stop_loss)
                    take_profit_price = current_price * (1 + self.profit_target)
                    
                    # 计算持仓量 - 基于风险比例
                    portfolio_value = self.risk_manager.portfolio_value
                    risk_amount = portfolio_value * self.risk_percent
                    position_size = risk_amount / (current_price * self.stop_loss)
                    
                    # 买入
                    position = self.buy(
                        stock_code=stock_code,
                        price=current_price,
                        shares=position_size,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price
                    )
                    
                    if position:
                        # 记录入场日期
                        self.position_entry_dates[stock_code] = date
                        logger.info(f"买入 {stock_code}: {position_size}股, 价格{current_price}, "
                                   f"动量得分{momentum_score:.4f}")
        
        return []
    
    def _select_stocks(self, date):
        """
        使用双动量策略选择股票
        
        Args:
            date: 当前日期
            
        Returns:
            dict: 选择的股票及其动量分数 {stock_code: momentum_score}
        """
        # 第一步: 计算所有股票的绝对动量
        absolute_momentum = {}
        
        for stock_code in self.engine.data.keys():
            stock_data = self.engine.data.get(stock_code)
            if stock_data is None or len(stock_data) < self.absolute_lookback:
                continue
                
            # 获取截止到当前日期的数据
            history = stock_data[stock_data.index <= date].copy()
            if len(history) < self.absolute_lookback:
                continue
            
            # 计算绝对动量 (当前价格相对于N天前价格的变化率)
            current_price = history['close'].iloc[-1]
            past_price = history['close'].iloc[-self.absolute_lookback]
            
            if past_price <= 0:
                continue
                
            abs_momentum = (current_price / past_price) - 1
            
            # 只保留绝对动量大于阈值的股票
            if abs_momentum > self.momentum_threshold:
                absolute_momentum[stock_code] = abs_momentum
        
        if not absolute_momentum:
            logger.info(f"没有股票通过绝对动量筛选")
            return {}
            
        logger.info(f"通过绝对动量筛选的股票数量: {len(absolute_momentum)}")
        
        # 第二步: 计算通过绝对动量筛选的股票的相对动量
        relative_momentum = {}
        
        for stock_code in absolute_momentum:
            stock_data = self.engine.data.get(stock_code)
            
            # 获取截止到当前日期的数据
            history = stock_data[stock_data.index <= date].copy()
            if len(history) < self.relative_lookback:
                continue
            
            # 计算相对动量 (结合多个时间周期的动量)
            # 使用3个不同周期的动量加权计算，增强稳定性
            short_momentum = history['close'].iloc[-1] / history['close'].iloc[-20] - 1  # 20天
            medium_momentum = history['close'].iloc[-1] / history['close'].iloc[-60] - 1  # 60天
            long_momentum = history['close'].iloc[-1] / history['close'].iloc[-min(self.relative_lookback, len(history)-1)] - 1  # 完整回看期
            
            # 加权计算综合动量分数 (短期占50%，中期占30%，长期占20%)
            momentum_score = 0.5 * short_momentum + 0.3 * medium_momentum + 0.2 * long_momentum
            
            # 添加技术指标评分增强动量判断
            tech_score = self._calculate_technical_score(history)
            combined_score = 0.7 * momentum_score + 0.3 * tech_score
            
            relative_momentum[stock_code] = combined_score
        
        # 第三步: 按相对动量排序并选择前N名
        sorted_stocks = sorted(relative_momentum.items(), key=lambda x: x[1], reverse=True)
        selected_stocks = dict(sorted_stocks[:self.top_n])
        
        logger.info(f"选定的股票: {list(selected_stocks.keys())}")
        
        return selected_stocks
    
    def _calculate_technical_score(self, history):
        """计算技术指标综合评分"""
        # 确保数据足够计算技术指标
        if len(history) < 60:
            return 0
            
        # 添加技术指标
        df = self.tech_indicators.add_all_indicators(history)
        
        # 获取最新的技术指标数据
        latest = df.iloc[-1]
        
        # 根据各指标给出评分 (0-1范围)
        score = 0
        total_weight = 0
        
        # 1. MACD指标 (20%)
        if not np.isnan(latest['macd']) and not np.isnan(latest['signal']):
            macd_score = 0
            # MACD为正且大于信号线
            if latest['macd'] > 0 and latest['macd'] > latest['signal']:
                macd_score = 0.5 + min(0.5, latest['macd_hist'] / latest['close'] * 100)  # 考虑MACD柱状图的强度
            # MACD为负但向上突破信号线
            elif latest['macd'] < 0 and latest['macd'] > latest['signal']:
                macd_score = 0.3
            score += 0.2 * macd_score
            total_weight += 0.2
            
        # 2. RSI指标 (20%)
        if not np.isnan(latest['rsi']):
            rsi_score = 0
            # RSI在50-70之间，上升趋势较强
            if 50 <= latest['rsi'] <= 70:
                rsi_score = (latest['rsi'] - 50) / 20
            # RSI在70-80之间，超买但仍有上升空间
            elif 70 < latest['rsi'] <= 80:
                rsi_score = 0.5
            score += 0.2 * rsi_score
            total_weight += 0.2
            
        # 3. 均线系统 (30%)
        ma_score = 0
        ma_count = 0
        
        for period in [5, 10, 20, 30, 60]:
            ma_key = f'ma{period}'
            if ma_key in df.columns and not np.isnan(latest[ma_key]):
                # 价格在均线之上得分高
                if latest['close'] > latest[ma_key]:
                    ma_score += (1 + (latest['close'] / latest[ma_key] - 1) * 5)  # 根据价格与均线距离加权
                    ma_count += 1
                    
        if ma_count > 0:
            ma_score = min(1, ma_score / ma_count)
            score += 0.3 * ma_score
            total_weight += 0.3
            
        # 4. 布林带指标 (15%)
        if 'bb_percent_b' in df.columns and not np.isnan(latest['bb_percent_b']):
            bb_score = 0
            # 位于布林带中上部(0.5-0.8)最理想
            if 0.5 <= latest['bb_percent_b'] <= 0.8:
                bb_score = (latest['bb_percent_b'] - 0.5) * 3.33  # 最高分为1
            # 位于布林带上部(0.8-1.0)仍可接受
            elif 0.8 < latest['bb_percent_b'] <= 1.0:
                bb_score = 1 - (latest['bb_percent_b'] - 0.8) * 2  # 0.8分到0.5分
            score += 0.15 * bb_score
            total_weight += 0.15
            
        # 5. 成交量指标 (15%)
        volume_score = 0
        if len(df) >= 20:
            # 计算当前成交量相对于20日平均的比值
            recent_vol_ratio = latest['volume'] / df['volume'].rolling(20).mean().iloc[-1]
            # 成交量温和放大得分高
            if 1.0 <= recent_vol_ratio <= 2.0:
                volume_score = recent_vol_ratio / 2
            # 成交量过大不一定是好事
            elif recent_vol_ratio > 2.0:
                volume_score = 1 - min(0.5, (recent_vol_ratio - 2) / 4)
            score += 0.15 * volume_score
            total_weight += 0.15
            
        # 归一化最终得分
        return score / total_weight if total_weight > 0 else 0
        
    def _check_exit_positions(self, date, data):
        """检查是否应该退出已有仓位(基于持有时间)"""
        positions_to_exit = []
        
        for stock_code, entry_date in self.position_entry_dates.items():
            if stock_code not in self.risk_manager.positions:
                # 持仓已经被清掉(可能是止损或止盈)，从记录中移除
                positions_to_exit.append(stock_code)
                continue
                
            # 计算持有天数
            if isinstance(date, str):
                date = pd.Timestamp(date)
            if isinstance(entry_date, str):
                entry_date = pd.Timestamp(entry_date)
                
            # 如果持有时间超过预定期限，平仓
            holding_days = (date - entry_date).days
            if holding_days >= self.holding_period:
                if stock_code in data:
                    stock_data = data[stock_code]
                    if date in stock_data.index:
                        current_price = stock_data.loc[date, 'close']
                        self.sell(stock_code, current_price)
                        logger.info(f"持有期到期卖出 {stock_code}: 价格{current_price}, 持有{holding_days}天")
                positions_to_exit.append(stock_code)
                
        # 清理记录
        for stock_code in positions_to_exit:
            if stock_code in self.position_entry_dates:
                del self.position_entry_dates[stock_code]
                
    def _get_current_positions(self):
        """获取当前持仓"""
        if not self.risk_manager:
            return {}
        return self.risk_manager.positions 