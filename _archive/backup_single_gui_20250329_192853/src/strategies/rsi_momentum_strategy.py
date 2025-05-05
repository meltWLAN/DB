#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RSI动量策略 (RSI Momentum Strategy)
结合RSI指标和价格动量的高胜率策略

策略原理:
1. 使用RSI指标识别超买超卖区域
2. 结合价格动量判断趋势强度
3. 在RSI反转时信号增强的基础上，进行交易决策

特点:
- 高胜率: 在超买超卖区域反转时胜率显著提高
- 波动率小: 通过严格的入场条件减少噪音交易
- 适应多种市场环境
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from ..backtest.strategy import BaseStrategy
from ..indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class RSIMomentumStrategy(BaseStrategy):
    """
    RSI动量策略
    
    策略核心:
    1. 使用RSI的超买超卖区域作为交易信号的基础
    2. 结合价格动量确认趋势方向
    3. 在确认的信号上使用仓位管理和止损策略
    """
    
    def __init__(self, 
                 rsi_period=14,           # RSI计算周期
                 rsi_upper=70,            # RSI超买阈值
                 rsi_lower=30,            # RSI超卖阈值
                 momentum_period=20,      # 动量计算周期
                 momentum_threshold=0.02, # 动量阈值
                 exit_rsi=50,             # 平仓RSI阈值
                 max_holding_period=15,   # 最大持有期(天)
                 max_positions=5,         # 最大持仓数
                 profit_target=0.15,      # 止盈比例
                 stop_loss=0.05,          # 止损比例
                 risk_per_trade=0.02):    # 每笔交易风险比例
        """
        初始化RSI动量策略
        
        Args:
            rsi_period: RSI计算周期
            rsi_upper: RSI超买阈值
            rsi_lower: RSI超卖阈值
            momentum_period: 动量计算周期
            momentum_threshold: 动量阈值
            exit_rsi: 平仓RSI阈值
            max_holding_period: 最大持有期(天)
            max_positions: 最大持仓数
            profit_target: 止盈比例
            stop_loss: 止损比例
            risk_per_trade: 每笔交易风险比例
        """
        super().__init__()
        self.rsi_period = rsi_period
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold
        self.exit_rsi = exit_rsi
        self.max_holding_period = max_holding_period
        self.max_positions = max_positions
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.risk_per_trade = risk_per_trade
        
        # 交易状态管理
        self.position_entry_dates = {}  # 记录入场日期
        self.signals = {}               # 信号记录
        self.tech_indicators = TechnicalIndicators()
        
        logger.info(f"初始化RSI动量策略: RSI周期={rsi_period}, 超买阈值={rsi_upper}, 超卖阈值={rsi_lower}")
    
    def initialize(self):
        """初始化策略，在回测开始前调用"""
        if self.engine is None:
            return
            
        # 加载历史数据
        self.history = self.engine.data
        logger.info(f"RSI动量策略历史数据加载完成，共{len(self.history)}只股票")
        
        # 初始化信号字典
        for stock_code in self.history.keys():
            self.signals[stock_code] = {
                'rsi_signal': 0,       # RSI信号：1买入，-1卖出，0无信号
                'momentum_signal': 0,  # 动量信号：1正动量，-1负动量，0无信号
                'combined_signal': 0,  # 综合信号
                'last_signal_date': None  # 上次信号日期
            }
    
    def on_data(self, date, data):
        """
        处理每日数据
        
        Args:
            date: 当前日期
            data: 当日市场数据
            
        Returns:
            orders: 交易指令列表
        """
        # 确保date是timestamp格式
        if isinstance(date, str):
            date = pd.Timestamp(date)
            
        # 更新持仓和检查止损止盈
        self.update_positions(date, data)
        
        # 检查是否应该基于持有时间退出持仓
        self._check_max_holding_period(date, data)
        
        # 获取当前持仓
        current_positions = self._get_current_positions()
        
        # 只有在持仓未满时继续
        if len(current_positions) >= self.max_positions:
            return []
            
        # 可用仓位数
        available_positions = self.max_positions - len(current_positions)
        
        # 计算每只股票的信号
        all_signals = self._calculate_signals(date, data)
        
        # 买入信号强度排序
        buy_signals = {}
        for stock_code, signal in all_signals.items():
            # 只考虑买入信号且不在当前持仓中的股票
            if signal['combined_signal'] > 0 and stock_code not in current_positions:
                buy_signals[stock_code] = signal['signal_strength']
                
        # 按信号强度排序
        sorted_signals = sorted(buy_signals.items(), key=lambda x: x[1], reverse=True)
        
        # 买入前N只股票（不超过可用仓位）
        stocks_to_buy = dict(sorted_signals[:available_positions])
        
        # 执行买入操作
        for stock_code, signal_strength in stocks_to_buy.items():
            self._execute_buy(date, stock_code, data, signal_strength)
            
        # 检查是否应该基于RSI穿越中值平仓
        for stock_code in list(current_positions.keys()):
            self._check_rsi_exit(date, stock_code, data)
            
        return []
    
    def _calculate_signals(self, date, data):
        """计算所有股票的信号"""
        signals = {}
        
        for stock_code, stock_data in data.items():
            # 跳过没有足够数据的股票
            if stock_code not in self.history:
                continue
                
            # 获取历史数据直到当前日期
            history = self.history[stock_code][self.history[stock_code].index <= date].copy()
            if len(history) < max(self.rsi_period, self.momentum_period) + 10:
                continue
                
            # 添加技术指标
            df = self.tech_indicators.add_all_indicators(history)
            
            # 确保当前日期的数据存在
            if date not in df.index:
                continue
                
            # 获取当前和前一天的RSI值
            current_rsi = df.loc[date, 'rsi']
            prev_rsi = df.loc[df.index[-2], 'rsi'] if len(df) > 1 else np.nan
            
            # 计算动量
            current_price = df.loc[date, 'close']
            past_price = df.loc[df.index[-self.momentum_period-1], 'close'] if len(df) > self.momentum_period else np.nan
            
            if np.isnan(current_rsi) or np.isnan(prev_rsi) or np.isnan(past_price):
                continue
                
            momentum = (current_price / past_price - 1) if past_price > 0 else 0
            
            # RSI信号计算
            rsi_signal = 0
            # RSI从超卖区域向上穿越 - 买入信号
            if prev_rsi <= self.rsi_lower and current_rsi > self.rsi_lower:
                rsi_signal = 1
            # RSI从超买区域向下穿越 - 卖出信号
            elif prev_rsi >= self.rsi_upper and current_rsi < self.rsi_upper:
                rsi_signal = -1
                
            # 动量信号计算
            momentum_signal = 0
            if momentum > self.momentum_threshold:
                momentum_signal = 1
            elif momentum < -self.momentum_threshold:
                momentum_signal = -1
                
            # 综合信号 (RSI信号 * 0.7 + 动量信号 * 0.3)
            combined_signal = 0
            # 买入条件：RSI从超卖区上穿且动量为正
            if rsi_signal == 1 and momentum_signal >= 0:
                combined_signal = 1
            # 卖出条件：RSI从超买区下穿且动量为负
            elif rsi_signal == -1 and momentum_signal <= 0:
                combined_signal = -1
                
            # 信号强度计算 (0-1之间)
            signal_strength = 0
            if combined_signal != 0:
                # RSI距离中值(50)越远，信号越强
                rsi_strength = abs(current_rsi - 50) / 50
                # 动量绝对值越大，信号越强
                momentum_strength = min(1, abs(momentum) / 0.1)  # 限制在0-1之间
                # 加权平均
                signal_strength = 0.7 * rsi_strength + 0.3 * momentum_strength
                
            # 更新信号
            signals[stock_code] = {
                'rsi': current_rsi,
                'momentum': momentum,
                'rsi_signal': rsi_signal,
                'momentum_signal': momentum_signal,
                'combined_signal': combined_signal,
                'signal_strength': signal_strength
            }
            
        return signals
        
    def _execute_buy(self, date, stock_code, data, signal_strength):
        """执行买入操作"""
        # 获取当日股票数据
        if stock_code in data:
            stock_data = data[stock_code]
            
            # 获取当日收盘价
            if date in stock_data.index:
                current_price = stock_data.loc[date, 'close']
                
                # 设置止损和止盈价格
                stop_loss_price = current_price * (1 - self.stop_loss)
                take_profit_price = current_price * (1 + self.profit_target)
                
                # 根据风险计算持仓规模
                portfolio_value = self.risk_manager.portfolio_value
                risk_amount = portfolio_value * self.risk_per_trade
                position_size = risk_amount / (current_price * self.stop_loss)
                
                # 执行买入
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
                    logger.info(f"RSI动量策略买入 {stock_code}: {position_size}股, 价格{current_price}, "
                               f"信号强度{signal_strength:.4f}")
                               
    def _check_rsi_exit(self, date, stock_code, data):
        """检查是否应该基于RSI值平仓"""
        # 股票今日数据
        if stock_code in self.history:
            # 获取历史数据直到当前日期
            history = self.history[stock_code][self.history[stock_code].index <= date].copy()
            
            # 添加技术指标
            df = self.tech_indicators.add_all_indicators(history)
            
            # 获取当前RSI值
            if date in df.index:
                current_rsi = df.loc[date, 'rsi']
                
                # 获取当前持仓方向
                direction = 1  # 假设所有持仓都是多头
                
                # RSI穿越中值退出条件
                # 多头持仓：当RSI从上方穿越中值，卖出
                if direction > 0 and current_rsi < self.exit_rsi:
                    if stock_code in data:
                        stock_data = data[stock_code]
                        if date in stock_data.index:
                            current_price = stock_data.loc[date, 'close']
                            self.sell(stock_code, current_price)
                            logger.info(f"RSI动量策略卖出 {stock_code}: RSI={current_rsi:.2f} 低于中值退出")
    
    def _check_max_holding_period(self, date, data):
        """检查是否超过最大持有期限"""
        positions_to_exit = []
        
        for stock_code, entry_date in self.position_entry_dates.items():
            if stock_code not in self.risk_manager.positions:
                # 持仓已经被清掉(可能是止损或止盈)，从记录中移除
                positions_to_exit.append(stock_code)
                continue
                
            # 计算持有天数
            if isinstance(entry_date, str):
                entry_date = pd.Timestamp(entry_date)
                
            # 如果持有时间超过预定期限，平仓
            holding_days = (date - entry_date).days
            if holding_days >= self.max_holding_period:
                if stock_code in data:
                    stock_data = data[stock_code]
                    if date in stock_data.index:
                        current_price = stock_data.loc[date, 'close']
                        self.sell(stock_code, current_price)
                        logger.info(f"RSI动量策略卖出 {stock_code}: 价格{current_price}, 持有{holding_days}天，超过最大持有期")
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