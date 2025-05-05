#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略模块
定义各种交易策略
"""

import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod

# 设置日志
logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """策略基类，定义了策略接口"""
    
    def __init__(self):
        """初始化策略"""
        self.engine = None
        self.risk_manager = None
        self.history = {}  # 保存策略所需的历史数据
        
    def set_engine(self, engine):
        """
        设置回测引擎
        
        Args:
            engine: 回测引擎实例
        """
        self.engine = engine
        
    def set_risk_manager(self, risk_manager):
        """
        设置风险管理器
        
        Args:
            risk_manager: 风险管理器实例
        """
        self.risk_manager = risk_manager
        
    def initialize(self):
        """初始化策略，在回测开始前调用"""
        # 在子类中实现
        pass
        
    @abstractmethod
    def on_data(self, date, data):
        """
        处理每日数据
        
        Args:
            date: 当前日期
            data: 当日市场数据
            
        Returns:
            orders: 交易指令列表
        """
        pass
        
    def buy(self, stock_code, price, shares=0, amount=0, stop_loss=None, take_profit=None):
        """
        买入股票
        
        Args:
            stock_code: 股票代码
            price: 买入价格
            shares: 买入数量（与amount二选一）
            amount: 买入金额（与shares二选一）
            stop_loss: 止损价
            take_profit: 止盈价
            
        Returns:
            position: 持仓信息
        """
        if self.risk_manager:
            position = self.risk_manager.add_position(
                stock_code=stock_code,
                price=price,
                shares=shares,
                amount=amount,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            return position
        return None
        
    def sell(self, stock_code, price=None):
        """
        卖出股票
        
        Args:
            stock_code: 股票代码
            price: 卖出价格，默认为当前价格
            
        Returns:
            position: 卖出的持仓信息
        """
        if self.risk_manager:
            position = self.risk_manager.remove_position(
                stock_code=stock_code,
                price=price
            )
            return position
        return None
        
    def update_positions(self, date, data):
        """
        更新持仓信息
        
        Args:
            date: 当前日期
            data: 当日市场数据
        """
        if not self.risk_manager:
            return
            
        # 获取当前持仓
        positions = self.risk_manager.positions.copy()
        
        for stock_code, position in positions.items():
            # 获取当日数据
            if stock_code in data:
                stock_data = data[stock_code]
                
                # 获取当日收盘价
                if date in stock_data.index:
                    current_price = stock_data.loc[date, 'close']
                    
                    # 更新持仓
                    self.risk_manager.update_position(stock_code, current_price)
                    
                    # 检查止损止盈
                    if self.risk_manager.check_stop_loss(stock_code):
                        self.sell(stock_code, current_price)
                        
                    if self.risk_manager.check_take_profit(stock_code):
                        self.sell(stock_code, current_price)


class MomentumStrategy(BaseStrategy):
    """动量策略：买入动量较好的股票，持有一段时间后卖出"""
    
    def __init__(self, lookback_period=20, momentum_threshold=0.05, profit_target=0.15, stop_loss=0.05, max_positions=5):
        """
        初始化动量策略
        
        Args:
            lookback_period: 回看期，用于计算动量的历史周期
            momentum_threshold: 动量阈值，只有动量大于阈值的股票才会被买入
            profit_target: 目标利润，达到这个利润就止盈
            stop_loss: 止损比例，亏损到这个比例就止损
            max_positions: 最大持仓数量
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_positions = max_positions
        
    def initialize(self):
        """初始化策略"""
        # 如果回测引擎不存在，直接返回
        if self.engine is None:
            return
            
        # 加载历史数据
        self.history = self.engine.data
        
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
        
        # 选择股票
        selected_stocks = self._select_stocks(date)
        
        # 买入选定的股票
        for stock_code, score in selected_stocks.items():
            # 获取当日数据
            if stock_code in data:
                stock_data = data[stock_code]
                
                # 获取当日收盘价
                if date in stock_data.index:
                    current_price = stock_data.loc[date, 'close']
                    
                    # 设置止损和止盈价格
                    stop_loss_price = current_price * (1 - self.stop_loss)
                    take_profit_price = current_price * (1 + self.profit_target)
                    
                    # 买入
                    self.buy(
                        stock_code=stock_code,
                        price=current_price,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price
                    )
        
        return []
        
    def _select_stocks(self, date):
        """
        根据动量分数选择股票
        
        Args:
            date: 当前日期
            
        Returns:
            dict: 选择的股票及其分数 {stock_code: score}
        """
        # 当前持有的股票
        current_positions = self._check_positions()
        
        # 计算所有股票的动量得分
        momentum_scores = {}
        
        for stock_code in self.engine.data.keys():
            # 如果已经持有该股票，跳过
            if stock_code in current_positions:
                continue
                
            # 获取该股票的历史数据
            stock_data = self.engine.data.get(stock_code)
            if stock_data is None:
                continue
                
            # 获取截止到当前日期的数据
            history = stock_data[stock_data.index <= date]
            if len(history) < self.lookback_period + 1:
                continue
                
            # 获取当前价格和历史价格
            current_price = history['close'].iloc[-1]
            previous_price = history['close'].iloc[-self.lookback_period-1]
            
            if previous_price <= 0:
                continue
                
            # 计算动量分数
            momentum = (current_price / previous_price) - 1
            
            # 如果动量大于阈值，加入候选
            if momentum > self.momentum_threshold:
                momentum_scores[stock_code] = momentum
                
        # 如果有足够的候选股票，选择动量最高的N只
        selected_stocks = {}
        if momentum_scores:
            # 按动量降序排序
            sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
            # 选择前N只股票，但不超过最大持仓数
            max_new_positions = self.max_positions - len(current_positions)
            selected_stocks = dict(sorted_stocks[:max_new_positions])
            
        return selected_stocks
        
    def _check_positions(self):
        """
        获取当前持仓信息
        
        Returns:
            dict: 当前持仓信息 {stock_code: position}
        """
        if not self.risk_manager:
            return {}
            
        return self.risk_manager.positions
