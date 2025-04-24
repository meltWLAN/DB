#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
风险管理模块
用于管理仓位、止盈止损和风险控制
"""

import logging
from typing import Dict, List, Union, Optional, Any
import pandas as pd
import numpy as np

# 设置日志
logger = logging.getLogger(__name__)

class RiskManager:
    """风险管理类"""
    
    def __init__(self, max_position_per_stock=0.1, max_position_per_industry=0.3,
                 max_industry_allocation=0.3, stop_loss=None, take_profit=None,
                 use_trailing_stop=True, max_drawdown=0.2, risk_free_rate=0.03,
                 atr_multiplier=2.0, trailing_stop_activation_pct=0.05):
        """初始化风险管理器"""
        self.positions = {}  # 当前持仓
        self.capital = 0.0   # 可用资金
        self.initial_capital = 0.0  # 初始资金
        
        # 风险控制参数
        self.max_position_per_stock = max_position_per_stock  # 单个股票最大仓位比例
        self.max_position_per_industry = max_position_per_industry  # 单个行业最大仓位比例
        self.max_industry_allocation = max_industry_allocation  # 单一行业最大配置比例
        self.use_trailing_stop = use_trailing_stop  # 是否使用追踪止损
        self.max_drawdown = max_drawdown  # 最大回撤
        self.risk_free_rate = risk_free_rate  # 无风险利率
        
        # 止损止盈参数
        self.stop_loss = stop_loss or {
            "fixed": 0.05,
            "trailing": 0.08,
            "time_based": 0.03,
            "atr_multiplier": atr_multiplier,
            "atr_period": 14,
            "trailing_percentage": trailing_stop_activation_pct
        }
        
        self.take_profit = take_profit or {
            "fixed": 0.15,
            "trailing": 0.1,
            "time_based": 0.2,
            "atr_multiplier": atr_multiplier * 1.5,
            "atr_period": 14,
            "trailing_percentage": trailing_stop_activation_pct * 2
        }
        
    def set_capital(self, capital: float) -> None:
        """
        设置初始资金
        
        Args:
            capital: 初始资金金额
        """
        self.capital = capital
        self.initial_capital = capital
        logger.info(f"设置初始资金: {capital:.2f}")
        
    def add_position(self, stock_code: str, price: float, 
                    shares: int = 0, amount: float = 0, 
                    stop_loss: float = None, take_profit: float = None) -> Dict:
        """
        添加持仓
        
        Args:
            stock_code: 股票代码
            price: 买入价格
            shares: 股数(与amount二选一)
            amount: 买入金额(与shares二选一)
            stop_loss: 止损价
            take_profit: 止盈价
            
        Returns:
            position: 持仓信息
        """
        # 检查股票代码
        if not stock_code:
            logger.error("无效的股票代码")
            return None
            
        # 检查是否已持有该股票
        if stock_code in self.positions:
            logger.warning(f"已持有股票 {stock_code}, 请使用update_position更新")
            return self.positions[stock_code]
            
        # 检查价格是否有效
        if price <= 0:
            logger.error(f"无效的买入价格: {price}")
            return None
            
        # 计算股数和金额
        if shares > 0:
            amount = shares * price
        elif amount > 0:
            shares = int(amount / price)
        else:
            # 如果未指定股数或金额，使用默认仓位比例
            amount = self.initial_capital * self.max_position_per_stock
            shares = int(amount / price)
            
        # 检查是否有足够的资金
        if amount > self.capital:
            logger.warning(f"资金不足, 需要 {amount:.2f}, 可用 {self.capital:.2f}")
            return None
            
        # 设置止损和止盈
        if stop_loss is None:
            stop_loss = price * 0.95  # 默认止损5%
            
        if take_profit is None:
            take_profit = price * 1.1  # 默认止盈10%
            
        # 创建持仓信息
        position = {
            'stock_code': stock_code,
            'shares': shares,
            'price': price,
            'amount': amount,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_price': price,
            'profit_loss': 0.0,
            'profit_loss_pct': 0.0
        }
        
        # 更新可用资金
        self.capital -= amount
        
        # 添加到持仓
        self.positions[stock_code] = position
        logger.info(f"添加持仓: {stock_code}, 价格: {price:.2f}, 数量: {shares}, 金额: {amount:.2f}")
        
        return position
        
    def update_position(self, stock_code: str, current_price: float) -> Dict:
        """
        更新持仓信息
        
        Args:
            stock_code: 股票代码
            current_price: 当前价格
            
        Returns:
            position: 更新后的持仓信息
        """
        # 检查是否持有该股票
        if stock_code not in self.positions:
            logger.warning(f"未持有股票 {stock_code}, 无法更新")
            return None
            
        # 获取持仓信息
        position = self.positions[stock_code]
        
        # 计算盈亏
        original_amount = position['amount']
        current_amount = position['shares'] * current_price
        profit_loss = current_amount - original_amount
        profit_loss_pct = profit_loss / original_amount
        
        # 更新持仓信息
        position['current_price'] = current_price
        position['profit_loss'] = profit_loss
        position['profit_loss_pct'] = profit_loss_pct
        
        # 更新持仓字典
        self.positions[stock_code] = position
        
        return position
        
    def check_stop_loss(self, stock_code: str, current_price: float = None) -> bool:
        """
        检查是否触发止损
        
        Args:
            stock_code: 股票代码
            current_price: 当前价格，默认为最近更新的价格
            
        Returns:
            bool: 是否触发止损
        """
        # 检查是否持有该股票
        if stock_code not in self.positions:
            return False
            
        # 获取持仓信息
        position = self.positions[stock_code]
        
        # 如果未指定当前价格，使用最近更新的价格
        if current_price is None:
            current_price = position['current_price']
            
        # 检查是否触发止损
        if current_price <= position['stop_loss']:
            logger.info(f"触发止损: {stock_code}, 当前价格: {current_price:.2f} <= 止损价: {position['stop_loss']:.2f}")
            return True
            
        return False
        
    def check_take_profit(self, stock_code: str, current_price: float = None) -> bool:
        """
        检查是否触发止盈
        
        Args:
            stock_code: 股票代码
            current_price: 当前价格，默认为最近更新的价格
            
        Returns:
            bool: 是否触发止盈
        """
        # 检查是否持有该股票
        if stock_code not in self.positions:
            return False
            
        # 获取持仓信息
        position = self.positions[stock_code]
        
        # 如果未指定当前价格，使用最近更新的价格
        if current_price is None:
            current_price = position['current_price']
            
        # 检查是否触发止盈
        if current_price >= position['take_profit']:
            logger.info(f"触发止盈: {stock_code}, 当前价格: {current_price:.2f} >= 止盈价: {position['take_profit']:.2f}")
            return True
            
        return False
