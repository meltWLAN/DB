#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测引擎模块
用于处理历史数据回测
"""

import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Any, Optional, Union

# 设置日志
logger = logging.getLogger(__name__)

class BacktestEngine:
    """回测引擎类，用于回测策略"""
    
    def __init__(self, data, strategy, start_date, end_date, initial_capital=1000000.0, commission=0.0003, slippage=0.0002):
        """
        初始化回测引擎
        
        Args:
            data: 历史数据，格式为 {stock_code: DataFrame}
            strategy: 回测策略
            start_date: 回测开始日期
            end_date: 回测结束日期
            initial_capital: 初始资金
            commission: 交易佣金
            slippage: 滑点
        """
        self.data = data  # 历史数据
        self.strategy = strategy  # 回测策略
        self.start_date = pd.to_datetime(start_date)  # 回测开始日期
        self.end_date = pd.to_datetime(end_date)  # 回测结束日期
        self.initial_capital = initial_capital  # 初始资金
        self.commission = commission  # 交易佣金
        self.slippage = slippage  # 滑点
        
        # 将策略与引擎关联
        self.strategy.set_engine(self)
        
        # 回测结果
        self.trades = []  # 交易记录
        self.daily_returns = []  # 每日收益
        self.equity_curve = []  # 权益曲线
        self.positions_history = []  # 持仓历史
        
        # 交易统计
        self.total_trades = 0  # 总交易次数
        self.winning_trades = 0  # 盈利交易次数
        self.losing_trades = 0  # 亏损交易次数
        self.total_return = 0.0  # 总收益率
        self.annual_return = 0.0  # 年化收益率
        self.max_drawdown = 0.0  # 最大回撤
        self.sharpe_ratio = 0.0  # 夏普比率
        self.win_rate = 0.0  # 胜率
        self.avg_holding_period = 0.0  # 平均持仓周期
        
        # 回测日期序列
        self.dates = self._generate_dates()
        
        logger.info(f"初始化回测引擎: {len(self.data)} 只股票, 时间范围: {start_date} 至 {end_date}")
    
    def _generate_dates(self):
        """生成回测日期序列"""
        all_dates = set()
        for stock_code, stock_data in self.data.items():
            dates = stock_data.index
            all_dates.update(dates)
            
        # 筛选在回测时间范围内的日期
        dates = sorted([date for date in all_dates if self.start_date <= date <= self.end_date])
        
        return dates
    
    def run(self):
        """运行回测"""
        # 初始化策略
        self.strategy.initialize()
        
        # 逐日进行回测
        logger.info(f"开始回测，日期范围: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}")
        
        for date in self.dates:
            # 获取当日数据
            daily_data = {}
            for stock_code, stock_data in self.data.items():
                if date in stock_data.index:
                    daily_data[stock_code] = stock_data
                    
            # 调用策略处理当日数据
            self.strategy.on_data(date, daily_data)
            
        # 计算回测指标
        self._calculate_performance()
        
        return self.get_performance()
    
    def _calculate_performance(self):
        """计算回测性能指标"""
        # 简化计算，只计算总收益率
        if self.positions_history and len(self.positions_history) > 0:
            initial_value = self.initial_capital
            final_value = self.positions_history[-1]['total_value']
            self.total_return = (final_value - initial_value) / initial_value
        
    def get_performance(self):
        """获取回测性能指标"""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio
        }
    
    def save_results(self, results_dir):
        """保存回测结果"""
        # 创建结果目录
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存性能指标
        performance = self.get_performance()
        pd.Series(performance).to_csv(os.path.join(results_dir, 'performance.csv'))
        
        logger.info(f"回测结果已保存至目录: {results_dir}")
