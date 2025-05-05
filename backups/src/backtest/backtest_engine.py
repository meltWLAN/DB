#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测引擎模块，提供对分析策略的回测功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
import os
import json
from ..strategies.analysis_strategies import TrendFollowingStrategy, ReversalStrategy, VolatilityBreakoutStrategy, MultiStrategyAnalyzer


class BacktestEngine:
    """回测引擎类"""
    
    def __init__(self, data, initial_capital=100000, commission_rate=0.0003, slippage=0.001):
        """初始化回测引擎
        
        Args:
            data: DataFrame，包含股票数据
            initial_capital: 初始资金，默认10万
            commission_rate: 交易手续费率，默认0.03%
            slippage: 滑点，默认0.1%
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.logger = logging.getLogger(__name__)
        
        # 初始化结果记录
        self.trades = []
        self.portfolio_values = []
        self.metrics = {}
    
    def run_backtest(self, strategy, start_date=None, end_date=None, lookback_period=30):
        """运行回测
        
        Args:
            strategy: 分析策略对象
            start_date: 回测开始日期，默认为数据的第一天
            end_date: 回测结束日期，默认为数据的最后一天
            lookback_period: 策略分析所需的历史数据天数
            
        Returns:
            dict: 回测结果
        """
        # 设置回测日期范围
        if start_date is None:
            start_date = self.data['date'].min()
        if end_date is None:
            end_date = self.data['date'].max()
        
        # 过滤数据
        backtest_data = self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date)].copy()
        
        # 确保数据按日期排序
        backtest_data = backtest_data.sort_values('date').reset_index(drop=True)
        
        if len(backtest_data) == 0:
            self.logger.error(f"回测数据为空，请检查日期范围 ({start_date} - {end_date})")
            return None
        
        # 初始化账户状态
        capital = self.initial_capital
        position = 0  # 持仓数量
        entry_price = 0  # 入场价格
        in_market = False  # 是否在市场中
        
        # 记录每日投资组合价值
        portfolio_values = []
        trade_history = []
        
        # 分段处理数据以进行回测
        for i in range(lookback_period, len(backtest_data)):
            current_date = backtest_data.iloc[i]['date']
            current_price = backtest_data.iloc[i]['close']
            
            # 使用过去N天数据进行分析
            analysis_data = backtest_data.iloc[i-lookback_period:i+1]
            
            # 运行策略分析
            try:
                result = strategy.analyze(analysis_data)
                score = result.get('score', 0)  # 对于单一策略
                
                if isinstance(strategy, MultiStrategyAnalyzer):
                    score = result.get('combined_score', 0)  # 对于多策略分析器
                
                # 交易信号逻辑
                # 入场信号：得分大于70，且当前未持仓
                if score > 70 and not in_market:
                    # 考虑滑点后的买入价
                    buy_price = current_price * (1 + self.slippage)
                    # 计算可买入的股数（取整）
                    shares_to_buy = int(capital / buy_price)
                    
                    if shares_to_buy > 0:
                        # 计算手续费
                        commission = buy_price * shares_to_buy * self.commission_rate
                        
                        # 更新状态
                        capital -= (buy_price * shares_to_buy + commission)
                        position = shares_to_buy
                        entry_price = buy_price
                        in_market = True
                        
                        # 记录交易
                        trade = {
                            'date': current_date,
                            'action': 'BUY',
                            'price': buy_price,
                            'shares': shares_to_buy,
                            'commission': commission,
                            'value': buy_price * shares_to_buy,
                            'score': score
                        }
                        trade_history.append(trade)
                        self.logger.info(f"买入信号：日期={current_date}，价格={buy_price:.2f}，数量={shares_to_buy}，分数={score}")
                
                # 出场信号：得分小于40，且当前持仓
                elif (score < 40 or i == len(backtest_data) - 1) and in_market:
                    # 考虑滑点后的卖出价
                    sell_price = current_price * (1 - self.slippage)
                    
                    # 计算手续费
                    commission = sell_price * position * self.commission_rate
                    
                    # 更新状态
                    capital += (sell_price * position - commission)
                    
                    # 记录交易
                    trade = {
                        'date': current_date,
                        'action': 'SELL',
                        'price': sell_price,
                        'shares': position,
                        'commission': commission,
                        'value': sell_price * position,
                        'score': score,
                        'profit': (sell_price - entry_price) * position - commission,
                        'profit_percent': ((sell_price / entry_price) - 1) * 100
                    }
                    trade_history.append(trade)
                    self.logger.info(f"卖出信号：日期={current_date}，价格={sell_price:.2f}，数量={position}，分数={score}，"
                                   f"利润={trade['profit']:.2f}，利润率={trade['profit_percent']:.2f}%")
                    
                    # 重置持仓状态
                    position = 0
                    entry_price = 0
                    in_market = False
            
            except Exception as e:
                self.logger.error(f"策略分析失败，日期={current_date}: {str(e)}")
            
            # 计算当前投资组合价值
            portfolio_value = capital
            if in_market:
                portfolio_value += position * current_price
            
            # 记录每日投资组合价值
            portfolio_values.append({
                'date': current_date,
                'price': current_price,
                'capital': capital,
                'position': position,
                'portfolio_value': portfolio_value,
                'in_market': in_market
            })
        
        # 保存结果
        self.trades = trade_history
        self.portfolio_values = portfolio_values
        
        # 计算回测指标
        self.calculate_metrics()
        
        # 构建回测结果
        backtest_result = {
            'trades': self.trades,
            'portfolio_values': self.portfolio_values,
            'metrics': self.metrics,
            'strategy_name': strategy.name if hasattr(strategy, 'name') else 'MultiStrategy',
            'start_date': str(start_date),
            'end_date': str(end_date),
            'initial_capital': self.initial_capital
        }
        
        return backtest_result
    
    def calculate_metrics(self):
        """计算回测指标"""
        if not self.portfolio_values:
            self.logger.warning("没有回测数据，无法计算指标")
            return
        
        # 创建投资组合价值DataFrame
        df = pd.DataFrame(self.portfolio_values)
        
        # 计算每日收益率
        df['daily_return'] = df['portfolio_value'].pct_change()
        
        # 计算累计收益率
        final_value = df.iloc[-1]['portfolio_value']
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # 计算年化收益率
        days = (df.iloc[-1]['date'] - df.iloc[0]['date']).days
        if days > 0:
            annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
        else:
            annualized_return = 0
        
        # 计算最大回撤
        df['cummax'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['cummax'] - df['portfolio_value']) / df['cummax'] * 100
        max_drawdown = df['drawdown'].max()
        
        # 计算夏普比率
        risk_free_rate = 0.03  # 假设年化无风险利率为3%
        daily_risk_free = (1 + risk_free_rate) ** (1/365) - 1
        
        excess_return = df['daily_return'] - daily_risk_free
        if len(excess_return) > 1 and excess_return.std() > 0:
            sharpe_ratio = (excess_return.mean() / excess_return.std()) * (252 ** 0.5)  # 252个交易日/年
        else:
            sharpe_ratio = 0
        
        # 计算交易次数和胜率
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            total_trades = len(trades_df[trades_df['action'] == 'SELL'])
            if total_trades > 0:
                winning_trades = len(trades_df[(trades_df['action'] == 'SELL') & (trades_df['profit'] > 0)])
                win_rate = (winning_trades / total_trades) * 100
                
                # 计算平均利润和平均亏损
                avg_profit = trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
                avg_loss = trades_df[trades_df['profit'] < 0]['profit'].mean() if (total_trades - winning_trades) > 0 else 0
                
                # 计算利润因子
                profit_factor = abs(trades_df[trades_df['profit'] > 0]['profit'].sum() / 
                                  trades_df[trades_df['profit'] < 0]['profit'].sum()) if trades_df[trades_df['profit'] < 0]['profit'].sum() != 0 else float('inf')
            else:
                win_rate = 0
                avg_profit = 0
                avg_loss = 0
                profit_factor = 0
        else:
            total_trades = 0
            win_rate = 0
            avg_profit = 0
            avg_loss = 0
            profit_factor = 0
        
        # 保存指标
        self.metrics = {
            'total_return': round(total_return, 2),
            'annualized_return': round(annualized_return, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'total_trades': total_trades,
            'win_rate': round(win_rate, 2),
            'avg_profit': round(avg_profit, 2) if avg_profit != 0 else 0,
            'avg_loss': round(avg_loss, 2) if avg_loss != 0 else 0,
            'profit_factor': round(profit_factor, 2),
            'final_capital': round(final_value, 2)
        }
    
    def plot_results(self, title=None, save_path=None):
        """绘制回测结果图表
        
        Args:
            title: 图表标题
            save_path: 保存路径，如果提供则保存图表
            
        Returns:
            None
        """
        if not self.portfolio_values:
            self.logger.warning("没有回测数据，无法绘制图表")
            return
        
        # 创建投资组合价值DataFrame
        df = pd.DataFrame(self.portfolio_values)
        
        # 转换日期为datetime格式
        df['date'] = pd.to_datetime(df['date'])
        
        # 创建一个带有两个y轴的图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # 设置图表标题
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            strategy_name = self.metrics.get('strategy_name', '未知策略')
            fig.suptitle(f"{strategy_name} 回测结果", fontsize=16)
        
        # 绘制价格线
        ax1.plot(df['date'], df['price'], label='价格', color='gray', alpha=0.6)
        
        # 绘制投资组合价值线
        port_line, = ax1.plot(df['date'], df['portfolio_value'], label='投资组合价值', color='blue', linewidth=2)
        
        # 绘制买入点和卖出点
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        
        if buy_trades:
            buy_dates = [pd.to_datetime(t['date']) for t in buy_trades]
            buy_prices = [t['price'] for t in buy_trades]
            ax1.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='买入点')
        
        if sell_trades:
            sell_dates = [pd.to_datetime(t['date']) for t in sell_trades]
            sell_prices = [t['price'] for t in sell_trades]
            ax1.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='卖出点')
        
        # 设置x轴格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        
        # 添加网格线
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 添加图例
        ax1.legend(loc='upper left')
        
        # 设置标签
        ax1.set_ylabel('价格 / 投资组合价值')
        ax1.set_title('价格走势和投资组合价值')
        
        # 绘制回撤图表
        df['cummax'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['cummax'] - df['portfolio_value']) / df['cummax'] * 100
        
        ax2.fill_between(df['date'], df['drawdown'], 0, color='coral', alpha=0.3)
        ax2.plot(df['date'], df['drawdown'], color='red', label='回撤%')
        
        # 设置y轴反向
        ax2.invert_yaxis()
        
        # 设置x轴格式
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        
        # 添加网格线
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # 设置标签
        ax2.set_xlabel('日期')
        ax2.set_ylabel('回撤 (%)')
        ax2.set_title('最大回撤')
        
        # 旋转x轴标签以避免重叠
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 添加回测指标文本框
        textstr = '\n'.join((
            f"总收益率: {self.metrics['total_return']:.2f}%",
            f"年化收益率: {self.metrics['annualized_return']:.2f}%",
            f"最大回撤: {self.metrics['max_drawdown']:.2f}%",
            f"夏普比率: {self.metrics['sharpe_ratio']:.2f}",
            f"交易次数: {self.metrics['total_trades']}",
            f"胜率: {self.metrics['win_rate']:.2f}%",
            f"利润因子: {self.metrics['profit_factor']:.2f}"
        ))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"回测结果图表已保存至 {save_path}")
        
        # 显示图表
        plt.show()
    
    def save_results(self, filepath):
        """保存回测结果到JSON文件
        
        Args:
            filepath: 保存路径
            
        Returns:
            bool: 是否保存成功
        """
        if not self.portfolio_values:
            self.logger.warning("没有回测数据，无法保存结果")
            return False
        
        # 准备结果数据
        results = {
            'trades': self.trades,
            'portfolio_values': [
                {k: str(v) if isinstance(v, (datetime, pd.Timestamp)) else v 
                 for k, v in val.items()}
                for val in self.portfolio_values
            ],
            'metrics': self.metrics,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存到JSON文件
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"回测结果已保存至 {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"保存回测结果失败: {str(e)}")
            return False


def run_backtest(stock_data, strategy_type='trend', start_date=None, end_date=None, initial_capital=100000, 
                 plot_results=True, save_plot=None, save_results=None):
    """运行回测的便捷函数
    
    Args:
        stock_data: DataFrame，包含股票数据
        strategy_type: 策略类型，可选 'trend', 'reversal', 'volatility', 'multi'
        start_date: 回测开始日期
        end_date: 回测结束日期
        initial_capital: 初始资金
        plot_results: 是否绘制结果图表
        save_plot: 图表保存路径
        save_results: 结果保存路径
        
    Returns:
        dict: 回测结果
    """
    # 初始化回测引擎
    engine = BacktestEngine(stock_data, initial_capital=initial_capital)
    
    # 选择策略
    if strategy_type == 'trend':
        strategy = TrendFollowingStrategy()
        strategy_name = "趋势跟踪策略"
    elif strategy_type == 'reversal':
        strategy = ReversalStrategy()
        strategy_name = "反转策略"
    elif strategy_type == 'volatility':
        strategy = VolatilityBreakoutStrategy()
        strategy_name = "波动率突破策略"
    elif strategy_type == 'multi':
        strategy = MultiStrategyAnalyzer()
        strategy_name = "多策略组合"
    else:
        raise ValueError(f"不支持的策略类型: {strategy_type}")
    
    # 运行回测
    result = engine.run_backtest(strategy, start_date=start_date, end_date=end_date)
    
    if result is None:
        return None
    
    # 绘制结果
    if plot_results:
        engine.plot_results(title=f"{strategy_name} 回测结果", save_path=save_plot)
    
    # 保存结果
    if save_results:
        engine.save_results(save_results)
    
    return result 