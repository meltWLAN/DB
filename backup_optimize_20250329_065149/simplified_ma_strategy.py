#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版均线交叉策略
用于演示和测试基本功能
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 确保目录存在
DATA_DIR = "./data"
RESULTS_DIR = "./results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "charts"), exist_ok=True)

def generate_mock_data(n_days=250):
    """生成模拟股票数据"""
    # 创建日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days * 1.5)  # 考虑周末和节假日
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    date_range = date_range[-n_days:]  # 确保只有n_days个数据
    
    # 生成模拟价格
    np.random.seed(42)  # 固定随机种子，确保结果可重现
    
    # 创建初始价格和日涨跌幅
    base_price = 50.0
    daily_returns = np.random.normal(0.0005, 0.018, n_days)  # 均值略大于0，标准差约1.8%
    
    # 添加一些趋势和周期
    trend = np.linspace(0, 0.3, n_days) + np.sin(np.linspace(0, 10, n_days)) * 0.1
    daily_returns = daily_returns + trend
    
    # 计算价格序列
    log_returns = np.log(1 + daily_returns)
    log_price = np.cumsum(log_returns) + np.log(base_price)
    price = np.exp(log_price)
    
    # 基于收盘价生成其他价格
    close = price
    high = close * np.random.uniform(1.01, 1.04, n_days)
    low = close * np.random.uniform(0.96, 0.99, n_days)
    
    # 开盘价在前一天收盘价附近波动
    open_price = np.zeros_like(close)
    open_price[0] = close[0] * np.random.uniform(0.98, 1.02)
    for i in range(1, n_days):
        open_price[i] = close[i-1] * np.random.uniform(0.99, 1.01)
    
    # 确保OHLC关系正确
    for i in range(n_days):
        high[i] = max(high[i], open_price[i], close[i])
        low[i] = min(low[i], open_price[i], close[i])
    
    # 创建成交量
    volume = np.random.normal(1e6, 3e5, n_days) * (1 + np.abs(daily_returns) * 10)
    volume = np.abs(volume)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    df.set_index('date', inplace=True)
    return df

def calculate_signals(df, short_ma=5, long_ma=20):
    """计算均线交叉信号"""
    # 复制数据
    data = df.copy()
    
    # 计算移动平均线
    data[f'MA{short_ma}'] = data['close'].rolling(window=short_ma).mean()
    data[f'MA{long_ma}'] = data['close'].rolling(window=long_ma).mean()
    
    # 计算金叉和死叉信号
    data['short_above_long'] = (data[f'MA{short_ma}'] > data[f'MA{long_ma}']).astype(int)
    data['signal'] = data['short_above_long'].diff()
    
    # 标记金叉(1)和死叉(-1)
    data['signal'] = data['signal'].map({1.0: 1, -1.0: -1, 0.0: 0})
    
    # 生成持仓状态: 1表示持有，0表示空仓
    data['position'] = 0
    
    # 根据信号调整持仓
    position = 0
    for i in range(len(data)):
        if data['signal'].iloc[i] == 1:  # 买入信号
            position = 1
        elif data['signal'].iloc[i] == -1:  # 卖出信号
            position = 0
        data['position'].iloc[i] = position
    
    return data

def backtest_strategy(data, initial_capital=100000, stop_loss_pct=0.05):
    """回测均线交叉策略"""
    # 复制数据，避免修改原始数据
    df = data.copy()
    
    # 计算每日回报
    df['returns'] = df['close'].pct_change()
    df['returns'].fillna(0, inplace=True)
    
    # 初始化回测数据
    df['capital'] = initial_capital  # 总资产
    df['shares'] = 0  # 持有股数
    df['cash'] = initial_capital  # 现金
    df['stop_loss'] = False  # 止损标记
    
    # 模拟交易
    shares = 0
    cash = initial_capital
    cost_price = 0
    
    for i in range(1, len(df)):
        # 昨天的持仓信号
        prev_position = df['position'].iloc[i-1]
        
        # 今天的持仓信号
        curr_position = df['position'].iloc[i]
        
        # 当前价格
        curr_price = df['close'].iloc[i]
        
        # 检查是否需要止损
        stop_loss = False
        if shares > 0 and curr_price < cost_price * (1 - stop_loss_pct):
            stop_loss = True
            curr_position = 0  # 强制卖出
            df['stop_loss'].iloc[i] = True
        
        # 持仓变化
        if curr_position != prev_position or stop_loss:
            # 买入信号
            if curr_position > prev_position:
                # 计算可买入股数（整数股）
                new_shares = int(cash // curr_price)
                if new_shares > 0:
                    shares = new_shares
                    cash -= shares * curr_price
                    cost_price = curr_price
            # 卖出信号
            elif curr_position < prev_position or stop_loss:
                cash += shares * curr_price
                shares = 0
                cost_price = 0
        
        # 更新持仓数据
        df['shares'].iloc[i] = shares
        df['cash'].iloc[i] = cash
        
        # 计算总资产
        df['capital'].iloc[i] = cash + shares * curr_price
    
    # 计算策略收益
    df['strategy_returns'] = df['capital'] / initial_capital - 1
    
    # 计算买入持有收益
    df['buy_hold_returns'] = df['close'] / df['close'].iloc[0] - 1
    
    # 计算最大回撤
    df['peak'] = df['capital'].cummax()
    df['drawdown'] = (df['capital'] - df['peak']) / df['peak']
    max_drawdown = abs(df['drawdown'].min())
    
    # 计算年化收益率
    days = (df.index[-1] - df.index[0]).days
    annual_return = (df['capital'].iloc[-1] / initial_capital) ** (365 / max(days, 1)) - 1
    
    # 统计交易次数
    trades = df[df['position'] != df['position'].shift(1)].copy()
    trades = trades.iloc[1:]  # 移除第一个交易日的变化
    
    # 统计交易胜率
    trade_returns = []
    buy_price = None
    for i in range(len(trades)):
        row = trades.iloc[i]
        if row['position'] == 1:  # 买入
            buy_price = row['close']
        elif row['position'] == 0 and buy_price is not None:  # 卖出
            sell_price = row['close']
            trade_return = sell_price / buy_price - 1
            trade_returns.append(trade_return)
            buy_price = None
    
    # 如果最后一个信号是买入，添加到目前为止的收益
    if df['position'].iloc[-1] == 1 and buy_price is not None:
        sell_price = df['close'].iloc[-1]
        trade_return = sell_price / buy_price - 1
        trade_returns.append(trade_return)
    
    # 计算胜率
    win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0
    
    # 返回回测结果
    results = {
        'backtest_data': df,
        'initial_capital': initial_capital,
        'final_capital': df['capital'].iloc[-1],
        'total_return': df['strategy_returns'].iloc[-1],
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'trade_count': len(trades) // 2,  # 一买一卖算一次完整交易
        'win_rate': win_rate,
        'buy_hold_return': df['buy_hold_returns'].iloc[-1]
    }
    
    return results

def plot_results(data, results, short_ma=5, long_ma=20, save_path=None):
    """绘制回测结果图表"""
    # 创建图表
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'均线交叉策略回测 (MA{short_ma}/MA{long_ma})', fontsize=16)
    
    # 子图布局
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
    
    # 第一个子图：价格和信号
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('价格与信号')
    
    # 绘制收盘价和移动平均线
    data['close'].plot(ax=ax1, color='black', lw=1, label='收盘价')
    data[f'MA{short_ma}'].plot(ax=ax1, color='blue', lw=1, label=f'MA{short_ma}')
    data[f'MA{long_ma}'].plot(ax=ax1, color='red', lw=1, label=f'MA{long_ma}')
    
    # 标记买入点
    buy_signals = data[data['signal'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='买入信号')
    
    # 标记卖出点
    sell_signals = data[data['signal'] == -1]
    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='卖出信号')
    
    # 标记止损点
    stop_loss = data[data['stop_loss'] == True]
    if not stop_loss.empty:
        ax1.scatter(stop_loss.index, stop_loss['close'], marker='x', color='black', s=100, label='止损')
    
    ax1.legend()
    ax1.grid(True)
    
    # 第二个子图：持仓状态
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.set_title('持仓状态')
    ax2.fill_between(data.index, 0, data['position'], color='skyblue')
    ax2.set_ylabel('持仓')
    ax2.set_yticks([0, 1])
    ax2.grid(True)
    
    # 第三个子图：资金曲线
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.set_title('资金曲线')
    ax3.plot(data.index, data['capital'], color='green', label='资金')
    ax3.set_ylabel('资金')
    ax3.legend()
    ax3.grid(True)
    
    # 第四个子图：策略收益vs买入持有
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax4.set_title('收益比较')
    data['strategy_returns'].plot(ax=ax4, color='blue', label='策略收益')
    data['buy_hold_returns'].plot(ax=ax4, color='orange', label='买入持有')
    ax4.set_ylabel('收益率')
    ax4.legend()
    ax4.grid(True)
    
    # 添加回测结果信息
    backtest_info = (
        f"初始资金: ¥{results['initial_capital']:,.2f}\n"
        f"最终资金: ¥{results['final_capital']:,.2f}\n"
        f"总收益率: {results['total_return']:.2%}\n"
        f"年化收益: {results['annual_return']:.2%}\n"
        f"最大回撤: {results['max_drawdown']:.2%}\n"
        f"交易次数: {results['trade_count']}\n"
        f"胜率: {results['win_rate']:.2%}\n"
        f"买入持有收益: {results['buy_hold_return']:.2%}"
    )
    
    # 在图表上添加文本框
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.05, backtest_info, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path)
        plt.close()
        logger.info(f"图表已保存到: {save_path}")
    else:
        plt.show()

def main():
    """主函数"""
    # 生成模拟数据
    logger.info("生成模拟股票数据...")
    stock_data = generate_mock_data(250)
    
    # 添加均线和信号
    logger.info("计算交易信号...")
    short_ma, long_ma = 5, 20
    signals_data = calculate_signals(stock_data, short_ma, long_ma)
    
    # 回测策略
    logger.info("执行策略回测...")
    results = backtest_strategy(signals_data)
    
    # 打印回测结果
    logger.info("回测完成！结果如下:")
    logger.info(f"初始资金: ¥{results['initial_capital']:,.2f}")
    logger.info(f"最终资金: ¥{results['final_capital']:,.2f}")
    logger.info(f"总收益率: {results['total_return']:.2%}")
    logger.info(f"年化收益: {results['annual_return']:.2%}")
    logger.info(f"最大回撤: {results['max_drawdown']:.2%}")
    logger.info(f"交易次数: {results['trade_count']}")
    logger.info(f"胜率: {results['win_rate']:.2%}")
    logger.info(f"买入持有收益: {results['buy_hold_return']:.2%}")
    
    # 绘制结果图表
    logger.info("绘制回测结果图表...")
    chart_path = os.path.join(RESULTS_DIR, "charts", "ma_strategy_backtest.png")
    plot_results(results['backtest_data'], results, short_ma, long_ma, chart_path)

if __name__ == "__main__":
    main() 