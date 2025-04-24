#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
均线交叉趋势跟踪策略
使用短期均线(如5日线)与长期均线(如20日线)的交叉作为买入卖出信号
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 确保src包可以被导入
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 导入配置
from src.enhanced.config.settings import LOG_DIR, DATA_DIR, RESULTS_DIR

# 创建必要目录
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "ma_strategy"), exist_ok=True)

# 配置日志
log_file = os.path.join(LOG_DIR, f"ma_crossover_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 策略参数
class StrategyParams:
    # 均线参数
    SHORT_MA = 5  # 短期均线
    LONG_MA = 20  # 长期均线
    
    # 交易参数
    STOP_LOSS_PCT = 0.05  # 止损比例
    TAKE_PROFIT_PCT = 0.15  # 止盈比例
    
    # 回测参数
    INITIAL_CAPITAL = 100000  # 初始资金
    POSITION_SIZE_PCT = 0.2  # 单次仓位比例
    MAX_POSITIONS = 5  # 最大同时持仓数量

def calculate_ma_signals(data):
    """
    计算均线交叉信号
    
    Args:
        data: 包含价格数据的DataFrame
        
    Returns:
        DataFrame: 添加了均线和信号的数据
    """
    # 确保数据按日期排序
    data = data.sort_values('date')
    
    # 计算短期和长期均线
    data[f'ma_{StrategyParams.SHORT_MA}'] = data['close'].rolling(window=StrategyParams.SHORT_MA).mean()
    data[f'ma_{StrategyParams.LONG_MA}'] = data['close'].rolling(window=StrategyParams.LONG_MA).mean()
    
    # 计算前一天的均线
    data[f'prev_ma_{StrategyParams.SHORT_MA}'] = data[f'ma_{StrategyParams.SHORT_MA}'].shift(1)
    data[f'prev_ma_{StrategyParams.LONG_MA}'] = data[f'ma_{StrategyParams.LONG_MA}'].shift(1)
    
    # 计算金叉(买入信号)和死叉(卖出信号)
    data['golden_cross'] = (data[f'ma_{StrategyParams.SHORT_MA}'] > data[f'ma_{StrategyParams.LONG_MA}']) & \
                           (data[f'prev_ma_{StrategyParams.SHORT_MA}'] <= data[f'prev_ma_{StrategyParams.LONG_MA}'])
    
    data['death_cross'] = (data[f'ma_{StrategyParams.SHORT_MA}'] < data[f'ma_{StrategyParams.LONG_MA}']) & \
                          (data[f'prev_ma_{StrategyParams.SHORT_MA}'] >= data[f'prev_ma_{StrategyParams.LONG_MA}'])
    
    # 计算买入卖出信号
    data['signal'] = 0  # 0: 无信号, 1: 买入, -1: 卖出
    data.loc[data['golden_cross'], 'signal'] = 1
    data.loc[data['death_cross'], 'signal'] = -1
    
    # 计算百分比变化
    data['pct_change'] = data['close'].pct_change()
    
    # 计算成交量变化
    data['volume_change'] = data['volume'].pct_change()
    data['volume_ma5'] = data['volume'].rolling(window=5).mean()
    data['volume_ma20'] = data['volume'].rolling(window=20).mean()
    
    return data

def backtest_strategy(data, initial_capital=StrategyParams.INITIAL_CAPITAL, 
                     position_size_pct=StrategyParams.POSITION_SIZE_PCT,
                     stop_loss_pct=StrategyParams.STOP_LOSS_PCT,
                     take_profit_pct=StrategyParams.TAKE_PROFIT_PCT):
    """
    回测均线交叉策略
    
    Args:
        data: 包含价格和信号的DataFrame
        initial_capital: 初始资金
        position_size_pct: 单次仓位比例
        stop_loss_pct: 止损比例
        take_profit_pct: 止盈比例
        
    Returns:
        dict: 回测结果
    """
    # 确保数据按日期排序
    data = data.sort_values('date')
    
    # 创建回测结果DataFrame
    backtest_results = data.copy()
    
    # 初始化回测变量
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    
    # 记录每日资金变化
    backtest_results['capital'] = initial_capital
    
    # 进行回测
    for i, row in backtest_results.iterrows():
        if i == 0:
            continue
            
        # 如果没有持仓，检查买入信号
        if position == 0 and row['signal'] == 1:
            # 计算买入数量
            position_size = capital * position_size_pct
            position = position_size / row['close']
            entry_price = row['close']
            capital -= position_size
            
            # 记录交易
            trades.append({
                'date': row['date'],
                'type': 'buy',
                'price': row['close'],
                'position': position,
                'capital': capital
            })
            
        # 如果持有仓位，检查卖出信号或止损/止盈
        elif position > 0:
            # 检查是否触发止损
            if row['close'] < entry_price * (1 - stop_loss_pct):
                # 止损卖出
                capital += position * row['close']
                
                # 记录交易
                trades.append({
                    'date': row['date'],
                    'type': 'stop_loss',
                    'price': row['close'],
                    'position': position,
                    'capital': capital
                })
                
                position = 0
                entry_price = 0
                
            # 检查是否触发止盈
            elif row['close'] > entry_price * (1 + take_profit_pct):
                # 止盈卖出
                capital += position * row['close']
                
                # 记录交易
                trades.append({
                    'date': row['date'],
                    'type': 'take_profit',
                    'price': row['close'],
                    'position': position,
                    'capital': capital
                })
                
                position = 0
                entry_price = 0
                
            # 检查是否有死叉卖出信号
            elif row['signal'] == -1:
                # 卖出信号卖出
                capital += position * row['close']
                
                # 记录交易
                trades.append({
                    'date': row['date'],
                    'type': 'sell',
                    'price': row['close'],
                    'position': position,
                    'capital': capital
                })
                
                position = 0
                entry_price = 0
        
        # 计算当前总资产
        portfolio_value = capital + (position * row['close'] if position > 0 else 0)
        backtest_results.at[i, 'capital'] = portfolio_value
    
    # 将最后的持仓卖出，计算最终资产
    if position > 0:
        final_price = backtest_results.iloc[-1]['close']
        capital += position * final_price
        
        # 记录最后一笔交易
        trades.append({
            'date': backtest_results.iloc[-1]['date'],
            'type': 'final_sell',
            'price': final_price,
            'position': position,
            'capital': capital
        })
    
    # 计算回测指标
    initial_value = initial_capital
    final_value = capital
    total_return = (final_value / initial_value - 1) * 100
    
    # 计算年化收益率
    days = (pd.to_datetime(backtest_results.iloc[-1]['date']) - pd.to_datetime(backtest_results.iloc[0]['date'])).days
    annual_return = ((1 + total_return / 100) ** (365 / max(days, 1)) - 1) * 100
    
    # 计算最大回撤
    backtest_results['portfolio_value'] = backtest_results['capital']
    backtest_results['cum_max'] = backtest_results['portfolio_value'].cummax()
    backtest_results['drawdown'] = (backtest_results['cum_max'] - backtest_results['portfolio_value']) / backtest_results['cum_max'] * 100
    max_drawdown = backtest_results['drawdown'].max()
    
    # 计算交易次数和胜率
    profit_trades = [t for t in trades if t['type'] in ('sell', 'take_profit', 'final_sell') and 
                    t['price'] * t['position'] > t['capital'] * position_size_pct]
    
    win_rate = len(profit_trades) / max(len(trades) / 2, 1) * 100 if trades else 0
    
    # 汇总结果
    results = {
        'initial_capital': initial_capital,
        'final_capital': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trade_count': len(trades) // 2,  # 买入+卖出算一次交易
        'trades': trades,
        'backtest_results': backtest_results
    }
    
    return results

def plot_ma_strategy(data, results, stock_code, stock_name, output_dir=None):
    """
    绘制均线交叉策略分析图
    
    Args:
        data: 股票数据DataFrame
        results: 回测结果字典
        stock_code: 股票代码
        stock_name: 股票名称
        output_dir: 输出目录
        
    Returns:
        str: 图表文件路径
    """
    # 创建子图布局
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # 格式化日期轴以避免中文显示问题
    dates = pd.to_datetime(data['date'])
    
    # 绘制价格和均线
    ax1.plot(dates, data['close'], 'b-', label='收盘价')
    ax1.plot(dates, data[f'ma_{StrategyParams.SHORT_MA}'], 'r-', 
             label=f'{StrategyParams.SHORT_MA}日均线')
    ax1.plot(dates, data[f'ma_{StrategyParams.LONG_MA}'], 'g-', 
             label=f'{StrategyParams.LONG_MA}日均线')
    
    # 标记买入点和卖出点
    buy_signals = data[data['signal'] == 1]
    sell_signals = data[data['signal'] == -1]
    
    ax1.scatter(pd.to_datetime(buy_signals['date']), buy_signals['close'], 
               color='red', marker='^', s=100, label='买入信号')
    ax1.scatter(pd.to_datetime(sell_signals['date']), sell_signals['close'], 
               color='green', marker='v', s=100, label='卖出信号')
    
    # 绘制实际交易点
    if 'trades' in results:
        for trade in results['trades']:
            if trade['type'] == 'buy':
                ax1.scatter(pd.to_datetime(trade['date']), trade['price'], 
                          color='purple', marker='*', s=150, label='_nolegend_')
            elif trade['type'] in ('sell', 'stop_loss', 'take_profit', 'final_sell'):
                ax1.scatter(pd.to_datetime(trade['date']), trade['price'], 
                          color='black', marker='x', s=150, label='_nolegend_')
    
    # 设置标题和标签
    title = f"{stock_name}({stock_code}) - 均线交叉策略回测"
    title += f"\n总收益: {results['total_return']:.2f}%, 年化: {results['annual_return']:.2f}%, 最大回撤: {results['max_drawdown']:.2f}%"
    ax1.set_title(title)
    ax1.set_ylabel('价格')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # 绘制成交量
    volume_colors = ['red' if x >= 0 else 'green' for x in data['pct_change']]
    ax2.bar(dates, data['volume'], color=volume_colors)
    ax2.plot(dates, data['volume_ma5'], 'r-', label='成交量5日均线')
    ax2.plot(dates, data['volume_ma20'], 'b-', label='成交量20日均线')
    ax2.set_ylabel('成交量')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # 绘制资金曲线
    if 'backtest_results' in results:
        backtest_data = results['backtest_results']
        ax3.plot(pd.to_datetime(backtest_data['date']), backtest_data['capital'], 'b-', label='资金曲线')
        ax3.set_ylabel('资金')
        ax3.set_xlabel('日期')
        ax3.grid(True)
        ax3.legend(loc='upper left')
        
        # 计算最大回撤区域
        highwater = backtest_data['capital'].cummax()
        drawdown = (highwater - backtest_data['capital']) / highwater
        
        # 找出最大回撤的开始和结束点
        max_dd_end_idx = drawdown.idxmax()
        temp = backtest_data.loc[:max_dd_end_idx]
        max_dd_start_idx = temp['capital'].idxmax()
        
        # 绘制最大回撤区域
        if max_dd_start_idx < max_dd_end_idx:
            ax3.axvspan(pd.to_datetime(backtest_data.loc[max_dd_start_idx, 'date']), 
                      pd.to_datetime(backtest_data.loc[max_dd_end_idx, 'date']), 
                      color='red', alpha=0.2)
    
    # 格式化x轴日期
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        chart_file = os.path.join(output_dir, f"{stock_code}_ma_strategy.png")
        plt.savefig(chart_file)
        plt.close()
        return chart_file
    else:
        plt.show()
        plt.close()
        return None

def analyze_stock_with_ma_strategy(tushare_fetcher, stock_code, start_date, end_date):
    """
    使用均线交叉策略分析单只股票
    
    Args:
        tushare_fetcher: TuShare数据获取器
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        dict: 分析结果
    """
    try:
        # 获取股票数据
        stock_data = tushare_fetcher.get_daily_data(stock_code, start_date, end_date)
        if stock_data is None or stock_data.empty:
            logger.warning(f"无法获取股票 {stock_code} 的数据")
            return None
        
        # 获取股票名称
        stock_list = tushare_fetcher.get_stock_list()
        stock_name = ""
        industry = ""
        if stock_list is not None and not stock_list.empty:
            stock_info = stock_list[stock_list['code'] == stock_code]
            if not stock_info.empty:
                stock_name = stock_info.iloc[0]['name']
                if 'industry' in stock_info.columns:
                    industry = stock_info.iloc[0]['industry']
        
        # 计算策略信号
        with_signals = calculate_ma_signals(stock_data)
        
        # 回测策略
        backtest_results = backtest_strategy(with_signals)
        
        # 计算当前信号
        latest_data = with_signals.iloc[-1]
        current_signal = latest_data['signal']
        
        # 获取最近的交易信号
        recent_signal = "无"
        for i in range(len(with_signals)-1, -1, -1):
            if with_signals.iloc[i]['signal'] != 0:
                if with_signals.iloc[i]['signal'] == 1:
                    recent_signal = "买入"
                else:
                    recent_signal = "卖出"
                break
        
        # 构建分析结果
        analysis = {
            'code': stock_code,
            'name': stock_name,
            'industry': industry,
            'latest_date': latest_data['date'],
            'close': latest_data['close'],
            'current_signal': "买入" if current_signal == 1 else "卖出" if current_signal == -1 else "持有",
            'recent_signal': recent_signal,
            'short_ma': latest_data[f'ma_{StrategyParams.SHORT_MA}'],
            'long_ma': latest_data[f'ma_{StrategyParams.LONG_MA}'],
            'ma_diff_pct': (latest_data[f'ma_{StrategyParams.SHORT_MA}'] / latest_data[f'ma_{StrategyParams.LONG_MA}'] - 1) * 100,
            'backtest_return': backtest_results['total_return'],
            'annual_return': backtest_results['annual_return'],
            'max_drawdown': backtest_results['max_drawdown'],
            'win_rate': backtest_results['win_rate'],
            'trade_count': backtest_results['trade_count'],
            'data': with_signals,
            'backtest_results': backtest_results
        }
        
        return analysis
    
    except Exception as e:
        logger.error(f"分析股票 {stock_code} 时出错: {str(e)}")
        return None

def main():
    """主函数"""
    try:
        logger.info("==== 均线交叉策略分析系统启动 ====")
        
        # 导入数据源
        from src.enhanced.data.fetchers.tushare_fetcher import EnhancedTushareFetcher
        
        # 初始化TuShare数据源
        logger.info("初始化TuShare数据源...")
        from src.enhanced.config.settings import DATA_SOURCE_CONFIG
        tushare_config = DATA_SOURCE_CONFIG.get('tushare', {})
        
        # 处理配置
        fetcher_config = {
            'token': tushare_config.get('token', ''),
            'rate_limit': tushare_config.get('rate_limit', {}).get('calls_per_minute', 500) / 60 if 'rate_limit' in tushare_config else 500 / 60,
            'connection_retries': tushare_config.get('retry', {}).get('max_retries', 3) if 'retry' in tushare_config else 3,
            'retry_delay': tushare_config.get('retry', {}).get('retry_interval', 5) if 'retry' in tushare_config else 5
        }
        
        # 初始化数据获取器
        tushare_fetcher = EnhancedTushareFetcher(fetcher_config)
        
        # 设置分析参数
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 获取一年数据
        
        # 获取股票列表
        logger.info("获取股票列表...")
        stock_list = tushare_fetcher.get_stock_list()
        if stock_list is None or stock_list.empty:
            logger.error("获取股票列表失败，无法继续分析")
            return
        
        # 筛选上证50指数成分股
        stock_codes = [
            "600519.SH",  # 贵州茅台
            "601318.SH",  # 中国平安
            "600036.SH",  # 招商银行
            "600276.SH",  # 恒瑞医药
            "601166.SH",  # 兴业银行
            "600887.SH",  # 伊利股份
            "601328.SH",  # 交通银行
            "601288.SH",  # 农业银行
            "600000.SH",  # 浦发银行
            "601398.SH",  # 工商银行
        ]
        
        # 分析结果列表
        strategy_results = []
        
        # 分析每只股票
        for stock_code in stock_codes:
            stock_info = stock_list[stock_list['code'] == stock_code]
            if stock_info.empty:
                logger.warning(f"找不到股票 {stock_code} 的信息，跳过")
                continue
                
            stock_name = stock_info.iloc[0]['name']
            
            logger.info(f"正在分析股票: {stock_name}({stock_code})...")
            result = analyze_stock_with_ma_strategy(tushare_fetcher, stock_code, start_date, end_date)
            
            if result:
                strategy_results.append(result)
                
                # 记录当前信号
                signal_str = result['current_signal']
                logger.info(f"股票 {stock_name}({stock_code}) 当前信号: {signal_str}")
                
                # 生成图表
                strategy_dir = os.path.join(RESULTS_DIR, "ma_strategy")
                chart_file = plot_ma_strategy(result['data'], result['backtest_results'], 
                                            stock_code, stock_name, strategy_dir)
                if chart_file:
                    logger.info(f"已生成图表: {chart_file}")
        
        # 按回测收益率排序结果
        strategy_results.sort(key=lambda x: x['backtest_return'], reverse=True)
        
        # 输出结果
        logger.info(f"分析完成，共分析 {len(strategy_results)} 只股票")
        if strategy_results:
            # 创建结果表格
            result_df = pd.DataFrame([
                {
                    '股票代码': s['code'],
                    '股票名称': s['name'],
                    '行业': s.get('industry', ''),
                    '收盘价': s['close'],
                    '当前信号': s['current_signal'],
                    '最近信号': s['recent_signal'],
                    f'{StrategyParams.SHORT_MA}日均线': f"{s['short_ma']:.2f}",
                    f'{StrategyParams.LONG_MA}日均线': f"{s['long_ma']:.2f}",
                    '均线差(%)': f"{s['ma_diff_pct']:.2f}%",
                    '回测收益(%)': f"{s['backtest_return']:.2f}%",
                    '年化收益(%)': f"{s['annual_return']:.2f}%",
                    '最大回撤(%)': f"{s['max_drawdown']:.2f}%",
                    '胜率(%)': f"{s['win_rate']:.2f}%",
                    '交易次数': s['trade_count']
                }
                for s in strategy_results
            ])
            
            # 根据当前信号分组
            buy_signals = result_df[result_df['当前信号'] == '买入']
            sell_signals = result_df[result_df['当前信号'] == '卖出']
            hold_signals = result_df[result_df['当前信号'] == '持有']
            
            # 保存结果
            result_file = os.path.join(RESULTS_DIR, f"ma_strategy_results_{datetime.now().strftime('%Y%m%d')}.csv")
            result_df.to_csv(result_file, index=False, encoding='utf-8-sig')
            logger.info(f"结果已保存至: {result_file}")
            
            # 显示买入信号的股票
            if not buy_signals.empty:
                print("\n== 买入信号股票 ==")
                print(buy_signals.to_string(index=False))
            
            # 显示卖出信号的股票
            if not sell_signals.empty:
                print("\n== 卖出信号股票 ==")
                print(sell_signals.to_string(index=False))
            
            # 显示所有结果
            print("\n== 全部分析结果 ==")
            print(result_df.to_string(index=False))
        
        logger.info("==== 均线交叉策略分析系统运行完成 ====")
        
    except Exception as e:
        logger.error(f"系统运行出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 