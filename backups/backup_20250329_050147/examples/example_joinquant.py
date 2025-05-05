#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用JoinQuant数据的分析示例脚本
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加当前目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 确保结果目录存在
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)

# 导入配置
from src.config import DATA_SOURCE_CONFIG

def get_jq_data(start_date, end_date, stock_codes):
    """从JoinQuant获取数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        stock_codes: 股票代码列表
        
    Returns:
        data: 股票数据字典 {股票代码: 数据DataFrame}
    """
    try:
        import jqdatasdk as jq
        
        # 获取JoinQuant配置
        jq_config = DATA_SOURCE_CONFIG.get('joinquant', {})
        username = jq_config.get('username', '')
        password = jq_config.get('password', '')
        
        if not username or not password:
            logger.error("JoinQuant账号未配置，请在src/config/__init__.py中设置")
            return {}
        
        # 登录
        logger.info("登录JoinQuant...")
        jq.auth(username, password)
        
        # 获取账号信息
        account_info = jq.get_account_info()
        logger.info(f"账号信息: {account_info}")
        
        # 保证日期类型是字符串
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
        # 获取交易日历
        trade_days = jq.get_trade_days(start_date, end_date)
        logger.info(f"获取到 {len(trade_days)} 个交易日")
        
        # 获取股票数据
        data = {}
        for stock_code in stock_codes:
            logger.info(f"获取 {stock_code} 数据...")
            
            # 获取价格数据
            df = jq.get_price(stock_code, start_date=start_date, end_date=end_date, 
                           frequency='daily', fields=['open', 'close', 'high', 'low', 'volume'])
            
            # 将日期从索引转为列
            df = df.reset_index()
            df.rename(columns={'index': 'date'}, inplace=True)
            
            # 计算技术指标
            # 移动平均线
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA10'] = df['close'].rolling(window=10).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            
            # MACD
            df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['DIF'] = df['EMA12'] - df['EMA26']
            df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
            df['MACD'] = 2 * (df['DIF'] - df['DEA'])
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            data[stock_code] = df
            logger.info(f"获取 {stock_code} 数据完成，包含 {len(df)} 条记录")
            
        # 登出
        jq.logout()
        logger.info("JoinQuant数据获取完成，已登出")
        
        return data
        
    except Exception as e:
        logger.error(f"获取JoinQuant数据失败: {str(e)}")
        return {}

def analyze_stock_data(data):
    """分析股票数据
    
    Args:
        data: 股票数据字典
        
    Returns:
        analysis_results: 分析结果
    """
    analysis_results = {}
    
    for stock_code, df in data.items():
        # 跳过数据不足的股票
        if len(df) < 20:
            logger.warning(f"{stock_code} 数据不足，跳过分析")
            continue
        
        # 计算收益率
        df['daily_return'] = df['close'].pct_change()
        
        # 计算累计收益率
        df['cumulative_return'] = (1 + df['daily_return']).cumprod()
        
        # 计算波动率（标准差）
        volatility = df['daily_return'].std() * np.sqrt(252)  # 年化波动率
        
        # 计算夏普比率（假设无风险收益率为0）
        annual_return = (df['cumulative_return'].iloc[-1] ** (252 / len(df))) - 1
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # 计算最大回撤
        df['cummax'] = df['cumulative_return'].cummax()
        df['drawdown'] = (df['cumulative_return'] / df['cummax']) - 1
        max_drawdown = df['drawdown'].min()
        
        # 计算均线金叉死叉次数
        df['golden_cross'] = (df['MA5'] > df['MA10']) & (df['MA5'].shift(1) <= df['MA10'].shift(1))
        df['death_cross'] = (df['MA5'] < df['MA10']) & (df['MA5'].shift(1) >= df['MA10'].shift(1))
        golden_cross_count = df['golden_cross'].sum()
        death_cross_count = df['death_cross'].sum()
        
        # 收集分析结果
        analysis_results[stock_code] = {
            'total_return': df['cumulative_return'].iloc[-1] - 1,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'golden_cross_count': golden_cross_count,
            'death_cross_count': death_cross_count,
            'avg_volume': df['volume'].mean(),
            'last_close': df['close'].iloc[-1],
            'last_date': df['date'].iloc[-1],
            'data_days': len(df)
        }
        
    return analysis_results

def plot_stock_charts(data, top_n=3, save_dir=None):
    """绘制股票图表
    
    Args:
        data: 股票数据字典
        top_n: 展示前N只表现最好的股票
        save_dir: 保存图表的目录
    """
    # 分析数据
    analysis_results = analyze_stock_data(data)
    
    # 按总收益率排序
    sorted_stocks = sorted(analysis_results.items(), key=lambda x: x[1]['total_return'], reverse=True)
    
    # 只展示前N只表现最好的股票
    top_stocks = sorted_stocks[:top_n]
    
    for stock_code, _ in top_stocks:
        df = data[stock_code]
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # 绘制K线图和均线
        ax1.plot(df['date'], df['close'], label='收盘价')
        ax1.plot(df['date'], df['MA5'], label='MA5')
        ax1.plot(df['date'], df['MA10'], label='MA10')
        ax1.plot(df['date'], df['MA20'], label='MA20')
        ax1.set_title(f'{stock_code} 价格走势')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制成交量
        ax2.bar(df['date'], df['volume'])
        ax2.set_title('成交量')
        ax2.grid(True)
        
        # 绘制MACD
        ax3.plot(df['date'], df['DIF'], label='DIF')
        ax3.plot(df['date'], df['DEA'], label='DEA')
        ax3.bar(df['date'], df['MACD'])
        ax3.set_title('MACD')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        if save_dir:
            save_path = os.path.join(save_dir, f"{stock_code.replace('.', '_')}_chart.png")
            plt.savefig(save_path)
            logger.info(f"保存图表: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    # 绘制收益率对比图
    plt.figure(figsize=(12, 6))
    
    for stock_code, _ in top_stocks:
        df = data[stock_code]
        plt.plot(df['date'], df['cumulative_return'], label=stock_code)
    
    plt.title('累计收益率对比')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    if save_dir:
        save_path = os.path.join(save_dir, "cumulative_returns.png")
        plt.savefig(save_path)
        logger.info(f"保存收益率对比图: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 创建分析结果表格
    results_df = pd.DataFrame(columns=[
        '股票代码', '总收益率', '年化收益率', '波动率', '夏普比率', '最大回撤', 
        '金叉次数', '死叉次数', '平均成交量', '最新收盘价', '数据天数'
    ])
    
    for i, (stock_code, result) in enumerate(sorted_stocks):
        results_df.loc[i] = [
            stock_code,
            f"{result['total_return']:.2%}",
            f"{result['annual_return']:.2%}",
            f"{result['volatility']:.2%}",
            f"{result['sharpe_ratio']:.2f}",
            f"{result['max_drawdown']:.2%}",
            result['golden_cross_count'],
            result['death_cross_count'],
            f"{result['avg_volume']:.0f}",
            f"{result['last_close']:.2f}",
            result['data_days']
        ]
    
    # 保存分析结果
    if save_dir:
        save_path = os.path.join(save_dir, "analysis_results.csv")
        results_df.to_csv(save_path, index=False)
        logger.info(f"保存分析结果: {save_path}")
        
        # 打印结果摘要
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        logger.info("\n" + str(results_df))
    
def main():
    """主函数"""
    logger.info("开始JoinQuant数据分析...")
    
    # 设置日期范围 (确保在您账号的有效期内)
    # 根据测试结果，您的账号数据范围为 2023-12-14 至 2024-12-20
    start_date = "2024-11-01"
    end_date = "2024-12-20"
    
    # 选择股票
    stock_codes = [
        '600519.XSHG',  # 贵州茅台
        '000858.XSHE',  # 五粮液
        '601318.XSHG',  # 中国平安
        '600036.XSHG',  # 招商银行
        '000333.XSHE',  # 美的集团
        '600276.XSHG',  # 恒瑞医药
        '002594.XSHE',  # 比亚迪
        '600900.XSHG',  # 长江电力
        '601888.XSHG',  # 中国中免
        '600028.XSHG',  # 中国石化
    ]
    
    # 获取数据
    stock_data = get_jq_data(start_date, end_date, stock_codes)
    
    # 检查数据
    if not stock_data:
        logger.error("未获取到数据，程序退出")
        return
    
    # 分析并绘制图表
    plot_stock_charts(stock_data, top_n=5, save_dir=results_dir)
    
    logger.info("JoinQuant数据分析完成")

if __name__ == "__main__":
    print("=" * 80)
    print(" JoinQuant数据分析示例 ".center(80, "="))
    print("=" * 80)
    
    # 运行主函数
    main()
    
    print("=" * 80)
    print(" 分析完成 ".center(80, "="))
    print("=" * 80) 