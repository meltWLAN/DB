#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版JoinQuant数据分析脚本
- 增加了更多股票
- 增加了行业分析
- 增加了相关性分析
- 增加了更多技术指标
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置中文字体
try:
    # 尝试使用系统中文字体
    font_path = '/System/Library/Fonts/PingFang.ttc'  # macOS中文字体
    if os.path.exists(font_path):
        chinese_font = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = chinese_font.get_name()
    else:
        # 如果找不到系统字体，使用matplotlib内置字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    # 如果设置字体失败，则忽略中文显示问题
    pass

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加当前目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 确保结果目录存在
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/enhanced_analysis")
os.makedirs(results_dir, exist_ok=True)

# 导入配置
from src.config import DATA_SOURCE_CONFIG

# 股票基本信息，用于显示中文名称和行业
STOCK_INFO = {
    # 金融板块
    '601318.XSHG': {'name': '中国平安', 'industry': '金融保险'},
    '600036.XSHG': {'name': '招商银行', 'industry': '银行'},
    '601166.XSHG': {'name': '兴业银行', 'industry': '银行'},
    '601328.XSHG': {'name': '交通银行', 'industry': '银行'},
    
    # 消费板块
    '600519.XSHG': {'name': '贵州茅台', 'industry': '白酒'},
    '000858.XSHE': {'name': '五粮液', 'industry': '白酒'},
    '600887.XSHG': {'name': '伊利股份', 'industry': '食品饮料'},
    '603288.XSHG': {'name': '海天味业', 'industry': '食品饮料'},
    
    # 科技板块
    '000333.XSHE': {'name': '美的集团', 'industry': '家电'},
    '002594.XSHE': {'name': '比亚迪', 'industry': '新能源车'},
    '600276.XSHG': {'name': '恒瑞医药', 'industry': '医药'},
    '002230.XSHE': {'name': '科大讯飞', 'industry': '人工智能'},
    
    # 能源板块
    '600900.XSHG': {'name': '长江电力', 'industry': '电力'},
    '601857.XSHG': {'name': '中国石油', 'industry': '石油'},
    '600028.XSHG': {'name': '中国石化', 'industry': '石油'},
    '601088.XSHG': {'name': '中国神华', 'industry': '煤炭'},
    
    # 互联网板块
    '000651.XSHE': {'name': '格力电器', 'industry': '家电'},
    '600809.XSHG': {'name': '山西汾酒', 'industry': '白酒'},
    '601888.XSHG': {'name': '中国中免', 'industry': '免税'},
    '600031.XSHG': {'name': '三一重工', 'industry': '机械设备'},
}

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
            stock_name = STOCK_INFO.get(stock_code, {}).get('name', stock_code)
            logger.info(f"获取 {stock_code} ({stock_name}) 数据...")
            
            # 获取价格数据
            df = jq.get_price(stock_code, start_date=start_date, end_date=end_date, 
                           frequency='daily', fields=['open', 'close', 'high', 'low', 'volume', 'money'])
            
            # 将日期从索引转为列
            df = df.reset_index()
            df.rename(columns={'index': 'date'}, inplace=True)
            
            # 添加股票信息
            df['code'] = stock_code
            if stock_code in STOCK_INFO:
                df['name'] = STOCK_INFO[stock_code]['name']
                df['industry'] = STOCK_INFO[stock_code]['industry']
            else:
                df['name'] = stock_code
                df['industry'] = '未知'
            
            # 计算技术指标
            # 1. 移动平均线
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA10'] = df['close'].rolling(window=10).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA60'] = df['close'].rolling(window=60).mean()
            
            # 2. MACD
            df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['DIF'] = df['EMA12'] - df['EMA26']
            df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
            df['MACD'] = 2 * (df['DIF'] - df['DEA'])
            
            # 3. RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 4. 布林带
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['STD20'] = df['close'].rolling(window=20).std()
            df['UPPER'] = df['MA20'] + 2 * df['STD20']
            df['LOWER'] = df['MA20'] - 2 * df['STD20']
            
            # 5. KDJ
            low_min = df['low'].rolling(window=9).min()
            high_max = df['high'].rolling(window=9).max()
            df['RSV'] = (df['close'] - low_min) / (high_max - low_min) * 100
            df['K'] = df['RSV'].ewm(alpha=1/3, adjust=False).mean()
            df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
            df['J'] = 3 * df['K'] - 2 * df['D']
            
            # 6. 成交量相关指标
            df['VOL_MA5'] = df['volume'].rolling(window=5).mean()
            df['VOL_MA10'] = df['volume'].rolling(window=10).mean()
            
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
        
        # 计算布林带突破次数
        df['upper_break'] = df['close'] > df['UPPER']
        df['lower_break'] = df['close'] < df['LOWER']
        upper_break_count = df['upper_break'].sum()
        lower_break_count = df['lower_break'].sum()
        
        # 计算KDJ金叉死叉次数
        df['kdj_golden_cross'] = (df['J'] > df['D']) & (df['J'].shift(1) <= df['D'].shift(1))
        df['kdj_death_cross'] = (df['J'] < df['D']) & (df['J'].shift(1) >= df['D'].shift(1))
        kdj_golden_cross_count = df['kdj_golden_cross'].sum()
        kdj_death_cross_count = df['kdj_death_cross'].sum()
        
        # 获取股票信息
        stock_name = df['name'].iloc[0] if 'name' in df.columns else stock_code
        industry = df['industry'].iloc[0] if 'industry' in df.columns else '未知'
        
        # 收集分析结果
        analysis_results[stock_code] = {
            'code': stock_code,
            'name': stock_name,
            'industry': industry,
            'total_return': df['cumulative_return'].iloc[-1] - 1,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'golden_cross_count': golden_cross_count,
            'death_cross_count': death_cross_count,
            'upper_break_count': upper_break_count,
            'lower_break_count': lower_break_count,
            'kdj_golden_cross_count': kdj_golden_cross_count,
            'kdj_death_cross_count': kdj_death_cross_count,
            'avg_volume': df['volume'].mean(),
            'avg_turnover': df['money'].mean() if 'money' in df.columns else 0,
            'last_close': df['close'].iloc[-1],
            'last_date': df['date'].iloc[-1],
            'data_days': len(df),
            'avg_rsi': df['RSI'].mean(),
            'last_rsi': df['RSI'].iloc[-1],
        }
        
    return analysis_results

def plot_stock_charts(data, top_n=5, save_dir=None):
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
    
    for stock_code, result in top_stocks:
        df = data[stock_code]
        stock_name = result['name']
        
        # 创建图表
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), sharex=True, 
                                              gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        
        # 绘制K线图和均线
        ax1.plot(df['date'], df['close'], label='收盘价')
        ax1.plot(df['date'], df['MA5'], label='MA5')
        ax1.plot(df['date'], df['MA10'], label='MA10')
        ax1.plot(df['date'], df['MA20'], label='MA20')
        ax1.plot(df['date'], df['UPPER'], 'r--', label='布林上轨')
        ax1.plot(df['date'], df['LOWER'], 'g--', label='布林下轨')
        ax1.set_title(f'{stock_code} {stock_name} - {result["industry"]} - 价格走势')
        ax1.legend(loc='best')
        ax1.grid(True)
        
        # 绘制成交量
        ax2.bar(df['date'], df['volume'], label='成交量')
        ax2.plot(df['date'], df['VOL_MA5'], 'r-', label='量5日均线')
        ax2.set_title('成交量')
        ax2.legend(loc='best')
        ax2.grid(True)
        
        # 绘制MACD
        ax3.plot(df['date'], df['DIF'], label='DIF')
        ax3.plot(df['date'], df['DEA'], label='DEA')
        ax3.bar(df['date'], df['MACD'], label='MACD')
        ax3.set_title('MACD')
        ax3.legend(loc='best')
        ax3.grid(True)
        
        # 绘制KDJ
        ax4.plot(df['date'], df['K'], 'b-', label='K')
        ax4.plot(df['date'], df['D'], 'y-', label='D')
        ax4.plot(df['date'], df['J'], 'm-', label='J')
        ax4.axhline(y=80, color='r', linestyle='--')
        ax4.axhline(y=20, color='g', linestyle='--')
        ax4.set_title('KDJ')
        ax4.legend(loc='best')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        if save_dir:
            save_path = os.path.join(save_dir, f"{stock_code.replace('.', '_')}_{stock_name}_chart.png")
            plt.savefig(save_path)
            logger.info(f"保存图表: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    # 绘制收益率对比图
    plt.figure(figsize=(14, 8))
    
    for stock_code, result in top_stocks:
        df = data[stock_code]
        plt.plot(df['date'], df['cumulative_return'], label=f"{stock_code} {result['name']}")
    
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
        '股票代码', '股票名称', '行业', '总收益率', '年化收益率', '波动率', '夏普比率', '最大回撤', 
        '均线金叉次数', '均线死叉次数', 'KDJ金叉次数', 'KDJ死叉次数',
        '布林上轨突破', '布林下轨突破', '平均成交量', '最新收盘价', '最新RSI'
    ])
    
    for i, (stock_code, result) in enumerate(sorted_stocks):
        results_df.loc[i] = [
            result['code'],
            result['name'],
            result['industry'],
            f"{result['total_return']:.2%}",
            f"{result['annual_return']:.2%}",
            f"{result['volatility']:.2%}",
            f"{result['sharpe_ratio']:.2f}",
            f"{result['max_drawdown']:.2%}",
            result['golden_cross_count'],
            result['death_cross_count'],
            result['kdj_golden_cross_count'],
            result['kdj_death_cross_count'],
            result['upper_break_count'],
            result['lower_break_count'],
            f"{result['avg_volume']:.0f}",
            f"{result['last_close']:.2f}",
            f"{result['last_rsi']:.2f}",
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
    
    return analysis_results

def plot_correlation_matrix(data, save_dir=None):
    """绘制相关性矩阵
    
    Args:
        data: 股票数据字典
        save_dir: 保存图表的目录
    """
    # 提取每只股票的收盘价
    prices = {}
    for stock_code, df in data.items():
        stock_name = STOCK_INFO.get(stock_code, {}).get('name', stock_code)
        prices[f"{stock_code}_{stock_name}"] = df.set_index('date')['close']
    
    # 创建一个DataFrame包含所有股票的收盘价
    price_df = pd.DataFrame(prices)
    
    # 计算日收益率
    returns_df = price_df.pct_change().dropna()
    
    # 计算相关性矩阵
    corr_matrix = returns_df.corr()
    
    # 绘制相关性热力图
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('股票收益率相关性矩阵')
    plt.tight_layout()
    
    # 保存图表
    if save_dir:
        save_path = os.path.join(save_dir, "correlation_matrix.png")
        plt.savefig(save_path)
        logger.info(f"保存相关性矩阵图: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return corr_matrix

def plot_industry_analysis(analysis_results, save_dir=None):
    """绘制行业分析图表
    
    Args:
        analysis_results: 分析结果
        save_dir: 保存图表的目录
    """
    # 将分析结果转换为DataFrame
    results_df = pd.DataFrame.from_dict(analysis_results, orient='index')
    
    # 按行业分组计算平均收益率
    industry_returns = results_df.groupby('industry')['total_return'].mean().sort_values(ascending=False)
    
    # 绘制行业平均收益率
    plt.figure(figsize=(12, 8))
    industry_returns.plot(kind='bar')
    plt.title('各行业平均收益率')
    plt.xlabel('行业')
    plt.ylabel('平均收益率')
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    
    # 保存图表
    if save_dir:
        save_path = os.path.join(save_dir, "industry_returns.png")
        plt.savefig(save_path)
        logger.info(f"保存行业收益率图: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 绘制行业波动率
    industry_volatility = results_df.groupby('industry')['volatility'].mean().sort_values(ascending=True)
    
    plt.figure(figsize=(12, 8))
    industry_volatility.plot(kind='bar')
    plt.title('各行业平均波动率')
    plt.xlabel('行业')
    plt.ylabel('平均波动率')
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    
    # 保存图表
    if save_dir:
        save_path = os.path.join(save_dir, "industry_volatility.png")
        plt.savefig(save_path)
        logger.info(f"保存行业波动率图: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 绘制行业夏普比率
    industry_sharpe = results_df.groupby('industry')['sharpe_ratio'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    industry_sharpe.plot(kind='bar')
    plt.title('各行业平均夏普比率')
    plt.xlabel('行业')
    plt.ylabel('平均夏普比率')
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    
    # 保存图表
    if save_dir:
        save_path = os.path.join(save_dir, "industry_sharpe.png")
        plt.savefig(save_path)
        logger.info(f"保存行业夏普比率图: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 创建行业分析表格
    industry_analysis = results_df.groupby('industry').agg({
        'total_return': 'mean',
        'annual_return': 'mean',
        'volatility': 'mean',
        'sharpe_ratio': 'mean',
        'max_drawdown': 'mean',
        'code': 'count'  # 计数每个行业有多少只股票
    }).rename(columns={'code': 'stock_count'}).sort_values('total_return', ascending=False)
    
    # 保存行业分析结果
    if save_dir:
        save_path = os.path.join(save_dir, "industry_analysis.csv")
        industry_analysis.to_csv(save_path)
        logger.info(f"保存行业分析结果: {save_path}")
        
        # 打印行业分析摘要
        logger.info("\n" + str(industry_analysis))
    
    return industry_analysis

def main():
    """主函数"""
    logger.info("开始增强版JoinQuant数据分析...")
    
    # 设置日期范围 (确保在您账号的有效期内)
    # 根据测试结果，您的账号数据范围为 2023-12-14 至 2024-12-20
    start_date = "2024-06-01"  # 调整为更早的日期，获取更长的数据周期
    end_date = "2024-12-20"    # 账号允许的最新日期
    
    # 选择股票 (使用STOCK_INFO中定义的所有股票)
    stock_codes = list(STOCK_INFO.keys())
    
    # 获取数据
    stock_data = get_jq_data(start_date, end_date, stock_codes)
    
    # 检查数据
    if not stock_data:
        logger.error("未获取到数据，程序退出")
        return
    
    # 分析并绘制图表
    analysis_results = plot_stock_charts(stock_data, top_n=10, save_dir=results_dir)
    
    # 绘制相关性矩阵
    corr_matrix = plot_correlation_matrix(stock_data, save_dir=results_dir)
    
    # 绘制行业分析
    industry_analysis = plot_industry_analysis(analysis_results, save_dir=results_dir)
    
    logger.info("增强版JoinQuant数据分析完成")

if __name__ == "__main__":
    print("=" * 80)
    print(" 增强版JoinQuant数据分析 ".center(80, "="))
    print("=" * 80)
    
    # 运行主函数
    main()
    
    print("=" * 80)
    print(" 分析完成 ".center(80, "="))
    print("=" * 80) 