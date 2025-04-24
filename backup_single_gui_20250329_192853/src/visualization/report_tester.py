"""
报告生成器测试脚本
用于演示如何使用报告生成器生成不同类型的报告
"""

import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import sys
import os
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# 导入报告生成器
from src.visualization.report_generator import ReportGenerator
from src.config.settings import RESULTS_DIR

def generate_sample_data():
    """生成样例数据用于测试"""
    # 生成日期范围
    date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    
    # 生成价格数据
    np.random.seed(42)
    prices = np.random.normal(loc=100, scale=10, size=len(date_range))
    prices = np.exp(np.cumsum(np.random.normal(loc=0.0001, scale=0.01, size=len(date_range))))
    prices *= 100
    
    # 创建DataFrame
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.01, len(prices))),
        'high': prices * (1 + abs(np.random.normal(0, 0.02, len(prices)))),
        'low': prices * (1 - abs(np.random.normal(0, 0.02, len(prices)))),
        'close': prices,
        'volume': np.random.randint(1000, 100000, len(prices))
    }, index=date_range)
    
    # 计算移动平均线
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma30'] = df['close'].rolling(window=30).mean()
    
    # 计算技术指标
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (std * 2)
    df['bb_lower'] = df['bb_middle'] - (std * 2)
    
    # KDJ
    low_min = df['low'].rolling(window=9).min()
    high_max = df['high'].rolling(window=9).max()
    df['rsv'] = (df['close'] - low_min) / (high_max - low_min) * 100
    df['k'] = df['rsv'].ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    
    # 去除NaN值
    df = df.dropna()
    
    return df

def test_stock_analysis_report():
    """测试生成股票分析报告"""
    logger.info("开始测试股票分析报告生成")
    
    # 创建报告生成器
    report_generator = ReportGenerator()
    
    # 生成样例数据
    stock_data = generate_sample_data()
    
    # 准备股票信息
    stock_info = {
        'code': '000001',
        'name': '平安银行',
        'industry': '银行',
        'market': '深交所',
        'description': '平安银行是中国平安保险(集团)股份有限公司控股的一家跨区域经营的股份制商业银行，为中国大陆12家全国性股份制商业银行之一。'
    }
    
    # 准备技术指标数据
    indicators_data = {
        'rsi': stock_data['rsi'],
        'macd': stock_data['macd'],
        'signal': stock_data['signal'],
        'macd_hist': stock_data['macd_hist'],
        'k': stock_data['k'],
        'd': stock_data['d'],
        'j': stock_data['j'],
        'bb_upper': stock_data['bb_upper'],
        'bb_middle': stock_data['bb_middle'],
        'bb_lower': stock_data['bb_lower'],
        'close': stock_data['close'],
        'ma5': stock_data['ma5'],
        'ma10': stock_data['ma10'],
        'ma20': stock_data['ma20']
    }
    
    # 准备投资建议
    recommendations = [
        {
            'action': '买入',
            'confidence': 0.85,
            'reason': '该股票RSI指标处于超卖区域，技术面显示有强烈反弹信号。'
        },
        {
            'action': '持有',
            'confidence': 0.60,
            'reason': '移动平均线呈现多头排列，中长期趋势向好。'
        },
        {
            'action': '观望',
            'confidence': 0.40,
            'reason': '近期市场波动较大，建议等待更明确的信号。'
        }
    ]
    
    # 准备分析总结
    summary = """
    <p>根据技术分析，该股票当前处于上升通道中，但短期可能面临回调压力。从基本面来看，公司业绩稳定增长，ROE保持在行业前列。</p>
    <p>考虑到当前市场环境和股票估值水平，建议投资者采取以下策略：</p>
    <ul>
        <li>短期投资者可在回调至支撑位时分批建仓</li>
        <li>中长期投资者可继续持有，并考虑在价格突破前高时加仓</li>
        <li>激进投资者可设置止损位，追踪短期趋势</li>
    </ul>
    <p>风险提示：宏观经济下行风险、行业政策变动风险、公司业绩不及预期风险等。</p>
    """
    
    # 生成报告
    report_path = report_generator.generate_stock_analysis_report(
        stock_data=stock_data,
        indicators_data=indicators_data,
        stock_info=stock_info,
        recommendations=recommendations,
        summary=summary,
        format="html"
    )
    
    logger.info(f"股票分析报告生成成功: {report_path}")
    return report_path

def test_backtest_report():
    """测试生成回测报告"""
    logger.info("开始测试回测报告生成")
    
    # 创建报告生成器
    report_generator = ReportGenerator()
    
    # 准备策略信息
    strategy_info = {
        'name': '双均线交叉策略',
        'period': '2023-01-01 至 2023-12-31',
        'initial_capital': 100000,
        'final_capital': 125000,
        'description': '该策略使用5日均线和20日均线的交叉作为交易信号，当5日均线上穿20日均线时买入，下穿时卖出。'
    }
    
    # 准备性能数据
    performance_data = {
        'total_return': 0.25,  # 总收益率
        'annual_return': 0.28,  # 年化收益率
        'sharpe_ratio': 1.35,  # 夏普比率
        'max_drawdown': 0.12,  # 最大回撤
        'win_rate': 0.65,  # 胜率
        'profit_factor': 1.8,  # 盈亏比
        'trade_count': 24,  # 交易次数
        'beta': 0.85,  # Beta
        'alpha': 0.05,  # Alpha
        'sortino_ratio': 1.45,  # 索提诺比率
        'information_ratio': 0.75,  # 信息比率
        'volatility': 0.15,  # 波动率
    }
    
    # 准备均线交叉策略的回测数据
    stock_data = generate_sample_data()
    
    # 生成交易记录
    trades = []
    is_holding = False
    entry_price = 0
    entry_date = None
    
    for i in range(1, len(stock_data)):
        prev_ma5 = stock_data['ma5'].iloc[i-1]
        prev_ma20 = stock_data['ma20'].iloc[i-1]
        curr_ma5 = stock_data['ma5'].iloc[i]
        curr_ma20 = stock_data['ma20'].iloc[i]
        
        # 买入信号: 5日均线上穿20日均线
        if not is_holding and prev_ma5 <= prev_ma20 and curr_ma5 > curr_ma20:
            is_holding = True
            entry_price = stock_data['close'].iloc[i]
            entry_date = stock_data.index[i]
            
        # 卖出信号: 5日均线下穿20日均线
        elif is_holding and prev_ma5 >= prev_ma20 and curr_ma5 < curr_ma20:
            exit_price = stock_data['close'].iloc[i]
            exit_date = stock_data.index[i]
            
            # 计算持有天数和收益率
            holding_days = (exit_date - entry_date).days
            return_rate = (exit_price / entry_price) - 1
            
            trades.append({
                'symbol': '000001',
                'direction': '多',
                'entry_date': entry_date,
                'entry_price': round(entry_price, 2),
                'exit_date': exit_date,
                'exit_price': round(exit_price, 2),
                'holding_days': holding_days,
                'return': return_rate
            })
            
            is_holding = False
    
    # 准备绘制权益曲线和回撤数据
    equity_curve = pd.Series(index=stock_data.index, data=1.0)
    benchmark = pd.Series(index=stock_data.index, data=stock_data['close'] / stock_data['close'].iloc[0])
    
    current_value = 1.0
    for trade in trades:
        idx = stock_data.index.get_indexer([trade['exit_date']])[0]
        current_value *= (1 + trade['return'])
        equity_curve.iloc[idx:] = current_value
    
    # 计算回撤
    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    
    # 添加到性能数据
    performance_data['equity_curve'] = equity_curve
    performance_data['benchmark'] = benchmark
    performance_data['drawdowns'] = drawdowns
    
    # 生成报告
    report_path = report_generator.generate_backtest_report(
        strategy_info=strategy_info,
        performance_data=performance_data,
        trades_data=trades,
        format="html"
    )
    
    logger.info(f"回测报告生成成功: {report_path}")
    return report_path

def main():
    """主函数"""
    try:
        # 测试股票分析报告
        stock_report_path = test_stock_analysis_report()
        print(f"股票分析报告生成路径: {stock_report_path}")
        
        # 测试回测报告
        backtest_report_path = test_backtest_report()
        print(f"回测报告生成路径: {backtest_report_path}")
        
        print("报告生成测试完成，可以在浏览器中打开以下路径查看生成的报告:")
        print(f"1. 股票分析报告: file://{stock_report_path}")
        print(f"2. 回测报告: file://{backtest_report_path}")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)

if __name__ == "__main__":
    main() 