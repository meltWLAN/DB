#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票分析系统主程序
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
import random
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.config import CONFIG
from src.data.data_manager import DataManager
from src.indicators.technical import TechnicalIndicators
from src.indicators.advanced_indicators import AdvancedIndicators
from src.strategies.analysis_strategies import TrendFollowingStrategy, ReversalStrategy, VolatilityBreakoutStrategy, MultiStrategyAnalyzer
from src.backtest.backtest_engine import BacktestEngine, run_backtest
from src.visualization.chart_generator import ChartGenerator, create_stock_dashboard


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stock_analysis.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='股票分析系统')
    
    # 基本参数
    parser.add_argument('--stocks', type=str, help='要分析的股票代码，如"000001,600519"')
    parser.add_argument('--mode', type=str, default='simple', 
                      choices=['simple', 'test', 'full', 'backtest', 'visualization'],
                      help='分析模式: simple=简单分析, test=测试模式, full=完整分析, backtest=回测模式, visualization=可视化')
    parser.add_argument('--start_date', type=str, help='开始日期，格式：YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, help='结束日期，格式：YYYY-MM-DD')
    
    # 数据相关参数
    parser.add_argument('--clear_cache', action='store_true', help='清除缓存数据')
    parser.add_argument('--use_test_data', action='store_true', help='使用测试数据')
    parser.add_argument('--data_source', type=str, default='akshare', choices=['tushare', 'akshare', 'both'],
                      help='数据源：tushare, akshare, both')
    
    # 分析相关参数
    parser.add_argument('--strategy', type=str, default='trend', 
                      choices=['trend', 'reversal', 'volatility', 'multi'],
                      help='分析策略：trend=趋势跟踪, reversal=反转策略, volatility=波动率突破, multi=多策略')
    parser.add_argument('--output', type=str, default='results', help='结果输出目录')
    parser.add_argument('--export_format', type=str, default='csv', choices=['csv', 'json'],
                      help='导出格式：csv或json')
    
    # 回测相关参数
    parser.add_argument('--initial_capital', type=float, default=100000, help='回测初始资金')
    parser.add_argument('--plot_backtest', action='store_true', help='绘制回测结果图表')
    
    # 可视化相关参数
    parser.add_argument('--chart_type', type=str, default='candlestick', 
                      choices=['candlestick', 'technical', 'comparison', 'dashboard'],
                      help='图表类型：candlestick=K线图, technical=技术指标图, comparison=对比图, dashboard=仪表板')
    parser.add_argument('--save_chart', action='store_true', help='保存图表')
    
    return parser.parse_args()

def generate_test_data(stock_code, days=90, start_date=None, end_date=None, volatility=0.02):
    """生成测试数据
    
    Args:
        stock_code: 股票代码
        days: 生成数据的天数
        start_date: 开始日期，如果为None则使用当前日期往前推days天
        end_date: 结束日期，如果为None则使用当前日期
        volatility: 股价波动率
        
    Returns:
        DataFrame: 生成的测试数据
    """
    # 设置日期范围
    if end_date is None:
        end_date = datetime.now().date()
    else:
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    if start_date is None:
        start_date = end_date - timedelta(days=days)
    else:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    
    # 生成日期序列（仅工作日）
    all_dates = []
    current_date = start_date
    while current_date <= end_date:
        # 跳过周末
        if current_date.weekday() < 5:  # 0-4 表示周一至周五
            all_dates.append(current_date)
        current_date += timedelta(days=1)
    
    # 初始价格
    base_price = 10 + random.random() * 10
    
    # 随机生成价格
    data = []
    prev_close = base_price
    for i, date in enumerate(all_dates):
        # 生成开盘价（基于前一天的收盘价）
        if i == 0:
            open_price = base_price
        else:
            # 开盘价在前一天收盘价基础上有一个小的随机变化
            open_price = prev_close * (1 + random.uniform(-0.01, 0.01))
        
        # 模拟趋势（循环上涨和下跌）
        trend_factor = 0.0003 * np.sin(i / 10)
        
        # 当日价格变化率
        daily_change = random.normalvariate(trend_factor, volatility)
        
        # 确保价格不会变成负数
        daily_change = max(daily_change, -0.1)
        
        # 生成高低价
        intraday_volatility = prev_close * volatility * 0.5
        high_price = max(open_price, prev_close * (1 + daily_change)) + random.uniform(0, intraday_volatility)
        low_price = min(open_price, prev_close * (1 + daily_change)) - random.uniform(0, intraday_volatility)
        low_price = max(low_price, 0.1)  # 确保价格为正
        
        # 生成收盘价
        close_price = prev_close * (1 + daily_change)
        close_price = max(close_price, 0.1)  # 确保价格为正
        
        # 生成成交量（与价格变动相关）
        volume_base = 100000 + random.uniform(-50000, 50000)
        volume = int(volume_base * (1 + abs(daily_change) * 10))  # 价格变动越大，成交量越大
        
        # 添加数据
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume,
            'daily_return': daily_change,
            'ts_code': stock_code
        })
        
        # 更新前一天收盘价
        prev_close = close_price
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 添加成交额
    df['amount'] = df['volume'] * df['close']
    
    return df

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 检查是否需要清除缓存
    if args.clear_cache:
        try:
            data_manager = DataManager()
            data_manager.clear_cache()
            print("缓存已清除")
            if '--clear_cache' in sys.argv and len(sys.argv) == 2:
                return  # 如果只有清除缓存参数，则执行完后退出
        except Exception as e:
            logger.error(f"清除缓存失败: {str(e)}")
    
    # 初始化数据管理器
    data_manager = DataManager(data_source=args.data_source)
    
    # 设置日期范围
    start_date = args.start_date or (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    
    # 解析股票代码列表
    if args.stocks:
        stock_codes = args.stocks.split(',')
    else:
        # 如果未提供股票代码，使用默认股票
        stock_codes = ["000001"]
    
    print(f"股票分析系统初始化完成，数据源: {args.data_source}")
    print(f"分析模式: {args.mode}, 策略: {args.strategy}")
    print(f"分析股票: {', '.join(stock_codes)}")
    print(f"日期范围: {start_date} 至 {end_date}")
    
    # 处理不同的分析模式
    if args.mode == 'test':
        # 测试模式：使用测试数据
        for stock_code in stock_codes:
            print(f"\n开始分析股票: {stock_code}")
            
            # 生成测试数据
            test_data = generate_test_data(stock_code, start_date=start_date, end_date=end_date)
            print(f"生成测试数据完成，共 {len(test_data)} 行")
            
            # 计算技术指标
            tech_indicators = TechnicalIndicators()
            adv_indicators = AdvancedIndicators()
            
            test_data = tech_indicators.calculate_all_indicators(test_data)
            test_data = adv_indicators.calculate_all_advanced_indicators(test_data)
            
            # 选择分析策略
            if args.strategy == 'trend':
                strategy = TrendFollowingStrategy()
            elif args.strategy == 'reversal':
                strategy = ReversalStrategy()
            elif args.strategy == 'volatility':
                strategy = VolatilityBreakoutStrategy()
            elif args.strategy == 'multi':
                strategy = MultiStrategyAnalyzer()
            
            # 运行分析
            result = strategy.analyze(test_data)
            
            # 输出分析结果
            if isinstance(result, dict):
                print("\n分析结果:")
                print(f"策略: {result.get('strategy_name', args.strategy)}")
                print(f"得分: {result.get('score', result.get('combined_score', 0))}")
                
                if args.strategy == 'multi':
                    print(f"股票状态: {result.get('stock_status', '未知')}")
                    print(f"推荐: {result.get('recommendation', '无')}")
                else:
                    signals = result.get('signals', [])
                    if signals:
                        print("\n检测到的信号:")
                        for signal in signals:
                            signal_type = signal.get('type', '')
                            signal_name = signal.get('name', '')
                            signal_weight = signal.get('weight', '')
                            print(f" - {signal_name} ({signal_type}, 权重:{signal_weight})")
            
            # 保存结果
            output_file = os.path.join(args.output, f"{stock_code}_test_analysis.{args.export_format}")
            os.makedirs(args.output, exist_ok=True)
            
            if args.export_format == 'csv':
                test_data.to_csv(output_file, index=False)
            else:
                # 为JSON格式准备数据
                json_data = {
                    'stock_code': stock_code,
                    'analysis_result': result,
                    'data_summary': {
                        'rows': len(test_data),
                        'start_date': str(test_data['date'].min()),
                        'end_date': str(test_data['date'].max()),
                        'min_price': float(test_data['low'].min()),
                        'max_price': float(test_data['high'].max()),
                        'avg_volume': float(test_data['volume'].mean())
                    }
                }
                
                with open(output_file, 'w') as f:
                    json.dump(json_data, f, indent=2, default=str)
            
            print(f"分析结果已保存至 {output_file}")
            
            # 可视化处理
            if args.chart_type or args.save_chart:
                chart_generator = ChartGenerator(output_dir=os.path.join(args.output, 'charts'))
                
                if args.chart_type == 'candlestick':
                    chart_generator.generate_candlestick_chart(
                        test_data, 
                        title=f"{stock_code} K线图",
                        indicators=['MA5', 'MA10', 'MA20'],
                        save_path=f"{stock_code}_candlestick.png" if args.save_chart else None
                    )
                elif args.chart_type == 'technical':
                    chart_generator.generate_technical_chart(
                        test_data,
                        title=f"{stock_code} 技术指标",
                        save_path=f"{stock_code}_technical.png" if args.save_chart else None
                    )
                elif args.chart_type == 'dashboard':
                    dashboard_path = os.path.join(args.output, 'dashboards', f"{stock_code}_dashboard.html")
                    create_stock_dashboard(
                        test_data,
                        dashboard_path,
                        stock_code=stock_code
                    )
                    print(f"股票分析仪表板已生成: {dashboard_path}")
    
    elif args.mode == 'backtest':
        # 回测模式
        for stock_code in stock_codes:
            print(f"\n开始回测股票: {stock_code}")
            
            try:
                # 获取股票数据
                if args.use_test_data:
                    stock_data = generate_test_data(stock_code, start_date=start_date, end_date=end_date)
                    print(f"使用测试数据进行回测，共 {len(stock_data)} 行")
                else:
                    stock_data = data_manager.get_stock_data(
                        stock_code=stock_code,
                        start_date=start_date,
                        end_date=end_date,
                        adjust='qfq'  # 前复权数据
                    )
                    print(f"获取股票数据完成，共 {len(stock_data)} 行")
                
                # 计算技术指标
                tech_indicators = TechnicalIndicators()
                adv_indicators = AdvancedIndicators()
                
                stock_data = tech_indicators.calculate_all_indicators(stock_data)
                stock_data = adv_indicators.calculate_all_advanced_indicators(stock_data)
                
                # 执行回测
                backtest_result = run_backtest(
                    stock_data=stock_data,
                    strategy_type=args.strategy,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=args.initial_capital,
                    plot_results=args.plot_backtest,
                    save_plot=os.path.join(args.output, 'charts', f"{stock_code}_backtest.png") if args.save_chart else None,
                    save_results=os.path.join(args.output, f"{stock_code}_backtest_result.json")
                )
                
                if backtest_result:
                    # 输出回测指标
                    metrics = backtest_result.get('metrics', {})
                    print("\n回测结果:")
                    print(f"策略: {args.strategy}")
                    print(f"总收益率: {metrics.get('total_return', 0):.2f}%")
                    print(f"年化收益率: {metrics.get('annualized_return', 0):.2f}%")
                    print(f"最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
                    print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
                    print(f"交易次数: {metrics.get('total_trades', 0)}")
                    print(f"胜率: {metrics.get('win_rate', 0):.2f}%")
                    print(f"利润因子: {metrics.get('profit_factor', 0):.2f}")
                    print(f"最终资金: {metrics.get('final_capital', 0):.2f}")
                
            except Exception as e:
                logger.error(f"回测股票 {stock_code} 时出错: {str(e)}")
    
    elif args.mode == 'visualization':
        # 可视化模式
        for stock_code in stock_codes:
            print(f"\n开始可视化股票: {stock_code}")
            
            try:
                # 获取股票数据
                if args.use_test_data:
                    stock_data = generate_test_data(stock_code, start_date=start_date, end_date=end_date)
                    print(f"使用测试数据进行可视化，共 {len(stock_data)} 行")
                else:
                    stock_data = data_manager.get_stock_data(
                        stock_code=stock_code,
                        start_date=start_date,
                        end_date=end_date,
                        adjust='qfq'  # 前复权数据
                    )
                    print(f"获取股票数据完成，共 {len(stock_data)} 行")
                
                # 计算技术指标
                tech_indicators = TechnicalIndicators()
                stock_data = tech_indicators.calculate_all_indicators(stock_data)
                
                # 创建可视化
                chart_generator = ChartGenerator(output_dir=os.path.join(args.output, 'charts'))
                
                if args.chart_type == 'candlestick':
                    chart_generator.generate_candlestick_chart(
                        stock_data, 
                        title=f"{stock_code} K线图",
                        indicators=['MA5', 'MA10', 'MA20'],
                        save_path=f"{stock_code}_candlestick.png" if args.save_chart else None
                    )
                    print(f"K线图生成完成")
                
                elif args.chart_type == 'technical':
                    chart_generator.generate_technical_chart(
                        stock_data,
                        title=f"{stock_code} 技术指标",
                        save_path=f"{stock_code}_technical.png" if args.save_chart else None
                    )
                    print(f"技术指标图生成完成")
                
                elif args.chart_type == 'comparison' and len(stock_codes) > 1:
                    # 收集多只股票的数据
                    all_stock_data = {}
                    for code in stock_codes:
                        if code == stock_code:
                            all_stock_data[code] = stock_data
                        else:
                            try:
                                if args.use_test_data:
                                    code_data = generate_test_data(code, start_date=start_date, end_date=end_date)
                                else:
                                    code_data = data_manager.get_stock_data(
                                        stock_code=code,
                                        start_date=start_date,
                                        end_date=end_date,
                                        adjust='qfq'
                                    )
                                all_stock_data[code] = code_data
                            except Exception as e:
                                logger.error(f"获取股票 {code} 数据时出错: {str(e)}")
                    
                    chart_generator.generate_comparison_chart(
                        all_stock_data,
                        title=f"股票价格对比图 ({start_date} - {end_date})",
                        save_path="stock_comparison.png" if args.save_chart else None
                    )
                    print(f"股票对比图生成完成")
                
                elif args.chart_type == 'dashboard':
                    dashboard_path = os.path.join(args.output, 'dashboards', f"{stock_code}_dashboard.html")
                    os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
                    
                    create_stock_dashboard(
                        stock_data,
                        dashboard_path,
                        stock_code=stock_code
                    )
                    print(f"股票分析仪表板已生成: {dashboard_path}")
            
            except Exception as e:
                logger.error(f"可视化股票 {stock_code} 时出错: {str(e)}")
    
    else:
        # 简单分析模式或完整分析模式
        for stock_code in stock_codes:
            print(f"\n开始分析股票: {stock_code}")
            
            try:
                # 获取股票数据
                if args.use_test_data:
                    stock_data = generate_test_data(stock_code, start_date=start_date, end_date=end_date)
                    print(f"使用测试数据进行分析，共 {len(stock_data)} 行")
                else:
                    stock_data = data_manager.get_stock_data(
                        stock_code=stock_code,
                        start_date=start_date,
                        end_date=end_date,
                        adjust='qfq'  # 前复权数据
                    )
                    print(f"获取股票数据完成，共 {len(stock_data)} 行")
                
                # 计算技术指标
                tech_indicators = TechnicalIndicators()
                adv_indicators = AdvancedIndicators()
                
                stock_data = tech_indicators.calculate_all_indicators(stock_data)
                
                if args.mode == 'full' or args.strategy in ['reversal', 'volatility', 'multi']:
                    stock_data = adv_indicators.calculate_all_advanced_indicators(stock_data)
                
                # 选择分析策略
                if args.strategy == 'trend':
                    strategy = TrendFollowingStrategy()
                elif args.strategy == 'reversal':
                    strategy = ReversalStrategy()
                elif args.strategy == 'volatility':
                    strategy = VolatilityBreakoutStrategy()
                elif args.strategy == 'multi':
                    strategy = MultiStrategyAnalyzer()
                
                # 运行分析
                result = strategy.analyze(stock_data)
                
                # 输出分析结果
                if isinstance(result, dict):
                    print("\n分析结果:")
                    print(f"策略: {result.get('strategy_name', args.strategy)}")
                    
                    if args.strategy == 'multi':
                        print(f"综合得分: {result.get('combined_score', 0):.2f}")
                        print(f"股票状态: {result.get('stock_status', '未知')}")
                        print(f"推荐: {result.get('recommendation', '无')}")
                        
                        # 显示顶部信号
                        top_signals = result.get('top_signals', [])
                        if top_signals:
                            print("\n顶部信号:")
                            for signal in top_signals[:5]:  # 只显示前5个信号
                                signal_type = signal.get('type', '')
                                signal_name = signal.get('name', '')
                                signal_weight = signal.get('weight', '')
                                signal_strategy = signal.get('strategy', '')
                                print(f" - {signal_name} ({signal_type}, 权重:{signal_weight}, 来源:{signal_strategy})")
                    else:
                        print(f"得分: {result.get('score', 0):.2f}")
                        
                        signals = result.get('signals', [])
                        if signals:
                            print("\n检测到的信号:")
                            for signal in signals:
                                signal_type = signal.get('type', '')
                                signal_name = signal.get('name', '')
                                signal_weight = signal.get('weight', '')
                                print(f" - {signal_name} ({signal_type}, 权重:{signal_weight})")
                
                # 保存结果
                output_file = os.path.join(args.output, f"{stock_code}_{start_date}_{end_date}.{args.export_format}")
                os.makedirs(args.output, exist_ok=True)
                
                if args.export_format == 'csv':
                    stock_data.to_csv(output_file, index=False)
                else:
                    # 为JSON格式准备数据
                    json_data = {
                        'stock_code': stock_code,
                        'analysis_result': result,
                        'data_summary': {
                            'rows': len(stock_data),
                            'start_date': str(stock_data['date'].min()),
                            'end_date': str(stock_data['date'].max()),
                            'min_price': float(stock_data['low'].min()),
                            'max_price': float(stock_data['high'].max()),
                            'avg_volume': float(stock_data['volume'].mean())
                        }
                    }
                    
                    with open(output_file, 'w') as f:
                        json.dump(json_data, f, indent=2, default=str)
                
                print(f"分析结果已保存至 {output_file}")
                
                # 如果需要生成图表
                if args.save_chart:
                    chart_generator = ChartGenerator(output_dir=os.path.join(args.output, 'charts'))
                    
                    chart_generator.generate_candlestick_chart(
                        stock_data, 
                        title=f"{stock_code} K线图",
                        indicators=['MA5', 'MA10', 'MA20'],
                        save_path=f"{stock_code}_candlestick.png"
                    )
                    
                    chart_generator.generate_technical_chart(
                        stock_data,
                        title=f"{stock_code} 技术指标",
                        save_path=f"{stock_code}_technical.png"
                    )
                    
                    print(f"图表已保存至 {os.path.join(args.output, 'charts')}")
                
                # 如果是完整分析且需要生成仪表板
                if args.mode == 'full' and args.chart_type == 'dashboard':
                    dashboard_path = os.path.join(args.output, 'dashboards', f"{stock_code}_dashboard.html")
                    os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
                    
                    create_stock_dashboard(
                        stock_data,
                        dashboard_path,
                        stock_code=stock_code
                    )
                    print(f"股票分析仪表板已生成: {dashboard_path}")
            
            except Exception as e:
                logger.error(f"分析股票 {stock_code} 时出错: {str(e)}")
    
    print("\n分析完成！")

if __name__ == '__main__':
    main() 