#!/usr/bin/env python3
"""
测试增强版动量分析器
"""
import os
import sys
import logging
from datetime import datetime

# 配置日志记录
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# 导入EnhancedMomentumAnalyzer
from enhanced_momentum_analysis import EnhancedMomentumAnalyzer

def test_single_stock(analyzer, ts_code):
    """测试单个股票分析"""
    print(f"\n测试单个股票分析: {ts_code}")
    
    # 1. 获取股票数据 - 获取更多历史数据（240天而不是默认的120天）
    stock_data = analyzer.get_stock_data(ts_code, days=240)
    if stock_data is None or stock_data.empty:
        print(f"无法获取股票数据: {ts_code}")
        return None
    
    print(f"成功获取股票数据, 行数: {len(stock_data)}")
    print(stock_data.head(3))
    
    # 2. 计算技术指标
    try:
        stock_data_with_indicators = analyzer.calculate_technical_indicators(stock_data)
        if stock_data_with_indicators is None:
            print(f"计算技术指标失败: {ts_code}")
            return None
        
        print(f"成功计算技术指标, 行数: {len(stock_data_with_indicators)}")
        # 打印部分列
        indicator_columns = [col for col in stock_data_with_indicators.columns if col.startswith('MA') or col in ['RSI', 'MACD']]
        print(f"技术指标列: {indicator_columns}")
        
        # 3. 计算得分和信号
        score, signals = analyzer._calculate_score_and_signals(stock_data_with_indicators)
        print(f"股票得分: {score}, 信号: {signals}")
        
        return {
            'ts_code': ts_code,
            'score': score,
            'signals': signals,
            'data': stock_data_with_indicators
        }
    except Exception as e:
        print(f"分析股票时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print(f"开始测试增强版动量分析器，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 获取当前日期字符串
    today_str = datetime.now().strftime('%Y%m%d')
    
    # 使用增强动量分析器计算人工智能行业股票
    analyzer = EnhancedMomentumAnalyzer(
        use_tushare=True,
        cache_timeout=86400  # 默认缓存有效期为一天
    )
    
    # 设置参数（在实例化后设置）
    analyzer.start_date = '20240601'  # 修改为更早的时间以获取更多历史数据
    analyzer.end_date = today_str
    analyzer.params['min_history_days'] = 10  # 设置最小历史数据要求为10天
    analyzer.use_parallel = True
    logger = logging.getLogger('enhanced_momentum_analysis')
    logger.setLevel(logging.DEBUG)
    
    # 获取股票列表
    stocks = analyzer.get_stock_list()
    if stocks is None or stocks.empty:
        print("获取股票列表失败")
        return
    
    print(f"获取到 {len(stocks)} 支股票")
    
    # 测试用10只股票
    test_stocks = stocks.head(10)
    print(f"测试股票: {test_stocks['ts_code'].tolist()}")
    
    # 测试直接分析单个股票 - 选择一个大型蓝筹股
    blue_chip_stock = '601318.SH'  # 中国平安
    result = test_single_stock(analyzer, blue_chip_stock)
    
    if result:
        # 测试生成分析图表
        try:
            chart_path = analyzer._generate_analysis_chart(result['data'], blue_chip_stock)
            print(f"图表生成结果: {chart_path}")
        except Exception as e:
            print(f"生成图表时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 测试手动分析一批单独的股票
    test_stock_codes = ['601318.SH', '600036.SH', '000651.SZ', '000333.SZ', '600519.SH']
    print(f"\n测试直接分析多个股票: {test_stock_codes}")
    
    results = []
    for ts_code in test_stock_codes:
        result = test_single_stock(analyzer, ts_code)
        if result:
            results.append(result)
    
    print(f"\n分析结果数量: {len(results)}")
    for result in results:
        print(f"股票: {result['ts_code']}, 得分: {result['score']}")
    
    print(f"\n测试完成，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 