#!/usr/bin/env python3
"""
测试单只股票的数据获取和动量分析功能
"""
import logging
from datetime import datetime
from enhanced_momentum_analysis import EnhancedMomentumAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_single_stock")

def test_single_stock(ts_code='000001.SZ'):
    """测试单只股票的分析功能"""
    logger.info(f"开始测试股票: {ts_code}")
    
    # 初始化分析器
    analyzer = EnhancedMomentumAnalyzer(use_tushare=True)
    
    # 获取股票数据
    logger.info(f"获取股票数据...")
    data = analyzer.get_stock_daily_data(ts_code)
    
    if data is None or data.empty:
        logger.error(f"获取股票数据失败")
        return False
    
    logger.info(f"获取到数据，条数: {len(data)}")
    logger.info(f"数据时间范围: {data.index.min()} 至 {data.index.max()}")
    
    # 打印原始数据的样本
    logger.info(f"原始数据示例: \n{data.head(2)}")
    
    # 计算动量和技术指标
    logger.info(f"计算技术指标...")
    data_with_indicators = analyzer.calculate_momentum(data)
    
    if data_with_indicators is None or data_with_indicators.empty:
        logger.error(f"计算技术指标失败")
        return False
    
    # 打印计算后的数据样本
    added_columns = set(data_with_indicators.columns) - set(data.columns)
    logger.info(f"添加的技术指标列: {added_columns}")
    
    # 计算动量得分
    logger.info(f"计算动量得分...")
    score, details = analyzer.calculate_momentum_score(data_with_indicators)
    
    logger.info(f"动量分析得分: {score}")
    logger.info(f"得分详情: {details}")
    
    # 获取最新数据
    latest = data_with_indicators.iloc[-1]
    logger.info(f"最新日期: {latest.name}, 收盘价: {latest['close']}")
    
    # 主要技术指标
    logger.info(f"主要技术指标:")
    logger.info(f"  RSI: {latest.get('rsi', 0):.2f}")
    logger.info(f"  MACD: {latest.get('macd', 0):.4f}")
    logger.info(f"  MACD信号线: {latest.get('signal', 0):.4f}")
    logger.info(f"  MACD柱状图: {latest.get('macd_hist', 0):.4f}")
    logger.info(f"  20日动量: {latest.get('momentum_20', 0):.2f}%")
    
    return True

if __name__ == "__main__":
    test_stocks = ['000001.SZ', '600000.SH', '300059.SZ']
    
    for stock in test_stocks:
        print(f"\n{'=' * 50}")
        test_single_stock(stock)
        print(f"{'=' * 50}\n") 