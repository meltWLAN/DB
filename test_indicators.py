#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术指标计算测试脚本
专门用于测试动量分析模块中技术指标计算函数的有效性
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入动量分析模块
try:
    from momentum_analysis_enhanced_performance import EnhancedMomentumAnalyzer
    modules_loaded = True
except ImportError as e:
    logger.error(f"导入动量分析模块失败: {str(e)}")
    modules_loaded = False

def create_test_data():
    """创建测试用的股票数据"""
    # 创建日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # 创建价格数据
    n = len(dates)
    np.random.seed(42)  # 设置随机种子确保结果可重现
    
    # 生成一个随机上升趋势的价格
    base_price = 10
    prices = np.cumsum(np.random.normal(0.01, 0.1, n)) + base_price
    prices = np.maximum(prices, 0.1)  # 确保价格为正
    
    # 创建交易量数据
    volumes = np.random.normal(1000000, 200000, n)
    volumes = np.maximum(volumes, 100000)  # 确保交易量为正
    
    # 创建DataFrame
    df = pd.DataFrame({
        'trade_date': dates,
        'ts_code': 'TEST001',
        'open': prices * np.random.normal(0.99, 0.01, n),
        'high': prices * np.random.normal(1.02, 0.01, n),
        'low': prices * np.random.normal(0.98, 0.01, n),
        'close': prices,
        'pre_close': np.roll(prices, 1),
        'vol': volumes,
        'amount': volumes * prices / 1000,
    })
    
    # 设置第一天的pre_close
    df.loc[0, 'pre_close'] = df.loc[0, 'open'] * 0.99
    
    # 计算涨跌幅
    df['change'] = df['close'] - df['pre_close']
    df['pct_chg'] = (df['change'] / df['pre_close']) * 100
    
    # 设置索引
    df.set_index('trade_date', inplace=True)
    
    return df

def test_indicator_calculation():
    """测试技术指标计算功能"""
    if not modules_loaded:
        logger.error("无法加载必要的模块，测试终止")
        return False
    
    try:
        # 初始化分析器
        analyzer = EnhancedMomentumAnalyzer(use_tushare=False)
        
        logger.info("创建测试数据...")
        test_data = create_test_data()
        
        logger.info(f"测试数据创建成功，共 {len(test_data)} 条记录")
        logger.info(f"示例数据:\n{test_data.head()}")
        
        # 原始方法计算技术指标
        logger.info("使用原始方法计算技术指标...")
        try:
            data_original = test_data.copy()
            data_with_indicators = analyzer.calculate_momentum_indicators(data_original)
            
            # 检查技术指标是否已计算
            expected_columns = ['ma20', 'ma60', 'momentum_20', 'rsi']
            missing_columns = [col for col in expected_columns if col not in data_with_indicators.columns]
            
            if not missing_columns:
                logger.info("原始方法指标计算成功")
                logger.info(f"计算结果示例:\n{data_with_indicators[['close'] + expected_columns].tail(5)}")
            else:
                logger.warning(f"部分技术指标计算失败，缺少: {missing_columns}")
        except Exception as e:
            logger.error(f"原始方法计算指标失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 向量化方法计算技术指标
        logger.info("\n使用向量化方法计算技术指标...")
        try:
            data_vector = test_data.copy()
            vectorized_data = analyzer.calculate_momentum_vectorized(data_vector)
            
            # 检查技术指标是否已计算
            expected_columns = ['ma20', 'ma60', 'momentum_20', 'rsi', 'macd']
            missing_columns = [col for col in expected_columns if col not in vectorized_data.columns]
            
            if not missing_columns:
                logger.info("向量化方法指标计算成功")
                logger.info(f"计算结果示例:\n{vectorized_data[['close'] + expected_columns].tail(5)}")
                
                # 绘制指标图表
                plot_indicators(vectorized_data)
            else:
                logger.warning(f"部分向量化技术指标计算失败，缺少: {missing_columns}")
                
                # 检查DataFrame的列
                logger.info(f"现有列: {vectorized_data.columns.tolist()}")
        except Exception as e:
            logger.error(f"向量化方法计算指标失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 检查真实股票数据
        logger.info("\n测试真实股票数据...")
        try:
            # 获取股票列表
            stock_list = analyzer.get_stock_list()
            if stock_list is not None and not stock_list.empty:
                test_code = stock_list.iloc[0]['ts_code']
                logger.info(f"获取 {test_code} 的实际数据")
                
                # 获取股票数据
                real_data = analyzer.get_stock_daily_data(test_code)
                
                if real_data is not None and not real_data.empty:
                    logger.info(f"成功获取 {test_code} 的日线数据，共 {len(real_data)} 条记录")
                    
                    # 检查数据列
                    logger.info(f"数据列: {real_data.columns.tolist()}")
                    
                    # 检查是否有空值
                    null_counts = real_data.isnull().sum()
                    if null_counts.sum() > 0:
                        logger.warning(f"数据中存在空值:\n{null_counts[null_counts > 0]}")
                    
                    # 尝试计算技术指标
                    try:
                        logger.info("尝试计算技术指标...")
                        real_data_indicators = analyzer.calculate_momentum_vectorized(real_data)
                        
                        if not real_data_indicators.empty:
                            logger.info("技术指标计算成功")
                            # 检查指标
                            indicator_cols = [col for col in real_data_indicators.columns if col not in real_data.columns]
                            logger.info(f"计算的指标: {indicator_cols}")
                        else:
                            logger.warning("技术指标计算结果为空")
                    except Exception as e:
                        logger.error(f"计算实际数据的技术指标时出错: {str(e)}")
                else:
                    logger.warning(f"无法获取 {test_code} 的数据")
        except Exception as e:
            logger.error(f"测试实际数据时出错: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def plot_indicators(data):
    """绘制技术指标图表"""
    try:
        plt.figure(figsize=(14, 10))
        
        # 绘制价格和移动平均线
        plt.subplot(3, 1, 1)
        plt.plot(data.index, data['close'], label='Close Price')
        plt.plot(data.index, data['ma20'], label='MA20')
        plt.plot(data.index, data['ma60'], label='MA60')
        plt.title('Price and Moving Averages')
        plt.legend()
        plt.grid(True)
        
        # 绘制动量指标
        plt.subplot(3, 1, 2)
        plt.plot(data.index, data['momentum_20'], label='Momentum 20')
        plt.title('20-Day Momentum')
        plt.legend()
        plt.grid(True)
        
        # 绘制RSI
        plt.subplot(3, 1, 3)
        plt.plot(data.index, data['rsi'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
        plt.title('RSI')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('test_indicators_chart.png')
        logger.info("指标图表已保存为 test_indicators_chart.png")
    except Exception as e:
        logger.error(f"绘制图表时出错: {str(e)}")

if __name__ == "__main__":
    logger.info("开始测试技术指标计算...")
    test_indicator_calculation()
    logger.info("技术指标计算测试完成") 