#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动量指标计算修复工具
解决技术指标计算在实际数据上的问题
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os

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

def fix_momentum_indicators(data):
    """
    修复技术指标计算函数，使其适用于实际数据
    
    Args:
        data: 原始股票数据DataFrame
        
    Returns:
        DataFrame: 包含技术指标的数据
    """
    if data.empty:
        logger.warning("输入数据为空")
        return data
    
    try:
        # 检查数据格式
        logger.debug(f"输入数据形状: {data.shape}, 列: {data.columns.tolist()}")
        
        # 检查并处理索引
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'trade_date' in data.columns:
                logger.debug("将trade_date列设置为索引")
                data['trade_date'] = pd.to_datetime(data['trade_date'])
                data.set_index('trade_date', inplace=True)
            else:
                logger.warning("数据没有日期索引或trade_date列")
        
        # 必要的列
        required_cols = ['close', 'high', 'low', 'vol']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.error(f"数据缺少必要的列: {missing_cols}")
            return data
        
        # 手动计算技术指标
        # 1. 移动平均线
        data['ma5'] = data['close'].rolling(window=5).mean()
        data['ma10'] = data['close'].rolling(window=10).mean()
        data['ma20'] = data['close'].rolling(window=20).mean()
        data['ma60'] = data['close'].rolling(window=60).mean()
        
        # 2. 动量指标
        data['momentum_20'] = data['close'] / data['close'].shift(20) - 1
        
        # 3. RSI指标
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # 4. MACD指标
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # 5. 成交量比率
        data['vol_ma20'] = data['vol'].rolling(window=20).mean()
        data['vol_ratio_20'] = data['vol'] / data['vol_ma20']
        
        logger.info(f"技术指标计算完成，共计算 {len(data.columns) - len(required_cols)} 个指标")
        
        return data
    except Exception as e:
        logger.error(f"技术指标计算失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return data

def calculate_momentum_indicators_fixed(analyzer, data):
    """
    修复后的技术指标计算函数包装器
    
    Args:
        analyzer: 动量分析器实例
        data: 原始股票数据DataFrame
        
    Returns:
        DataFrame: 包含技术指标的数据
    """
    try:
        # 首先尝试原始向量化方法
        result = analyzer.calculate_momentum_vectorized(data.copy())
        
        # 检查结果是否有效
        if result is not None and not result.empty:
            # 检查是否计算了指标
            indicator_cols = ['ma20', 'ma60', 'momentum_20', 'rsi', 'macd']
            missing_indicators = [col for col in indicator_cols if col not in result.columns]
            
            if not missing_indicators:
                logger.info("原始向量化方法计算成功")
                return result
            
            logger.warning(f"原始向量化方法缺少指标: {missing_indicators}")
        
        # 如果原始方法失败，使用修复版本
        logger.info("使用修复版本计算技术指标")
        fixed_result = fix_momentum_indicators(data.copy())
        
        return fixed_result
    except Exception as e:
        logger.error(f"计算技术指标时出错: {str(e)}")
        # 返回带有基本指标的原始数据
        return fix_momentum_indicators(data.copy())

def patch_momentum_analyzer():
    """
    修补动量分析器，使技术指标计算更加稳健
    
    Returns:
        EnhancedMomentumAnalyzer: 修补后的分析器类
    """
    if not modules_loaded:
        logger.error("无法加载动量分析模块，无法修补")
        return None
    
    # 创建修补后的分析器类
    class PatchedMomentumAnalyzer(EnhancedMomentumAnalyzer):
        def calculate_momentum_vectorized(self, data):
            """重写向量化计算方法"""
            return calculate_momentum_indicators_fixed(super(), data)
        
        def analyze_single_stock_optimized(self, stock_data):
            """重写单只股票分析方法"""
            ts_code = stock_data['ts_code']
            name = stock_data['name']
            
            try:
                logger.info(f"正在分析: {name}({ts_code})")
                
                # 获取股票数据
                data = self.get_stock_daily_data(ts_code)
                if data.empty:
                    logger.warning(f"无法获取{ts_code}的数据，跳过分析")
                    return None
                
                # 使用修补后的技术指标计算函数
                data = calculate_momentum_indicators_fixed(self, data)
                
                if data.empty:
                    logger.warning(f"计算{ts_code}的技术指标失败，跳过分析")
                    return None
                
                # 继续使用原来的评分方法
                min_score = stock_data.get('min_score', 60)
                score, score_details = self.calculate_momentum_score_optimized(data)
                
                if score >= min_score:
                    # 获取最新数据
                    latest = data.iloc[-1]
                    
                    # 构建结果
                    result = {
                        'ts_code': ts_code,
                        'name': name,
                        'close': latest['close'],
                        'momentum_20': latest.get('momentum_20', 0),
                        'rsi': latest.get('rsi', 0),
                        'macd': latest.get('macd', 0),
                        'vol_ratio': latest.get('vol_ratio_20', 1),
                        'score': score,
                        'score_details': score_details,
                        'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'data': data
                    }
                    
                    return result
                return None
            except Exception as e:
                logger.error(f"分析{name}({ts_code})时出错: {str(e)}")
                return None
    
    logger.info("动量分析器已修补，技术指标计算更加稳健")
    return PatchedMomentumAnalyzer

def test_patched_analyzer():
    """测试修补后的分析器"""
    if not modules_loaded:
        logger.error("无法加载必要的模块，测试终止")
        return
    
    # 获取修补后的分析器类
    PatchedAnalyzer = patch_momentum_analyzer()
    
    # 创建修补后的分析器实例
    analyzer = PatchedAnalyzer(use_tushare=True, cache_size=50)
    
    # 获取股票列表
    stock_list = analyzer.get_stock_list()
    
    if stock_list is None or stock_list.empty:
        logger.error("获取股票列表失败")
        return
    
    # 测试单只股票分析
    test_stock = stock_list.iloc[0].to_dict()
    ts_code = test_stock['ts_code']
    name = test_stock['name']
    
    logger.info(f"测试分析 {name}({ts_code})")
    
    # 获取股票数据
    data = analyzer.get_stock_daily_data(ts_code)
    
    if data is None or data.empty:
        logger.error(f"无法获取 {ts_code} 的数据")
        return
    
    # 计算技术指标
    logger.info("计算技术指标...")
    data_with_indicators = analyzer.calculate_momentum_vectorized(data)
    
    # 验证指标计算结果
    indicators = ['ma20', 'ma60', 'momentum_20', 'rsi', 'macd']
    missing = [ind for ind in indicators if ind not in data_with_indicators.columns]
    
    if missing:
        logger.error(f"仍然缺少指标: {missing}")
    else:
        logger.info("所有指标计算成功!")
        
        # 绘制图表
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(data_with_indicators.index[-60:], data_with_indicators['close'][-60:], label='收盘价')
        plt.plot(data_with_indicators.index[-60:], data_with_indicators['ma20'][-60:], label='MA20')
        plt.plot(data_with_indicators.index[-60:], data_with_indicators['ma60'][-60:], label='MA60')
        plt.legend()
        plt.title(f"{name} ({ts_code}) - 价格和均线")
        
        plt.subplot(2, 1, 2)
        plt.plot(data_with_indicators.index[-60:], data_with_indicators['rsi'][-60:])
        plt.axhline(y=70, color='r', linestyle='--')
        plt.axhline(y=30, color='g', linestyle='--')
        plt.title('RSI指标')
        
        plt.tight_layout()
        plt.savefig(f"{ts_code}_indicators.png")
        logger.info(f"已保存技术指标图表到 {ts_code}_indicators.png")
    
    # 测试股票分析
    result = analyzer.analyze_single_stock_optimized(test_stock)
    
    if result is not None:
        logger.info(f"分析结果: 得分={result['score']}")
        logger.info(f"详细得分: {result['score_details']}")
    else:
        logger.warning("分析未返回结果，可能是得分未达到最低要求")
    
    # 测试批量分析
    logger.info("测试批量分析...")
    results = analyzer.analyze_stocks(stock_list.head(5), min_score=50)
    
    logger.info(f"批量分析完成，符合条件的股票数量: {len(results)}")
    if results:
        for i, res in enumerate(results):
            logger.info(f"结果 {i+1}: {res['name']} ({res['ts_code']}) - 得分: {res['score']}")

if __name__ == "__main__":
    logger.info("开始测试修补后的动量分析器...")
    test_patched_analyzer()
    logger.info("修补后的动量分析器测试完成") 