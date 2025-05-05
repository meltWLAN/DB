#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动量分析模块完整修复工具
解决技术指标计算和评分系统问题
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
import json

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
        
        # 2. 动量指标 (5日和20日)
        data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
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
        data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['signal']
        
        # 5. 成交量比率
        data['vol_ma20'] = data['vol'].rolling(window=20).mean()
        data['vol_ratio_5'] = data['vol'] / data['vol'].rolling(window=5).mean()
        data['vol_ratio_20'] = data['vol'] / data['vol_ma20']
        
        # 6. 趋势强度指标
        data['trend_20'] = (data['close'] - data['ma20']) / data['ma20']
        
        logger.info(f"技术指标计算完成，共计算 {len(data.columns) - len(required_cols)} 个指标")
        
        return data
    except Exception as e:
        logger.error(f"技术指标计算失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return data

def calculate_momentum_score_fixed(data):
    """
    修复的动量评分计算方法
    
    Args:
        data: 包含技术指标的股票数据
    
    Returns:
        tuple: (总得分, 各项得分明细字典)
    """
    if data.empty or len(data) < 60:
        return 0, {}
    
    try:
        # 获取最新数据
        latest = data.iloc[-1]
        
        # 初始化得分详情
        score_details = {}
        
        # 1. 价格动量得分 (0-25分)
        try:
            momentum_5 = latest.get('momentum_5', 0)
            momentum_20 = latest.get('momentum_20', 0)
            
            # 短期动量
            if momentum_5 > 0.05:
                score_momentum_5 = 15
            elif momentum_5 > 0.02:
                score_momentum_5 = 10
            elif momentum_5 > 0:
                score_momentum_5 = 5
            else:
                score_momentum_5 = 0
            
            # 长期动量
            if momentum_20 > 0.15:
                score_momentum_20 = 10
            elif momentum_20 > 0.08:
                score_momentum_20 = 7
            elif momentum_20 > 0:
                score_momentum_20 = 5
            else:
                score_momentum_20 = 0
            
            # 总动量得分
            momentum_score = score_momentum_5 + score_momentum_20
            score_details['momentum'] = momentum_score
        except Exception as e:
            logger.error(f"计算价格动量得分失败: {str(e)}")
            momentum_score = 0
            score_details['momentum'] = 0
        
        # 2. RSI得分 (0-20分)
        try:
            rsi = latest.get('rsi', 50)
            
            if rsi > 70:
                rsi_score = 20
            elif rsi > 60:
                rsi_score = 15
            elif rsi > 50:
                rsi_score = 10
            elif rsi > 40:
                rsi_score = 5
            else:
                rsi_score = 0
                
            score_details['rsi'] = rsi_score
        except Exception as e:
            logger.error(f"计算RSI得分失败: {str(e)}")
            rsi_score = 0
            score_details['rsi'] = 0
        
        # 3. MACD得分 (0-20分)
        try:
            macd = latest.get('macd', 0)
            macd_signal = latest.get('signal', 0)
            macd_hist = latest.get('macd_hist', 0)
            
            # MACD值为正
            if macd > 0:
                macd_value_score = 10
            else:
                macd_value_score = 0
            
            # MACD柱状图为正且上升
            prev_hist = data.iloc[-2].get('macd_hist', 0)
            if macd_hist > 0 and macd_hist > prev_hist:
                macd_hist_score = 10
            elif macd_hist > 0:
                macd_hist_score = 5
            else:
                macd_hist_score = 0
            
            # 总MACD得分
            macd_score = macd_value_score + macd_hist_score
            score_details['macd'] = macd_score
        except Exception as e:
            logger.error(f"计算MACD指标得分失败: {str(e)}")
            macd_score = 0
            score_details['macd'] = 0
        
        # 4. 移动平均线得分 (0-15分)
        try:
            close = latest.get('close', 0)
            ma20 = latest.get('ma20', 0)
            ma60 = latest.get('ma60', 0)
            
            # 收盘价相对于20日均线
            if close > ma20 * 1.05:
                ma20_score = 8
            elif close > ma20:
                ma20_score = 5
            else:
                ma20_score = 0
            
            # 20日均线相对于60日均线
            if ma20 > ma60 * 1.03:
                ma_trend_score = 7
            elif ma20 > ma60:
                ma_trend_score = 4
            else:
                ma_trend_score = 0
            
            # 总均线得分
            ma_score = ma20_score + ma_trend_score
            score_details['ma'] = ma_score
        except Exception as e:
            logger.error(f"计算均线得分失败: {str(e)}")
            ma_score = 0
            score_details['ma'] = 0
        
        # 5. 成交量得分 (0-20分)
        try:
            vol_ratio_5 = latest.get('vol_ratio_5', 1)
            vol_ratio_20 = latest.get('vol_ratio_20', 1)
            
            # 5日成交量比
            if vol_ratio_5 > 1.5:
                vol_short_score = 10
            elif vol_ratio_5 > 1.2:
                vol_short_score = 7
            elif vol_ratio_5 > 1:
                vol_short_score = 5
            else:
                vol_short_score = 0
            
            # 20日成交量比
            if vol_ratio_20 > 1.3:
                vol_long_score = 10
            elif vol_ratio_20 > 1.1:
                vol_long_score = 7
            elif vol_ratio_20 > 1:
                vol_long_score = 5
            else:
                vol_long_score = 0
            
            # 总成交量得分
            vol_score = max(vol_short_score, vol_long_score)
            score_details['volume'] = vol_score
        except Exception as e:
            logger.error(f"计算成交量变化得分失败: {str(e)}")
            vol_score = 0
            score_details['volume'] = 0
        
        # 计算总分
        total_score = momentum_score + rsi_score + macd_score + ma_score + vol_score
        
        # 返回总分和详细得分
        return total_score, score_details
        
    except Exception as e:
        logger.error(f"计算动量得分失败: {str(e)}")
        return 0, {}

def patch_momentum_analyzer():
    """
    修补动量分析器，使技术指标计算和评分系统更加稳健
    
    Returns:
        EnhancedMomentumAnalyzer: 修补后的分析器类
    """
    if not modules_loaded:
        logger.error("无法加载动量分析模块，无法修补")
        return None
    
    # 创建修补后的分析器类
    class FixedMomentumAnalyzer(EnhancedMomentumAnalyzer):
        def calculate_momentum_vectorized(self, data):
            """重写向量化计算方法"""
            try:
                # 首先尝试原始向量化方法
                result = super().calculate_momentum_vectorized(data.copy())
                
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
        
        def calculate_momentum_score_optimized(self, data):
            """重写评分方法"""
            return calculate_momentum_score_fixed(data)
        
        def analyze_single_stock_optimized(self, stock_data):
            """重写单只股票分析方法"""
            ts_code = stock_data['ts_code']
            name = stock_data['name']
            industry = stock_data.get('industry', '')
            
            try:
                logger.info(f"正在分析: {name}({ts_code})")
                
                # 获取股票数据
                data = self.get_stock_daily_data(ts_code)
                if data is None or data.empty:
                    logger.warning(f"无法获取{ts_code}的数据，跳过分析")
                    return None
                
                # 计算技术指标
                data = self.calculate_momentum_vectorized(data)
                if data is None or data.empty:
                    logger.warning(f"计算{ts_code}的技术指标失败，跳过分析")
                    return None
                
                # 计算评分
                min_score = stock_data.get('min_score', 60)
                score, score_details = self.calculate_momentum_score_optimized(data)
                
                logger.info(f"{name}({ts_code}) 得分: {score}, 阈值: {min_score}")
                
                if score >= min_score:
                    # 获取最新数据
                    latest = data.iloc[-1]
                    
                    # 构建结果
                    result = {
                        'ts_code': ts_code,
                        'name': name,
                        'industry': industry,
                        'close': latest['close'],
                        'momentum_20': latest.get('momentum_20', 0) * 100,  # 转换为百分比
                        'rsi': latest.get('rsi', 0),
                        'macd': latest.get('macd', 0),
                        'vol_ratio': latest.get('vol_ratio_20', 1),
                        'score': score,
                        'score_details': score_details,
                        'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    }
                    
                    logger.info(f"{name}({ts_code}) 符合条件，得分: {score}")
                    return result
                
                logger.info(f"{name}({ts_code}) 不符合条件，得分: {score}")
                return None
            except Exception as e:
                logger.error(f"分析{name}({ts_code})时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
    
    logger.info("动量分析器已修补，技术指标计算和评分系统更加稳健")
    return FixedMomentumAnalyzer

def test_fixed_analyzer():
    """测试修补后的分析器"""
    if not modules_loaded:
        logger.error("无法加载必要的模块，测试终止")
        return
    
    # 获取修补后的分析器类
    FixedAnalyzer = patch_momentum_analyzer()
    
    # 创建修补后的分析器实例
    analyzer = FixedAnalyzer(use_tushare=True, cache_size=50)
    
    # 获取股票列表
    stock_list = analyzer.get_stock_list()
    
    if stock_list is None or stock_list.empty:
        logger.error("获取股票列表失败")
        return
    
    # 创建结果目录
    results_dir = "momentum_results"
    os.makedirs(results_dir, exist_ok=True)
    
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
        plt.figure(figsize=(12, 12))
        
        # 绘制价格和移动平均线
        plt.subplot(3, 1, 1)
        plt.plot(data_with_indicators.index[-60:], data_with_indicators['close'][-60:], label='收盘价')
        plt.plot(data_with_indicators.index[-60:], data_with_indicators['ma20'][-60:], label='MA20')
        plt.plot(data_with_indicators.index[-60:], data_with_indicators['ma60'][-60:], label='MA60')
        plt.legend()
        plt.grid(True)
        plt.title(f"{name} ({ts_code}) - 价格和均线")
        
        # 绘制MACD
        plt.subplot(3, 1, 2)
        plt.plot(data_with_indicators.index[-60:], data_with_indicators['macd'][-60:], label='MACD')
        plt.plot(data_with_indicators.index[-60:], data_with_indicators['signal'][-60:], label='Signal')
        plt.bar(data_with_indicators.index[-60:], data_with_indicators['macd_hist'][-60:], label='Histogram')
        plt.legend()
        plt.grid(True)
        plt.title('MACD指标')
        
        # 绘制RSI
        plt.subplot(3, 1, 3)
        plt.plot(data_with_indicators.index[-60:], data_with_indicators['rsi'][-60:])
        plt.axhline(y=70, color='r', linestyle='--')
        plt.axhline(y=30, color='g', linestyle='--')
        plt.grid(True)
        plt.title('RSI指标')
        
        plt.tight_layout()
        chart_path = os.path.join(results_dir, f"{ts_code}_indicators.png")
        plt.savefig(chart_path)
        logger.info(f"已保存技术指标图表到 {chart_path}")
    
    # 测试评分系统
    score, details = analyzer.calculate_momentum_score_optimized(data_with_indicators)
    logger.info(f"动量评分: {score}")
    logger.info(f"详细得分: {details}")
    
    # 降低分数阈值以便观察结果
    test_stock['min_score'] = 30
    
    # 测试单只股票分析
    result = analyzer.analyze_single_stock_optimized(test_stock)
    
    if result is not None:
        logger.info(f"分析结果: 得分={result['score']}")
        logger.info(f"指标: 动量={result['momentum_20']:.2f}%, RSI={result['rsi']:.2f}")
        # 保存结果
        with open(os.path.join(results_dir, f"{ts_code}_analysis.json"), 'w', encoding='utf-8') as f:
            # 创建可序列化的结果
            serializable_result = {k: v for k, v in result.items() if k != 'data'}
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
    else:
        logger.warning("分析未返回结果，得分可能未达到要求")
    
    # 测试批量分析
    logger.info("测试批量分析(5只股票，分数阈值=30)...")
    # 为所有股票设置较低阈值以便于观察结果
    temp_stocks = stock_list.head(5).copy()
    temp_stocks['min_score'] = 30
    
    results = analyzer.analyze_stocks(temp_stocks, min_score=30)
    
    logger.info(f"批量分析完成，符合条件的股票数量: {len(results)}")
    if results:
        # 保存结果到CSV
        result_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'data' and k != 'score_details'}
                                for r in results])
        csv_path = os.path.join(results_dir, "momentum_analysis_results.csv")
        result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"已保存结果到 {csv_path}")
        
        # 显示结果
        for i, res in enumerate(results):
            logger.info(f"结果 {i+1}: {res['name']} ({res['ts_code']}) - 得分: {res['score']}")

def main():
    """主函数"""
    logger.info("开始执行动量分析模块完整修复...")
    
    # 测试修补后的分析器
    test_fixed_analyzer()
    
    logger.info("动量分析模块完整修复测试完成")
    
    # 创建使用说明
    logger.info("\n使用说明:")
    logger.info("1. 在您的代码中导入FixedMomentumAnalyzer:")
    logger.info("   from momentum_fix_complete import patch_momentum_analyzer")
    logger.info("   FixedMomentumAnalyzer = patch_momentum_analyzer()")
    logger.info("2. 创建分析器实例:")
    logger.info("   analyzer = FixedMomentumAnalyzer()")
    logger.info("3. 使用分析器分析股票:")
    logger.info("   results = analyzer.analyze_stocks(stock_list)")

if __name__ == "__main__":
    main() 