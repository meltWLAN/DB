#!/usr/bin/env python3
"""
增强版动量分析模块验证测试
使用最先进的方法验证测试动量分析模块，确保其准确性和可靠性
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import pytest
from typing import Dict, List, Optional, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("momentum_validation")

# 导入增强版动量分析器
from enhanced_momentum_analysis import EnhancedMomentumAnalyzer

class MomentumValidationTester:
    """动量分析验证测试器"""
    
    def __init__(self):
        self.analyzer = EnhancedMomentumAnalyzer(use_tushare=True)
        self.test_stocks = ['000001.SZ', '600000.SH', '300059.SZ']  # 测试股票列表
        self.test_periods = [5, 10, 20, 30, 60]  # 测试周期
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """验证数据质量
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        if df is None or df.empty:
            logger.error("数据为空")
            return False
            
        # 检查必要列是否存在
        required_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"缺少必要列: {missing_columns}")
            return False
            
        # 检查数据完整性
        if df.isnull().any().any():
            logger.warning("数据中存在空值")
            return False
            
        # 检查数据范围
        if (df['close'] <= 0).any():
            logger.error("存在无效的收盘价")
            return False
            
        return True
    
    def test_technical_indicators(self) -> bool:
        """测试技术指标计算"""
        logger.info("=== 测试技术指标计算 ===")
        
        for ts_code in self.test_stocks:
            logger.info(f"测试股票: {ts_code}")
            
            # 获取股票数据
            df = self.analyzer.get_stock_data(ts_code, days=60)
            if not self.validate_data_quality(df):
                continue
                
            # 计算技术指标
            try:
                df_with_indicators = self.analyzer.calculate_technical_indicators(df)
                
                # 验证技术指标
                required_indicators = ['MA5', 'MA10', 'MA20', 'RSI', 'MACD']
                for indicator in required_indicators:
                    if indicator not in df_with_indicators.columns:
                        logger.error(f"缺少技术指标: {indicator}")
                        return False
                        
                # 验证指标值范围
                if not (0 <= df_with_indicators['RSI'].max() <= 100):
                    logger.error("RSI值超出有效范围")
                    return False
                    
                logger.info(f"股票 {ts_code} 技术指标测试通过")
                
            except Exception as e:
                logger.error(f"计算技术指标时出错: {str(e)}")
                return False
                
        return True
    
    def test_momentum_scoring(self) -> bool:
        """测试动量评分系统"""
        logger.info("=== 测试动量评分系统 ===")
        
        for ts_code in self.test_stocks:
            logger.info(f"测试股票: {ts_code}")
            
            try:
                score = self.analyzer.calculate_enhanced_momentum_score(ts_code)
                
                # 验证评分范围
                if not (0 <= score <= 100):
                    logger.error(f"动量评分超出有效范围: {score}")
                    return False
                    
                logger.info(f"股票 {ts_code} 动量评分: {score}")
                
            except Exception as e:
                logger.error(f"计算动量评分时出错: {str(e)}")
                return False
                
        return True
    
    def test_industry_analysis(self) -> bool:
        """测试行业分析功能"""
        logger.info("=== 测试行业分析功能 ===")
        
        for ts_code in self.test_stocks:
            logger.info(f"测试股票: {ts_code}")
            
            try:
                # 获取行业信息
                industry = self.analyzer.get_stock_industry(ts_code)
                if not industry:
                    logger.error(f"无法获取股票 {ts_code} 的行业信息")
                    return False
                    
                # 分析行业动量
                industry_momentum = self.analyzer.analyze_industry_momentum(ts_code)
                if industry_momentum is None:
                    logger.error(f"行业动量分析失败: {ts_code}")
                    return False
                    
                logger.info(f"股票 {ts_code} 行业分析测试通过")
                
            except Exception as e:
                logger.error(f"行业分析时出错: {str(e)}")
                return False
                
        return True
    
    def run_all_tests(self) -> bool:
        """运行所有测试"""
        logger.info("开始运行动量分析模块验证测试")
        
        tests = [
            self.test_technical_indicators,
            self.test_momentum_scoring,
            self.test_industry_analysis
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                logger.info(f"测试 {test.__name__} {'通过' if result else '失败'}")
            except Exception as e:
                logger.error(f"测试 {test.__name__} 执行出错: {str(e)}")
                results.append(False)
        
        return all(results)

def main():
    """主函数"""
    tester = MomentumValidationTester()
    success = tester.run_all_tests()
    
    if success:
        logger.info("所有测试通过！动量分析模块验证成功")
    else:
        logger.error("测试失败！请检查日志了解详细信息")
        
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 