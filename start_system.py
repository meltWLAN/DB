#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
系统启动脚本
初始化并启动增强动量分析系统
"""

import os
import logging
from datetime import datetime
from enhanced_momentum_analysis import EnhancedMomentumAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"system_start_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8')
    ]
)

logger = logging.getLogger("system_starter")

def main():
    """主函数"""
    try:
        logger.info("开始初始化增强动量分析系统")
        
        # 创建必要的目录
        for dir_path in ["logs", "data", "results", "results/enhanced_charts"]:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"确保目录存在: {dir_path}")
        
        # 初始化增强动量分析器
        analyzer = EnhancedMomentumAnalyzer(use_tushare=True)
        logger.info("成功初始化增强动量分析器")
        
        # 获取股票列表
        stocks = analyzer.get_stock_list()
        if stocks is not None and len(stocks) > 0:
            logger.info(f"成功获取股票列表，共 {len(stocks)} 支股票")
            
            # 测试分析一支股票
            test_stock = stocks.iloc[0]
            logger.info(f"测试分析第一支股票: {test_stock['name']}({test_stock['ts_code']})")
            
            results = analyzer.analyze_stocks_enhanced([test_stock], min_score=50)
            if results:
                logger.info("系统测试成功，可以开始使用")
            else:
                logger.warning("系统测试未返回结果，但初始化完成")
        else:
            logger.warning("获取股票列表失败，但系统已初始化")
        
        logger.info("系统启动完成")
        return analyzer
        
    except Exception as e:
        logger.error(f"系统启动过程中发生错误: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 