#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票选股策略主程序
"""

import logging
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.enhanced.config.settings import LOG_DIR
from src.enhanced.strategies.stock_picker import StockPicker

# 配置日志
log_file = os.path.join(LOG_DIR, f"stock_picker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """主函数"""
    try:
        logger.info("正在启动股票选股策略...")
        
        # 初始化选股策略
        stock_picker = StockPicker()
        
        # 获取股票推荐
        logger.info("正在分析股票...")
        recommendations = stock_picker.get_stock_recommendations(top_n=10)
        
        if not recommendations:
            logger.warning("未找到符合条件的股票")
            return
        
        # 输出推荐结果
        logger.info("\n=== 股票推荐列表 ===")
        for i, stock in enumerate(recommendations, 1):
            logger.info(f"\n{i}. 股票代码: {stock['stock_code']}")
            logger.info(f"   当前价格: {stock['current_price']:.2f}")
            logger.info(f"   涨跌幅: {stock['change_pct']*100:.2f}%")
            logger.info(f"   成交量: {stock['volume']:.0f}")
            logger.info(f"   风险分数: {stock['risk_score']:.2f}")
            logger.info(f"   推荐理由: {stock['recommendation_reason']}")
        
        # 保存结果到CSV文件
        output_file = os.path.join(LOG_DIR, f"stock_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df = pd.DataFrame(recommendations)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"\n推荐结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 