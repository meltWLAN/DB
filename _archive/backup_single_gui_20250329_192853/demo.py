"""
数据增强系统演示脚本
展示系统的主要功能,包括:
- 数据获取和处理
- 数据质量检查
- 数据分析和预测
- 结果输出和可视化
"""

import logging
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.enhanced.config.settings import LOG_DIR
from src.enhanced.data.data_manager import EnhancedDataManager
from src.enhanced.data.quality.data_quality_checker import DataQualityChecker
from src.enhanced.strategies.stock_picker import StockPicker

# 配置日志
log_file = os.path.join(LOG_DIR, f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def demonstrate_data_processing():
    """演示数据处理功能"""
    try:
        logger.info("=== 开始演示数据处理功能 ===")
        
        # 初始化数据管理器
        data_manager = EnhancedDataManager()
        
        # 获取示例股票数据
        stock_code = "000001"  # 平安银行
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        logger.info(f"获取股票 {stock_code} 的历史数据...")
        df = data_manager.get_stock_data(stock_code, start_date, end_date)
        
        if df is not None and not df.empty:
            logger.info(f"成功获取数据,共 {len(df)} 条记录")
            logger.info("\n数据预览:")
            logger.info(df.head())
            
            # 计算基本统计信息
            logger.info("\n基本统计信息:")
            logger.info(df.describe())
            
            # 绘制价格走势图
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['close'])
            plt.title(f"{stock_code} 股价走势")
            plt.xlabel("日期")
            plt.ylabel("价格")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存图表
            output_dir = Path(LOG_DIR) / "plots"
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / f"price_trend_{stock_code}.png")
            plt.close()
            logger.info(f"价格走势图已保存到: {output_dir / f'price_trend_{stock_code}.png'}")
            
    except Exception as e:
        logger.error(f"数据处理演示出错: {str(e)}", exc_info=True)

def demonstrate_data_quality():
    """演示数据质量检查功能"""
    try:
        logger.info("\n=== 开始演示数据质量检查功能 ===")
        
        # 初始化数据质量检查器
        quality_checker = DataQualityChecker()
        
        # 获取示例数据
        data_manager = EnhancedDataManager()
        stock_code = "000001"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        df = data_manager.get_stock_data(stock_code, start_date, end_date)
        if df is None or df.empty:
            logger.error("无法获取数据用于质量检查")
            return
        
        # 执行数据质量检查
        logger.info("执行数据质量检查...")
        quality_report = quality_checker.check_data_quality(df)
        
        # 输出检查结果
        logger.info("\n数据质量检查报告:")
        for check_type, results in quality_report.items():
            logger.info(f"\n{check_type}:")
            for result in results:
                logger.info(f"  - {result}")
                
    except Exception as e:
        logger.error(f"数据质量检查演示出错: {str(e)}", exc_info=True)

def demonstrate_stock_picking():
    """演示股票选股功能"""
    try:
        logger.info("\n=== 开始演示股票选股功能 ===")
        
        # 初始化选股策略
        stock_picker = StockPicker()
        
        # 获取股票推荐
        logger.info("分析股票并获取推荐...")
        recommendations = stock_picker.get_stock_recommendations(top_n=5)
        
        if not recommendations:
            logger.warning("未找到符合条件的股票")
            return
        
        # 输出推荐结果
        logger.info("\n股票推荐列表:")
        for i, stock in enumerate(recommendations, 1):
            logger.info(f"\n{i}. 股票代码: {stock['stock_code']}")
            logger.info(f"   当前价格: {stock['current_price']:.2f}")
            logger.info(f"   涨跌幅: {stock['change_pct']*100:.2f}%")
            logger.info(f"   风险分数: {stock['risk_score']:.2f}")
            logger.info(f"   推荐理由: {stock['recommendation_reason']}")
            
    except Exception as e:
        logger.error(f"股票选股演示出错: {str(e)}", exc_info=True)

def main():
    """主函数"""
    try:
        logger.info("开始数据增强系统演示...")
        
        # 演示各个功能模块
        demonstrate_data_processing()
        demonstrate_data_quality()
        demonstrate_stock_picking()
        
        logger.info("\n演示完成!")
        
    except Exception as e:
        logger.error(f"演示程序运行出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 