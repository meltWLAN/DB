#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
市场概览功能测试脚本
用于验证市场概览功能是否正常工作，并分析性能
"""

import logging
import time
import json
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_market_overview():
    """测试市场概览功能"""
    logger.info("=" * 50)
    logger.info("开始测试市场概览功能...")
    logger.info("=" * 50)
    
    try:
        # 导入数据源管理器
        from src.enhanced.data.fetchers.data_source_manager import DataSourceManager
        
        # 初始化数据源管理器
        logger.info("初始化DataSourceManager...")
        start_time = time.time()
        manager = DataSourceManager()
        init_time = time.time() - start_time
        logger.info(f"初始化耗时: {init_time:.2f}秒")
        
        # 获取数据源状态
        status = manager.get_data_sources_status()
        logger.info(f"可用数据源: {status}")
        
        # 获取最新交易日
        latest_date = manager.get_latest_trading_date()
        logger.info(f"最新交易日: {latest_date}")
        
        # 测试1: 获取最新市场概览
        logger.info("\n测试1: 获取最新市场概览...")
        start_time = time.time()
        market_overview = manager.get_market_overview()
        elapsed = time.time() - start_time
        
        if market_overview and isinstance(market_overview, dict):
            logger.info(f"成功获取市场概览，包含 {len(market_overview)} 个字段，耗时: {elapsed:.2f}秒")
            
            # 打印市场概览关键数据
            print("\n市场概览关键数据:")
            print(f"日期: {market_overview.get('date', 'N/A')}")
            
            if 'up_count' in market_overview and 'down_count' in market_overview:
                total = market_overview.get('up_count', 0) + market_overview.get('down_count', 0) + market_overview.get('flat_count', 0)
                up_ratio = market_overview.get('up_count', 0) / total * 100 if total > 0 else 0
                down_ratio = market_overview.get('down_count', 0) / total * 100 if total > 0 else 0
                
                print(f"涨跌家数: {market_overview.get('up_count', 0)}涨({up_ratio:.1f}%) / {market_overview.get('down_count', 0)}跌({down_ratio:.1f}%)")
            
            if 'limit_up_count' in market_overview and 'limit_down_count' in market_overview:
                print(f"涨停/跌停: {market_overview.get('limit_up_count', 0)}涨停 / {market_overview.get('limit_down_count', 0)}跌停")
            
            if 'avg_change_pct' in market_overview:
                print(f"平均涨跌幅: {market_overview.get('avg_change_pct', 0):.2f}%")
            
            if 'total_amount' in market_overview:
                amount_billion = market_overview.get('total_amount', 0) / 100000000  # 转换为亿元
                print(f"总成交额: {amount_billion:.2f}亿元")
            
            if 'limit_up_stocks' in market_overview and market_overview['limit_up_stocks']:
                print("\n部分涨停股:")
                for stock in market_overview['limit_up_stocks'][:5]:  # 只显示前5个
                    print(f"  {stock.get('code', 'N/A')} {stock.get('name', 'N/A')}")
            
            # 保存结果到文件
            with open('market_overview_result.json', 'w', encoding='utf-8') as f:
                json.dump(market_overview, f, ensure_ascii=False, indent=2)
            logger.info("市场概览结果已保存到 market_overview_result.json")
        else:
            logger.warning(f"获取市场概览失败或返回非字典类型: {type(market_overview)}")
        
        # 测试2: 测试连续获取性能(模拟缓存效果)
        logger.info("\n测试2: 测试连续获取性能...")
        start_time = time.time()
        market_overview2 = manager.get_market_overview()
        elapsed2 = time.time() - start_time
        
        logger.info(f"第二次获取市场概览耗时: {elapsed2:.2f}秒 (首次: {elapsed:.2f}秒)")
        if elapsed2 < elapsed:
            speedup = (elapsed - elapsed2) / elapsed * 100
            logger.info(f"性能提升: {speedup:.1f}%")
        
        # 测试3: 获取指定日期的市场概览
        if latest_date:
            # 获取上一个交易日
            previous_date = manager.get_previous_trading_date(latest_date)
            if previous_date:
                logger.info(f"\n测试3: 获取指定日期({previous_date})的市场概览...")
                start_time = time.time()
                previous_overview = manager.get_market_overview(previous_date)
                elapsed3 = time.time() - start_time
                
                if previous_overview and isinstance(previous_overview, dict):
                    logger.info(f"成功获取 {previous_date} 的市场概览，包含 {len(previous_overview)} 个字段，耗时: {elapsed3:.2f}秒")
                    
                    if 'up_count' in previous_overview and 'down_count' in previous_overview:
                        total = previous_overview.get('up_count', 0) + previous_overview.get('down_count', 0) + previous_overview.get('flat_count', 0)
                        up_ratio = previous_overview.get('up_count', 0) / total * 100 if total > 0 else 0
                        
                        print(f"\n{previous_date} 市场概览:")
                        print(f"涨跌家数: {previous_overview.get('up_count', 0)}涨({up_ratio:.1f}%) / {previous_overview.get('down_count', 0)}跌")
                else:
                    logger.warning(f"获取 {previous_date} 的市场概览失败")
        
        logger.info("\n=" * 50)
        logger.info("市场概览功能测试完成!")
        logger.info("=" * 50)
        
        # 返回成功标志
        return True
        
    except Exception as e:
        logger.error(f"测试市场概览功能失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_market_overview() 