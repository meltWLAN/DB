#!/usr/bin/env python3
"""
数据源接口验证测试脚本
用于测试各个数据获取接口的参数和功能
"""
import os
import sys
import logging
import json
import traceback
from datetime import datetime, timedelta
import pandas as pd

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_interface(source_manager, interface_name, *args, **kwargs):
    """测试单个接口函数
    
    Args:
        source_manager: 数据源管理器实例
        interface_name: 接口函数名
        *args, **kwargs: 传递给接口的参数
        
    Returns:
        bool: 测试结果，成功返回True，失败返回False
    """
    try:
        logger.info(f"测试接口: {interface_name}, 参数: {args}, {kwargs}")
        
        # 获取接口方法
        method = getattr(source_manager, interface_name)
        
        # 调用接口
        start_time = datetime.now()
        result = method(*args, **kwargs)
        end_time = datetime.now()
        
        # 记录执行时间
        execution_time = (end_time - start_time).total_seconds()
        
        # 检查结果
        if result is None:
            logger.warning(f"接口 {interface_name} 返回None，耗时: {execution_time:.2f}秒")
            return False
            
        if isinstance(result, pd.DataFrame):
            if result.empty:
                logger.warning(f"接口 {interface_name} 返回空DataFrame，耗时: {execution_time:.2f}秒")
                return False
            else:
                row_count = len(result)
                col_count = len(result.columns)
                logger.info(f"接口 {interface_name} 返回DataFrame: {row_count}行 x {col_count}列，耗时: {execution_time:.2f}秒")
                logger.info(f"列名: {list(result.columns)}")
                # 显示前3行数据
                if row_count > 0:
                    logger.info(f"数据样例(前3行):\n{result.head(3)}")
        elif isinstance(result, dict):
            key_count = len(result)
            logger.info(f"接口 {interface_name} 返回字典: {key_count}个键，耗时: {execution_time:.2f}秒")
            logger.info(f"键名: {list(result.keys())}")
        elif isinstance(result, list):
            item_count = len(result)
            logger.info(f"接口 {interface_name} 返回列表: {item_count}个元素，耗时: {execution_time:.2f}秒")
            if item_count > 0:
                logger.info(f"第一个元素类型: {type(result[0])}")
                
        return True
            
    except Exception as e:
        logger.error(f"测试接口 {interface_name} 失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False
        
def test_tushare_fetcher():
    """测试TuShare数据获取器"""
    logger.info("=" * 50)
    logger.info("测试TuShare数据获取器...")
    
    try:
        from src.enhanced.data.fetchers.tushare_fetcher import EnhancedTushareFetcher
        from src.enhanced.config.settings import DATA_SOURCE_CONFIG
        
        # 初始化TuShare获取器
        config = DATA_SOURCE_CONFIG.get('tushare', {})
        tushare_fetcher = EnhancedTushareFetcher(config)
        
        # 测试健康检查
        logger.info("测试健康检查...")
        health_status = tushare_fetcher.check_health()
        logger.info(f"TuShare健康状态: {health_status}")
        
        # 测试获取股票列表
        test_interface(tushare_fetcher, "get_stock_list")
        
        # 测试获取日线数据
        today = datetime.now()
        one_month_ago = (today - timedelta(days=30)).strftime('%Y-%m-%d')
        today_str = today.strftime('%Y-%m-%d')
        test_interface(tushare_fetcher, "get_daily_data", "000001.SZ", one_month_ago, today_str)
        
        # 测试获取指数数据
        test_interface(tushare_fetcher, "get_stock_index_data", "000001.SH", one_month_ago, today_str)
        
        # 测试获取行业列表
        test_interface(tushare_fetcher, "get_industry_list")
        
        # 测试获取市场概览
        test_interface(tushare_fetcher, "get_market_overview", today_str)
        
        logger.info("TuShare数据获取器测试完成")
        return True
        
    except Exception as e:
        logger.error(f"测试TuShare数据获取器失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False
        
def test_akshare_fetcher():
    """测试AKShare数据获取器"""
    logger.info("=" * 50)
    logger.info("测试AKShare数据获取器...")
    
    try:
        from src.enhanced.data.fetchers.akshare_fetcher import EnhancedAKShareFetcher
        from src.enhanced.config.settings import DATA_SOURCE_CONFIG
        
        # 初始化AKShare获取器
        config = DATA_SOURCE_CONFIG.get('akshare', {})
        akshare_fetcher = EnhancedAKShareFetcher(config)
        
        # 测试健康检查
        logger.info("测试健康检查...")
        health_status = akshare_fetcher.check_health()
        logger.info(f"AKShare健康状态: {health_status}")
        
        # 测试获取股票列表
        test_interface(akshare_fetcher, "get_stock_list")
        
        # 测试获取日线数据
        today = datetime.now()
        one_month_ago = (today - timedelta(days=30)).strftime('%Y-%m-%d')
        today_str = today.strftime('%Y-%m-%d')
        test_interface(akshare_fetcher, "get_daily_data", "000001.SZ", one_month_ago, today_str)
        
        # 测试获取指数数据
        test_interface(akshare_fetcher, "get_stock_index_data", "000001.SH", one_month_ago, today_str)
        
        # 测试获取行业列表
        test_interface(akshare_fetcher, "get_industry_list")
        
        # 测试获取市场概览
        test_interface(akshare_fetcher, "get_market_overview", today_str)
        
        logger.info("AKShare数据获取器测试完成")
        return True
        
    except Exception as e:
        logger.error(f"测试AKShare数据获取器失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False
        
def test_data_source_manager():
    """测试数据源管理器"""
    logger.info("=" * 50)
    logger.info("测试数据源管理器...")
    
    try:
        from src.enhanced.data.fetchers.data_source_manager import DataSourceManager
        
        # 初始化数据源管理器
        data_manager = DataSourceManager()
        
        # 获取可用数据源
        available_sources = data_manager.available_sources
        logger.info(f"可用数据源: {available_sources}")
        
        # 测试获取最新交易日期
        latest_date = data_manager.get_latest_trading_date()
        logger.info(f"最新交易日期: {latest_date}")
        
        # 如果无法获取最新交易日期，使用当前日期
        today = datetime.now()
        if latest_date is None:
            latest_date = today.strftime('%Y-%m-%d')
            
        # 获取前一个交易日
        prev_date = data_manager.get_previous_trading_date(latest_date, 1)
        logger.info(f"前一个交易日: {prev_date}")
        
        # 获取前30个交易日
        start_date = data_manager.get_previous_trading_date(latest_date, 30)
        if start_date is None:
            start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # 测试获取股票列表
        test_interface(data_manager, "get_stock_list")
        
        # 测试获取日线数据
        test_interface(data_manager, "get_daily_data", "000001.SZ", start_date, latest_date)
        
        # 测试获取指数数据
        test_interface(data_manager, "get_stock_index_data", "000001.SH", start_date, latest_date)
        
        # 测试获取行业列表
        test_interface(data_manager, "get_industry_list")
        
        # 测试获取行业股票
        # 先获取行业列表
        industry_list = data_manager.get_industry_list()
        if industry_list is not None and not industry_list.empty:
            first_industry = industry_list.iloc[0]
            industry_code = first_industry.get('industry_code', '')
            if industry_code:
                test_interface(data_manager, "get_industry_stocks", industry_code)
        
        # 测试获取行业表现
        test_interface(data_manager, "get_industry_performance", latest_date)
        
        # 测试获取市场概览
        test_interface(data_manager, "get_market_overview", latest_date)
        
        # 测试获取全部股票当日数据
        test_interface(data_manager, "get_all_stock_data_on_date", latest_date)
        
        logger.info("数据源管理器测试完成")
        return True
        
    except Exception as e:
        logger.error(f"测试数据源管理器失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """主测试函数"""
    logger.info("开始数据源接口验证测试...")
    
    # 测试TuShare数据获取器
    tushare_result = test_tushare_fetcher()
    
    # 测试AKShare数据获取器
    akshare_result = test_akshare_fetcher()
    
    # 测试数据源管理器
    manager_result = test_data_source_manager()
    
    # 输出总结
    logger.info("=" * 50)
    logger.info("数据源接口验证测试结果总结:")
    logger.info(f"TuShare数据获取器: {'成功' if tushare_result else '失败'}")
    logger.info(f"AKShare数据获取器: {'成功' if akshare_result else '失败'}")
    logger.info(f"数据源管理器: {'成功' if manager_result else '失败'}")
    
    success_count = sum([tushare_result, akshare_result, manager_result])
    total_count = 3
    success_rate = success_count / total_count * 100
    
    logger.info(f"总体测试成功率: {success_rate:.2f}% ({success_count}/{total_count})")
    
    # 创建日志文件
    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"data_source_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    return success_count == total_count

if __name__ == "__main__":
    main() 