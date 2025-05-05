#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票分析系统启动脚本
初始化和测试数据源管理器
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from functools import wraps

# 确保src包可以被导入
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 创建必要目录
from src.enhanced.config.settings import LOG_DIR, ENHANCED_CACHE_DIR
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ENHANCED_CACHE_DIR, exist_ok=True)

# 配置日志
log_file = os.path.join(LOG_DIR, f"system_startup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 定义装饰器
def with_cache(method_name: str):
    """
    装饰器: 为方法添加缓存功能
    
    Args:
        method_name: 方法名
        
    Returns:
        callable: 装饰后的方法
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'cache_enabled') or not self.cache_enabled:
                return func(self, *args, **kwargs)
                
            # 从关键字参数中提取强制刷新标志
            force_refresh = kwargs.pop('force_refresh', False) if 'force_refresh' in kwargs else False
            
            # 生成缓存键
            if hasattr(self, '_get_cache_key'):
                cache_key = self._get_cache_key(method_name, *args, **kwargs)
                
                # 如果不需要强制刷新，尝试从缓存获取
                if not force_refresh and hasattr(self, '_get_from_cache'):
                    cached_data = self._get_from_cache(cache_key)
                    if cached_data is not None:
                        return cached_data
                
                # 执行原始方法
                result = func(self, *args, **kwargs)
                
                # 如果有结果，保存到缓存
                if result is not None and hasattr(self, '_save_to_cache'):
                    self._save_to_cache(cache_key, result)
                    
                return result
            else:
                # 如果没有缓存机制，直接执行原方法
                return func(self, *args, **kwargs)
        return wrapper
    return decorator

# 定义故障转移装饰器
def with_failover(method_name: str):
    """
    装饰器: 为方法添加故障转移逻辑
    
    Args:
        method_name: 数据源的方法名
        
    Returns:
        callable: 装饰后的方法
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 确保第一个参数是self
            instance = args[0]
            
            # 从关键字参数中提取首选数据源
            preferred_sources = kwargs.pop('preferred_sources', None)
            
            # 记录遇到的异常
            exceptions = {}
            
            # 如果提供了首选数据源，先尝试它们
            if preferred_sources:
                for source_name in preferred_sources:
                    if not instance.health_check(source_name):
                        logger.debug(f"跳过不健康的首选数据源: {source_name}")
                        continue
                    
                    data_source = instance.data_sources[source_name]
                    try:
                        logger.debug(f"尝试使用首选数据源 {source_name} 的 {method_name} 方法")
                        method = getattr(data_source, method_name)
                        return method(*args[1:], **kwargs)
                    except Exception as e:
                        exceptions[source_name] = str(e)
                        logger.warning(f"使用数据源 {source_name} 失败: {str(e)}")
                        # 标记为不健康
                        instance.health_status[source_name] = False
            
            # 尝试主数据源(如果它不在首选数据源中)
            if instance.primary_source and (not preferred_sources or instance.primary_source not in preferred_sources):
                if instance.health_check(instance.primary_source):
                    try:
                        logger.debug(f"尝试使用主数据源 {instance.primary_source} 的 {method_name} 方法")
                        method = getattr(instance.data_sources[instance.primary_source], method_name)
                        return method(*args[1:], **kwargs)
                    except Exception as e:
                        exceptions[instance.primary_source] = str(e)
                        logger.warning(f"使用主数据源 {instance.primary_source} 失败: {str(e)}")
                        # 标记为不健康
                        instance.health_status[instance.primary_source] = False
            
            # 尝试其他健康的数据源
            for source_name, data_source in instance.data_sources.items():
                # 跳过已尝试过的
                if (preferred_sources and source_name in preferred_sources) or source_name == instance.primary_source:
                    continue
                
                if not instance.health_check(source_name):
                    logger.debug(f"跳过不健康的数据源: {source_name}")
                    continue
                
                try:
                    logger.debug(f"尝试使用备用数据源 {source_name} 的 {method_name} 方法")
                    method = getattr(data_source, method_name)
                    return method(*args[1:], **kwargs)
                except Exception as e:
                    exceptions[source_name] = str(e)
                    logger.warning(f"使用数据源 {source_name} 失败: {str(e)}")
                    # 标记为不健康
                    instance.health_status[source_name] = False
            
            # 所有数据源都失败了
            logger.error(f"所有数据源的 {method_name} 方法都失败了")
            for source_name, error in exceptions.items():
                logger.error(f"  - {source_name}: {error}")
            
            return None
        
        return wrapper
    return decorator

# 在全局命名空间中定义装饰器名称，以便可以在导入前被使用
import builtins
builtins.with_cache = with_cache
builtins.with_failover = with_failover

# 导入配置和数据源管理器
from src.enhanced.data.fetchers.data_source_manager import DataSourceManager

def main():
    """主函数"""
    try:
        logger.info("==== 股票分析系统启动 ====")
        
        # 初始化数据源管理器
        logger.info("初始化数据源管理器...")
        data_source_manager = DataSourceManager()
        
        # 获取数据源状态
        sources_status = data_source_manager.get_data_sources_status()
        logger.info(f"数据源状态: {sources_status}")
        
        # 测试获取股票列表
        logger.info("测试获取股票列表...")
        stock_list = data_source_manager.get_stock_list()
        if stock_list is not None and not stock_list.empty:
            logger.info(f"成功获取股票列表，共 {len(stock_list)} 只股票")
            logger.info("\n示例数据:\n" + str(stock_list.head()))
        else:
            logger.warning("获取股票列表失败")
            
        # 测试获取行业列表
        logger.info("测试获取行业列表...")
        industry_list = data_source_manager.get_industry_list()
        if industry_list is not None and not industry_list.empty:
            logger.info(f"成功获取行业列表，共 {len(industry_list)} 个行业")
            logger.info("\n示例数据:\n" + str(industry_list.head()))
        else:
            logger.warning("获取行业列表失败")
            
        # 测试获取单只股票的日线数据
        stock_code = "000001.SZ"  # 平安银行
        start_date = "2024-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"测试获取股票 {stock_code} 的日线数据...")
        daily_data = data_source_manager.get_daily_data(stock_code, start_date, end_date)
        if daily_data is not None and not daily_data.empty:
            logger.info(f"成功获取股票 {stock_code} 的日线数据，共 {len(daily_data)} 条记录")
            logger.info("\n示例数据:\n" + str(daily_data.head()))
        else:
            logger.warning(f"获取股票 {stock_code} 的日线数据失败")
            
        logger.info("==== 股票分析系统启动完成 ====")
        
        # 返回数据源管理器实例以便可以在交互式环境中使用
        return data_source_manager
        
    except Exception as e:
        logger.error(f"系统启动出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    dsm = main() 