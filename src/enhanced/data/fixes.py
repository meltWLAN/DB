#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据源修复模块
在系统启动时自动应用所有数据源修复
"""

import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

def apply_data_source_fixes():
    """应用数据源修复"""
    logger.info("应用数据源修复...")
    
    try:
        # 1. 修复TuShare数据接口
        from src.enhanced.data.fetchers.tushare_fetcher import EnhancedTushareFetcher
        
        # 保存原始实现
        original_get_stock_index_data = EnhancedTushareFetcher.get_stock_index_data
        
        # 定义修复后的方法
        def fixed_get_stock_index_data(self, index_code, start_date=None, end_date=None):
            """修复的获取指数数据方法"""
            try:
                # 标准化指数代码
                if '.' not in index_code:
                    if index_code.startswith('000'):
                        index_code = f"{index_code}.SH"
                    elif index_code.startswith('399'):
                        index_code = f"{index_code}.SZ"
                
                # 标准化日期格式
                if start_date and '-' in start_date:
                    start_date = start_date.replace('-', '')
                if end_date and '-' in end_date:
                    end_date = end_date.replace('-', '')
                
                # 调用原始方法
                return original_get_stock_index_data(self, index_code, start_date, end_date)
            except Exception as e:
                logger.error(f"获取指数 {index_code} 数据失败: {str(e)}")
                return None
        
        # 应用修复
        EnhancedTushareFetcher.get_stock_index_data = fixed_get_stock_index_data
        logger.info("已修复TuShare指数数据获取方法")
        
        # 2. 修复DataSourceManager
        from src.enhanced.data.fetchers.data_source_manager import DataSourceManager
        
        # 修复get_market_overview方法
        original_get_market_overview = DataSourceManager.get_market_overview
        
        def fixed_get_market_overview(self, trade_date=None):
            """修复的市场概览获取方法"""
            try:
                # 调用原始方法
                result = original_get_market_overview(self, trade_date)
                
                # 处理返回类型
                import pandas as pd
                if isinstance(result, pd.DataFrame):
                    if not result.empty:
                        # 转换为字典
                        dict_result = {}
                        for col in result.columns:
                            dict_result[col] = result.iloc[0][col]
                        
                        # 确保日期存在
                        if 'date' not in dict_result and trade_date:
                            dict_result['date'] = trade_date
                        
                        return dict_result
                    else:
                        return {'date': trade_date or datetime.now().strftime('%Y-%m-%d')}
                
                elif result is None:
                    return {'date': trade_date or datetime.now().strftime('%Y-%m-%d')}
                
                elif isinstance(result, dict):
                    return result
                
                else:
                    return {'date': trade_date or datetime.now().strftime('%Y-%m-%d')}
                
            except Exception as e:
                logger.error(f"获取市场概览失败: {str(e)}")
                return {'date': trade_date or datetime.now().strftime('%Y-%m-%d')}
        
        # 应用修复
        DataSourceManager.get_market_overview = fixed_get_market_overview
        logger.info("已修复市场概览获取方法")
        
        # 3. 添加/修复get_stock_data方法
        if not hasattr(DataSourceManager, 'get_stock_data'):
            def get_stock_data(self, stock_code, start_date=None, end_date=None, limit=None):
                """获取股票日线数据"""
                try:
                    # 调用get_daily_data方法
                    data = self.get_daily_data(stock_code, start_date, end_date)
                    
                    if data is not None and not data.empty:
                        # 处理日期列名兼容性
                        date_col = 'date' if 'date' in data.columns else 'trade_date'
                        
                        # 根据限制条数截取数据
                        if limit is not None and len(data) > limit:
                            data = data.sort_values(date_col, ascending=False).head(limit).sort_values(date_col)
                        
                        return data
                    else:
                        return None
                    
                except Exception as e:
                    logger.error(f"获取股票 {stock_code} 数据失败: {str(e)}")
                    return None
            
            # 添加方法
            DataSourceManager.get_stock_data = get_stock_data
            logger.info("已添加get_stock_data方法")
        
        return True
        
    except Exception as e:
        logger.error(f"应用数据源修复失败: {str(e)}")
        return False

# 在模块导入时自动应用修复
apply_data_source_fixes()
