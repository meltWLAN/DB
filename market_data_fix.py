#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
市场数据获取修复补丁
修复TuShare、AKShare数据接口和DataSourceManager的性能问题
"""

import logging
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('market_data_fix.log')
    ]
)
logger = logging.getLogger()

def apply_market_data_fixes():
    """应用市场数据修复补丁"""
    logger.info("开始应用市场数据修复补丁...")
    
    # 先测试当前状态
    logger.info("测试当前数据源状态...")
    test_data_source()
    
    # 应用修复
    try:
        # 导入需要修改的模块
        from src.enhanced.data.fetchers.data_source_manager import DataSourceManager
        from src.enhanced.data.fetchers.tushare_fetcher import EnhancedTushareFetcher
        
        # 1. 修复EnhancedTushareFetcher中的get_stock_index_data方法
        logger.info("\n1. 修复指数代码解析和日期处理问题...")
        patch_tushare_fetcher()
        
        # 2. 修复DataSourceManager中的get_market_overview方法
        logger.info("\n2. 优化市场概览获取性能...")
        patch_data_source_manager()
        
        # 3. 修复日期格式转换和种子值处理
        logger.info("\n3. 修复数据模拟和日期格式问题...")
        patch_mock_data_generation()
        
        # 验证修复效果
        logger.info("\n验证修复效果...")
        test_data_source()
        
        logger.info("\n市场数据修复补丁应用完成！")
        
    except Exception as e:
        logger.error(f"应用修复补丁过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

def patch_tushare_fetcher():
    """修复TuShare获取器中的问题"""
    try:
        from src.enhanced.data.fetchers.tushare_fetcher import EnhancedTushareFetcher
        old_get_stock_index_data = EnhancedTushareFetcher.get_stock_index_data
        
        def patched_get_stock_index_data(self, index_code: str, start_date: str, end_date: str = None):
            """修复后的获取指数数据方法"""
            try:
                # 标准化日期格式 YYYY-MM-DD -> YYYYMMDD
                start_date_fmt = start_date.replace('-', '') if start_date else (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                end_date_fmt = end_date.replace('-', '') if end_date else datetime.now().strftime('%Y%m%d')
                
                # 标准化指数代码，确保包含交易所后缀
                if '.' not in index_code:
                    if index_code.startswith('000') or index_code.startswith('399'):
                        # 根据前缀判断交易所
                        if index_code.startswith('000'):
                            full_code = f"{index_code}.SH"  # 上证系列指数
                        elif index_code.startswith('399'):
                            full_code = f"{index_code}.SZ"  # 深证系列指数
                        else:
                            full_code = index_code
                        logger.info(f"标准化指数代码: {index_code} -> {full_code}")
                        index_code = full_code
                
                logger.info(f"获取指数 {index_code} 数据, 时间范围: {start_date_fmt} - {end_date_fmt}")
                
                # 调用原始方法
                return old_get_stock_index_data(self, index_code, start_date, end_date)
            except Exception as e:
                logger.error(f"获取指数 {index_code} 数据失败: {str(e)}")
                return None
        
        # 替换方法
        EnhancedTushareFetcher.get_stock_index_data = patched_get_stock_index_data
        logger.info("成功修复TuShare获取器的指数数据获取方法")
        
    except Exception as e:
        logger.error(f"修复TuShare获取器失败: {str(e)}")

def patch_data_source_manager():
    """修复DataSourceManager中的性能问题"""
    try:
        from src.enhanced.data.fetchers.data_source_manager import DataSourceManager
        old_get_market_overview = DataSourceManager.get_market_overview
        
        def patched_get_market_overview(self, trade_date: str = None):
            """优化的市场概览获取方法"""
            # 检查缓存
            if self.cache_enabled:
                cache_key = self._get_cache_key("get_market_overview", trade_date)
                cached_data = self._get_from_cache(cache_key)
                if cached_data is not None:
                    logger.info(f"从缓存获取市场概览数据: {trade_date}")
                    return cached_data
            
            try:
                # 获取所有股票当日行情
                all_data = self.get_all_stock_data_on_date(trade_date)
                if all_data is None or all_data.empty:
                    logger.warning(f"获取 {trade_date} 的市场概览失败：无法获取行情数据")
                    return {}
                
                # 计算涨跌家数
                up_count = len(all_data[all_data['close'] > all_data['open']])
                down_count = len(all_data[all_data['close'] < all_data['open']])
                flat_count = len(all_data) - up_count - down_count
                
                # 计算总成交量和成交额
                total_volume = all_data['volume'].sum() if 'volume' in all_data.columns else 0
                total_amount = all_data['amount'].sum() if 'amount' in all_data.columns else 0
                
                # 计算平均涨跌幅
                all_data['change_pct'] = (all_data['close'] - all_data['open']) / all_data['open'] * 100
                avg_change_pct = all_data['change_pct'].mean()
                
                # 组装结果
                result = {
                    'date': trade_date,
                    'up_count': up_count,
                    'down_count': down_count,
                    'flat_count': flat_count,
                    'total_count': len(all_data),
                    'total_volume': total_volume,
                    'total_amount': total_amount,
                    'avg_change_pct': avg_change_pct,
                    'turnover_rate': total_volume / total_amount * 100 if total_amount > 0 else 0,
                }
                
                # 保存到缓存
                if self.cache_enabled:
                    self._save_to_cache(cache_key, result)
                
                logger.info(f"成功获取 {trade_date} 的市场概览数据")
                return result
                
            except Exception as e:
                logger.error(f"获取市场概览数据失败: {str(e)}")
                return {}
        
        # 替换方法
        DataSourceManager.get_market_overview = patched_get_market_overview
        logger.info("成功优化DataSourceManager的市场概览获取方法")
        
        # 添加(或修复)get_stock_data方法
        def patched_get_stock_data(self, stock_code: str, start_date: str = None, end_date: str = None, limit: int = None):
            """修复后的股票数据获取方法"""
            try:
                # 如果未提供起始日期但提供了limit，设置默认起始日期
                if start_date is None and limit is not None:
                    from datetime import datetime, timedelta
                    end_date = datetime.now().strftime('%Y-%m-%d') if end_date is None else end_date
                    start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=limit * 2)).strftime('%Y-%m-%d')
                    
                # 调用daily_data方法
                data = self.get_daily_data(stock_code, start_date, end_date)
                
                # 如果获取成功，进行处理
                if data is not None and not data.empty:
                    # 确保date列存在
                    date_col = 'date' if 'date' in data.columns else 'trade_date'
                    
                    # 如果提供了limit，截取最近的数据
                    if limit is not None and len(data) > limit:
                        data = data.sort_values(date_col, ascending=False).head(limit).sort_values(date_col)
                        
                    return data
                else:
                    logger.warning(f"无法获取股票 {stock_code} 数据")
                    return None
            except Exception as e:
                logger.error(f"获取股票 {stock_code} 日线数据失败: {e}")
                return None
        
        # 添加方法
        DataSourceManager.get_stock_data = patched_get_stock_data
        logger.info("成功添加/修复DataSourceManager的get_stock_data方法")
        
    except Exception as e:
        logger.error(f"修复DataSourceManager失败: {str(e)}")

def patch_mock_data_generation():
    """修复模拟数据生成和日期格式转换问题"""
    try:
        from src.enhanced.data.fetchers.data_source_manager import DataSourceManager
        old_generate_mock_index_data = DataSourceManager._generate_mock_index_data
        
        def patched_generate_mock_index_data(self, index_code: str, start_date: str = None, end_date: str = None):
            """修复后的模拟指数数据生成方法"""
            try:
                # 解析日期
                from datetime import datetime, timedelta
                if end_date is None:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                if start_date is None:
                    # 默认生成30天的数据
                    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                
                # 创建日期范围
                date_range = []
                curr_dt = start_dt
                while curr_dt <= end_dt:
                    # 跳过周末
                    if curr_dt.weekday() < 5:  # 0-4是周一到周五
                        date_range.append(curr_dt.strftime('%Y-%m-%d'))
                    curr_dt += timedelta(days=1)
                
                # 使用稳定的种子确保同一指数每次生成的模拟数据一致
                import numpy as np
                
                # 修复: 确保种子在有效范围内 (0到2^32-1)
                # 使用简单的数值哈希函数
                def simple_hash(s):
                    h = 0
                    for c in s:
                        h = (31 * h + ord(c)) & 0x7FFFFFFF
                    return h
                
                seed = simple_hash(index_code) % (2**32 - 1)
                logger.info(f"为指数 {index_code} 生成种子值: {seed}")
                np.random.seed(seed)
                
                # 基础价格根据指数类型设置
                if '000001.SH' in index_code:  # 上证指数
                    base_price = 3000 + np.random.uniform(-300, 300)
                elif '399001' in index_code:  # 深证成指
                    base_price = 10000 + np.random.uniform(-1000, 1000)
                elif '399006' in index_code:  # 创业板指
                    base_price = 2000 + np.random.uniform(-200, 200)
                elif '000300' in index_code:  # 沪深300
                    base_price = 3500 + np.random.uniform(-350, 350)
                elif '000016' in index_code:  # 上证50
                    base_price = 2500 + np.random.uniform(-250, 250)
                elif '000905' in index_code:  # 中证500
                    base_price = 6000 + np.random.uniform(-600, 600)
                else:
                    base_price = 1000 + np.random.uniform(-100, 100)
                
                # 生成价格和成交量
                prices = []
                volumes = []
                for i in range(len(date_range)):
                    if i == 0:
                        prices.append(base_price)
                    else:
                        change_pct = np.random.normal(0, 0.01)  # 每日涨跌幅，均值为0，标准差为1%
                        prices.append(prices[i-1] * (1 + change_pct))
                    volumes.append(np.random.randint(100000, 10000000))  # 随机成交量
                
                # 创建DataFrame
                data = {
                    'ts_code': index_code,
                    'trade_date': date_range,
                    'open': [price * (1 + np.random.normal(0, 0.003)) for price in prices],
                    'high': [price * (1 + abs(np.random.normal(0, 0.005))) for price in prices],
                    'low': [price * (1 - abs(np.random.normal(0, 0.005))) for price in prices],
                    'close': prices,
                    'vol': volumes,
                    'amount': [vol * price / 1000 * np.random.uniform(0.95, 1.05) for vol, price in zip(volumes, prices)]
                }
                
                # 计算涨跌幅
                data['change'] = [0] + [prices[i] - prices[i-1] for i in range(1, len(prices))]
                data['pct_chg'] = [0] + [(prices[i] / prices[i-1] - 1) * 100 for i in range(1, len(prices))]
                
                # 转换trade_date列格式为字符串，兼容性更好
                mock_df = pd.DataFrame(data)
                
                # 添加date列，与trade_date保持一致
                mock_df['date'] = mock_df['trade_date']
                
                # 重置随机数种子
                np.random.seed(None)
                
                logger.info(f"成功生成 {index_code} 的模拟指数数据，包含 {len(mock_df)} 条记录")
                return mock_df
                
            except Exception as e:
                logger.error(f"生成模拟指数数据出错: {str(e)}")
                # 出错时返回至少包含基本结构的空DataFrame
                import pandas as pd
                empty_df = pd.DataFrame({
                    'ts_code': [index_code],
                    'trade_date': [datetime.now().strftime('%Y-%m-%d')],
                    'date': [datetime.now().strftime('%Y-%m-%d')],
                    'open': [0], 'high': [0], 'low': [0], 'close': [0],
                    'vol': [0], 'amount': [0], 'pct_chg': [0]
                })
                return empty_df
        
        # 替换方法
        DataSourceManager._generate_mock_index_data = patched_generate_mock_index_data
        logger.info("成功修复模拟数据生成方法")
        
    except Exception as e:
        logger.error(f"修复模拟数据生成失败: {str(e)}")

def test_data_source():
    """测试数据源功能"""
    try:
        from src.enhanced.data.fetchers.data_source_manager import DataSourceManager
        
        # 初始化数据源管理器
        logger.info("初始化DataSourceManager...")
        manager = DataSourceManager()
        
        # 测试获取指数数据
        logger.info("测试获取上证指数数据...")
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        start_time = time.time()
        index_data = manager.get_stock_index_data('000001.SH', start_date, end_date)
        elapsed = time.time() - start_time
        
        if index_data is not None and not index_data.empty:
            logger.info(f"成功获取上证指数数据，共{len(index_data)}条记录，耗时：{elapsed:.2f}秒")
        else:
            logger.warning(f"未获取到上证指数数据，耗时：{elapsed:.2f}秒")
        
        # 测试获取市场概览
        logger.info("测试获取市场概览...")
        start_time = time.time()
        market_overview = manager.get_market_overview()
        elapsed = time.time() - start_time
        
        if market_overview and isinstance(market_overview, dict) and len(market_overview) > 0:
            logger.info(f"成功获取市场概览，耗时：{elapsed:.2f}秒")
            logger.info(f"市场概览包含 {len(market_overview)} 个数据项")
        else:
            logger.warning(f"未获取到市场概览，耗时：{elapsed:.2f}秒")
        
        # 测试get_stock_data方法
        logger.info("测试获取股票数据...")
        start_time = time.time()
        stock_data = manager.get_stock_data('000001.SZ', start_date, end_date)
        elapsed = time.time() - start_time
        
        if stock_data is not None and not stock_data.empty:
            logger.info(f"成功获取平安银行股票数据，共{len(stock_data)}条记录，耗时：{elapsed:.2f}秒")
        else:
            logger.warning(f"未获取到平安银行股票数据，耗时：{elapsed:.2f}秒")
            
            # 测试get_daily_data方法
            logger.info("尝试使用get_daily_data方法...")
            start_time = time.time()
            daily_data = manager.get_daily_data('000001.SZ', start_date, end_date)
            elapsed = time.time() - start_time
            
            if daily_data is not None and not daily_data.empty:
                logger.info(f"成功使用get_daily_data获取平安银行数据，共{len(daily_data)}条记录，耗时：{elapsed:.2f}秒")
            else:
                logger.warning(f"使用get_daily_data也未获取到平安银行数据，耗时：{elapsed:.2f}秒")
        
        return True
        
    except Exception as e:
        logger.error(f"测试数据源失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    apply_market_data_fixes() 