#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
热门股票控制器
实现热门股票分析功能与GUI的交互
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# 导入热门股票跟踪器
from src.strategies.hot_stock_tracker import HotStockTracker

# 尝试导入增强版数据模块（如果可用）
HAS_ENHANCED_MODULES = False
try:
    from src.enhanced.data.fetchers.tushare_fetcher import EnhancedTushareFetcher
    from src.enhanced.data.processors.optimized_processor import EnhancedDataProcessor
    HAS_ENHANCED_MODULES = True
except ImportError:
    logging.warning("未能导入增强版数据模块，将使用基础版功能")

# 设置日志
logger = logging.getLogger(__name__)

class HotStockController:
    """热门股票控制器，处理前端请求并返回分析结果"""
    
    def __init__(self, config=None, tushare_fetcher=None, data_processor=None):
        """初始化控制器
        
        Args:
            config: 配置字典
            tushare_fetcher: 数据获取器
            data_processor: 数据处理器
        """
        self.config = config or {}
        
        # 默认配置
        self.default_config = {
            'api_token': '',  # TuShare API Token
            'data_quality_threshold': 60,  # 数据质量阈值
            'max_missing_days': 5,  # 最大允许缺失天数
            'min_trading_days': 60,  # 最小交易天数要求
            'connection_retries': 3,  # 连接重试次数
            'retry_delay': 2,  # 重试延迟（秒）
        }
        
        # 合并配置
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # 如果没有提供数据获取器和处理器，尝试创建增强版模块的实例
        if tushare_fetcher is None and HAS_ENHANCED_MODULES:
            logger.info("创建默认的增强版TuShare数据获取器")
            tushare_fetcher = EnhancedTushareFetcher(config)
        
        if data_processor is None and HAS_ENHANCED_MODULES:
            logger.info("创建默认的增强版数据处理器")
            data_processor = EnhancedDataProcessor(config)
        
        # 创建热门股票跟踪器
        self.tracker = HotStockTracker(tushare_fetcher, data_processor)
        logger.info("热门股票控制器初始化完成")
        
        # 创建结果目录
        self.results_dir = os.path.join('results', 'hot_stocks')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_consecutive_limit_up_stocks(self, consecutive_days=1, end_date=None):
        """获取连续涨停股票
        
        Args:
            consecutive_days: 连续涨停天数
            end_date: 结束日期，格式 YYYY-MM-DD
            
        Returns:
            pd.DataFrame: 连续涨停股票数据
        """
        try:
            logger.info(f"请求获取连续{consecutive_days}天涨停的股票，截止日期: {end_date or '今日'}")
            result = self.tracker.get_consecutive_limit_up_stocks(consecutive_days, end_date)
            return result
        except Exception as e:
            logger.error(f"获取连续涨停股票时出错: {str(e)}")
            return pd.DataFrame()
    
    def get_potential_breakout_stocks(self, threshold_score=70, end_date=None):
        """获取潜在暴涨股
        
        Args:
            threshold_score: 最小爆发潜力得分 (0-100)
            end_date: 结束日期，格式 YYYY-MM-DD
            
        Returns:
            pd.DataFrame: 潜在暴涨股数据
        """
        try:
            logger.info(f"请求获取潜在暴涨股，得分阈值: {threshold_score}，截止日期: {end_date or '今日'}")
            result = self.tracker.identify_potential_breakout_stocks(end_date, threshold_score)
            return result
        except Exception as e:
            logger.error(f"获取潜在暴涨股时出错: {str(e)}")
            return pd.DataFrame()
    
    def get_hot_sectors(self, top_n=10, consecutive_days=3, end_date=None):
        """获取热门板块
        
        Args:
            top_n: 返回前N个热门板块
            consecutive_days: 分析的天数
            end_date: 结束日期，格式 YYYY-MM-DD
            
        Returns:
            Dict: 热门板块数据
        """
        try:
            logger.info(f"请求获取热门板块，Top {top_n}，分析天数: {consecutive_days}，截止日期: {end_date or '今日'}")
            result = self.tracker.analyze_hot_sectors(top_n, consecutive_days, end_date)
            return result
        except Exception as e:
            logger.error(f"获取热门板块时出错: {str(e)}")
            return {'hot_sectors': []}
    
    def predict_continuation(self, stock_code, end_date=None):
        """预测涨停延续性
        
        Args:
            stock_code: 股票代码
            end_date: 结束日期，格式 YYYY-MM-DD
            
        Returns:
            Dict: 预测结果
        """
        try:
            logger.info(f"请求预测股票 {stock_code} 的涨停延续性，截止日期: {end_date or '今日'}")
            result = self.tracker.predict_limit_up_continuation(stock_code, end_date)
            return result
        except Exception as e:
            logger.error(f"预测涨停延续性时出错: {str(e)}")
            return {'continuation_probability': 0, 'error': str(e)}
    
    def get_comprehensive_report(self, end_date=None):
        """获取综合分析报告
        
        Args:
            end_date: 结束日期，格式 YYYY-MM-DD
            
        Returns:
            Dict: 综合分析报告
        """
        try:
            logger.info(f"请求生成综合分析报告，截止日期: {end_date or '今日'}")
            
            # 获取各类数据
            limit_up_1d = self.get_consecutive_limit_up_stocks(1, end_date)
            limit_up_2d = self.get_consecutive_limit_up_stocks(2, end_date)
            limit_up_3d = self.get_consecutive_limit_up_stocks(3, end_date)
            
            potential_stocks = self.get_potential_breakout_stocks(70, end_date)
            
            hot_sectors = self.get_hot_sectors(10, 3, end_date)
            
            # 创建报告
            report = {
                'date': end_date or datetime.now().strftime('%Y-%m-%d'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'consecutive_limit_up': {
                    '1day': limit_up_1d.to_dict('records') if not limit_up_1d.empty else [],
                    '2days': limit_up_2d.to_dict('records') if not limit_up_2d.empty else [],
                    '3days': limit_up_3d.to_dict('records') if not limit_up_3d.empty else [],
                    'summary': {
                        'total_1day': len(limit_up_1d),
                        'total_2days': len(limit_up_2d),
                        'total_3days': len(limit_up_3d),
                    }
                },
                'potential_breakout': {
                    'stocks': potential_stocks.to_dict('records') if not potential_stocks.empty else [],
                    'total': len(potential_stocks)
                },
                'hot_sectors': hot_sectors,
                'market_heat': hot_sectors.get('market_heat', 50) if isinstance(hot_sectors, dict) else 50
            }
            
            # 保存报告
            self._save_report(report)
            
            return report
        except Exception as e:
            logger.error(f"生成综合分析报告时出错: {str(e)}")
            return {
                'date': end_date or datetime.now().strftime('%Y-%m-%d'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
    
    def _save_report(self, report):
        """保存分析报告
        
        Args:
            report: 分析报告字典
        """
        try:
            # 创建时间戳文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(self.results_dir, f'hot_stock_report_{timestamp}.json')
            
            # 保存为JSON
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"热门股票分析报告已保存到 {file_path}")
            
        except Exception as e:
            logger.error(f"保存热门股票分析报告时出错: {str(e)}")
    
    def convert_to_gui_format(self, data, data_type='limit_up'):
        """转换数据为GUI显示格式
        
        Args:
            data: 原始数据 (DataFrame或Dict)
            data_type: 数据类型 ('limit_up', 'potential', 'sectors')
            
        Returns:
            List: 适合GUI显示的数据列表
        """
        try:
            if data_type == 'limit_up':
                if isinstance(data, pd.DataFrame) and not data.empty:
                    result = []
                    for _, row in data.iterrows():
                        item = {
                            'code': row['code'],
                            'name': row['name'] if 'name' in row else '',
                            'industry': row['industry'] if 'industry' in row else '',
                            'consecutive_days': str(row['consecutive_days']) if 'consecutive_days' in row else '1',
                            'last_close': f"{row['last_close']:.2f}" if 'last_close' in row else '0.00',
                            'change_percent': f"{row['change_percent']:.2f}%" if 'change_percent' in row else '0.00%',
                            'volume_ratio': f"{row['volume_ratio']:.2f}" if 'volume_ratio' in row else '0.00',
                            'last_date': row['last_date'] if 'last_date' in row else ''
                        }
                        result.append(item)
                    return result
                return []
                
            elif data_type == 'potential':
                if isinstance(data, pd.DataFrame) and not data.empty:
                    result = []
                    for _, row in data.iterrows():
                        item = {
                            'code': row['code'],
                            'name': row['name'] if 'name' in row else '',
                            'industry': row['industry'] if 'industry' in row else '',
                            'breakout_score': f"{row['breakout_score']:.1f}" if 'breakout_score' in row else '0.0',
                            'close': f"{row['close']:.2f}" if 'close' in row else '0.00',
                            'change_percent': f"{row['change_percent']:.2f}%" if 'change_percent' in row else '0.00%',
                            'volume_ratio': f"{row['volume_ratio']:.2f}" if 'volume_ratio' in row else '0.00',
                            'money_flow_score': f"{row['money_flow_score']:.1f}" if 'money_flow_score' in row else '0.0',
                            'macd_signal': row['macd_signal'] if 'macd_signal' in row else ''
                        }
                        result.append(item)
                    return result
                return []
                
            elif data_type == 'sectors':
                if isinstance(data, dict) and 'hot_sectors' in data:
                    result = []
                    for sector in data['hot_sectors']:
                        item = {
                            'sector': sector['sector'],
                            'limit_up_count': str(sector['limit_up_count']) if 'limit_up_count' in sector else '0',
                            'avg_change': f"{sector['avg_change']:.2f}%" if 'avg_change' in sector else '0.00%',
                            'top_performer_name': sector['top_performer']['name'] if 'top_performer' in sector else '',
                            'top_performer_code': sector['top_performer']['code'] if 'top_performer' in sector else '',
                            'top_performer_change': f"{sector['top_performer']['change']:.2f}%" if 'top_performer' in sector else '0.00%',
                            'score': f"{sector['score']:.1f}" if 'score' in sector else '0.0'
                        }
                        result.append(item)
                    return result
                return []
                
            else:
                logger.warning(f"未知的数据类型: {data_type}")
                return []
                
        except Exception as e:
            logger.error(f"转换GUI格式时出错: {str(e)}")
            return [] 