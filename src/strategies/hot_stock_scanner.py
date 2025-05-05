#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
热门股票扫描器
提供连续涨停股扫描、潜在暴涨股分析和热门板块识别功能
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stock_data_storage import StockData

class HotStockScanner:
    """热门股票扫描器"""
    
    def __init__(self):
        """初始化扫描器"""
        self.logger = logging.getLogger(__name__)
        self.stock_data = StockData()
        
        # 创建结果目录
        self.results_dir = os.path.join('results', 'hot_stocks')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger.info("热门股票扫描器初始化完成")
    
    def _is_limit_up(self, day_data):
        """判断是否涨停
        
        Args:
            day_data: 单日交易数据
            
        Returns:
            bool: 是否涨停
        """
        if not day_data:
            return False
            
        # 获取收盘价和前一日收盘价
        close = day_data.get('close', 0)
        prev_close = day_data.get('prev_close', 0)
        
        if prev_close <= 0:
            return False
            
        # 计算涨幅
        change_pct = (close - prev_close) / prev_close * 100
        
        # 判断是否涨停（考虑不同市场的涨停幅度）
        if 'code' in day_data:
            code = day_data['code']
            # 科创板和创业板涨跌幅限制为20%
            if code.startswith(('688', '300')):
                return change_pct >= 19.5
        
        # 主板和其他板块涨跌幅限制为10%
        return change_pct >= 9.5
        
    def scan_consecutive_limit_up(self, days=2):
        """
        扫描连续涨停股票
        
        Args:
            days (int): 连续涨停天数，默认为2天
            
        Returns:
            list: 连续涨停股票列表，每个元素为字典，包含股票代码、名称、连续涨停天数等信息
        """
        try:
            self.logger.info(f"开始扫描连续{days}天涨停股票...")
            
            # 获取股票列表
            stock_list = self.stock_data.get_stock_list()
            hot_stocks = []
            
            for stock_code in stock_list:
                try:
                    # 获取股票详情
                    stock_details = self.stock_data.get_stock_details(stock_code)
                    if not stock_details:
                        continue
                        
                    # 获取最近的交易数据
                    daily_data = stock_details.get('daily_data', [])
                    if len(daily_data) < days:
                        continue
                    
                    # 检查最近n天是否连续涨停
                    recent_data = daily_data[-days:]
                    is_consecutive = True
                    
                    for day_data in recent_data:
                        if not self._is_limit_up(day_data):
                            is_consecutive = False
                            break
                    
                    if is_consecutive:
                        hot_stocks.append({
                            'code': stock_code,
                            'name': stock_details.get('name', ''),
                            'consecutive_days': days,
                            'last_price': recent_data[-1].get('close', 0),
                            'volume': recent_data[-1].get('volume', 0),
                            'turnover': recent_data[-1].get('turnover', 0)
                        })
                        
                except Exception as e:
                    self.logger.error(f"处理股票{stock_code}时出错: {str(e)}")
                    continue
            
            self.logger.info(f"扫描完成，找到{len(hot_stocks)}只连续{days}天涨停股票")
            return hot_stocks
            
        except Exception as e:
            self.logger.error(f"扫描连续涨停股票时出错: {str(e)}")
            return []
            
    def analyze_hot_sectors(self):
        """分析热门板块
        
        Returns:
            dict: 热门板块分析结果
        """
        try:
            self.logger.info("开始分析热门板块")
            
            # 获取所有行业
            industries = self.stock_data.get_industries()
            
            # 存储结果
            hot_sectors = []
            
            # 分析每个行业
            for industry in industries:
                try:
                    # 获取行业股票
                    stocks = self.stock_data.get_stocks_by_industry(industry)
                    
                    if not stocks:
                        continue
                    
                    # 统计数据
                    limit_up_count = 0
                    total_change = 0
                    max_change = 0
                    top_performer = None
                    
                    # 遍历行业股票
                    for code in stocks:
                        stock_detail = self.stock_data.get_stock_details(code)
                        daily_data = stock_detail.get('daily_data', [])
                        
                        if daily_data:
                            # 获取最新交易日数据
                            latest_data = daily_data[-1]
                            
                            # 计算涨跌幅
                            if len(daily_data) > 1:
                                prev_close = daily_data[-2].get('close', 0)
                                if prev_close > 0:
                                    change = (latest_data.get('close', 0) - prev_close) / prev_close * 100
                                    
                                    # 统计涨停
                                    if change >= 9.5:
                                        limit_up_count += 1
                                    
                                    # 更新行业涨跌幅
                                    total_change += change
                                    
                                    # 更新最强个股
                                    if change > max_change:
                                        max_change = change
                                        top_performer = {
                                            'code': code,
                                            'name': self.stock_data.get_stock_name(code),
                                            'change': change
                                        }
                    
                    # 计算行业平均涨跌幅
                    avg_change = total_change / len(stocks) if stocks else 0
                    
                    # 计算行业热度得分
                    score = (limit_up_count / len(stocks) * 50) + (avg_change * 5)
                    
                    # 添加到结果
                    hot_sectors.append({
                        'sector': industry,
                        'limit_up_count': limit_up_count,
                        'avg_change': round(avg_change, 2),
                        'top_performer': top_performer,
                        'score': round(score, 1)
                    })
                
                except Exception as e:
                    self.logger.error(f"处理行业 {industry} 时出错: {str(e)}")
                    continue
            
            # 按热度得分排序
            hot_sectors.sort(key=lambda x: x['score'], reverse=True)
            
            self.logger.info(f"分析完成，处理了 {len(hot_sectors)} 个行业")
            return {
                'hot_sectors': hot_sectors,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'market_heat': round(sum(sector['score'] for sector in hot_sectors) / len(hot_sectors), 1) if hot_sectors else 50
            }
            
        except Exception as e:
            self.logger.error(f"分析热门板块时出错: {str(e)}")
            return {'hot_sectors': []} 