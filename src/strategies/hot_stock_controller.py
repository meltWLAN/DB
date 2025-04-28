#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
暴涨股捕捉控制器
负责将暴涨股跟踪器集成到主系统，处理GUI回调和数据展示
"""

import os
import sys
import time
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable

# 导入策略模块
from src.strategies.hot_stock_tracker import HotStockTracker

# 设置日志
logger = logging.getLogger(__name__)

class HotStockController:
    """暴涨股捕捉控制器，处理暴涨股模块与GUI的交互"""
    
    def __init__(self, tushare_fetcher=None, data_processor=None):
        """初始化控制器
        
        Args:
            tushare_fetcher: TuShare数据获取器
            data_processor: 数据处理器
        """
        self.tracker = HotStockTracker(tushare_fetcher, data_processor)
        self.last_results = {}
        self.analysis_thread = None
        self.running = False
        
        # 创建结果目录
        self.results_dir = os.path.join('results', 'hot_stocks')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def scan_limit_up_stocks(self, days=1, end_date=None, gui_callback=None):
        """扫描连续涨停股票
        
        Args:
            days: 连续涨停天数
            end_date: 结束日期
            gui_callback: GUI回调函数
            
        Returns:
            pd.DataFrame: 涨停股票数据
        """
        if gui_callback:
            gui_callback("状态", f"正在扫描连续{days}天涨停的股票...")
        
        try:
            # 获取连续涨停股
            limit_up_stocks = self.tracker.get_consecutive_limit_up_stocks(days, end_date)
            
            if limit_up_stocks is not None and not limit_up_stocks.empty:
                if gui_callback:
                    gui_callback("状态", f"找到 {len(limit_up_stocks)} 只连续{days}天涨停的股票")
                    gui_callback("涨停股", limit_up_stocks.to_dict('records'))
                
                # 保存结果
                self.last_results[f'limit_up_{days}'] = limit_up_stocks
                
                return limit_up_stocks
            else:
                if gui_callback:
                    gui_callback("状态", f"未找到连续{days}天涨停的股票")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"扫描连续涨停股票时出错: {str(e)}")
            if gui_callback:
                gui_callback("状态", f"扫描连续涨停股票时出错: {str(e)}")
            return pd.DataFrame()
    
    def scan_potential_breakout_stocks(self, threshold_score=70, end_date=None, gui_callback=None):
        """扫描潜在暴涨股
        
        Args:
            threshold_score: 最低爆发潜力得分
            end_date: 结束日期
            gui_callback: GUI回调函数
            
        Returns:
            pd.DataFrame: 潜在暴涨股数据
        """
        if gui_callback:
            gui_callback("状态", "正在分析潜在爆发股票...")
        
        try:
            # 获取潜在爆发股
            potential_stocks = self.tracker.identify_potential_breakout_stocks(end_date, threshold_score)
            
            if potential_stocks is not None and not potential_stocks.empty:
                if gui_callback:
                    gui_callback("状态", f"找到 {len(potential_stocks)} 只潜在爆发股票")
                    gui_callback("潜在爆发股", potential_stocks.to_dict('records'))
                
                # 保存结果
                self.last_results['potential_breakout'] = potential_stocks
                
                return potential_stocks
            else:
                if gui_callback:
                    gui_callback("状态", "未找到符合条件的潜在爆发股票")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"扫描潜在暴涨股时出错: {str(e)}")
            if gui_callback:
                gui_callback("状态", f"扫描潜在暴涨股时出错: {str(e)}")
            return pd.DataFrame()
    
    def analyze_hot_sectors(self, top_n=10, consecutive_days=3, end_date=None, gui_callback=None):
        """分析热门板块
        
        Args:
            top_n: 返回前N个热门板块
            consecutive_days: 分析的天数
            end_date: 结束日期
            gui_callback: GUI回调函数
            
        Returns:
            Dict: 热门板块数据
        """
        if gui_callback:
            gui_callback("状态", "正在分析热门板块...")
        
        try:
            # 获取热门板块
            hot_sectors = self.tracker.analyze_hot_sectors(top_n, consecutive_days, end_date)
            
            if hot_sectors and 'hot_sectors' in hot_sectors and hot_sectors['hot_sectors']:
                if gui_callback:
                    gui_callback("状态", f"分析完成，找到 {len(hot_sectors['hot_sectors'])} 个热门板块")
                    gui_callback("热门板块", hot_sectors)
                
                # 保存结果
                self.last_results['hot_sectors'] = hot_sectors
                
                return hot_sectors
            else:
                if gui_callback:
                    gui_callback("状态", "未找到热门板块数据")
                return {'hot_sectors': []}
            
        except Exception as e:
            logger.error(f"分析热门板块时出错: {str(e)}")
            if gui_callback:
                gui_callback("状态", f"分析热门板块时出错: {str(e)}")
            return {'hot_sectors': []}
    
    def predict_continuation(self, stock_code, end_date=None, gui_callback=None):
        """预测涨停是否会延续
        
        Args:
            stock_code: 股票代码
            end_date: 结束日期
            gui_callback: GUI回调函数
            
        Returns:
            Dict: 预测结果
        """
        if gui_callback:
            gui_callback("状态", f"正在分析 {stock_code} 涨停延续性...")
        
        try:
            # 获取预测结果
            prediction = self.tracker.predict_limit_up_continuation(stock_code, end_date)
            
            if prediction and 'continuation_probability' in prediction:
                # 格式化输出
                prob = prediction['continuation_probability']
                level = "很高" if prob >= 80 else "高" if prob >= 60 else "中等" if prob >= 40 else "低" if prob >= 20 else "很低"
                
                if gui_callback:
                    gui_callback("状态", f"{stock_code} 明日涨停概率: {prob:.2f}%，级别: {level}")
                    gui_callback("涨停预测", prediction)
                
                return prediction
            else:
                if gui_callback:
                    gui_callback("状态", f"{stock_code} 不符合预测条件")
                return {'continuation_probability': 0}
            
        except Exception as e:
            logger.error(f"预测涨停延续性时出错: {str(e)}")
            if gui_callback:
                gui_callback("状态", f"预测涨停延续性时出错: {str(e)}")
            return {'continuation_probability': 0, 'error': str(e)}
    
    def generate_comprehensive_report(self, top_n=20, end_date=None, gui_callback=None):
        """生成综合暴涨股分析报告
        
        Args:
            top_n: 每类返回的股票数量
            end_date: 结束日期
            gui_callback: GUI回调函数
            
        Returns:
            Dict: 分析报告
        """
        if self.running:
            if gui_callback:
                gui_callback("状态", "已有分析任务在运行，请等待完成")
            return None
        
        self.running = True
        
        if gui_callback:
            gui_callback("状态", "开始生成暴涨股综合分析报告...")
        
        # 创建线程执行分析
        def run_analysis():
            try:
                # 生成报告
                report = self.tracker.generate_hot_stock_report(top_n, end_date)
                
                if report and 'error' not in report:
                    # 保存结果
                    self.last_results['comprehensive_report'] = report
                    
                    # 格式化市场热度
                    market_heat = report.get('market_indicators', {}).get('market_heat', 0)
                    heat_level = report.get('market_indicators', {}).get('heat_level', '未知')
                    
                    if gui_callback:
                        gui_callback("状态", f"报告生成完成！市场热度: {market_heat:.2f}，级别: {heat_level}")
                        gui_callback("综合报告", report)
                else:
                    error = report.get('error', '未知错误')
                    if gui_callback:
                        gui_callback("状态", f"报告生成失败: {error}")
                
                self.running = False
                return report
                
            except Exception as e:
                logger.error(f"生成综合报告时出错: {str(e)}")
                if gui_callback:
                    gui_callback("状态", f"生成综合报告时出错: {str(e)}")
                self.running = False
                return None
        
        # 启动分析线程
        self.analysis_thread = threading.Thread(target=run_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        return None  # 结果将通过回调函数返回
    
    def export_results(self, result_type, file_path, gui_callback=None):
        """导出分析结果
        
        Args:
            result_type: 结果类型，可选值：limit_up_1, limit_up_2, limit_up_3, potential_breakout, hot_sectors, comprehensive_report
            file_path: 文件路径
            gui_callback: GUI回调函数
            
        Returns:
            bool: 是否导出成功
        """
        try:
            # 检查结果是否存在
            if result_type not in self.last_results:
                if gui_callback:
                    gui_callback("状态", f"未找到{result_type}类型的结果")
                return False
            
            result = self.last_results[result_type]
            
            # 根据结果类型和格式进行导出
            if isinstance(result, pd.DataFrame):
                # DataFrame导出为CSV
                result.to_csv(file_path, index=False, encoding='utf-8-sig')
                if gui_callback:
                    gui_callback("状态", f"已将结果导出至 {file_path}")
                return True
            elif isinstance(result, dict):
                # 字典导出为JSON
                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                if gui_callback:
                    gui_callback("状态", f"已将结果导出至 {file_path}")
                return True
            else:
                if gui_callback:
                    gui_callback("状态", f"不支持的结果类型: {type(result)}")
                return False
            
        except Exception as e:
            logger.error(f"导出结果时出错: {str(e)}")
            if gui_callback:
                gui_callback("状态", f"导出结果时出错: {str(e)}")
            return False
    
    def get_market_heat_indicator(self, end_date=None, gui_callback=None):
        """获取市场热度指标
        
        Args:
            end_date: 结束日期
            gui_callback: GUI回调函数
            
        Returns:
            Dict: 市场热度指标
        """
        if gui_callback:
            gui_callback("状态", "正在分析市场热度...")
        
        try:
            # 获取涨停股数据
            limit_up_1d = self.tracker.get_consecutive_limit_up_stocks(1, end_date)
            limit_up_2d = self.tracker.get_consecutive_limit_up_stocks(2, end_date)
            limit_up_3d = self.tracker.get_consecutive_limit_up_stocks(3, end_date)
            
            # 计算市场热度
            market_heat = 0
            
            # 基于涨停股票数量计算热度
            limit_up_1d_count = len(limit_up_1d) if not limit_up_1d.empty else 0
            limit_up_2d_count = len(limit_up_2d) if not limit_up_2d.empty else 0
            limit_up_3d_count = len(limit_up_3d) if not limit_up_3d.empty else 0
            
            if limit_up_1d_count > 0:
                limit_up_ratio = min(1.0, limit_up_1d_count / 100)
                consecutive_ratio = (limit_up_2d_count * 2 + limit_up_3d_count * 3) / 50
                
                market_heat = (limit_up_ratio * 40 + consecutive_ratio * 60) * 100
                market_heat = min(100, max(0, market_heat))
            
            # 确定热度级别
            heat_level = '火爆' if market_heat >= 80 else \
                         '热' if market_heat >= 60 else \
                         '温和' if market_heat >= 40 else \
                         '冷' if market_heat >= 20 else '极冷'
            
            # 创建结果
            result = {
                'date': end_date or datetime.now().strftime('%Y-%m-%d'),
                'market_heat': market_heat,
                'heat_level': heat_level,
                'limit_up_counts': {
                    '1day': limit_up_1d_count,
                    '2days': limit_up_2d_count,
                    '3days': limit_up_3d_count
                }
            }
            
            if gui_callback:
                gui_callback("状态", f"市场热度: {market_heat:.2f}，级别: {heat_level}")
                gui_callback("市场热度", result)
            
            return result
            
        except Exception as e:
            logger.error(f"获取市场热度指标时出错: {str(e)}")
            if gui_callback:
                gui_callback("状态", f"获取市场热度指标时出错: {str(e)}")
            return {
                'date': end_date or datetime.now().strftime('%Y-%m-%d'),
                'market_heat': 0,
                'heat_level': '未知',
                'error': str(e)
            }
    
    def visualize_market_heat_trend(self, days=30, gui_callback=None):
        """可视化市场热度趋势
        
        Args:
            days: 分析的天数
            gui_callback: GUI回调函数
            
        Returns:
            Dict: 市场热度趋势数据
        """
        if gui_callback:
            gui_callback("状态", f"正在分析近{days}天市场热度趋势...")
        
        try:
            # 获取日期序列
            end_date = datetime.now()
            date_list = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
            date_list.reverse()  # 按时间升序排列
            
            # 收集每日热度
            heat_trend = []
            
            for date in date_list:
                # 获取当日市场热度
                heat_data = self.get_market_heat_indicator(date)
                
                heat_trend.append({
                    'date': date,
                    'market_heat': heat_data['market_heat'],
                    'heat_level': heat_data['heat_level'],
                    'limit_up_1d': heat_data['limit_up_counts']['1day'],
                    'limit_up_2d': heat_data['limit_up_counts']['2days'],
                    'limit_up_3d': heat_data['limit_up_counts']['3days']
                })
            
            # 创建结果
            result = {
                'start_date': date_list[0],
                'end_date': date_list[-1],
                'days': days,
                'trend_data': heat_trend
            }
            
            if gui_callback:
                gui_callback("状态", f"市场热度趋势分析完成，共{days}天数据")
                gui_callback("热度趋势", result)
            
            return result
            
        except Exception as e:
            logger.error(f"分析市场热度趋势时出错: {str(e)}")
            if gui_callback:
                gui_callback("状态", f"分析市场热度趋势时出错: {str(e)}")
            return {
                'error': str(e)
            } 