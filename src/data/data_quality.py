#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
import re
from datetime import datetime, timedelta

class DataQuality:
    @staticmethod
    def validate_stock_code(stock_code: str) -> bool:
        """
        验证股票代码格式是否正确
        """
        # A股股票代码格式验证
        pattern = r'^[0-9]{6}$'
        return bool(re.match(pattern, stock_code))
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> Tuple[bool, str]:
        """
        验证日期范围是否有效
        """
        try:
            start = datetime.strptime(start_date, '%Y%m%d')
            end = datetime.strptime(end_date, '%Y%m%d')
            
            # 检查日期范围
            if start > end:
                return False, "开始日期不能晚于结束日期"
            
            # 检查是否超过合理范围
            if end > datetime.now():
                return False, "结束日期不能晚于当前日期"
            
            if start < datetime(1990, 1, 1):  # A股最早数据
                return False, "开始日期不能早于1990年"
            
            # 检查日期范围是否过大
            if (end - start).days > 3650:  # 约10年
                return False, "日期范围不能超过10年"
            
            return True, "日期范围有效"
            
        except ValueError:
            return False, "日期格式无效，请使用YYYYMMDD格式"
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame, stock_code: str) -> Dict[str, Any]:
        """
        检查数据质量并生成报告
        """
        if df is None or df.empty:
            return {
                'status': 'error',
                'message': '数据为空'
            }
        
        report = {
            'status': 'ok',
            'warnings': [],
            'statistics': {},
            'completeness': {},
            'validity': {},
            'consistency': {},
            'timeliness': {}
        }
        
        try:
            # 1. 完整性检查
            DataQuality._check_completeness(df, report)
            
            # 2. 有效性检查
            DataQuality._check_validity(df, report)
            
            # 3. 一致性检查
            DataQuality._check_consistency(df, report)
            
            # 4. 时效性检查
            DataQuality._check_timeliness(df, report)
            
            # 5. 基本统计信息
            DataQuality._calculate_statistics(df, report)
            
            # 根据警告数量确定状态
            if len(report['warnings']) > 0:
                report['status'] = 'warning'
            
            return report
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'数据质量检查失败: {str(e)}'
            }
    
    @staticmethod
    def _check_completeness(df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """检查数据完整性"""
        # 检查必要字段
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            report['warnings'].append(f"缺失必要字段: {', '.join(missing_columns)}")
        
        # 检查缺失值
        missing_stats = df[required_columns].isnull().sum()
        report['completeness']['missing_values'] = missing_stats.to_dict()
        
        # 检查交易日是否连续
        date_gaps = DataQuality._check_date_continuity(df)
        if date_gaps:
            report['warnings'].append(f"发现{len(date_gaps)}个交易日缺失")
            report['completeness']['missing_dates'] = date_gaps
    
    @staticmethod
    def _check_validity(df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """检查数据有效性"""
        validity_issues = []
        
        # 价格有效性
        if (df['close'] <= 0).any():
            validity_issues.append("存在无效的收盘价（<=0）")
        
        if (df['high'] < df['low']).any():
            validity_issues.append("存在最高价低于最低价的异常")
        
        if ((df['open'] > df['high']) | (df['open'] < df['low'])).any():
            validity_issues.append("存在开盘价超出日内范围的异常")
        
        if ((df['close'] > df['high']) | (df['close'] < df['low'])).any():
            validity_issues.append("存在收盘价超出日内范围的异常")
        
        # 成交量有效性
        if (df['volume'] < 0).any():
            validity_issues.append("存在负的成交量")
        
        report['validity']['issues'] = validity_issues
        if validity_issues:
            report['warnings'].extend(validity_issues)
    
    @staticmethod
    def _check_consistency(df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """检查数据一致性"""
        consistency_issues = []
        
        # 检查价格波动是否合理（日内波动不超过20%）
        daily_change = abs(df['close'].pct_change())
        abnormal_changes = daily_change[daily_change > 0.20]
        if not abnormal_changes.empty:
            consistency_issues.append(f"发现{len(abnormal_changes)}天价格波动超过20%")
        
        # 检查成交量突变
        volume_change = abs(df['volume'].pct_change())
        abnormal_volumes = volume_change[volume_change > 5]  # 成交量变化超过500%
        if not abnormal_volumes.empty:
            consistency_issues.append(f"发现{len(abnormal_volumes)}天成交量突变（变化>500%）")
        
        report['consistency']['issues'] = consistency_issues
        if consistency_issues:
            report['warnings'].extend(consistency_issues)
    
    @staticmethod
    def _check_timeliness(df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """检查数据时效性"""
        if df.empty:
            return
        
        latest_date = df.index.max()
        days_old = (pd.Timestamp.now() - latest_date).days
        
        report['timeliness']['latest_date'] = latest_date.strftime('%Y-%m-%d')
        report['timeliness']['days_since_update'] = days_old
        
        if days_old > 3:  # 假设数据应该每3天更新一次
            report['warnings'].append(f"数据可能过时，最后更新日期: {latest_date.strftime('%Y-%m-%d')}")
    
    @staticmethod
    def _calculate_statistics(df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """计算基本统计信息"""
        stats = {}
        
        # 数据范围
        stats['date_range'] = {
            'start': df.index.min().strftime('%Y-%m-%d'),
            'end': df.index.max().strftime('%Y-%m-%d'),
            'trading_days': len(df)
        }
        
        # 价格统计
        stats['price'] = {
            'mean': df['close'].mean(),
            'std': df['close'].std(),
            'min': df['close'].min(),
            'max': df['close'].max(),
            'latest': df['close'].iloc[-1]
        }
        
        # 成交量统计
        stats['volume'] = {
            'mean': df['volume'].mean(),
            'std': df['volume'].std(),
            'min': df['volume'].min(),
            'max': df['volume'].max(),
            'latest': df['volume'].iloc[-1]
        }
        
        # 波动率
        stats['volatility'] = {
            'daily_returns_std': df['close'].pct_change().std(),
            'price_range_mean': ((df['high'] - df['low']) / df['low']).mean()
        }
        
        report['statistics'] = stats
    
    @staticmethod
    def _check_date_continuity(df: pd.DataFrame) -> List[str]:
        """检查交易日期的连续性"""
        if df.empty:
            return []
        
        # 获取所有交易日
        dates = df.index.sort_values()
        
        # 生成理论上应该存在的所有交易日
        all_dates = pd.date_range(start=dates.min(), end=dates.max(), freq='B')
        
        # 找出缺失的交易日
        missing_dates = all_dates.difference(dates)
        
        return [d.strftime('%Y-%m-%d') for d in missing_dates] 