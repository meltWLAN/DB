#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据质量检查器
提供数据一致性和完整性校验，以及异常检测与修复
"""

import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
import re
import warnings
from functools import wraps

from ...config.settings import DATA_QUALITY_CONFIG

# 设置日志
logger = logging.getLogger(__name__)

class DataQualityChecker:
    """
    数据质量检查器
    检查和修复常见的数据质量问题
    """
    
    def __init__(self):
        """初始化数据质量检查器"""
        # 启用数据验证
        self.validation_enabled = DATA_QUALITY_CONFIG.get('validation_enabled', True)
        # 启用自动修复
        self.auto_fix_enabled = DATA_QUALITY_CONFIG.get('auto_fix_enabled', True)
        # 质量阈值
        self.quality_threshold = DATA_QUALITY_CONFIG.get('validation_threshold', 0.95)
        
        # 验证规则
        self.validation_rules = {}
        
        # 初始化默认规则
        self._init_default_rules()
        
        logger.info("数据质量检查器初始化完成")
    
    def _init_default_rules(self):
        """初始化默认验证规则"""
        # 日期列验证
        self.add_validation_rule(
            'date', 
            lambda s: pd.to_datetime(s, errors='coerce').notna(),
            "日期格式无效"
        )
        
        # 开盘价验证 - 必须为正数
        self.add_validation_rule(
            'open',
            lambda s: (s > 0),
            "开盘价必须为正数"
        )
        
        # 收盘价验证 - 必须为正数
        self.add_validation_rule(
            'close',
            lambda s: (s > 0),
            "收盘价必须为正数"
        )
        
        # 最高价验证 - 必须大于等于收盘价和开盘价
        self.add_validation_rule(
            'high',
            lambda s, df: (s >= df['close']) & (s >= df['open']),
            "最高价必须大于等于收盘价和开盘价"
        )
        
        # 最低价验证 - 必须小于等于收盘价和开盘价
        self.add_validation_rule(
            'low',
            lambda s, df: (s <= df['close']) & (s <= df['open']),
            "最低价必须小于等于收盘价和开盘价"
        )
        
        # 成交量验证 - 必须为非负数
        self.add_validation_rule(
            'volume',
            lambda s: (s >= 0),
            "成交量必须为非负数"
        )
        
        # 成交额验证 - 必须为非负数
        self.add_validation_rule(
            'amount',
            lambda s: (s >= 0),
            "成交额必须为非负数"
        )
    
    def add_validation_rule(self, column: str, rule_func: Callable, error_message: str):
        """
        添加验证规则
        
        Args:
            column: 列名
            rule_func: 验证函数，接收Series和可选的DataFrame，返回布尔值Series
            error_message: 错误消息
        """
        if column not in self.validation_rules:
            self.validation_rules[column] = []
        
        self.validation_rules[column].append({
            'rule': rule_func,
            'message': error_message
        })
        
        logger.debug(f"添加验证规则: {column} - {error_message}")
    
    def validate_data(self, data: pd.DataFrame) -> Dict:
        """
        验证数据质量
        
        Args:
            data: 要验证的DataFrame
            
        Returns:
            Dict: 质量报告
        """
        if not self.validation_enabled:
            logger.info("数据验证已禁用")
            return {'validation_enabled': False}
        
        if data is None or data.empty:
            logger.warning("无数据可验证")
            return {'empty_data': True}
        
        # 准备质量报告
        quality_report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'row_count': len(data),
            'column_count': len(data.columns),
            'columns': list(data.columns),
            'missing_values': {},
            'validation_errors': {},
            'duplicate_rows': int(data.duplicated().sum()),
            'quality_score': 1.0,
            'passes_threshold': True
        }
        
        # 检查缺失值
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            if missing_count > 0:
                quality_report['missing_values'][column] = {
                    'count': int(missing_count),
                    'percentage': float((missing_count / len(data)) * 100)
                }
        
        # 应用验证规则
        total_validations = 0
        failed_validations = 0
        
        for column, rules in self.validation_rules.items():
            if column not in data.columns:
                continue
            
            quality_report['validation_errors'][column] = []
            
            for rule in rules:
                # 应用规则
                try:
                    if rule['rule'].__code__.co_argcount > 1:
                        # 如果规则函数需要额外的DataFrame参数
                        result = rule['rule'](data[column], data)
                    else:
                        result = rule['rule'](data[column])
                    
                    invalid_mask = ~result
                    invalid_count = invalid_mask.sum()
                    total_validations += len(data)
                    failed_validations += invalid_count
                    
                    if invalid_count > 0:
                        error_result = {
                            'message': rule['message'],
                            'count': int(invalid_count),
                            'percentage': float((invalid_count / len(data)) * 100),
                            'first_10_indices': data.index[invalid_mask].tolist()[:10]
                        }
                        quality_report['validation_errors'][column].append(error_result)
                except Exception as e:
                    logger.error(f"验证规则执行失败: {column} - {rule['message']}, 错误: {str(e)}")
        
        # 计算质量分数
        if total_validations > 0:
            quality_score = 1.0 - (failed_validations / total_validations)
            quality_report['quality_score'] = float(quality_score)
            quality_report['passes_threshold'] = quality_score >= self.quality_threshold
        
        # 汇总验证结果
        if not quality_report['passes_threshold']:
            logger.warning(f"数据质量低于阈值: {quality_report['quality_score']:.2f} < {self.quality_threshold}")
        
        return quality_report
    
    def fix_data_issues(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        修复数据问题
        
        Args:
            data: 要修复的DataFrame
            
        Returns:
            Tuple[DataFrame, Dict]: 修复后的数据和修复报告
        """
        if data is None or data.empty:
            return data, {'empty_data': True}
        
        if not self.auto_fix_enabled:
            logger.info("自动修复已禁用")
            return data, {'auto_fix_enabled': False}
        
        df = data.copy()
        
        # 准备修复报告
        fix_report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_row_count': len(data),
            'fixed_issues': {},
            'rows_modified': 0,
            'rows_dropped': 0
        }
        
        # 1. 删除重复行
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            df = df.drop_duplicates()
            fix_report['fixed_issues']['duplicates'] = {
                'count': int(duplicate_count),
                'action': 'dropped'
            }
            fix_report['rows_dropped'] += duplicate_count
            logger.info(f"删除了 {duplicate_count} 行重复数据")
        
        # 2. 修复日期列
        if 'date' in df.columns:
            invalid_dates = pd.to_datetime(df['date'], errors='coerce').isna()
            invalid_count = invalid_dates.sum()
            if invalid_count > 0:
                # 尝试修复日期格式
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                # 无法修复的日期，用前一个有效日期填充
                df['date'] = df['date'].fillna(method='ffill')
                # 依然无效的日期行删除
                still_invalid = df['date'].isna()
                if still_invalid.any():
                    df = df.dropna(subset=['date'])
                    fix_report['rows_dropped'] += still_invalid.sum()
                
                fix_report['fixed_issues']['invalid_dates'] = {
                    'count': int(invalid_count),
                    'action': 'fixed or dropped'
                }
                logger.info(f"修复了 {invalid_count} 个无效日期")
        
        # 3. 修复价格数据
        price_columns = ['open', 'high', 'low', 'close']
        for column in [col for col in price_columns if col in df.columns]:
            # 替换负值为NaN
            negative_mask = df[column] <= 0
            negative_count = negative_mask.sum()
            if negative_count > 0:
                df.loc[negative_mask, column] = np.nan
                fix_report['fixed_issues'][f'negative_{column}'] = {
                    'count': int(negative_count),
                    'action': 'set to NaN'
                }
                logger.info(f"将 {negative_count} 个负 {column} 值设为NaN")
        
        # 4. 确保价格的逻辑关系正确
        if all(col in df.columns for col in price_columns):
            # 修复high < open/close的情况
            invalid_high = (df['high'] < df['open']) | (df['high'] < df['close'])
            invalid_high_count = invalid_high.sum()
            if invalid_high_count > 0:
                df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close']].max(axis=1)
                fix_report['fixed_issues']['invalid_high'] = {
                    'count': int(invalid_high_count),
                    'action': 'set to max(open, close)'
                }
                logger.info(f"修复了 {invalid_high_count} 行最高价不符合逻辑关系的数据")
            
            # 修复low > open/close的情况
            invalid_low = (df['low'] > df['open']) | (df['low'] > df['close'])
            invalid_low_count = invalid_low.sum()
            if invalid_low_count > 0:
                df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'close']].min(axis=1)
                fix_report['fixed_issues']['invalid_low'] = {
                    'count': int(invalid_low_count),
                    'action': 'set to min(open, close)'
                }
                logger.info(f"修复了 {invalid_low_count} 行最低价不符合逻辑关系的数据")
        
        # 5. 填充缺失值
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                # 根据列类型选择填充方法
                if column in price_columns:
                    # 价格列用前值填充
                    df[column] = df[column].fillna(method='ffill')
                    still_missing = df[column].isnull().sum()
                    if still_missing > 0:
                        # 如果仍有缺失，使用后值填充
                        df[column] = df[column].fillna(method='bfill')
                elif column == 'volume' or column == 'amount':
                    # 成交量和成交额用0填充
                    df[column] = df[column].fillna(0)
                else:
                    # 其他列用前值填充
                    df[column] = df[column].fillna(method='ffill')
                
                fix_report['fixed_issues'][f'missing_{column}'] = {
                    'count': int(missing_count),
                    'action': 'filled'
                }
                logger.info(f"填充了 {column} 列的 {missing_count} 个缺失值")
        
        # 6. 检测数值列的异常值并处理
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if column in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                # 使用IQR方法检测异常值
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # 设置上下限
                lower_bound = Q1 - 3 * IQR  # 使用3倍IQR作为阈值
                upper_bound = Q3 + 3 * IQR
                
                # 检测异常值
                outliers = ((df[column] < lower_bound) | (df[column] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    if column in ['volume', 'amount']:
                        # 成交量和成交额异常值直接截断
                        df.loc[df[column] < lower_bound, column] = 0
                        df.loc[df[column] > upper_bound, column] = upper_bound
                    else:
                        # 价格异常值使用移动中位数替换
                        window_size = 5
                        df.loc[outliers, column] = df[column].rolling(window=window_size, center=True, min_periods=1).median().loc[outliers]
                    
                    fix_report['fixed_issues'][f'outliers_{column}'] = {
                        'count': int(outlier_count),
                        'action': 'fixed'
                    }
                    logger.info(f"修复了 {column} 列的 {outlier_count} 个异常值")
        
        # 统计修改的行数
        fix_report['rows_modified'] = len(data) - fix_report['rows_dropped'] - len(df)
        fix_report['final_row_count'] = len(df)
        
        return df, fix_report
    
    def process_data_quality(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        处理数据质量，包括验证和修复
        
        Args:
            data: 要处理的DataFrame
            
        Returns:
            Tuple[DataFrame, Dict]: 处理后的数据和质量报告
        """
        if data is None or data.empty:
            return data, {'empty_data': True}
        
        # 1. 验证数据
        validation_report = self.validate_data(data)
        
        # 2. 如果需要修复，则进行修复
        fixed_data, fix_report = self.fix_data_issues(data) if self.auto_fix_enabled else (data, {'auto_fix_enabled': False})
        
        # 3. 合并报告
        quality_report = {
            'validation': validation_report,
            'fix': fix_report if self.auto_fix_enabled else {'auto_fix_enabled': False}
        }
        
        return fixed_data, quality_report
    
    def export_report(self, report: Dict, file_path: str):
        """
        导出质量报告到文件
        
        Args:
            report: 质量报告字典
            file_path: 导出文件路径
        """
        try:
            # 将NumPy数据类型转换为Python原生类型
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                                    np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                else:
                    return obj
            
            # 转换报告中的NumPy类型
            report_converted = convert_numpy_types(report)
            
            with open(file_path, 'w') as f:
                json.dump(report_converted, f, indent=2)
            logger.info(f"质量报告已导出到 {file_path}")
        except Exception as e:
            logger.error(f"导出质量报告失败: {str(e)}") 