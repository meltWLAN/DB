#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据质量检查模块
提供一系列检查方法验证股票数据的完整性和质量
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings

# 配置日志
logger = logging.getLogger(__name__)

class DataQualityChecker:
    """数据质量检查器类"""
    
    def __init__(self, thresholds=None):
        """初始化数据质量检查器
        
        Args:
            thresholds: 质量检查阈值字典，为None时使用默认值
        """
        # 默认质量检查阈值
        self.default_thresholds = {
            'min_data_points': 20,          # 最少数据点数量
            'max_missing_pct': 5.0,         # 最大缺失值百分比
            'max_outlier_z_score': 3.0,     # 最大异常值Z分数
            'min_trading_days_ratio': 0.7,  # 最小交易日比例
            'max_price_gap_pct': 20.0,      # 最大价格跳跃百分比
            'min_volume': 50000,            # 最小成交量
            'max_high_low_ratio': 15.0,     # 最大高低价比率
            'max_price_zero_pct': 1.0,      # 最大价格为零的百分比
            'min_coverage_days': 60,        # 最小覆盖天数
            'max_duplicate_pct': 0.5        # 最大重复数据百分比
        }
        
        # 使用自定义阈值覆盖默认值
        self.thresholds = self.default_thresholds.copy()
        if thresholds:
            for key, value in thresholds.items():
                if key in self.thresholds:
                    self.thresholds[key] = value
        
        logger.info("数据质量检查器初始化完成")
    
    def check_data_quality(self, df, code=None, full_check=True):
        """检查数据质量
        
        Args:
            df: 股票数据DataFrame
            code: 股票代码，可选
            full_check: 是否进行完整检查
            
        Returns:
            dict: 检查结果，包含pass, score, details和reason
        """
        if df is None or df.empty:
            return {
                'pass': False,
                'score': 0,
                'reason': "数据为空",
                'details': {},
                'code': code
            }
        
        problems = []
        scores = {}
        
        # 基本检查 - 所有情况都会进行
        self._check_data_points(df, problems, scores)
        self._check_missing_values(df, problems, scores)
        
        # 仅当数据不为空且包含必要字段时进行进一步检查
        if full_check and len(df) > 0:
            if 'close' in df.columns:
                self._check_price_outliers(df, problems, scores)
                self._check_price_zeros(df, problems, scores)
                
                if 'high' in df.columns and 'low' in df.columns:
                    self._check_high_low_ratio(df, problems, scores)
            
            if 'vol' in df.columns or 'volume' in df.columns:
                self._check_volume(df, problems, scores)
            
            # 如果df有日期索引或日期列，检查交易日连续性
            if df.index.name == 'trade_date' or 'trade_date' in df.columns:
                self._check_trading_continuity(df, problems, scores)
                self._check_coverage_days(df, problems, scores)
            
            # 检查重复数据
            self._check_duplicates(df, problems, scores)
        
        # 计算总分
        if scores:
            total_score = sum(scores.values()) / len(scores)
        else:
            total_score = 0
        
        # 结果
        result = {
            'pass': len(problems) == 0,
            'score': total_score,
            'reason': "; ".join(problems) if problems else "数据质量良好",
            'details': scores,
            'code': code
        }
        
        return result
    
    def _check_data_points(self, df, problems, scores):
        """检查数据点数量"""
        min_points = self.thresholds['min_data_points']
        if len(df) < min_points:
            problems.append(f"数据点数量不足 (实际: {len(df)}, 最少需要: {min_points})")
            scores['data_points'] = 0
        else:
            scores['data_points'] = min(100, int(len(df) / min_points * 100))
    
    def _check_missing_values(self, df, problems, scores):
        """检查缺失值百分比"""
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        max_missing = self.thresholds['max_missing_pct']
        if missing_pct > max_missing:
            problems.append(f"缺失值过多 (实际: {missing_pct:.2f}%, 最大允许: {max_missing}%)")
            scores['missing_data'] = 0
        else:
            scores['missing_data'] = int(100 - (missing_pct / max_missing * 100))
    
    def _check_price_outliers(self, df, problems, scores):
        """使用Z分数检测异常值"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            close_prices = df['close'].dropna()
            if len(close_prices) > 0:
                # 使用Z分数检测异常值
                z_scores = np.abs((close_prices - close_prices.mean()) / close_prices.std())
                max_z = self.thresholds['max_outlier_z_score']
                outliers = z_scores[z_scores > max_z]
                if len(outliers) > 0:
                    outlier_pct = len(outliers) / len(close_prices) * 100
                    problems.append(f"存在异常值 (异常值比例: {outlier_pct:.2f}%, 最大Z分数: {z_scores.max():.2f})")
                    scores['outliers'] = int(100 - (outlier_pct * 10))  # 异常值每1%扣10分
                else:
                    scores['outliers'] = 100
    
    def _check_trading_continuity(self, df, problems, scores):
        """检查交易日连续性"""
        # 获取日期列
        if df.index.name == 'trade_date':
            dates = df.index
        else:
            dates = pd.to_datetime(df['trade_date'])
        
        # 排序日期
        dates = sorted(dates)
        
        if len(dates) > 1:
            # 检查日期间隔
            date_diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            
            # 排除周末和节假日，认为3天以内的差距是正常的
            normal_diffs = [diff for diff in date_diffs if diff <= 3]
            
            # 计算连续性比例
            continuity_ratio = len(normal_diffs) / len(date_diffs)
            min_ratio = self.thresholds['min_trading_days_ratio']
            
            if continuity_ratio < min_ratio:
                problems.append(f"交易日不连续 (连续交易日比例: {continuity_ratio:.2f}, 最小要求: {min_ratio})")
                scores['continuity'] = int(continuity_ratio / min_ratio * 100)
            else:
                scores['continuity'] = 100
    
    def _check_volume(self, df, problems, scores):
        """检查成交量"""
        # 确定使用哪个列作为成交量
        vol_col = 'vol' if 'vol' in df.columns else 'volume'
        
        # 检查平均成交量
        mean_volume = df[vol_col].mean()
        min_volume = self.thresholds['min_volume']
        
        if mean_volume < min_volume:
            problems.append(f"平均成交量过低 (实际: {mean_volume:.2f}, 最小要求: {min_volume})")
            vol_score = int(mean_volume / min_volume * 100)
            scores['volume'] = max(0, min(100, vol_score))
        else:
            scores['volume'] = 100
    
    def _check_high_low_ratio(self, df, problems, scores):
        """检查最高价和最低价的比率"""
        # 计算每日最高价/最低价比率
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratio = df['high'] / df['low'].replace(0, np.nan)
            max_ratio = ratio.max()
            
            threshold = self.thresholds['max_high_low_ratio']
            if max_ratio > threshold:
                problems.append(f"最高价/最低价比率异常 (最大比率: {max_ratio:.2f}, 最大允许: {threshold})")
                scores['high_low_ratio'] = int(100 - (max_ratio / threshold - 1) * 100)
            else:
                scores['high_low_ratio'] = 100
    
    def _check_price_zeros(self, df, problems, scores):
        """检查价格为零的情况"""
        zero_prices = (df['close'] == 0).sum()
        zero_pct = zero_prices / len(df) * 100
        
        max_zero_pct = self.thresholds['max_price_zero_pct']
        if zero_pct > max_zero_pct:
            problems.append(f"存在价格为零的数据 (比例: {zero_pct:.2f}%, 最大允许: {max_zero_pct}%)")
            scores['zero_prices'] = int(100 - (zero_pct / max_zero_pct * 100))
        else:
            scores['zero_prices'] = 100
    
    def _check_coverage_days(self, df, problems, scores):
        """检查数据覆盖的天数"""
        # 获取日期列
        if df.index.name == 'trade_date':
            dates = df.index
        else:
            dates = pd.to_datetime(df['trade_date'])
        
        # 计算覆盖天数
        if len(dates) > 0:
            date_range = (max(dates) - min(dates)).days
            min_days = self.thresholds['min_coverage_days']
            
            if date_range < min_days:
                problems.append(f"数据覆盖天数不足 (实际: {date_range}天, 最少需要: {min_days}天)")
                scores['coverage'] = int(date_range / min_days * 100)
            else:
                scores['coverage'] = 100
    
    def _check_duplicates(self, df, problems, scores):
        """检查重复数据"""
        dup_count = df.duplicated().sum()
        dup_pct = dup_count / len(df) * 100
        
        max_dup_pct = self.thresholds['max_duplicate_pct']
        if dup_pct > max_dup_pct:
            problems.append(f"存在重复数据 (比例: {dup_pct:.2f}%, 最大允许: {max_dup_pct}%)")
            scores['duplicates'] = int(100 - (dup_pct / max_dup_pct * 100))
        else:
            scores['duplicates'] = 100

    def get_quality_report(self, df, code=None):
        """生成详细的质量报告
        
        Args:
            df: 股票数据DataFrame
            code: 股票代码，可选
            
        Returns:
            dict: 详细的质量报告
        """
        # 基础质量检查
        basic_check = self.check_data_quality(df, code)
        
        # 扩展报告数据
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'code': code,
            'basic_result': basic_check,
            'data_stats': {
                'row_count': len(df),
                'column_count': len(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percent': (df.isnull().sum() / len(df) * 100).to_dict(),
                'data_types': df.dtypes.astype(str).to_dict()
            }
        }
        
        # 添加数值列统计
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            report['numeric_stats'] = {}
            for col in numeric_cols:
                if col in df.columns:
                    series = df[col].dropna()
                    if len(series) > 0:
                        report['numeric_stats'][col] = {
                            'min': float(series.min()),
                            'max': float(series.max()),
                            'mean': float(series.mean()),
                            'median': float(series.median()),
                            'std': float(series.std()),
                            'zeros_count': int((series == 0).sum()),
                            'zeros_percent': float((series == 0).sum() / len(series) * 100)
                        }
        
        # 如果有日期数据，添加日期统计
        date_col = None
        if df.index.name == 'trade_date':
            date_col = df.index
        elif 'trade_date' in df.columns:
            date_col = pd.to_datetime(df['trade_date'])
        
        if date_col is not None:
            dates = sorted(date_col)
            if len(dates) > 0:
                report['date_stats'] = {
                    'start_date': min(dates).strftime('%Y-%m-%d'),
                    'end_date': max(dates).strftime('%Y-%m-%d'),
                    'date_range_days': (max(dates) - min(dates)).days,
                    'avg_days_between': 0
                }
                
                if len(dates) > 1:
                    diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                    report['date_stats']['avg_days_between'] = sum(diffs) / len(diffs)
                    report['date_stats']['max_days_between'] = max(diffs)
                    report['date_stats']['data_completeness'] = len(dates) / (report['date_stats']['date_range_days'] + 1)
        
        return report

# 创建单例实例
data_quality_checker = DataQualityChecker()

if __name__ == "__main__":
    # 测试代码
    import pandas as pd
    import numpy as np
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据集
    # 生成一个正常的股票数据
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='B')
    base_price = 100
    
    data = {
        'trade_date': dates,
        'open': [base_price + i * 0.1 + np.random.normal(0, 1) for i in range(100)],
        'high': [base_price + i * 0.1 + 2 + np.random.normal(0, 1) for i in range(100)],
        'low': [base_price + i * 0.1 - 2 + np.random.normal(0, 1) for i in range(100)],
        'close': [base_price + i * 0.1 + np.random.normal(0, 1) for i in range(100)],
        'vol': [100000 + np.random.randint(-50000, 50000) for _ in range(100)],
        'amount': [1000000 + np.random.randint(-500000, 500000) for _ in range(100)]
    }
    
    good_df = pd.DataFrame(data)
    good_df.set_index('trade_date', inplace=True)
    
    # 创建一个有问题的数据集
    bad_data = data.copy()
    # 添加缺失值
    for col in ['open', 'high', 'low', 'close']:
        for i in range(10):
            idx = np.random.randint(0, 100)
            bad_data[col][idx] = np.nan
    
    # 添加异常值
    bad_data['close'][50] = base_price * 5  # 异常收盘价
    bad_data['vol'][30] = 0  # 成交量为0
    
    # 添加时间断层
    bad_dates = dates.tolist()
    for i in range(5):
        idx = np.random.randint(20, 80)
        bad_dates.pop(idx)
    
    bad_data['trade_date'] = bad_dates[:95]  # 使长度匹配
    
    bad_df = pd.DataFrame(bad_data)
    bad_df.set_index('trade_date', inplace=True)
    
    # 创建检查器
    checker = DataQualityChecker()
    
    # 检查好的数据
    print("检查良好数据...")
    good_result = checker.check_data_quality(good_df, 'TEST001')
    print(f"检查结果: {'通过' if good_result['pass'] else '不通过'}")
    print(f"质量得分: {good_result['score']:.2f}")
    print(f"原因: {good_result['reason']}")
    
    # 检查有问题的数据
    print("\n检查有问题数据...")
    bad_result = checker.check_data_quality(bad_df, 'TEST002')
    print(f"检查结果: {'通过' if bad_result['pass'] else '不通过'}")
    print(f"质量得分: {bad_result['score']:.2f}")
    print(f"原因: {bad_result['reason']}")
    
    # 生成详细报告
    print("\n生成详细质量报告...")
    report = checker.get_quality_report(bad_df, 'TEST002')
    print(f"报告时间: {report['timestamp']}")
    print(f"行数: {report['data_stats']['row_count']}")
    print(f"列数: {report['data_stats']['column_count']}")
    
    if 'date_stats' in report:
        print(f"数据开始日期: {report['date_stats']['start_date']}")
        print(f"数据结束日期: {report['date_stats']['end_date']}")
        print(f"数据范围天数: {report['date_stats']['date_range_days']}")
        print(f"平均间隔天数: {report['date_stats']['avg_days_between']:.2f}") 