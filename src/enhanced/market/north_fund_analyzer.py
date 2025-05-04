#!/usr/bin/env python3
"""
北向资金分析器模块
提供北向资金流向的高级分析功能
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
from functools import wraps

# 配置日志
logger = logging.getLogger(__name__)

def with_cache(cache_file=None, expire_hours=24):
    """缓存装饰器，用于缓存函数结果"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 如果未提供缓存文件，则使用默认缓存文件
            nonlocal cache_file
            if cache_file is None:
                instance = args[0]
                if hasattr(instance, 'cache_dir'):
                    func_name = func.__name__
                    cache_file = os.path.join(instance.cache_dir, f"{func_name}_cache.pkl")
            
            # 如果缓存文件存在且未过期，则返回缓存结果
            if os.path.exists(cache_file):
                file_mtime = os.path.getmtime(cache_file)
                file_datetime = datetime.fromtimestamp(file_mtime)
                now = datetime.now()
                time_diff = now - file_datetime
                
                if time_diff.total_seconds() < expire_hours * 3600:
                    try:
                        with open(cache_file, 'rb') as f:
                            result = pickle.load(f)
                            logger.debug(f"使用缓存结果: {cache_file}")
                            return result
                    except Exception as e:
                        logger.warning(f"读取缓存失败: {e}")
            
            # 执行原始函数
            result = func(*args, **kwargs)
            
            # 将结果保存到缓存
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                    logger.debug(f"已缓存结果到: {cache_file}")
            except Exception as e:
                logger.warning(f"缓存结果失败: {e}")
            
            return result
        
        return wrapper
    
    return decorator

class NorthFundAnalyzer:
    """北向资金分析器
    提供北向资金流向的各种分析功能
    """
    
    def __init__(self, cache_dir="cache/north_fund"):
        """初始化北向资金分析器
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"北向资金分析器初始化完成，缓存目录: {self.cache_dir}")
    
    @with_cache(expire_hours=24)
    def get_north_fund_flow(self, start_date=None, end_date=None):
        """获取北向资金流向数据
        
        Args:
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'，默认为今天
            
        Returns:
            pandas.DataFrame: 北向资金流向数据
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
        logger.info(f"获取北向资金流向数据: {start_date} 至 {end_date}")
        
        try:
            # 实际实现中应该从数据源获取数据
            # TODO: 实现真实数据获取
            # 从API或本地数据源获取北向资金流向数据
            
            # 如果无法获取数据，返回空DataFrame
            logger.warning("无法获取北向资金流向数据，返回空DataFrame")
            return pd.DataFrame(columns=['trade_date', 'north_money', 'south_money', 'north_acc', 'south_acc'])
            
        except Exception as e:
            logger.error(f"获取北向资金流向数据失败: {str(e)}")
            return pd.DataFrame(columns=['trade_date', 'north_money', 'south_money', 'north_acc', 'south_acc'])
    
    @with_cache(expire_hours=24)
    def get_sector_allocation(self, date=None):
        """获取北向资金行业配置数据
        
        Args:
            date: 日期，格式为'YYYY-MM-DD'，默认为最新交易日
            
        Returns:
            pandas.DataFrame: 行业配置数据
        """
        logger.info(f"获取北向资金行业配置数据: {date}")
        
        try:
            # 实际实现中应该从数据源获取数据
            # TODO: 实现真实数据获取
            # 从API或本地数据源获取北向资金行业配置数据
            
            # 如果无法获取数据，返回空DataFrame
            logger.warning("无法获取北向资金行业配置数据，返回空DataFrame")
            return pd.DataFrame(columns=['industry', 'weight', 'stock_count', 'total_ratio', 'total_vol'])
            
        except Exception as e:
            logger.error(f"获取北向资金行业配置数据失败: {str(e)}")
            return pd.DataFrame(columns=['industry', 'weight', 'stock_count', 'total_ratio', 'total_vol'])
    
    def analyze_fund_flow_trend(self, days=30):
        """分析北向资金流向趋势
        
        Args:
            days: 分析天数
            
        Returns:
            dict: 趋势分析结果
        """
        logger.info(f"分析北向资金流向趋势，天数: {days}")
        
        try:
            # 获取资金流向数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df = self.get_north_fund_flow(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # 如果数据为空，返回空结果
            if df.empty:
                logger.warning("北向资金流向数据为空，无法分析趋势")
                return {
                    'latest_net_flow': 0,
                    'avg_net_flow': 0,
                    'trend': "未知",
                    'trend_strength': 0,
                    'consecutive_days': 0,
                    'correlation_with_market': 0,
                    'analysis_period': days
                }
            
            # 确保数据按日期排序
            df = df.sort_values('trade_date')
            
            # 提取最近的资金流向数据
            recent_df = df.tail(days)
            
            # 计算最新净流入
            latest_net_flow = recent_df['north_money'].iloc[-1]
            
            # 计算平均净流入
            avg_net_flow = recent_df['north_money'].mean()
            
            # 判断趋势
            # 使用简单的线性回归斜率判断趋势
            x = np.arange(len(recent_df))
            y = recent_df['north_money'].values
            slope, _ = np.polyfit(x, y, 1)
            
            if slope > 0:
                trend = "上升"
                # 计算趋势强度 - 斜率相对于平均值的比例
                trend_strength = min(100, max(0, (slope / avg_net_flow) * 100)) if avg_net_flow > 0 else 50
            else:
                trend = "下降"
                # 计算趋势强度 - 斜率相对于平均值的比例，取绝对值
                trend_strength = min(100, max(0, (-slope / avg_net_flow) * 100)) if avg_net_flow > 0 else 50
            
            # 计算连续流入/流出天数
            flows = recent_df['north_money'].values
            consecutive_days = 1
            for i in range(len(flows)-2, -1, -1):
                if (flows[i] > 0 and flows[-1] > 0) or (flows[i] < 0 and flows[-1] < 0):
                    consecutive_days += 1
                else:
                    break
            
            # 计算与大盘的相关性
            corr_with_market = 0
            
            # 返回分析结果
            result = {
                'latest_net_flow': latest_net_flow,
                'avg_net_flow': avg_net_flow,
                'trend': trend,
                'trend_strength': trend_strength,
                'consecutive_days': consecutive_days,
                'correlation_with_market': corr_with_market,
                'analysis_period': days
            }
            
            return result
            
        except Exception as e:
            logger.error(f"分析北向资金流向趋势失败: {str(e)}")
            return {
                'latest_net_flow': 0,
                'avg_net_flow': 0,
                'trend': "未知",
                'trend_strength': 0,
                'consecutive_days': 0,
                'correlation_with_market': 0,
                'analysis_period': days,
                'error': str(e)
            }
    
    def plot_fund_flow_trend(self, days=30, save_path=None):
        """绘制北向资金流向趋势图
        
        Args:
            days: 天数
            save_path: 保存路径
            
        Returns:
            str: 图表保存路径
        """
        logger.info(f"绘制北向资金流向趋势图，天数: {days}")
        
        # 获取资金流向数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = self.get_north_fund_flow(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if df.empty:
            logger.warning("北向资金流向数据为空，无法绘制趋势图")
            return None
        
        # 确保数据按日期排序
        df = df.sort_values('trade_date')
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 创建两个子图
        gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
        
        # 第一个子图：日净流入
        ax1 = plt.subplot(gs[0])
        
        # 绘制日净流入柱状图
        colors = np.where(df['north_money'] > 0, 'red', 'green')
        ax1.bar(df['trade_date'], df['north_money'], color=colors, alpha=0.7)
        
        # 添加标题和标签
        ax1.set_title('北向资金日净流入', fontsize=14)
        ax1.set_ylabel('净流入金额（万元）', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        plt.xticks(rotation=45)
        
        # 第二个子图：累计净流入
        ax2 = plt.subplot(gs[1])
        
        # 绘制累计净流入曲线
        ax2.plot(df['trade_date'], df['north_acc'], 'b-', linewidth=2)
        
        # 添加标题和标签
        ax2.set_title('北向资金累计净流入', fontsize=14)
        ax2.set_ylabel('累计金额（万元）', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        plt.xticks(rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"已保存趋势图至: {save_path}")
            except Exception as e:
                logger.error(f"保存趋势图失败: {e}")
                save_path = None
        
        # 关闭图表
        plt.close()
        
        return save_path

    def plot_sector_allocation(self, top_n=10, save_path=None):
        """绘制北向资金行业配置图
        
        Args:
            top_n: 显示前N个行业
            save_path: 保存路径
            
        Returns:
            str: 图表保存路径
        """
        logger.info(f"绘制北向资金行业配置图，top_n: {top_n}")
        
        # 获取行业配置数据
        df = self.get_sector_allocation()
        
        if df.empty:
            logger.warning("北向资金行业配置数据为空，无法绘制配置图")
            return None
        
        # 按权重排序
        df = df.sort_values('weight', ascending=False).head(top_n)
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 创建柱状图
        sns.barplot(x='weight', y='industry', data=df, palette='viridis')
        
        # 添加标题和标签
        plt.title('北向资金行业配置（前{}名）'.format(top_n), fontsize=14)
        plt.xlabel('权重 (%)', fontsize=12)
        plt.ylabel('行业', fontsize=12)
        
        # 添加数值标签
        for i, v in enumerate(df['weight']):
            plt.text(v + 0.2, i, f"{v:.2f}%", va='center')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"已保存配置图至: {save_path}")
            except Exception as e:
                logger.error(f"保存配置图失败: {e}")
                save_path = None
        
        # 关闭图表
        plt.close()
        
        return save_path
    
    def analyze_sector_changes(self, period_days=30):
        """分析行业配置变化
        
        Args:
            period_days: 比较周期（天）
            
        Returns:
            dict: 分析结果
        """
        logger.info(f"分析行业配置变化，周期: {period_days}天")
        
        # 获取当前行业配置
        current_allocation = self.get_sector_allocation()
        
        if current_allocation.empty:
            logger.warning("当前行业配置数据为空，无法分析变化")
            return {
                'period_days': period_days,
                'top_increasing': [],
                'top_decreasing': [],
                'status': '数据不足'
            }
        
        # 获取之前的行业配置（实际应用中需要从数据源获取历史数据）
        # 在此返回空结果，表示无法获取历史数据
        logger.warning("无法获取历史行业配置数据，无法分析变化")
        return {
            'period_days': period_days,
            'top_increasing': [],
            'top_decreasing': [],
            'status': '数据不足'
        }
    
    def get_top_increasing_sectors(self, top_n=5):
        """获取配置比例增加最多的行业
        
        Args:
            top_n: 返回前N个行业
            
        Returns:
            list: 行业列表
        """
        logger.info(f"获取配置比例增加最多的行业，top_n: {top_n}")
        
        # 分析行业变化
        changes = self.analyze_sector_changes()
        
        # 返回配置增加最多的行业
        return changes.get('top_increasing', [])
    
    def get_top_decreasing_sectors(self, top_n=5):
        """获取配置比例减少最多的行业
        
        Args:
            top_n: 返回前N个行业
            
        Returns:
            list: 行业列表
        """
        logger.info(f"获取配置比例减少最多的行业，top_n: {top_n}")
        
        # 分析行业变化
        changes = self.analyze_sector_changes()
        
        # 返回配置减少最多的行业
        return changes.get('top_decreasing', [])
    
    def generate_fund_flow_report(self, days=30, save_path=None):
        """生成北向资金流向分析报告
        
        Args:
            days: 分析天数
            save_path: 报告保存路径
            
        Returns:
            dict: 报告内容
        """
        logger.info(f"生成北向资金流向分析报告，天数: {days}")
        
        # 分析资金流向趋势
        trend_analysis = self.analyze_fund_flow_trend(days)
        
        # 绘制资金流向趋势图
        if save_path:
            trend_chart_path = os.path.join(os.path.dirname(save_path), 'north_fund_trend.png')
        else:
            trend_chart_path = os.path.join(self.cache_dir, 'north_fund_trend.png')
        
        self.plot_fund_flow_trend(days, trend_chart_path)
        
        # 分析行业配置
        sector_allocation = self.get_sector_allocation()
        
        # 绘制行业配置图
        if save_path:
            sector_chart_path = os.path.join(os.path.dirname(save_path), 'sector_allocation.png')
        else:
            sector_chart_path = os.path.join(self.cache_dir, 'sector_allocation.png')
        
        self.plot_sector_allocation(10, sector_chart_path)
        
        # 分析行业变化
        sector_changes = self.analyze_sector_changes()
        
        # 生成报告内容
        report = {
            'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_period': days,
            'fund_flow_trend': trend_analysis,
            'sector_allocation': sector_allocation.to_dict('records') if not sector_allocation.empty else [],
            'sector_changes': sector_changes,
            'charts': {
                'trend_chart': trend_chart_path,
                'sector_chart': sector_chart_path
            }
        }
        
        # 保存报告
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(report, f, ensure_ascii=False, indent=2)
                logger.info(f"已保存报告至: {save_path}")
            except Exception as e:
                logger.error(f"保存报告失败: {e}")
        
        return report 