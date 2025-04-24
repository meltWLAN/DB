#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图表生成模块，用于生成股票分析可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf
import seaborn as sns
from datetime import datetime
import os
import logging


class ChartGenerator:
    """图表生成器类"""
    
    def __init__(self, output_dir='charts', dpi=100, style='default'):
        """初始化图表生成器
        
        Args:
            output_dir: 图表输出目录
            dpi: 图表分辨率
            style: 图表样式，可选 'default', 'dark', 'light'
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.style = style
        self.logger = logging.getLogger(__name__)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图表样式
        if style == 'dark':
            plt.style.use('dark_background')
        elif style == 'light':
            plt.style.use('seaborn-v0_8-bright')
    
    def format_date_axis(self, ax):
        """格式化日期轴
        
        Args:
            ax: matplotlib轴对象
        """
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def generate_candlestick_chart(self, data, title=None, indicators=None, save_path=None):
        """生成K线图
        
        Args:
            data: DataFrame，包含OHLCV数据
            title: 图表标题
            indicators: 要显示的指标列表，如 ['MA5', 'MA10', 'MA20']
            save_path: 保存路径，如果提供则保存图表
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        # 准备数据
        df = data.copy()
        
        # 确保日期是正确的格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # 确保OHLC数据列名符合mplfinance要求
        col_map = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        # 重命名列
        for old, new in col_map.items():
            if old in df.columns:
                df[new] = df[old]
        
        # 设置mpf样式
        mpf_style = mpf.make_mpf_style(
            marketcolors=mpf.make_marketcolors(
                up='red', down='green',
                edge={'up': 'red', 'down': 'green'},
                wick={'up': 'red', 'down': 'green'},
                volume={'up': 'red', 'down': 'green'}
            ),
            gridstyle=':',
            gridaxis='both'
        )
        
        # 准备额外指标
        addon_plots = []
        if indicators:
            for indicator in indicators:
                if indicator in df.columns:
                    addon_plots.append(
                        mpf.make_addplot(df[indicator], panel=0, secondary_y=False, 
                                          color='blue' if indicator == 'MA5' else 
                                                'orange' if indicator == 'MA10' else 
                                                'purple' if indicator == 'MA20' else 'gray')
                    )
        
        # 生成K线图
        fig, axes = mpf.plot(
            df,
            type='candle',
            style=mpf_style,
            title=title or '股票K线图',
            ylabel='价格',
            ylabel_lower='成交量',
            volume=True,
            figsize=(12, 8),
            addplot=addon_plots if addon_plots else None,
            returnfig=True
        )
        
        # 保存图表
        if save_path:
            path = os.path.join(self.output_dir, save_path)
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"K线图已保存至 {path}")
        
        return fig
    
    def generate_technical_chart(self, data, title=None, indicators=None, save_path=None):
        """生成技术指标图表
        
        Args:
            data: DataFrame，包含股票数据和技术指标
            title: 图表标题
            indicators: 要显示的指标配置，例如：
                       {'panel1': ['close', 'MA5', 'MA10'], 
                        'panel2': ['volume'], 
                        'panel3': ['macd', 'signal']}
            save_path: 保存路径，如果提供则保存图表
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        # 准备数据
        df = data.copy()
        
        # 确保日期是正确的格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            df['date'] = pd.date_range(start='2020-01-01', periods=len(df))
        
        # 设置默认指标
        if indicators is None:
            indicators = {
                'panel1': ['close', 'MA5', 'MA10', 'MA20'],
                'panel2': ['volume'],
                'panel3': ['macd', 'signal', 'hist']
            }
        
        # 计算所需的面板数量
        panel_count = len(indicators)
        
        # 创建图表和子图
        fig, axes = plt.subplots(panel_count, 1, figsize=(12, 8 + 2 * panel_count),
                               gridspec_kw={'height_ratios': [3] + [1] * (panel_count - 1)})
        
        if panel_count == 1:
            axes = [axes]
        
        # 设置图表标题
        if title:
            fig.suptitle(title, fontsize=16)
        
        # 遍历每个面板绘制指标
        for i, (panel_name, panel_indicators) in enumerate(indicators.items()):
            ax = axes[i]
            
            # 绘制该面板的指标
            for indicator in panel_indicators:
                if indicator in df.columns:
                    # 特殊处理MACD柱状图
                    if indicator == 'hist':
                        ax.bar(df['date'], df[indicator], color=['g' if x < 0 else 'r' for x in df[indicator]], 
                              alpha=0.5, label='MACD Histogram')
                    # 特殊处理成交量
                    elif indicator == 'volume':
                        ax.bar(df['date'], df[indicator], color=['g' if df['close'].iloc[i] < df['close'].iloc[i-1] 
                                                         else 'r' for i in range(1, len(df))], 
                              alpha=0.5, label='Volume')
                    # 其他线图指标
                    else:
                        ax.plot(df['date'], df[indicator], label=indicator.upper())
            
            # 设置x轴格式
            self.format_date_axis(ax)
            
            # 添加网格和图例
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper left')
            
            # 设置y轴标签
            if panel_name == 'panel1':
                ax.set_ylabel('价格')
            elif 'volume' in panel_indicators:
                ax.set_ylabel('成交量')
            elif 'macd' in panel_indicators:
                ax.set_ylabel('MACD')
            elif 'rsi' in panel_indicators:
                ax.set_ylabel('RSI')
                ax.axhline(y=70, color='r', linestyle='-', alpha=0.3)
                ax.axhline(y=30, color='g', linestyle='-', alpha=0.3)
            else:
                ax.set_ylabel(panel_name)
        
        # 设置最后一个面板的x轴标签
        axes[-1].set_xlabel('日期')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            path = os.path.join(self.output_dir, save_path)
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"技术指标图表已保存至 {path}")
        
        return fig
    
    def generate_comparison_chart(self, data_dict, title=None, save_path=None):
        """生成多只股票比较图表
        
        Args:
            data_dict: 字典，键为股票名称，值为包含股票数据的DataFrame
            title: 图表标题
            save_path: 保存路径，如果提供则保存图表
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制每只股票的价格走势
        for stock_name, df in data_dict.items():
            # 确保日期是正确的格式
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # 标准化价格为基准日的百分比变化
            base_price = df.iloc[0]['close']
            normalized = df['close'] / base_price * 100 - 100
            
            # 绘制线图
            ax.plot(df['date'], normalized, label=f"{stock_name} ({df.iloc[-1]['close']:.2f})")
        
        # 设置图表标题
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('股票价格对比图 (基准日=0%)', fontsize=14)
        
        # 设置轴标签
        ax.set_xlabel('日期')
        ax.set_ylabel('价格变化 (%)')
        
        # 设置x轴格式
        self.format_date_axis(ax)
        
        # 添加网格和图例
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')
        
        # 添加0线
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            path = os.path.join(self.output_dir, save_path)
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"对比图表已保存至 {path}")
        
        return fig
    
    def generate_distribution_chart(self, data, field='daily_return', title=None, save_path=None):
        """生成分布图表
        
        Args:
            data: DataFrame，包含股票数据
            field: 要分析分布的字段
            title: 图表标题
            save_path: 保存路径，如果提供则保存图表
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        # 准备数据
        df = data.copy()
        
        if field not in df.columns:
            self.logger.error(f"字段 '{field}' 不存在于数据中")
            return None
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制分布图
        sns.histplot(df[field], kde=True, ax=ax)
        
        # 添加均值和中位数线
        mean_val = df[field].mean()
        median_val = df[field].median()
        
        ax.axvline(x=mean_val, color='r', linestyle='--', alpha=0.8, label=f'均值: {mean_val:.4f}')
        ax.axvline(x=median_val, color='g', linestyle='--', alpha=0.8, label=f'中位数: {median_val:.4f}')
        
        # 设置图表标题
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'{field} 分布图', fontsize=14)
        
        # 设置轴标签
        ax.set_xlabel(field)
        ax.set_ylabel('频率')
        
        # 添加网格和图例
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            path = os.path.join(self.output_dir, save_path)
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"分布图表已保存至 {path}")
        
        return fig
    
    def generate_correlation_matrix(self, data_dict, title=None, save_path=None):
        """生成相关性矩阵
        
        Args:
            data_dict: 字典，键为股票名称，值为包含股票数据的DataFrame
            title: 图表标题
            save_path: 保存路径，如果提供则保存图表
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        # 提取每只股票的收盘价
        close_prices = {}
        
        for stock_name, df in data_dict.items():
            # 确保日期是正确的格式
            if 'date' in df.columns:
                df = df.set_index('date')
            
            close_prices[stock_name] = df['close']
        
        # 创建价格DataFrame
        price_df = pd.DataFrame(close_prices)
        
        # 计算日收益率
        returns_df = price_df.pct_change().dropna()
        
        # 计算相关性矩阵
        corr_matrix = returns_df.corr()
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制热力图
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   center=0, square=True, linewidths=0.5, ax=ax)
        
        # 设置图表标题
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('股票收益率相关性矩阵', fontsize=14)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            path = os.path.join(self.output_dir, save_path)
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"相关性矩阵已保存至 {path}")
        
        return fig


def create_stock_dashboard(stock_data, output_path, stock_code=None, timeframe='daily'):
    """创建股票综合分析仪表板
    
    Args:
        stock_data: DataFrame，包含股票数据和技术指标
        output_path: 输出路径
        stock_code: 股票代码
        timeframe: 数据时间帧，如 'daily', 'weekly' 等
        
    Returns:
        str: 仪表板HTML路径
    """
    # 创建图表生成器
    chart_gen = ChartGenerator(output_dir=os.path.dirname(output_path), style='light')
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 生成各种图表
    price_chart_path = f"{stock_code}_price_chart.png"
    indicators_chart_path = f"{stock_code}_indicators_chart.png"
    volume_chart_path = f"{stock_code}_volume_chart.png"
    
    # 生成K线图
    chart_gen.generate_candlestick_chart(
        stock_data, 
        title=f"{stock_code} K线图 ({timeframe})" if stock_code else "股票K线图",
        indicators=['MA5', 'MA10', 'MA20'],
        save_path=price_chart_path
    )
    
    # 生成技术指标图表
    chart_gen.generate_technical_chart(
        stock_data,
        title=f"{stock_code} 技术指标 ({timeframe})" if stock_code else "技术指标图表",
        indicators={
            'panel1': ['close', 'MA5', 'MA10', 'MA20'],
            'panel2': ['volume'],
            'panel3': ['macd', 'signal', 'hist'],
            'panel4': ['rsi']
        },
        save_path=indicators_chart_path
    )
    
    # 构建HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{'股票分析仪表板 - ' + stock_code if stock_code else '股票分析仪表板'}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }}
            .chart-container {{
                margin-bottom: 30px;
            }}
            .chart {{
                width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            h1, h2 {{
                color: #333;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #777;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{'股票分析仪表板 - ' + stock_code if stock_code else '股票分析仪表板'}</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="chart-container">
                <h2>K线图</h2>
                <img class="chart" src="{price_chart_path}" alt="K线图">
            </div>
            
            <div class="chart-container">
                <h2>技术指标</h2>
                <img class="chart" src="{indicators_chart_path}" alt="技术指标图表">
            </div>
            
            <div class="footer">
                <p>© {datetime.now().year} 股票分析系统</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"股票分析仪表板已生成: {output_path}")
    
    return output_path 