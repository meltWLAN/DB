"""
股票图表可视化模块
负责绘制各种类型的股票图表
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, List, Tuple, Optional, Union, Any
import mplfinance as mpf
from datetime import datetime, timedelta
from ..config.settings import GUI_CONFIG

class StockChartManager:
    """股票图表管理器，负责创建和管理各种类型的股票图表"""
    
    def __init__(self, theme_config: Optional[Dict[str, Any]] = None):
        """
        初始化图表管理器
        
        Args:
            theme_config: 主题配置
        """
        # 设置主题配置
        if theme_config is None:
            self.theme = {
                "up_color": GUI_CONFIG["chart_colors"]["up"],
                "down_color": GUI_CONFIG["chart_colors"]["down"],
                "volume_color": GUI_CONFIG["chart_colors"]["volume"],
                "ma_colors": [
                    GUI_CONFIG["chart_colors"]["ma5"],
                    GUI_CONFIG["chart_colors"]["ma10"],
                    GUI_CONFIG["chart_colors"]["ma20"],
                    GUI_CONFIG["chart_colors"]["ma30"]
                ],
                "grid_color": "#E0E0E0",
                "bg_color": "#FFFFFF",
                "text_color": "#333333",
            }
        else:
            self.theme = theme_config
            
        # 设置全局样式
        self._set_style()
        
    def _set_style(self) -> None:
        """设置图表样式"""
        # 设置seaborn样式
        sns.set_style("ticks", {
            "axes.grid": True,
            "grid.color": self.theme["grid_color"],
            "axes.facecolor": self.theme["bg_color"],
            "text.color": self.theme["text_color"],
            "axes.labelcolor": self.theme["text_color"],
            "xtick.color": self.theme["text_color"],
            "ytick.color": self.theme["text_color"],
        })
        
        # 设置matplotlib参数
        plt.rcParams.update({
            "figure.facecolor": self.theme["bg_color"],
            "axes.facecolor": self.theme["bg_color"],
            "savefig.facecolor": self.theme["bg_color"],
        })
        
    def update_theme(self, theme_config: Dict[str, Any]) -> None:
        """
        更新图表主题
        
        Args:
            theme_config: 新的主题配置
        """
        self.theme = theme_config
        self._set_style()
        
    def create_candlestick_chart(self, df: pd.DataFrame, parent=None, 
                                title: str = "股票价格图", figsize: Tuple[int, int] = (12, 8)) -> Union[None, Tuple]:
        """
        创建K线图
        
        Args:
            df: 股票数据，必须包含open, high, low, close和volume列
            parent: tkinter父容器，如果提供则返回tkinter画布
            title: 图表标题
            figsize: 图表大小
            
        Returns:
            如果parent为None，返回None；否则返回(fig, canvas)
        """
        # 确保数据包含必要的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"数据必须包含以下列: {', '.join(required_cols)}")
            
        # 确保索引是日期类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            
        # 设置mplfinance样式
        mc = mpf.make_marketcolors(
            up=self.theme["up_color"],
            down=self.theme["down_color"],
            volume=self.theme["volume_color"],
            edge='inherit',
            wick='inherit'
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='--',
            gridcolor=self.theme["grid_color"],
            figcolor=self.theme["bg_color"],
            facecolor=self.theme["bg_color"],
            edgecolor=self.theme["text_color"],
            y_on_right=False
        )
        
        # 创建附加图表
        ap = []
        
        # 添加移动平均线
        if 'ma5' in df.columns and 'ma10' in df.columns and 'ma20' in df.columns:
            ap.append(mpf.make_addplot(df['ma5'], color=self.theme["ma_colors"][0], width=1, panel=0))
            ap.append(mpf.make_addplot(df['ma10'], color=self.theme["ma_colors"][1], width=1, panel=0))
            ap.append(mpf.make_addplot(df['ma20'], color=self.theme["ma_colors"][2], width=1, panel=0))
            
        # 如果是在Tkinter应用中使用
        if parent is not None:
            # 创建图表并返回Figure和Canvas对象
            fig = mpf.figure(figsize=figsize, style=s)
            ax1 = fig.add_subplot(5, 1, (1, 4))  # 价格图占4/5
            ax2 = fig.add_subplot(5, 1, 5, sharex=ax1)  # 成交量图占1/5
            
            mpf.plot(
                df,
                type='candle',
                style=s,
                ax=ax1,
                volume=ax2,
                axtitle=title,
                xrotation=20,
                datetime_format='%Y-%m-%d',
                addplot=ap,
                figscale=1.2
            )
            
            # 创建Canvas
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill='both', expand=True)
            
            return fig, canvas
        else:
            # 直接绘制并显示图表
            mpf.plot(
                df,
                type='candle',
                style=s,
                title=title,
                volume=True,
                figsize=figsize,
                xrotation=20,
                datetime_format='%Y-%m-%d',
                addplot=ap,
                figscale=1.2,
                panel_ratios=(4, 1)
            )
            return None
            
    def create_line_chart(self, df: pd.DataFrame, y_column: str, parent=None,
                         title: str = "股票走势图", figsize: Tuple[int, int] = (12, 6)) -> Union[None, Tuple]:
        """
        创建折线图
        
        Args:
            df: 股票数据
            y_column: 要绘制的数据列名
            parent: tkinter父容器
            title: 图表标题
            figsize: 图表大小
            
        Returns:
            如果parent为None，返回None；否则返回(fig, canvas)
        """
        # 确保数据包含必要的列
        if y_column not in df.columns:
            raise ValueError(f"数据必须包含列: {y_column}")
            
        # 创建图表
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        
        # 绘制折线图
        ax.plot(df.index, df[y_column], color=self.theme["ma_colors"][0], linewidth=1.5)
        
        # 设置图表标题和标签
        ax.set_title(title)
        ax.set_ylabel(y_column)
        
        # 设置网格和背景
        ax.grid(True, linestyle='--', alpha=0.7, color=self.theme["grid_color"])
        ax.set_facecolor(self.theme["bg_color"])
        
        # 格式化x轴日期
        if isinstance(df.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
            
        if parent is not None:
            # 创建Canvas
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill='both', expand=True)
            
            return fig, canvas
        else:
            plt.show()
            return None
            
    def create_multi_line_chart(self, df: pd.DataFrame, y_columns: List[str], parent=None,
                               title: str = "技术指标图", figsize: Tuple[int, int] = (12, 6)) -> Union[None, Tuple]:
        """
        创建多线折线图
        
        Args:
            df: 股票数据
            y_columns: 要绘制的数据列名列表
            parent: tkinter父容器
            title: 图表标题
            figsize: 图表大小
            
        Returns:
            如果parent为None，返回None；否则返回(fig, canvas)
        """
        # 确保数据包含必要的列
        for col in y_columns:
            if col not in df.columns:
                raise ValueError(f"数据必须包含列: {col}")
                
        # 创建图表
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        
        # 绘制多条折线
        for i, col in enumerate(y_columns):
            color = self.theme["ma_colors"][i % len(self.theme["ma_colors"])]
            ax.plot(df.index, df[col], color=color, linewidth=1.5, label=col)
            
        # 设置图表标题和标签
        ax.set_title(title)
        ax.legend()
        
        # 设置网格和背景
        ax.grid(True, linestyle='--', alpha=0.7, color=self.theme["grid_color"])
        ax.set_facecolor(self.theme["bg_color"])
        
        # 格式化x轴日期
        if isinstance(df.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
            
        if parent is not None:
            # 创建Canvas
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill='both', expand=True)
            
            return fig, canvas
        else:
            plt.show()
            return None
            
    def create_technical_indicators_chart(self, df: pd.DataFrame, parent=None,
                                         title: str = "技术指标分析", figsize: Tuple[int, int] = (12, 10)) -> Union[None, Tuple]:
        """
        创建技术指标组合图
        
        Args:
            df: 股票数据
            parent: tkinter父容器
            title: 图表标题
            figsize: 图表大小
            
        Returns:
            如果parent为None，返回None；否则返回(fig, canvas)
        """
        # 创建图表
        fig = Figure(figsize=figsize, dpi=100)
        
        # 确定要绘制的指标
        indicators = []
        if 'rsi' in df.columns:
            indicators.append(('rsi', '相对强弱指标(RSI)'))
        if 'macd' in df.columns and 'signal' in df.columns:
            indicators.append(('macd', 'MACD'))
        if 'k' in df.columns and 'd' in df.columns:
            indicators.append(('kdj', 'KDJ'))
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            indicators.append(('bollinger', '布林带'))
            
        # 没有可用指标
        if not indicators:
            fig.suptitle("没有可用的技术指标")
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "没有找到技术指标数据", ha='center', va='center')
            
            if parent is not None:
                canvas = FigureCanvasTkAgg(fig, master=parent)
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.pack(fill='both', expand=True)
                return fig, canvas
            else:
                plt.show()
                return None
                
        # 设置图表布局
        n_indicators = len(indicators)
        grid_height = n_indicators + 1  # 加1是为了价格图
        price_height = 2  # 价格图高度是其他图的2倍
        total_height = grid_height + price_height - 1
        
        # 价格图
        ax_price = fig.add_subplot(total_height, 1, (1, price_height))
        ax_price.set_title("价格走势")
        ax_price.plot(df.index, df['close'], color='black', linewidth=1.5, label='收盘价')
        
        # 添加移动平均线
        if 'ma5' in df.columns:
            ax_price.plot(df.index, df['ma5'], color=self.theme["ma_colors"][0], linewidth=1, label='MA5')
        if 'ma10' in df.columns:
            ax_price.plot(df.index, df['ma10'], color=self.theme["ma_colors"][1], linewidth=1, label='MA10')
        if 'ma20' in df.columns:
            ax_price.plot(df.index, df['ma20'], color=self.theme["ma_colors"][2], linewidth=1, label='MA20')
            
        ax_price.legend(loc='upper left')
        ax_price.grid(True, linestyle='--', alpha=0.7, color=self.theme["grid_color"])
        
        # 添加技术指标
        for i, (indicator_name, indicator_title) in enumerate(indicators):
            ax = fig.add_subplot(total_height, 1, price_height + i + 1, sharex=ax_price)
            ax.set_title(indicator_title)
            
            if indicator_name == 'rsi':
                ax.plot(df.index, df['rsi'], color=self.theme["ma_colors"][0], linewidth=1.5)
                ax.axhline(70, color='r', linestyle='--', alpha=0.5)
                ax.axhline(30, color='g', linestyle='--', alpha=0.5)
                ax.set_ylim(0, 100)
                
            elif indicator_name == 'macd':
                ax.plot(df.index, df['macd'], color=self.theme["ma_colors"][0], linewidth=1.5, label='MACD')
                ax.plot(df.index, df['signal'], color=self.theme["ma_colors"][1], linewidth=1.5, label='Signal')
                
                # 添加MACD柱状图
                if 'macd_hist' in df.columns:
                    for j in range(len(df)):
                        if j > 0:  # 跳过第一个点
                            hist = df['macd_hist'].iloc[j]
                            color = self.theme["up_color"] if hist > 0 else self.theme["down_color"]
                            ax.fill_between([df.index[j-1], df.index[j]], 
                                           [0, 0], 
                                           [df['macd_hist'].iloc[j-1], hist], 
                                           color=color, alpha=0.5)
                ax.legend(loc='upper left')
                
            elif indicator_name == 'kdj':
                ax.plot(df.index, df['k'], color=self.theme["ma_colors"][0], linewidth=1.5, label='K')
                ax.plot(df.index, df['d'], color=self.theme["ma_colors"][1], linewidth=1.5, label='D')
                if 'j' in df.columns:
                    ax.plot(df.index, df['j'], color=self.theme["ma_colors"][2], linewidth=1.5, label='J')
                ax.axhline(80, color='r', linestyle='--', alpha=0.5)
                ax.axhline(20, color='g', linestyle='--', alpha=0.5)
                ax.set_ylim(0, 100)
                ax.legend(loc='upper left')
                
            elif indicator_name == 'bollinger':
                ax.plot(df.index, df['close'], color='black', linewidth=1.5, label='收盘价')
                ax.plot(df.index, df['bb_upper'], color=self.theme["ma_colors"][0], linestyle='--', linewidth=1.5, label='上轨')
                ax.plot(df.index, df['bb_middle'], color=self.theme["ma_colors"][1], linewidth=1.5, label='中轨')
                ax.plot(df.index, df['bb_lower'], color=self.theme["ma_colors"][2], linestyle='--', linewidth=1.5, label='下轨')
                ax.legend(loc='upper left')
                
            ax.grid(True, linestyle='--', alpha=0.7, color=self.theme["grid_color"])
            
        # 格式化x轴日期
        if isinstance(df.index, pd.DatetimeIndex):
            ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
            
        # 调整布局
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)
        
        if parent is not None:
            # 创建Canvas
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill='both', expand=True)
            
            return fig, canvas
        else:
            plt.show()
            return None
            
    def create_volume_profile_chart(self, df: pd.DataFrame, parent=None,
                                   title: str = "成交量分布图", figsize: Tuple[int, int] = (12, 8)) -> Union[None, Tuple]:
        """
        创建成交量分布图
        
        Args:
            df: 股票数据，必须包含close和volume列
            parent: tkinter父容器
            title: 图表标题
            figsize: 图表大小
            
        Returns:
            如果parent为None，返回None；否则返回(fig, canvas)
        """
        # 确保数据包含必要的列
        required_cols = ['close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"数据必须包含以下列: {', '.join(required_cols)}")
            
        # 创建图表
        fig = Figure(figsize=figsize, dpi=100)
        
        # 设置网格
        grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
        
        # 主图 - 价格
        ax_main = fig.add_subplot(grid[0:3, 0:3])
        # 成交量直方图 - 右侧
        ax_vol_profile = fig.add_subplot(grid[0:3, 3], sharey=ax_main)
        # 成交量图 - 底部
        ax_volume = fig.add_subplot(grid[3, 0:3], sharex=ax_main)
        
        # 价格图
        ax_main.plot(df.index, df['close'], color='black', linewidth=1.5)
        # 添加移动平均线
        if 'ma5' in df.columns:
            ax_main.plot(df.index, df['ma5'], color=self.theme["ma_colors"][0], linewidth=1, label='MA5')
        if 'ma10' in df.columns:
            ax_main.plot(df.index, df['ma10'], color=self.theme["ma_colors"][1], linewidth=1, label='MA10')
        if 'ma20' in df.columns:
            ax_main.plot(df.index, df['ma20'], color=self.theme["ma_colors"][2], linewidth=1, label='MA20')
            
        ax_main.set_title(title)
        ax_main.legend(loc='upper left')
        ax_main.grid(True, linestyle='--', alpha=0.7, color=self.theme["grid_color"])
        
        # 计算价格水平和对应的成交量
        price_min = df['close'].min()
        price_max = df['close'].max()
        price_range = price_max - price_min
        n_bins = 50
        bin_size = price_range / n_bins
        
        # 创建价格区间
        price_bins = np.linspace(price_min, price_max, n_bins)
        volume_by_price = [0] * len(price_bins)
        
        # 统计每个价格区间的成交量
        for i, (_, row) in enumerate(df.iterrows()):
            bin_index = min(int((row['close'] - price_min) / bin_size), n_bins - 1)
            volume_by_price[bin_index] += row['volume']
            
        # 绘制成交量分布
        ax_vol_profile.barh(price_bins, volume_by_price, height=bin_size, color=self.theme["volume_color"], alpha=0.7)
        ax_vol_profile.set_xticks([])  # 隐藏x轴刻度
        ax_vol_profile.set_title("成交量分布")
        
        # 成交量图
        for i in range(len(df)):
            color = self.theme["up_color"] if i > 0 and df['close'].iloc[i] >= df['close'].iloc[i-1] else self.theme["down_color"]
            ax_volume.bar(df.index[i], df['volume'].iloc[i], color=color, alpha=0.7)
            
        ax_volume.set_title("成交量")
        
        # 格式化x轴日期
        if isinstance(df.index, pd.DatetimeIndex):
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_main.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
            
        # 调整布局
        plt.tight_layout()
        
        if parent is not None:
            # 创建Canvas
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill='both', expand=True)
            
            return fig, canvas
        else:
            plt.show()
            return None
            
    def create_performance_chart(self, performance_data: Dict[str, Union[pd.Series, List[float], np.ndarray]],
                                parent=None, title: str = "策略表现", figsize: Tuple[int, int] = (12, 8)) -> Union[None, Tuple]:
        """
        创建策略表现图表
        
        Args:
            performance_data: 性能数据字典，必须包含：
                - 'returns': 策略收益率序列
                - 'benchmark_returns': 基准收益率序列(可选)
                - 'drawdowns': 回撤序列(可选)
            parent: tkinter父容器
            title: 图表标题
            figsize: 图表大小
            
        Returns:
            如果parent为None，返回None；否则返回(fig, canvas)
        """
        # 检查必要数据
        if 'returns' not in performance_data:
            raise ValueError("性能数据必须包含'returns'键")
            
        returns = performance_data['returns']
        has_benchmark = 'benchmark_returns' in performance_data
        has_drawdowns = 'drawdowns' in performance_data
        
        # 创建图表
        fig = Figure(figsize=figsize, dpi=100)
        
        # 确定子图数量和布局
        n_plots = 1 + has_drawdowns
        
        # 累积收益图
        ax_returns = fig.add_subplot(n_plots, 1, 1)
        
        # 将收益率转换为累积收益率
        cumulative_returns = (1 + returns).cumprod() - 1
        ax_returns.plot(cumulative_returns.index, cumulative_returns, 
                      linewidth=2, color=self.theme["ma_colors"][0], label='策略')
        
        # 绘制基准收益率（如果有）
        if has_benchmark:
            benchmark_returns = performance_data['benchmark_returns']
            cumulative_benchmark = (1 + benchmark_returns).cumprod() - 1
            ax_returns.plot(cumulative_benchmark.index, cumulative_benchmark, 
                          linewidth=2, color=self.theme["ma_colors"][1], label='基准')
            
        ax_returns.set_title("累积收益率")
        ax_returns.legend(loc='upper left')
        ax_returns.grid(True, linestyle='--', alpha=0.7, color=self.theme["grid_color"])
        ax_returns.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        ax_returns.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # 绘制回撤（如果有）
        if has_drawdowns:
            drawdowns = performance_data['drawdowns']
            ax_drawdowns = fig.add_subplot(n_plots, 1, 2, sharex=ax_returns)
            ax_drawdowns.fill_between(drawdowns.index, 0, -drawdowns, 
                                    color=self.theme["down_color"], alpha=0.5)
            ax_drawdowns.set_title("回撤")
            ax_drawdowns.grid(True, linestyle='--', alpha=0.7, color=self.theme["grid_color"])
            ax_drawdowns.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{-y:.1%}'))
            
        # 格式化x轴日期
        if isinstance(returns.index, pd.DatetimeIndex):
            ax_returns.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_returns.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
            
        # 调整布局
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)
        
        if parent is not None:
            # 创建Canvas
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill='both', expand=True)
            
            return fig, canvas
        else:
            plt.show()
            return None 