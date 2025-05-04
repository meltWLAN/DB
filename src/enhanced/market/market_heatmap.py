#!/usr/bin/env python3
"""
市场热力图模块
提供各种市场热力图可视化功能
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

# 配置日志
logger = logging.getLogger(__name__)

class MarketHeatmap:
    """市场热力图类，提供多种热力图可视化功能"""
    
    def __init__(self, save_dir="results/heatmaps"):
        """初始化市场热力图
        
        Args:
            save_dir: 图表保存目录
        """
        self.save_dir = save_dir
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置自定义颜色映射
        # 创建红绿配色的颜色映射（红色表示上涨，绿色表示下跌）
        self.rg_cmap = LinearSegmentedColormap.from_list(
            'red_green', 
            [(0, 'darkgreen'), (0.5, 'white'), (1, 'darkred')]
        )
        
        # 创建蓝色渐变的颜色映射（用于情绪等指标）
        self.blue_cmap = LinearSegmentedColormap.from_list(
            'blue_grad', 
            [(0, 'lightblue'), (1, 'darkblue')]
        )
        
        logger.info("市场热力图初始化完成")
    
    def create_industry_performance_heatmap(self, industry_data, title="行业表现热力图", 
                                           figsize=(12, 8), save_path=None):
        """创建行业表现热力图
        
        Args:
            industry_data: 包含行业名称和涨跌幅的DataFrame
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
            
        Returns:
            str: 图表保存路径
        """
        logger.info("创建行业表现热力图")
        
        # 确保数据中包含必要的列
        required_cols = ['name', 'change']
        if not all(col in industry_data.columns for col in required_cols):
            raise ValueError(f"行业数据必须包含以下列：{required_cols}")
        
        # 按涨跌幅排序
        sorted_data = industry_data.sort_values('change', ascending=False).reset_index(drop=True)
        
        # 创建图表
        plt.figure(figsize=figsize)
        
        # 计算规范化的大小 (面积与涨跌幅成正比，但保证最小值不为0)
        sizes = np.abs(sorted_data['change'])
        min_size, max_size = sizes.min(), sizes.max()
        if min_size == max_size:
            normalized_sizes = np.ones(len(sizes)) * 2000
        else:
            normalized_sizes = ((sizes - min_size) / (max_size - min_size) * 0.8 + 0.2) * 2000
        
        # 创建二维网格坐标，用于放置气泡
        num_items = len(sorted_data)
        cols = min(5, num_items)  # 最多5列
        rows = (num_items - 1) // cols + 1
        
        # 计算位置
        positions = []
        for i in range(num_items):
            row = i // cols
            col = i % cols
            # 交错排列，使视觉效果更好
            if row % 2 == 1:
                col = cols - 1 - col
            positions.append((col, rows - 1 - row))  # 翻转行，使第一行在顶部
        
        # 提取坐标
        x_pos = [pos[0] for pos in positions]
        y_pos = [pos[1] for pos in positions]
        
        # 绘制气泡
        bubbles = plt.scatter(
            x_pos, y_pos,
            s=normalized_sizes,  # 气泡大小
            c=sorted_data['change'],  # 颜色映射数据
            cmap=self.rg_cmap,  # 颜色映射
            alpha=0.7,  # 透明度
            edgecolors='black',  # 气泡边框颜色
            linewidths=1  # 气泡边框宽度
        )
        
        # 添加行业标签
        for i, (x, y) in enumerate(zip(x_pos, y_pos)):
            name = sorted_data['name'].iloc[i]
            change = sorted_data['change'].iloc[i]
            plt.annotate(
                f"{name}\n{change:.2f}%",
                xy=(x, y),
                ha='center',
                va='center',
                fontsize=9,
                weight='bold',
                color='black' if abs(change) < max(abs(sorted_data['change'])) * 0.7 else 'white'
            )
        
        # 设置颜色条
        cbar = plt.colorbar(bubbles)
        cbar.set_label('涨跌幅 (%)')
        
        # 设置标题
        plt.title(title, fontsize=16, pad=20)
        
        # 去除坐标轴
        plt.axis('off')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"行业表现热力图已保存至: {save_path}")
            plt.close()
            return save_path
        else:
            # 生成默认保存路径
            default_path = os.path.join(self.save_dir, f"industry_heatmap_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(default_path, dpi=100, bbox_inches='tight')
            logger.info(f"行业表现热力图已保存至: {default_path}")
            plt.close()
            return default_path
    
    def create_stock_change_heatmap(self, stock_data, groupby='industry', color_by='change',
                                   title="个股涨跌幅热力图", figsize=(14, 10), save_path=None):
        """创建个股涨跌幅热力图
        
        Args:
            stock_data: 包含股票信息的DataFrame
            groupby: 按哪一列分组
            color_by: 按哪一列着色
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
            
        Returns:
            str: 图表保存路径
        """
        logger.info("创建个股涨跌幅热力图")
        
        # 确保数据中包含必要的列
        required_cols = ['name', groupby, color_by]
        if not all(col in stock_data.columns for col in required_cols):
            raise ValueError(f"股票数据必须包含以下列：{required_cols}")
        
        # 按行业分组并计算每个股票在行业内的排名
        groups = stock_data.groupby(groupby)
        
        # 创建图表
        plt.figure(figsize=figsize)
        
        # 计算网格布局
        num_groups = len(groups)
        cols = min(3, num_groups)  # 最多3列
        rows = (num_groups - 1) // cols + 1
        
        # 遍历每个行业绘制小型热力图
        for i, (group_name, group_data) in enumerate(groups):
            # 计算子图位置
            ax = plt.subplot(rows, cols, i + 1)
            
            # 按涨跌幅排序
            sorted_data = group_data.sort_values(color_by, ascending=False).reset_index(drop=True)
            
            # 计算网格大小
            num_stocks = len(sorted_data)
            grid_size = int(np.ceil(np.sqrt(num_stocks)))
            
            # 创建网格索引
            indices = np.arange(num_stocks)
            grid_indices = np.unravel_index(indices[:num_stocks], (grid_size, grid_size))
            
            # 创建热力图数据
            heatmap_data = np.zeros((grid_size, grid_size))
            heatmap_data[:] = np.nan  # 将未使用的单元格设为NaN
            for j, (row, col) in enumerate(zip(*grid_indices)):
                if j < num_stocks:
                    heatmap_data[row, col] = sorted_data[color_by].iloc[j]
            
            # 绘制热力图
            sns.heatmap(
                heatmap_data,
                cmap=self.rg_cmap,
                center=0,
                annot=False,
                cbar=False,
                square=True,
                ax=ax,
                mask=np.isnan(heatmap_data)
            )
            
            # 添加股票标签
            for j, (row, col) in enumerate(zip(*grid_indices)):
                if j < num_stocks:
                    text = sorted_data['name'].iloc[j]
                    val = sorted_data[color_by].iloc[j]
                    # 根据数值选择文本颜色
                    text_color = 'white' if abs(val) > 3 else 'black'
                    ax.text(
                        col + 0.5, row + 0.5, 
                        f"{text}\n{val:.2f}%", 
                        ha='center', va='center',
                        fontsize=8,
                        color=text_color
                    )
            
            # 设置标题
            ax.set_title(f"{group_name} ({num_stocks}只)", fontsize=12)
            
            # 去除刻度标签
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 添加全局颜色条
        plt.subplots_adjust(right=0.9)
        cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=self.rg_cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('涨跌幅 (%)')
        
        # 设置标题
        plt.suptitle(title, fontsize=16, y=0.98)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"个股涨跌幅热力图已保存至: {save_path}")
            plt.close()
            return save_path
        else:
            # 生成默认保存路径
            default_path = os.path.join(self.save_dir, f"stock_heatmap_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(default_path, dpi=100, bbox_inches='tight')
            logger.info(f"个股涨跌幅热力图已保存至: {default_path}")
            plt.close()
            return default_path
    
    def create_market_emotion_heatmap(self, emotion_indicators, title="市场情绪热力图", 
                                     figsize=(10, 6), save_path=None):
        """创建市场情绪热力图
        
        Args:
            emotion_indicators: 情绪指标字典
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
            
        Returns:
            str: 图表保存路径
        """
        logger.info("创建市场情绪热力图")
        
        # 提取数据
        indicators = list(emotion_indicators.keys())
        values = list(emotion_indicators.values())
        
        # 确保有足够的数据
        if not indicators or not values:
            logger.error("情绪指标数据为空")
            return None
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算网格大小
        num_indicators = len(indicators)
        cols = min(4, num_indicators)  # 最多4列
        rows = (num_indicators - 1) // cols + 1
        
        # 创建网格索引
        indices = np.arange(num_indicators)
        grid_indices = np.unravel_index(indices[:num_indicators], (rows, cols))
        
        # 创建热力图数据
        heatmap_data = np.zeros((rows, cols))
        heatmap_data[:] = np.nan  # 将未使用的单元格设为NaN
        
        # 获取数据范围
        min_val, max_val = min(values), max(values)
        norm = plt.Normalize(min_val, max_val)
        
        # 填充热力图数据
        for i, (row, col) in enumerate(zip(*grid_indices)):
            if i < num_indicators:
                heatmap_data[row, col] = values[i]
        
        # 创建自定义颜色映射
        cmap = plt.cm.RdYlGn  # 红黄绿色映射
        
        # 绘制热力图
        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            annot=False,  # 不添加数值标注
            cbar=True,    # 显示颜色条
            square=True,
            ax=ax,
            mask=np.isnan(heatmap_data),  # 隐藏NaN单元格
            cbar_kws={"label": "情绪指数 (低 -> 高)", "orientation": "horizontal", "shrink": 0.8}
        )
        
        # 添加指标标签
        for i, (row, col) in enumerate(zip(*grid_indices)):
            if i < num_indicators:
                val = values[i]
                # 根据值的大小选择文本颜色
                norm_val = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                text_color = 'black' if 0.3 < norm_val < 0.7 else 'white'
                
                ax.text(
                    col + 0.5, row + 0.5, 
                    f"{indicators[i]}\n{val:.2f}", 
                    ha='center', va='center',
                    fontsize=10,
                    color=text_color,
                    weight='bold'
                )
        
        # 设置标题
        ax.set_title(title, fontsize=16, pad=20)
        
        # 移除刻度标签
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"市场情绪热力图已保存至: {save_path}")
            plt.close()
            return save_path
        else:
            # 生成默认保存路径
            default_path = os.path.join(self.save_dir, f"emotion_heatmap_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(default_path, dpi=100, bbox_inches='tight')
            logger.info(f"市场情绪热力图已保存至: {default_path}")
            plt.close()
            return default_path
    
    def create_correlation_heatmap(self, correlation_data, title="相关性热力图", 
                                  figsize=(10, 8), save_path=None):
        """创建相关性热力图
        
        Args:
            correlation_data: 相关性矩阵DataFrame
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
            
        Returns:
            str: 图表保存路径
        """
        logger.info("创建相关性热力图")
        
        # 创建图表
        plt.figure(figsize=figsize)
        
        # 设置自定义颜色映射 - 相关性由-1到1
        corr_cmap = LinearSegmentedColormap.from_list(
            'corr_cmap', 
            [(0, 'blue'), (0.5, 'white'), (1, 'red')]
        )
        
        # 绘制热力图
        mask = np.zeros_like(correlation_data, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True  # 只显示下三角
        
        sns.heatmap(
            correlation_data,
            mask=mask,
            cmap=corr_cmap,
            vmin=-1, vmax=1,
            center=0,
            annot=True,  # 显示数值
            fmt=".2f",  # 数值格式
            linewidths=0.5,  # 单元格边框宽度
            cbar_kws={"shrink": 0.8},  # 颜色条大小
            square=True  # 正方形单元格
        )
        
        # 设置标题
        plt.title(title, fontsize=16, pad=20)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"相关性热力图已保存至: {save_path}")
            plt.close()
            return save_path
        else:
            # 生成默认保存路径
            default_path = os.path.join(self.save_dir, f"correlation_heatmap_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(default_path, dpi=100, bbox_inches='tight')
            logger.info(f"相关性热力图已保存至: {default_path}")
            plt.close()
            return default_path
    
    def create_multi_timeframe_heatmap(self, data, timeframes=['日', '周', '月'], 
                                     title="多时间周期热力图", figsize=(14, 8), save_path=None):
        """创建多时间周期热力图
        
        Args:
            data: 包含不同时间周期数据的字典，格式为{timeframe: DataFrame}
            timeframes: 时间周期名称列表
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
            
        Returns:
            str: 图表保存路径
        """
        logger.info("创建多时间周期热力图")
        
        # 创建图表
        plt.figure(figsize=figsize)
        
        # 时间周期数量
        num_timeframes = len(timeframes)
        
        # 遍历每个时间周期
        for i, timeframe in enumerate(timeframes):
            if timeframe not in data:
                logger.warning(f"时间周期 {timeframe} 数据不存在")
                continue
                
            # 获取当前时间周期的数据
            current_data = data[timeframe]
            
            # 确保数据有名称和涨跌幅列
            if 'name' not in current_data.columns or 'change' not in current_data.columns:
                logger.warning(f"时间周期 {timeframe} 数据缺少必要列")
                continue
            
            # 按涨跌幅排序
            sorted_data = current_data.sort_values('change', ascending=False).head(20)
            
            # 创建子图
            ax = plt.subplot(1, num_timeframes, i + 1)
            
            # 绘制条形图
            bars = ax.barh(
                sorted_data['name'],
                sorted_data['change'],
                color=np.where(sorted_data['change'] > 0, 'red', 'green'),
                alpha=0.7
            )
            
            # 添加数值标签
            for bar in bars:
                width = bar.get_width()
                label_x = width if width >= 0 else width - 1
                ax.text(
                    label_x,
                    bar.get_y() + bar.get_height()/2,
                    f"{width:.2f}%",
                    va='center',
                    fontsize=8,
                    color='black'
                )
            
            # 设置标题
            ax.set_title(f"{timeframe}涨跌幅", fontsize=12)
            
            # 设置网格线
            ax.grid(True, linestyle='--', alpha=0.3, axis='x')
            
            # 设置零线
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # 设置适当的坐标范围
            max_val = max(abs(sorted_data['change'].max()), abs(sorted_data['change'].min()))
            ax.set_xlim(-max_val * 1.2, max_val * 1.2)
        
        # 设置总标题
        plt.suptitle(title, fontsize=16, y=0.98)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"多时间周期热力图已保存至: {save_path}")
            plt.close()
            return save_path
        else:
            # 生成默认保存路径
            default_path = os.path.join(self.save_dir, f"timeframe_heatmap_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(default_path, dpi=100, bbox_inches='tight')
            logger.info(f"多时间周期热力图已保存至: {default_path}")
            plt.close()
            return default_path


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建热力图生成器
    heatmap = MarketHeatmap()
    
    # 创建模拟行业数据
    industry_data = pd.DataFrame({
        'name': ['电子', '医药', '食品饮料', '银行', '房地产', 
                '汽车', '计算机', '通信', '电气设备', '建筑'],
        'change': [3.5, 2.1, 1.8, -0.5, -1.2, 
                  2.8, 3.2, 1.5, -0.8, -1.5]
    })
    
    # 创建模拟股票数据
    stocks = []
    industries = ['电子', '医药', '食品饮料', '银行', '房地产']
    for industry in industries:
        for i in range(5):
            change = np.random.uniform(-5, 5)
            stocks.append({
                'name': f"{industry}股票{i+1}",
                'industry': industry,
                'change': change
            })
    stock_data = pd.DataFrame(stocks)
    
    # 创建模拟情绪指标
    emotion_indicators = {
        '量比': 1.2,
        '涨跌家数比': 1.5,
        '北向资金': 25.6,
        'MACD': 0.02,
        'RSI': 65,
        '恐慌指数': 25,
        '市场宽度': 0.68,
        '换手率': 1.8,
        '情绪指数': 0.75
    }
    
    # 创建模拟相关性数据
    corr_data = pd.DataFrame(np.random.uniform(-1, 1, size=(5, 5)), 
                       columns=['沪指', '深成指', '创业板', '上证50', '北向资金'])
    # 使矩阵对称
    corr_data = (corr_data + corr_data.T) / 2
    np.fill_diagonal(corr_data.values, 1)
    
    # 生成热力图
    heatmap.create_industry_performance_heatmap(industry_data)
    heatmap.create_stock_change_heatmap(stock_data)
    heatmap.create_market_emotion_heatmap(emotion_indicators)
    heatmap.create_correlation_heatmap(corr_data) 