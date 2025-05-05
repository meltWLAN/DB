"""
图表面板组件
提供股票图表显示功能
"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
                           QPushButton, QLabel)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph import PlotWidget

from services.chart_service import ChartService
from utils.logger import get_logger

logger = get_logger(__name__)

class ChartPanel(QWidget):
    """图表面板组件"""
    
    def __init__(self):
        super().__init__()
        self.chart_service = ChartService()
        self.current_stock = None
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 图表控制栏
        control_layout = QHBoxLayout()
        
        # 时间周期选择
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1分钟", "5分钟", "15分钟", "30分钟", 
                                     "60分钟", "日线", "周线", "月线"])
        self.timeframe_combo.currentTextChanged.connect(self._on_timeframe_changed)
        control_layout.addWidget(QLabel("时间周期:"))
        control_layout.addWidget(self.timeframe_combo)
        
        # 指标选择
        self.indicator_combo = QComboBox()
        self.indicator_combo.addItems(["K线", "MA", "MACD", "RSI", "BOLL"])
        self.indicator_combo.currentTextChanged.connect(self._on_indicator_changed)
        control_layout.addWidget(QLabel("指标:"))
        control_layout.addWidget(self.indicator_combo)
        
        # 刷新按钮
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.update_data)
        control_layout.addWidget(refresh_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # 图表区域
        self.chart = PlotWidget()
        self.chart.setBackground('w')
        self.chart.showGrid(True, True)
        self.chart.setLabel('left', '价格')
        self.chart.setLabel('bottom', '时间')
        layout.addWidget(self.chart)
        
        # 初始化图表样式
        self._init_chart_style()
    
    def _init_chart_style(self):
        """初始化图表样式"""
        # 设置颜色
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        # 设置字体
        font = {'family': 'Arial', 'size': 10}
        pg.setConfigOption('font', font)
    
    def set_stock(self, stock_code: str):
        """设置当前股票
        
        Args:
            stock_code: 股票代码
        """
        self.current_stock = stock_code
        self.update_data()
    
    def update_data(self):
        """更新图表数据"""
        if not self.current_stock:
            return
        
        try:
            # 获取图表数据
            timeframe = self.timeframe_combo.currentText()
            indicator = self.indicator_combo.currentText()
            chart_data = self.chart_service.get_chart_data(
                self.current_stock, timeframe, indicator
            )
            
            # 清除旧图表
            self.chart.clear()
            
            # 绘制新图表
            if indicator == "K线":
                self._plot_candlestick(chart_data)
            else:
                self._plot_indicator(chart_data, indicator)
            
            logger.info(f"Updated chart for {self.current_stock}")
        except Exception as e:
            logger.error(f"Error updating chart: {str(e)}")
    
    def _plot_candlestick(self, data: dict):
        """绘制K线图
        
        Args:
            data: 图表数据
        """
        # 绘制K线
        for i in range(len(data['open'])):
            # 计算K线位置
            x = i
            open_price = data['open'][i]
            close_price = data['close'][i]
            high_price = data['high'][i]
            low_price = data['low'][i]
            
            # 设置颜色
            color = 'g' if close_price >= open_price else 'r'
            
            # 绘制K线实体
            self.chart.addLine(x=x, y=open_price, x2=x, y2=close_price,
                             pen=pg.mkPen(color, width=3))
            
            # 绘制上下影线
            self.chart.addLine(x=x, y=low_price, x2=x, y2=high_price,
                             pen=pg.mkPen(color, width=1))
    
    def _plot_indicator(self, data: dict, indicator: str):
        """绘制技术指标
        
        Args:
            data: 图表数据
            indicator: 指标名称
        """
        x = np.arange(len(data['values']))
        
        if indicator == "MA":
            # 绘制多条均线
            for i, ma_data in enumerate(data['values']):
                self.chart.plot(x, ma_data, pen=pg.mkPen(
                    color=self._get_indicator_color(indicator, i),
                    width=2
                ))
        else:
            # 绘制单个指标
            self.chart.plot(x, data['values'], pen=pg.mkPen(
                color=self._get_indicator_color(indicator),
                width=2
            ))
    
    def _get_indicator_color(self, indicator: str, index: int = 0) -> str:
        """获取指标颜色
        
        Args:
            indicator: 指标名称
            index: 指标索引（用于多条线的情况）
            
        Returns:
            str: 颜色代码
        """
        colors = {
            "MA": ['#FF0000', '#00FF00', '#0000FF', '#FF00FF'],
            "MACD": ['#0000FF', '#FF0000', '#00FF00'],
            "RSI": '#FF00FF',
            "BOLL": ['#0000FF', '#FF0000', '#00FF00']
        }
        
        if isinstance(colors[indicator], list):
            return colors[indicator][index % len(colors[indicator])]
        return colors[indicator]
    
    def _on_timeframe_changed(self, timeframe: str):
        """时间周期变更处理
        
        Args:
            timeframe: 新的时间周期
        """
        if self.current_stock:
            self.update_data()
    
    def _on_indicator_changed(self, indicator: str):
        """指标变更处理
        
        Args:
            indicator: 新的指标
        """
        if self.current_stock:
            self.update_data()
    
    def update_settings(self, settings: dict):
        """更新设置
        
        Args:
            settings: 新的设置
        """
        try:
            self.chart_service.update_settings(settings)
            if self.current_stock:
                self.update_data()
            logger.info("Updated chart settings")
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}") 