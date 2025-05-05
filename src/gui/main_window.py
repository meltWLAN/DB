"""
主窗口模块
提供桌面应用的主界面
"""

import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QTabWidget, QStatusBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon

from config.settings import GUI_CONFIG
from gui.widgets.stock_list import StockListWidget
from gui.widgets.analysis_panel import AnalysisPanel
from gui.widgets.chart_panel import ChartPanel
from gui.widgets.control_panel import ControlPanel
from utils.logger import get_logger

logger = get_logger(__name__)

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("股票分析系统")
        self.setMinimumSize(1200, 800)
        
        # 设置主题
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {GUI_CONFIG['themes']['default']['background']};
            }}
            QLabel {{
                color: {GUI_CONFIG['themes']['default']['text']};
            }}
            QPushButton {{
                background-color: {GUI_CONFIG['themes']['default']['button']};
                color: {GUI_CONFIG['themes']['default']['button_text']};
                border: 1px solid {GUI_CONFIG['themes']['default']['accent']};
                padding: 5px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {GUI_CONFIG['themes']['default']['highlight']};
            }}
        """)
        
        # 初始化UI
        self._init_ui()
        
        # 初始化定时器
        self._init_timer()
        
        # 连接信号
        self._connect_signals()
    
    def _init_ui(self):
        """初始化UI组件"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 股票列表
        self.stock_list = StockListWidget()
        left_layout.addWidget(self.stock_list)
        
        # 控制面板
        self.control_panel = ControlPanel()
        left_layout.addWidget(self.control_panel)
        
        # 右侧面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 标签页
        self.tab_widget = QTabWidget()
        
        # 分析面板
        self.analysis_panel = AnalysisPanel()
        self.tab_widget.addTab(self.analysis_panel, "分析")
        
        # 图表面板
        self.chart_panel = ChartPanel()
        self.tab_widget.addTab(self.chart_panel, "图表")
        
        right_layout.addWidget(self.tab_widget)
        
        # 添加到主布局
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 3)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
    
    def _init_timer(self):
        """初始化定时器"""
        self.update_timer = QTimer()
        self.update_timer.setInterval(GUI_CONFIG['refresh_interval'])
        self.update_timer.timeout.connect(self._update_data)
    
    def _connect_signals(self):
        """连接信号和槽"""
        # 股票选择
        self.stock_list.stock_selected.connect(self._on_stock_selected)
        
        # 控制面板信号
        self.control_panel.start_analysis.connect(self._on_start_analysis)
        self.control_panel.stop_analysis.connect(self._on_stop_analysis)
        self.control_panel.settings_changed.connect(self._on_settings_changed)
    
    def _on_stock_selected(self, stock_code: str):
        """股票选择处理
        
        Args:
            stock_code: 股票代码
        """
        logger.info(f"Selected stock: {stock_code}")
        self.analysis_panel.set_stock(stock_code)
        self.chart_panel.set_stock(stock_code)
    
    def _on_start_analysis(self):
        """开始分析处理"""
        logger.info("Starting analysis")
        self.status_bar.showMessage("分析中...")
        self.update_timer.start()
    
    def _on_stop_analysis(self):
        """停止分析处理"""
        logger.info("Stopping analysis")
        self.status_bar.showMessage("已停止")
        self.update_timer.stop()
    
    def _on_settings_changed(self, settings: dict):
        """设置变更处理
        
        Args:
            settings: 新的设置
        """
        logger.info(f"Settings changed: {settings}")
        # 更新相关组件
        self.analysis_panel.update_settings(settings)
        self.chart_panel.update_settings(settings)
    
    def _update_data(self):
        """更新数据"""
        try:
            # 更新当前选中的股票数据
            current_stock = self.stock_list.get_selected_stock()
            if current_stock:
                self.analysis_panel.update_data()
                self.chart_panel.update_data()
        except Exception as e:
            logger.error(f"Error updating data: {str(e)}")
            self.status_bar.showMessage(f"更新失败: {str(e)}")
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        logger.info("Closing application")
        self.update_timer.stop()
        event.accept() 