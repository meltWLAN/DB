"""
分析面板组件
提供股票分析结果显示功能
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                           QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from services.analysis_service import AnalysisService
from utils.logger import get_logger

logger = get_logger(__name__)

class AnalysisPanel(QWidget):
    """分析面板组件"""
    
    def __init__(self):
        super().__init__()
        self.analysis_service = AnalysisService()
        self.current_stock = None
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 基本信息
        self.basic_info = QTableWidget()
        self.basic_info.setColumnCount(2)
        self.basic_info.setHorizontalHeaderLabels(["指标", "数值"])
        self.basic_info.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.basic_info.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.basic_info)
        
        # 技术指标
        self.technical_indicators = QTableWidget()
        self.technical_indicators.setColumnCount(2)
        self.technical_indicators.setHorizontalHeaderLabels(["指标", "数值"])
        self.technical_indicators.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.technical_indicators.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.technical_indicators)
        
        # 动量分析
        self.momentum_analysis = QTableWidget()
        self.momentum_analysis.setColumnCount(2)
        self.momentum_analysis.setHorizontalHeaderLabels(["指标", "数值"])
        self.momentum_analysis.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.momentum_analysis.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.momentum_analysis)
    
    def set_stock(self, stock_code: str):
        """设置当前股票
        
        Args:
            stock_code: 股票代码
        """
        self.current_stock = stock_code
        self.update_data()
    
    def update_data(self):
        """更新分析数据"""
        if not self.current_stock:
            return
        
        try:
            # 获取分析数据
            analysis_data = self.analysis_service.get_analysis(self.current_stock)
            
            # 更新基本信息
            self._update_table(self.basic_info, analysis_data['basic_info'])
            
            # 更新技术指标
            self._update_table(self.technical_indicators, analysis_data['technical_indicators'])
            
            # 更新动量分析
            self._update_table(self.momentum_analysis, analysis_data['momentum_analysis'])
            
            logger.info(f"Updated analysis data for {self.current_stock}")
        except Exception as e:
            logger.error(f"Error updating analysis data: {str(e)}")
    
    def _update_table(self, table: QTableWidget, data: dict):
        """更新表格数据
        
        Args:
            table: 表格组件
            data: 数据字典
        """
        table.setRowCount(len(data))
        for i, (key, value) in enumerate(data.items()):
            # 指标名称
            name_item = QTableWidgetItem(key)
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(i, 0, name_item)
            
            # 指标值
            value_item = QTableWidgetItem(str(value))
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            # 根据数值设置颜色
            if isinstance(value, (int, float)):
                if value > 0:
                    value_item.setForeground(QColor('green'))
                elif value < 0:
                    value_item.setForeground(QColor('red'))
            
            table.setItem(i, 1, value_item)
    
    def update_settings(self, settings: dict):
        """更新设置
        
        Args:
            settings: 新的设置
        """
        try:
            self.analysis_service.update_settings(settings)
            if self.current_stock:
                self.update_data()
            logger.info("Updated analysis settings")
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}") 