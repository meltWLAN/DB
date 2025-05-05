"""
股票列表组件
提供股票列表显示和选择功能
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
                           QPushButton, QLabel, QLineEdit, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem

from services.stock_service import StockService
from utils.logger import get_logger

logger = get_logger(__name__)

class StockListWidget(QWidget):
    """股票列表组件"""
    
    stock_selected = pyqtSignal(str)  # 股票选择信号
    
    def __init__(self):
        super().__init__()
        self.stock_service = StockService()
        self._init_ui()
        self._load_stocks()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 搜索栏
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索股票...")
        self.search_input.textChanged.connect(self._filter_stocks)
        search_layout.addWidget(self.search_input)
        
        # 行业筛选
        self.industry_filter = QComboBox()
        self.industry_filter.addItem("全部行业")
        self.industry_filter.currentTextChanged.connect(self._filter_stocks)
        search_layout.addWidget(self.industry_filter)
        
        layout.addLayout(search_layout)
        
        # 股票列表
        self.stock_list = QListWidget()
        self.stock_list.itemClicked.connect(self._on_stock_selected)
        layout.addWidget(self.stock_list)
        
        # 刷新按钮
        refresh_btn = QPushButton("刷新列表")
        refresh_btn.clicked.connect(self._load_stocks)
        layout.addWidget(refresh_btn)
    
    def _load_stocks(self):
        """加载股票列表"""
        try:
            # 获取股票列表
            stocks = self.stock_service.get_stock_list()
            
            # 更新行业筛选器
            industries = set(stock['industry'] for stock in stocks)
            current_industry = self.industry_filter.currentText()
            self.industry_filter.clear()
            self.industry_filter.addItem("全部行业")
            self.industry_filter.addItems(sorted(industries))
            if current_industry in industries:
                self.industry_filter.setCurrentText(current_industry)
            
            # 更新股票列表
            self.stock_list.clear()
            for stock in stocks:
                item = f"{stock['code']} - {stock['name']} ({stock['industry']})"
                self.stock_list.addItem(item)
            
            logger.info(f"Loaded {len(stocks)} stocks")
        except Exception as e:
            logger.error(f"Error loading stocks: {str(e)}")
    
    def _filter_stocks(self):
        """过滤股票列表"""
        search_text = self.search_input.text().lower()
        industry = self.industry_filter.currentText()
        
        for i in range(self.stock_list.count()):
            item = self.stock_list.item(i)
            item_text = item.text().lower()
            
            # 检查搜索文本和行业
            show = (search_text in item_text and 
                   (industry == "全部行业" or industry in item_text))
            item.setHidden(not show)
    
    def _on_stock_selected(self, item):
        """股票选择处理
        
        Args:
            item: 选中的列表项
        """
        stock_code = item.text().split()[0]
        self.stock_selected.emit(stock_code)
    
    def get_selected_stock(self) -> str:
        """获取当前选中的股票代码
        
        Returns:
            str: 股票代码，如果没有选中则返回空字符串
        """
        current_item = self.stock_list.currentItem()
        if current_item:
            return current_item.text().split()[0]
        return "" 