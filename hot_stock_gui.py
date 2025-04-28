#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
暴涨股捕捉GUI界面
提供连续涨停股票扫描、潜在暴涨股识别和热门板块分析功能
"""

import os
import sys
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QComboBox, QSpinBox, QCheckBox, QSplitter,
                             QGroupBox, QFormLayout, QLineEdit, QMessageBox,
                             QProgressBar, QTextEdit, QRadioButton, QButtonGroup,
                             QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate, QTimer
from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap
import pyqtgraph as pg
from hot_stock_scanner import HotStockScanner

# 导入自定义模块
from src.strategies.hot_stock_controller import HotStockController
from src.enhanced.data.fetchers.tushare_fetcher import EnhancedTushareFetcher
from src.enhanced.data.processors.optimized_processor import EnhancedDataProcessor
from stock_data_storage import StockData
from enhanced_momentum_analysis import EnhancedMomentumAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hot_stock_gui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ScannerThread(QThread):
    """Thread for running scanner operations without blocking the GUI"""
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, scanner, scan_type, params=None):
        super().__init__()
        self.scanner = scanner
        self.scan_type = scan_type
        self.params = params or {}
        
    def run(self):
        try:
            result = {}
            if self.scan_type == "consecutive_limit_up":
                result["data"] = self.scanner.scan_consecutive_limit_up(
                    days_look_back=self.params.get("days_look_back", 10),
                    min_consecutive_days=self.params.get("min_consecutive_days", 2),
                    refresh_cache=self.params.get("refresh_cache", False)
                )
            elif self.scan_type == "potential_breakout":
                result["data"] = self.scanner.scan_potential_breakout(
                    days_look_back=self.params.get("days_look_back", 20),
                    min_score=self.params.get("min_score", 70)
                )
            elif self.scan_type == "hot_sectors":
                result["data"] = self.scanner.get_hot_sectors(
                    top_n=self.params.get("top_n", 5)
                )
            elif self.scan_type == "predict_continuation":
                result["data"] = self.scanner.predict_limit_up_continuation(
                    stock_code=self.params.get("stock_code", ""),
                    consecutive_days=self.params.get("consecutive_days", 2)
                )
                
            self.result_signal.emit(result)
        except Exception as e:
            logger.error(f"Error in scanner thread: {e}", exc_info=True)
            self.error_signal.emit(str(e))

class HotStockGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scanner = HotStockScanner()
        self.stock_data = StockData()
        self.init_ui()
        self.current_thread = None
        
    def init_ui(self):
        self.setWindowTitle("Hot Stock Scanner")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.setup_consecutive_limit_up_tab()
        self.setup_potential_breakout_tab()
        self.setup_hot_sectors_tab()
        self.setup_prediction_tab()
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def setup_consecutive_limit_up_tab(self):
        """Setup tab for consecutive limit-up stocks scanning"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Control panel
        control_group = QGroupBox("Scan Controls")
        control_layout = QFormLayout()
        
        # Days to look back
        self.days_look_back = QSpinBox()
        self.days_look_back.setRange(5, 60)
        self.days_look_back.setValue(10)
        control_layout.addRow("Days to Look Back:", self.days_look_back)
        
        # Minimum consecutive days
        self.min_consecutive_days = QSpinBox()
        self.min_consecutive_days.setRange(1, 10)
        self.min_consecutive_days.setValue(2)
        control_layout.addRow("Min Consecutive Days:", self.min_consecutive_days)
        
        # Refresh cache checkbox
        self.refresh_cache = QCheckBox("Refresh Cache")
        control_layout.addRow("", self.refresh_cache)
        
        # Scan button
        self.scan_button = QPushButton("Scan Consecutive Limit-Up Stocks")
        self.scan_button.clicked.connect(self.scan_consecutive_limit_up)
        control_layout.addRow("", self.scan_button)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Results table
        self.limit_up_table = QTableWidget()
        self.limit_up_table.setColumnCount(7)
        self.limit_up_table.setHorizontalHeaderLabels([
            "Stock Code", "Stock Name", "Consecutive Days", 
            "Last Close", "Change (%)", "Volume Ratio", "Details"
        ])
        self.limit_up_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.limit_up_table)
        
        self.tabs.addTab(tab, "连续涨停扫描")

    def setup_potential_breakout_tab(self):
        """Setup tab for potential breakout stocks scanning"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Control panel
        control_group = QGroupBox("Breakout Scan Controls")
        control_layout = QFormLayout()
        
        # Days to look back
        self.breakout_days = QSpinBox()
        self.breakout_days.setRange(10, 120)
        self.breakout_days.setValue(20)
        control_layout.addRow("Days to Look Back:", self.breakout_days)
        
        # Minimum score
        self.min_score = QSpinBox()
        self.min_score.setRange(50, 100)
        self.min_score.setValue(70)
        control_layout.addRow("Minimum Score:", self.min_score)
        
        # Scan button
        self.breakout_button = QPushButton("Scan Potential Breakout Stocks")
        self.breakout_button.clicked.connect(self.scan_potential_breakout)
        control_layout.addRow("", self.breakout_button)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Results table
        self.breakout_table = QTableWidget()
        self.breakout_table.setColumnCount(8)
        self.breakout_table.setHorizontalHeaderLabels([
            "Stock Code", "Stock Name", "Breakout Score", 
            "Momentum Score", "Volume Score", "RSI", 
            "MACD Signal", "Details"
        ])
        self.breakout_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.breakout_table)
        
        self.tabs.addTab(tab, "潜在暴涨股")

    def setup_hot_sectors_tab(self):
        """Setup tab for hot sectors analysis"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Control panel
        control_group = QGroupBox("Hot Sectors Controls")
        control_layout = QFormLayout()
        
        # Top N sectors
        self.top_n_sectors = QSpinBox()
        self.top_n_sectors.setRange(3, 20)
        self.top_n_sectors.setValue(5)
        control_layout.addRow("Top N Sectors:", self.top_n_sectors)
        
        # Scan button
        self.sectors_button = QPushButton("Get Hot Sectors")
        self.sectors_button.clicked.connect(self.get_hot_sectors)
        control_layout.addRow("", self.sectors_button)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Results table
        self.sectors_table = QTableWidget()
        self.sectors_table.setColumnCount(5)
        self.sectors_table.setHorizontalHeaderLabels([
            "Sector Name", "Limit-Up Count", "Avg Change (%)", 
            "Top Performer", "Score"
        ])
        self.sectors_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.sectors_table)
        
        self.tabs.addTab(tab, "热门板块分析")

    def setup_prediction_tab(self):
        """Setup tab for limit-up continuation prediction"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Control panel
        control_group = QGroupBox("Prediction Controls")
        control_layout = QFormLayout()
        
        # Stock code input
        self.stock_code_input = QLineEdit()
        control_layout.addRow("Stock Code:", self.stock_code_input)
        
        # Consecutive days so far
        self.consecutive_days_input = QSpinBox()
        self.consecutive_days_input.setRange(1, 10)
        self.consecutive_days_input.setValue(2)
        control_layout.addRow("Consecutive Days So Far:", self.consecutive_days_input)
        
        # Predict button
        self.predict_button = QPushButton("Predict Continuation")
        self.predict_button.clicked.connect(self.predict_continuation)
        control_layout.addRow("", self.predict_button)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Results display
        self.prediction_results = QWidget()
        prediction_layout = QVBoxLayout()
        self.prediction_results.setLayout(prediction_layout)
        
        self.prediction_label = QLabel("Enter a stock code and click Predict")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 16px;")
        prediction_layout.addWidget(self.prediction_label)
        
        # Detailed results table
        self.prediction_table = QTableWidget()
        self.prediction_table.setColumnCount(5)
        self.prediction_table.setHorizontalHeaderLabels([
            "Factor", "Value", "Weight", "Score", "Interpretation"
        ])
        self.prediction_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        prediction_layout.addWidget(self.prediction_table)
        
        layout.addWidget(self.prediction_results)
        
        self.tabs.addTab(tab, "涨停延续性预测")

    def scan_consecutive_limit_up(self):
        """Handle scanning for consecutive limit-up stocks"""
        self.statusBar().showMessage("Scanning for consecutive limit-up stocks...")
        self.scan_button.setEnabled(False)
        
        params = {
            "days_look_back": self.days_look_back.value(),
            "min_consecutive_days": self.min_consecutive_days.value(),
            "refresh_cache": self.refresh_cache.isChecked()
        }
        
        self.current_thread = ScannerThread(
            self.scanner, "consecutive_limit_up", params
        )
        self.current_thread.result_signal.connect(self.display_consecutive_limit_up_results)
        self.current_thread.error_signal.connect(self.handle_error)
        self.current_thread.finished.connect(lambda: self.scan_button.setEnabled(True))
        self.current_thread.start()
    
    def scan_potential_breakout(self):
        """Handle scanning for potential breakout stocks"""
        self.statusBar().showMessage("Scanning for potential breakout stocks...")
        self.breakout_button.setEnabled(False)
        
        params = {
            "days_look_back": self.breakout_days.value(),
            "min_score": self.min_score.value()
        }
        
        self.current_thread = ScannerThread(
            self.scanner, "potential_breakout", params
        )
        self.current_thread.result_signal.connect(self.display_potential_breakout_results)
        self.current_thread.error_signal.connect(self.handle_error)
        self.current_thread.finished.connect(lambda: self.breakout_button.setEnabled(True))
        self.current_thread.start()
    
    def get_hot_sectors(self):
        """Handle getting hot sectors"""
        self.statusBar().showMessage("Analyzing hot sectors...")
        self.sectors_button.setEnabled(False)
        
        params = {
            "top_n": self.top_n_sectors.value()
        }
        
        self.current_thread = ScannerThread(
            self.scanner, "hot_sectors", params
        )
        self.current_thread.result_signal.connect(self.display_hot_sectors_results)
        self.current_thread.error_signal.connect(self.handle_error)
        self.current_thread.finished.connect(lambda: self.sectors_button.setEnabled(True))
        self.current_thread.start()
    
    def predict_continuation(self):
        """Handle predicting limit-up continuation"""
        stock_code = self.stock_code_input.text().strip()
        if not stock_code:
            QMessageBox.warning(self, "Input Error", "Please enter a stock code.")
            return
        
        self.statusBar().showMessage(f"Predicting continuation for {stock_code}...")
        self.predict_button.setEnabled(False)
        
        params = {
            "stock_code": stock_code,
            "consecutive_days": self.consecutive_days_input.value()
        }
        
        self.current_thread = ScannerThread(
            self.scanner, "predict_continuation", params
        )
        self.current_thread.result_signal.connect(self.display_prediction_results)
        self.current_thread.error_signal.connect(self.handle_error)
        self.current_thread.finished.connect(lambda: self.predict_button.setEnabled(True))
        self.current_thread.start()
        
    def display_consecutive_limit_up_results(self, result):
        """Display results of consecutive limit-up stocks scan"""
        data = result.get("data", [])
        self.limit_up_table.setRowCount(0)  # Clear the table
        
        if not data:
            self.statusBar().showMessage("No consecutive limit-up stocks found.")
            return
            
        self.limit_up_table.setRowCount(len(data))
        
        for row, stock in enumerate(data):
            # Stock code
            self.limit_up_table.setItem(row, 0, QTableWidgetItem(stock.get("code", "")))
            # Stock name
            self.limit_up_table.setItem(row, 1, QTableWidgetItem(stock.get("name", "")))
            # Consecutive days
            self.limit_up_table.setItem(row, 2, QTableWidgetItem(str(stock.get("consecutive_days", 0))))
            # Last close
            self.limit_up_table.setItem(row, 3, QTableWidgetItem(f"{stock.get('last_close', 0):.2f}"))
            # Change percentage
            change_item = QTableWidgetItem(f"{stock.get('change_percent', 0):.2f}%")
            if stock.get('change_percent', 0) >= 9.5:
                change_item.setBackground(QColor(255, 100, 100))  # Light red for limit-up
            self.limit_up_table.setItem(row, 4, change_item)
            # Volume ratio
            self.limit_up_table.setItem(row, 5, QTableWidgetItem(f"{stock.get('volume_ratio', 0):.2f}"))
            
            # Details button
            details_btn = QPushButton("View")
            details_btn.clicked.connect(lambda checked, c=stock.get("code", ""): self.view_stock_details(c))
            self.limit_up_table.setCellWidget(row, 6, details_btn)
            
        self.statusBar().showMessage(f"Found {len(data)} consecutive limit-up stocks.")
    
    def display_potential_breakout_results(self, result):
        """Display results of potential breakout stocks scan"""
        data = result.get("data", [])
        self.breakout_table.setRowCount(0)  # Clear the table
        
        if not data:
            self.statusBar().showMessage("No potential breakout stocks found.")
            return
            
        self.breakout_table.setRowCount(len(data))
        
        for row, stock in enumerate(data):
            # Stock code
            self.breakout_table.setItem(row, 0, QTableWidgetItem(stock.get("code", "")))
            # Stock name
            self.breakout_table.setItem(row, 1, QTableWidgetItem(stock.get("name", "")))
            
            # Breakout score
            score_item = QTableWidgetItem(f"{stock.get('breakout_score', 0):.1f}")
            score = stock.get('breakout_score', 0)
            if score >= 80:
                score_item.setBackground(QColor(100, 255, 100))  # Light green for high score
            elif score >= 70:
                score_item.setBackground(QColor(200, 255, 100))  # Light yellow-green for medium score
            self.breakout_table.setItem(row, 2, score_item)
            
            # Momentum score
            self.breakout_table.setItem(row, 3, QTableWidgetItem(f"{stock.get('momentum_score', 0):.1f}"))
            # Volume score
            self.breakout_table.setItem(row, 4, QTableWidgetItem(f"{stock.get('volume_score', 0):.1f}"))
            # RSI
            self.breakout_table.setItem(row, 5, QTableWidgetItem(f"{stock.get('rsi', 0):.1f}"))
            
            # MACD signal
            macd_signal = stock.get('macd_signal', '')
            macd_item = QTableWidgetItem(macd_signal)
            if macd_signal == 'BUY':
                macd_item.setBackground(QColor(100, 255, 100))  # Light green for buy
            elif macd_signal == 'SELL':
                macd_item.setBackground(QColor(255, 100, 100))  # Light red for sell
            self.breakout_table.setItem(row, 6, macd_item)
            
            # Details button
            details_btn = QPushButton("View")
            details_btn.clicked.connect(lambda checked, c=stock.get("code", ""): self.view_stock_details(c))
            self.breakout_table.setCellWidget(row, 7, details_btn)
            
        self.statusBar().showMessage(f"Found {len(data)} potential breakout stocks.")
    
    def display_hot_sectors_results(self, result):
        """Display results of hot sectors analysis"""
        data = result.get("data", [])
        self.sectors_table.setRowCount(0)  # Clear the table
        
        if not data:
            self.statusBar().showMessage("No hot sectors found.")
            return
            
        self.sectors_table.setRowCount(len(data))
        
        for row, sector in enumerate(data):
            # Sector name
            self.sectors_table.setItem(row, 0, QTableWidgetItem(sector.get("sector", "")))
            # Limit-up count
            self.sectors_table.setItem(row, 1, QTableWidgetItem(str(sector.get("limit_up_count", 0))))
            # Average change
            avg_change_item = QTableWidgetItem(f"{sector.get('avg_change', 0):.2f}%")
            if sector.get('avg_change', 0) >= 5:
                avg_change_item.setBackground(QColor(255, 100, 100))  # Light red for high change
            self.sectors_table.setItem(row, 2, avg_change_item)
            # Top performer
            top_performer = sector.get("top_performer", {})
            self.sectors_table.setItem(row, 3, QTableWidgetItem(
                f"{top_performer.get('name', '')} ({top_performer.get('code', '')})"
            ))
            # Score
            score_item = QTableWidgetItem(f"{sector.get('score', 0):.1f}")
            score = sector.get('score', 0)
            if score >= 80:
                score_item.setBackground(QColor(100, 255, 100))  # Light green for high score
            elif score >= 70:
                score_item.setBackground(QColor(200, 255, 100))  # Light yellow-green for medium score
            self.sectors_table.setItem(row, 4, score_item)
            
        self.statusBar().showMessage(f"Found {len(data)} hot sectors.")
    
    def display_prediction_results(self, result):
        """Display results of limit-up continuation prediction"""
        data = result.get("data", {})
        
        # Update prediction label
        stock_code = self.stock_code_input.text().strip()
        stock_name = data.get("name", "")
        probability = data.get("probability", 0) * 100
        recommendation = data.get("recommendation", "")
        
        if probability >= 70:
            color = "green"
        elif probability >= 50:
            color = "orange"
        else:
            color = "red"
            
        self.prediction_label.setText(
            f"<b>{stock_name} ({stock_code})</b><br>"
            f"Continuation Probability: <font color='{color}'>{probability:.1f}%</font><br>"
            f"Recommendation: <b>{recommendation}</b>"
        )
        
        # Update factors table
        factors = data.get("factors", [])
        self.prediction_table.setRowCount(0)  # Clear the table
        
        if not factors:
            self.statusBar().showMessage("No prediction factors available.")
            return
            
        self.prediction_table.setRowCount(len(factors))
        
        for row, factor in enumerate(factors):
            # Factor name
            self.prediction_table.setItem(row, 0, QTableWidgetItem(factor.get("name", "")))
            # Value
            self.prediction_table.setItem(row, 1, QTableWidgetItem(str(factor.get("value", ""))))
            # Weight
            self.prediction_table.setItem(row, 2, QTableWidgetItem(f"{factor.get('weight', 0) * 100:.0f}%"))
            
            # Score
            score = factor.get('score', 0)
            score_item = QTableWidgetItem(f"{score:.1f}")
            if score >= 0.7:
                score_item.setBackground(QColor(100, 255, 100))  # Light green for high score
            elif score >= 0.5:
                score_item.setBackground(QColor(200, 255, 100))  # Light yellow-green for medium score
            elif score < 0.3:
                score_item.setBackground(QColor(255, 100, 100))  # Light red for low score
            self.prediction_table.setItem(row, 3, score_item)
            
            # Interpretation
            self.prediction_table.setItem(row, 4, QTableWidgetItem(factor.get("interpretation", "")))
            
        self.statusBar().showMessage(f"Prediction completed for {stock_name} ({stock_code}).")
    
    def view_stock_details(self, stock_code):
        """Show detailed information for a stock"""
        try:
            # Get stock details
            stock_data = self.stock_data.get_stock_details(stock_code)
            if not stock_data:
                QMessageBox.warning(self, "Error", f"No details available for stock {stock_code}")
                return
                
            # Create and show a dialog with stock details
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Details for {stock_data.get('name', '')} ({stock_code})")
            dialog.setMinimumSize(800, 600)
            
            layout = QVBoxLayout()
            dialog.setLayout(layout)
            
            # Create a tab widget for different views
            tabs = QTabWidget()
            layout.addWidget(tabs)
            
            # Price chart tab
            price_tab = QWidget()
            price_layout = QVBoxLayout()
            price_tab.setLayout(price_layout)
            
            # Create chart using pyqtgraph
            chart_view = pg.PlotWidget()
            price_layout.addWidget(chart_view)
            
            # Plot price data if available
            if 'daily_data' in stock_data:
                daily_data = stock_data['daily_data']
                dates = range(len(daily_data))
                prices = [d.get('close', 0) for d in daily_data]
                
                chart_view.plot(dates, prices, pen='b')
                chart_view.setLabel('left', 'Price')
                chart_view.setLabel('bottom', 'Trading Days')
                
            tabs.addTab(price_tab, "Price Chart")
            
            # Information tab
            info_tab = QWidget()
            info_layout = QVBoxLayout()
            info_tab.setLayout(info_layout)
            
            # Create a form to display stock information
            info_form = QFormLayout()
            
            for key, value in stock_data.items():
                if key not in ['daily_data', 'momentum_analysis', 'name']:
                    info_form.addRow(f"{key.replace('_', ' ').title()}:", QLabel(str(value)))
            
            info_layout.addLayout(info_form)
            tabs.addTab(info_tab, "Information")
            
            # Button to close the dialog
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.exec_()
            
        except Exception as e:
            logger.error(f"Error displaying stock details: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to display stock details: {str(e)}")
    
    def handle_error(self, error_message):
        """Handle errors from scanner threads"""
        QMessageBox.critical(self, "Error", error_message)
        self.statusBar().showMessage(f"Error: {error_message}")
        logger.error(f"Scanner error: {error_message}")
        
        # Re-enable all scan buttons
        self.scan_button.setEnabled(True)
        self.breakout_button.setEnabled(True)
        self.sectors_button.setEnabled(True)
        self.predict_button.setEnabled(True)

# Main function
def main():
    app = QApplication(sys.argv)
    window = HotStockGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
