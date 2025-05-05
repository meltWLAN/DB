"""
应用程序入口模块
提供桌面应用的启动入口
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.main_window import MainWindow
from ..utils.logger import get_logger

logger = get_logger(__name__)

def run_app():
    """运行应用程序"""
    try:
        # 创建应用实例
        app = QApplication(sys.argv)
        
        # 设置应用属性
        app.setApplicationName("股票分析系统")
        app.setOrganizationName("StockAnalysis")
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
        app.setStyle('Fusion')  # 使用Fusion风格，看起来更现代
        
        # 创建主窗口
        window = MainWindow()
        window.show()
        
        # 运行应用
        logger.info("Application started")
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_app() 