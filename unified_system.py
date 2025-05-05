#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一系统启动器
整合所有分析模块，提供统一的启动和管理界面
"""

import os
import sys
import traceback

# 添加当前目录和src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

import logging
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import json
import threading
from PIL import Image, ImageTk

# 设置日志
def setup_logging():
    """配置日志系统"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f'unified_system_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('UnifiedSystem')

# 创建主日志记录器
logger = setup_logging()

# 尝试导入核心模块
try:
    # 导入核心分析模块
    from strategies.hot_stock_scanner import HotStockScanner
    from strategies.momentum_analysis import MomentumAnalyzer
    from strategies.ma_cross_strategy import MACrossStrategy
    from strategies.financial_analysis import FinancialAnalyzer
    logger.info("成功导入核心分析模块")
except ImportError as e:
    logger.error(f"导入核心分析模块失败: {str(e)}")
    logger.error(f"导入错误详情: {traceback.format_exc()}")

# 尝试导入机器学习模块
try:
    # 强制使用当前目录下的ml_momentum_model.py
    import importlib.util
    ml_path = os.path.join(current_dir, 'ml_momentum_model.py')
    if os.path.exists(ml_path):
        spec = importlib.util.spec_from_file_location("ml_module", ml_path)
        ml_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ml_module)
        MLMomentumModel = ml_module.MLMomentumModel
        logger.info("成功从根目录导入MLMomentumModel")
    else:
        logger.error(f"ML模型文件不存在: {ml_path}")
        raise ImportError(f"找不到ML动量模型文件: {ml_path}")
except ImportError as e:
    logger.error(f"导入机器学习模块失败: {str(e)}")
    logger.error(f"可能需要检查ml_momentum_model.py文件是否存在")
    logger.error(f"导入错误详情: {traceback.format_exc()}")

# 尝试导入数据管理模块
try:
    from stock_data_storage import StockData
    from enhanced_data_provider import EnhancedDataProvider
    from data_quality_checker import DataQualityChecker
    logger.info("成功导入数据管理模块")
except ImportError as e:
    logger.error(f"导入数据管理模块失败: {str(e)}")
    logger.error(f"导入错误详情: {traceback.format_exc()}")

# 尝试导入风险管理模块
try:
    from risk.risk_management import RiskManager
    logger.info("成功导入风险管理模块")
except ImportError as e:
    logger.error(f"导入风险管理模块失败: {str(e)}")
    logger.error(f"导入错误详情: {traceback.format_exc()}")

# 尝试导入回测模块
try:
    from backtest.backtest_engine import BacktestEngine
    logger.info("成功导入回测模块")
except ImportError as e:
    logger.error(f"导入回测模块失败: {str(e)}")
    logger.error(f"导入错误详情: {traceback.format_exc()}")

# 尝试导入分析模块
try:
    from analysis.capital_flow import CapitalFlowAnalyzer
    from analysis.sentiment import SentimentAnalyzer
    logger.info("成功导入分析模块")
except ImportError as e:
    logger.error(f"导入分析模块失败: {str(e)}")
    logger.error(f"导入错误详情: {traceback.format_exc()}")

# 尝试导入通知模块
try:
    from notification.notifier import NotificationManager
    logger.info("成功导入通知模块")
except ImportError as e:
    logger.error(f"导入通知模块失败: {str(e)}")
    logger.error(f"导入错误详情: {traceback.format_exc()}")

class UnifiedSystem:
    """统一系统类"""
    
    def __init__(self):
        """初始化统一系统"""
        self.logger = logger
        self.init_components()
        
    def init_components(self):
        """初始化所有组件"""
        try:
            # 创建必要的目录
            for directory in ['data', 'cache', 'results', 'models', 'config', 'logs']:
                os.makedirs(directory, exist_ok=True)
            
            self.components = {}
            self.init_status = {}
            
            # 初始化数据管理组件
            try:
                self.components['stock_data'] = StockData()
                self.components['data_provider'] = EnhancedDataProvider()
                self.components['data_checker'] = DataQualityChecker()
                self.init_status['data'] = True
                self.logger.info("数据管理组件初始化成功")
            except Exception as e:
                self.logger.error(f"数据管理组件初始化失败: {str(e)}")
                self.init_status['data'] = False
            
            # 初始化核心分析组件
            try:
                self.components['hot_stock_scanner'] = HotStockScanner()
                self.components['momentum_analyzer'] = MomentumAnalyzer()
                self.components['ma_strategy'] = MACrossStrategy()
                self.components['financial_analyzer'] = FinancialAnalyzer()
                self.init_status['analysis'] = True
                self.logger.info("核心分析组件初始化成功")
            except Exception as e:
                self.logger.error(f"核心分析组件初始化失败: {str(e)}")
                self.init_status['analysis'] = False
            
            # 初始化机器学习组件
            try:
                if 'MLMomentumModel' in globals():
                    self.components['ml_momentum'] = MLMomentumModel()
                    self.init_status['ml'] = True
                    self.logger.info("机器学习组件初始化成功")
                else:
                    self.logger.error("MLMomentumModel未定义，无法初始化机器学习组件")
                    self.init_status['ml'] = False
            except Exception as e:
                self.logger.error(f"机器学习组件初始化失败: {str(e)}")
                self.init_status['ml'] = False
            
            # 初始化风险管理组件
            try:
                self.components['risk_manager'] = RiskManager()
                self.init_status['risk'] = True
                self.logger.info("风险管理组件初始化成功")
            except Exception as e:
                self.logger.error(f"风险管理组件初始化失败: {str(e)}")
                self.init_status['risk'] = False
            
            # 初始化回测组件
            try:
                self.components['backtest_engine'] = BacktestEngine()
                self.init_status['backtest'] = True
                self.logger.info("回测组件初始化成功")
            except Exception as e:
                self.logger.error(f"回测组件初始化失败: {str(e)}")
                self.init_status['backtest'] = False
            
            # 初始化分析组件
            try:
                self.components['capital_flow'] = CapitalFlowAnalyzer()
                self.components['sentiment'] = SentimentAnalyzer()
                self.init_status['adv_analysis'] = True
                self.logger.info("高级分析组件初始化成功")
            except Exception as e:
                self.logger.error(f"高级分析组件初始化失败: {str(e)}")
                self.init_status['adv_analysis'] = False
            
            # 初始化通知组件
            try:
                self.components['notification'] = NotificationManager()
                self.init_status['notification'] = True
                self.logger.info("通知组件初始化成功")
            except Exception as e:
                self.logger.error(f"通知组件初始化失败: {str(e)}")
                self.init_status['notification'] = False
            
            # 检查初始化状态
            failed_components = [k for k, v in self.init_status.items() if not v]
            if failed_components:
                self.logger.warning(f"以下组件初始化失败: {', '.join(failed_components)}")
            else:
                self.logger.info("所有组件初始化成功")
            
        except Exception as e:
            self.logger.error(f"组件初始化过程中发生错误: {str(e)}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            raise
            
    def launch_gui(self):
        """启动GUI界面"""
        try:
            # 为GUI模块创建临时的MLMomentumModel
            if 'MLMomentumModel' in globals():
                # 添加到内置命名空间，供GUI模块导入
                import builtins
                builtins.MLMomentumModel = MLMomentumModel
                self.logger.info("已将MLMomentumModel添加到内置命名空间")
            
            # 导入GUI模块
            try:
                from stock_analysis_gui import StockAnalysisGUI
                from ml_momentum_gui import MLMomentumGUI
                from hot_stock_gui import HotStockGUI
                
                # 创建主窗口
                root = tk.Tk()
                root.title("统一股票分析系统")
                root.geometry("1200x800")
                
                # 显示初始化状态
                failed_components = [k for k, v in self.init_status.items() if not v]
                if failed_components:
                    msg = f"警告: 以下组件初始化失败: {', '.join(failed_components)}\n系统可能无法正常工作"
                    messagebox.showwarning("初始化警告", msg)
                
                # 创建标签页
                notebook = ttk.Notebook(root)
                notebook.pack(expand=True, fill='both')
                
                # 添加各个模块的GUI
                # 创建各个模块的标签页
                stock_analysis_tab = StockAnalysisGUI(notebook)
                ml_momentum_tab = MLMomentumGUI(notebook)
                hot_stock_tab = HotStockGUI(notebook)
                
                # 运行主循环
                root.mainloop()
                
            except ImportError as e:
                self.logger.error(f"导入GUI模块失败: {str(e)}")
                self.logger.error(f"错误详情: {traceback.format_exc()}")
                
                # 如果GUI导入失败，尝试创建一个简单的错误提示界面
                root = tk.Tk()
                root.title("股票分析系统 - 错误")
                root.geometry("800x400")
                
                frame = ttk.Frame(root, padding=20)
                frame.pack(fill=tk.BOTH, expand=True)
                
                ttk.Label(frame, text="启动系统时发生错误", font=("Arial", 14, "bold")).pack(pady=10)
                ttk.Label(frame, text=f"错误信息: {str(e)}", wraplength=700).pack(pady=10)
                ttk.Label(frame, text="请检查日志文件获取更多信息").pack(pady=10)
                
                # 显示日志区域
                log_frame = ttk.LabelFrame(frame, text="最近日志")
                log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
                
                log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
                log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                
                # 添加最新日志内容
                log_files = [f for f in os.listdir('logs') if f.startswith('unified_system_')]
                if log_files:
                    latest_log = os.path.join('logs', sorted(log_files)[-1])
                    try:
                        with open(latest_log, 'r', encoding='utf-8') as f:
                            log_content = f.readlines()
                            # 只显示最后20行
                            for line in log_content[-20:]:
                                log_text.insert(tk.END, line)
                    except Exception as e:
                        log_text.insert(tk.END, f"无法读取日志文件: {str(e)}")
                
                ttk.Button(frame, text="退出", command=root.destroy).pack(pady=10)
                
                root.mainloop()
                
        except Exception as e:
            self.logger.error(f"启动GUI失败: {str(e)}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            raise
            
    def start_cli(self):
        """启动命令行界面"""
        self.logger.info("启动命令行界面")
        # 显示系统状态
        print("=== 股票分析系统 CLI 模式 ===")
        
        # 显示组件状态
        print("\n组件状态:")
        for name, status in self.init_status.items():
            print(f"  - {name}: {'正常' if status else '失败'}")
        
        # 简单的CLI菜单
        while True:
            print("\n可用命令:")
            print("1. 运行动量分析")
            print("2. 运行均线交叉策略")
            print("3. 运行机器学习分析")
            print("4. 扫描热门股票")
            print("5. 退出")
            
            choice = input("\n请输入命令编号: ")
            
            if choice == '1':
                print("运行动量分析...")
                # TODO: 实现命令行版动量分析
            elif choice == '2':
                print("运行均线交叉策略...")
                # TODO: 实现命令行版均线交叉策略
            elif choice == '3':
                print("运行机器学习分析...")
                # TODO: 实现命令行版机器学习分析
            elif choice == '4':
                print("扫描热门股票...")
                # TODO: 实现命令行版热门股票扫描
            elif choice == '5':
                print("退出系统...")
                break
            else:
                print("无效的命令，请重新输入")
        
    def start_web(self):
        """启动Web界面"""
        self.logger.info("启动Web界面")
        # TODO: 实现Web界面

def main():
    """主函数"""
    try:
        # 创建统一系统实例
        system = UnifiedSystem()
        
        # 根据命令行参数选择启动方式
        if len(sys.argv) > 1 and sys.argv[1] == '--cli':
            system.start_cli()
        elif len(sys.argv) > 1 and sys.argv[1] == '--web':
            system.start_web()
        else:
            # 默认启动GUI界面
            system.launch_gui()
        
    except Exception as e:
        logging.error(f"系统启动失败: {str(e)}")
        logging.error(f"错误详情: {traceback.format_exc()}")
        print(f"系统启动失败: {str(e)}")
        print("请检查日志文件获取更多信息")
        if hasattr(sys, 'exit'):
            sys.exit(1)

if __name__ == "__main__":
    main() 