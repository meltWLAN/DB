"""
GUI控制器
连接GUI界面与分析模块的功能类
"""
import os
import sys
import logging
import pandas as pd
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import csv
# 导入增强API可靠性模块
try:
    from enhance_api_reliability import (
        enhance_get_stock_name, 
        enhance_get_stock_names_batch, 
        enhance_get_stock_industry,
        with_retry
    )
    HAS_ENHANCED_API = True
    logger = logging.getLogger(__name__)
    logger.info("成功加载增强API可靠性模块到GUI控制器")
except ImportError:
    HAS_ENHANCED_API = False
    logger = logging.getLogger(__name__)
    logger.warning("无法加载增强API可靠性模块到GUI控制器，将使用基本API功能")
# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))
# 导入分析模块
from momentum_analysis import MomentumAnalyzer
from ma_cross_strategy import MACrossStrategy
from financial_analysis import FinancialAnalyzer
# 导入项目配置
try:
    from src.enhanced.config.settings import LOG_DIR, DATA_DIR, RESULTS_DIR
except ImportError:
    # 设置默认配置
    LOG_DIR = "./logs"
    DATA_DIR = "./data"
    RESULTS_DIR = "./results"
# 配置日志
logger = logging.getLogger(__name__)
class GuiController:
    """GUI控制器类，连接GUI与分析模块"""
    def __init__(self, use_tushare=True, cache_limit=128):
        """初始化控制器
        
        Args:
            use_tushare: 是否使用Tushare数据源
            cache_limit: 缓存限制大小
        """
        self.use_tushare = use_tushare
        self._data_cache = {}
        self._cache_keys = []
        self._cache_limit = cache_limit
        
        # 记录是否使用增强API
        self.using_enhanced_api = HAS_ENHANCED_API
        
        # 初始化分析器
        self.momentum_analyzer = MomentumAnalyzer(use_tushare=use_tushare)
        self.ma_strategy = MACrossStrategy(use_tushare=use_tushare)
        self.financial_analyzer = FinancialAnalyzer(use_tushare=use_tushare)
        self.stock_list = None
        self.momentum_results = None
        self.ma_results = None
        self.combined_results = None
        self.financial_results = None
        
        # 增强型API缓存
        self._stock_name_cache = {}
        self._industry_cache = {}
        
        # 确保数据目录存在
        os.makedirs("data", exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    def _get_cached_data(self, key):
        """从缓存获取数据"""
        if key in self._data_cache:
            # 更新访问顺序
            self._cache_keys.remove(key)
            self._cache_keys.append(key)
            return self._data_cache[key]
        return None
    
    def _set_cached_data(self, key, data):
        """设置缓存数据"""
        # 如果缓存已满，删除最早访问的项目
        if len(self._cache_keys) >= self._cache_limit and self._cache_keys:
            oldest_key = self._cache_keys.pop(0)
            if oldest_key in self._data_cache:
                del self._data_cache[oldest_key]
        
        # 添加新项目到缓存
        if key not in self._data_cache:
            self._cache_keys.append(key)
        self._data_cache[key] = data
    
    def clear_cache(self):
        """清空缓存"""
        self._data_cache.clear()
        self._cache_keys.clear()
    
    def get_stock_list(self, industry=None):
        """获取股票列表"""
        try:
            self.stock_list = self.momentum_analyzer.get_stock_list(industry)
            return self.stock_list
        except Exception as e:
            logger.error(f"获取股票列表失败: {str(e)}")
            return pd.DataFrame()
    def get_stock_industries(self):
        """获取所有行业列表"""
        if self.stock_list is None:
            self.stock_list = self.get_stock_list()
        if 'industry' in self.stock_list.columns:
            industries = sorted(self.stock_list['industry'].dropna().unique().tolist())
            return ["全部"] + industries
        return ["全部"]
    def run_momentum_analysis(self, sample_size=100, industry=None, min_score=60, gui_callback=None):
        """运行动量分析"""
        try:
            # 获取股票列表
            if self.stock_list is None or (industry is not None and industry != "全部"):
                self.stock_list = self.get_stock_list(industry)
            
            if self.stock_list.empty:
                if gui_callback:
                    gui_callback("状态", "获取股票列表失败，请检查网络连接或数据源配置")
                return None
            
            # 记录参数信息
            logger.info(f"启动动量分析 - 样本数量: {sample_size}, 行业: {industry}, 最低分数: {min_score}")
            
            if gui_callback:
                gui_callback("状态", f"开始分析 {len(self.stock_list)} 支股票的动量...")
            
            # 运行分析
            def run_analysis():
                try:
                    # 确保有默认结果，即使分析出错
                    default_results = []
                    
                    try:
                        # 分析股票
                        self.momentum_results = self.momentum_analyzer.analyze_stocks(
                            self.stock_list, sample_size=sample_size, min_score=min_score)
                        
                        # 如果结果为空或None，使用空列表
                        if not self.momentum_results:
                            logger.warning("动量分析返回了空结果或None")
                            self.momentum_results = []
                    except Exception as analysis_error:
                        # 记录错误但不中断
                        logger.error(f"动量分析器内部错误: {str(analysis_error)}", exc_info=True)
                        # 使用空结果继续
                        self.momentum_results = default_results
                    
                    # 记录动量分析完成状态
                    logger.info(f"动量分析完成，结果数量: {len(self.momentum_results)}")
                    
                    # 打印结果样本用于调试
                    if self.momentum_results and len(self.momentum_results) > 0:
                        logger.info(f"动量分析首条结果: {self.momentum_results[0]}")
                    
                    # 确保总是返回结果，即使是空列表
                    if gui_callback:
                        gui_callback("结果", self.momentum_results)
                        
                        # 根据结果提供更详细的状态信息
                        if len(self.momentum_results) > 0:
                            gui_callback("状态", f"动量分析完成，找到 {len(self.momentum_results)} 支符合条件的股票")
                        else:
                            gui_callback("状态", "动量分析完成，未找到符合条件的股票。尝试降低最低分数或更换行业。")
                    
                except Exception as e:
                    logger.error(f"动量分析线程出错: {str(e)}", exc_info=True)
                    # 确保在任何情况下都返回结果
                    if gui_callback:
                        gui_callback("状态", f"动量分析过程中出错: {str(e)}")
                        # 即使出错也返回空结果，让调用者可以继续
                        gui_callback("结果", [])
            
            # 在新线程中运行分析
            analysis_thread = threading.Thread(target=run_analysis)
            analysis_thread.daemon = True
            analysis_thread.start()
            
            return True
        
        except Exception as e:
            logger.error(f"启动动量分析失败: {str(e)}", exc_info=True)
            if gui_callback:
                gui_callback("状态", f"启动动量分析失败: {str(e)}")
                # 即使启动失败也返回空结果
                gui_callback("结果", [])
            return None
    def run_ma_cross_strategy(self, short_ma=5, long_ma=20, initial_capital=100000,
                             stop_loss_pct=0.05, sample_size=100, industry=None, gui_callback=None):
        """运行均线交叉策略"""
        try:
            # 获取股票列表
            if self.stock_list is None or (industry is not None and industry != "全部"):
                self.stock_list = self.get_stock_list(industry)
            
            if self.stock_list.empty:
                if gui_callback:
                    gui_callback("状态", "获取股票列表失败，请检查网络连接或数据源配置")
                return None
            
            # 记录策略参数
            logger.info(f"启动均线交叉策略 - 短期均线: {short_ma}, 长期均线: {long_ma}, 初始资金: {initial_capital}, 止损比例: {stop_loss_pct}, 样本数: {sample_size}, 行业: {industry}")
            
            if gui_callback:
                gui_callback("状态", f"开始分析 {len(self.stock_list)} 支股票的均线交叉策略...")
            
            # 运行策略
            def run_strategy():
                try:
                    # 确保有默认结果，即使分析出错
                    default_results = []
                    
                    try:
                        # 运行策略分析
                        self.ma_results = self.ma_strategy.run_strategy(
                            self.stock_list, short_ma=short_ma, long_ma=long_ma,
                            initial_capital=initial_capital, stop_loss_pct=stop_loss_pct/100,
                            sample_size=sample_size)
                        
                        # 检查结果有效性
                        if not self.ma_results:
                            logger.warning("均线交叉策略返回了空结果或None")
                            self.ma_results = []
                    except Exception as strategy_error:
                        # 记录错误但不中断
                        logger.error(f"均线交叉策略内部错误: {str(strategy_error)}", exc_info=True)
                        # 使用空结果继续
                        self.ma_results = default_results
                    
                    # 记录策略分析完成状态
                    logger.info(f"均线交叉策略分析完成，结果数量: {len(self.ma_results)}")
                    
                    # 打印结果样本用于调试
                    if self.ma_results and len(self.ma_results) > 0:
                        logger.info(f"均线交叉策略首条结果: {self.ma_results[0]}")
                    
                    # 确保总是返回结果，即使是空列表
                    if gui_callback:
                        gui_callback("结果", self.ma_results)
                        
                        # 根据结果提供更详细的状态信息
                        if len(self.ma_results) > 0:
                            buy_signals = sum(1 for r in self.ma_results if r.get('current_signal') == '买入')
                            sell_signals = sum(1 for r in self.ma_results if r.get('current_signal') == '卖出')
                            hold_signals = sum(1 for r in self.ma_results if r.get('current_signal') == '持有')
                            
                            gui_callback("状态", f"均线交叉策略分析完成，共{len(self.ma_results)}支股票 (买入:{buy_signals}, 卖出:{sell_signals}, 持有:{hold_signals})")
                        else:
                            gui_callback("状态", "均线交叉策略分析完成，未找到有效结果。尝试调整均线参数或更换行业。")
                
                except Exception as e:
                    logger.error(f"均线交叉策略线程出错: {str(e)}", exc_info=True)
                    # 确保在任何情况下都返回结果
                    if gui_callback:
                        gui_callback("状态", f"均线交叉策略执行过程中出错: {str(e)}")
                        # 即使出错也返回空结果，让调用者可以继续
                        gui_callback("结果", [])
            
            # 在新线程中运行策略
            strategy_thread = threading.Thread(target=run_strategy)
            strategy_thread.daemon = True
            strategy_thread.start()
            
            return True
        
        except Exception as e:
            logger.error(f"启动均线交叉策略失败: {str(e)}", exc_info=True)
            if gui_callback:
                gui_callback("状态", f"启动均线交叉策略失败: {str(e)}")
                # 即使启动失败也返回空结果
                gui_callback("结果", [])
            return None
    def get_market_overview(self, gui_callback=None):
        """获取市场概览信息"""
        try:
            if gui_callback:
                gui_callback("状态", "正在获取市场概览数据...")
            # 这里可以通过Tushare或其他数据源获取市场概览数据
            # 暂时使用模拟数据
            overview_data = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "indices": [
                    {"name": "上证指数", "code": "000001.SH", "close": 3258.35, "change": 0.65},
                    {"name": "深证成指", "code": "399001.SZ", "close": 10523.67, "change": 0.87},
                    {"name": "创业板指", "code": "399006.SZ", "close": 2075.21, "change": 1.05},
                ],
                "market_stats": {
                    "total_turnover": 876532000000,  # 总成交额(亿元)
                    "up_count": 2534,  # 上涨家数
                    "down_count": 1875,  # 下跌家数
                    "flat_count": 245,  # 平盘家数
                    "limit_up_count": 87,  # 涨停家数
                    "limit_down_count": 15  # 跌停家数
                },
                "industry_performance": [
                    {"name": "医药生物", "change": 2.15, "up_count": 120, "down_count": 45, "total_count": 165, "leading_up": "科兴生物", "leading_down": "华润医药"},
                    {"name": "电子", "change": 1.87, "up_count": 98, "down_count": 52, "total_count": 150, "leading_up": "京东方A", "leading_down": "三安光电"},
                    {"name": "计算机", "change": 1.65, "up_count": 85, "down_count": 40, "total_count": 125, "leading_up": "浪潮信息", "leading_down": "东方财富"},
                    {"name": "有色金属", "change": 1.52, "up_count": 75, "down_count": 35, "total_count": 110, "leading_up": "中国铝业", "leading_down": "紫金矿业"},
                    {"name": "通信", "change": 1.32, "up_count": 65, "down_count": 30, "total_count": 95, "leading_up": "中兴通讯", "leading_down": "华为科技"},
                    {"name": "传媒", "change": 1.25, "up_count": 60, "down_count": 25, "total_count": 85, "leading_up": "分众传媒", "leading_down": "华录百纳"},
                    {"name": "电气设备", "change": 1.12, "up_count": 55, "down_count": 20, "total_count": 75, "leading_up": "特变电工", "leading_down": "思源电气"},
                    {"name": "汽车", "change": 0.95, "up_count": 50, "down_count": 15, "total_count": 65, "leading_up": "长城汽车", "leading_down": "上汽集团"},
                    {"name": "机械设备", "change": 0.85, "up_count": 45, "down_count": 10, "total_count": 55, "leading_up": "三一重工", "leading_down": "徐工机械"},
                    {"name": "食品饮料", "change": 0.75, "up_count": 40, "down_count": 5, "total_count": 45, "leading_up": "贵州茅台", "leading_down": "五粮液"},
                    {"name": "银行", "change": 0.65, "up_count": 35, "down_count": 2, "total_count": 37, "leading_up": "工商银行", "leading_down": "建设银行"},
                    {"name": "房地产", "change": -0.45, "up_count": 20, "down_count": 40, "total_count": 60, "leading_up": "万科A", "leading_down": "保利地产"},
                    {"name": "钢铁", "change": -0.55, "up_count": 15, "down_count": 45, "total_count": 60, "leading_up": "宝钢股份", "leading_down": "鞍钢股份"},
                    {"name": "煤炭", "change": -0.65, "up_count": 10, "down_count": 50, "total_count": 60, "leading_up": "中国神华", "leading_down": "兖州煤业"},
                    {"name": "石油石化", "change": -0.75, "up_count": 5, "down_count": 55, "total_count": 60, "leading_up": "中国石油", "leading_down": "中国石化"}
                ],
                # 当前热门板块数据
                "hot_sectors": [
                    {"name": "人工智能", "change": 3.25, "turnover": 125.32, "up_count": 35, "down_count": 5, "leading_stock": "科大讯飞"},
                    {"name": "半导体芯片", "change": 2.87, "turnover": 98.76, "up_count": 28, "down_count": 7, "leading_stock": "中芯国际"},
                    {"name": "新能源汽车", "change": 2.45, "turnover": 87.45, "up_count": 25, "down_count": 8, "leading_stock": "比亚迪"},
                    {"name": "医疗器械", "change": 2.12, "turnover": 76.23, "up_count": 22, "down_count": 6, "leading_stock": "迈瑞医疗"},
                    {"name": "云计算", "change": 1.98, "turnover": 68.54, "up_count": 20, "down_count": 5, "leading_stock": "阿里云"},
                    {"name": "5G通信", "change": 1.75, "turnover": 65.32, "up_count": 18, "down_count": 6, "leading_stock": "中兴通讯"},
                    {"name": "生物医药", "change": 1.65, "turnover": 58.97, "up_count": 15, "down_count": 5, "leading_stock": "恒瑞医药"}
                ],
                # 未来热门板块预测数据
                "future_hot_sectors": [
                    {"name": "量子计算", "predicted_change": 5.35, "attention_index": 95, "fund_inflow": 35.45, "growth_score": 92, "recommendation": "政策支持+技术突破，未来发展潜力巨大"},
                    {"name": "生物技术", "predicted_change": 4.87, "attention_index": 88, "fund_inflow": 28.76, "growth_score": 90, "recommendation": "医疗革新需求强劲，研发投入持续增加"},
                    {"name": "绿色能源", "predicted_change": 4.25, "attention_index": 85, "fund_inflow": 25.32, "growth_score": 88, "recommendation": "碳中和政策推动，产业链完善"},
                    {"name": "元宇宙", "predicted_change": 3.95, "attention_index": 82, "fund_inflow": 22.45, "growth_score": 85, "recommendation": "虚拟现实融合加速，应用场景扩展"},
                    {"name": "高端制造", "predicted_change": 3.65, "attention_index": 80, "fund_inflow": 18.67, "growth_score": 83, "recommendation": "产业升级趋势明显，自主可控需求强"}
                ]
            }
            if gui_callback:
                gui_callback("状态", "市场概览数据获取完成")
                gui_callback("结果", overview_data)
            return overview_data
        except Exception as e:
            logger.error(f"获取市场概览失败: {str(e)}")
            if gui_callback:
                gui_callback("状态", f"获取市场概览失败: {str(e)}")
            return None
    def export_results(self, result_type, file_path, gui_callback=None):
        """导出分析结果"""
        try:
            if gui_callback:
                gui_callback("状态", f"正在导出{result_type}结果到{file_path}...")
            if result_type == "动量分析" and self.momentum_results:
                # 创建结果DataFrame
                result_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'data' and k != 'score_details'}
                                        for r in self.momentum_results])
                # 根据文件扩展名导出
                if file_path.endswith('.csv'):
                    result_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                elif file_path.endswith('.xlsx'):
                    result_df.to_excel(file_path, index=False)
                else:
                    result_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                if gui_callback:
                    gui_callback("状态", f"成功导出动量分析结果到{file_path}")
                return True
            elif result_type == "均线策略" and self.ma_results:
                # 创建结果DataFrame
                result_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'data' and k != 'stats'}
                                        for r in self.ma_results])
                # 根据文件扩展名导出
                if file_path.endswith('.csv'):
                    result_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                elif file_path.endswith('.xlsx'):
                    result_df.to_excel(file_path, index=False)
                else:
                    result_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                if gui_callback:
                    gui_callback("状态", f"成功导出均线交叉策略结果到{file_path}")
                return True
            elif result_type == "组合策略" and hasattr(self, 'combined_results') and self.combined_results:
                # 创建结果DataFrame
                result_df = pd.DataFrame(self.combined_results)
                # 根据文件扩展名导出
                if file_path.endswith('.csv'):
                    result_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                elif file_path.endswith('.xlsx'):
                    result_df.to_excel(file_path, index=False)
                else:
                    result_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                if gui_callback:
                    gui_callback("状态", f"成功导出组合策略结果到{file_path}")
                return True
            else:
                if gui_callback:
                    gui_callback("状态", f"没有可导出的{result_type}结果")
                return False
        except Exception as e:
            logger.error(f"导出结果失败: {str(e)}")
            if gui_callback:
                gui_callback("状态", f"导出结果失败: {str(e)}")
            return False
    def save_combined_results(self, combined_results, gui_callback=None):
        """保存组合策略的结果
        Args:
            combined_results: 组合策略的分析结果
            gui_callback: GUI回调函数
        Returns:
            bool: 是否保存成功
        """
        try:
            if combined_results:
                self.combined_results = combined_results
                if gui_callback:
                    gui_callback("状态", f"成功保存组合策略结果，共 {len(combined_results)} 条记录")
                return True
            return False
        except Exception as e:
            logger.error(f"保存组合策略结果失败: {str(e)}")
            if gui_callback:
                gui_callback("状态", f"保存组合策略结果失败: {str(e)}")
            return False
    def run_combined_strategy(self, momentum_weight=0.6, ma_weight=0.4, lookback_period=20, 
                             rsi_threshold=70, ma_short=5, ma_long=20, initial_capital=100000,
                             stop_loss_pct=0.05, sample_size=100, industry=None, gui_callback=None):
        """运行组合策略（动量分析+均线交叉）"""
        try:
            # 获取股票列表
            if self.stock_list is None or (industry is not None and industry != "全部"):
                self.stock_list = self.get_stock_list(industry)
                
            if self.stock_list.empty:
                if gui_callback:
                    gui_callback("状态", "获取股票列表失败，请检查网络连接或数据源配置")
                    # 确保返回空结果
                    gui_callback("结果", [])
                return None
            
            # 记录策略参数
            logger.info(f"启动组合策略 - 动量权重: {momentum_weight}, 均线权重: {ma_weight}, "
                       f"回溯期: {lookback_period}, RSI阈值: {rsi_threshold}, "
                       f"短期均线: {ma_short}, 长期均线: {ma_long}, "
                       f"初始资金: {initial_capital}, 止损比例: {stop_loss_pct}, "
                       f"样本数: {sample_size}, 行业: {industry}")
            
            if gui_callback:
                gui_callback("状态", f"开始分析 {len(self.stock_list)} 支股票的组合策略...")
            
            # 通过事件标志来跟踪动量和均线策略的完成情况
            momentum_done = threading.Event()
            ma_done = threading.Event()
            
            # 存储策略结果的变量
            momentum_results = []
            ma_results = []
            
            # 动量策略回调
            def momentum_callback(msg_type, data):
                nonlocal momentum_results
                if msg_type == "状态":
                    if gui_callback:
                        gui_callback("状态", f"组合策略: {data}")
                elif msg_type == "结果":
                    momentum_results = data if data else []
                    logger.info(f"组合策略: 动量分析完成，获得结果数量: {len(momentum_results)}")
                    momentum_done.set()  # 标记动量分析完成
                    
                    if gui_callback:
                        gui_callback("状态", "组合策略: 动量分析完成，继续进行均线分析...")
            
            # 均线策略回调
            def ma_callback(msg_type, data):
                nonlocal ma_results
                if msg_type == "状态":
                    if gui_callback:
                        gui_callback("状态", f"组合策略: {data}")
                elif msg_type == "结果":
                    ma_results = data if data else []
                    logger.info(f"组合策略: 均线分析完成，获得结果数量: {len(ma_results)}")
                    ma_done.set()  # 标记均线分析完成
                    
                    if gui_callback:
                        gui_callback("状态", "组合策略: 均线分析完成，开始合并结果...")
            
            # 运行组合策略
            def run_combined():
                try:
                    # 运行动量分析
                    self.run_momentum_analysis(
                        lookback_period=lookback_period,
                        rsi_threshold=rsi_threshold,
                        sample_size=sample_size,
                        industry=industry,
                        gui_callback=momentum_callback
                    )
                    
                    # 运行均线交叉策略
                    self.run_ma_cross_strategy(
                        short_ma=ma_short,
                        long_ma=ma_long,
                        initial_capital=initial_capital,
                        stop_loss_pct=stop_loss_pct,
                        sample_size=sample_size,
                        industry=industry,
                        gui_callback=ma_callback
                    )
                    
                    # 等待两个策略完成，最多等待120秒
                    all_completed = all([
                        momentum_done.wait(120),  # 等待动量分析完成
                        ma_done.wait(120)         # 等待均线分析完成
                    ])
                    
                    if not all_completed:
                        logger.warning("组合策略: 一个或多个子策略未在规定时间内完成，将使用已有结果继续")
                    
                    # 记录两个策略的结果状态
                    logger.info(f"组合策略: 动量分析结果数量: {len(momentum_results)}, 均线分析结果数量: {len(ma_results)}")
                    
                    # 检查两个策略是否都有结果
                    if not momentum_results and not ma_results:
                        logger.warning("组合策略: 两个子策略都没有返回结果")
                        if gui_callback:
                            gui_callback("状态", "组合策略: 未找到符合条件的股票，尝试调整参数")
                            gui_callback("结果", [])
                        return
                    
                    # 将两个策略结果合并
                    combined_results = []
                    
                    # 创建股票代码到结果的映射
                    momentum_dict = {result['code']: result for result in momentum_results} if momentum_results else {}
                    ma_dict = {result['code']: result for result in ma_results} if ma_results else {}
                    
                    # 获取所有唯一的股票代码
                    all_codes = set(momentum_dict.keys()).union(set(ma_dict.keys()))
                    logger.info(f"组合策略: 合并结果中的唯一股票数量: {len(all_codes)}")
                    
                    # 处理每支股票
                    for code in all_codes:
                        try:
                            # 初始化组合结果
                            combined_result = {'code': code, 'name': '', 'combined_score': 0}
                            
                            # 从动量结果获取信息
                            if code in momentum_dict:
                                momentum_result = momentum_dict[code]
                                combined_result['name'] = momentum_result.get('name', '')
                                combined_result['momentum_score'] = momentum_result.get('total_score', 0)
                                # 复制其他动量分析相关字段
                                for key in ['price_momentum', 'volume_momentum', 'rsi', 'macd', 'kdj']:
                                    if key in momentum_result:
                                        combined_result[key] = momentum_result[key]
                            else:
                                combined_result['momentum_score'] = 0
                                logger.debug(f"组合策略: 股票 {code} 没有动量分析结果")
                            
                            # 从均线结果获取信息
                            if code in ma_dict:
                                ma_result = ma_dict[code]
                                if not combined_result['name'] and 'name' in ma_result:
                                    combined_result['name'] = ma_result.get('name', '')
                                combined_result['ma_signal'] = ma_result.get('current_signal', '无信号')
                                combined_result['ma_score'] = 1 if ma_result.get('current_signal') == '买入' else 0
                                # 复制其他均线分析相关字段
                                for key in ['profit_loss', 'win_rate', 'max_drawdown']:
                                    if key in ma_result:
                                        combined_result[key] = ma_result[key]
                            else:
                                combined_result['ma_score'] = 0
                                combined_result['ma_signal'] = '无信号'
                                logger.debug(f"组合策略: 股票 {code} 没有均线分析结果")
                            
                            # 计算组合得分
                            momentum_score = combined_result.get('momentum_score', 0)
                            ma_score = combined_result.get('ma_score', 0)
                            
                            # 归一化动量得分到0-1范围
                            norm_momentum = min(max(momentum_score / 100, 0), 1) if momentum_score else 0
                            
                            # 计算加权组合得分
                            combined_result['combined_score'] = (
                                norm_momentum * momentum_weight + 
                                ma_score * ma_weight
                            )
                            
                            # 添加到结果列表
                            combined_results.append(combined_result)
                        except Exception as stock_error:
                            logger.error(f"组合策略: 处理股票 {code} 时出错: {str(stock_error)}", exc_info=True)
                    
                    # 按组合得分排序
                    combined_results = sorted(combined_results, key=lambda x: x.get('combined_score', 0), reverse=True)
                    
                    # 记录结果数量和得分范围
                    score_range = (
                        combined_results[0].get('combined_score', 0) if combined_results else 0,
                        combined_results[-1].get('combined_score', 0) if combined_results else 0
                    )
                    logger.info(f"组合策略: 合并完成，共 {len(combined_results)} 支股票，得分范围: {score_range}")
                    
                    # 保存结果
                    if combined_results:
                        try:
                            # 设置日期时间戳
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filepath = f"./results/combined_strategy_{timestamp}.csv"
                            
                            # 确保目录存在
                            os.makedirs("./results", exist_ok=True)
                            
                            # 将结果保存为CSV
                            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                                writer = csv.DictWriter(f, fieldnames=combined_results[0].keys())
                                writer.writeheader()
                                writer.writerows(combined_results)
                                
                            logger.info(f"组合策略: 结果已保存至 {filepath}")
                            
                            if gui_callback:
                                gui_callback("状态", f"组合策略: 分析完成，共 {len(combined_results)} 支股票")
                                gui_callback("结果", combined_results)
                                
                        except Exception as save_error:
                            logger.error(f"组合策略: 保存结果时出错: {str(save_error)}", exc_info=True)
                            # 即使保存失败也要尝试返回结果
                            if gui_callback:
                                gui_callback("状态", f"组合策略: 分析完成但保存失败: {str(save_error)}")
                                gui_callback("结果", combined_results)
                    else:
                        logger.warning("组合策略: 未生成任何合并结果")
                        if gui_callback:
                            gui_callback("状态", "组合策略: 未找到满足条件的股票")
                            gui_callback("结果", [])
                
                except Exception as e:
                    logger.error(f"组合策略执行过程中出错: {str(e)}", exc_info=True)
                    if gui_callback:
                        gui_callback("状态", f"组合策略执行过程中出错: {str(e)}")
                        # 确保即使出错也返回一个空的结果列表
                        gui_callback("结果", [])
            
            # 在新线程中运行组合策略
            combined_thread = threading.Thread(target=run_combined)
            combined_thread.daemon = True
            combined_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"启动组合策略失败: {str(e)}", exc_info=True)
            if gui_callback:
                gui_callback("状态", f"启动组合策略失败: {str(e)}")
                # 确保在任何情况下都返回结果
                gui_callback("结果", [])
            return None
    def run_financial_analysis(self, sample_size=100, industry=None, min_score=60, gui_callback=None):
        """运行财务分析"""
        try:
            # 获取股票列表
            if self.stock_list is None or (industry is not None and industry != "全部"):
                self.stock_list = self.get_stock_list(industry)
            if self.stock_list.empty:
                if gui_callback:
                    gui_callback("状态", "获取股票列表失败，请检查网络连接或数据源配置")
                return None
            if gui_callback:
                gui_callback("状态", f"开始分析 {len(self.stock_list)} 支股票的财务状况...")
            
            # 运行财务分析
            def run_analysis():
                try:
                    # 分析股票
                    self.financial_results = self.financial_analyzer.analyze_financial_stocks(
                        self.stock_list, sample_size=sample_size, min_score=min_score)
                    
                    # 打印结果用于调试
                    print(f"财务分析完成，结果数量: {len(self.financial_results)}")
                    if self.financial_results and len(self.financial_results) > 0:
                        print(f"第一条结果: {self.financial_results[0]}")
                    
                    # 在分析完成后调用GUI回调
                    if gui_callback:
                        if self.financial_results:
                            # 直接调用回调，将结果传给GUI
                            gui_callback("结果", self.financial_results)
                        else:
                            gui_callback("状态", "财务分析完成，未找到符合条件的股票")
                except Exception as e:
                    logger.error(f"财务分析过程中出错: {str(e)}", exc_info=True)
                    if gui_callback:
                        gui_callback("状态", f"财务分析过程中出错: {str(e)}")
            
            # 在新线程中运行分析
            analysis_thread = threading.Thread(target=run_analysis)
            analysis_thread.daemon = True
            analysis_thread.start()
            return True
        except Exception as e:
            logger.error(f"启动财务分析失败: {str(e)}", exc_info=True)
            if gui_callback:
                gui_callback("状态", f"启动财务分析失败: {str(e)}")
            return None
    
    def get_financial_indicator(self, ts_code, start_date=None, end_date=None, period=None, gui_callback=None):
        """获取股票财务指标数据"""
        try:
            if gui_callback:
                gui_callback("状态", f"正在获取 {ts_code} 的财务指标数据...")
            
            # 获取财务指标
            financial_data = self.financial_analyzer.get_financial_indicator(
                ts_code, start_date=start_date, end_date=end_date, period=period)
            
            if financial_data.empty:
                if gui_callback:
                    gui_callback("状态", f"未找到 {ts_code} 的财务指标数据")
                return None
            
            if gui_callback:
                gui_callback("结果", financial_data)
            
            return financial_data
        except Exception as e:
            logger.error(f"获取财务指标数据失败: {str(e)}", exc_info=True)
            if gui_callback:
                gui_callback("状态", f"获取财务指标数据失败: {str(e)}")
            return None
    
    def get_stk_holders(self, ts_code, start_date=None, end_date=None, gui_callback=None):
        """获取股东人数数据"""
        try:
            if gui_callback:
                gui_callback("状态", f"正在获取 {ts_code} 的股东人数数据...")
            
            # 获取股东人数
            holder_data = self.financial_analyzer.get_stk_holders(
                ts_code, start_date=start_date, end_date=end_date)
            
            if gui_callback:
                if not holder_data.empty:
                    gui_callback("结果", holder_data)
                else:
                    gui_callback("状态", f"未找到 {ts_code} 的股东人数数据")
            
            return holder_data
        except Exception as e:
            logger.error(f"获取股东人数数据失败: {str(e)}", exc_info=True)
            if gui_callback:
                gui_callback("状态", f"获取股东人数数据失败: {str(e)}")
            return None
    
    def get_institutional_survey(self, ts_code, start_date=None, end_date=None, gui_callback=None):
        """获取机构调研数据"""
        try:
            if gui_callback:
                gui_callback("状态", f"正在获取 {ts_code} 的机构调研数据...")
            
            # 获取机构调研
            survey_data = self.financial_analyzer.get_institutional_survey(
                ts_code, start_date=start_date, end_date=end_date)
            
            if gui_callback:
                if not survey_data.empty:
                    gui_callback("结果", survey_data)
                else:
                    gui_callback("状态", f"未找到 {ts_code} 的机构调研数据")
            
            return survey_data
        except Exception as e:
            logger.error(f"获取机构调研数据失败: {str(e)}", exc_info=True)
            if gui_callback:
                gui_callback("状态", f"获取机构调研数据失败: {str(e)}")
            return None
    
    def get_chip_distribution(self, ts_code, trade_date=None, gui_callback=None):
        """获取筹码分布数据"""
        try:
            if gui_callback:
                gui_callback("状态", f"正在获取 {ts_code} 的筹码分布数据...")
            
            # 获取筹码分布
            chip_data = self.financial_analyzer.get_chips_data(ts_code, trade_date=trade_date)
            
            if gui_callback:
                if not chip_data.empty:
                    gui_callback("结果", chip_data)
                else:
                    gui_callback("状态", f"未找到 {ts_code} 的筹码分布数据")
            
            return chip_data
        except Exception as e:
            logger.error(f"获取筹码分布数据失败: {str(e)}", exc_info=True)
            if gui_callback:
                gui_callback("状态", f"获取筹码分布数据失败: {str(e)}")
            return None
    
    def run_combined_financial_technical(self, momentum_weight, financial_weight, sample_size, industry, min_score, gui_callback=None):
        """运行结合动量分析和财务分析的综合策略"""
        try:
            # 获取股票列表
            if self.stock_list is None or (industry is not None and industry != "全部"):
                self.stock_list = self.get_stock_list(industry)
            if self.stock_list.empty:
                if gui_callback:
                    gui_callback("状态", "获取股票列表失败，请检查网络连接或数据源配置")
                return None
            
            if gui_callback:
                gui_callback("状态", f"开始综合分析 {len(self.stock_list)} 支股票...")
            
            # 运行分析
            def run_analysis():
                try:
                    # 先进行财务分析
                    financial_results = self.financial_analyzer.analyze_financial_stocks(
                        self.stock_list, sample_size=sample_size, min_score=0)  # 不设最低分，后面综合时筛选
                    
                    # 再进行动量分析
                    def momentum_callback(status_type, data):
                        if status_type == "结果":
                            # 在两个分析都完成后，进行综合
                            combined_results = self.financial_analyzer.combine_financial_technical(
                                financial_results, data, financial_weight)
                            
                            # 根据最低得分筛选
                            filtered_results = [r for r in combined_results if r['combined_score'] >= min_score]
                            
                            # 保存结果并回调
                            self.combined_results = filtered_results
                            if gui_callback:
                                if filtered_results:
                                    gui_callback("结果", filtered_results)
                                else:
                                    gui_callback("状态", "综合分析完成，但未找到符合最低得分的股票")
                    
                    # 运行动量分析
                    self.run_momentum_analysis(
                        sample_size=sample_size, 
                        industry=industry, 
                        min_score=0,  # 不设最低分
                        gui_callback=momentum_callback
                    )
                except Exception as e:
                    logger.error(f"综合分析过程中出错: {str(e)}", exc_info=True)
                    if gui_callback:
                        gui_callback("状态", f"综合分析过程中出错: {str(e)}")
            
            # 在新线程中运行分析
            analysis_thread = threading.Thread(target=run_analysis)
            analysis_thread.daemon = True
            analysis_thread.start()
            return True
        except Exception as e:
            logger.error(f"启动综合分析失败: {str(e)}", exc_info=True)
            if gui_callback:
                gui_callback("状态", f"启动综合分析失败: {str(e)}")
            return None
    
    def get_stock_name(self, ts_code):
        """获取股票名称（支持增强API）"""
        # 检查本地缓存
        if ts_code in self._stock_name_cache:
            return self._stock_name_cache[ts_code]
        
        # 使用增强API
        if self.using_enhanced_api:
            try:
                name = enhance_get_stock_name(ts_code)
                self._stock_name_cache[ts_code] = name
                return name
            except Exception as e:
                logger.warning(f"使用增强API获取股票名称失败: {e}")
                # 失败后尝试使用基本方法
        
        # 使用基本方法
        try:
            # 尝试从stock_list中查找
            if self.stock_list is not None and not self.stock_list.empty and 'ts_code' in self.stock_list.columns and 'name' in self.stock_list.columns:
                matched = self.stock_list[self.stock_list['ts_code'] == ts_code]
                if not matched.empty:
                    name = matched.iloc[0]['name']
                    self._stock_name_cache[ts_code] = name
                    return name
            
            # 尝试使用动量分析器获取
            name = self.momentum_analyzer.get_stock_name(ts_code)
            if name:
                self._stock_name_cache[ts_code] = name
                return name
        except Exception as e:
            logger.error(f"获取股票名称失败: {e}")
        
        # 返回默认名称
        default_name = f"股票{ts_code.split('.')[0]}"
        self._stock_name_cache[ts_code] = default_name
        return default_name
    
    def get_stock_names_batch(self, ts_codes):
        """批量获取股票名称（支持增强API）"""
        results = {}
        missing_codes = []
        
        # 先检查本地缓存
        for ts_code in ts_codes:
            if ts_code in self._stock_name_cache:
                results[ts_code] = self._stock_name_cache[ts_code]
            else:
                missing_codes.append(ts_code)
        
        # 如果有未缓存的代码，使用增强API批量获取
        if missing_codes and self.using_enhanced_api:
            try:
                batch_results = enhance_get_stock_names_batch(missing_codes)
                # 更新结果和缓存
                for code, name in batch_results.items():
                    results[code] = name
                    self._stock_name_cache[code] = name
                # 清空已处理的代码
                missing_codes = []
            except Exception as e:
                logger.warning(f"使用增强API批量获取股票名称失败: {e}")
                # 失败的代码会继续使用基本方法处理
        
        # 对于仍然缺失的代码，逐个使用基本方法获取
        for ts_code in missing_codes:
            results[ts_code] = self.get_stock_name(ts_code)
        
        return results
    
    def get_stock_industry(self, ts_code):
        """获取股票行业（支持增强API）"""
        # 检查本地缓存
        if ts_code in self._industry_cache:
            return self._industry_cache[ts_code]
        
        # 使用增强API
        if self.using_enhanced_api:
            try:
                industry = enhance_get_stock_industry(ts_code)
                self._industry_cache[ts_code] = industry
                return industry
            except Exception as e:
                logger.warning(f"使用增强API获取股票行业失败: {e}")
                # 失败后尝试使用基本方法
        
        # 使用基本方法
        try:
            # 尝试从stock_list中查找
            if self.stock_list is not None and not self.stock_list.empty and 'ts_code' in self.stock_list.columns and 'industry' in self.stock_list.columns:
                matched = self.stock_list[self.stock_list['ts_code'] == ts_code]
                if not matched.empty and pd.notna(matched.iloc[0]['industry']):
                    industry = matched.iloc[0]['industry']
                    self._industry_cache[ts_code] = industry
                    return industry
            
            # 尝试使用动量分析器获取
            if hasattr(self.momentum_analyzer, 'get_stock_industry'):
                industry = self.momentum_analyzer.get_stock_industry(ts_code)
                if industry:
                    self._industry_cache[ts_code] = industry
                    return industry
        except Exception as e:
            logger.error(f"获取股票行业失败: {e}")
        
        # 返回默认行业
        default_industry = "未知行业"
        self._industry_cache[ts_code] = default_industry
        return default_industry
    
    def clear_enhanced_cache(self):
        """清除增强API相关的缓存"""
        self._stock_name_cache.clear()
        self._industry_cache.clear()
        
        # 如果存在增强API的缓存更新函数，也调用它
        if self.using_enhanced_api:
            try:
                from enhance_api_reliability import update_cache_now
                update_cache_now()
                logger.info("已清除增强API缓存")
            except Exception as e:
                logger.error(f"清除增强API缓存失败: {e}")
        
        return {
            "status": "success",
            "cleared": True,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_api_stats(self):
        """获取API统计信息"""
        if not self.using_enhanced_api:
            return {
                "status": "not_available",
                "message": "未使用增强API模块"
            }
        
        try:
            from enhance_api_reliability import get_cache_manager
            stats = get_cache_manager()
            return {
                "status": "success",
                "data": stats,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.error(f"获取API统计信息失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            }