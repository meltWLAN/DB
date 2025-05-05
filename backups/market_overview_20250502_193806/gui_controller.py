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
    # 使用LRU缓存优化数据加载
    def __init__(self, use_tushare=False, cache_limit=128):
        """初始化控制器
        
        Args:
            use_tushare: 是否使用Tushare数据源
            cache_limit: 缓存限制大小
        """
        self.use_tushare = use_tushare
        self._data_cache = {}
        self._cache_keys = []
        self._cache_limit = cache_limit
        
        # 确保数据目录存在
        os.makedirs("data", exist_ok=True)
    
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
        
    """GUI控制器类，连接GUI与分析模块"""
    def __init__(self, use_tushare=True):
        """初始化控制器"""
        self.momentum_analyzer = MomentumAnalyzer(use_tushare=use_tushare)
        self.ma_strategy = MACrossStrategy(use_tushare=use_tushare)
        self.financial_analyzer = FinancialAnalyzer(use_tushare=use_tushare)
        self.stock_list = None
        self.momentum_results = None
        self.ma_results = None
        self.combined_results = None
        self.financial_results = None
        # 确保目录存在
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
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
            if gui_callback:
                gui_callback("状态", f"开始分析 {len(self.stock_list)} 支股票的动量...")
            # 运行分析
            def run_analysis():
                try:
                    # 分析股票
                    self.momentum_results = self.momentum_analyzer.analyze_stocks(
                        self.stock_list, sample_size=sample_size, min_score=min_score)
                    # 打印结果用于调试
                    print(f"动量分析完成，结果数量: {len(self.momentum_results)}")
                    if self.momentum_results and len(self.momentum_results) > 0:
                        print(f"第一条结果: {self.momentum_results[0]}")
                    # 在分析完成后调用GUI回调
                    if gui_callback:
                        if self.momentum_results:
                            # 直接调用回调，将结果传给GUI
                            gui_callback("结果", self.momentum_results)
                        else:
                            gui_callback("状态", "动量分析完成，未找到符合条件的股票")
                except Exception as e:
                    logger.error(f"分析过程中出错: {str(e)}", exc_info=True)
                    if gui_callback:
                        gui_callback("状态", f"分析过程中出错: {str(e)}")
            # 在新线程中运行分析
            analysis_thread = threading.Thread(target=run_analysis)
            analysis_thread.daemon = True
            analysis_thread.start()
            return True
        except Exception as e:
            logger.error(f"启动动量分析失败: {str(e)}", exc_info=True)
            if gui_callback:
                gui_callback("状态", f"启动动量分析失败: {str(e)}")
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
            if gui_callback:
                gui_callback("状态", f"开始分析 {len(self.stock_list)} 支股票的均线交叉策略...")
            # 运行策略
            def run_strategy():
                try:
                    # 运行策略分析
                    self.ma_results = self.ma_strategy.run_strategy(
                        self.stock_list, short_ma=short_ma, long_ma=long_ma,
                        initial_capital=initial_capital, stop_loss_pct=stop_loss_pct/100,
                        sample_size=sample_size)
                    # 在分析完成后调用GUI回调
                    if gui_callback:
                        if self.ma_results:
                            # 直接调用回调，将结果传给GUI
                            gui_callback("结果", self.ma_results)
                        else:
                            gui_callback("状态", "均线交叉策略分析完成，未得到有效结果")
                except Exception as e:
                    logger.error(f"策略执行过程中出错: {str(e)}")
                    if gui_callback:
                        gui_callback("状态", f"策略执行过程中出错: {str(e)}")
            # 在新线程中运行策略
            strategy_thread = threading.Thread(target=run_strategy)
            strategy_thread.daemon = True
            strategy_thread.start()
            return True
        except Exception as e:
            logger.error(f"启动均线交叉策略失败: {str(e)}")
            if gui_callback:
                gui_callback("状态", f"启动均线交叉策略失败: {str(e)}")
            return None
    def get_market_overview(self, gui_callback=None):
        """获取市场概览信息"""
        try:
            if gui_callback:
                gui_callback("状态", "正在获取市场概览数据...")
            
            # 尝试从数据源管理器获取真实市场数据
            try:
                from src.enhanced.data.fetchers.data_source_manager import DataSourceManager
                data_manager = DataSourceManager()
                
                # 获取当前交易日期
                latest_trade_date = data_manager.get_latest_trading_date()
                
                # 获取市场概览基础数据
                market_data = data_manager.get_market_overview(latest_trade_date)
                if not market_data:
                    raise ValueError("无法获取市场概览数据")
                
                # 获取指数数据
                indices_data = []
                index_codes = ["000001.SH", "399001.SZ", "399006.SZ", "000016.SH", "000300.SH", "000905.SH"]
                index_names = ["上证指数", "深证成指", "创业板指", "上证50", "沪深300", "中证500"]
                
                for i, code in enumerate(index_codes):
                    try:
                        # 获取最近5个交易日的指数数据以计算趋势
                        prev_date = data_manager.get_previous_trading_date(latest_trade_date, 5)
                        index_df = data_manager.get_stock_index_data(code, prev_date, latest_trade_date)
                        
                        if index_df is not None and not index_df.empty:
                            # 获取最新行情
                            latest_data = index_df.iloc[-1]
                            
                            # 计算5日涨跌幅
                            change_5d = ((latest_data['close'] / index_df.iloc[0]['close']) - 1) * 100 if len(index_df) > 1 else 0
                            
                            # 计算量比（当日成交量/5日平均成交量）
                            volume_ratio = latest_data['volume'] / index_df['volume'].mean() if 'volume' in index_df.columns else 1
                            
                            indices_data.append({
                                "name": index_names[i],
                                "code": code,
                                "close": latest_data['close'],
                                "change": ((latest_data['close'] / latest_data['open']) - 1) * 100 if 'open' in latest_data else 0,
                                "change_5d": change_5d,
                                "volume": latest_data['volume'] if 'volume' in latest_data else 0,
                                "amount": latest_data['amount'] if 'amount' in latest_data else 0,
                                "volume_ratio": volume_ratio,
                                "trend": self._analyze_trend(index_df) if hasattr(self, '_analyze_trend') else "中性"
                            })
                    except Exception as e:
                        logger.error(f"获取指数 {code} 数据失败: {str(e)}")
                        # 添加一个基本的占位数据
                        indices_data.append({
                            "name": index_names[i],
                            "code": code,
                            "close": 0,
                            "change": 0,
                            "change_5d": 0,
                            "volume": 0,
                            "amount": 0,
                            "volume_ratio": 0,
                            "trend": "中性"
                        })
                
                # 获取行业表现数据
                industry_performance = data_manager.get_industry_performance(latest_trade_date)
                industries_data = []
                
                if industry_performance is not None and not industry_performance.empty:
                    for _, row in industry_performance.iterrows():
                        industry_code = row.get('industry_code', '')
                        industry_name = row.get('industry_name', '')
                        change = row.get('change_pct', 0)
                        
                        # 获取行业成分股
                        try:
                            industry_stocks = data_manager.get_industry_stocks(industry_code)
                            if industry_stocks is not None and not industry_stocks.empty:
                                # 按涨跌幅排序
                                industry_stocks_data = []
                                for _, stock in industry_stocks.iterrows():
                                    stock_code = stock.get('ts_code', '')
                                    stock_data = data_manager.get_daily_data(stock_code, 
                                                                            data_manager.get_previous_trading_date(latest_trade_date, 1), 
                                                                            latest_trade_date)
                                    if stock_data is not None and not stock_data.empty:
                                        change_pct = ((stock_data.iloc[-1]['close'] / stock_data.iloc[-1]['open']) - 1) * 100
                                        industry_stocks_data.append({
                                            'code': stock_code,
                                            'name': stock.get('name', ''),
                                            'change_pct': change_pct
                                        })
                                
                                # 按涨跌幅排序
                                industry_stocks_data.sort(key=lambda x: x['change_pct'], reverse=True)
                                
                                # 找出领涨股和领跌股
                                leading_up = industry_stocks_data[0] if industry_stocks_data else {'name': '', 'change_pct': 0}
                                leading_down = industry_stocks_data[-1] if industry_stocks_data else {'name': '', 'change_pct': 0}
                                
                                # 计算上涨和下跌股票数量
                                up_count = sum(1 for stock in industry_stocks_data if stock['change_pct'] > 0)
                                down_count = sum(1 for stock in industry_stocks_data if stock['change_pct'] < 0)
                                
                                industries_data.append({
                                    'name': industry_name,
                                    'code': industry_code,
                                    'change': change,
                                    'up_count': up_count,
                                    'down_count': down_count,
                                    'total_count': len(industry_stocks_data),
                                    'leading_up': leading_up['name'],
                                    'leading_up_change': leading_up['change_pct'],
                                    'leading_down': leading_down['name'],
                                    'leading_down_change': leading_down['change_pct'],
                                    'strength_index': self._calculate_industry_strength(up_count, down_count, change) if hasattr(self, '_calculate_industry_strength') else 50
                                })
                        except Exception as e:
                            logger.error(f"处理行业 {industry_name} 数据失败: {str(e)}")
                
                # 按行业强度指数排序
                industries_data.sort(key=lambda x: x.get('strength_index', 0), reverse=True)
                
                # 提取热门板块数据（强度指数最高的几个行业）
                hot_sectors = industries_data[:7] if len(industries_data) >= 7 else industries_data
                
                # 分析市场情绪和预测未来热门板块 
                market_sentiment = self._analyze_market_sentiment(market_data, indices_data)
                future_hot_sectors = self._predict_future_hot_sectors(industries_data, market_sentiment)
                
                # 构建市场概览数据
                overview_data = {
                    "date": latest_trade_date,
                    "indices": indices_data,
                    "market_stats": {
                        "total_turnover": market_data.get('total_amount', 0),
                        "up_count": market_data.get('up_count', 0), 
                        "down_count": market_data.get('down_count', 0),
                        "flat_count": market_data.get('flat_count', 0),
                        "limit_up_count": market_data.get('limit_up_count', 0),
                        "limit_down_count": market_data.get('limit_down_count', 0),
                        "turnover_rate": market_data.get('turnover_rate', 0),
                        "market_sentiment": market_sentiment
                    },
                    "industry_performance": industries_data,
                    "hot_sectors": [{
                        "name": sector.get('name', ''),
                        "change": sector.get('change', 0),
                        "turnover": sector.get('turnover', 0),  # 这个数据可能需要额外获取 
                        "up_count": sector.get('up_count', 0),
                        "down_count": sector.get('down_count', 0),
                        "leading_stock": sector.get('leading_up', '')
                    } for sector in hot_sectors],
                    "future_hot_sectors": future_hot_sectors
                }
                
                if gui_callback:
                    gui_callback("状态", "市场概览数据获取完成")
                    gui_callback("结果", overview_data)
                return overview_data
            except Exception as api_error:
                logger.error(f"尝试获取真实市场数据失败，使用模拟数据: {str(api_error)}")
                if gui_callback:
                    gui_callback("状态", "无法获取实时市场数据，使用模拟数据")
                
                # 使用模拟数据
                from datetime import datetime
                today = datetime.now().strftime('%Y-%m-%d')
                
                # 模拟指数数据
                indices_data = [
                    {"name": "上证指数", "code": "000001.SH", "close": 3150.78, "change": 0.85, "change_5d": 2.15, "volume": 1200000000, "amount": 1500000000, "volume_ratio": 1.02, "trend": "上涨"},
                    {"name": "深证成指", "code": "399001.SZ", "close": 10230.56, "change": 1.05, "change_5d": 2.85, "volume": 1000000000, "amount": 1200000000, "volume_ratio": 1.08, "trend": "上涨"},
                    {"name": "创业板指", "code": "399006.SZ", "close": 2180.45, "change": 1.25, "change_5d": 3.25, "volume": 800000000, "amount": 950000000, "volume_ratio": 1.15, "trend": "强势上涨"},
                    {"name": "上证50", "code": "000016.SH", "close": 3050.67, "change": 0.65, "change_5d": 1.85, "volume": 500000000, "amount": 750000000, "volume_ratio": 0.95, "trend": "震荡"},
                    {"name": "沪深300", "code": "000300.SH", "close": 4120.34, "change": 0.75, "change_5d": 2.05, "volume": 650000000, "amount": 900000000, "volume_ratio": 1.05, "trend": "上涨"},
                    {"name": "中证500", "code": "000905.SH", "close": 6680.12, "change": 0.90, "change_5d": 2.35, "volume": 750000000, "amount": 850000000, "volume_ratio": 1.10, "trend": "上涨"}
                ]
                
                # 模拟行业数据
                industries_data = [
                    {"name": "电子", "code": "ELE", "change": 1.75, "up_count": 58, "down_count": 25, "total_count": 90, "leading_up": "科大讯飞", "leading_up_change": 5.63, "leading_down": "华工科技", "leading_down_change": -2.41, "strength_index": 85},
                    {"name": "医疗健康", "code": "MED", "change": 1.25, "up_count": 65, "down_count": 30, "total_count": 100, "leading_up": "迈瑞医疗", "leading_up_change": 4.21, "leading_down": "通策医疗", "leading_down_change": -1.85, "strength_index": 80},
                    {"name": "半导体", "code": "SEM", "change": 1.95, "up_count": 42, "down_count": 15, "total_count": 60, "leading_up": "中芯国际", "leading_up_change": 6.37, "leading_down": "北方华创", "leading_down_change": -1.28, "strength_index": 90},
                    {"name": "新能源", "code": "NER", "change": 1.45, "up_count": 72, "down_count": 28, "total_count": 105, "leading_up": "宁德时代", "leading_up_change": 3.97, "leading_down": "亿纬锂能", "leading_down_change": -2.05, "strength_index": 82},
                    {"name": "计算机", "code": "COM", "change": 1.15, "up_count": 48, "down_count": 32, "total_count": 85, "leading_up": "浪潮信息", "leading_up_change": 3.82, "leading_down": "用友网络", "leading_down_change": -1.79, "strength_index": 75},
                    {"name": "消费", "code": "CON", "change": 0.95, "up_count": 52, "down_count": 38, "total_count": 95, "leading_up": "贵州茅台", "leading_up_change": 2.84, "leading_down": "伊利股份", "leading_down_change": -1.63, "strength_index": 70},
                    {"name": "金融", "code": "FIN", "change": 0.65, "up_count": 35, "down_count": 25, "total_count": 65, "leading_up": "招商银行", "leading_up_change": 2.15, "leading_down": "中国平安", "leading_down_change": -1.42, "strength_index": 65},
                    {"name": "有色金属", "code": "MET", "change": 1.35, "up_count": 45, "down_count": 30, "total_count": 80, "leading_up": "紫金矿业", "leading_up_change": 4.52, "leading_down": "洛阳钼业", "leading_down_change": -1.95, "strength_index": 77}
                ]
                
                # 模拟热门板块
                hot_sectors = [
                    {"name": "半导体", "change": 1.95, "turnover": 85.2, "up_count": 42, "down_count": 15, "leading_stock": "中芯国际"},
                    {"name": "电子", "change": 1.75, "turnover": 92.5, "up_count": 58, "down_count": 25, "leading_stock": "科大讯飞"},
                    {"name": "新能源", "change": 1.45, "turnover": 105.8, "up_count": 72, "down_count": 28, "leading_stock": "宁德时代"},
                    {"name": "医疗健康", "change": 1.25, "turnover": 78.3, "up_count": 65, "down_count": 30, "leading_stock": "迈瑞医疗"},
                    {"name": "有色金属", "change": 1.35, "turnover": 68.9, "up_count": 45, "down_count": 30, "leading_stock": "紫金矿业"},
                    {"name": "计算机", "change": 1.15, "turnover": 72.4, "up_count": 48, "down_count": 32, "leading_stock": "浪潮信息"},
                    {"name": "消费", "change": 0.95, "turnover": 89.7, "up_count": 52, "down_count": 38, "leading_stock": "贵州茅台"}
                ]
                
                # 模拟未来热门板块预测
                future_hot_sectors = [
                    {"name": "半导体", "predicted_change": 3.5, "attention_index": 92, "fund_inflow": 15.8, "growth_score": 95, "recommendation": "强势行业,主力资金持续流入,中芯国际等龙头表现优异"},
                    {"name": "人工智能", "predicted_change": 3.2, "attention_index": 90, "fund_inflow": 14.5, "growth_score": 93, "recommendation": "行业基本面向好,整体趋势向上,关注度提升"},
                    {"name": "新能源", "predicted_change": 2.8, "attention_index": 88, "fund_inflow": 13.2, "growth_score": 91, "recommendation": "行业处于成长期,中长期向好"},
                    {"name": "医疗器械", "predicted_change": 2.5, "attention_index": 85, "fund_inflow": 12.7, "growth_score": 88, "recommendation": "防御性较强,估值处于合理区间"},
                    {"name": "云计算", "predicted_change": 2.3, "attention_index": 82, "fund_inflow": 11.8, "growth_score": 86, "recommendation": "数字经济发展核心,成长确定性高"}
                ]
                
                # 市场情绪
                market_sentiment = "偏乐观"
                
                # 构建模拟的市场概览数据
                overview_data = {
                    "date": today,
                    "indices": indices_data,
                    "market_stats": {
                        "total_turnover": 7850000000000,
                        "up_count": 2150, 
                        "down_count": 1450,
                        "flat_count": 250,
                        "limit_up_count": 35,
                        "limit_down_count": 8,
                        "turnover_rate": 1.85,
                        "market_sentiment": market_sentiment
                    },
                    "industry_performance": industries_data,
                    "hot_sectors": hot_sectors,
                    "future_hot_sectors": future_hot_sectors
                }
                
                if gui_callback:
                    gui_callback("状态", "使用模拟数据完成市场概览")
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
    def run_combined_strategy(self, momentum_weight, ma_weight, sample_size, industry, min_score, short_ma, long_ma, initial_capital, stop_loss_pct, gui_callback):
        """
        运行组合策略分析
        :param momentum_weight: 动量策略权重 (0-1)
        :param ma_weight: 均线策略权重 (0-1)
        :param sample_size: 分析的股票数量
        :param industry: 行业过滤
        :param min_score: 最低动量得分
        :param short_ma: 短期均线
        :param long_ma: 长期均线
        :param initial_capital: 初始资金
        :param stop_loss_pct: 止损百分比
        :param gui_callback: GUI回调函数
        """
        # 通知GUI分析开始
        gui_callback("状态", "开始运行组合策略分析...")
        # 创建一个线程来执行策略分析
        def run_analysis():
            try:
                logger.info(f"开始组合策略分析，参数: 动量权重={momentum_weight}, 均线权重={ma_weight}, 样本数={sample_size}, 行业={industry}")
                # 获取股票列表
                stock_list = self.get_stock_list(industry)
                # 过滤行业
                if industry != "全部" and 'industry' in stock_list.columns:
                    stock_list = stock_list[stock_list['industry'] == industry]
                # 随机选择股票，如果样本数大于可用股票数
                if sample_size < len(stock_list):
                    stock_list = stock_list.sample(sample_size)
                stock_codes = stock_list['ts_code'].tolist()
                # 通知GUI正在获取股票数据
                gui_callback("状态", f"组合策略分析: 正在获取 {len(stock_codes)} 支股票的数据...")
                # 并行分析多只股票
                results = []
                # 创建临时回调函数来收集结果
                momentum_results = []
                ma_results = []
                # 动量分析回调
                def momentum_callback(status_type, data):
                    if status_type == "结果" and isinstance(data, list):
                        nonlocal momentum_results
                        momentum_results = data
                        gui_callback("状态", f"组合策略: 动量分析完成，继续进行均线分析...")
                # 均线分析回调
                def ma_callback(status_type, data):
                    if status_type == "结果" and isinstance(data, list):
                        nonlocal ma_results
                        ma_results = data
                        gui_callback("状态", f"组合策略: 均线分析完成，开始组合评分...")
                # 运行动量分析
                self.run_momentum_analysis(
                    sample_size=sample_size,
                    industry=industry,
                    min_score=min_score,
                    gui_callback=momentum_callback
                )
                # 运行均线交叉分析
                self.run_ma_cross_strategy(
                    short_ma=short_ma,
                    long_ma=long_ma,
                    initial_capital=initial_capital,
                    stop_loss_pct=stop_loss_pct,
                    sample_size=sample_size,
                    industry=industry,
                    gui_callback=ma_callback
                )
                # 等待两个分析都完成
                wait_count = 0
                max_wait = 120  # 最多等待120秒
                while (not momentum_results or not ma_results) and wait_count < max_wait:
                    import time
                    time.sleep(1)
                    wait_count += 1
                if wait_count >= max_wait:
                    gui_callback("状态", "组合策略分析超时，请重试")
                    return
                # 合并结果
                combined_results = []
                momentum_map = {item['ts_code']: item for item in momentum_results}
                ma_map = {item['ts_code']: item for item in ma_results}
                # 获取所有出现在任一分析中的股票代码
                all_codes = list(set(list(momentum_map.keys()) + list(ma_map.keys())))
                for code in all_codes:
                    # 如果股票在两个分析中都有结果，则合并
                    if code in momentum_map and code in ma_map:
                        m_item = momentum_map[code]
                        ma_item = ma_map[code]
                        # 创建合并项
                        combined_item = {
                            'ts_code': code,
                            'name': m_item.get('name', ma_item.get('name', '')),
                            'industry': m_item.get('industry', ma_item.get('industry', '')),
                            'close': m_item.get('close', ma_item.get('close', 0)),
                            'momentum_score': m_item.get('momentum_score', m_item.get('score', 0)),
                            'ma_score': ma_item.get('score', 0),
                            'ma_signal': ma_item.get('current_signal', '无信号'),
                            'total_return': ma_item.get('total_return', 0),
                            'win_rate': ma_item.get('win_rate', 0)
                        }
                        # 计算组合得分
                        # 动量分数已经是0-100
                        momentum_contribution = momentum_weight * combined_item['momentum_score']
                        # 均线信号转换为分数: 买入=100, 持有=50, 卖出=0
                        ma_signal_score = 0
                        if combined_item['ma_signal'] == '买入':
                            ma_signal_score = 100
                        elif combined_item['ma_signal'] == '持有':
                            ma_signal_score = 50
                        ma_contribution = ma_weight * ma_signal_score
                        # 计算组合得分
                        combined_item['combined_score'] = momentum_contribution + ma_contribution
                        # 添加到结果列表
                        combined_results.append(combined_item)
                # 按组合得分排序
                combined_results = sorted(combined_results, key=lambda x: x['combined_score'], reverse=True)
                # 添加排名
                for i, item in enumerate(combined_results):
                    item['rank'] = i + 1
                # 如果结果不为空，保存到CSV并创建结果图表
                if combined_results:
                    # 保存结果到CSV
                    output_dir = os.path.join('results', 'combined_strategy')
                    os.makedirs(output_dir, exist_ok=True)
                    # 创建CSV文件名，包含时间戳
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    csv_path = os.path.join(output_dir, f'combined_results_{timestamp}.csv')
                    # 转换为DataFrame并保存
                    result_df = pd.DataFrame(combined_results)
                    result_df.to_csv(csv_path, index=False)
                    logger.info(f"已将组合策略结果保存到 {csv_path}")
                    # 通知GUI分析完成
                    gui_callback("状态", f"组合策略分析完成，共有 {len(combined_results)} 支股票的结果")
                    gui_callback("结果", combined_results)
                else:
                    logger.warning("组合策略分析没有产生任何结果")
                    gui_callback("状态", "组合策略分析未产生任何结果")
            except Exception as e:
                logger.error(f"组合策略分析出错: {str(e)}", exc_info=True)
                gui_callback("状态", f"组合策略分析错误: {str(e)}")
        # 启动分析线程
        analysis_thread = threading.Thread(target=run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        return analysis_thread
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

    def _analyze_trend(self, data_df):
        """分析股票或指数的趋势"""
        if data_df is None or len(data_df) < 3:
            return "中性"
            
        try:
            # 计算MA5和MA10
            data_df['ma5'] = data_df['close'].rolling(window=5).mean()
            data_df['ma10'] = data_df['close'].rolling(window=10).mean()
            
            # 获取最新数据
            latest = data_df.iloc[-1]
            
            # 判断价格位置
            price_above_ma5 = latest['close'] > latest['ma5'] if not pd.isna(latest['ma5']) else False
            price_above_ma10 = latest['close'] > latest['ma10'] if not pd.isna(latest['ma10']) else False
            
            # 判断均线方向
            ma5_trend_up = False
            ma10_trend_up = False
            
            if len(data_df) >= 3:
                ma5_values = data_df['ma5'].dropna().tail(3).values
                if len(ma5_values) == 3:
                    ma5_trend_up = ma5_values[2] > ma5_values[0]
                
                ma10_values = data_df['ma10'].dropna().tail(3).values
                if len(ma10_values) == 3:
                    ma10_trend_up = ma10_values[2] > ma10_values[0]
            
            # 综合判断趋势
            if price_above_ma5 and price_above_ma10 and ma5_trend_up and ma10_trend_up:
                return "强势上涨"
            elif price_above_ma5 and ma5_trend_up:
                return "上涨"
            elif not price_above_ma5 and not price_above_ma10 and not ma5_trend_up and not ma10_trend_up:
                return "强势下跌"
            elif not price_above_ma5 and not ma5_trend_up:
                return "下跌"
            else:
                return "震荡"
        except Exception as e:
            logger.error(f"分析趋势失败: {str(e)}")
            return "中性"
            
    def _calculate_industry_strength(self, up_count, down_count, change_pct):
        """计算行业强度指数"""
        if up_count + down_count == 0:
            return 0
            
        # 上涨家数占比
        up_ratio = up_count / (up_count + down_count)
        
        # 综合考虑上涨家数比例和平均涨幅
        strength = (up_ratio * 70) + (change_pct * 3)
        
        return min(100, max(0, strength))
        
    def _analyze_market_sentiment(self, market_data, indices_data):
        """分析市场情绪"""
        try:
            # 提取市场数据
            up_count = market_data.get('up_count', 0)
            down_count = market_data.get('down_count', 0)
            flat_count = market_data.get('flat_count', 0)
            limit_up_count = market_data.get('limit_up_count', 0)
            limit_down_count = market_data.get('limit_down_count', 0)
            
            total_count = up_count + down_count + flat_count
            if total_count == 0:
                return "中性"
                
            # 计算多空比
            bull_bear_ratio = up_count / down_count if down_count > 0 else float('inf')
            
            # 计算涨跌比
            up_down_ratio = up_count / total_count if total_count > 0 else 0
            
            # 分析主要指数表现
            index_changes = [index_data.get('change', 0) for index_data in indices_data]
            avg_index_change = sum(index_changes) / len(index_changes) if len(index_changes) > 0 else 0
            
            # 综合分析市场情绪
            if bull_bear_ratio > 2.5 and up_down_ratio > 0.7 and limit_up_count > 30 and avg_index_change > 1:
                return "极度乐观"
            elif bull_bear_ratio > 1.8 and up_down_ratio > 0.65 and limit_up_count > 20 and avg_index_change > 0.5:
                return "乐观"
            elif bull_bear_ratio > 1.2 and up_down_ratio > 0.55 and avg_index_change > 0:
                return "偏乐观"
            elif bull_bear_ratio < 0.4 and up_down_ratio < 0.3 and limit_down_count > 30 and avg_index_change < -1:
                return "极度悲观"
            elif bull_bear_ratio < 0.6 and up_down_ratio < 0.35 and limit_down_count > 20 and avg_index_change < -0.5:
                return "悲观"
            elif bull_bear_ratio < 0.8 and up_down_ratio < 0.45 and avg_index_change < 0:
                return "偏悲观"
            else:
                return "中性"
        except Exception as e:
            logger.error(f"分析市场情绪失败: {str(e)}")
            return "中性"
            
    def _predict_future_hot_sectors(self, industries_data, market_sentiment):
        """预测未来热门板块"""
        try:
            # 按强度和变化趋势预测未来热门板块
            future_candidates = []
            
            # 为每个行业评分
            scored_industries = []
            for industry in industries_data:
                # 基础分 = 行业强度
                base_score = industry.get('strength_index', 0)
                
                # 资金关注度权重 (这里用上涨股票比例作为代理指标)
                fund_attention = industry.get('up_count', 0) / max(1, industry.get('total_count', 1)) * 100
                
                # 计算最终得分
                final_score = base_score * 0.7 + fund_attention * 0.3
                
                scored_industries.append({
                    'name': industry.get('name', ''),
                    'score': final_score,
                    'strength': industry.get('strength_index', 0),
                    'up_ratio': industry.get('up_count', 0) / max(1, industry.get('total_count', 1)),
                    'change': industry.get('change', 0),
                    'leading_stock': industry.get('leading_up', '')
                })
            
            # 按得分排序
            scored_industries.sort(key=lambda x: x['score'], reverse=True)
            
            # 根据市场情绪调整推荐逻辑
            if "乐观" in market_sentiment:
                # 乐观市场偏向高强度和高涨幅行业
                candidates = scored_industries[:10]
            elif "悲观" in market_sentiment:
                # 悲观市场偏向防御性行业 (这里简化处理，实际应有更复杂的逻辑)
                candidates = [ind for ind in scored_industries if ind['change'] > 0][:10]
            else:
                # 中性市场提供平衡选择
                candidates = scored_industries[:10]
            
            # 生成未来热门板块预测
            future_hot_sectors = []
            for i, candidate in enumerate(candidates[:5]):  # 取前5个
                predicted_change = min(10, max(1, candidate['change'] + 2)) if candidate['change'] > 0 else min(5, max(1, 3))
                
                # 生成推荐理由
                if candidate['score'] > 80:
                    reason = f"强势行业,主力资金持续流入,{candidate['leading_stock']}等龙头表现优异"
                elif candidate['score'] > 60:
                    reason = f"行业基本面向好,整体趋势向上,关注度提升"
                else:
                    reason = f"近期有回暖迹象,可能迎来阶段性机会"
                
                future_hot_sectors.append({
                    'name': candidate['name'],
                    'predicted_change': predicted_change,
                    'attention_index': min(100, max(50, candidate['score'])),
                    'fund_inflow': candidate['up_ratio'] * 20,  # 模拟资金流入数据
                    'growth_score': min(100, max(30, candidate['score'] * 1.1)),
                    'recommendation': reason
                })
            
            return future_hot_sectors
            
        except Exception as e:
            logger.error(f"预测未来热门板块失败: {str(e)}")
            # 返回一个基本的预测
            return [
                {"name": "数字经济", "predicted_change": 3.5, "attention_index": 85, "fund_inflow": 15.45, "growth_score": 88, "recommendation": "政策持续支持,板块景气度高"},
                {"name": "新能源", "predicted_change": 3.2, "attention_index": 82, "fund_inflow": 12.76, "growth_score": 85, "recommendation": "行业处于成长期,中长期向好"},
                {"name": "医疗健康", "predicted_change": 2.8, "attention_index": 80, "fund_inflow": 10.32, "growth_score": 82, "recommendation": "防御性较强,估值处于合理区间"},
                {"name": "先进制造", "predicted_change": 2.5, "attention_index": 75, "fund_inflow": 8.67, "growth_score": 80, "recommendation": "产业升级持续,自主可控推进"},
                {"name": "消费升级", "predicted_change": 2.2, "attention_index": 72, "fund_inflow": 7.54, "growth_score": 78, "recommendation": "消费回暖预期,政策利好频出"}
            ]