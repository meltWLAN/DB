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
        self.stock_list = None
        self.momentum_results = None
        self.ma_results = None
        self.combined_results = None
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
                stock_list = self.data_manager.get_stock_list()
                # 过滤行业
                if industry != "全部":
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