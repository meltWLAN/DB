"""
报告生成器模块
用于自动生成股票分析和回测报告
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import logging
import os

from .templates import StockAnalysisReport, BacktestReport
from .charts import StockChartManager
from ..config.settings import RESULTS_DIR, GUI_CONFIG

logger = logging.getLogger(__name__)

class ReportGenerator:
    """报告生成器类，负责生成各种类型的报告"""
    
    def __init__(self, output_dir: Optional[Path] = None, theme_config: Optional[Dict[str, Any]] = None):
        """
        初始化报告生成器
        
        Args:
            output_dir: 报告输出目录
            theme_config: 主题配置
        """
        self.output_dir = output_dir if output_dir is not None else RESULTS_DIR / "reports"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化图表管理器
        self.chart_manager = StockChartManager(theme_config)
        
        logger.info(f"报告生成器初始化，输出目录: {self.output_dir}")
        
    def generate_stock_analysis_report(self, stock_data: pd.DataFrame, 
                                     indicators_data: Dict[str, Any],
                                     stock_info: Dict[str, Any],
                                     recommendations: List[Dict[str, Any]],
                                     summary: str = "",
                                     title: Optional[str] = None,
                                     format: str = "html") -> str:
        """
        生成股票分析报告
        
        Args:
            stock_data: 股票价格数据DataFrame，必须包含open, high, low, close, volume列
            indicators_data: 技术指标数据字典
            stock_info: 股票信息字典，必须包含code, name, industry, market等字段
            recommendations: 投资建议列表
            summary: 分析总结
            title: 报告标题，如果为None则自动生成
            format: 报告格式，支持"html", "pdf", "json"
            
        Returns:
            str: 生成的报告文件路径
        """
        # 检查必要的数据
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in stock_data.columns for col in required_cols):
            raise ValueError(f"股票数据必须包含以下列: {', '.join(required_cols)}")
        
        required_info = ['code', 'name', 'industry', 'market']
        if not all(field in stock_info for field in required_info):
            raise ValueError(f"股票信息必须包含以下字段: {', '.join(required_info)}")
            
        # 自动生成标题（如果未提供）
        if title is None:
            title = f"{stock_info['name']}({stock_info['code']}) - 股票分析报告"
            
        # 处理技术指标数据
        indicators = self._process_indicators(indicators_data)
        
        # 准备报告数据
        report_data = {
            "stock_info": stock_info,
            "price_data": stock_data,
            "indicators": indicators,
            "recommendations": recommendations,
            "summary": summary
        }
        
        # 创建报告对象并生成报告
        report = StockAnalysisReport(title=title, output_dir=self.output_dir)
        report_path = report.generate_report(report_data, format=format)
        
        logger.info(f"已生成股票分析报告: {report_path}")
        return report_path
        
    def generate_backtest_report(self, strategy_info: Dict[str, Any],
                               performance_data: Dict[str, Any],
                               trades_data: Union[pd.DataFrame, List[Dict[str, Any]]],
                               title: Optional[str] = None,
                               format: str = "html") -> str:
        """
        生成回测报告
        
        Args:
            strategy_info: 策略信息字典，必须包含name, period, initial_capital, final_capital等字段
            performance_data: 性能数据字典，必须包含关键性能指标
            trades_data: 交易记录数据
            title: 报告标题，如果为None则自动生成
            format: 报告格式，支持"html", "pdf", "json"
            
        Returns:
            str: 生成的报告文件路径
        """
        # 检查必要的数据
        required_info = ['name', 'period', 'initial_capital', 'final_capital']
        if not all(field in strategy_info for field in required_info):
            raise ValueError(f"策略信息必须包含以下字段: {', '.join(required_info)}")
            
        required_performance = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        if not all(field in performance_data for field in required_performance):
            raise ValueError(f"性能数据必须包含以下字段: {', '.join(required_performance)}")
            
        # 自动生成标题（如果未提供）
        if title is None:
            title = f"{strategy_info['name']} - 策略回测报告"
            
        # 准备报告数据
        report_data = {
            "strategy_info": strategy_info,
            "performance": performance_data,
            "trades": trades_data
        }
        
        # 创建报告对象并生成报告
        report = BacktestReport(title=title, output_dir=self.output_dir)
        report_path = report.generate_report(report_data, format=format)
        
        logger.info(f"已生成回测报告: {report_path}")
        return report_path
        
    def generate_multi_stock_comparison_report(self, stocks_data: Dict[str, pd.DataFrame],
                                            stocks_info: Dict[str, Dict[str, Any]],
                                            comparison_period: str,
                                            title: str = "多股票对比分析报告",
                                            format: str = "html") -> str:
        """
        生成多股票对比分析报告
        
        Args:
            stocks_data: 多个股票的价格数据，键为股票代码，值为价格DataFrame
            stocks_info: 多个股票的信息，键为股票代码，值为股票信息字典
            comparison_period: 对比周期描述
            title: 报告标题
            format: 报告格式，支持"html", "pdf", "json"
            
        Returns:
            str: 生成的报告文件路径
        """
        # 实现多股票对比报告的生成逻辑
        # TODO: 实现此功能，可能需要创建新的报告模板
        logger.warning("多股票对比分析报告功能尚未实现")
        return ""
        
    def generate_portfolio_report(self, portfolio_data: Dict[str, Any],
                               stocks_data: Dict[str, pd.DataFrame],
                               performance_data: Dict[str, Any],
                               title: str = "投资组合报告",
                               format: str = "html") -> str:
        """
        生成投资组合报告
        
        Args:
            portfolio_data: 投资组合数据
            stocks_data: 组合中各股票的价格数据
            performance_data: 组合的性能数据
            title: 报告标题
            format: 报告格式，支持"html", "pdf", "json"
            
        Returns:
            str: 生成的报告文件路径
        """
        # 实现投资组合报告的生成逻辑
        # TODO: 实现此功能，可能需要创建新的报告模板
        logger.warning("投资组合报告功能尚未实现")
        return ""
        
    def batch_generate_reports(self, batch_config: Dict[str, Any]) -> List[str]:
        """
        批量生成报告
        
        Args:
            batch_config: 批量生成报告的配置
            
        Returns:
            List[str]: 所有生成的报告文件路径列表
        """
        report_paths = []
        
        # 实现批量报告生成的逻辑
        # TODO: 实现此功能
        logger.warning("批量生成报告功能尚未实现")
        return report_paths
        
    def _process_indicators(self, indicators_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理技术指标数据为报告格式
        
        Args:
            indicators_data: 技术指标数据字典
            
        Returns:
            List[Dict[str, Any]]: 格式化后的技术指标列表
        """
        indicators = []
        
        # 处理 RSI
        if 'rsi' in indicators_data:
            rsi_value = indicators_data['rsi']
            if isinstance(rsi_value, pd.Series):
                rsi_value = rsi_value.iloc[-1]
                
            signal = '买入' if rsi_value < 30 else '卖出' if rsi_value > 70 else '中性'
            indicators.append({
                'name': 'RSI(相对强弱指标)',
                'value': f"{rsi_value:.2f}",
                'signal': signal,
                'description': 'RSI低于30视为超卖，高于70视为超买'
            })
            
        # 处理 MACD
        if all(k in indicators_data for k in ['macd', 'signal', 'macd_hist']):
            macd = indicators_data['macd'].iloc[-1] if isinstance(indicators_data['macd'], pd.Series) else indicators_data['macd']
            signal = indicators_data['signal'].iloc[-1] if isinstance(indicators_data['signal'], pd.Series) else indicators_data['signal']
            hist = indicators_data['macd_hist'].iloc[-1] if isinstance(indicators_data['macd_hist'], pd.Series) else indicators_data['macd_hist']
            
            macd_signal = '买入' if hist > 0 and macd > signal else '卖出' if hist < 0 and macd < signal else '中性'
            indicators.append({
                'name': 'MACD(平滑异同移动平均线)',
                'value': f"MACD: {macd:.4f}, Signal: {signal:.4f}, Hist: {hist:.4f}",
                'signal': macd_signal,
                'description': '当MACD穿过信号线向上时为买入信号，向下时为卖出信号'
            })
            
        # 处理 KDJ
        if all(k in indicators_data for k in ['k', 'd']):
            k = indicators_data['k'].iloc[-1] if isinstance(indicators_data['k'], pd.Series) else indicators_data['k']
            d = indicators_data['d'].iloc[-1] if isinstance(indicators_data['d'], pd.Series) else indicators_data['d']
            
            kdj_signal = '买入' if k < 20 or (k > d and k < 50) else '卖出' if k > 80 or (k < d and k > 50) else '中性'
            indicators.append({
                'name': 'KDJ(随机指标)',
                'value': f"K: {k:.2f}, D: {d:.2f}",
                'signal': kdj_signal,
                'description': 'K值低于20为超卖，高于80为超买；K线穿过D线向上为买入信号，向下为卖出信号'
            })
            
        # 处理布林带
        if all(k in indicators_data for k in ['bb_upper', 'bb_middle', 'bb_lower']):
            close = indicators_data.get('close', None)
            if close is not None:
                if isinstance(close, pd.Series):
                    close = close.iloc[-1]
                    
                upper = indicators_data['bb_upper'].iloc[-1] if isinstance(indicators_data['bb_upper'], pd.Series) else indicators_data['bb_upper']
                middle = indicators_data['bb_middle'].iloc[-1] if isinstance(indicators_data['bb_middle'], pd.Series) else indicators_data['bb_middle']
                lower = indicators_data['bb_lower'].iloc[-1] if isinstance(indicators_data['bb_lower'], pd.Series) else indicators_data['bb_lower']
                
                bollinger_signal = '买入' if close < lower else '卖出' if close > upper else '中性'
                indicators.append({
                    'name': '布林带(Bollinger Bands)',
                    'value': f"上轨: {upper:.2f}, 中轨: {middle:.2f}, 下轨: {lower:.2f}, 当前: {close:.2f}",
                    'signal': bollinger_signal,
                    'description': '价格触及下轨为买入信号，触及上轨为卖出信号'
                })
                
        # 处理移动平均线
        if all(k in indicators_data for k in ['ma5', 'ma10', 'ma20']):
            ma5 = indicators_data['ma5'].iloc[-1] if isinstance(indicators_data['ma5'], pd.Series) else indicators_data['ma5']
            ma10 = indicators_data['ma10'].iloc[-1] if isinstance(indicators_data['ma10'], pd.Series) else indicators_data['ma10']
            ma20 = indicators_data['ma20'].iloc[-1] if isinstance(indicators_data['ma20'], pd.Series) else indicators_data['ma20']
            
            ma_signal = '买入' if ma5 > ma10 > ma20 else '卖出' if ma5 < ma10 < ma20 else '中性'
            indicators.append({
                'name': '移动平均线(MA)',
                'value': f"MA5: {ma5:.2f}, MA10: {ma10:.2f}, MA20: {ma20:.2f}",
                'signal': ma_signal,
                'description': '短期均线上穿长期均线为金叉买入信号，下穿为死叉卖出信号'
            })
            
        # 可以添加更多技术指标的处理...
            
        return indicators
        
    def _save_chart_to_file(self, chart_data, chart_type: str, filename: str) -> str:
        """
        将图表保存为文件
        
        Args:
            chart_data: 图表数据
            chart_type: 图表类型
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        # 根据图表类型调用不同的绘图方法
        # 此处需要根据实际的图表管理器接口进行适配
        # TODO: 实现此功能
        return "" 