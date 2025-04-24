#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票数据分析系统主程序
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# 添加当前目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from src.config import (
    DATA_DIR, STOCK_SELECTION_PARAMS, NOTIFICATION_CONFIG,
    RISK_CONTROL_PARAMS, BACKTEST_PARAMS
)
from src.data.fetcher import StockDataFetcher
from src.data.processor import StockDataProcessor
from src.indicators.technical import TechnicalIndicators
from src.analysis.capital_flow import CapitalFlowAnalyzer
from src.analysis.sentiment import SentimentAnalyzer
from src.utils.logger import setup_logger
from src.risk.risk_management import RiskManager
from src.notification.notifier import NotificationManager
from src.backtest.engine import BacktestEngine
from src.backtest.strategy import MomentumStrategy
from src.strategies.limit_up_capture_strategy import LimitUpCaptureStrategy

# 设置日志
logger = setup_logger("main")

class StockAnalysisSystem:
    """股票涨停和大幅上涨捕捉系统的主类"""
    
    def __init__(self):
        """初始化系统"""
        # 创建数据目录
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # 初始化组件
        self.data_fetcher = StockDataFetcher()
        self.data_processor = StockDataProcessor()
        self.technical_indicators = TechnicalIndicators()
        self.capital_flow_analyzer = CapitalFlowAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # 初始化风险管理器
        self.risk_manager = RiskManager(
            max_position_per_stock=RISK_CONTROL_PARAMS.get("max_position_per_stock", 0.05),
            max_position_per_industry=RISK_CONTROL_PARAMS.get("max_position_per_industry", 0.20),
            max_industry_allocation=RISK_CONTROL_PARAMS.get("max_industry_allocation", 0.30),
            stop_loss=RISK_CONTROL_PARAMS.get("stop_loss", 0.05),
            take_profit=RISK_CONTROL_PARAMS.get("take_profit", 0.15),
            use_trailing_stop=RISK_CONTROL_PARAMS.get("use_trailing_stop", True),
            max_drawdown=RISK_CONTROL_PARAMS.get("max_drawdown", 0.10),
            risk_free_rate=RISK_CONTROL_PARAMS.get("risk_free_rate", 0.03),
        )
        
        # 初始化通知管理器
        self.notification_manager = NotificationManager(NOTIFICATION_CONFIG)
        
        # 股票数据缓存
        self.stock_list = None
        self.industry_list = None
        self.stock_data_cache = {}
        self.industry_data_cache = {}
        self.limit_up_stocks = {}
        
        logger.info("股票分析系统初始化完成")
    
    def prepare_data(self, start_date=None, end_date=None):
        """准备数据
        
        Args:
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        Returns:
            bool: 是否成功
        """
        # 处理日期参数
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # 验证日期格式
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"日期格式错误，请使用YYYY-MM-DD格式。start_date:{start_date}, end_date:{end_date}")
            return False
            
        # 验证起始日期不大于结束日期
        if start_date > end_date:
            logger.error(f"起始日期({start_date})不能大于结束日期({end_date})")
            return False
        
        logger.info(f"准备数据，起始日期: {start_date}, 结束日期: {end_date}")
        
        # 获取股票列表
        self.stock_list = self.data_fetcher.get_stock_list()
        if self.stock_list is not None:
            logger.info(f"获取到 {len(self.stock_list)} 只股票")
        else:
            logger.error("获取股票列表失败")
            return False
        
        # 获取行业列表
        self.industry_list = self.data_fetcher.get_industry_list()
        if self.industry_list is not None:
            logger.info(f"获取到 {len(self.industry_list)} 个行业")
        else:
            logger.warning("获取行业列表失败，将不进行行业分析")
        
        # 获取连续涨停股票并保存
        limit_up_results = {}
        for days in STOCK_SELECTION_PARAMS['continuous_limit_up_days']:
            try:
                limit_up_stocks = self.data_fetcher.get_continuous_limit_up_stocks(days=days, end_date=end_date)
                if limit_up_stocks is not None and not limit_up_stocks.empty:
                    logger.info(f"获取到 {len(limit_up_stocks)} 只连续 {days} 天涨停的股票")
                    # 保存到文件
                    file_path = os.path.join(DATA_DIR, f"continuous_limit_up_{days}d_{end_date}.csv")
                    limit_up_stocks.to_csv(file_path, index=False)
                    # 保存到结果字典
                    limit_up_results[days] = limit_up_stocks
                else:
                    logger.info(f"没有找到连续 {days} 天涨停的股票")
                    limit_up_results[days] = pd.DataFrame()
            except Exception as e:
                logger.error(f"获取连续{days}天涨停股票失败: {str(e)}")
                limit_up_results[days] = pd.DataFrame()
        
        # 保存涨停结果到系统中
        self.limit_up_stocks = limit_up_results
        
        return True
    
    def analyze_stock(self, stock_code, start_date, end_date=None):
        """分析单个股票
        
        Args:
            stock_code: 股票代码
            start_date: 起始日期
            end_date: 结束日期，默认为今天
            
        Returns:
            dict: 分析结果
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"分析股票 {stock_code}, 起始日期: {start_date}, 结束日期: {end_date}")
        
        # 1. 获取日线数据
        daily_data = self.data_fetcher.get_daily_data(stock_code, start_date, end_date)
        if daily_data is None or daily_data.empty:
            logger.warning(f"获取股票 {stock_code} 日线数据失败")
            return None
        
        # 2. 处理数据（清洗、计算收益率等）
        daily_data = self.data_processor.clean_daily_data(daily_data)
        
        # 3. 计算技术指标
        with_indicators = self.technical_indicators.calculate_all_indicators(daily_data)
        
        # 4. 资金流向分析
        with_mfi = self.capital_flow_analyzer.calculate_mfi(with_indicators)
        
        # 5. 获取个股资金流向数据
        fund_flow_data = self.data_fetcher.get_stock_fund_flow(stock_code, start_date, end_date)
        if fund_flow_data is not None and not fund_flow_data.empty:
            fund_flow_analysis = self.capital_flow_analyzer.analyze_fund_flow(fund_flow_data)
            # TODO: 将资金流向分析结果与技术指标结合
        
        # 6. 计算涨停检测
        with_limit_up = self.data_processor.detect_limit_up(with_mfi)
        
        # 缓存股票数据
        self.stock_data_cache[stock_code] = with_limit_up
        
        # 7. 计算综合评分
        scores = self.calculate_stock_score(with_limit_up)
        
        # 8. 整理分析结果
        result = {
            'stock_code': stock_code,
            'stock_name': self.get_stock_name(stock_code),
            'latest_date': with_limit_up['date'].max(),
            'latest_price': with_limit_up.iloc[-1]['close'],
            'price_change': with_limit_up.iloc[-1]['daily_return'] if 'daily_return' in with_limit_up.columns else None,
            'score': scores,
            'signals': self.extract_signals(with_limit_up),
            'data': with_limit_up
        }
        
        return result
    
    def get_stock_name(self, stock_code):
        """获取股票名称
        
        Args:
            stock_code: 股票代码
            
        Returns:
            str: 股票名称
        """
        if self.stock_list is None:
            return stock_code
        
        # 根据实际的股票列表结构调整查询方式
        if 'ts_code' in self.stock_list.columns and 'name' in self.stock_list.columns:
            filtered = self.stock_list[self.stock_list['ts_code'] == stock_code]
            if not filtered.empty:
                return filtered.iloc[0]['name']
        
        if 'symbol' in self.stock_list.columns and 'name' in self.stock_list.columns:
            symbol = stock_code.split('.')[0] if '.' in stock_code else stock_code
            filtered = self.stock_list[self.stock_list['symbol'] == symbol]
            if not filtered.empty:
                return filtered.iloc[0]['name']
        
        return stock_code
    
    def extract_signals(self, stock_data):
        """提取股票的技术信号
        
        Args:
            stock_data: DataFrame，包含股票数据和技术指标
            
        Returns:
            list: 技术信号列表
        """
        if stock_data is None or stock_data.empty:
            return []
        
        signals = []
        latest_data = stock_data.iloc[-1]
        
        # 检查是否是涨停股
        if 'is_limit_up' in latest_data and latest_data['is_limit_up']:
            signals.append({
                'type': 'limit_up',
                'description': '涨停',
                'strength': 'strong'
            })
        
        # 检查连续涨停
        if 'consecutive_limit_up' in latest_data and latest_data['consecutive_limit_up'] > 1:
            signals.append({
                'type': 'consecutive_limit_up',
                'description': f'连续涨停{latest_data["consecutive_limit_up"]}天',
                'strength': 'very_strong'
            })
        
        # 检查均线金叉
        if 'ma_5_cross_above_ma_20' in latest_data and latest_data['ma_5_cross_above_ma_20']:
            signals.append({
                'type': 'golden_cross',
                'description': '5日均线上穿20日均线',
                'strength': 'medium'
            })
        
        # 检查MACD金叉
        if 'macd_golden_cross' in latest_data and latest_data['macd_golden_cross']:
            signals.append({
                'type': 'macd_golden_cross',
                'description': 'MACD金叉',
                'strength': 'medium'
            })
        
        # 检查RSI突破
        if 'rsi_cross_above_50' in latest_data and latest_data['rsi_cross_above_50']:
            signals.append({
                'type': 'rsi_bullish',
                'description': 'RSI突破50，趋势转强',
                'strength': 'medium'
            })
        
        # 检查布林带突破
        if 'bb_break_upper' in latest_data and latest_data['bb_break_upper']:
            signals.append({
                'type': 'bollinger_breakout',
                'description': '股价突破布林带上轨',
                'strength': 'strong'
            })
        
        # 检查成交量放大
        if 'volume_surge' in latest_data and latest_data['volume_surge']:
            signals.append({
                'type': 'volume_surge',
                'description': '成交量显著放大',
                'strength': 'medium'
            })
        
        # 检查价格突破前期高点
        if 'break_high_20d' in latest_data and latest_data['break_high_20d']:
            signals.append({
                'type': 'break_high',
                'description': '突破20日新高',
                'strength': 'strong'
            })
        
        # 检查MFI超卖反转
        if 'mfi_cross_above_oversold' in latest_data and latest_data['mfi_cross_above_oversold']:
            signals.append({
                'type': 'mfi_oversold_reversal',
                'description': 'MFI超卖区域反转',
                'strength': 'medium'
            })
        
        return signals
    
    def calculate_stock_score(self, stock_data):
        """计算股票的综合评分
        
        Args:
            stock_data: DataFrame，包含股票数据和技术指标
            
        Returns:
            float: 综合评分 (0-100)
        """
        if stock_data is None or stock_data.empty or len(stock_data) < 2:
            return 0
        
        latest_data = stock_data.iloc[-1]
        
        # 初始分数
        score = 50
        
        # 1. 涨停和连续涨停加分
        if 'is_limit_up' in latest_data and latest_data['is_limit_up']:
            score += 10
            
            if 'consecutive_limit_up' in latest_data:
                consecutive_days = latest_data['consecutive_limit_up']
                if consecutive_days > 1:
                    score += min(consecutive_days * 5, 20)  # 最多加20分
        
        # 2. 技术指标加分
        
        # 均线多头排列
        if 'ma_bull_alignment' in latest_data and latest_data['ma_bull_alignment']:
            score += 5
        
        # MACD金叉
        if 'macd_golden_cross' in latest_data and latest_data['macd_golden_cross']:
            score += 5
        
        # RSI强势
        if 'rsi' in latest_data:
            rsi_value = latest_data['rsi']
            if 50 <= rsi_value <= 70:  # 强势但不过热
                score += 3
            elif rsi_value > 70:  # 可能过热
                score -= 2
        
        # 布林带突破
        if 'bb_break_upper' in latest_data and latest_data['bb_break_upper']:
            score += 5
        
        # 成交量放大
        if 'volume_ratio_5d' in latest_data and latest_data['volume_ratio_5d'] > 1.5:
            score += 3
        
        # 价格突破前期高点
        if 'break_high_20d' in latest_data and latest_data['break_high_20d']:
            score += 5
        
        # 资金流向
        if 'mfi' in latest_data:
            mfi_value = latest_data['mfi']
            if 50 <= mfi_value <= 80:  # 资金流入但不过热
                score += 5
            elif mfi_value > 80:  # 可能过热
                score -= 2
        
        # 确保分数在0-100范围内
        score = max(0, min(100, score))
        
        return score
    
    def screen_stocks(self, min_score=None):
        """筛选高分股票
        
        Args:
            min_score: 最小评分，默认使用配置文件中的阈值
            
        Returns:
            list: 筛选出的股票列表
        """
        if min_score is None:
            min_score = STOCK_SELECTION_PARAMS['score_threshold']
        
        selected_stocks = []
        
        for stock_code, stock_data in self.stock_data_cache.items():
            score = self.calculate_stock_score(stock_data)
            if score >= min_score:
                selected_stocks.append({
                    'stock_code': stock_code,
                    'stock_name': self.get_stock_name(stock_code),
                    'score': score,
                    'signals': self.extract_signals(stock_data),
                    'latest_price': stock_data.iloc[-1]['close'],
                    'price_change': stock_data.iloc[-1]['daily_return'] if 'daily_return' in stock_data.columns else None
                })
        
        # 按评分降序排序
        selected_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        return selected_stocks
    
    def batch_analyze_stocks(self, stock_codes, start_date, end_date=None, max_workers=10):
        """批量分析多只股票
        
        Args:
            stock_codes: 股票代码列表
            start_date: 起始日期
            end_date: 结束日期，默认为今天
            max_workers: 最大线程数
            
        Returns:
            list: 分析结果列表
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"批量分析 {len(stock_codes)} 只股票")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {
                executor.submit(self.analyze_stock, stock, start_date, end_date): stock 
                for stock in stock_codes
            }
            
            for future in future_to_stock:
                stock = future_to_stock[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"分析股票 {stock} 时出错: {e}")
        
        logger.info(f"成功分析 {len(results)} 只股票")
        
        return results
    
    def monitor_stocks_realtime(self, stock_codes=None, interval=None):
        """实时监控股票价格和交易情况
        
        Args:
            stock_codes: 要监控的股票代码列表，默认为已筛选的高分股票
            interval: 监控间隔（秒），默认使用配置文件中的设置
        """
        if interval is None:
            interval = STOCK_SELECTION_PARAMS['update_frequency']
        
        if stock_codes is None:
            # 使用已筛选的高分股票
            selected_stocks = self.screen_stocks()
            stock_codes = [stock['stock_code'] for stock in selected_stocks]
            
            # 如果没有筛选出股票，则使用连续涨停股票
            if not stock_codes:
                for days in sorted(STOCK_SELECTION_PARAMS['continuous_limit_up_days'], reverse=True):
                    file_path = os.path.join(DATA_DIR, f"continuous_limit_up_{days}d_{datetime.now().strftime('%Y-%m-%d')}.csv")
                    if os.path.exists(file_path):
                        limit_up_stocks = pd.read_csv(file_path)
                        if 'ts_code' in limit_up_stocks.columns:
                            stock_codes = limit_up_stocks['ts_code'].tolist()
                        elif 'code' in limit_up_stocks.columns:
                            stock_codes = limit_up_stocks['code'].tolist()
                        break
        
        if not stock_codes:
            logger.warning("没有要监控的股票")
            return
        
        # 限制监控的股票数量
        max_stocks = STOCK_SELECTION_PARAMS['max_stocks_to_monitor']
        if len(stock_codes) > max_stocks:
            logger.info(f"限制监控股票数量为 {max_stocks}")
            stock_codes = stock_codes[:max_stocks]
        
        logger.info(f"开始实时监控 {len(stock_codes)} 只股票，间隔 {interval} 秒")
        
        try:
            while True:
                logger.info(f"获取 {len(stock_codes)} 只股票的实时数据")
                
                # 这里应该实现获取实时行情数据的逻辑
                # 由于实时行情通常需要特定的API和认证，这里仅作为示例框架
                
                # TODO: 获取实时行情数据
                # TODO: 分析实时数据
                # TODO: 发送实时警报
                
                # 等待下一次更新
                time.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("用户中断，停止监控")
        except Exception as e:
            logger.error(f"监控过程中出错: {e}")
    
    def generate_report(self, results, output_dir=None):
        """生成分析报告
        
        Args:
            results: 分析结果列表
            output_dir: 输出目录，默认为data目录下的reports子目录
            
        Returns:
            str: 报告文件路径
        """
        if output_dir is None:
            output_dir = os.path.join(DATA_DIR, "reports")
            
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"stock_analysis_report_{timestamp}.csv")
        
        # 准备报告数据
        report_data = []
        for result in results:
            if not result:
                continue
                
            report_row = {
                'stock_code': result['stock_code'],
                'stock_name': result['stock_name'],
                'latest_date': result['latest_date'],
                'latest_price': result['latest_price'],
                'price_change': result['price_change'],
                'score': result['score'],
                'signals': ', '.join([signal['description'] for signal in result['signals']])
            }
            report_data.append(report_row)
        
        # 创建DataFrame并保存
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(report_file, index=False)
        
        logger.info(f"分析报告已保存到 {report_file}")
        
        return report_file
    
    def backtest_strategy(self, strategy_type="momentum", start_date=None, end_date=None, 
                          stock_codes=None, initial_capital=None):
        """回测策略
        
        Args:
            strategy_type: 策略类型，默认为"momentum"
            start_date: 回测起始日期，默认为配置中的值
            end_date: 回测结束日期，默认为配置中的值
            stock_codes: 待回测的股票列表，默认为None表示使用筛选出的股票
            initial_capital: 初始资金，默认为配置中的值
            
        Returns:
            dict: 回测结果
        """
        # 使用配置中的回测参数
        if start_date is None:
            start_date = BACKTEST_PARAMS['start_date']
        if end_date is None:
            end_date = BACKTEST_PARAMS['end_date']
        if initial_capital is None:
            initial_capital = BACKTEST_PARAMS['initial_capital']
        
        # 如果未提供股票列表，使用筛选的股票
        if stock_codes is None:
            # 筛选评分高的股票
            high_score_stocks = self.screen_stocks(min_score=STOCK_SELECTION_PARAMS['score_threshold'])
            stock_codes = [stock['stock_code'] for stock in high_score_stocks]
            if not stock_codes:
                logger.warning("没有符合条件的股票可供回测")
                return None
        
        # 获取回测数据
        backtest_data = {}
        for stock_code in stock_codes:
            # 如果缓存中有数据，直接使用
            if stock_code in self.stock_data_cache:
                stock_data = self.stock_data_cache[stock_code]
                # 过滤日期范围
                stock_data = stock_data[(stock_data['date'] >= start_date) & (stock_data['date'] <= end_date)]
            else:
                # 获取并分析股票数据
                result = self.analyze_stock(stock_code, start_date, end_date)
                if result is None:
                    continue
                stock_data = result['data']
            
            # 将日期设为索引
            stock_data_indexed = stock_data.set_index('date')
            # 添加行业信息
            if self.stock_list is not None and 'industry' in self.stock_list.columns:
                filtered = self.stock_list[self.stock_list['ts_code'] == stock_code]
                if not filtered.empty and 'industry' in filtered.iloc[0]:
                    industry = filtered.iloc[0]['industry']
                else:
                    industry = "未知"
            else:
                industry = "未知"
            
            # 添加行业字段
            stock_data_indexed['industry'] = industry
            # 添加股票代码字段
            stock_data_indexed['stock_code'] = stock_code
            
            backtest_data[stock_code] = stock_data_indexed
        
        if not backtest_data:
            logger.error("无法获取回测数据")
            return None
        
        # 创建策略
        if strategy_type == "momentum":
            strategy = MomentumStrategy(
                config=self.config,
                lookback_period=20,
                momentum_threshold=0.05,
                profit_target=RISK_CONTROL_PARAMS["take_profit"]["fixed"],
                stop_loss=RISK_CONTROL_PARAMS["stop_loss"]["fixed"],
                max_positions=5
            )
        else:
            logger.error(f"不支持的策略类型: {strategy_type}")
            return None
        
        # 创建回测引擎
        engine = BacktestEngine(
            data=backtest_data,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            commission=BACKTEST_PARAMS['commission'],
            slippage=BACKTEST_PARAMS['slippage'],
            risk_manager=self.risk_manager
        )
        
        # 设置风险管理器
        engine.set_risk_manager(self.risk_manager)
        
        # 运行回测
        logger.info(f"开始回测策略 {strategy_type}, 起始日期: {start_date}, 结束日期: {end_date}")
        results = engine.run()
        
        # 保存回测结果
        output_dir = os.path.join(DATA_DIR, "backtest_results", strategy_type, datetime.now().strftime("%Y%m%d_%H%M%S"))
        engine.save_results(output_dir)
        
        logger.info(f"回测完成，结果已保存至 {output_dir}")
        logger.info(engine.get_summary())
        
        return {
            'performance': results,
            'output_dir': output_dir
        }
    
    def generate_trading_signals(self, min_score=None):
        """生成交易信号
        
        Args:
            min_score: 最低评分阈值，默认为None表示使用配置中的值
            
        Returns:
            list: 交易信号列表
        """
        if min_score is None:
            min_score = STOCK_SELECTION_PARAMS['score_threshold']
        
        # 筛选高评分股票
        high_score_stocks = self.screen_stocks(min_score=min_score)
        
        # 生成交易信号
        signals = []
        for stock in high_score_stocks[:STOCK_SELECTION_PARAMS['max_positions']]:
            stock_signals = stock['signals']
            
            # 只保留强烈买入信号
            buy_signals = [s for s in stock_signals if s['strength'] in ['strong', 'very_strong']]
            
            if buy_signals:
                # 计算买入数量
                price = stock['latest_price']
                risk_score = 1 - stock['score']['total'] / 100  # 评分越高，风险越低
                
                # 获取行业信息
                industry = "未知"
                if self.stock_list is not None and 'industry' in self.stock_list.columns:
                    filtered = self.stock_list[self.stock_list['ts_code'] == stock['stock_code']]
                    if not filtered.empty and 'industry' in filtered.iloc[0]:
                        industry = filtered.iloc[0]['industry']
                
                # 使用风险管理器计算仓位
                quantity = self.risk_manager.calculate_position_size(
                    stock_code=stock['stock_code'],
                    price=price,
                    risk_score=risk_score
                )
                
                if quantity > 0:
                    signals.append({
                        'stock_code': stock['stock_code'],
                        'stock_name': stock['stock_name'],
                        'action': 'buy',
                        'price': price,
                        'quantity': quantity,
                        'date': stock['latest_date'],
                        'industry': industry,
                        'signals': buy_signals,
                        'score': stock['score']
                    })
        
        logger.info(f"生成 {len(signals)} 个交易信号")
        return signals
    
    def send_trading_signals(self, signals):
        """发送交易信号通知
        
        Args:
            signals: 交易信号列表
        """
        if not signals:
            logger.info("没有需要发送的交易信号")
            return
        
        # 生成通知内容
        for signal in signals:
            stock_code = signal['stock_code']
            stock_name = signal['stock_name']
            action = signal['action']
            price = signal['price']
            
            # 涨跌幅
            if 'price_change' in signal and signal['price_change'] is not None:
                change_pct = signal['price_change']
            else:
                change_pct = 0
            
            # 信号描述
            signal_desc = ', '.join([s['description'] for s in signal['signals']])
            
            # 发送通知
            self.notification_manager.send_stock_alert(
                stock_code=f"{stock_code} {stock_name}",
                alert_type="买入信号",
                price=price,
                change_pct=change_pct,
                message=signal_desc
            )
            
            logger.info(f"已发送 {stock_code} {stock_name} 的买入信号通知")
    
    def run(self, start_date=None, end_date=None):
        """运行系统
        
        Args:
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        Returns:
            dict: 系统运行结果
        """
        try:
            # 1. 准备数据
            success = self.prepare_data(start_date, end_date)
            if not success:
                logger.error("数据准备失败，系统退出")
                return {
                    "status": "error",
                    "message": "数据准备失败"
                }
            
            # 2. 分析股票
            logger.info("开始分析筛选出的股票")
            analysis_results = []
            
            # 分析连续涨停的股票
            for days, limit_up_df in self.limit_up_stocks.items():
                if limit_up_df is not None and not limit_up_df.empty:
                    logger.info(f"分析 {len(limit_up_df)} 只连续{days}天涨停的股票")
                    
                    for _, row in limit_up_df.iterrows():
                        stock_code = row['ts_code']
                        stock_name = row['name']
                        
                        # 分析股票
                        result = self.analyze_stock(stock_code, start_date, end_date)
                        if result:
                            analysis_results.append(result)
                else:
                    logger.info(f"没有连续{days}天涨停的股票，跳过分析")
            
            # 3. 生成交易信号
            trading_signals = []
            
            # 筛选评分高的股票
            high_score_stocks = [result for result in analysis_results if result['score'] >= STOCK_SELECTION_PARAMS['score_threshold']]
            logger.info(f"筛选出 {len(high_score_stocks)} 只高评分股票")
            
            # 按照评分排序
            if high_score_stocks:
                high_score_stocks.sort(key=lambda x: x['score'], reverse=True)
                
                # 生成交易信号
                for stock in high_score_stocks[:STOCK_SELECTION_PARAMS['max_positions']]:
                    # 直接构建交易信号
                    signal = {
                        'stock_code': stock['stock_code'],
                        'stock_name': stock['stock_name'],
                        'action': 'buy',
                        'price': stock['latest_price'],
                        'signals': stock['signals'],
                        'score': stock['score']
                    }
                    trading_signals.append(signal)
            
            # 4. 发送通知
            if trading_signals:
                logger.info(f"生成了 {len(trading_signals)} 个交易信号")
                for signal in trading_signals:
                    logger.info(f"交易信号: {signal['stock_code']} {signal['stock_name']} - 评分: {signal['score']}")
            else:
                logger.info("没有生成任何交易信号")
            
            # 5. 生成分析报告
            if analysis_results:
                self.generate_report(analysis_results)
            else:
                logger.info("没有分析结果，不生成报告")
            
            logger.info("系统运行完成，状态: success")
            logger.info(f"分析了 {len(analysis_results)} 只股票，筛选出 {len(high_score_stocks)} 只高评分股票")
            logger.info(f"生成了 {len(trading_signals)} 个交易信号")
            
            return {
                "status": "success",
                "analysis_count": len(analysis_results),
                "high_score_count": len(high_score_stocks) if 'high_score_stocks' in locals() else 0,
                "signal_count": len(trading_signals),
                "signals": trading_signals
            }
        
        except Exception as e:
            logger.error(f"系统运行错误: {str(e)}")
            logger.exception("详细错误信息:")
            return {
                "status": "error",
                "message": str(e)
            }

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="股票涨停和大幅上涨捕捉系统")
    parser.add_argument('--all', action='store_true', help='分析所有股票')
    parser.add_argument('--start-date', type=str, help='起始日期，格式为YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, help='结束日期，格式为YYYY-MM-DD')
    parser.add_argument('--no-report', action='store_true', help='不输出报告')
    parser.add_argument('--backtest', action='store_true', help='运行策略回测')
    
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 创建系统实例
    system = StockAnalysisSystem()
    
    # 运行系统
    result = system.run(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # 输出运行结果
    logger.info(f"系统运行完成，状态: {result['status']}")
    if result['status'] == 'success':
        logger.info(f"分析了 {result['analysis_count']} 只股票，筛选出 {result['high_score_count']} 只高评分股票")
        logger.info(f"生成了 {result['signal_count']} 个交易信号") 