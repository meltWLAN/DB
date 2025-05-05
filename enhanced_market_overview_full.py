"""
增强版市场概览模块
提供多维度的市场分析、预测和可视化功能
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from pathlib import Path
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置日志
logger = logging.getLogger(__name__)

# 尝试导入数据源管理器
try:
    from src.enhanced.data.fetchers.data_source_manager import DataSourceManager
except ImportError:
    logger.error("无法导入DataSourceManager，市场概览模块将无法正常工作")


class EnhancedMarketOverview:
    """增强版市场概览类"""

    def __init__(self, cache_dir: str = 'cache/market_overview'):
        """初始化市场概览类

        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # 初始化数据源管理器
        try:
            self.data_manager = DataSourceManager()
        except Exception as e:
            logger.error(f"初始化数据源管理器失败: {str(e)}")
            self.data_manager = None

        # 初始化市场预测模型
        self.prediction_models = {}
        self._init_prediction_models()

    def _init_prediction_models(self):
        """初始化预测模型"""
        # 这里将根据需要加载预训练模型
        model_path = os.path.join('models', 'market_sentiment_model.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.prediction_models['sentiment'] = pickle.load(f)
                logger.info("成功加载市场情绪预测模型")
            except Exception as e:
                logger.error(f"加载市场情绪预测模型失败: {str(e)}")

    def get_market_overview(self, trade_date: str = None) -> Dict[str, Any]:
        """获取综合市场概览

        Args:
            trade_date: 交易日期，默认为最新交易日

        Returns:
            Dict: 市场概览数据
        """
        if self.data_manager is None:
            logger.error("数据源管理器未初始化，无法获取市场概览")
            return {}

        # 获取最新交易日期
        if trade_date is None:
            # 尝试获取最新交易日期
            trade_date = self.data_manager.get_latest_trading_date()
            # 如果获取最新交易日期失败，使用当前日期
            if trade_date is None:
                logger.warning("无法获取最新交易日期，使用当前日期")
                from datetime import datetime
                trade_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"开始获取 {trade_date} 的市场概览数据")

        # 获取基础市场数据
        base_market_data = self._get_base_market_data(trade_date)
        if base_market_data is None or (isinstance(base_market_data, dict) and len(base_market_data) == 0) or (isinstance(base_market_data, pd.DataFrame) and base_market_data.empty):
            logger.error(f"无法获取 {trade_date} 的基础市场数据")
            base_market_data = {}

        # 获取指数数据
        indices_data = self._get_indices_data(trade_date)

        # 获取行业数据
        industry_data = self._get_industry_data(trade_date)

        # 获取宏观经济数据
        macro_data = self._get_macro_economic_data(trade_date)

        # 获取资金流向数据
        money_flow_data = self._get_money_flow_data(trade_date)

        # 分析市场情绪
        market_sentiment = self._analyze_market_sentiment(
            base_market_data, indices_data, money_flow_data)

        # 预测未来热门板块
        future_hot_sectors = self._predict_future_hot_sectors(
            industry_data, market_sentiment)

        # 合并所有数据
        overview_data = {
            "date": trade_date,
            "market_base": base_market_data,
            "indices": indices_data,
            "industry": industry_data,
            "macro_economic": macro_data,
            "money_flow": money_flow_data,
            "market_sentiment": market_sentiment,
            "future_hot_sectors": future_hot_sectors
        }

        # Check if we have meaningful data
        has_data = (len(indices_data) > 0 or
                    (isinstance(base_market_data, dict) and len(base_market_data) > 0) or
                    len(industry_data) > 0)

        # Log based on data availability
        if not has_data:
            logger.warning(f"未能获取到 {trade_date} 的有效市场数据")
        else:
            logger.info(f"成功获取 {trade_date} 的市场概览数据")

        return overview_data

    def _get_base_market_data(self, trade_date: str) -> Dict[str, Any]:
        """获取基础市场数据

        Args:
            trade_date: 交易日期

        Returns:
            Dict: 基础市场数据
        """
        try:
            # 直接从数据源管理器获取市场概览
            base_data = self.data_manager.get_market_overview(trade_date)
            if base_data is None or (isinstance(base_data, dict) and len(base_data) == 0):
                logger.warning(f"无法通过数据源管理器获取 {trade_date} 的市场概览，尝试手动计算")
                base_data = self._calculate_market_base_data(trade_date)

            return base_data
        except Exception as e:
            logger.error(f"获取基础市场数据失败: {str(e)}")
            return {}

    def _calculate_market_base_data(self, trade_date: str) -> Dict[str, Any]:
        """手动计算基础市场数据

        Args:
            trade_date: 交易日期

        Returns:
            Dict: 计算的基础市场数据
        """
        try:
            # 获取所有股票当日行情
            all_stocks = self.data_manager.get_all_stock_data_on_date(
                trade_date)
            if all_stocks is None or (isinstance(all_stocks, pd.DataFrame) and all_stocks.empty):
                logger.error(f"无法获取 {trade_date} 的所有股票行情数据")
                return {}

            # 计算涨跌家数
            up_count = len(all_stocks[all_stocks['pct_chg'] > 0])
            down_count = len(all_stocks[all_stocks['pct_chg'] < 0])
            flat_count = len(all_stocks) - up_count - down_count

            # 计算涨停跌停数量
            limit_up_count = len(all_stocks[all_stocks['pct_chg'] > 9.5])
            limit_down_count = len(all_stocks[all_stocks['pct_chg'] < -9.5])

            # 计算成交量和成交额
            total_volume = all_stocks['vol'].sum(
            ) if 'vol' in all_stocks.columns else 0
            total_amount = all_stocks['amount'].sum(
            ) if 'amount' in all_stocks.columns else 0

            # 计算平均涨跌幅
            avg_change = all_stocks['pct_chg'].mean(
            ) if 'pct_chg' in all_stocks.columns else 0

            return {
                'date': trade_date,
                'up_count': up_count,
                'down_count': down_count,
                'flat_count': flat_count,
                'total_count': len(all_stocks),
                'limit_up_count': limit_up_count,
                'limit_down_count': limit_down_count,
                'total_volume': total_volume,
                'total_amount': total_amount,
                'avg_change_pct': avg_change,
                'turnover_rate': total_volume / total_amount * 100 if total_amount > 0 else 0
            }
        except Exception as e:
            logger.error(f"计算基础市场数据失败: {str(e)}")
            return {}

    def _get_indices_data(self, trade_date: str) -> List[Dict[str, Any]]:
        """获取主要指数数据

        Args:
            trade_date: 交易日期

        Returns:
            List[Dict]: 指数数据列表
        """
        try:
            # 主要指数代码和名称
            indices = [
                {'code': '000001.SH', 'name': '上证指数', 'type': '综合'},
                {'code': '399001.SZ', 'name': '深证成指', 'type': '综合'},
                {'code': '399006.SZ', 'name': '创业板指', 'type': '综合'},
                {'code': '000300.SH', 'name': '沪深300', 'type': '综合'},
                {'code': '000016.SH', 'name': '上证50', 'type': '蓝筹'},
                {'code': '000905.SH', 'name': '中证500', 'type': '中小'},
                {'code': '000852.SH', 'name': '中证1000', 'type': '小盘'},
                {'code': '000688.SH', 'name': '科创50', 'type': '科技'},
                {'code': '000922.SH', 'name': '中证红利', 'type': '红利'},
                {'code': '399550.SZ', 'name': '央企创新', 'type': '国企'}
            ]

            # 获取最近N个交易日，用于计算趋势和短期表现
            days_back = 20
            start_date = self.data_manager.get_previous_trading_date(
                trade_date, days_back)
            if not start_date:
                # 如果无法获取前一交易日，使用简单的日期减法
                start_date = (datetime.strptime(trade_date, '%Y-%m-%d') -
                              timedelta(days=days_back)).strftime('%Y-%m-%d')

            results = []

            # 获取指数数据
            for idx in indices:
                try:
                    index_data = self._get_single_index_data(
                        idx['code'], idx['name'], idx['type'], start_date, trade_date
                    )
                    if index_data:
                        results.append(index_data)
                except Exception as e:
                    logger.error(f"获取 {idx['name']} 指数数据失败: {str(e)}")

            # 按类型和名称排序
            results = sorted(results, key=lambda x: (
                x.get('type', ''), x.get('name', '')))

            logger.info(f"成功获取 {len(results)} 个指数的数据")
            return results

        except Exception as e:
            logger.error(f"获取指数数据失败: {str(e)}")
            return []

    def _get_single_index_data(self, code: str, name: str, index_type: str,
                               start_date: str, end_date: str) -> Dict[str, Any]:
        """获取单个指数的数据

        Args:
            code: 指数代码
            name: 指数名称
            index_type: 指数类型
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict: 指数数据
        """
        try:
            # 获取指数日线数据
            df = self.data_manager.get_stock_index_data(
                code, start_date, end_date)

            # 检查数据有效性 - 使用明确的空值检查
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                logger.warning(f"无法获取指数 {name}({code}) 的数据，返回基本信息")
                return {
                    'code': code,
                    'name': name,
                    'type': index_type,
                    'close': 0,
                    'change': 0,
                    'change_5d': 0,
                    'change_20d': 0,
                    'volume': 0,
                    'amount': 0,
                    'volume_ratio': 1,
                    'turnover_rate': 0,
                    'pe': 0,
                    'trend': '未知',
                    'strength': 0
                }

            # 添加技术指标
            df = self._calculate_technical_indicators(df)

            # 获取最近一个交易日的数据
            latest_data = df.iloc[-1] if len(df) > 0 else None
            if latest_data is None:
                logger.warning(f"指数 {name}({code}) 数据为空，返回基本信息")
                return {
                    'code': code,
                    'name': name,
                    'type': index_type,
                    'close': 0,
                    'change': 0,
                    'change_5d': 0,
                    'change_20d': 0,
                    'volume': 0,
                    'amount': 0,
                    'volume_ratio': 1,
                    'turnover_rate': 0,
                    'pe': 0,
                    'trend': '未知',
                    'strength': 0
                }

            # 计算近期表现
            change_5d = self._calculate_period_change(df, 5)
            change_20d = self._calculate_period_change(df, 20)

            # 分析趋势和强度
            trend, strength = self._analyze_index_trend(df)

            # 计算量比（当日成交量/5日平均成交量）
            volume_ratio = 1.0
            if 'volume' in df.columns and len(df) >= 5:
                avg_vol_5d = df['volume'].iloc[-5:].mean() if len(df) >= 5 else df['volume'].mean()
                volume_ratio = latest_data['volume'] / avg_vol_5d if avg_vol_5d > 0 else 1.0

            # 构建结果
            result = {
                'code': code,
                'name': name,
                'type': index_type,
                'close': float(latest_data['close']),
                'change': float(latest_data['pct_chg']) if 'pct_chg' in latest_data else 0,
                'change_5d': float(change_5d),
                'change_20d': float(change_20d),
                'volume': float(latest_data['volume']) if 'volume' in latest_data else 0,
                'amount': float(latest_data['amount']) if 'amount' in latest_data else 0,
                'volume_ratio': float(volume_ratio),
                'turnover_rate': float(latest_data['turnover_rate']) if 'turnover_rate' in latest_data else 0,
                'pe': float(latest_data['pe']) if 'pe' in latest_data else 0,
                'trend': str(trend),
                'strength': float(strength)
            }

            # 确保所有字段都是基本类型，而不是numpy类型
            for key, value in result.items():
                if hasattr(value, 'item'):
                    result[key] = value.item()

            return result
        except Exception as e:
            logger.error(f"获取指数 {name}({code}) 数据失败: {str(e)}")
            return {
                'code': code,
                'name': name,
                'type': index_type,
                'close': 0,
                'change': 0,
                'change_5d': 0,
                'change_20d': 0,
                'volume': 0,
                'amount': 0,
                'volume_ratio': 1,
                'turnover_rate': 0,
                'pe': 0,
                'trend': '未知',
                'strength': 0
            }

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 添加了技术指标的数据框
        """
        try:
            # 确保数据框有足够的行
            if len(df) < 30:
                logger.warning(f"数据行数不足，无法计算某些技术指标")
                return df

            # 计算RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # 计算MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # 计算布林带
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['std20'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['ma20'] + (df['std20'] * 2)
            df['lower_band'] = df['ma20'] - (df['std20'] * 2)

            return df

        except Exception as e:
            logger.error(f"计算技术指标失败: {str(e)}")
            return df

    def _calculate_period_change(self, df: pd.DataFrame, days: int) -> float:
        """计算指定周期的涨跌幅

        Args:
            df: 数据框
            days: 天数

        Returns:
            float: 涨跌幅(%)
        """
        try:
            if len(df) <= days:
                return 0.0

            return ((df['close'].iloc[-1] / df['close'].iloc[-days-1]) - 1) * 100
        except Exception as e:
            logger.error(f"计算周期涨跌幅失败: {str(e)}")
            return 0.0

    def _analyze_index_trend(self, df: pd.DataFrame) -> Tuple[str, float]:
        """分析指数趋势

        Args:
            df: 数据框

        Returns:
            Tuple[str, float]: 趋势和强度
        """
        try:
            # 确保数据框有足够的行
            if len(df) < 20:
                return "未知", 0.0

            # 计算短中期均线
            ma5 = df['close'].rolling(window=5).mean()
            ma10 = df['close'].rolling(window=10).mean()
            ma20 = df['close'].rolling(window=20).mean()

            # 获取最新值
            latest_ma5 = ma5.iloc[-1]
            latest_ma10 = ma10.iloc[-1]
            latest_ma20 = ma20.iloc[-1]
            latest_close = df['close'].iloc[-1]

            # 判断多头排列、空头排列
            bull_alignment = latest_ma5 > latest_ma10 > latest_ma20 and latest_close > latest_ma5
            bear_alignment = latest_ma5 < latest_ma10 < latest_ma20 and latest_close < latest_ma5

            # 计算20日线的斜率(简单线性回归)
            ma20_recent = ma20.tail(10).values
            x = np.arange(len(ma20_recent))
            slope, _ = np.polyfit(x, ma20_recent, 1)

            # 计算趋势强度 (0-100)
            # 考虑均线排列、斜率、最近N天涨跌幅、MACD柱状图等
            strength_factors = []

            # 均线排列因子
            if bull_alignment:
                strength_factors.append(80)
            elif bear_alignment:
                strength_factors.append(20)
            else:
                # 部分均线多头或空头排列
                if latest_ma5 > latest_ma10:
                    strength_factors.append(60)
                elif latest_ma5 < latest_ma10:
                    strength_factors.append(40)
                else:
                    strength_factors.append(50)

            # 斜率因子 (归一化到0-100)
            normalized_slope = min(max((slope * 1000 + 50), 0), 100)
            strength_factors.append(normalized_slope)

            # 最近涨跌幅因子
            change_5d = self._calculate_period_change(df, 5)
            change_factor = min(max((change_5d + 10) * 5, 0), 100)
            strength_factors.append(change_factor)

            # MACD柱状图因子
            if 'macd_hist' in df.columns:
                recent_hist = df['macd_hist'].tail(3).mean()
                macd_factor = min(max((recent_hist * 50 + 50), 0), 100)
                strength_factors.append(macd_factor)

            # 计算平均强度
            strength = np.mean(strength_factors)

            # 确定趋势类型
            if strength >= 70:
                trend = "强势上涨"
            elif strength >= 60:
                trend = "上涨"
            elif strength >= 45:
                trend = "震荡偏多"
            elif strength >= 40:
                trend = "震荡"
            elif strength >= 30:
                trend = "震荡偏空"
            elif strength >= 20:
                trend = "下跌"
            else:
                trend = "强势下跌"

            return trend, strength

        except Exception as e:
            logger.error(f"分析指数趋势失败: {str(e)}")
            return "未知", 50.0

    def _calculate_index_score(self, index_data: Dict[str, Any]) -> float:
        """计算指数综合得分

        Args:
            index_data: 指数数据

        Returns:
            float: 综合得分(0-100)
        """
        try:
            score_factors = []

            # 短期涨跌幅因子 (5日)
            change_5d = index_data.get('change_5d', 0)
            change_5d_score = min(max((change_5d + 5) * 10, 0), 100)
            score_factors.append(change_5d_score * 0.2)  # 权重0.2

            # 中期涨跌幅因子 (20日)
            change_20d = index_data.get('change_20d', 0)
            change_20d_score = min(max((change_20d + 10) * 5, 0), 100)
            score_factors.append(change_20d_score * 0.15)  # 权重0.15

            # 趋势强度因子
            trend_strength = index_data.get('strength', 50)
            score_factors.append(trend_strength * 0.25)  # 权重0.25

            # RSI因子 (RSI在30-70之间为健康，过高或过低都不是好信号)
            rsi = index_data.get('rsi', 50)
            if rsi < 30:
                rsi_score = rsi * 100 / 30  # 0-30映射到0-100
            elif rsi > 70:
                rsi_score = (100 - rsi) * 100 / 30  # 70-100映射到100-0
            else:
                rsi_score = 100  # 30-70区间给满分
            score_factors.append(rsi_score * 0.15)  # 权重0.15

            # MACD柱状图因子
            macd_hist = index_data.get('macd_hist', 0)
            macd_score = min(max((macd_hist * 50 + 50), 0), 100)
            score_factors.append(macd_score * 0.15)  # 权重0.15

            # 成交量因子
            volume_ratio = index_data.get('volume_ratio', 1.0)
            volume_score = min(max((volume_ratio - 0.5) * 100, 0), 100)
            score_factors.append(volume_score * 0.1)  # 权重0.1

            # 计算总分
            total_score = sum(score_factors)

            return round(total_score, 1)

        except Exception as e:
            logger.error(f"计算指数得分失败: {str(e)}")
            return 50.0

    def _get_industry_data(self, trade_date: str) -> List[Dict[str, Any]]:
        """获取行业数据

        Args:
            trade_date: 交易日期

        Returns:
            List[Dict]: 行业数据列表
        """
        try:
            # 获取行业表现数据
            industry_performance = self.data_manager.get_industry_performance(
                trade_date)
            if industry_performance is None or industry_performance.empty:
                logger.warning(f"无法获取 {trade_date} 的行业表现数据")
                return []

            # 准备结果列表
            results = []

            # 处理每个行业数据
            for _, row in industry_performance.iterrows():
                industry_code = row.get('industry_code', '')
                industry_name = row.get('industry_name', '')

                if not industry_code or not industry_name:
                    continue

                # 提取行业数据
                industry_data = {
                    'code': industry_code,
                    'name': industry_name,
                    'change': float(row.get('change_pct', 0)),
                    'turnover': float(row.get('turnover', 0)) if 'turnover' in row else 0,
                    'pe': float(row.get('pe', 0)) if 'pe' in row else 0,
                    'total_market_cap': float(row.get('total_market_cap', 0)) if 'total_market_cap' in row else 0,
                    'volume': float(row.get('volume', 0)) if 'volume' in row else 0,
                    'amount': float(row.get('amount', 0)) if 'amount' in row else 0
                }

                # 获取行业成分股
                industry_stocks = self.data_manager.get_industry_stocks(
                    industry_code)

                # 如果无法获取成分股，跳过
                if industry_stocks is None or industry_stocks.empty:
                    logger.warning(
                        f"无法获取行业 {industry_name}({industry_code}) 的成分股数据")
                    continue

                # 统计上涨和下跌股票数量
                stock_change_data = self._get_industry_stock_changes(
                    industry_stocks, trade_date)

                # 添加到行业数据中
                industry_data.update(stock_change_data)

                # 计算行业强度指数
                industry_data['strength_index'] = self._calculate_industry_strength(
                    industry_data.get('up_count', 0),
                    industry_data.get('down_count', 0),
                    industry_data.get('change', 0)
                )

                # 获取最近N天的表现和趋势
                industry_trend_data = self._get_industry_trend(
                    industry_code, trade_date)
                industry_data.update(industry_trend_data)

                # 添加到结果列表
                results.append(industry_data)

            # 按行业强度指数排序
            results.sort(key=lambda x: x.get(
                'strength_index', 0), reverse=True)

            logger.info(f"成功获取 {len(results)} 个行业的数据")
            return results

        except Exception as e:
            logger.error(f"获取行业数据失败: {str(e)}")
            return []

    def _get_industry_stock_changes(self, industry_stocks: pd.DataFrame, trade_date: str) -> Dict[str, Any]:
        """获取行业成分股的涨跌情况

        Args:
            industry_stocks: 行业成分股数据框
            trade_date: 交易日期

        Returns:
            Dict: 行业成分股涨跌统计数据
        """
        try:
            # 获取前一个交易日
            prev_date = self.data_manager.get_previous_trading_date(
                trade_date, 1)

            # 初始化统计数据
            stats = {
                'up_count': 0,
                'down_count': 0,
                'flat_count': 0,
                'limit_up_count': 0,
                'limit_down_count': 0,
                'total_count': len(industry_stocks),
                'leading_up': {'code': '', 'name': '', 'change': 0},
                'leading_down': {'code': '', 'name': '', 'change': 0}
            }

            # 获取每只股票的涨跌情况
            stock_changes = []

            for _, stock in industry_stocks.iterrows():
                ts_code = stock.get('ts_code', '')
                stock_name = stock.get('name', '')

                if not ts_code:
                    continue

                # 获取股票行情数据
                stock_data = self.data_manager.get_daily_data(
                    ts_code, prev_date, trade_date)

                # 如果无法获取行情数据，跳过
                if stock_data is None or stock_data.empty or len(stock_data) < 1:
                    continue

                # 获取最新行情
                latest = stock_data.iloc[-1]

                # 计算涨跌幅
                change_pct = latest.get(
                    'pct_chg', 0) if 'pct_chg' in latest else 0

                # 统计涨跌家数
                if change_pct > 0:
                    stats['up_count'] += 1
                elif change_pct < 0:
                    stats['down_count'] += 1
                else:
                    stats['flat_count'] += 1

                # 统计涨停跌停
                if change_pct > 9.5:
                    stats['limit_up_count'] += 1
                elif change_pct < -9.5:
                    stats['limit_down_count'] += 1

                # 记录股票涨跌情况
                stock_changes.append({
                    'code': ts_code,
                    'name': stock_name,
                    'change': float(change_pct)
                })

            # 有效数据不为空时，找出领涨领跌股
            if stock_changes:
                # 按涨跌幅排序
                stock_changes.sort(key=lambda x: x['change'], reverse=True)

                # 领涨股
                stats['leading_up'] = stock_changes[0]

                # 领跌股
                stats['leading_down'] = stock_changes[-1]

            return stats

        except Exception as e:
            logger.error(f"获取行业成分股涨跌情况失败: {str(e)}")
            return {
                'up_count': 0, 'down_count': 0, 'flat_count': 0,
                'limit_up_count': 0, 'limit_down_count': 0, 'total_count': 0,
                'leading_up': {'code': '', 'name': '', 'change': 0},
                'leading_down': {'code': '', 'name': '', 'change': 0}
            }

    def _calculate_industry_strength(self, up_count: int, down_count: int, change_pct: float) -> float:
        """计算行业强度指数

        Args:
            up_count: 上涨家数
            down_count: 下跌家数
            change_pct: 行业涨跌幅

        Returns:
            float: 行业强度指数(0-100)
        """
        try:
            total_count = up_count + down_count

            if total_count == 0:
                return 50.0

            # 计算上涨比例因子 (0-100)
            up_ratio = up_count / total_count * 100

            # 计算行业涨跌幅因子 (归一化到0-100)
            change_score = min(max((change_pct + 10) * 5, 0), 100)

            # 综合得分 (权重：上涨比例70%，涨跌幅30%)
            strength_index = up_ratio * 0.7 + change_score * 0.3

            return round(strength_index, 1)

        except Exception as e:
            logger.error(f"计算行业强度指数失败: {str(e)}")
            return 50.0

    def _get_industry_trend(self, industry_code: str, trade_date: str) -> Dict[str, Any]:
        """获取行业趋势数据

        Args:
            industry_code: 行业代码
            trade_date: 交易日期

        Returns:
            Dict: 行业趋势数据
        """
        try:
            # 获取最近20个交易日
            start_date = self.data_manager.get_previous_trading_date(
                trade_date, 20)

            # 初始化结果
            result = {
                'trend': '震荡',
                'change_5d': 0.0,
                'change_10d': 0.0,
                'change_20d': 0.0,
                'momentum_score': 50.0
            }

            # 获取行业指数数据（如果数据源提供）
            # 这里假设数据源没有提供行业指数数据，使用成分股平均值代替

            # 获取行业成分股
            industry_stocks = self.data_manager.get_industry_stocks(
                industry_code)
            if industry_stocks is None or industry_stocks.empty:
                return result

            # 从成分股中随机抽取一部分（最多10只）进行分析
            sample_size = min(10, len(industry_stocks))
            sampled_stocks = industry_stocks.sample(sample_size) if len(
                industry_stocks) > sample_size else industry_stocks

            # 收集各个周期的涨跌幅
            changes_5d = []
            changes_10d = []
            changes_20d = []

            for _, stock in sampled_stocks.iterrows():
                ts_code = stock.get('ts_code', '')

                if not ts_code:
                    continue

                # 获取股票行情数据
                stock_data = self.data_manager.get_daily_data(
                    ts_code, start_date, trade_date)

                # 如果无法获取行情数据或数据不足，跳过
                if stock_data is None or stock_data.empty or len(stock_data) < 5:
                    continue

                # 计算各个周期的涨跌幅
                if len(stock_data) >= 5:
                    changes_5d.append(
                        ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[-5]) - 1) * 100)

                if len(stock_data) >= 10:
                    changes_10d.append(
                        ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[-10]) - 1) * 100)

                if len(stock_data) >= 20:
                    changes_20d.append(
                        ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[-20]) - 1) * 100)

            # 计算平均涨跌幅
            if changes_5d:
                result['change_5d'] = round(np.mean(changes_5d), 2)

            if changes_10d:
                result['change_10d'] = round(np.mean(changes_10d), 2)

            if changes_20d:
                result['change_20d'] = round(np.mean(changes_20d), 2)

            # 根据涨跌幅判断趋势
            if result['change_5d'] > 3 and result['change_10d'] > 5:
                result['trend'] = '强势上涨'
            elif result['change_5d'] > 1 and result['change_10d'] > 0:
                result['trend'] = '上涨'
            elif result['change_5d'] < -3 and result['change_10d'] < -5:
                result['trend'] = '强势下跌'
            elif result['change_5d'] < -1 and result['change_10d'] < 0:
                result['trend'] = '下跌'
            elif abs(result['change_5d']) < 1:
                result['trend'] = '震荡'
            elif result['change_5d'] > 0:
                result['trend'] = '震荡偏多'
            else:
                result['trend'] = '震荡偏空'

            # 计算动量得分 (0-100)
            momentum_factor_5d = min(
                max((result['change_5d'] + 5) * 10, 0), 100)
            momentum_factor_10d = min(
                max((result['change_10d'] + 10) * 5, 0), 100)
            momentum_factor_20d = min(
                max((result['change_20d'] + 20) * 2.5, 0), 100)

            # 加权计算动量得分（短期权重高）
            result['momentum_score'] = round(
                momentum_factor_5d * 0.5 + momentum_factor_10d * 0.3 + momentum_factor_20d * 0.2,
                1
            )

            return result

        except Exception as e:
            logger.error(f"获取行业趋势数据失败: {str(e)}")
            return {
                'trend': '震荡',
                'change_5d': 0.0,
                'change_10d': 0.0,
                'change_20d': 0.0,
                'momentum_score': 50.0
            }

    def _get_macro_economic_data(self, trade_date: str) -> Dict[str, Any]:
        """获取宏观经济数据

        Args:
            trade_date: 交易日期

        Returns:
            Dict: 宏观经济数据
        """
        # 待实现具体内容
        return {}

    def _get_money_flow_data(self, trade_date: str) -> Dict[str, Any]:
        """获取资金流向数据

        Args:
            trade_date: 交易日期

        Returns:
            Dict: 资金流向数据
        """
        # 待实现具体内容
        return {}

    def _analyze_market_sentiment(self, base_data: Dict[str, Any], indices_data: List[Dict[str, Any]],
                                  money_flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析市场情绪

        Args:
            base_data: 基础市场数据
            indices_data: 指数数据
            money_flow_data: 资金流向数据

        Returns:
            Dict: 市场情绪分析
        """
        try:
            # 简单实现，仅基于涨跌家数和涨跌幅度
            up_count = base_data.get('up_count', 0) if isinstance(
                base_data, dict) else 0
            down_count = base_data.get(
                'down_count', 0) if isinstance(base_data, dict) else 0
            avg_change = base_data.get(
                'avg_change_pct', 0) if isinstance(base_data, dict) else 0

            # 计算人气指数 (0-100)
            popularity = min(100, max(
                0, 50 + (up_count - down_count) / (up_count + down_count + 0.001) * 50))

            # 计算强弱指数 (0-100)
            strength = min(100, max(0, 50 + avg_change * 10))

            # 综合判断
            if strength > 70 and popularity > 70:
                status = "非常强势"
                advice = "市场热度高，注意风险，谨慎追高"
            elif strength > 60 and popularity > 60:
                status = "强势"
                advice = "市场上涨动能充足，关注强势板块"
            elif strength > 50 and popularity > 50:
                status = "偏强"
                advice = "市场偏强运行，把握结构性机会"
            elif strength < 30 and popularity < 30:
                status = "非常弱势"
                advice = "市场悲观情绪浓厚，等待企稳信号"
            elif strength < 40 and popularity < 40:
                status = "弱势"
                advice = "市场承压下行，注意控制仓位"
            elif strength < 50 and popularity < 50:
                status = "偏弱"
                advice = "市场偏弱运行，降低预期，控制风险"
            else:
                status = "震荡"
                advice = "市场震荡整理，等待方向选择"

            return {
                "status": status,
                "advice": advice,
                "popularity_index": popularity,
                "strength_index": strength,
                "up_down_ratio": up_count / (down_count + 0.001),
                "avg_change": avg_change
            }
        except Exception as e:
            logger.error(f"分析市场情绪失败: {str(e)}")
            return {
                "status": "未知",
                "advice": "数据不足，无法分析市场情绪",
                "popularity_index": 50,
                "strength_index": 50,
                "up_down_ratio": 1,
                "avg_change": 0
            }

    def _predict_future_hot_sectors(self, industry_data: List[Dict[str, Any]],
                                    market_sentiment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """预测未来热门板块

        Args:
            industry_data: 行业数据
            market_sentiment: 市场情绪

        Returns:
            List[Dict]: 未来热门板块
        """
        try:
            # 检查industry_data是否为None或空列表
            if industry_data is None or not isinstance(industry_data, list) or len(industry_data) == 0:
                logger.warning("行业数据为空，无法预测未来热门板块")
                return []

            # 计算潜力分数
            scored_industries = []
            for ind in industry_data:
                if not isinstance(ind, dict):
                    continue

                # 基础分数来自当前强度
                base_score = ind.get('strength_index', 50)

                # 上涨股票比例因子
                up_ratio = ind.get('up_count', 0) / \
                (ind.get('total_count', 1) or 1)
                up_factor = up_ratio * 20

                # 涨幅因子
                change_factor = ind.get('change', 0) * 2

                # 综合计算潜力分数
                potential_score = min(
                    100, max(0, base_score + up_factor + change_factor))

                scored_industries.append({
                    'name': ind.get('name', '未知'),
                    'code': ind.get('code', ''),
                    'current_change': ind.get('change', 0),
                    # 简单预测，实际应使用模型
                    'predicted_change': ind.get('change', 0) * 1.2,
                    'attention_index': potential_score,
                    'fund_inflow': ind.get('fund_inflow', 0),
                    'growth_score': potential_score,
                    'recommendation': self._generate_sector_recommendation(ind, potential_score)
                })

            # 按潜力分数排序
            scored_industries.sort(
                key=lambda x: x['attention_index'], reverse=True)

            # 返回前5个
            return scored_industries[:5]
        except Exception as e:
            logger.error(f"预测未来热门板块失败: {str(e)}")
            return []

    def _generate_sector_recommendation(self, industry_data: Dict[str, Any], potential_score: float) -> str:
        """生成行业推荐建议

        Args:
            industry_data: in行业数据
            potential_score: 潜力分数

        Returns:
            str: 推荐建议
        """
        try:
            name = industry_data.get('name', '未知行业')
            change = industry_data.get('change', 0)
            up_count = industry_data.get('up_count', 0)
            total_count = industry_data.get('total_count', 0)

            # 基于分数给出不同建议
            if potential_score > 80:
                return f"{name}行业表现强势，上涨股票占比高，有望持续领涨市场"
            elif potential_score > 70:
                return f"{name}行业动能较强，关注龙头股机会"
            elif potential_score > 60:
                return f"{name}行业近期表现活跃，可择机布局"
            elif potential_score > 50:
                return f"{name}行业有一定潜力，建议观望或小仓位试探"
            else:
                return f"{name}行业表现平淡，暂不建议重点关注"
        except Exception as e:
            logger.error(f"生成行业推荐建议失败: {str(e)}")
            return "数据不足，无法给出具体建议"

    def generate_market_report(self, trade_date: str = None) -> str:
        """生成市场报告

        Args:
            trade_date: 交易日期

        Returns:
            str: 市场报告文本
        """
        # 获取市场概览数据
        overview = self.get_market_overview(trade_date)
        if not overview:
            return "无法生成市场报告：数据获取失败"

        # 生成报告文本
        report = []
        report.append(f"# 市场概览报告 ({overview['date']})")
        report.append("\n## 市场状况")

        # 添加更多报告内容...

        return "\n".join(report)
