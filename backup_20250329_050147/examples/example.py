#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票涨停与大幅上涨捕捉系统示例脚本
演示如何使用系统的核心功能，包括风险管理和回测
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 添加当前目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from src.config import (
    DATA_DIR, STOCK_SELECTION_PARAMS, NOTIFICATION_CONFIG,
    RISK_CONTROL_PARAMS, BACKTEST_PARAMS
)
from src.data import StockDataFetcher, StockDataProcessor
from src.indicators import TechnicalIndicators
from src.risk.risk_management import RiskManager
from src.backtest.engine import BacktestEngine
from src.backtest.strategy import MomentumStrategy

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建结果目录
os.makedirs(os.path.join(DATA_DIR, "backtest_demo"), exist_ok=True)

def demo_risk_management():
    """
    演示风险管理功能
    """
    logger.info("=== 风险管理功能演示 ===")
    
    # 初始化风险管理器
    risk_manager = RiskManager(
        max_position_risk=0.02,
        max_portfolio_risk=0.05,
        max_industry_allocation=0.30,
        default_stop_loss_pct=0.05,
        default_take_profit_pct=0.15,
        use_trailing_stop=True
    )
    
    # 设置资金
    initial_capital = 1000000
    risk_manager.set_capital(initial_capital)
    logger.info(f"设置资金: {initial_capital:,.2f}")
    
    # 定义一些股票
    stocks = [
        {"code": "000001.SZ", "name": "平安银行", "price": 15.25, "industry": "银行", "risk_score": 0.5, "atr": 0.35},
        {"code": "600036.SH", "name": "招商银行", "price": 38.50, "industry": "银行", "risk_score": 0.4, "atr": 0.50},
        {"code": "000651.SZ", "name": "格力电器", "price": 42.30, "industry": "家电", "risk_score": 0.6, "atr": 0.60},
        {"code": "000333.SZ", "name": "美的集团", "price": 60.25, "industry": "家电", "risk_score": 0.7, "atr": 0.75},
        {"code": "600519.SH", "name": "贵州茅台", "price": 1825.00, "industry": "白酒", "risk_score": 0.3, "atr": 25.00}
    ]
    
    # 计算并添加仓位
    for stock in stocks:
        # 计算仓位大小
        position_size = risk_manager.calculate_position_size(
            stock_code=stock["code"],
            price=stock["price"],
            risk_score=stock["risk_score"]
        )
        
        # 计算金额
        amount = position_size * stock["price"]
        
        logger.info(f"计算 {stock['code']} {stock['name']} 仓位: {position_size} 股，金额: {amount:.2f}")
        
        # 添加仓位
        success = risk_manager.add_position(
            stock_code=stock["code"],
            quantity=position_size,
            price=stock["price"],
            industry=stock["industry"],
            risk_score=stock["risk_score"]
        )
        
        if success:
            logger.info(f"添加仓位成功: {stock['code']} {stock['name']}")
        else:
            logger.warning(f"添加仓位失败: {stock['code']} {stock['name']}")
    
    # 获取当前仓位状态
    positions = risk_manager.get_position_status()
    
    logger.info("\n当前仓位状态:")
    for code, position in positions.items():
        logger.info(f"{code}: 数量: {position['quantity']}, 入场价: {position['entry_price']:.2f}, 止损价: {position['stop_loss']:.2f}, 止盈价: {position['take_profit']:.2f}")
    
    # 验证投资组合风险
    risk_status = risk_manager.validate_portfolio()
    
    logger.info(f"\n投资组合风险校验: {'通过' if risk_status[0] else '不通过'}")
    logger.info(f"投资组合总价值: {risk_status[1]['portfolio_value']:.2f}")
    logger.info(f"可用资金: {risk_status[1]['available_capital']:.2f}")
    logger.info(f"投资组合风险: {risk_status[1]['portfolio_risk_pct']*100:.2f}% (最大允许: {risk_manager.max_portfolio_risk*100:.2f}%)")
    
    logger.info("\n行业配置:")
    for industry, allocation in risk_status[1]['industry_allocation_pct'].items():
        logger.info(f"{industry}: {allocation*100:.2f}% (最大允许: {risk_manager.max_industry_allocation*100:.2f}%)")
    
    # 模拟价格变动
    logger.info("\n模拟股价变动:")
    
    # 更新每只股票的价格，涨幅8%
    for stock in stocks:
        new_price = stock["price"] * 1.08
        risk_manager.update_position(stock["code"], new_price)
        
        # 检查是否触发止损或止盈
        stop_loss_triggered = risk_manager.check_stop_loss(stock["code"])
        take_profit_triggered = risk_manager.check_take_profit(stock["code"])
        
        logger.info(f"{stock['code']} 价格更新为 {new_price:.2f} (上涨8%), 是否触发平仓: {stop_loss_triggered or take_profit_triggered}")
    
    # 再次验证投资组合风险
    risk_status = risk_manager.validate_portfolio()
    
    logger.info(f"\n价格变动后投资组合风险校验: {'通过' if risk_status[0] else '不通过'}")
    logger.info(f"投资组合总价值: {risk_status[1]['portfolio_value']:.2f}")

def demo_backtest():
    """
    演示回测功能
    """
    logger.info("=== 回测功能演示 ===")
    
    # 初始化风险管理器
    risk_manager = RiskManager(
        max_position_risk=0.02,
        max_portfolio_risk=0.05,
        max_industry_allocation=0.30,
        default_stop_loss_pct=0.05,
        default_take_profit_pct=0.15,
        use_trailing_stop=True
    )
    
    # 定义回测参数
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    stock_codes = ["000001.SZ", "000651.SZ", "000333.SZ", "600036.SH", "600519.SH"]
    
    logger.info(f"创建模拟回测数据, 股票: {stock_codes}, 时间: {start_date} 到 {end_date}")
    
    # 创建模拟数据
    backtest_data = {}
    
    # 设置日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B'表示工作日
    
    # 行业映射字典
    industry_map = {
        "000001.SZ": "银行",
        "000651.SZ": "家电",
        "000333.SZ": "家电",
        "600036.SH": "银行",
        "600519.SH": "白酒"
    }
    
    # 为每只股票创建模拟数据
    for stock_code in stock_codes:
        try:
            # 设置初始价格
            if stock_code == "600519.SH":  # 贵州茅台价格较高
                initial_price = 1800.0
            else:
                initial_price = 40.0 + np.random.rand() * 30.0  # 40-70之间的随机价格
            
            # 创建价格序列，加入一些随机走势
            np.random.seed(hash(stock_code) % 100)  # 使每只股票有不同但可重现的走势
            
            # 为了增加更多动量选股机会，我们为每只股票生成不同特征的价格序列
            # 一半股票有上升趋势，一半股票有震荡趋势
            is_trend_stock = hash(stock_code) % 2 == 0
            
            # 增加噪声
            noise = np.random.normal(0.0005, 0.015, len(date_range))
            
            # 创建基础趋势
            if is_trend_stock:
                # 上升趋势股，平均每天上涨0.2%
                base_trend = np.linspace(0, 0.2, len(date_range))
            else:
                # 震荡股，有小幅上升趋势
                base_trend = np.linspace(0, 0.05, len(date_range))
            
            # 添加周期性波动，模拟市场周期，周期长度为20-60天
            cycle_length = 20 + hash(stock_code) % 40
            cycles = 0.15 * np.sin(np.linspace(0, 2*np.pi * (len(date_range)/cycle_length), len(date_range)))
            
            # 添加几次明显的上涨阶段，以满足动量策略需要
            strong_momentum_periods = []
            num_momentum_periods = 2 + hash(stock_code) % 3  # 2-4次强势上涨期
            
            for i in range(num_momentum_periods):
                # 随机确定强势上涨的起始位置
                start_idx = (i * len(date_range) // num_momentum_periods) + (hash(stock_code) % 30)
                if start_idx >= len(date_range) - 30:
                    start_idx = len(date_range) - 30
                
                # 强势上涨持续15-25天
                duration = 15 + hash((stock_code, i)) % 10
                if start_idx + duration >= len(date_range):
                    duration = len(date_range) - start_idx - 1
                
                # 创建强势上涨期（每天平均涨1%）
                strong_momentum = np.zeros(len(date_range))
                strong_momentum[start_idx:start_idx+duration] = 0.01
                
                strong_momentum_periods.append(strong_momentum)
            
            # 组合所有因素：基础噪声 + 基础趋势 + 周期波动 + 强势上涨期
            daily_returns = noise * 0.3 + base_trend * 0.003 + cycles * 0.005
            
            # 添加强势动量阶段
            for momentum_period in strong_momentum_periods:
                daily_returns += momentum_period
            
            # 计算价格序列
            close_prices = initial_price * np.cumprod(1 + daily_returns)
            
            # 确保价格在合理范围内
            # 对于贵州茅台，价格范围在1500-2500之间
            # 对于其他股票，价格范围在20-100之间
            if stock_code == "600519.SH":  # 贵州茅台
                close_prices = np.clip(close_prices, 1500, 2500)
            else:
                close_prices = np.clip(close_prices, 20, 100)
            
            # 生成开盘价、最高价和最低价
            open_prices = close_prices * (1 + np.random.normal(0, 0.005, len(date_range)))
            high_prices = np.maximum(close_prices, open_prices) * (1 + np.abs(np.random.normal(0, 0.008, len(date_range))))
            low_prices = np.minimum(close_prices, open_prices) * (1 - np.abs(np.random.normal(0, 0.008, len(date_range))))
            
            # 生成成交量
            volumes = np.random.normal(1000000, 300000, len(date_range))
            volumes = np.maximum(volumes, 100000)  # 确保成交量为正
            
            # 生成成交额
            amounts = close_prices * volumes
            
            # 创建DataFrame
            df = pd.DataFrame({
                'date': date_range,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volumes,
                'amount': amounts,
                'industry': industry_map.get(stock_code, "其他")
            })
            
            # 设置日期为索引
            df.set_index('date', inplace=True)
            
            # 计算一些技术指标
            # 5日移动平均线
            df['ma_5'] = df['close'].rolling(window=5).mean()
            # 10日移动平均线
            df['ma_10'] = df['close'].rolling(window=10).mean()
            # 20日移动平均线
            df['ma_20'] = df['close'].rolling(window=20).mean()
            # 60日移动平均线
            df['ma_60'] = df['close'].rolling(window=60).mean()
            
            # 计算ATR
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    np.abs(df['high'] - df['close'].shift(1)),
                    np.abs(df['low'] - df['close'].shift(1))
                )
            )
            
            df['atr_14'] = df['true_range'].rolling(window=14).mean()
            
            # 计算MACD
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # 计算RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算BOLL
            df['boll_mid'] = df['close'].rolling(window=20).mean()
            df['boll_std'] = df['close'].rolling(window=20).std()
            df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
            df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
            
            # 计算日收益率
            df['daily_return'] = df['close'].pct_change()
            
            # 记录数据
            backtest_data[stock_code] = df
            
            logger.info(f"创建 {stock_code} 的模拟数据: {len(df)} 条记录, 初始价格: {df['close'].iloc[0]:.2f}, 最终价格: {df['close'].iloc[-1]:.2f}")
            
        except Exception as e:
            logger.error(f"创建 {stock_code} 模拟数据失败: {str(e)}")
    
    # 创建自定义策略类，继承自带的策略并增加调试功能
    class DebugMomentumStrategy(MomentumStrategy):
        def _select_stocks(self, date: str, data: Dict[str, Dict]):
            """增强版的选股方法，添加详细调试日志"""
            logger.info(f"开始选股，当前日期: {date}")
            logger.info(f"动量策略参数 - 回溯期: {self.lookback_period}天, 动量阈值: {self.momentum_threshold*100:.2f}%, 止盈: {self.profit_target*100:.2f}%, 止损: {self.stop_loss*100:.2f}%")
            
            total_stocks = len(data)
            skipped_positions = 0
            skipped_no_history = 0
            skipped_short_history = 0
            skipped_negative_index = 0
            skipped_low_momentum = 0
            valid_stocks = 0
            
            momentum_scores = []
            
            logger.info(f"分析 {total_stocks} 只股票...")
            
            for stock_code, stock_data in data.items():
                # 跳过已持仓的股票
                if stock_code in self.get_positions():
                    skipped_positions += 1
                    continue
                
                # 如果历史数据不足，跳过
                if stock_code not in self.history:
                    skipped_no_history += 1
                    continue
                
                df = self.history[stock_code]
                if len(df) < self.lookback_period + 1:
                    logger.debug(f"{stock_code} 历史数据不足 {self.lookback_period + 1} 条，跳过")
                    skipped_short_history += 1
                    continue
                
                # 获取今天和回溯期前的收盘价
                current_price = stock_data.get('close', 0)
                try:
                    hist_price_idx = df.index.get_loc(date) - self.lookback_period
                    if hist_price_idx < 0:
                        logger.debug(f"{stock_code} 历史价格索引为负，跳过")
                        skipped_negative_index += 1
                        continue
                    
                    hist_price = df.iloc[hist_price_idx]['close']
                    
                    # 计算动量得分 (简单的价格变化百分比)
                    momentum = (current_price - hist_price) / hist_price
                    
                    logger.info(f"{stock_code} 当前价格: {current_price:.2f}, {self.lookback_period}天前价格: {hist_price:.2f}, 动量: {momentum*100:.2f}%, 阈值: {self.momentum_threshold*100:.2f}%")
                    
                    # 跳过动量小于阈值的股票
                    if momentum < self.momentum_threshold:
                        logger.info(f"{stock_code} 动量 {momentum*100:.2f}% 小于阈值 {self.momentum_threshold*100:.2f}%，跳过")
                        skipped_low_momentum += 1
                        continue
                    
                    # 计算额外的技术指标得分，这里简单地使用最近5天的涨幅
                    recent_change = 0
                    if len(df) >= 5:
                        recent_idx = df.index.get_loc(date)
                        if recent_idx >= 5:
                            recent_price = df.iloc[recent_idx - 5]['close']
                            recent_change = (current_price - recent_price) / recent_price
                            logger.info(f"{stock_code} 5天涨幅: {recent_change*100:.2f}%")
                    
                    # 计算综合得分 (70% 动量 + 30% 近期涨幅)
                    score = 0.7 * momentum + 0.3 * recent_change
                    
                    # 添加到动量得分列表
                    momentum_scores.append((stock_code, score, current_price))
                    logger.info(f"{stock_code} 综合得分: {score*100:.2f}%")
                    valid_stocks += 1
                    
                except Exception as e:
                    logger.error(f"{stock_code} 选股过程出错: {str(e)}")
            
            # 汇总选股结果
            logger.info(f"选股汇总: 总共 {total_stocks} 只股票, 有效 {valid_stocks} 只")
            logger.info(f"跳过原因: 已持仓 {skipped_positions}, 无历史数据 {skipped_no_history}, " 
                      f"历史数据太短 {skipped_short_history}, 索引为负 {skipped_negative_index}, "
                      f"动量不足 {skipped_low_momentum}")
            
            # 按动量得分排序 (从大到小)
            momentum_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 更新选股结果
            available_slots = self.max_positions - len(self.get_positions())
            self.selected_stocks = [item[0] for item in momentum_scores[:available_slots]]
            
            if self.selected_stocks:
                logger.info(f"选出 {len(self.selected_stocks)} 只动量最强的股票: {', '.join(self.selected_stocks)}")
            else:
                logger.info("没有选出任何股票")
    
    # 创建并初始化策略
    strategy = DebugMomentumStrategy(
        lookback_period=10,
        momentum_threshold=0.02,
        profit_target=0.10,
        stop_loss=0.05,
        max_positions=5
    )
    
    # 创建回测引擎
    engine = BacktestEngine(
        data=backtest_data,
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=1000000.0,
        commission=0.0003,
        slippage=0.0001,
        risk_manager=risk_manager
    )
    
    # 设置风险管理器
    risk_manager.set_capital(1000000.0)
    engine.set_risk_manager(risk_manager)
    
    # 确保策略可以访问数据
    strategy.set_engine(engine)
    strategy.initialize()
    
    # 检查策略是否正确加载了历史数据
    logger.info(f"策略历史数据加载检查: 历史数据数量 = {len(strategy.history)}")
    if len(strategy.history) == 0:
        logger.error("策略没有加载任何历史数据！")
        # 手动设置历史数据
        strategy.history = backtest_data
        logger.info(f"手动设置历史数据后，数据数量 = {len(strategy.history)}")
    
    logger.info("开始运行回测...")
    
    # 运行回测
    results = engine.run()
    
    logger.info("\n回测结果:")
    
    # 输出回测结果摘要
    summary = f"""回测结果摘要
----------------
初始资金: {engine.initial_capital:,.2f}
最终资金: {engine.total_capital:,.2f}
总收益率: {results['total_return']*100:.2f}%
年化收益率: {results['annual_return']*100:.2f}%
最大回撤: {abs(results['max_drawdown'])*100:.2f}%
夏普比率: {results['sharpe_ratio']:.2f}
交易次数: {results['trade_count']}
胜率: {results['win_rate']*100:.2f}%
盈亏比: {results['profit_factor']:.2f}
平均持仓天数: {results.get('avg_hold_days', 0):.1f}
平均盈利: {results.get('avg_profit', 0):.2f}
平均亏损: {results.get('avg_loss', 0):.2f}
"""
    logger.info(summary)
    
    # 保存回测结果
    try:
        os.makedirs("results/backtest_demo", exist_ok=True)
        
        # 保存权益曲线数据
        equity_df = pd.DataFrame(engine.daily_equity)
        equity_df.to_csv("results/backtest_demo/equity.csv", index=False)
        
        # 保存表现指标
        with open("results/backtest_demo/performance.json", "w") as f:
            import json
            json.dump(results, f, indent=2)
        
        # 绘制权益曲线
        engine.plot_equity_curve(save_path="results/backtest_demo/equity_curve.png")
        
        logger.info("回测结果已保存至 results/backtest_demo")
    except Exception as e:
        logger.error(f"保存回测结果失败: {str(e)}")

def demo_real_data_backtest():
    """
    使用真实历史数据进行回测
    """
    
    # 初始化日志
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("RealDataBacktest")
    
    # 导入我们创建的多数据源模块
    from src.data.multi_data_source import MultiDataSource
    from src.risk.risk_management import RiskManager
    from src.backtest.engine import BacktestEngine
    from src.backtest.strategy import MomentumStrategy
    import pandas as pd
    import os
    
    # 设置结果目录
    results_dir = "results/real_data_backtest"
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化数据源（使用Tushare和AKShare）
    # 替换为您的Tushare token
    tushare_token = "b82eb228e75ef3b76f18633ef5b0d3b9ef70904d883be5bbee8a2321"
    data_source = MultiDataSource(
        tushare_token=tushare_token,
        use_tushare=True,
        use_akshare=True,
        cache_dir="./data/cache"
    )
    
    # 设置回测参数
    start_date = "2021-01-01"
    end_date = "2021-12-31"
    initial_capital = 1000000.0  # 初始资金：100万
    
    # 选择回测股票
    stock_codes = [
        "600519.SH",  # 贵州茅台
        "000858.SZ",  # 五粮液
        "601318.SH",  # 中国平安
        "600036.SH",  # 招商银行
        "000333.SZ"   # 美的集团
    ]
    
    logger.info(f"开始获取回测数据，时间范围：{start_date} 至 {end_date}")
    logger.info(f"股票列表：{stock_codes}")
    
    # 获取回测所需的历史数据
    backtest_data = data_source.get_backtest_data(
        stock_codes=stock_codes,
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"  # 使用前复权数据
    )
    
    # 检查数据是否获取成功
    valid_stocks = [code for code, df in backtest_data.items() if df is not None and len(df) > 0]
    logger.info(f"成功获取数据的股票数量: {len(valid_stocks)}/{len(stock_codes)}")
    
    if not valid_stocks:
        logger.error("未能获取任何有效的股票数据，回测终止")
        return
    
    # 重新设置可用股票
    stock_codes = valid_stocks
    
    # 创建DEBUG版本的动量策略
    class DebugMomentumStrategy(MomentumStrategy):
        """带有调试功能的动量策略"""
        
        def _select_stocks(self, date):
            """
            根据动量分数选择股票
            
            Args:
                date: 当前日期
                
            Returns:
                dict: 选择的股票及其分数 {stock_code: score}
            """
            logger.info(f"开始选股 - 日期: {date.strftime('%Y-%m-%d')}")
            logger.info(f"策略参数: 回看期={self.lookback_period}天, 动量阈值={self.momentum_threshold*100:.2f}%, 止盈={self.profit_target*100:.2f}%, 止损={self.stop_loss*100:.2f}%")
            
            # 初始化各种跳过原因的计数器
            skip_already_held = 0
            skip_no_history = 0
            skip_short_history = 0
            skip_negative_index = 0
            skip_low_momentum = 0
            valid_stocks = 0
            
            # 当前持有的股票
            current_positions = self._check_positions()
            logger.info(f"当前已持有 {len(current_positions)} 只股票")
            
            # 计算所有股票的动量得分
            momentum_scores = {}
            all_stocks_count = len(self.engine.data.keys())
            logger.info(f"总共分析 {all_stocks_count} 只股票")
            
            for stock_code in self.engine.data.keys():
                # 如果已经持有该股票，跳过
                if stock_code in current_positions:
                    skip_already_held += 1
                    logger.debug(f"跳过 {stock_code}: 已经持有")
                    continue
                
                # 获取该股票的历史数据
                stock_data = self.engine.data.get(stock_code)
                if stock_data is None:
                    skip_no_history += 1
                    logger.debug(f"跳过 {stock_code}: 没有历史数据")
                    continue
                
                # 获取截止到当前日期的数据
                history = stock_data[stock_data.index <= date]
                if len(history) < self.lookback_period + 1:
                    skip_short_history += 1
                    logger.debug(f"跳过 {stock_code}: 历史数据不足，仅有 {len(history)} 天，需要 {self.lookback_period + 1} 天")
                    continue
                
                # 获取当前价格和历史价格
                current_price = history['close'].iloc[-1]
                previous_price = history['close'].iloc[-self.lookback_period-1]
                
                if previous_price <= 0:
                    skip_negative_index += 1
                    logger.debug(f"跳过 {stock_code}: 历史价格非正数 ({previous_price})")
                    continue
                
                # 计算动量分数
                momentum = (current_price / previous_price) - 1
                logger.debug(f"股票 {stock_code}: 当前价格={current_price:.2f}, {self.lookback_period}天前价格={previous_price:.2f}, 动量={momentum*100:.2f}%")
                
                # 如果动量大于阈值，加入候选
                if momentum > self.momentum_threshold:
                    momentum_scores[stock_code] = momentum
                    valid_stocks += 1
                    logger.info(f"有效股票 {stock_code}: 动量={momentum*100:.2f}% > 阈值{self.momentum_threshold*100:.2f}%")
                else:
                    skip_low_momentum += 1
                
            # 日志记录选股结果
            logger.info(f"选股结果汇总 - 总股票数: {all_stocks_count}, 有效股票数: {valid_stocks}")
            logger.info(f"跳过原因 - 已持有: {skip_already_held}, 无历史数据: {skip_no_history}, 历史数据不足: {skip_short_history}, 负数指标: {skip_negative_index}, 动量不足: {skip_low_momentum}")
            
            # 如果有足够的候选股票，选择动量最高的N只
            selected_stocks = {}
            if momentum_scores:
                # 按动量降序排序
                sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
                # 选择前N只股票，但不超过最大持仓数
                max_new_positions = self.max_positions - len(current_positions)
                selected_stocks = dict(sorted_stocks[:max_new_positions])
                logger.info(f"最终选择了 {len(selected_stocks)} 只股票进行买入")
                for stock, score in selected_stocks.items():
                    logger.info(f"选择买入 {stock}: 动量分数={score*100:.2f}%")
            else:
                logger.info("没有选出任何股票")
            
            return selected_stocks
    
    # 初始化风险管理器
    risk_manager = RiskManager()
    risk_manager.set_capital(initial_capital)
    
    # 初始化动量策略
    strategy = DebugMomentumStrategy(
        lookback_period=10,           # 回看期: 10天
        momentum_threshold=0.02,      # 动量阈值: 2%
        profit_target=0.10,           # 止盈: 10%
        stop_loss=0.05,               # 止损: 5%
        max_positions=4               # 最大持仓数: 4
    )
    strategy.set_risk_manager(risk_manager)
    
    # 初始化回测引擎
    engine = BacktestEngine(
        data=backtest_data,
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission=0.0003,  # 手续费: 万三
        slippage=0.0002     # 滑点: 万二
    )
    
    # 运行回测
    logger.info("开始回测...")
    engine.run()
    
    # 输出回测结果
    performance = engine.get_performance()
    logger.info("回测完成，性能指标:")
    for key, value in performance.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.2%}")
        else:
            logger.info(f"{key}: {value}")
    
    # 保存回测结果
    engine.save_results(results_dir)
    logger.info(f"回测结果已保存至 {results_dir}")

if __name__ == "__main__":
    print("=" * 50)
    print("示例程序启动")
    print("=" * 50)
    
    # 选择要运行的示例
    # demo_risk_management()  # 风险管理示例
    # demo_backtest()  # 回测示例
    demo_real_data_backtest()  # 真实数据回测示例
    
    print("=" * 50)
    print("示例程序结束")
    print("=" * 50) 