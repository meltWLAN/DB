#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
行业动量交易策略
基于行业表现和个股动量选择交易标的
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置中文字体
try:
    # 尝试使用系统中文字体
    font_path = '/System/Library/Fonts/PingFang.ttc'  # macOS中文字体
    if os.path.exists(font_path):
        chinese_font = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = chinese_font.get_name()
    else:
        # 如果找不到系统字体，使用matplotlib内置字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    # 如果设置字体失败，则忽略中文显示问题
    pass

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加当前目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 确保结果目录存在
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/strategy_backtest")
os.makedirs(results_dir, exist_ok=True)

# 导入配置
from src.config import DATA_SOURCE_CONFIG

class IndustryMomentumStrategy:
    """行业动量交易策略
    
    策略说明：
    1. 选择表现最好的行业（基于回测结果）
    2. 在该行业中选择具有强势动量的个股
    3. 使用技术指标（MACD、RSI、KDJ）确认交易信号
    4. 实施仓位管理和风险控制
    """
    
    def __init__(self, start_date, end_date, industry_analysis_file, stock_analysis_file):
        """初始化策略
        
        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
            industry_analysis_file: 行业分析结果文件路径
            stock_analysis_file: 个股分析结果文件路径
        """
        self.start_date = start_date
        self.end_date = end_date
        self.industry_analysis = pd.read_csv(industry_analysis_file, index_col='industry')
        self.stock_analysis = pd.read_csv(stock_analysis_file)
        self.selected_stocks = []
        self.positions = {}  # 持仓 {股票代码: 持仓比例}
        self.signals = {}    # 交易信号 {股票代码: 信号}
        self.performance = {}  # 策略表现
        
        # 策略参数
        self.top_industry_count = 3        # 选择前N个行业
        self.top_stock_per_industry = 2    # 每个行业选择前N只股票
        self.position_size = 0.2           # 单只股票最大仓位
        self.rsi_oversold = 30             # RSI超卖阈值
        self.rsi_overbought = 70           # RSI超买阈值
        self.stop_loss = -0.05             # 止损阈值
        self.take_profit = 0.15            # 止盈阈值
        
        logger.info(f"初始化行业动量策略: {start_date} 至 {end_date}")
    
    def select_industries(self):
        """选择表现最好的行业"""
        # 根据夏普比率排序
        top_industries = self.industry_analysis.sort_values('sharpe_ratio', ascending=False).head(self.top_industry_count)
        logger.info(f"选择的行业: {top_industries.index.tolist()}")
        return top_industries.index.tolist()
    
    def select_stocks(self, industries):
        """在选定行业中选择个股
        
        Args:
            industries: 选定的行业列表
            
        Returns:
            selected_stocks: 选定的股票列表
        """
        selected_stocks = []
        
        for industry in industries:
            # 找出该行业的股票
            industry_stocks = self.stock_analysis[self.stock_analysis['行业'] == industry]
            
            # 如果该行业有股票，选择表现最好的前N只
            if not industry_stocks.empty:
                # 根据夏普比率排序
                top_stocks = industry_stocks.sort_values('夏普比率', ascending=False).head(self.top_stock_per_industry)
                industry_selected = top_stocks['股票代码'].tolist()
                selected_stocks.extend(industry_selected)
                
                logger.info(f"行业 {industry} 选择的股票: {industry_selected}")
        
        self.selected_stocks = selected_stocks
        return selected_stocks
    
    def get_stock_data(self, stock_codes):
        """获取选定股票的历史数据
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            data: 股票数据字典 {股票代码: 数据DataFrame}
        """
        try:
            import jqdatasdk as jq
            
            # 获取JoinQuant配置
            jq_config = DATA_SOURCE_CONFIG.get('joinquant', {})
            username = jq_config.get('username', '')
            password = jq_config.get('password', '')
            
            if not username or not password:
                logger.error("JoinQuant账号未配置，请在src/config/__init__.py中设置")
                return {}
            
            # 登录
            logger.info("登录JoinQuant...")
            jq.auth(username, password)
            
            # 获取账号信息
            account_info = jq.get_account_info()
            logger.info(f"账号信息: {account_info}")
            
            # 获取股票数据
            data = {}
            for stock_code in stock_codes:
                logger.info(f"获取 {stock_code} 数据...")
                
                # 获取价格数据
                df = jq.get_price(stock_code, start_date=self.start_date, end_date=self.end_date, 
                               frequency='daily', fields=['open', 'close', 'high', 'low', 'volume', 'money'])
                
                # 将日期从索引转为列
                df = df.reset_index()
                df.rename(columns={'index': 'date'}, inplace=True)
                
                # 计算技术指标
                # 1. 移动平均线
                df['MA5'] = df['close'].rolling(window=5).mean()
                df['MA10'] = df['close'].rolling(window=10).mean()
                df['MA20'] = df['close'].rolling(window=20).mean()
                df['MA60'] = df['close'].rolling(window=60).mean()
                
                # 2. MACD
                df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
                df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
                df['DIF'] = df['EMA12'] - df['EMA26']
                df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
                df['MACD'] = 2 * (df['DIF'] - df['DEA'])
                
                # 3. RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # 4. KDJ
                low_min = df['low'].rolling(window=9).min()
                high_max = df['high'].rolling(window=9).max()
                df['RSV'] = (df['close'] - low_min) / (high_max - low_min) * 100
                df['K'] = df['RSV'].ewm(alpha=1/3, adjust=False).mean()
                df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
                df['J'] = 3 * df['K'] - 2 * df['D']
                
                # 5. 布林带
                df['MA20'] = df['close'].rolling(window=20).mean()
                df['STD20'] = df['close'].rolling(window=20).std()
                df['UPPER'] = df['MA20'] + 2 * df['STD20']
                df['LOWER'] = df['MA20'] - 2 * df['STD20']
                
                # 6. 动量指标
                df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
                df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
                df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
                
                # 计算收益率
                df['daily_return'] = df['close'].pct_change()
                
                # 计算累计收益率
                df['cumulative_return'] = (1 + df['daily_return']).cumprod()
                
                data[stock_code] = df
                logger.info(f"获取 {stock_code} 数据完成，包含 {len(df)} 条记录")
                
            # 登出
            jq.logout()
            logger.info("JoinQuant数据获取完成，已登出")
            
            return data
                
        except Exception as e:
            logger.error(f"获取JoinQuant数据失败: {str(e)}")
            return {}
    
    def generate_signals(self, data):
        """生成交易信号
        
        Args:
            data: 股票数据字典 {股票代码: 数据DataFrame}
            
        Returns:
            signals: 交易信号字典 {股票代码: DataFrame包含信号}
        """
        signals = {}
        
        for stock_code, df in data.items():
            # 创建信号列
            df['signal'] = 0  # 0: 无信号, 1: 买入, -1: 卖出
            
            # 生成买入信号
            # 1. MACD金叉
            df['macd_golden_cross'] = (df['DIF'] > df['DEA']) & (df['DIF'].shift(1) <= df['DEA'].shift(1))
            
            # 2. RSI低位反转
            df['rsi_bullish'] = (df['RSI'] > 30) & (df['RSI'].shift(1) <= 30)
            
            # 3. KDJ金叉
            df['kdj_golden_cross'] = (df['J'] > df['D']) & (df['J'].shift(1) <= df['D'].shift(1))
            
            # 4. 价格突破MA20
            df['price_breakout'] = (df['close'] > df['MA20']) & (df['close'].shift(1) <= df['MA20'].shift(1))
            
            # 5. 动量加速
            df['momentum_increasing'] = (df['momentum_5d'] > df['momentum_5d'].shift(1)) & (df['momentum_5d'] > 0)
            
            # 综合买入信号
            df['buy_signal'] = (
                (df['macd_golden_cross'] | df['kdj_golden_cross']) &  # 技术指标金叉
                (df['RSI'] < 70) &  # RSI未超买
                (df['close'] > df['MA20']) &  # 价格在MA20之上
                (df['momentum_5d'] > 0)  # 5日动量为正
            )
            
            # 生成卖出信号
            # 1. MACD死叉
            df['macd_death_cross'] = (df['DIF'] < df['DEA']) & (df['DIF'].shift(1) >= df['DEA'].shift(1))
            
            # 2. RSI超买
            df['rsi_bearish'] = (df['RSI'] > 70)
            
            # 3. KDJ死叉
            df['kdj_death_cross'] = (df['J'] < df['D']) & (df['J'].shift(1) >= df['D'].shift(1))
            
            # 4. 价格跌破MA20
            df['price_breakdown'] = (df['close'] < df['MA20']) & (df['close'].shift(1) >= df['MA20'].shift(1))
            
            # 5. 动量减弱
            df['momentum_decreasing'] = (df['momentum_5d'] < df['momentum_5d'].shift(1)) & (df['momentum_5d'] < 0)
            
            # 综合卖出信号
            df['sell_signal'] = (
                (df['macd_death_cross'] | df['kdj_death_cross']) |  # 技术指标死叉
                (df['RSI'] > 70) |  # RSI超买
                (df['close'] < df['MA20'] * 0.95) |  # 价格明显跌破MA20
                (df['momentum_5d'] < -0.03)  # 5日动量明显为负
            )
            
            # 设置信号
            df.loc[df['buy_signal'], 'signal'] = 1
            df.loc[df['sell_signal'], 'signal'] = -1
            
            signals[stock_code] = df
            
        return signals
    
    def backtest(self):
        """回测策略"""
        # 1. 选择行业
        top_industries = self.select_industries()
        
        # 2. 选择个股
        selected_stocks = self.select_stocks(top_industries)
        
        # 3. 获取历史数据
        stock_data = self.get_stock_data(selected_stocks)
        
        # 4. 生成交易信号
        self.signals = self.generate_signals(stock_data)
        
        # 5. 模拟交易
        # 初始资金
        initial_capital = 1000000
        capital = initial_capital
        # 持仓
        positions = {stock: 0 for stock in selected_stocks}
        # 每日净值
        daily_net_values = []
        # 交易记录
        trades = []
        
        # 获取所有日期
        all_dates = sorted(set().union(*[df['date'].tolist() for df in stock_data.values()]))
        
        # 按日期模拟交易
        for date in all_dates:
            daily_value = capital
            
            # 更新持仓价值
            for stock, position in positions.items():
                if position > 0:
                    # 获取当日收盘价
                    df = stock_data[stock]
                    if date in df['date'].values:
                        price = df.loc[df['date'] == date, 'close'].values[0]
                        daily_value += position * price
            
            # 记录当日净值
            daily_net_values.append({
                'date': date,
                'net_value': daily_value,
                'return': daily_value / initial_capital - 1
            })
            
            # 交易决策
            for stock in selected_stocks:
                df = stock_data[stock]
                if date in df['date'].values:
                    idx = df.loc[df['date'] == date].index[0]
                    signal = df.loc[idx, 'signal']
                    price = df.loc[idx, 'close']
                    
                    # 买入
                    if signal == 1 and positions[stock] == 0 and capital > 0:
                        # 计算买入数量
                        amount = min(capital, initial_capital * self.position_size)
                        shares = int(amount / price)
                        cost = shares * price
                        
                        if shares > 0:
                            positions[stock] = shares
                            capital -= cost
                            
                            trades.append({
                                'date': date,
                                'stock': stock,
                                'action': 'BUY',
                                'price': price,
                                'shares': shares,
                                'amount': cost
                            })
                            
                            logger.info(f"买入 {stock}: {shares}股, 价格{price}, 总额{cost}")
                    
                    # 卖出
                    elif (signal == -1 or self._check_stop_conditions(stock, price)) and positions[stock] > 0:
                        shares = positions[stock]
                        amount = shares * price
                        
                        positions[stock] = 0
                        capital += amount
                        
                        trades.append({
                            'date': date,
                            'stock': stock,
                            'action': 'SELL',
                            'price': price,
                            'shares': shares,
                            'amount': amount
                        })
                        
                        logger.info(f"卖出 {stock}: {shares}股, 价格{price}, 总额{amount}")
        
        # 计算最终市值
        final_value = capital
        for stock, position in positions.items():
            if position > 0:
                # 获取最后一天的收盘价
                df = stock_data[stock]
                price = df['close'].iloc[-1]
                final_value += position * price
        
        # 计算策略回报
        total_return = final_value / initial_capital - 1
        
        # 计算年化收益率
        days = (datetime.strptime(self.end_date, '%Y-%m-%d') - datetime.strptime(self.start_date, '%Y-%m-%d')).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # 计算最大回撤
        net_value_df = pd.DataFrame(daily_net_values)
        net_value_df['cummax'] = net_value_df['net_value'].cummax()
        net_value_df['drawdown'] = net_value_df['net_value'] / net_value_df['cummax'] - 1
        max_drawdown = net_value_df['drawdown'].min()
        
        # 计算夏普比率
        if len(net_value_df) > 1:
            daily_returns = net_value_df['net_value'].pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
        else:
            sharpe_ratio = 0
        
        # 记录回测结果
        self.performance = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades,
            'daily_net_values': daily_net_values
        }
        
        # 打印回测结果摘要
        logger.info(f"回测结果摘要:")
        logger.info(f"初始资金: {initial_capital}")
        logger.info(f"最终市值: {final_value:.2f}")
        logger.info(f"总收益率: {total_return:.2%}")
        logger.info(f"年化收益率: {annual_return:.2%}")
        logger.info(f"最大回撤: {max_drawdown:.2%}")
        logger.info(f"夏普比率: {sharpe_ratio:.2f}")
        logger.info(f"交易次数: {len(trades)}")
        
        return self.performance
    
    def _check_stop_conditions(self, stock, current_price):
        """检查止盈止损条件
        
        Args:
            stock: 股票代码
            current_price: 当前价格
            
        Returns:
            bool: 是否触发止盈止损
        """
        # 获取该股票的买入记录
        buy_records = [t for t in self.performance.get('trades', []) 
                      if t['stock'] == stock and t['action'] == 'BUY']
        
        if not buy_records:
            return False
        
        # 获取最近一次买入价格
        last_buy = buy_records[-1]
        buy_price = last_buy['price']
        
        # 计算当前收益率
        current_return = current_price / buy_price - 1
        
        # 止损
        if current_return <= self.stop_loss:
            logger.info(f"{stock} 触发止损: 买入价 {buy_price}, 当前价 {current_price}, 收益率 {current_return:.2%}")
            return True
        
        # 止盈
        if current_return >= self.take_profit:
            logger.info(f"{stock} 触发止盈: 买入价 {buy_price}, 当前价 {current_price}, 收益率 {current_return:.2%}")
            return True
        
        return False
    
    def plot_results(self, save_dir=None):
        """绘制回测结果图表
        
        Args:
            save_dir: 保存图表的目录
        """
        if not self.performance:
            logger.error("没有回测结果，请先运行backtest()")
            return
        
        # 1. 绘制净值曲线
        net_value_df = pd.DataFrame(self.performance['daily_net_values'])
        plt.figure(figsize=(14, 6))
        plt.plot(net_value_df['date'], net_value_df['net_value'])
        plt.title('策略净值曲线')
        plt.grid(True)
        
        # 保存图表
        if save_dir:
            save_path = os.path.join(save_dir, "net_value_curve.png")
            plt.savefig(save_path)
            logger.info(f"保存净值曲线图: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # 2. 绘制回撤曲线
        net_value_df['cummax'] = net_value_df['net_value'].cummax()
        net_value_df['drawdown'] = net_value_df['net_value'] / net_value_df['cummax'] - 1
        
        plt.figure(figsize=(14, 6))
        plt.plot(net_value_df['date'], net_value_df['drawdown'])
        plt.title('策略回撤曲线')
        plt.grid(True)
        
        # 保存图表
        if save_dir:
            save_path = os.path.join(save_dir, "drawdown_curve.png")
            plt.savefig(save_path)
            logger.info(f"保存回撤曲线图: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # 3. 绘制个股收益贡献
        trades_df = pd.DataFrame(self.performance['trades'])
        if not trades_df.empty:
            stock_returns = {}
            
            for stock in trades_df['stock'].unique():
                stock_trades = trades_df[trades_df['stock'] == stock]
                buy_amount = stock_trades[stock_trades['action'] == 'BUY']['amount'].sum()
                sell_amount = stock_trades[stock_trades['action'] == 'SELL']['amount'].sum()
                
                # 考虑未平仓的持仓
                positions = {s: 0 for s in trades_df['stock'].unique()}
                for _, row in stock_trades.iterrows():
                    if row['action'] == 'BUY':
                        positions[stock] += row['shares']
                    else:
                        positions[stock] -= row['shares']
                
                # 如果有未平仓的持仓，估算其价值
                if positions[stock] > 0:
                    # 获取最后一个交易日价格
                    last_price = stock_trades.iloc[-1]['price']
                    sell_amount += positions[stock] * last_price
                
                if buy_amount > 0:
                    stock_returns[stock] = sell_amount / buy_amount - 1
                else:
                    stock_returns[stock] = 0
            
            # 绘制个股收益贡献
            if stock_returns:
                plt.figure(figsize=(14, 6))
                plt.bar(stock_returns.keys(), [r * 100 for r in stock_returns.values()])
                plt.title('个股收益贡献（%）')
                plt.grid(True, axis='y')
                plt.xticks(rotation=45)
                
                # 保存图表
                if save_dir:
                    save_path = os.path.join(save_dir, "stock_returns.png")
                    plt.savefig(save_path)
                    logger.info(f"保存个股收益贡献图: {save_path}")
                else:
                    plt.show()
                
                plt.close()
        
        # 4. 保存交易记录
        if save_dir and not trades_df.empty:
            save_path = os.path.join(save_dir, "trade_records.csv")
            trades_df.to_csv(save_path, index=False)
            logger.info(f"保存交易记录: {save_path}")
        
        # 5. 保存回测结果摘要
        if save_dir:
            with open(os.path.join(save_dir, "performance_summary.txt"), 'w') as f:
                f.write(f"回测区间: {self.start_date} 至 {self.end_date}\n")
                f.write(f"初始资金: {self.performance['initial_capital']}\n")
                f.write(f"最终市值: {self.performance['final_value']:.2f}\n")
                f.write(f"总收益率: {self.performance['total_return']:.2%}\n")
                f.write(f"年化收益率: {self.performance['annual_return']:.2%}\n")
                f.write(f"最大回撤: {self.performance['max_drawdown']:.2%}\n")
                f.write(f"夏普比率: {self.performance['sharpe_ratio']:.2f}\n")
                f.write(f"交易次数: {len(self.performance['trades'])}\n")
                
                f.write("\n选择的行业:\n")
                for industry in self.select_industries():
                    f.write(f"- {industry}\n")
                
                f.write("\n选择的股票:\n")
                for stock in self.selected_stocks:
                    f.write(f"- {stock}\n")
            
            logger.info(f"保存回测结果摘要: {os.path.join(save_dir, 'performance_summary.txt')}")

def main():
    """主函数"""
    logger.info("开始运行行业动量策略回测...")
    
    # 设置回测日期
    start_date = "2024-06-01"
    end_date = "2024-12-20"
    
    # 分析结果文件
    industry_analysis_file = "results/enhanced_analysis/industry_analysis.csv"
    stock_analysis_file = "results/enhanced_analysis/analysis_results.csv"
    
    # 创建策略实例
    strategy = IndustryMomentumStrategy(
        start_date=start_date,
        end_date=end_date,
        industry_analysis_file=industry_analysis_file,
        stock_analysis_file=stock_analysis_file
    )
    
    # 运行回测
    strategy.backtest()
    
    # 绘制结果
    strategy.plot_results(save_dir=results_dir)
    
    logger.info("行业动量策略回测完成")

if __name__ == "__main__":
    print("=" * 80)
    print(" 行业动量策略回测 ".center(80, "="))
    print("=" * 80)
    
    # 运行主函数
    main()
    
    print("=" * 80)
    print(" 回测完成 ".center(80, "="))
    print("=" * 80) 