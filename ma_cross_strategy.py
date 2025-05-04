"""
均线交叉策略模块
提供基于均线交叉的交易策略，实现金叉/死叉信号生成及回测
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from pathlib import Path
import tushare as ts
import warnings
warnings.filterwarnings('ignore')
# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))
# 导入项目配置
try:
    from src.enhanced.config.settings import TUSHARE_TOKEN, LOG_DIR, DATA_DIR, RESULTS_DIR
except ImportError:
    # 设置默认配置
    TUSHARE_TOKEN = ""
    LOG_DIR = "./logs"
    DATA_DIR = "./data"
    RESULTS_DIR = "./results"
# 确保目录存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "ma_charts"), exist_ok=True)
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"ma_strategy_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# 设置Tushare
if not TUSHARE_TOKEN:
    # 直接在代码中设置Token（如果配置文件中没有设置）
    TUSHARE_TOKEN = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
else:
    pro = None
    logger.warning("未设置Tushare Token，将使用本地数据")
class MACrossStrategy:
    """均线交叉策略类，实现金叉/死叉信号生成及回测"""
    def __init__(self, use_tushare=True):
        """初始化均线交叉策略"""
        self.use_tushare = use_tushare
        if use_tushare and not TUSHARE_TOKEN:
            logger.warning("未设置Tushare Token，将使用本地数据")
            self.use_tushare = False
        self.data_cache = {}  # 数据缓存
    def get_stock_list(self, industry=None):
        """获取股票列表，可按行业筛选"""
        if self.use_tushare:
            try:
                # 获取所有股票列表
                stocks = pro.stock_basic(exchange='', list_status='L',
                                         fields='ts_code,symbol,name,area,industry,list_date')
                # 行业筛选
                if industry and industry != "全部":
                    stocks = stocks[stocks['industry'] == industry]
                return stocks
            except Exception as e:
                logger.error(f"从Tushare获取股票列表失败: {str(e)}")
                # 尝试使用备用数据
                return self._get_local_stock_list(industry)
        else:
            return self._get_local_stock_list(industry)
    def _get_local_stock_list(self, industry=None):
        """从本地获取股票列表（备用方法）"""
        try:
            # 尝试从本地文件读取
            stock_file = os.path.join(DATA_DIR, "stock_list.csv")
            if os.path.exists(stock_file):
                stocks = pd.read_csv(stock_file)
                # 行业筛选
                if industry and industry != "全部":
                    stocks = stocks[stocks['industry'] == industry]
                return stocks
            else:
                # 创建模拟数据
                logger.warning("无法获取股票列表，创建模拟数据")
                data = {
                    'ts_code': ['000001.SZ', '000002.SZ', '000063.SZ', '000066.SZ', '000333.SZ'],
                    'symbol': ['000001', '000002', '000063', '000066', '000333'],
                    'name': ['平安银行', '万科A', '中兴通讯', '中国长城', '美的集团'],
                    'area': ['深圳', '深圳', '深圳', '深圳', '广东'],
                    'industry': ['银行', '房地产', '通信', '计算机', '家电'],
                    'list_date': ['19910403', '19910129', '19971118', '19970620', '20130918']
                }
                stocks = pd.DataFrame(data)
                # 行业筛选
                if industry and industry != "全部":
                    stocks = stocks[stocks['industry'] == industry]
                return stocks
        except Exception as e:
            logger.error(f"从本地获取股票列表失败: {str(e)}")
            return pd.DataFrame(columns=['ts_code', 'symbol', 'name', 'area', 'industry', 'list_date'])
    def get_stock_daily_data(self, ts_code, start_date=None, end_date=None):
        """获取股票日线数据"""
        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        # 检查缓存
        cache_key = f"{ts_code}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        if self.use_tushare:
            try:
                # 从Tushare获取日线数据
                df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df.empty:
                    # 尝试使用备用API
                    df = ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if not df.empty:
                    # 确保日期列为索引并按日期排序
                    if 'trade_date' in df.columns:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        df.sort_values('trade_date', inplace=True)
                        df.set_index('trade_date', inplace=True)
                    # 缓存数据
                    self.data_cache[cache_key] = df
                    return df
                else:
                    logger.warning(f"获取{ts_code}的日线数据为空")
                    return self._get_local_stock_data(ts_code, start_date, end_date)
            except Exception as e:
                logger.error(f"从Tushare获取{ts_code}的日线数据失败: {str(e)}")
                return self._get_local_stock_data(ts_code, start_date, end_date)
        else:
            return self._get_local_stock_data(ts_code, start_date, end_date)
    def _get_local_stock_data(self, ts_code, start_date=None, end_date=None):
        """从本地获取股票日线数据（备用方法）"""
        try:
            # 尝试从本地文件读取
            stock_file = os.path.join(DATA_DIR, f"{ts_code.replace('.', '_')}.csv")
            if os.path.exists(stock_file):
                df = pd.read_csv(stock_file)
                # 处理日期
                if 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    df.sort_values('trade_date', inplace=True)
                    # 日期筛选
                    if start_date:
                        start_date = pd.to_datetime(start_date)
                        df = df[df['trade_date'] >= start_date]
                    if end_date:
                        end_date = pd.to_datetime(end_date)
                        df = df[df['trade_date'] <= end_date]
                    df.set_index('trade_date', inplace=True)
                return df
            else:
                # 创建模拟数据
                logger.warning(f"无法获取{ts_code}的日线数据，创建模拟数据")
                # 生成日期序列
                if not end_date:
                    end_date = datetime.now().strftime('%Y%m%d')
                if not start_date:
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                date_range = pd.date_range(start=start_date, end=end_date, freq='B')
                # 生成模拟价格和交易量
                n = len(date_range)
                seed = sum(ord(c) for c in ts_code)  # 根据股票代码生成种子
                np.random.seed(seed)
                # 创建基础价格序列 (基于随机游走)
                base_price = 50 + np.random.rand() * 50  # 基础价格50-100之间随机
                price_changes = np.random.normal(0, 1, n) * 0.5  # 每日价格变化
                # 添加一些趋势和周期性
                trend = np.linspace(0, 5, n) * np.random.choice([-1, 1]) * 0.2  # 添加轻微趋势
                cycle = np.sin(np.linspace(0, 4*np.pi, n)) * 5  # 添加周期波动
                # 合并所有因素
                cumulative_changes = np.cumsum(price_changes) + trend + cycle
                close = base_price + cumulative_changes
                close = np.maximum(close, base_price * 0.5)  # 确保价格不会太低
                # 根据收盘价生成其他价格
                daily_volatility = np.random.uniform(0.01, 0.03, n)  # 每日波动率
                # 确保价格关系合理
                high = close * (1 + daily_volatility)
                low = close * (1 - daily_volatility)
                # 随机开盘价，但确保在最高价和最低价之间
                rand_factor = np.random.rand(n)  # 0到1之间的随机数
                open_price = low + rand_factor * (high - low)
                # 重新检查确保high > low，有时浮点数计算可能导致问题
                high = np.maximum(high, np.maximum(close, open_price) * 1.001)
                low = np.minimum(low, np.minimum(close, open_price) * 0.999)
                # 确保所有价格都是正数
                high = np.maximum(high, 0.01)
                low = np.maximum(low, 0.01)
                close = np.maximum(close, 0.01)
                open_price = np.maximum(open_price, 0.01)
                # 成交量与价格变化正相关
                price_change = np.abs(np.diff(close, prepend=close[0]))
                vol_base = np.random.normal(1e6, 2e5, n)  # 基础成交量
                vol = vol_base * (1 + 3 * price_change / close)  # 价格变化大时，成交量增加
                vol = np.abs(vol)  # 确保成交量为正
                # 计算成交额和其他指标
                amount = vol * close  # 成交额
                change = np.diff(close, prepend=close[0])  # 价格变化
                pct_chg = change / np.roll(close, 1) * 100  # 涨跌幅
                pct_chg[0] = 0  # 第一天的涨跌幅设为0
                # 创建DataFrame
                data = {
                    'ts_code': [ts_code] * n,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'vol': vol,
                    'amount': amount,
                    'change': change,
                    'pct_chg': pct_chg
                }
                df = pd.DataFrame(data, index=date_range)
                # 恢复随机种子
                np.random.seed(None)
                # 保存到本地文件，便于下次使用
                os.makedirs(os.path.dirname(stock_file) if os.path.dirname(stock_file) else '.', exist_ok=True)
                df_to_save = df.reset_index()
                df_to_save.rename(columns={'index': 'trade_date'}, inplace=True)
                df_to_save.to_csv(stock_file, index=False)
                logger.info(f"已将模拟数据保存至: {stock_file}")
                return df
        except Exception as e:
            logger.error(f"从本地获取{ts_code}的日线数据失败: {str(e)}")
            return pd.DataFrame()
    def calculate_ma_signals(self, data, short_ma=5, long_ma=20):
        """计算均线交叉信号"""
        if data.empty:
            return data
        df = data.copy()
        # 计算移动平均线
        df[f'ma{short_ma}'] = df['close'].rolling(window=short_ma).mean()
        df[f'ma{long_ma}'] = df['close'].rolling(window=long_ma).mean()
        # 计算信号
        df['signal'] = 0
        # 金叉信号：短期均线上穿长期均线
        df.loc[(df[f'ma{short_ma}'] > df[f'ma{long_ma}']) &
               (df[f'ma{short_ma}'].shift(1) <= df[f'ma{long_ma}'].shift(1)), 'signal'] = 1
        # 死叉信号：短期均线下穿长期均线
        df.loc[(df[f'ma{short_ma}'] < df[f'ma{long_ma}']) &
               (df[f'ma{short_ma}'].shift(1) >= df[f'ma{long_ma}'].shift(1)), 'signal'] = -1
        # 持仓状态：1表示持有，0表示空仓
        df['position'] = 0
        # 第一个信号产生之前的位置设为0
        first_valid_index = df[df['signal'] != 0].first_valid_index()
        if first_valid_index:
            position = 0
            for i, row in df.loc[first_valid_index:].iterrows():
                if row['signal'] == 1:
                    position = 1
                elif row['signal'] == -1:
                    position = 0
                df.loc[i, 'position'] = position
        # 删除NaN值
        df.dropna(inplace=True)
        return df
    def backtest_strategy(self, data, initial_capital=100000, stop_loss_pct=0.05):
        """回测均线交叉策略"""
        if data.empty or 'position' not in data.columns:
            logger.warning("数据为空或未包含持仓信息，无法进行回测")
            return None
        # 拷贝数据，避免修改原始数据
        df = data.copy()
        # 初始化回测数据
        df['capital'] = initial_capital  # 初始都设为初始资金
        df['shares'] = 0  # 持仓数量
        df['cost_price'] = 0.0  # 持仓成本
        df['stop_loss'] = False  # 是否触发止损
        df['strategy_return'] = 0.0  # 策略收益率
        # 股票价格变化
        if 'pct_change' not in df.columns:
            df['pct_change'] = df['close'].pct_change()
            df['pct_change'].fillna(0, inplace=True)
        # 确保索引是有序的
        df = df.sort_index()
        # 模拟交易
        position = 0  # 当前持仓状态
        shares = 0  # 持有股数
        cost_price = 0  # 持仓成本
        capital = initial_capital  # 当前资金
        # 循环模拟每个交易日
        for i in range(len(df)):
            current_date = df.index[i]
            try:
                # 当前价格
                current_price = df.loc[current_date, 'close']
                # 当前持仓信号
                current_position = df.loc[current_date, 'position']
                # 前一个交易日的持仓信号
                prev_position = 0 if i == 0 else df.loc[df.index[i-1], 'position']
                # 检查是否触发止损
                stop_loss_triggered = False
                if position > 0 and shares > 0 and cost_price > 0:
                    if (current_price / cost_price - 1) < -stop_loss_pct:
                        stop_loss_triggered = True
                        current_position = 0  # 强制卖出
                        df.loc[current_date, 'stop_loss'] = True
                # 持仓变化
                if current_position != prev_position or stop_loss_triggered:
                    # 买入信号
                    if current_position > prev_position:
                        # 计算可买入的股数（整数）
                        new_shares = int(capital // current_price)
                        if new_shares > 0:
                            cost = new_shares * current_price
                            shares = new_shares
                            capital -= cost
                            cost_price = current_price
                    # 卖出信号
                    elif current_position < prev_position or stop_loss_triggered:
                        if shares > 0:
                            # 卖出所有股票
                            capital += shares * current_price
                            shares = 0
                            cost_price = 0
                # 更新当前持仓状态
                position = current_position
                # 计算当前总资产
                total_assets = capital + shares * current_price
                # 记录到DataFrame
                df.loc[current_date, 'shares'] = shares
                df.loc[current_date, 'cost_price'] = cost_price
                df.loc[current_date, 'capital'] = total_assets
                df.loc[current_date, 'strategy_return'] = (total_assets / initial_capital) - 1
            except Exception as e:
                logger.error(f"回测时出错，日期: {current_date}, 错误: {str(e)}")
        # 计算回测指标
        # 最大回撤
        df['peak'] = df['capital'].cummax()
        df['drawdown'] = (df['capital'] - df['peak']) / df['peak']
        max_drawdown = abs(df['drawdown'].min()) if not df['drawdown'].isna().all() else 0
        # 年化收益率
        days = (df.index[-1] - df.index[0]).days
        if days > 0:
            annual_return = (df['capital'].iloc[-1] / initial_capital) ** (365 / days) - 1
        else:
            annual_return = 0
        # 总收益率
        total_return = (df['capital'].iloc[-1] / initial_capital) - 1
        # 计算胜率
        trade_dates = df[df['position'] != df['position'].shift(1)].index
        if len(trade_dates) >= 2:
            buy_dates = []
            sell_dates = []
            for i in range(len(trade_dates)):
                date = trade_dates[i]
                if i == 0:
                    prev_pos = 0
                else:
                    prev_pos = df.loc[trade_dates[i-1]:date, 'position'].iloc[0]
                curr_pos = df.loc[date, 'position']
                if curr_pos > prev_pos:  # 买入点
                    buy_dates.append(date)
                elif curr_pos < prev_pos:  # 卖出点
                    if buy_dates:  # 确保有买入才记录卖出
                        sell_dates.append(date)
            # 计算每次交易的收益率
            trade_returns = []
            for i in range(min(len(buy_dates), len(sell_dates))):
                buy_price = df.loc[buy_dates[i], 'close']
                sell_price = df.loc[sell_dates[i], 'close']
                trade_return = sell_price / buy_price - 1
                trade_returns.append(trade_return)
            # 如果最后一个交易还未卖出，计算到最后一天的收益
            if len(buy_dates) > len(sell_dates):
                buy_price = df.loc[buy_dates[-1], 'close']
                sell_price = df.iloc[-1]['close']
                trade_return = sell_price / buy_price - 1
                trade_returns.append(trade_return)
            # 计算胜率
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0
        else:
            win_rate = 0
        # 当前信号
        latest_pos = df['position'].iloc[-1]
        current_signal = "买入" if latest_pos > 0 else "卖出"
        # 返回回测结果
        return {
            'backtest_data': df,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'current_signal': current_signal
        }
    def backtest_strategy_vectorized(self, data, initial_capital=100000, stop_loss_pct=0.05):
        """使用矢量化操作回测均线交叉策略 (优化速度版本)"""
        if data.empty or 'position' not in data.columns:
            logger.warning("数据为空或未包含持仓信息，无法进行回测")
            return None
            
        # 拷贝数据，避免修改原始数据
        df = data.copy()
        
        # 获取价格和仓位数据
        close_prices = df['close'].values
        positions = df['position'].values
        dates = df.index
        
        # 初始化数组
        n = len(df)
        capitals = np.zeros(n)
        shares_arr = np.zeros(n)
        cost_prices = np.zeros(n)
        stop_losses = np.zeros(n, dtype=bool)
        
        # 初始资金和状态
        capital = initial_capital
        shares = 0
        cost_price = 0
        
        # 使用numpy进行快速计算
        for i in range(n):
            current_price = close_prices[i]
            current_position = positions[i]
            prev_position = 0 if i == 0 else positions[i-1]
            
            # 检查是否触发止损
            stop_loss_triggered = False
            if shares > 0 and cost_price > 0:
                if (current_price / cost_price - 1) < -stop_loss_pct:
                    stop_loss_triggered = True
                    current_position = 0  # 强制卖出
                    stop_losses[i] = True
            
            # 持仓变化
            if current_position != prev_position or stop_loss_triggered:
                # 买入信号
                if current_position > prev_position:
                    # 计算可买入的股数（整数）
                    new_shares = int(capital // current_price)
                    if new_shares > 0:
                        cost = new_shares * current_price
                        shares = new_shares
                        capital -= cost
                        cost_price = current_price
                # 卖出信号
                elif current_position < prev_position or stop_loss_triggered:
                    if shares > 0:
                        # 卖出所有股票
                        capital += shares * current_price
                        shares = 0
                        cost_price = 0
            
            # 记录当前状态
            shares_arr[i] = shares
            cost_prices[i] = cost_price
            capitals[i] = capital + shares * current_price
        
        # 将结果写回DataFrame
        df['shares'] = shares_arr
        df['cost_price'] = cost_prices
        df['stop_loss'] = stop_losses
        df['capital'] = capitals
        df['strategy_return'] = (df['capital'] / initial_capital) - 1
        
        # 计算回测指标
        df['peak'] = df['capital'].cummax()
        df['drawdown'] = (df['capital'] - df['peak']) / df['peak']
        max_drawdown = abs(df['drawdown'].min()) if not df['drawdown'].isna().all() else 0
        
        # 年化收益率
        days = (dates[-1] - dates[0]).days
        if days > 0:
            annual_return = (capitals[-1] / initial_capital) ** (365 / days) - 1
        else:
            annual_return = 0
            
        # 总收益率
        total_return = (capitals[-1] / initial_capital) - 1
        
        # 计算胜率 - 优化计算方式，避免循环
        # 找出所有交易点
        position_changes = np.diff(positions, prepend=0)
        buy_points = position_changes > 0
        sell_points = position_changes < 0
        
        # 如果有买卖点，计算胜率
        if np.any(buy_points) and np.any(sell_points):
            buy_prices = close_prices[buy_points]
            # 对每个买入点，找到下一个卖出点
            trades = []
            buy_indices = np.where(buy_points)[0]
            sell_indices = np.where(sell_points)[0]
            
            for buy_idx in buy_indices:
                # 找到下一个卖出点
                next_sells = sell_indices[sell_indices > buy_idx]
                if len(next_sells) > 0:
                    sell_idx = next_sells[0]
                    trades.append(close_prices[sell_idx] / close_prices[buy_idx] - 1)
            
            # 如果最后一个交易还未卖出，计算到最后一天的收益
            if buy_indices[-1] > sell_indices[-1] if len(sell_indices) > 0 else True:
                trades.append(close_prices[-1] / close_prices[buy_indices[-1]] - 1)
                
            # 计算胜率
            win_rate = np.sum(np.array(trades) > 0) / len(trades) if trades else 0
        else:
            win_rate = 0
            
        # 当前信号
        latest_pos = positions[-1]
        current_signal = "买入" if latest_pos > 0 else "卖出"
        
        # 返回回测结果
        return {
            'backtest_data': df,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'current_signal': current_signal
        }
    def plot_ma_strategy_chart(self, data, backtest_result, stock_code, stock_name, save_path=None):
        """绘制均线交叉策略回测图表"""
        if data.empty or backtest_result is None:
            logger.warning(f"无法绘制{stock_code}的图表，数据为空")
            return False
        # 创建图表
        fig = plt.figure(figsize=(14, 10))
        # 设置网格
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        # 第一个子图：K线、均线和交易信号
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title(f"{stock_name}({stock_code}) 均线交叉策略回测", fontsize=15)
        # 绘制收盘价
        ax1.plot(data.index, data['close'], label='收盘价', color='black', alpha=0.75)
        # 获取均线列
        ma_cols = [col for col in data.columns if col.startswith('ma')]
        # 绘制均线
        for col in ma_cols:
            ax1.plot(data.index, data[col], label=col.upper(), alpha=0.75)
        # 绘制买入信号
        buy_signals = data[data['signal'] == 1]
        ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='red', s=100, label='买入信号')
        # 绘制卖出信号
        sell_signals = data[data['signal'] == -1]
        ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='green', s=100, label='卖出信号')
        # 绘制止损信号
        if 'stop_loss' in data.columns:
            stop_loss_signals = data[data['stop_loss'] == True]
            if not stop_loss_signals.empty:
                ax1.scatter(stop_loss_signals.index, stop_loss_signals['close'], marker='x', color='purple', s=100, label='止损信号')
        ax1.legend(loc='best')
        ax1.grid(True)
        # 第二个子图：持仓状态
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax2.fill_between(data.index, 0, data['position'], color='skyblue', alpha=0.5)
        ax2.set_ylabel('持仓')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['空仓', '持仓'])
        ax2.grid(True)
        # 第三个子图：资金曲线
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax3.plot(data.index, data['capital'], label='资金曲线', color='blue')
        ax3.set_ylabel('资金')
        ax3.legend(loc='best')
        ax3.grid(True)
        # 添加回测结果信息
        info_text = (
            f"总收益率: {backtest_result['total_return']:.2%}\n"
            f"年化收益率: {backtest_result['annual_return']:.2%}\n"
            f"最大回撤: {backtest_result['max_drawdown']:.2%}\n"
            f"胜率: {backtest_result['win_rate']:.2%}\n"
            f"当前信号: {backtest_result['current_signal']}"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.05, info_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        # 调整布局
        plt.tight_layout()
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            logger.info(f"已保存{stock_code}的图表到{save_path}")
            plt.close(fig)
            return True
        else:
            plt.show()
            return True
    def run_strategy(self, stock_list, short_ma=5, long_ma=20, initial_capital=100000, stop_loss_pct=0.05, sample_size=None):
        """对股票列表运行均线交叉策略"""
        results = []
        
        # 如果提供了sample_size，则仅分析指定数量的股票
        if sample_size and sample_size < len(stock_list):
            stock_list = stock_list.head(sample_size)
            
        total = len(stock_list)
        for idx, (_, stock) in enumerate(stock_list.iterrows()):
            try:
                ts_code = stock['ts_code']
                name = stock['name']
                industry = stock.get('industry', '')
                logger.info(f"策略进度: {idx+1}/{total} - 正在分析: {name}({ts_code})")
                
                # 获取日线数据
                data = self.get_stock_daily_data(ts_code)
                if data.empty:
                    logger.warning(f"无法获取{ts_code}的数据，跳过分析")
                    continue
                    
                # 计算均线和信号
                data = self.calculate_ma_signals(data, short_ma=short_ma, long_ma=long_ma)
                if data.empty:
                    logger.warning(f"计算{ts_code}的信号失败，跳过分析")
                    continue
                    
                # 回测策略（使用优化版本）
                try:
                    backtest_result = self.backtest_strategy_vectorized(data, initial_capital=initial_capital, stop_loss_pct=stop_loss_pct)
                    if backtest_result is None:
                        logger.warning(f"回测{ts_code}的策略失败，跳过分析")
                        continue
                        
                    # 获取最新价格
                    latest = data.iloc[-1]
                    
                    # 转换信号为标准文本格式
                    signal_text = "无信号"
                    raw_signal = backtest_result['current_signal']
                    if raw_signal == "买入":
                        signal_text = "买入信号"
                    elif raw_signal == "卖出":
                        signal_text = "卖出信号"
                    elif latest['position'] > 0:
                        signal_text = "持有多头"
                    elif latest['position'] == 0 and latest.get('signal') == 0:
                        signal_text = "观望"
                        
                    # 保存结果
                    result = {
                        'ts_code': ts_code,
                        'name': name,
                        'industry': industry,
                        'close': latest['close'],
                        'current_signal': signal_text,
                        'short_ma': short_ma,
                        'long_ma': long_ma,
                        'total_return': backtest_result['total_return'],
                        'annual_return': backtest_result['annual_return'],
                        'max_drawdown': backtest_result['max_drawdown'],
                        'win_rate': backtest_result['win_rate'],
                        'data': data,
                        'backtest_result': backtest_result
                    }
                    results.append(result)
                    
                    # 生成图表
                    chart_path = os.path.join(RESULTS_DIR, "ma_charts", f"{ts_code}_ma_cross.png")
                    self.plot_ma_strategy_chart(data, backtest_result, ts_code, name, save_path=chart_path)
                    
                except Exception as e:
                    logger.error(f"回测{ts_code}的策略失败: {str(e)}")
                    # 简化的结果，包含基本信息但没有回测数据
                    result = {
                        'ts_code': ts_code,
                        'name': name,
                        'industry': industry,
                        'close': data.iloc[-1]['close'] if not data.empty else 0,
                        'current_signal': '无信号',
                        'total_return': 0.0,
                        'annual_return': 0.0,
                        'max_drawdown': 0.0,
                        'win_rate': 0.0
                    }
                    results.append(result)
            except Exception as e:
                logger.error(f"分析{stock.get('name', '')}({stock.get('ts_code', '')})时出错: {str(e)}")
                continue
                
        # 按收益率排序
        results.sort(key=lambda x: x['total_return'], reverse=True)
        
        # 将结果保存为CSV
        if results:
            result_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'data' and k != 'backtest_result'}
                                    for r in results])
            csv_path = os.path.join(RESULTS_DIR, f"ma_strategy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"已将策略结果保存至: {csv_path}")
            
        return results
# 运行测试
if __name__ == "__main__":
    # 测试
    strategy = MACrossStrategy(use_tushare=True)
    # 获取股票列表
    stocks = strategy.get_stock_list()
    print(f"获取到 {len(stocks)} 支股票")
    # 分析前10支股票
    results = strategy.run_strategy(stocks.head(10), short_ma=5, long_ma=20)
    # 输出结果
    for r in results:
        print(f"{r['name']}({r['ts_code']}): 信号={r['current_signal']}, 收益率={r['total_return']:.2%}, 最大回撤={r['max_drawdown']:.2%}")