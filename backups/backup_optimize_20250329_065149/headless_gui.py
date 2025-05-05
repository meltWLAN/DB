#!/usr/bin/env python3
"""
股票分析系统 - 无界面分析版本
使用matplotlib作为后端，避免tkinter问题
"""
import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as pl
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
# 配置日志
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/headless_gui_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HeadlessGUI")
# 确保目录存在
for dirname in ["data", "logs", "results", "charts"]:
    os.makedirs(os.path.join(os.getcwd(), dirname), exist_ok=True)
class StockAnalyzer:
    """股票分析类"""
    def __init__(self):
        """初始化分析器"""
        self.data_dir = "data"
        self.results_dir = "results"
        self.charts_dir = "charts"
        logger.info("初始化股票分析器")
    def get_stock_list(self):
        """获取股票列表"""
        stock_files = list(Path(self.data_dir).glob("*.csv"))
        return [f.stem for f in stock_files]
    def read_stock_data(self, stock_code):
        """读取股票数据"""
        try:
            file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
            if not os.path.exists(file_path):
                logger.warning(f"找不到股票数据文件: {file_path}")
                return None
            df = pd.read_csv(file_path)
            # 确保日期列是日期类型
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.error(f"读取股票数据出错: {str(e)}")
            return None
    def calculate_technical_indicators(self, df):
        """计算技术指标"""
        try:
            # 检查必要的列是否存在
            required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"缺少必要的列: {missing_columns}")
                # 检查是否有对应的小写列名
                for col in missing_columns:
                    lower_col = col.lower()
                    if lower_col in df.columns:
                        df[col] = df[lower_col]
            # 确保所有必要的列都存在
            if not all(col in df.columns for col in required_columns):
                available_cols = ', '.join(df.columns)
                logger.warning(f"无法计算技术指标，缺少必要的列。可用列: {available_cols}")
                return df
            # 计算移动平均线
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA10'] = df['Close'].rolling(window=10).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA60'] = df['Close'].rolling(window=60).mean()
            # 计算MACD
            df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['Histogram'] = df['MACD'] - df['Signal']
            # 计算RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            # 计算布林带
            df['MB'] = df['Close'].rolling(window=20).mean()
            df['STD'] = df['Close'].rolling(window=20).std()
            df['UB'] = df['MB'] + 2 * df['STD']
            df['LB'] = df['MB'] - 2 * df['STD']
            # 计算成交量变化
            df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_Change'] = df['Volume'] / df['Volume_MA5']
            return df
        except Exception as e:
            logger.error(f"计算技术指标出错: {str(e)}")
            return df
    def analyze_stock(self, stock_code):
        """分析单只股票"""
        logger.info(f"开始分析股票: {stock_code}")
        # 读取数据
        df = self.read_stock_data(stock_code)
        if df is None or df.empty:
            logger.warning(f"没有有效的数据用于分析: {stock_code}")
            return None
        # 计算技术指标
        df = self.calculate_technical_indicators(df)
        # 计算动量得分
        df['Momentum_Score'] = self.calculate_momentum_score(df)
        # 计算均线交叉信号
        df['MA_Cross_Signal'] = self.calculate_ma_cross_signal(df)
        # 保存结果
        result_file = os.path.join(self.results_dir, f"{stock_code}_analysis.csv")
        df.to_csv(result_file)
        logger.info(f"分析结果已保存到: {result_file}")
        # 绘制图表
        self.plot_stock_chart(df, stock_code)
        return df
    def calculate_momentum_score(self, df):
        """计算动量得分"""
        try:
            if 'Close' not in df.columns:
                return pd.Series(0, index=df.index)
            # 计算价格动量
            price_change_5d = df['Close'].pct_change(5) * 100
            price_change_20d = df['Close'].pct_change(20) * 100
            # 计算相对强弱
            rsi_signal = (df['RSI'] - 50) / 50 if 'RSI' in df.columns else 0
            # 计算MACD信号
            macd_signal = df['Histogram'] / df['Close'] * 100 if 'Histogram' in df.columns else 0
            # 计算成交量信号
            volume_signal = (df['Volume_Change'] - 1) * 10 if 'Volume_Change' in df.columns else 0
            # 计算总得分 (满分100)
            momentum_score = (
                price_change_5d * 2 +  # 权重20%
                price_change_20d * 1 +  # 权重10%
                rsi_signal * 30 +  # 权重30%
                macd_signal * 20 +  # 权重20%
                volume_signal * 20  # 权重20%
            )
            # 归一化到0-100
            score = momentum_score.clip(0, 100)
            return score
        except Exception as e:
            logger.error(f"计算动量得分出错: {str(e)}")
            return pd.Series(0, index=df.index)
    def calculate_ma_cross_signal(self, df):
        """计算均线交叉信号"""
        try:
            if 'MA5' not in df.columns or 'MA20' not in df.columns:
                return pd.Series(0, index=df.index)
            # 计算短期均线与长期均线的交叉
            df['MA5_Over_MA20'] = df['MA5'] > df['MA20']
            # 计算金叉和死叉
            signal = pd.Series(0, index=df.index)
            for i in range(1, len(df)):
                if df['MA5_Over_MA20'].iloc[i] and not df['MA5_Over_MA20'].iloc[i-1]:
                    # 金叉信号
                    signal.iloc[i] = 1
                elif not df['MA5_Over_MA20'].iloc[i] and df['MA5_Over_MA20'].iloc[i-1]:
                    # 死叉信号
                    signal.iloc[i] = -1
            return signal
        except Exception as e:
            logger.error(f"计算均线交叉信号出错: {str(e)}")
            return pd.Series(0, index=df.index)
    def plot_stock_chart(self, df, stock_code):
        """绘制股票图表"""
        try:
            # 创建一个4x1的子图布局
            fig = plt.figure(figsize=(14, 10))
            # 绘制价格和均线
            ax1 = plt.subplot(4, 1, 1)
            ax1.plot(df.index, df['Close'], label='收盘价', color='black')
            if 'MA5' in df.columns:
                ax1.plot(df.index, df['MA5'], label='MA5', color='blue')
            if 'MA20' in df.columns:
                ax1.plot(df.index, df['MA20'], label='MA20', color='red')
            if 'MA60' in df.columns:
                ax1.plot(df.index, df['MA60'], label='MA60', color='green')
            # 添加布林带
            if 'UB' in df.columns and 'LB' in df.columns:
                ax1.fill_between(df.index, df['UB'], df['LB'], color='gray', alpha=0.2)
            ax1.set_title(f'{stock_code} 价格走势')
            ax1.set_ylabel('价格')
            ax1.legend(loc='upper left')
            ax1.grid(True)
            # 绘制成交量
            ax2 = plt.subplot(4, 1, 2, sharex=ax1)
            ax2.bar(df.index, df['Volume'], color='gray', alpha=0.7)
            if 'Volume_MA5' in df.columns:
                ax2.plot(df.index, df['Volume_MA5'], color='red', linewidth=1.5)
            ax2.set_ylabel('成交量')
            ax2.grid(True)
            # 绘制MACD
            ax3 = plt.subplot(4, 1, 3, sharex=ax1)
            if all(col in df.columns for col in ['MACD', 'Signal', 'Histogram']):
                ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
                ax3.plot(df.index, df['Signal'], label='Signal', color='red')
                ax3.bar(df.index, df['Histogram'], color='gray', alpha=0.7)
                ax3.set_ylabel('MACD')
                ax3.legend(loc='upper left')
                ax3.grid(True)
            # 绘制RSI
            ax4 = plt.subplot(4, 1, 4, sharex=ax1)
            if 'RSI' in df.columns:
                ax4.plot(df.index, df['RSI'], label='RSI', color='purple')
                ax4.axhline(y=70, color='red', linestyle='--')
                ax4.axhline(y=30, color='green', linestyle='--')
                ax4.set_ylim(0, 100)
                ax4.set_ylabel('RSI')
                ax4.grid(True)
            # 设置x轴格式
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            fig.autofmt_xdate()
            plt.tight_layout()
            # 保存图表
            chart_file = os.path.join(self.charts_dir, f"{stock_code}_chart.png")
            plt.savefig(chart_file, dpi=100)
            plt.close()
            logger.info(f"图表已保存到: {chart_file}")
            return chart_file
        except Exception as e:
            logger.error(f"绘制图表出错: {str(e)}")
            return None
    def analyze_multiple_stocks(self, stock_codes=None, limit=10):
        """分析多只股票"""
        if stock_codes is None:
            stock_codes = self.get_stock_list()
            if limit > 0 and limit < len(stock_codes):
                stock_codes = stock_codes[:limit]
        logger.info(f"开始分析 {len(stock_codes)} 只股票")
        results = []
        for stock_code in stock_codes:
            df = self.analyze_stock(stock_code)
            if df is not None and not df.empty:
                # 获取最新数据
                latest = df.iloc[-1]
                result = {
                    'Stock': stock_code,
                    'Price': latest['Close'] if 'Close' in latest else 0,
                    'Change_1d': df['Close'].pct_change(1).iloc[-1] * 100 if 'Close' in df else 0,
                    'Change_5d': df['Close'].pct_change(5).iloc[-1] * 100 if 'Close' in df else 0,
                    'RSI': latest['RSI'] if 'RSI' in latest else 0,
                    'MACD': latest['MACD'] if 'MACD' in latest else 0,
                    'Signal': latest['Signal'] if 'Signal' in latest else 0,
                    'Momentum_Score': latest['Momentum_Score'] if 'Momentum_Score' in latest else 0,
                    'MA_Cross_Signal': latest['MA_Cross_Signal'] if 'MA_Cross_Signal' in latest else 0
                }
                results.append(result)
        # 创建结果DataFrame
        if results:
            results_df = pd.DataFrame(results)
            # 保存结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(self.results_dir, f"analysis_results_{timestamp}.csv")
            results_df.to_csv(result_file, index=False)
            logger.info(f"综合分析结果已保存到: {result_file}")
            # 按动量得分排序
            results_df = results_df.sort_values('Momentum_Score', ascending=False)
            return results_df
        else:
            logger.warning("没有产生任何分析结果")
            return pd.DataFrame()
    def generate_sample_data(self, num_stocks=5):
        """生成示例数据"""
        logger.info(f"生成 {num_stocks} 只股票的示例数据")
        # 生成日期范围（过去一年的交易日）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        for i in range(num_stocks):
            stock_code = f"SAMPLE{i+1:03d}"
            logger.info(f"生成 {stock_code} 的数据")
            # 设置随机种子以确保可重复性
            np.random.seed(i + 42)
            # 生成起始价格
            base_price = np.random.uniform(50, 200)
            # 生成价格序列
            prices = [base_price]
            for _ in range(1, len(dates)):
                # 添加随机波动
                daily_return = np.random.normal(0.0002, 0.018)  # 均值略大于0，产生上升趋势
                price = prices[-1] * (1 + daily_return)
                prices.append(price)
            prices = np.array(prices)
            # 生成开盘价、最高价、最低价
            opens = prices * (1 + np.random.normal(0, 0.005, len(prices)))
            highs = np.maximum(prices, opens) * (1 + np.random.uniform(0, 0.015, len(prices)))
            lows = np.minimum(prices, opens) * (1 - np.random.uniform(0, 0.015, len(prices)))
            # 生成成交量
            volume_base = np.random.randint(100000, 1000000)
            volumes = np.random.normal(volume_base, volume_base * 0.3, len(dates))
            volumes = np.maximum(volumes, 10000)  # 确保成交量为正
            # 创建DataFrame
            df = pd.DataFrame({
                'Date': dates,
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': prices,
                'Volume': volumes.astype(int)
            })
            # 保存到CSV
            file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
            df.to_csv(file_path, index=False)
            logger.info(f"示例数据已保存到: {file_path}")
        return True
def print_menu():
    """打印菜单"""
    print("\n" + "=" * 50)
    print(" 股票分析系统 - 无界面版")
    print("=" * 50)
    print("1. 生成示例数据")
    print("2. 分析单只股票")
    print("3. 批量分析股票")
    print("4. 查看可用股票")
    print("5. 查看分析结果")
    print("6. 退出")
    print("-" * 50)
    return input("请选择操作 [1-6]: ")
def main():
    """主函数"""
    print("=" * 60)
    print("股票分析系统 - 无界面版 (使用matplotlib后端)")
    print("=" * 60)
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"工作目录: {os.getcwd()}")
    print("-" * 60)
    analyzer = StockAnalyzer()
    while True:
        choice = print_menu()
        if choice == '1':
            num_stocks = input("请输入要生成的股票数量 [默认5]: ").strip()
            num_stocks = int(num_stocks) if num_stocks.isdigit() else 5
            analyzer.generate_sample_data(num_stocks)
            print(f"已生成 {num_stocks} 只股票的示例数据")
        elif choice == '2':
            stocks = analyzer.get_stock_list()
            if not stocks:
                print("没有可用的股票数据，请先生成示例数据")
                continue
            print("\n可用的股票:")
            for i, stock in enumerate(stocks):
                print(f"{i+1}. {stock}")
            stock_idx = input("\n请选择要分析的股票 [1-{}]: ".format(len(stocks)))
            try:
                idx = int(stock_idx) - 1
                if 0 <= idx < len(stocks):
                    stock_code = stocks[idx]
                    print(f"\n开始分析 {stock_code}...")
                    df = analyzer.analyze_stock(stock_code)
                    if df is not None:
                        print(f"分析完成，结果已保存")
                else:
                    print("无效的选择")
            except ValueError:
                print("请输入有效的数字")
        elif choice == '3':
            stocks = analyzer.get_stock_list()
            if not stocks:
                print("没有可用的股票数据，请先生成示例数据")
                continue
            limit = input(f"请输入要分析的股票数量 [默认全部 {len(stocks)}]: ").strip()
            limit = int(limit) if limit.isdigit() else len(stocks)
            print(f"\n开始批量分析 {min(limit, len(stocks))} 只股票...")
            results = analyzer.analyze_multiple_stocks(limit=limit)
            if not results.empty:
                print("\n分析完成，前10只表现最佳的股票:")
                print(results.head(10))
        elif choice == '4':
            stocks = analyzer.get_stock_list()
            if not stocks:
                print("没有可用的股票数据，请先生成示例数据")
                continue
            print(f"\n可用的股票列表 (共 {len(stocks)} 只):")
            for i in range(0, len(stocks), 5):
                batch = stocks[i:i+5]
                print(", ".join(batch))
        elif choice == '5':
            result_files = list(Path(analyzer.results_dir).glob("*.csv"))
            if not result_files:
                print("没有可用的分析结果")
                continue
            print("\n可用的分析结果:")
            for i, file in enumerate(result_files):
                print(f"{i+1}. {file.name}")
            file_idx = input("\n请选择要查看的结果文件 [1-{}]: ".format(len(result_files)))
            try:
                idx = int(file_idx) - 1
                if 0 <= idx < len(result_files):
                    file_path = result_files[idx]
                    results = pd.read_csv(file_path)
                    print(f"\n文件 {file_path.name} 的内容 (前10行):")
                    print(results.head(10))
                else:
                    print("无效的选择")
            except ValueError:
                print("请输入有效的数字")
            except Exception as e:
                print(f"读取结果文件出错: {str(e)}")
        elif choice == '6':
            print("退出程序")
            break
        else:
            print("无效的选择，请重试")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
        print(f"程序运行出错: {str(e)}")
    print("\n程序已退出")