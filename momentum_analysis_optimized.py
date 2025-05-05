"""
动量分析模块 (优化版本)
提供股票动量分析的相关功能，包括技术指标计算、筛选和评分
支持多进程处理和缓存优化，加快分析速度
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
from multiprocessing import Pool, cpu_count
import functools
import pickle
import hashlib

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
os.makedirs(os.path.join(RESULTS_DIR, "charts"), exist_ok=True)

# 添加缓存目录
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 配置日志
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

# 缓存装饰器
def disk_cache(func):
    """简单的磁盘缓存装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 创建缓存键
        key_parts = [func.__name__]
        key_parts.extend([str(arg) for arg in args])
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key = "_".join(key_parts)
        # 使用MD5哈希作为文件名
        hash_obj = hashlib.md5(key.encode())
        cache_file = os.path.join(CACHE_DIR, f"{hash_obj.hexdigest()}.pkl")
        
        # 检查缓存是否存在
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                # 如果加载缓存失败，继续执行原始函数
                pass
                
        # 执行原始函数
        result = func(*args, **kwargs)
        
        # 保存结果到缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"缓存保存失败: {str(e)}")
            
        return result
    return wrapper

class MomentumAnalyzer:
    """动量分析器类，提供动量分析相关功能"""
    def __init__(self, use_tushare=True, use_multiprocessing=True, workers=None):
        """初始化动量分析器"""
        self.use_tushare = use_tushare
        if use_tushare and not TUSHARE_TOKEN:
            logger.warning("未设置Tushare Token，将使用本地数据")
            self.use_tushare = False
        self.data_cache = {}  # 数据缓存
        
        # 多进程设置
        self.use_multiprocessing = use_multiprocessing
        self.workers = workers if workers is not None else max(1, cpu_count() - 1)
        
    @disk_cache
    def get_stock_daily_data_cached(self, ts_code, start_date=None, end_date=None, use_tushare=True):
        """带缓存的股票日线数据获取（供磁盘缓存使用）"""
        if use_tushare:
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
                    return df
                else:
                    logger.warning(f"获取{ts_code}的日线数据为空")
                    return self._get_local_stock_data(ts_code, start_date, end_date)
            except Exception as e:
                logger.error(f"从Tushare获取{ts_code}的日线数据失败: {str(e)}")
                return self._get_local_stock_data(ts_code, start_date, end_date)
        else:
            return self._get_local_stock_data(ts_code, start_date, end_date)
            
    def get_stock_daily_data(self, ts_code, start_date=None, end_date=None):
        """获取股票日线数据（带内存和磁盘缓存）"""
        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            
        # 检查内存缓存
        cache_key = f"{ts_code}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
            
        # 使用磁盘缓存获取数据
        df = self.get_stock_daily_data_cached(ts_code, start_date, end_date, self.use_tushare)
        
        # 存入内存缓存
        self.data_cache[cache_key] = df
        return df
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
                close = np.random.normal(100, 10, n).cumsum() + 3000
                open_price = close * np.random.normal(1, 0.01, n)
                high = np.maximum(close, open_price) * np.random.normal(1.02, 0.01, n)
                low = np.minimum(close, open_price) * np.random.normal(0.98, 0.01, n)
                vol = np.random.normal(100000, 20000, n) * (1 + 0.1 * np.sin(np.arange(n) / 10))
                vol = np.abs(vol)
                # 创建DataFrame
                data = {
                    'ts_code': [ts_code] * n,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'vol': vol,
                    'amount': vol * close
                }
                df = pd.DataFrame(data, index=date_range)
                return df
        except Exception as e:
            logger.error(f"从本地获取{ts_code}的日线数据失败: {str(e)}")
            return pd.DataFrame()
    def calculate_momentum(self, data):
        """计算动量指标"""
        if data.empty:
            return data
        df = data.copy()
        # 计算移动平均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        # 计算动量
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        df['momentum_60'] = df['close'] / df['close'].shift(60) - 1
        # 计算相对强弱指数 (RSI)
        delta = df['close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        # 计算MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['signal']
        # 计算KDJ指标
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        df['k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['d'] = df['k'].rolling(window=3).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        # 计算成交量变化
        df['vol_ratio_5'] = df['vol'] / df['vol'].rolling(window=5).mean()
        df['vol_ratio_20'] = df['vol'] / df['vol'].rolling(window=20).mean()
        # 计算布林带
        df['boll_mid'] = df['close'].rolling(window=20).mean()
        df['boll_std'] = df['close'].rolling(window=20).std()
        df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
        # 去除NaN值
        df.dropna(inplace=True)
        return df
    def calculate_momentum_score(self, data):
        """计算动量综合评分"""
        if data.empty or len(data) < 60:
            return 0, {}
        # 获取最新数据
        latest = data.iloc[-1]
        # 计算各个指标得分
        scores = {}
        # 1. 价格相对于均线得分 (0-20分)
        ma_score = 0
        if latest['close'] > latest['ma5']:
            ma_score += 5
        if latest['close'] > latest['ma10']:
            ma_score += 5
        if latest['close'] > latest['ma20']:
            ma_score += 5
        if latest['close'] > latest['ma60']:
            ma_score += 5
        scores['ma_score'] = ma_score
        # 2. 动量得分 (0-25分)
        momentum_score = 0
        if latest['momentum_5'] > 0:
            momentum_score += 5
        if latest['momentum_10'] > 0:
            momentum_score += 5
        if latest['momentum_20'] > 0:
            momentum_score += 8
        if latest['momentum_60'] > 0:
            momentum_score += 7
        scores['momentum_score'] = momentum_score
        # 3. RSI得分 (0-15分)
        rsi_score = 0
        if 40 <= latest['rsi'] <= 80:
            rsi_score += 7
        if 50 <= latest['rsi'] <= 70:
            rsi_score += 8
        scores['rsi_score'] = rsi_score
        # 4. MACD得分 (0-15分)
        macd_score = 0
        if latest['macd'] > 0:
            macd_score += 7
        if latest['macd'] > latest['signal']:
            macd_score += 8
        scores['macd_score'] = macd_score
        # 5. KDJ得分 (0-15分)
        kdj_score = 0
        if latest['k'] > latest['d']:
            kdj_score += 7
        if 20 <= latest['k'] <= 80:
            kdj_score += 8
        scores['kdj_score'] = kdj_score
        # 6. 成交量得分 (0-10分)
        volume_score = 0
        if latest['vol_ratio_5'] > 1:
            volume_score += 5
        if latest['vol_ratio_20'] > 1:
            volume_score += 5
        scores['volume_score'] = volume_score
        # 计算总分
        total_score = sum(scores.values())
        # 返回总分和各项得分
        return total_score, scores
    def plot_stock_chart(self, data, stock_code, stock_name, score_details, save_path=None):
        """绘制股票K线和技术指标图"""
        if data.empty:
            logger.warning(f"无法绘制{stock_code}的图表，数据为空")
            return False
        # 创建图表
        fig = plt.figure(figsize=(14, 14))
        # 设置网格
        gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 1, 1])
        # 第一个子图：K线和均线
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title(f"{stock_name}({stock_code}) 动量分析 - 总分: {sum(score_details.values())}", fontsize=15)
        # 绘制K线
        for i in range(len(data)):
            date = data.index[i]
            open_price, close, high, low = data.iloc[i][['open', 'close', 'high', 'low']]
            # 确定颜色（红涨绿跌）
            color = 'red' if close >= open_price else 'green'
            # 绘制影线
            ax1.plot([date, date], [low, high], color=color, linewidth=1)
            # 绘制实体
            rect_height = abs(open_price - close)
            rect_bottom = min(open_price, close)
            ax1.add_patch(plt.Rectangle((date - pd.Timedelta(days=0.4), rect_bottom),
                                       pd.Timedelta(days=0.8), rect_height,
                                       edgecolor=color, facecolor=color))
        # 绘制均线
        ax1.plot(data.index, data['ma5'], label='MA5', color='blue', linewidth=1)
        ax1.plot(data.index, data['ma10'], label='MA10', color='orange', linewidth=1)
        ax1.plot(data.index, data['ma20'], label='MA20', color='purple', linewidth=1)
        ax1.plot(data.index, data['ma60'], label='MA60', color='brown', linewidth=1)
        # 绘制布林带
        ax1.plot(data.index, data['boll_mid'], label='BOLL', color='black', linewidth=1)
        ax1.plot(data.index, data['boll_upper'], label='Upper', color='gray', linewidth=1, linestyle='--')
        ax1.plot(data.index, data['boll_lower'], label='Lower', color='gray', linewidth=1, linestyle='--')
        ax1.legend(loc='best')
        ax1.grid(True)
        # 第二个子图：成交量
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        for i in range(len(data)):
            date = data.index[i]
            open_price, close = data.iloc[i][['open', 'close']]
            color = 'red' if close >= open_price else 'green'
            ax2.bar(date, data.iloc[i]['vol'], color=color, width=pd.Timedelta(days=0.8))
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        # 第三个子图：MACD
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax3.plot(data.index, data['macd'], label='MACD', color='blue', linewidth=1)
        ax3.plot(data.index, data['signal'], label='Signal', color='red', linewidth=1)
        # 绘制MACD柱状图
        for i in range(len(data)):
            date = data.index[i]
            hist = data.iloc[i]['macd_hist']
            color = 'red' if hist >= 0 else 'green'
            ax3.bar(date, hist, color=color, width=pd.Timedelta(days=0.8))
        ax3.legend(loc='best')
        ax3.set_ylabel('MACD')
        ax3.grid(True)
        # 第四个子图：RSI
        ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
        ax4.plot(data.index, data['rsi'], label='RSI', color='purple', linewidth=1)
        ax4.axhline(y=30, color='green', linestyle='--')
        ax4.axhline(y=70, color='red', linestyle='--')
        ax4.legend(loc='best')
        ax4.set_ylabel('RSI')
        ax4.set_ylim(0, 100)
        ax4.grid(True)
        # 第五个子图：KDJ
        ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)
        ax5.plot(data.index, data['k'], label='K', color='blue', linewidth=1)
        ax5.plot(data.index, data['d'], label='D', color='yellow', linewidth=1)
        ax5.plot(data.index, data['j'], label='J', color='magenta', linewidth=1)
        ax5.axhline(y=20, color='green', linestyle='--')
        ax5.axhline(y=80, color='red', linestyle='--')
        ax5.legend(loc='best')
        ax5.set_ylabel('KDJ')
        ax5.grid(True)
        # 添加得分信息
        score_text = "\n".join([f"{k}: {v}" for k, v in score_details.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.05, score_text, transform=ax1.transAxes, fontsize=9,
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
    # 添加单个股票分析函数（用于多进程处理）
    def analyze_single_stock(self, stock_data):
        """分析单个股票（用于多进程）"""
        ts_code = stock_data['ts_code']
        name = stock_data['name']
        industry = stock_data.get('industry', '')
        min_score = stock_data.get('min_score', 60)
        
        try:
            logger.info(f"正在分析: {name}({ts_code})")
            
            # 获取日线数据
            data = self.get_stock_daily_data(ts_code)
            if data.empty:
                logger.warning(f"无法获取{ts_code}的数据，跳过分析")
                return None
                
            # 计算技术指标
            data = self.calculate_momentum(data)
            if data.empty:
                logger.warning(f"计算{ts_code}的技术指标失败，跳过分析")
                return None
                
            # 计算动量得分
            score, score_details = self.calculate_momentum_score(data)
            if score >= min_score:
                # 获取最新数据
                latest = data.iloc[-1]
                
                # 保存分析结果
                result = {
                    'ts_code': ts_code,
                    'name': name,
                    'industry': industry,
                    'close': latest['close'],
                    'momentum_20': latest.get('momentum_20', 0),
                    'momentum_20d': latest.get('momentum_20', 0),
                    'rsi': latest.get('rsi', 0),
                    'macd': latest.get('macd', 0),
                    'macd_hist': latest.get('macd_hist', 0),
                    'volume_ratio': latest.get('vol_ratio_20', 1),
                    'score': score,
                    'score_details': score_details,
                    'data': data
                }
                
                # 生成图表
                chart_path = os.path.join(RESULTS_DIR, "charts", f"{ts_code}_momentum.png")
                self.plot_stock_chart(data, ts_code, name, score_details, save_path=chart_path)
                
                return result
        except Exception as e:
            logger.error(f"分析{name}({ts_code})时出错: {str(e)}")
            
        return None
        
    def analyze_stocks(self, stock_list, sample_size=100, min_score=60):
        """分析股票列表，找出具有强劲动量的股票（支持多进程）"""
        results = []
        
        # 确保股票列表不为空
        if stock_list.empty:
            logger.error("股票列表为空，无法进行分析")
            return results
            
        # 记录原始股票数量
        original_count = len(stock_list)
        logger.info(f"准备分析 {original_count} 支股票")
        
        # 限制样本大小
        if sample_size < len(stock_list):
            stock_list = stock_list.sample(sample_size)
            logger.info(f"从 {original_count} 支股票中随机选择 {sample_size} 支进行分析")
        else:
            logger.info(f"分析全部 {original_count} 支股票")
            
        total = len(stock_list)
        logger.info(f"开始分析 {total} 支股票，使用{'多进程' if self.use_multiprocessing else '单进程'}")
        
        # 准备任务列表
        tasks = []
        for _, stock in stock_list.iterrows():
            stock_data = stock.to_dict()
            stock_data['min_score'] = min_score
            tasks.append(stock_data)
            
        # 使用多进程处理
        if self.use_multiprocessing and total > 10:  # 股票数量超过10只才值得使用多进程
            logger.info(f"启动 {self.workers} 个工作进程")
            with Pool(processes=self.workers) as pool:
                # 使用多进程映射函数
                pool_results = pool.map(self.analyze_single_stock, tasks)
                
                # 筛选有效结果
                for result in pool_results:
                    if result is not None:
                        results.append(result)
                        
        # 使用单进程处理
        else:
            for idx, task in enumerate(tasks):
                logger.info(f"分析进度: {idx+1}/{total}")
                result = self.analyze_single_stock(task)
                if result is not None:
                    results.append(result)
                    
        # 按得分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"分析完成，符合条件的股票数量: {len(results)}")
        
        # 将结果保存为CSV
        if results:
            result_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'data' and k != 'score_details'}
                                    for r in results])
            csv_path = os.path.join(RESULTS_DIR, f"momentum_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"已将分析结果保存至: {csv_path}")
            
        return results
# 运行测试
if __name__ == "__main__":
    # 测试
    analyzer = MomentumAnalyzer(use_tushare=True)
    # 获取股票列表
    stocks = analyzer.get_stock_list()
    print(f"获取到 {len(stocks)} 支股票")
    # 分析前20支股票
    results = analyzer.analyze_stocks(stocks.head(20), min_score=50)
    # 输出结果
    for r in results:
        print(f"{r['name']}({r['ts_code']}): 得分={r['score']}, 动量20={r['momentum_20']:.2%}, RSI={r['rsi']:.2f}")