"""
股票动量分析增强版
查找近期具有强劲动量的股票，增加KDJ和DMI指标，并按行业分类
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt

# 确保src包可以被导入
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 导入配置
from src.enhanced.config.settings import LOG_DIR, DATA_DIR, RESULTS_DIR

# 创建必要目录
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "charts"), exist_ok=True)

# 配置日志
log_file = os.path.join(LOG_DIR, f"enhanced_momentum_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 定义增强版动量分析函数
def calculate_enhanced_momentum(data, window=20):
    """计算更全面的股票动量指标
    
    Args:
        data: 股票数据DataFrame
        window: 动量窗口期，默认20天
        
    Returns:
        DataFrame: 添加了动量指标的数据
    """
    # 确保数据按日期排序
    data = data.sort_values('date')
    
    # 计算百分比变化
    data['pct_change'] = data['close'].pct_change()
    
    # 计算累积收益
    data['cum_return'] = (1 + data['pct_change']).cumprod()
    
    # 计算n日动量（当前价格/n天前价格 - 1）
    data[f'momentum_{window}d'] = data['close'] / data['close'].shift(window) - 1
    data[f'momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data[f'momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # 计算相对强度指标 (RSI)
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # 计算MACD
    data['ema12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['ema26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = data['ema12'] - data['ema26']
    data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['signal']
    
    # 计算5日, 10日和20日移动平均线
    data['ma5'] = data['close'].rolling(window=5).mean()
    data['ma10'] = data['close'].rolling(window=10).mean()
    data['ma20'] = data['close'].rolling(window=20).mean()
    data['ma60'] = data['close'].rolling(window=60).mean()
    
    # 计算均线多头排列 (ma5 > ma10 > ma20)
    data['golden_cross'] = (data['ma5'] > data['ma10']) & (data['ma10'] > data['ma20'])
    
    # 计算布林带
    data['middle_band'] = data['close'].rolling(window=20).mean()
    data['std'] = data['close'].rolling(window=20).std()
    data['upper_band'] = data['middle_band'] + (data['std'] * 2)
    data['lower_band'] = data['middle_band'] - (data['std'] * 2)
    
    # 计算交易量变化
    data['volume_change'] = data['volume'].pct_change()
    data['volume_ma5'] = data['volume'].rolling(window=5).mean()
    data['volume_ma20'] = data['volume'].rolling(window=20).mean()
    
    # 计算KDJ指标
    low_min = data['low'].rolling(window=9).min()
    high_max = data['high'].rolling(window=9).max()
    
    data['K'] = 100 * ((data['close'] - low_min) / (high_max - low_min))
    data['D'] = data['K'].rolling(window=3).mean()
    data['J'] = 3 * data['K'] - 2 * data['D']
    
    # 计算DMI指标(趋向指标)
    data['tr1'] = abs(data['high'] - data['low'])
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr'] = data['tr'].rolling(window=14).mean()
    
    data['up_move'] = data['high'] - data['high'].shift(1)
    data['down_move'] = data['low'].shift(1) - data['low']
    
    data['plus_dm'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0), data['up_move'], 0)
    data['minus_dm'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0), data['down_move'], 0)
    
    data['plus_di'] = 100 * (data['plus_dm'].rolling(window=14).mean() / data['atr'])
    data['minus_di'] = 100 * (data['minus_dm'].rolling(window=14).mean() / data['atr'])
    data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
    data['adx'] = data['dx'].rolling(window=14).mean()
    
    # 计算量价关系指标
    data['volume_price_trend'] = (data['volume'] * data['pct_change']).cumsum()
    
    return data

def analyze_stock(tushare_fetcher, stock_code, start_date, end_date):
    """分析单只股票的动量
    
    Args:
        tushare_fetcher: TuShare数据获取器
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        dict: 分析结果
    """
    try:
        # 获取股票数据
        stock_data = tushare_fetcher.get_daily_data(stock_code, start_date, end_date)
        if stock_data is None or stock_data.empty:
            logger.warning(f"无法获取股票 {stock_code} 的数据")
            return None
            
        # 计算动量指标
        with_momentum = calculate_enhanced_momentum(stock_data)
        
        # 获取最新数据
        latest_data = with_momentum.iloc[-1]
        
        # 判断动量信号
        is_momentum_20d = latest_data.get('momentum_20d', 0) > 0.1  # 20日动量大于10%
        is_momentum_10d = latest_data.get('momentum_10d', 0) > 0.05  # 10日动量大于5%
        is_momentum_5d = latest_data.get('momentum_5d', 0) > 0.03  # 5日动量大于3%
        
        is_rsi_high = latest_data.get('rsi', 0) > 70  # RSI大于70
        is_macd_positive = latest_data.get('macd_hist', 0) > 0  # MACD柱状图为正
        is_above_ma20 = latest_data.get('close', 0) > latest_data.get('ma20', 0)  # 价格在20日均线上方
        is_volume_high = latest_data.get('volume', 0) > latest_data.get('volume_ma20', 0) * 1.5  # 成交量大于20日均量的1.5倍
        is_golden_cross = latest_data.get('golden_cross', False)  # 均线多头排列
        
        # KDJ指标判断
        is_kdj_golden_cross = latest_data.get('K', 0) > latest_data.get('D', 0)  # K线上穿D线
        is_kdj_high = latest_data.get('J', 0) > 80  # J值大于80
        
        # DMI指标判断
        is_dmi_positive = latest_data.get('plus_di', 0) > latest_data.get('minus_di', 0)  # +DI大于-DI
        is_adx_strong = latest_data.get('adx', 0) > 25  # ADX大于25表示趋势强
        
        # 计算综合得分 (0-100)
        score = 0
        if is_momentum_20d: score += 15
        if is_momentum_10d: score += 10
        if is_momentum_5d: score += 5
        if is_rsi_high: score += 10
        if is_macd_positive: score += 10
        if is_above_ma20: score += 10
        if is_volume_high: score += 10
        if is_golden_cross: score += 10
        if is_kdj_golden_cross: score += 10
        if is_kdj_high: score += 5
        if is_dmi_positive: score += 10
        if is_adx_strong: score += 5
        
        # 获取股票名称和行业
        stock_list = tushare_fetcher.get_stock_list()
        stock_name = ""
        industry = ""
        if stock_list is not None and not stock_list.empty:
            stock_info = stock_list[stock_list['code'] == stock_code]
            if not stock_info.empty:
                stock_name = stock_info.iloc[0]['name']
                if 'industry' in stock_info.columns:
                    industry = stock_info.iloc[0]['industry']
        
        # 构建分析结果
        analysis = {
            'code': stock_code,
            'name': stock_name,
            'industry': industry,
            'date': latest_data['date'],
            'close': latest_data['close'],
            'momentum_20d': latest_data.get('momentum_20d', 0),
            'momentum_10d': latest_data.get('momentum_10d', 0),
            'momentum_5d': latest_data.get('momentum_5d', 0),
            'rsi': latest_data.get('rsi', 0),
            'macd_hist': latest_data.get('macd_hist', 0),
            'K': latest_data.get('K', 0),
            'D': latest_data.get('D', 0),
            'J': latest_data.get('J', 0),
            'adx': latest_data.get('adx', 0),
            'plus_di': latest_data.get('plus_di', 0),
            'minus_di': latest_data.get('minus_di', 0),
            'golden_cross': is_golden_cross,
            'kdj_golden_cross': is_kdj_golden_cross,
            'dmi_positive': is_dmi_positive,
            'volume_ratio': latest_data.get('volume', 0) / latest_data.get('volume_ma20', 1) if 'volume_ma20' in latest_data else 0,
            'score': score,
            'is_strong_momentum': score >= 70,
            'data': with_momentum
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"分析股票 {stock_code} 时出错: {str(e)}")
        return None

def plot_enhanced_stock_chart(analysis_result, output_dir=None):
    """绘制增强版股票走势图，包含KDJ和DMI指标
    
    Args:
        analysis_result: 分析结果
        output_dir: 输出目录
        
    Returns:
        str: 图表文件路径
    """
    if not analysis_result or 'data' not in analysis_result:
        return None
    
    stock_data = analysis_result['data'].tail(60)  # 最近60个交易日
    
    # 创建子图布局
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # 格式化日期轴以避免中文显示问题
    dates = pd.to_datetime(stock_data['date'])
    date_strs = [d.strftime('%m-%d') for d in dates]
    
    # 绘制K线图和均线
    ax1.plot(date_strs, stock_data['close'], 'b-', label='收盘价')
    ax1.plot(date_strs, stock_data['ma5'], 'r--', label='MA5')
    ax1.plot(date_strs, stock_data['ma10'], 'g--', label='MA10')
    ax1.plot(date_strs, stock_data['ma20'], 'y--', label='MA20')
    ax1.plot(date_strs, stock_data['upper_band'], 'c:', label='上轨')
    ax1.plot(date_strs, stock_data['lower_band'], 'c:', label='下轨')
    
    # 设置标题
    score_text = f"{analysis_result['name']}({analysis_result['code']}) - 评分: {analysis_result['score']}"
    if analysis_result.get('industry'):
        score_text += f" - 行业: {analysis_result['industry']}"
    ax1.set_title(score_text)
    ax1.set_ylabel('价格')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # 绘制成交量
    volume_colors = ['red' if x >= 0 else 'green' for x in stock_data['pct_change']]
    ax2.bar(date_strs, stock_data['volume'], color=volume_colors)
    ax2.plot(date_strs, stock_data['volume_ma5'], 'r-', label='成交量MA5')
    ax2.plot(date_strs, stock_data['volume_ma20'], 'b-', label='成交量MA20')
    ax2.set_ylabel('成交量')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # 绘制MACD
    ax3.plot(date_strs, stock_data['macd'], 'b-', label='MACD')
    ax3.plot(date_strs, stock_data['signal'], 'r-', label='Signal')
    macd_colors = ['red' if x >= 0 else 'green' for x in stock_data['macd_hist']]
    ax3.bar(date_strs, stock_data['macd_hist'], color=macd_colors)
    ax3.set_ylabel('MACD')
    ax3.grid(True)
    ax3.legend(loc='upper left')
    
    # 绘制KDJ和ADX指标
    ax4.plot(date_strs, stock_data['K'], 'r-', label='K值')
    ax4.plot(date_strs, stock_data['D'], 'g-', label='D值')
    ax4.plot(date_strs, stock_data['J'], 'b-', label='J值')
    ax4.plot(date_strs, stock_data['adx'], 'k-', label='ADX')
    ax4.axhline(y=80, color='grey', linestyle='--')
    ax4.axhline(y=20, color='grey', linestyle='--')
    ax4.set_ylabel('KDJ & ADX')
    ax4.set_xlabel('日期')
    ax4.grid(True)
    ax4.legend(loc='upper left')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        chart_file = os.path.join(output_dir, f"{analysis_result['code']}_chart.png")
        plt.savefig(chart_file)
        plt.close()
        return chart_file
    else:
        plt.show()
        plt.close()
        return None

def main():
    """主函数"""
    try:
        logger.info("==== 增强版股票动量分析系统启动 ====")
        
        # 导入数据源
        from src.enhanced.data.fetchers.tushare_fetcher import EnhancedTushareFetcher
        
        # 初始化TuShare数据源
        logger.info("初始化TuShare数据源...")
        from src.enhanced.config.settings import DATA_SOURCE_CONFIG
        tushare_config = DATA_SOURCE_CONFIG.get('tushare', {})
        
        # 处理配置
        fetcher_config = {
            'token': tushare_config.get('token', ''),
            'rate_limit': tushare_config.get('rate_limit', {}).get('calls_per_minute', 500) / 60 if 'rate_limit' in tushare_config else 500 / 60,
            'connection_retries': tushare_config.get('retry', {}).get('max_retries', 3) if 'retry' in tushare_config else 3,
            'retry_delay': tushare_config.get('retry', {}).get('retry_interval', 5) if 'retry' in tushare_config else 5
        }
        
        # 初始化数据获取器
        tushare_fetcher = EnhancedTushareFetcher(fetcher_config)
        
        # 设置分析参数
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')  # 获取120天数据用于计算指标
        
        # 获取股票列表
        logger.info("获取股票列表...")
        stock_list = tushare_fetcher.get_stock_list()
        if stock_list is None or stock_list.empty:
            logger.error("获取股票列表失败，无法继续分析")
            return
        
        # 如果股票列表没有行业信息列，添加基本的行业分类
        if 'industry' not in stock_list.columns:
            logger.warning("股票列表中没有行业信息，使用股票代码前缀进行分类")
            def get_industry_by_code(code):
                if code.startswith(('600', '601', '603')):
                    return "沪市主板"
                elif code.startswith('000'):
                    return "深市主板"
                elif code.startswith('002'):
                    return "中小板"
                elif code.startswith('300'):
                    return "创业板"
                elif code.startswith('688'):
                    return "科创板"
                else:
                    return "其他"
            
            stock_list['industry'] = stock_list['code'].apply(get_industry_by_code)
            
        # 按行业分类股票
        industry_groups = stock_list.groupby('industry')
        
        # 分析结果列表
        strong_momentum_stocks = []
        
        # 设置每个行业分析的股票数量
        stocks_per_industry = 10
        
        # 按行业分析股票
        for industry_name, industry_stocks in industry_groups:
            logger.info(f"分析行业: {industry_name}，共 {len(industry_stocks)} 只股票")
            
            # 确保每个行业不超过指定数量
            if len(industry_stocks) > stocks_per_industry:
                # 随机选择股票以避免偏差
                sample_stocks = industry_stocks.sample(stocks_per_industry)
            else:
                sample_stocks = industry_stocks
                
            # 分析每只股票
            for idx, stock_row in sample_stocks.iterrows():
                stock_code = stock_row['code']
                stock_name = stock_row['name']
                
                logger.info(f"正在分析股票: {stock_name}({stock_code}) - {industry_name}...")
                result = analyze_stock(tushare_fetcher, stock_code, start_date, end_date)
                
                if result and result.get('is_strong_momentum', False):
                    strong_momentum_stocks.append(result)
                    logger.info(f"发现强势动量股票: {stock_name}({stock_code}), 行业: {industry_name}, 得分: {result['score']}")
                    
                    # 生成图表
                    charts_dir = os.path.join(RESULTS_DIR, "charts")
                    chart_file = plot_enhanced_stock_chart(result, charts_dir)
                    if chart_file:
                        logger.info(f"已生成图表: {chart_file}")
        
        # 按得分排序动量股票
        strong_momentum_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        # 输出结果
        logger.info(f"分析完成，共发现 {len(strong_momentum_stocks)} 只强势动量股票")
        if strong_momentum_stocks:
            # 创建结果表格
            result_df = pd.DataFrame([
                {
                    '股票代码': s['code'],
                    '股票名称': s['name'],
                    '行业': s.get('industry', ''),
                    '收盘价': s['close'],
                    '20日动量': f"{s['momentum_20d']*100:.2f}%",
                    '10日动量': f"{s['momentum_10d']*100:.2f}%",
                    '5日动量': f"{s['momentum_5d']*100:.2f}%",
                    'RSI': f"{s['rsi']:.2f}",
                    'MACD柱状图': f"{s['macd_hist']:.4f}",
                    'KDJ-J值': f"{s['J']:.2f}",
                    'ADX': f"{s['adx']:.2f}",
                    '成交量比': f"{s['volume_ratio']:.2f}",
                    '得分': s['score']
                }
                for s in strong_momentum_stocks
            ])
            
            # 保存结果
            result_file = os.path.join(RESULTS_DIR, f"enhanced_momentum_stocks_{datetime.now().strftime('%Y%m%d')}.csv")
            result_df.to_csv(result_file, index=False, encoding='utf-8-sig')
            logger.info(f"结果已保存至: {result_file}")
            
            # 行业分布统计
            if 'industry' in stock_list.columns:
                industry_stats = result_df.groupby('行业').size().reset_index(name='股票数量')
                industry_stats = industry_stats.sort_values('股票数量', ascending=False)
                
                # 保存行业分布
                industry_file = os.path.join(RESULTS_DIR, f"industry_momentum_stats_{datetime.now().strftime('%Y%m%d')}.csv")
                industry_stats.to_csv(industry_file, index=False, encoding='utf-8-sig')
                logger.info(f"行业分布统计已保存至: {industry_file}")
                
                # 显示行业分布
                print("\n强势动量股票行业分布:")
                print(industry_stats.to_string(index=False))
            
            # 显示结果
            print("\n强势动量股票:")
            print(result_df.to_string(index=False))
        
        logger.info("==== 增强版股票动量分析系统运行完成 ====")
        
    except Exception as e:
        logger.error(f"系统运行出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 