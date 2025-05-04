#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通过Tushare API获取真实市场数据
用于市场概览功能
"""

import logging
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置Tushare token
TUSHARE_TOKEN = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

def get_market_overview_data():
    """获取市场概览数据"""
    logger.info("开始从Tushare获取市场概览数据...")
    
    try:
        # 获取当前日期
        today = datetime.now().strftime('%Y%m%d')
        # 获取上一个交易日
        last_trade_date = get_last_trade_date()
        
        if not last_trade_date:
            logger.error("无法获取最近交易日期")
            return None
        
        logger.info(f"获取 {last_trade_date} 的市场数据")
        
        # 获取股票日线行情
        daily_data = get_daily_data(last_trade_date)
        if daily_data is None or len(daily_data) == 0:
            logger.error("无法获取股票日线行情数据")
            return None
        
        # 计算市场统计数据
        market_stats = calculate_market_stats(daily_data)
        
        # 获取涨停和跌停股票
        limit_up_stocks, limit_down_stocks = get_limit_stocks(daily_data)
        
        # 合并所有数据
        market_overview = {
            'date': last_trade_date,
            'up_count': market_stats['up_count'],
            'down_count': market_stats['down_count'],
            'flat_count': market_stats['flat_count'],
            'total_count': market_stats['total_count'],
            'total_volume': market_stats['total_volume'],
            'total_amount': market_stats['total_amount'],
            'avg_change_pct': market_stats['avg_change_pct'],
            'limit_up_count': len(limit_up_stocks),
            'limit_down_count': len(limit_down_stocks),
            'limit_up_stocks': limit_up_stocks,
            'limit_down_stocks': limit_down_stocks,
            'turnover_rate': market_stats['turnover_rate']
        }
        
        logger.info(f"成功获取市场概览数据，共 {market_overview['total_count']} 只股票")
        return market_overview
    
    except Exception as e:
        logger.error(f"获取市场概览数据时出错: {str(e)}")
        return None

def get_last_trade_date():
    """获取最近的交易日期"""
    try:
        # 尝试获取最近10天的交易日历
        today = datetime.now()
        start_date = (today - timedelta(days=10)).strftime('%Y%m%d')
        end_date = today.strftime('%Y%m%d')
        
        # 获取交易日历
        trade_cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open=1)
        
        if not trade_cal.empty:
            # 获取最近的交易日期
            last_trade_date = trade_cal['cal_date'].iloc[-1]
            return last_trade_date
        return None
    
    except Exception as e:
        logger.error(f"获取最近交易日期时出错: {str(e)}")
        return None

def get_daily_data(trade_date):
    """获取指定日期的股票日线行情"""
    try:
        # 获取所有股票的日线行情
        df = pro.daily(trade_date=trade_date)
        
        # 如果数据为空，返回None
        if df.empty:
            return None
        
        # 获取股票名称和行业信息
        stock_basic = pro.stock_basic(exchange='', list_status='L', 
                                      fields='ts_code,name,industry,market')
        
        # 合并数据
        if not stock_basic.empty:
            df = pd.merge(df, stock_basic, on='ts_code', how='left')
        
        return df
    
    except Exception as e:
        logger.error(f"获取股票日线行情时出错: {str(e)}")
        return None

def calculate_market_stats(daily_data):
    """计算市场统计数据"""
    # 初始化结果
    stats = {
        'up_count': 0,
        'down_count': 0,
        'flat_count': 0,
        'total_count': len(daily_data),
        'total_volume': 0,
        'total_amount': 0,
        'avg_change_pct': 0,
        'turnover_rate': 0
    }
    
    try:
        # 计算涨跌家数
        stats['up_count'] = len(daily_data[daily_data['pct_chg'] > 0])
        stats['down_count'] = len(daily_data[daily_data['pct_chg'] < 0])
        stats['flat_count'] = len(daily_data[daily_data['pct_chg'] == 0])
        
        # 计算总成交量和总成交额
        stats['total_volume'] = float(daily_data['vol'].sum())
        stats['total_amount'] = float(daily_data['amount'].sum())
        
        # 计算平均涨跌幅
        stats['avg_change_pct'] = float(daily_data['pct_chg'].mean())
        
        # 计算换手率
        if 'turnover_rate' in daily_data.columns:
            stats['turnover_rate'] = float(daily_data['turnover_rate'].mean())
        else:
            # 如果没有换手率字段，用成交量和总量的比值估算
            stats['turnover_rate'] = stats['total_volume'] / stats['total_amount'] * 100 if stats['total_amount'] > 0 else 0
        
        return stats
    
    except Exception as e:
        logger.error(f"计算市场统计数据时出错: {str(e)}")
        return stats

def get_limit_stocks(daily_data):
    """获取涨停和跌停股票"""
    limit_up_stocks = []
    limit_down_stocks = []
    
    try:
        # 选择涨停股票 (涨幅大于9.5%)
        limit_up_df = daily_data[daily_data['pct_chg'] > 9.5]
        # 选择跌停股票 (跌幅小于-9.5%)
        limit_down_df = daily_data[daily_data['pct_chg'] < -9.5]
        
        # 转换为列表形式
        for _, row in limit_up_df.iterrows():
            limit_up_stocks.append({
                'code': row['ts_code'],
                'name': row['name'] if 'name' in row else row['ts_code']
            })
        
        for _, row in limit_down_df.iterrows():
            limit_down_stocks.append({
                'code': row['ts_code'],
                'name': row['name'] if 'name' in row else row['ts_code']
            })
        
        return limit_up_stocks, limit_down_stocks
    
    except Exception as e:
        logger.error(f"获取涨停跌停股票时出错: {str(e)}")
        return [], []

def get_industry_data():
    """获取行业数据"""
    try:
        # 获取最近交易日期
        trade_date = get_last_trade_date()
        if not trade_date:
            return []
        
        # 获取行业列表
        industry_list = pro.index_classify(level='L1', src='SW')
        
        industry_data = []
        for _, row in industry_list.iterrows():
            # 获取行业成分股
            index_stocks = pro.index_member(index_code=row['index_code'])
            
            if index_stocks.empty:
                continue
            
            # 获取这些股票的行情数据
            stocks_list = ','.join(index_stocks['con_code'].tolist())
            
            # 由于股票太多，分批处理
            batch_size = 50
            all_stocks = index_stocks['con_code'].tolist()
            
            up_count = 0
            down_count = 0
            flat_count = 0
            leading_stock = {"name": "", "code": "", "change": 0}
            
            for i in range(0, len(all_stocks), batch_size):
                batch = all_stocks[i:i+batch_size]
                stocks_str = ','.join(batch)
                
                try:
                    # 获取行情数据
                    batch_data = pro.daily(ts_code=stocks_str, trade_date=trade_date)
                    
                    if batch_data.empty:
                        continue
                    
                    # 统计涨跌家数
                    up_count += len(batch_data[batch_data['pct_chg'] > 0])
                    down_count += len(batch_data[batch_data['pct_chg'] < 0])
                    flat_count += len(batch_data[batch_data['pct_chg'] == 0])
                    
                    # 找出领涨股
                    if not batch_data.empty:
                        max_pct_row = batch_data.loc[batch_data['pct_chg'].idxmax()]
                        if max_pct_row['pct_chg'] > leading_stock["change"]:
                            # 获取股票名称
                            stock_info = pro.stock_basic(ts_code=max_pct_row['ts_code'], fields='ts_code,name')
                            stock_name = stock_info['name'].iloc[0] if not stock_info.empty else max_pct_row['ts_code']
                            
                            leading_stock = {
                                "name": stock_name,
                                "code": max_pct_row['ts_code'],
                                "change": max_pct_row['pct_chg']
                            }
                except Exception as e:
                    logger.warning(f"获取行业成分股行情失败: {str(e)}")
            
            # 计算行业平均涨跌幅
            total_count = up_count + down_count + flat_count
            if total_count > 0:
                # 获取行业指数涨跌幅
                try:
                    index_data = pro.index_daily(ts_code=row['index_code'], trade_date=trade_date)
                    industry_change = float(index_data['pct_chg'].iloc[0]) if not index_data.empty else 0
                except:
                    # 如果无法获取指数数据，使用估算值
                    industry_change = leading_stock["change"] * (up_count / total_count) if up_count > 0 else 0
                
                industry_data.append({
                    "name": row['industry_name'],
                    "code": row['index_code'],
                    "change": industry_change,
                    "up_count": up_count,
                    "down_count": down_count,
                    "flat_count": flat_count,
                    "total_count": total_count,
                    "leading_stock": leading_stock
                })
        
        return industry_data
    
    except Exception as e:
        logger.error(f"获取行业数据时出错: {str(e)}")
        return []

def get_index_data():
    """获取主要指数数据"""
    try:
        # 获取最近交易日期
        trade_date = get_last_trade_date()
        if not trade_date:
            return []
        
        # 主要指数列表
        index_list = [
            {'ts_code': '000001.SH', 'name': '上证指数'},
            {'ts_code': '399001.SZ', 'name': '深证成指'},
            {'ts_code': '399006.SZ', 'name': '创业板指'},
            {'ts_code': '000016.SH', 'name': '上证50'},
            {'ts_code': '000300.SH', 'name': '沪深300'},
            {'ts_code': '000905.SH', 'name': '中证500'}
        ]
        
        index_data = []
        for index in index_list:
            try:
                # 获取指数日线数据
                df = pro.index_daily(ts_code=index['ts_code'], start_date=(datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=10)).strftime('%Y%m%d'), end_date=trade_date)
                
                if df.empty:
                    continue
                
                # 获取最新数据
                latest = df.iloc[0]
                
                # 计算5日涨跌幅
                change_5d = 0
                if len(df) >= 5:
                    change_5d = (latest['close'] - df.iloc[4]['close']) / df.iloc[4]['close'] * 100
                
                # 添加到结果中
                index_data.append({
                    'name': index['name'],
                    'code': index['ts_code'],
                    'close': float(latest['close']),
                    'change': float(latest['pct_chg']),
                    'change_5d': float(change_5d),
                    'volume': float(latest['vol']),
                    'amount': float(latest['amount']),
                    'trend': get_trend(latest['pct_chg'])
                })
            except Exception as e:
                logger.warning(f"获取指数 {index['name']} 数据失败: {str(e)}")
        
        return index_data
    
    except Exception as e:
        logger.error(f"获取指数数据时出错: {str(e)}")
        return []

def get_trend(change):
    """根据涨跌幅判断趋势"""
    if change > 1.5:
        return "强势上涨"
    elif change > 0:
        return "上涨"
    elif change < -1.5:
        return "强势下跌"
    elif change < 0:
        return "下跌"
    else:
        return "震荡"

if __name__ == "__main__":
    # 测试获取市场概览数据
    market_data = get_market_overview_data()
    if market_data:
        print(f"成功获取市场概览数据，共 {market_data['total_count']} 只股票")
        print(f"涨跌家数: {market_data['up_count']}涨 / {market_data['down_count']}跌")
        print(f"涨停/跌停: {market_data['limit_up_count']}涨停 / {market_data['limit_down_count']}跌停")
        print(f"平均涨跌幅: {market_data['avg_change_pct']:.2f}%")
    
    # 测试获取指数数据
    index_data = get_index_data()
    if index_data:
        print(f"\n成功获取 {len(index_data)} 个指数数据")
        for idx in index_data:
            print(f"{idx['name']}: {idx['close']:.2f} ({idx['change']:.2f}%)")
    
    # 测试获取行业数据
    industry_data = get_industry_data()
    if industry_data:
        print(f"\n成功获取 {len(industry_data)} 个行业数据")
        for ind in sorted(industry_data, key=lambda x: x['change'], reverse=True)[:5]:
            print(f"{ind['name']}: {ind['change']:.2f}% (上涨家数: {ind['up_count']})") 