#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
独立的市场概览测试脚本
直接生成和保存模拟市场数据，不依赖DataSourceManager
"""

import logging
import time
import numpy as np
import pandas as pd
import json
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_mock_market_data(trade_date=None):
    """生成模拟市场数据"""
    # 使用固定种子确保可重复
    np.random.seed(42)
    
    # 设置日期
    if trade_date is None:
        trade_date = datetime.now().strftime('%Y-%m-%d')
    
    # 生成股票数据
    num_stocks = 1000  # 模拟1000只股票
    
    # 股票代码和名称
    codes = []
    names = []
    for i in range(1, num_stocks + 1):
        # 生成股票代码 (沪深市场混合)
        if i % 3 == 0:  # 创业板
            codes.append(f"300{i%1000:03d}")
        elif i % 3 == 1:  # 沪市
            codes.append(f"600{i%1000:03d}")
        else:  # 深市
            codes.append(f"000{i%1000:03d}")
        
        # 生成股票名称
        industry_names = ["科技", "金融", "医药", "能源", "消费", "工业", "材料", "通信"]
        type_names = ["股份", "科技", "集团", "电子", "食品", "制药", "新材料", "软件"]
        
        name = f"{np.random.choice(industry_names)}{np.random.choice(type_names)}{i%100:02d}"
        names.append(name)
    
    # 生成开盘价
    open_prices = np.random.uniform(5, 100, num_stocks)
    
    # 生成涨跌幅 - 正态分布，均值为0，标准差为2%
    change_pcts = np.random.normal(0, 0.02, num_stocks)
    
    # 计算收盘价
    close_prices = open_prices * (1 + change_pcts)
    
    # 生成最高价和最低价
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.uniform(0, 0.02, num_stocks))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.uniform(0, 0.02, num_stocks))
    
    # 生成成交量和成交额
    volumes = np.random.randint(10000, 10000000, num_stocks)
    amounts = volumes * (open_prices + close_prices) / 2
    
    # 创建涨停和跌停股票
    # 涨停: 最后50只股票设置为涨停
    for i in range(num_stocks - 50, num_stocks):
        change_pcts[i] = 0.1  # 10%涨幅
        close_prices[i] = open_prices[i] * 1.1
        high_prices[i] = close_prices[i]
    
    # 跌停: 倒数第51-80只股票设置为跌停
    for i in range(num_stocks - 80, num_stocks - 50):
        change_pcts[i] = -0.1  # -10%跌幅
        close_prices[i] = open_prices[i] * 0.9
        low_prices[i] = close_prices[i]
    
    # 创建DataFrame
    df = pd.DataFrame({
        'code': codes,
        'name': names,
        'date': trade_date,
        'open': open_prices,
        'close': close_prices,
        'high': high_prices,
        'low': low_prices,
        'volume': volumes,
        'amount': amounts,
        'change_pct': change_pcts * 100  # 转换为百分比
    })
    
    # 重置随机种子
    np.random.seed(None)
    
    logger.info(f"成功生成 {len(df)} 只股票的模拟市场数据")
    return df

def generate_market_overview(all_data, trade_date=None):
    """根据市场数据生成市场概览"""
    if trade_date is None:
        trade_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"根据 {len(all_data)} 只股票数据生成市场概览...")
    
    # 计算涨跌家数
    up_count = len(all_data[all_data['close'] > all_data['open']])
    down_count = len(all_data[all_data['close'] < all_data['open']])
    flat_count = len(all_data) - up_count - down_count
    
    # 计算总成交量和成交额
    total_volume = all_data['volume'].sum() if 'volume' in all_data.columns else 0
    total_amount = all_data['amount'].sum() if 'amount' in all_data.columns else 0
    
    # 计算平均涨跌幅
    if 'change_pct' not in all_data.columns:
        all_data['change_pct'] = (all_data['close'] - all_data['open']) / all_data['open'] * 100
    
    avg_change_pct = all_data['change_pct'].mean()
    
    # 获取涨停和跌停股票
    limit_up_stocks = all_data[all_data['change_pct'] > 9.5][['code', 'name']].to_dict('records')
    limit_down_stocks = all_data[all_data['change_pct'] < -9.5][['code', 'name']].to_dict('records')
    
    # 组装结果
    result = {
        'date': trade_date,
        'up_count': int(up_count),
        'down_count': int(down_count),
        'flat_count': int(flat_count),
        'total_count': len(all_data),
        'total_volume': float(total_volume),
        'total_amount': float(total_amount),
        'avg_change_pct': float(avg_change_pct),
        'limit_up_count': len(limit_up_stocks),
        'limit_down_count': len(limit_down_stocks),
        'limit_up_stocks': limit_up_stocks[:10],  # 只返回前10只
        'limit_down_stocks': limit_down_stocks[:10],  # 只返回前10只
        'turnover_rate': float(total_volume / total_amount * 100) if total_amount > 0 else 0
    }
    
    logger.info(f"成功生成市场概览，包含 {len(result)} 个字段")
    return result

def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("开始独立测试市场概览功能...")
    logger.info("=" * 50)
    
    # 生成模拟市场数据
    market_data = generate_mock_market_data()
    
    # 生成市场概览
    market_overview = generate_market_overview(market_data)
    
    # 打印市场概览关键数据
    print("\n市场概览关键数据:")
    print(f"日期: {market_overview.get('date', 'N/A')}")
    
    if 'up_count' in market_overview and 'down_count' in market_overview:
        total = market_overview.get('up_count', 0) + market_overview.get('down_count', 0) + market_overview.get('flat_count', 0)
        up_ratio = market_overview.get('up_count', 0) / total * 100 if total > 0 else 0
        down_ratio = market_overview.get('down_count', 0) / total * 100 if total > 0 else 0
        
        print(f"涨跌家数: {market_overview.get('up_count', 0)}涨({up_ratio:.1f}%) / {market_overview.get('down_count', 0)}跌({down_ratio:.1f}%)")
    
    if 'limit_up_count' in market_overview and 'limit_down_count' in market_overview:
        print(f"涨停/跌停: {market_overview.get('limit_up_count', 0)}涨停 / {market_overview.get('limit_down_count', 0)}跌停")
    
    if 'avg_change_pct' in market_overview:
        print(f"平均涨跌幅: {market_overview.get('avg_change_pct', 0):.2f}%")
    
    if 'total_amount' in market_overview:
        amount_billion = market_overview.get('total_amount', 0) / 100000000  # 转换为亿元
        print(f"总成交额: {amount_billion:.2f}亿元")
    
    if 'limit_up_stocks' in market_overview and market_overview['limit_up_stocks']:
        print("\n部分涨停股:")
        for stock in market_overview['limit_up_stocks'][:5]:  # 只显示前5个
            print(f"  {stock.get('code', 'N/A')} {stock.get('name', 'N/A')}")
    
    # 保存结果到文件
    with open('standalone_market_overview.json', 'w', encoding='utf-8') as f:
        json.dump(market_overview, f, ensure_ascii=False, indent=2)
    logger.info("市场概览结果已保存到 standalone_market_overview.json")
    
    logger.info("\n" + "=" * 50)
    logger.info("独立测试市场概览功能完成!")
    logger.info("=" * 50)
    
    return True

if __name__ == "__main__":
    main() 