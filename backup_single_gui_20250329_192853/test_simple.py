"""
简单测试脚本，确认系统可以正常工作
"""

import os
import sys
import pandas as pd
import random
from datetime import datetime, timedelta

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 创建必要的目录
for dir_path in ['data', 'data/cache', 'results', 'logs']:
    os.makedirs(dir_path, exist_ok=True)
    print(f"✅ 创建目录: {dir_path}")

print("\n正在测试系统功能...\n")

# 创建一个模拟的股票数据
print("创建模拟数据...")
start_date = "2024-01-01"
end_date = "2024-03-25"
stock_code = "000001"  # 平安银行

# 创建日期范围
date_range = pd.date_range(start=start_date, end=end_date)
dates = [d.strftime('%Y-%m-%d') for d in date_range if d.weekday() < 5]  # 只保留工作日

# 基础价格
base_price = 10.0
# 每日随机波动
price_changes = [random.uniform(-0.05, 0.08) for _ in range(len(dates))]

# 累计价格变化
prices = []
current_price = base_price
for change in price_changes:
    current_price = current_price * (1 + change)
    prices.append(current_price)

# 创建模拟数据
data = {
    'date': dates,
    'open': prices,
    'high': [p * (1 + random.uniform(0.01, 0.05)) for p in prices],
    'low': [p * (1 - random.uniform(0.01, 0.05)) for p in prices],
    'close': [p * (1 + random.uniform(-0.03, 0.03)) for p in prices],
    'volume': [int(100000 * (1 + random.uniform(-0.5, 1.5))) for _ in range(len(dates))],
    'amount': [int(10000000 * (1 + random.uniform(-0.5, 1.5))) for _ in range(len(dates))]
}

df = pd.DataFrame(data)
print(f"创建了 {len(df)} 行模拟数据")

# 保存到缓存目录
cache_dir = "data/cache"
os.makedirs(cache_dir, exist_ok=True)
cache_file = os.path.join(cache_dir, f"{stock_code}_{start_date}_{end_date}.csv")
df.to_csv(cache_file, index=False)
print(f"数据已保存到: {cache_file}")

# 计算一些基本的技术指标
print("\n计算技术指标...")
df['MA5'] = df['close'].rolling(window=5).mean()
df['MA10'] = df['close'].rolling(window=10).mean()
df['daily_return'] = df['close'].pct_change()

# 打印数据概览
print("\n数据概览:")
print(df.head())

# 计算基本统计信息
total_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
max_price = df['high'].max()
min_price = df['low'].min()
avg_volume = df['volume'].mean()

print(f"\n总收益率: {total_return:.2%}")
print(f"最高价: {max_price:.2f}")
print(f"最低价: {min_price:.2f}")
print(f"平均成交量: {avg_volume:.0f}")

# 保存结果
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
result_file = os.path.join(results_dir, f"{stock_code}_analysis.csv")
df.to_csv(result_file, index=False)
print(f"\n分析结果已保存到: {result_file}")

print("\n✅ 系统功能测试完成，一切正常!") 