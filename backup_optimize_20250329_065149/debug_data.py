#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据调试工具
用于检查生成的模拟数据格式和结构
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 设置数据目录
DATA_DIR = "./data"

def check_stock_data(stock_code):
    """检查股票数据的格式和结构"""
    try:
        # 构建文件路径
        file_path = os.path.join(DATA_DIR, f"{stock_code.replace('.', '_')}.csv")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误：文件 {file_path} 不存在")
            return None
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 打印数据基本信息
        print(f"\n数据文件: {file_path}")
        print(f"数据行数: {len(df)}")
        print(f"数据列数: {len(df.columns)}")
        print(f"数据列: {', '.join(df.columns)}")
        print(f"数据起止日期: {df['trade_date'].min()} 至 {df['trade_date'].max()}")
        
        # 检查数据内容
        print("\n数据前5行:")
        print(df.head())
        
        # 检查数据是否有空值
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print("\n数据存在空值:")
            for col in null_counts.index:
                if null_counts[col] > 0:
                    print(f"  - {col}: {null_counts[col]} 个空值")
        else:
            print("\n数据完整，无空值")
        
        # 检查日期是否连续
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.sort_values('trade_date', inplace=True)
        date_diff = df['trade_date'].diff().dt.days
        if date_diff.max() > 3:  # 允许周末的差距
            large_gaps = df.loc[date_diff > 3, 'trade_date']
            print(f"\n警告：数据存在较大日期间隔，最大间隔为 {date_diff.max()} 天")
            print(f"大间隔出现在：{', '.join(large_gaps.astype(str))}")
        else:
            print("\n日期连续性良好")
        
        # 检查价格是否合理
        price_cols = ['open', 'high', 'low', 'close']
        if all(col in df.columns for col in price_cols):
            price_range = (df[price_cols].min().min(), df[price_cols].max().max())
            print(f"\n价格范围: {price_range[0]:.2f} - {price_range[1]:.2f}")
            
            # 检查是否有异常值（价格为0或异常大）
            if (df[price_cols] <= 0).any().any():
                print("警告：存在价格小于等于0的异常值")
            if (df[price_cols] > 10000).any().any():
                print("警告：存在价格超过10000的异常值")
                
            # 检查是否满足high >= open/close >= low
            invalid_rows = df[(df['high'] < df['open']) | 
                              (df['high'] < df['close']) | 
                              (df['low'] > df['open']) | 
                              (df['low'] > df['close'])]
            if not invalid_rows.empty:
                print(f"\n警告：存在 {len(invalid_rows)} 行数据不满足价格逻辑关系")
                print("样例:")
                print(invalid_rows.head())
            else:
                print("\n价格逻辑关系检查通过")
        
        # 绘制价格图表
        plt.figure(figsize=(12, 6))
        plt.plot(df['trade_date'], df['close'], label='收盘价')
        plt.title(f'{stock_code} 收盘价')
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图表
        chart_path = os.path.join(DATA_DIR, f"{stock_code.replace('.', '_')}_chart.png")
        plt.savefig(chart_path)
        print(f"\n已保存价格图表至 {chart_path}")
        
        return df
        
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
        return None

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        stock_code = sys.argv[1]
    else:
        # 默认检查第一个股票
        stock_file = next((f for f in os.listdir(DATA_DIR) if f.endswith('.csv')), None)
        if stock_file:
            stock_code = stock_file.replace('_', '.').rstrip('.csv')
        else:
            print("错误：未找到任何股票数据文件")
            return
    
    # 检查数据
    df = check_stock_data(stock_code)
    
    # 如果成功获取数据，显示统计信息
    if df is not None:
        print("\n数据统计信息:")
        print(df.describe())

if __name__ == "__main__":
    main() 