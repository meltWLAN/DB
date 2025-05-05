#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
调试脚本，用于找出技术指标计算中的问题
"""

import pandas as pd
import numpy as np
import traceback
import sys
from pathlib import Path

# 添加src目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入模块
try:
    from src.indicators.advanced_indicators import AdvancedIndicators
    from src.data.data_fetcher import DataFetcher
except ImportError as e:
    print(f"导入模块时出错: {e}")
    sys.exit(1)

def simulate_data():
    """创建模拟数据进行测试"""
    # 创建日期索引
    date_range = pd.date_range(start="2020-01-01", end="2020-12-31", freq="B")
    
    # 生成随机价格
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.normal(0, 1, len(date_range)))
    
    # 创建OHLCV数据
    data = pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(1000, 10000, len(date_range))
    }, index=date_range)
    
    return data

def main():
    """主函数"""
    # 创建一组测试数据
    test_df = simulate_data()
    print(f"测试数据: shape={test_df.shape}, 类型={type(test_df)}")
    
    # 使用真实数据获取器
    print("\n测试数据获取器:")
    fetcher = DataFetcher()
    sample_data = fetcher.get_stock_data("600000", "2020-01-01", "2020-12-31")
    print(f"实际数据: shape={sample_data.shape if sample_data is not None else None}, 类型={type(sample_data)}")
    
    if not isinstance(sample_data, pd.DataFrame):
        try:
            # 尝试转换为DataFrame
            if isinstance(sample_data, np.ndarray):
                cols = ["open", "high", "low", "close", "volume"]
                sample_data = pd.DataFrame(sample_data, columns=cols)
                print(f"转换后: shape={sample_data.shape}, 类型={type(sample_data)}")
        except Exception as e:
            print(f"转换数据格式时出错: {e}")
    
    # 测试技术指标计算 - 模拟数据
    print("\n测试技术指标计算 - 模拟数据:")
    try:
        indicators_df = AdvancedIndicators.add_advanced_indicators(test_df)
        print(f"计算结果: shape={indicators_df.shape}, 类型={type(indicators_df)}")
        print(f"计算的指标列: {[col for col in indicators_df.columns if col not in test_df.columns][:5]}...")
    except Exception as e:
        print(f"计算指标时出错: {e}")
        print(traceback.format_exc())
    
    # 测试技术指标计算 - 真实数据
    print("\n测试技术指标计算 - 真实数据:")
    try:
        if isinstance(sample_data, pd.DataFrame):
            indicators_real = AdvancedIndicators.add_advanced_indicators(sample_data)
            print(f"计算结果: shape={indicators_real.shape}, 类型={type(indicators_real)}")
        else:
            print("跳过真实数据测试，因为获取的数据不是DataFrame")
    except Exception as e:
        print(f"计算指标时出错: {e}")
        print(traceback.format_exc())
    
    # 测试NumPy数组情况
    print("\n测试NumPy数组情况:")
    try:
        # 创建一个numpy数组
        np_data = test_df.values
        print(f"NumPy数组: shape={np_data.shape}, 类型={type(np_data)}")
        
        # 尝试直接计算
        np_result = AdvancedIndicators.add_advanced_indicators(np_data)
        print(f"NumPy计算结果: shape={np_result.shape if np_result is not None else None}, 类型={type(np_result)}")
    except Exception as e:
        print(f"处理NumPy数组时出错: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 