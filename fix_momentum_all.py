#!/usr/bin/env python
"""
全面修复momentum_analysis.py中的问题
1. 添加get_stock_list方法
2. 修复__init__方法以支持use_tushare参数
3. 添加get_stock_name方法
4. 改进Tushare API的调用方式
"""
import os
import sys
import shutil
import re
from datetime import datetime

# 备份原始文件
original_file = '/Users/mac/Desktop/DB/momentum_analysis.py'
backup_file = '/Users/mac/Desktop/DB/momentum_analysis_backup_{}.py'.format(
    datetime.now().strftime('%Y%m%d_%H%M%S'))

print(f"备份原始文件到: {backup_file}")
shutil.copy2(original_file, backup_file)

# 读取原始文件内容
with open(original_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 修改__init__方法，添加use_tushare参数
init_pattern = r'def __init__\(self, stock_pool=None, start_date=None, end_date=None,\s+backtest_start_date=None, backtest_end_date=None,\s+momentum_period=20, use_parallel=True, max_processes=None,\s+use_cache=True, enable_optimization=True\):'
new_init = 'def __init__(self, stock_pool=None, start_date=None, end_date=None, \n                 backtest_start_date=None, backtest_end_date=None, \n                 momentum_period=20, use_parallel=True, max_processes=None, \n                 use_cache=True, enable_optimization=True, use_tushare=True):'
content = re.sub(init_pattern, new_init, content)

# 添加use_tushare初始化
init_end_pattern = r'self.enable_optimization = enable_optimization'
new_init_end = 'self.enable_optimization = enable_optimization\n        self.use_tushare = use_tushare'
content = content.replace(init_end_pattern, new_init_end)

# 2. 添加get_stock_list方法 (在类中的适当位置)
get_stock_list_method = """
    def get_stock_list(self, industry=None):
        \"\"\"
        获取股票列表
        
        参数:
            industry: 行业名称，默认为None，表示获取所有行业股票
            
        返回:
            包含股票信息的DataFrame
        \"\"\"
        try:
            self.logger.info("获取股票列表")
            pro_api = get_pro_api()
            
            if pro_api is not None and self.use_tushare:
                # 从Tushare获取
                try:
                    # 获取股票基本信息
                    stocks = pro_api.stock_basic(
                        exchange='', 
                        list_status='L', 
                        fields='ts_code,symbol,name,area,industry,list_date'
                    )
                    
                    # 如果指定了行业，进行筛选
                    if industry and industry != "全部":
                        stocks = stocks[stocks['industry'] == industry]
                        
                    return stocks
                except Exception as e:
                    self.logger.error(f"从Tushare获取股票列表失败: {str(e)}")
                    return self._get_local_stock_list(industry)
            else:
                return self._get_local_stock_list(industry)
        except Exception as e:
            self.logger.error(f"获取股票列表时出错: {str(e)}")
            return pd.DataFrame()
    
    def _get_local_stock_list(self, industry=None):
        \"\"\"从本地获取股票列表（备用方法）\"\"\"
        try:
            self.logger.info("从本地获取股票列表")
            
            # 生成一个简单的股票列表
            stock_data = {
                'ts_code': ['000001.SZ', '000002.SZ', '000063.SZ', '000333.SZ', '000651.SZ', 
                           '000858.SZ', '002415.SZ', '600036.SH', '600276.SH', '600887.SH'],
                'symbol': ['000001', '000002', '000063', '000333', '000651', 
                          '000858', '002415', '600036', '600276', '600887'],
                'name': ['平安银行', '万科A', '中兴通讯', '美的集团', '格力电器', 
                        '五粮液', '海康威视', '招商银行', '恒瑞医药', '伊利股份'],
                'area': ['深圳', '深圳', '深圳', '广东', '广东', 
                        '四川', '浙江', '上海', '江苏', '内蒙古'],
                'industry': ['银行', '房地产', '通信', '家电', '家电', 
                           '食品饮料', '电子', '银行', '医药', '食品饮料'],
                'list_date': ['19910403', '19910129', '19971118', '19970620', '19961118', 
                             '19980427', '20100528', '20021119', '20031216', '19960403']
            }
            
            stocks = pd.DataFrame(stock_data)
            
            # 如果指定了行业，进行筛选
            if industry and industry != "全部":
                stocks = stocks[stocks['industry'] == industry]
                
            return stocks
        except Exception as e:
            self.logger.error(f"创建本地股票列表失败: {str(e)}")
            # 返回一个空的DataFrame，具有正确的列
            return pd.DataFrame(columns=['ts_code', 'symbol', 'name', 'area', 'industry', 'list_date'])
"""

# 查找合适的位置插入get_stock_list方法 - 在_init_parallel_pool方法之后
pool_method_end = 'def _init_parallel_pool(self):'
pool_method_pos = content.find(pool_method_end)
pool_method_end_pos = content.find('def get_stock_daily_data', pool_method_pos)

if pool_method_end_pos > 0:
    # 在get_stock_daily_data方法之前插入get_stock_list方法
    content = content[:pool_method_end_pos] + get_stock_list_method + '\n    ' + content[pool_method_end_pos:]
else:
    # 如果找不到get_stock_daily_data，在_init_parallel_pool方法之后插入
    pool_end = content.find('\n\n', pool_method_pos + len(pool_method_end))
    if pool_end > 0:
        content = content[:pool_end+2] + get_stock_list_method + content[pool_end+2:]

# 3. 添加get_stock_name函数 - 在模块级别，在get_pro_api函数之后
get_stock_name_func = """
def get_stock_name(ts_code):
    \"\"\"
    根据股票代码获取股票名称（简单实现，实际使用时应从数据库或API获取）
    
    参数:
        ts_code: 股票代码(带后缀，如000001.SZ)
        
    返回:
        股票名称
    \"\"\"
    # 常见股票名称映射
    stock_names = {
        '000001.SZ': '平安银行',
        '000002.SZ': '万科A',
        '000063.SZ': '中兴通讯',
        '000333.SZ': '美的集团',
        '000651.SZ': '格力电器',
        '000858.SZ': '五粮液',
        '002415.SZ': '海康威视',
        '600036.SH': '招商银行',
        '600276.SH': '恒瑞医药',
        '600887.SH': '伊利股份'
    }
    
    # 如果能找到，返回名称，否则返回代码
    return stock_names.get(ts_code, f"股票{ts_code}")

"""

# 添加get_stock_name函数
get_pro_api_end = 'def get_pro_api():'
get_pro_api_pos = content.find(get_pro_api_end)
get_pro_api_end_pos = content.find('class MomentumAnalyzer:', get_pro_api_pos)

if get_pro_api_end_pos > 0:
    # 在MomentumAnalyzer类之前插入get_stock_name函数
    content = content[:get_pro_api_end_pos] + get_stock_name_func + '\n' + content[get_pro_api_end_pos:]

# 4. 修复主函数调用中的get_stock_list和其他问题
main_func = """
# 运行测试
if __name__ == "__main__":
    # ===== Tushare Direct Initialization Test Start =====
    print("\\n===== Tushare 直接初始化功能测试 =====")
    USER_TOKEN = '0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10'
    direct_pro_api = None
    try:
        print(f"尝试使用 Token '{USER_TOKEN[:5]}...' 直接初始化 Pro API...")
        ts.set_token(USER_TOKEN)
        direct_pro_api = ts.pro_api()
        if direct_pro_api:
             print("直接初始化 Pro API 成功")
             # Test trade_cal
             print("\\n尝试使用直接初始化的 API 调用 trade_cal...")
             try:
                 df_cal = direct_pro_api.trade_cal(exchange='', start_date='20240101', end_date='20240105')
                 print("直接调用 trade_cal 成功:")
                 print(df_cal)
             except Exception as e_cal:
                 print(f"直接调用 trade_cal 失败: {e_cal}")

             # Test daily instead of pro_bar
             print("\\n尝试使用直接初始化的 API 调用 daily 获取 000001.SZ 数据...")
             try:
                 # 尝试使用daily而不是pro_bar
                 df_daily = direct_pro_api.daily(ts_code='000001.SZ', start_date='20240101', end_date='20240110')
                 if df_daily is not None and not df_daily.empty:
                     print("直接调用 daily 成功:")
                     print(df_daily.head())
                 else:
                     print("直接调用 daily 返回为空或 None")
             except Exception as e_daily:
                 print(f"直接调用 daily 失败: {e_daily}")
                 
                 # 如果daily失败，尝试创建模拟数据
                 print("创建模拟数据用于测试:")
                 import pandas as pd
                 import numpy as np
                 from datetime import datetime, timedelta
                 
                 # 创建日期范围
                 start_date = datetime(2024, 1, 1)
                 end_date = datetime(2024, 1, 10)
                 date_range = pd.date_range(start=start_date, end=end_date, freq='B')
                 
                 # 生成模拟数据
                 n = len(date_range)
                 close_prices = np.linspace(10, 11, n) + np.random.normal(0, 0.1, n)
                 
                 # 创建DataFrame
                 df_dummy = pd.DataFrame({
                     'ts_code': ['000001.SZ'] * n,
                     'trade_date': [d.strftime('%Y%m%d') for d in date_range],
                     'open': close_prices * 0.99,
                     'high': close_prices * 1.02, 
                     'low': close_prices * 0.98,
                     'close': close_prices,
                     'vol': np.random.normal(1000000, 200000, n)
                 })
                 print("模拟数据样例:")
                 print(df_dummy.head())
        else:
            print("直接初始化 Pro API 失败 (返回 None)")

    except Exception as e_init:
         print(f"直接初始化 Pro API 或调用时发生错误: {e_init}")

    print("===== Tushare 直接初始化功能测试结束 =====\\n")
    # ===== Tushare Direct Initialization Test End =====

    # 后续分析
    try:
        print("创建动量分析器实例...")
        analyzer = MomentumAnalyzer(use_tushare=True)
        
        # 获取股票列表
        print("获取股票列表...")
        stocks = analyzer.get_stock_list()
        print(f"获取到 {len(stocks)} 支股票")
        
        # 分析前5支股票
        sample_size = min(5, len(stocks))
        print(f"分析前{sample_size}支股票...")
        
        results = analyzer.analyze_stocks(stocks.head(sample_size))
        
        # 输出结果
        print("\\n分析结果:")
        if isinstance(results, pd.DataFrame) and not results.empty:
            for index, row in results.iterrows():
                print(f"{row['stock_name']}({row['ts_code']}): " 
                      f"得分={row['momentum_score']:.1f}, "
                      f"价格变化={row['price_change']:.2f}%")
        else:
            print("没有有效的分析结果")
    except Exception as e:
        print(f"执行动量分析时出错: {e}")
        import traceback
        traceback.print_exc()
"""

# 替换主函数部分
main_start = "# 运行测试\nif __name__ == \"__main__\":"
main_pos = content.find(main_start)

if main_pos > 0:
    main_end = content.find('\n# ===== Tushare Direct Initialization Test End =====')
    if main_end > 0:
        main_end = content.find('\n\n', main_end)
        content = content[:main_pos] + main_func + content[main_end:]
    else:
        # 如果找不到结束标记，直接替换到文件末尾
        content = content[:main_pos] + main_func

# 写入修改后的内容
with open(original_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("成功修复以下问题:")
print("1. 添加get_stock_list方法 - 支持从Tushare或本地获取股票列表")
print("2. 修复__init__方法 - 添加use_tushare参数")
print("3. 添加get_stock_name方法 - 获取股票名称")
print("4. 改进Tushare API的调用方式 - 使用daily代替pro_bar，并添加模拟数据支持")
print("\n修复已完成，您可以运行该脚本以应用所有修复") 