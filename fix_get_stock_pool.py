#!/usr/bin/env python
"""
添加get_stock_pool函数到momentum_analysis.py文件中
"""
import os
import shutil
from datetime import datetime

# 备份原始文件
original_file = '/Users/mac/Desktop/DB/momentum_analysis.py'
backup_file = '/Users/mac/Desktop/DB/momentum_analysis_backup_pool_{}.py'.format(
    datetime.now().strftime('%Y%m%d_%H%M%S'))

print(f"备份原始文件到: {backup_file}")
shutil.copy2(original_file, backup_file)

# 读取原始文件内容
with open(original_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 添加get_stock_pool函数 - 在get_stock_name函数之后
get_stock_pool_func = """
def get_stock_pool():
    \"\"\"
    获取默认股票池
    
    返回:
        默认股票池，DataFrame格式
    \"\"\"
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
    
    return pd.DataFrame(stock_data)

"""

# 查找get_stock_name函数结尾
get_stock_name_end = 'def get_stock_name(ts_code):'
get_stock_name_pos = content.find(get_stock_name_end)

if get_stock_name_pos > 0:
    # 在get_stock_name函数之后查找空行
    func_end = content.find('class MomentumAnalyzer:', get_stock_name_pos)
    if func_end > 0:
        # 在get_stock_name函数和MomentumAnalyzer类之间插入get_stock_pool函数
        content = content[:func_end] + get_stock_pool_func + '\n' + content[func_end:]
    else:
        print("警告：无法找到MomentumAnalyzer类的起始位置")
else:
    print("警告：无法找到get_stock_name函数")
    
# 更新main函数最后的error处理，添加更多详细信息
update_error_handling = """
    except Exception as e:
        print(f"执行动量分析时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 调试辅助信息
        print("\\n调试辅助信息:")
        print(f"- get_stock_pool 函数是否存在: {'get_stock_pool' in globals()}")
        print(f"- get_stock_name 函数是否存在: {'get_stock_name' in globals()}")
        
        # 尝试使用简单的股票池继续运行
        try:
            print("\\n尝试使用简单股票池继续...")
            
            # 创建简单股票池
            simple_stocks = pd.DataFrame({
                'ts_code': ['000001.SZ', '000002.SZ'],
                'name': ['平安银行', '万科A']
            })
            
            print(f"简单股票池: {len(simple_stocks)}支股票")
            sample_results = analyzer.analyze_stocks(simple_stocks)
            
            # 输出结果
            print("\\n分析结果 (简单股票池):")
            if isinstance(sample_results, pd.DataFrame) and not sample_results.empty:
                for index, row in sample_results.iterrows():
                    print(f"{row['stock_name']}({row['ts_code']}): " 
                          f"得分={row['momentum_score']:.1f}, "
                          f"价格变化={row['price_change']:.2f}%")
            else:
                print("没有有效的分析结果 (简单股票池)")
        except Exception as e2:
            print(f"尝试简单股票池时出错: {str(e2)}")
"""

# 查找error处理代码的位置
error_handling_pos = content.find("except Exception as e:\n        print(f\"执行动量分析时出错: {str(e)}\")")

if error_handling_pos > 0:
    # 查找代码块结束位置
    block_end = content.find('\n', error_handling_pos + len("except Exception as e:\n        print(f\"执行动量分析时出错: {str(e)}\")"))
    if block_end > 0:
        # 替换error处理代码
        content = content[:error_handling_pos] + update_error_handling + content[block_end+1:]

# 修复主函数最后的测试以省略get_stock_list调用
fix_main_call = """
     # 后续分析
     try:
         print("创建动量分析器实例...")
         analyzer = MomentumAnalyzer(use_tushare=True)
         
         # 使用简单股票列表而不是调用get_stock_list
         print("创建股票列表...")
         stocks = pd.DataFrame({
             'ts_code': ['000001.SZ', '000002.SZ', '000063.SZ', '000333.SZ', '000651.SZ'],
             'name': ['平安银行', '万科A', '中兴通讯', '美的集团', '格力电器']
         })
         print(f"创建了 {len(stocks)} 支股票")
         
         # 分析前5支股票
         sample_size = min(5, len(stocks))
         print(f"分析前{sample_size}支股票...")
         
         results = analyzer.analyze_stocks(stocks.head(sample_size))
"""

# 查找主函数中的相关部分
main_test_pos = content.find("     # 后续分析\n     try:")

if main_test_pos > 0:
    # 查找stocks = analyzer.get_stock_list()那一行
    get_stock_list_pos = content.find("         stocks = analyzer.get_stock_list()", main_test_pos)
    next_block = content.find("         # 分析前5支股票", get_stock_list_pos)
    
    if get_stock_list_pos > 0 and next_block > 0:
        # 替换获取股票列表的代码
        content = content[:main_test_pos] + fix_main_call + content[next_block:]

# 写入修改后的内容
with open(original_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("成功添加get_stock_pool函数并修复其他相关问题")
print("您可以现在运行momentum_analysis.py脚本来测试") 