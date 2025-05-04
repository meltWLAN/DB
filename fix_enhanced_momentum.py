#!/usr/bin/env python3
"""
修复增强型动量分析器的缓存方法问题
"""
import os
import re

def fix_enhanced_momentum_file():
    """修复enhanced_momentum_analysis.py中的_cached_get_stock_data方法缺失问题"""
    filename = "enhanced_momentum_analysis.py"
    
    # 检查文件是否存在
    if not os.path.exists(filename):
        print(f"错误: 找不到文件 {filename}")
        return False
    
    # 读取文件内容
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找类定义的位置
    class_pattern = r'class EnhancedMomentumAnalyzer\(MomentumAnalyzer\):'
    class_match = re.search(class_pattern, content)
    if not class_match:
        print("错误: 找不到EnhancedMomentumAnalyzer类定义")
        return False
    
    class_end = class_match.end()
    
    # 查找__init__方法的结束位置
    init_pattern = r'def __init__.*?}(\s*\n)'
    init_match = re.search(init_pattern, content[class_end:], re.DOTALL)
    if not init_match:
        print("错误: 找不到__init__方法")
        return False
    
    insert_pos = class_end + init_match.end()
    
    # 要插入的代码
    cached_method_code = '''
    # 添加兼容方法：修复函数名不匹配问题
    def _cached_get_stock_data(self, ts_code, start_date, end_date, use_tushare):
        """兼容旧版API的缓存数据获取方法"""
        if use_tushare:
            try:
                # 从Tushare获取日线数据
                df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df.empty:
                    # 尝试使用备用API
                    df = ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if not df.empty:
                    # 确保日期列为索引并按日期排序
                    if "trade_date" in df.columns:
                        df["trade_date"] = pd.to_datetime(df["trade_date"])
                        df.sort_values("trade_date", inplace=True)
                        df.set_index("trade_date", inplace=True)
                    return df
                else:
                    logger.warning(f"获取{ts_code}的日线数据为空")
                    return self._get_local_stock_data(ts_code, start_date, end_date)
            except Exception as e:
                logger.error(f"从Tushare获取{ts_code}的日线数据失败: {str(e)}")
                return self._get_local_stock_data(ts_code, start_date, end_date)
        else:
            return self._get_local_stock_data(ts_code, start_date, end_date)
    '''
    
    # 插入代码
    new_content = content[:insert_pos] + cached_method_code + content[insert_pos:]
    
    # 备份原始文件
    backup_filename = f"{filename}.bak"
    with open(backup_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"已备份原始文件到 {backup_filename}")
    
    # 写入新内容
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"已成功修复 {filename}")
    
    return True

if __name__ == "__main__":
    fix_enhanced_momentum_file() 