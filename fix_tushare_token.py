#!/usr/bin/env python3
"""
修复Tushare token处理脚本
解决momentum_analysis.py中依赖token文件导致的错误
"""
import os
import re
import shutil
from datetime import datetime

# 要修复的文件
TARGET_FILE = 'momentum_analysis.py'

# Tushare token
TUSHARE_TOKEN = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"

# 备份文件
def backup_file(file_path):
    """创建文件备份"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{file_path}_backup_{timestamp}"
    shutil.copy2(file_path, backup_path)
    print(f"已创建备份文件: {backup_path}")
    return backup_path

# 修复token处理
def fix_token_handling(file_path):
    """修复token处理，替换为直接使用token值"""
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return False
    
    # 备份文件
    backup_file(file_path)
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找并替换token处理代码
    # 模式1: pro = ts.pro_api()
    content = re.sub(
        r'pro = ts\.pro_api\(\)',
        f'# 直接使用token初始化Tushare API\npro = ts.pro_api("{TUSHARE_TOKEN}")',
        content
    )
    
    # 模式2: 如果有通过get_token获取的代码，也替换掉
    content = re.sub(
        r'token = upass\.get_token\(\)',
        f'token = "{TUSHARE_TOKEN}"  # 使用硬编码token替代文件读取',
        content
    )
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已修复文件: {file_path}")
    return True

if __name__ == "__main__":
    print(f"开始修复Tushare token处理，时间: {datetime.now()}")
    if fix_token_handling(TARGET_FILE):
        print("修复成功！")
    else:
        print("修复失败！")
    print(f"完成时间: {datetime.now()}") 