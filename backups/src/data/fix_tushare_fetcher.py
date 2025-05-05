#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复Tushare数据获取模块的use_cache属性问题和优化API调用
"""

import os
import sys
import re

# 添加项目根目录到PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def fix_tushare_fetcher():
    """修复TushareFetcher类中的问题"""
    print("正在修复TushareFetcher类...")
    
    # 确定文件路径
    file_path = 'src/data/tushare_fetcher.py'
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在!")
        return False
    
    # 备份原文件
    backup_path = file_path + '.bak'
    try:
        with open(file_path, 'r') as f:
            original_content = f.read()
        
        with open(backup_path, 'w') as f:
            f.write(original_content)
        print(f"已备份原文件到 {backup_path}")
    except Exception as e:
        print(f"备份文件失败: {e}")
        return False
    
    # 修复步骤1: 添加use_cache属性
    modified_content = original_content
    init_pattern = r'def __init__\(self, config=None\):(.*?)# 初始化pro_api'
    init_replacement = '''def __init__(self, config=None):
        """初始化Tushare数据获取器
        
        Args:
            config: 配置字典
        """
        # 导入依赖
        import tushare as ts
        import logging
        import os
        import time
        from datetime import datetime
        
        # 设置logger
        self.logger = logging.getLogger(__name__)
        
        # 读取配置
        if config is None:
            from src.config import DATA_SOURCE_CONFIG
            if 'tushare' in DATA_SOURCE_CONFIG:
                config = DATA_SOURCE_CONFIG['tushare']
            else:
                config = {}
        
        self.config = config
        self.token = config.get('token', '')
        self.timeout = config.get('timeout', 60)
        self.max_retry = config.get('max_retry', 3)
        self.retry_delay = config.get('retry_delay', 5)
        self.rate_limit = config.get('rate_limit', 300)  # 降低API调用频率
        self.concurrent_calls = config.get('concurrent_calls', 5)  # 降低并发调用数
        self.use_cache = config.get('use_cache', True)  # 添加缓存开关属性
        
        # API调用计数和时间记录
        self.api_call_count = 0
        self.api_call_times = []
        
        # 创建缓存目录
        from src.config import CACHE_DIR
        self.cache_dir = os.path.join(CACHE_DIR, 'tushare')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 初始化pro_api'''
    
    # 修复步骤2: 优化API调用
    api_call_pattern = r'def api_call\(self, func, \*\*kwargs\):(.*?)return None'
    api_call_replacement = '''def api_call(self, func, **kwargs):
        """封装API调用，包含重试和错误处理
        
        Args:
            func: API函数
            **kwargs: 函数参数
            
        Returns:
            pandas.DataFrame: API调用结果
        """
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retry:
            try:
                # 检查API调用频率
                self._check_rate_limit()
                
                # 记录API调用
                self.api_call_count += 1
                
                # 调用API并记录时间
                start_time = time.time()
                result = func(**kwargs)
                elapsed_time = time.time() - start_time
                
                # 记录调用成功
                self.logger.debug(f"API调用成功，耗时 {elapsed_time:.2f} 秒")
                
                # 如果是空DataFrame，也视为失败
                if isinstance(result, pd.DataFrame) and result.empty:
                    self.logger.warning(f"API返回空DataFrame")
                    time.sleep(self.retry_delay)  # 在重试之前等待
                    retry_count += 1
                    continue
                
                # 添加随机延迟，避免连续请求过快
                import random
                delay = random.uniform(0.5, 1.5)
                time.sleep(delay)
                
                return result
                
            except Exception as e:
                retry_count += 1
                last_error = e
                
                if retry_count < self.max_retry:
                    # 指数增加重试延迟
                    delay = self.retry_delay * (2 ** (retry_count - 1))
                    self.logger.warning(f"API调用失败，{delay}秒后重试: {str(e)}")
                    time.sleep(delay)  # 重试前延迟
                else:
                    self.logger.error(f"达到最大重试次数，API调用失败: {str(e)}")
        
        # 所有重试都失败了
        self.logger.error(f"API调用最终失败: {str(last_error)}")
        return None'''
    
    # 修复步骤3: 添加get_today方法
    today_method = '''
    def get_today(self):
        """获取今天的日期，格式为YYYYMMDD"""
        return datetime.now().strftime('%Y%m%d')
    '''
    
    # 修复步骤4: 修复use_cache检查部分
    cache_check_pattern = r'if not result_df\.empty and self\.use_cache:'
    cache_check_replacement = 'if not result_df.empty and hasattr(self, "use_cache") and self.use_cache:'
    
    cache_read_pattern = r'if os\.path\.exists\(cache_file\) and self\.use_cache:'
    cache_read_replacement = 'if os.path.exists(cache_file) and hasattr(self, "use_cache") and self.use_cache:'
    
    # 执行替换
    try:
        # 对大段代码使用非贪婪匹配，避免匹配过多
        modified_content = re.sub(init_pattern, init_replacement, modified_content, flags=re.DOTALL)
        modified_content = re.sub(api_call_pattern, api_call_replacement, modified_content, flags=re.DOTALL)
        
        # 检查是否需要添加get_today方法
        if 'def get_today(self):' not in modified_content:
            # 找到get_continuous_limit_up_stocks方法之前的位置
            get_stocks_pos = modified_content.find('def get_continuous_limit_up_stocks')
            if get_stocks_pos != -1:
                # 在get_continuous_limit_up_stocks之前插入get_today方法
                modified_content = modified_content[:get_stocks_pos] + today_method + modified_content[get_stocks_pos:]
        
        # 修复缓存检查
        modified_content = modified_content.replace(cache_check_pattern, cache_check_replacement)
        modified_content = modified_content.replace(cache_read_pattern, cache_read_replacement)
        
        # 写入修改后的内容
        with open(file_path, 'w') as f:
            f.write(modified_content)
        
        print(f"成功修复 {file_path}!")
        return True
    
    except Exception as e:
        print(f"修复文件失败: {e}")
        # 恢复备份
        try:
            with open(backup_path, 'r') as f:
                backup_content = f.read()
            
            with open(file_path, 'w') as f:
                f.write(backup_content)
            print(f"已恢复原文件")
        except Exception as restore_error:
            print(f"恢复原文件失败: {restore_error}")
        
        return False

if __name__ == "__main__":
    fix_tushare_fetcher() 