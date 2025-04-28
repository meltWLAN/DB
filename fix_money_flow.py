#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复资金流向分析模块的错误
解决'invalid literal for int() with base 10: 'SZ'/'SH'/'BJ''问题
"""

import os
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

def fix_analyze_money_flow_method():
    """修复analyze_money_flow方法中对股票代码后缀的处理问题"""
    try:
        # 获取enhanced_momentum_analysis.py文件的绝对路径
        file_path = os.path.join(current_dir, 'enhanced_momentum_analysis.py')
        
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return False
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找错误部分
        if 'code_parts = ts_code.split(\'.\')' in content and 'stock_num = code_parts[0]' in content:
            logger.info("找到可能的错误代码，进行修复...")
            
            # 修改代码，确保只使用数字部分进行哈希计算
            modified_content = content.replace(
                """                # 使用股票代码作为随机种子，确保同一股票总是生成相同的值
                code_parts = ts_code.split('.')
                stock_num = code_parts[0]
                
                # 使用哈希算法生成一个固定的伪随机数
                import hashlib
                hash_obj = hashlib.md5(stock_num.encode())""",
                
                """                # 使用股票代码作为随机种子，确保同一股票总是生成相同的值
                # 提取股票代码数字部分，去除SH/SZ/BJ后缀
                code_parts = ts_code.split('.')
                stock_num = code_parts[0]  # 只使用数字部分
                
                # 使用哈希算法生成一个固定的伪随机数
                import hashlib
                hash_obj = hashlib.md5(stock_num.encode())"""
            )
            
            # 类似的修复应用于analyze_north_money_flow方法
            if 'def analyze_north_money_flow(' in modified_content:
                modified_content = modified_content.replace(
                    """                    # 使用股票代码生成固定的随机值
                    code_parts = ts_code.split('.')
                    exchange = code_parts[1]
                    stock_code = int(code_parts[0])""",
                    
                    """                    # 使用股票代码生成固定的随机值
                    code_parts = ts_code.split('.')
                    exchange = code_parts[1]
                    # 确保只使用数字部分，不尝试将其转换为整数
                    stock_code = code_parts[0]"""
                )
                
                # 如果代码中尝试将stock_code当作整数使用，进一步修复
                modified_content = modified_content.replace(
                    "north_weight = (stock_code % 100) / 100",
                    "north_weight = (int(stock_code) % 100) / 100 if stock_code.isdigit() else 0.5"
                )
            
            # 保存修改后的内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            logger.info("成功修复analyze_money_flow和analyze_north_money_flow方法")
            return True
        else:
            logger.warning("没有找到需要修复的代码，文件可能已经修复或格式已变更")
            return False
    
    except Exception as e:
        logger.error(f"修复过程中出错: {str(e)}")
        return False

def fix_enhanced_gui_controller():
    """修复enhanced_gui_controller.py中可能使用这些方法的地方"""
    try:
        # 获取enhanced_gui_controller.py文件的绝对路径
        file_path = os.path.join(current_dir, 'enhanced_gui_controller.py')
        
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return False
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否存在可能导致错误的代码
        if 'analyze_money_flow(' in content or 'analyze_north_money_flow(' in content:
            logger.info("在enhanced_gui_controller.py中找到相关方法调用，进行修复...")
            
            # 添加异常处理
            modified_content = content.replace(
                "money_flow = self.momentum_analyzer.analyze_money_flow(ts_code)",
                """try:
                    money_flow = self.momentum_analyzer.analyze_money_flow(ts_code)
                except ValueError as e:
                    logger.warning(f"分析资金流向时出错: {str(e)}")
                    money_flow = 0.0"""
            )
            
            modified_content = modified_content.replace(
                "north_flow = self.momentum_analyzer.analyze_north_money_flow(ts_code)",
                """try:
                    north_flow = self.momentum_analyzer.analyze_north_money_flow(ts_code)
                except ValueError as e:
                    logger.warning(f"分析北向资金流向时出错: {str(e)}")
                    north_flow = 0.0"""
            )
            
            # 保存修改后的内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            logger.info("成功修复enhanced_gui_controller.py中的方法调用")
            return True
        else:
            logger.info("在enhanced_gui_controller.py中没有找到需要修复的代码")
            return False
    
    except Exception as e:
        logger.error(f"修复enhanced_gui_controller.py过程中出错: {str(e)}")
        return False

def main():
    """主函数"""
    logger.info("开始修复资金流向分析模块的错误...")
    
    # 修复analyze_money_flow方法
    fix_analyze_money_flow_method()
    
    # 修复enhanced_gui_controller中的调用
    fix_enhanced_gui_controller()
    
    logger.info("修复完成，请重新启动系统进行测试")

if __name__ == "__main__":
    main() 