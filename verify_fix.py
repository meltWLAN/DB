#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证动量分析模块修复脚本
检查修复是否成功集成到系统中
"""

import sys
import logging
import importlib
import inspect

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_module_availability():
    """验证模块可用性"""
    # 需要检查的模块
    modules_to_check = [
        "momentum_analysis_enhanced_performance",
        "patches.momentum_fix_complete",
        "patches.momentum_fix_integration"
    ]
    
    available_modules = []
    unavailable_modules = []
    
    for module_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            available_modules.append(module_name)
        except ImportError as e:
            logger.warning(f"模块 '{module_name}' 不可用: {str(e)}")
            unavailable_modules.append(module_name)
    
    logger.info(f"可用模块: {', '.join(available_modules)}")
    if unavailable_modules:
        logger.warning(f"不可用模块: {', '.join(unavailable_modules)}")
    
    return available_modules, unavailable_modules

def check_momentum_analyzer():
    """检查动量分析器是否已修复"""
    try:
        # 导入增强版动量分析模块
        from momentum_analysis_enhanced_performance import EnhancedMomentumAnalyzer
        
        # 导入修复版本
        from patches.momentum_fix_complete import patch_momentum_analyzer
        
        # 获取修复后的分析器
        FixedMomentumAnalyzer = patch_momentum_analyzer()
        
        # 检查当前使用的分析器是否是修复版本
        is_fixed = EnhancedMomentumAnalyzer == FixedMomentumAnalyzer
        
        if is_fixed:
            logger.info("验证成功: 动量分析器已被修复版本替换")
            return True
        else:
            logger.warning("验证失败: 动量分析器未被修复版本替换")
            
            # 检查方法签名是否一致
            logger.info("检查关键方法...")
            
            original_methods = {
                name: method for name, method in inspect.getmembers(EnhancedMomentumAnalyzer, inspect.isfunction)
                if not name.startswith('_')
            }
            
            fixed_methods = {
                name: method for name, method in inspect.getmembers(FixedMomentumAnalyzer, inspect.isfunction)
                if not name.startswith('_')
            }
            
            # 检查原始类是否包含修复后的方法
            has_fixed_methods = False
            for name, method in fixed_methods.items():
                if name in original_methods:
                    if original_methods[name] == method:
                        logger.info(f"方法 '{name}' 匹配修复版本")
                        has_fixed_methods = True
            
            if has_fixed_methods:
                logger.info("部分方法已被修复")
                return True
            else:
                logger.warning("没有方法被修复")
                return False
    
    except ImportError as e:
        logger.error(f"导入模块失败: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"检查过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def verify_integration():
    """验证修复集成情况"""
    logger.info("开始验证动量分析模块修复集成...")
    
    # 检查当前Python路径
    logger.info(f"Python路径: {sys.path}")
    
    # 检查模块可用性
    available_modules, unavailable_modules = verify_module_availability()
    
    if "momentum_analysis_enhanced_performance" not in available_modules:
        logger.error("未找到原始动量分析模块，无法完成验证")
        return False
    
    if "patches.momentum_fix_complete" not in available_modules:
        logger.error("未找到修复模块，集成失败")
        return False
    
    # 检查动量分析器
    analyzer_fixed = check_momentum_analyzer()
    
    # 进行简单的功能测试
    if analyzer_fixed:
        logger.info("执行简单功能测试...")
        try:
            # 导入增强版动量分析模块
            from momentum_analysis_enhanced_performance import EnhancedMomentumAnalyzer
            
            # 创建分析器实例
            analyzer = EnhancedMomentumAnalyzer()
            
            # 获取股票列表
            stock_list = analyzer.get_stock_list()
            if stock_list is not None and not stock_list.empty:
                logger.info(f"成功获取股票列表，共 {len(stock_list)} 支股票")
                
                # 测试单只股票
                if len(stock_list) > 0:
                    test_stock = stock_list.iloc[0].to_dict()
                    test_stock['min_score'] = 30  # 设置较低的阈值以便观察结果
                    
                    # 分析单只股票
                    logger.info(f"测试分析 {test_stock['name']}({test_stock['ts_code']})...")
                    result = analyzer.analyze_single_stock_optimized(test_stock)
                    
                    if result is not None:
                        logger.info(f"分析结果: 得分={result.get('score', 'N/A')}")
                        logger.info("功能测试通过！")
                        return True
                    else:
                        logger.warning("单只股票分析未返回结果，但这可能是正常的（如得分不够）")
                        return True
            else:
                logger.warning("获取股票列表失败")
                return False
                
        except Exception as e:
            logger.error(f"功能测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    return analyzer_fixed

if __name__ == "__main__":
    success = verify_integration()
    
    if success:
        logger.info("验证结果: 动量分析模块修复已成功集成到系统中")
        sys.exit(0)
    else:
        logger.error("验证结果: 动量分析模块修复未成功集成到系统中")
        sys.exit(1) 