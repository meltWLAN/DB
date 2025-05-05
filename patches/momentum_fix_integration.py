#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动量分析模块修复集成文件
在系统启动时自动加载修复
"""

import logging
import sys
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # 导入修复模块
    from momentum_fix_complete import patch_momentum_analyzer
    
    # 检查原始模块
    try:
        from momentum_analysis_enhanced_performance import EnhancedMomentumAnalyzer as OriginalAnalyzer
        
        # 获取修复后的分析器
        FixedMomentumAnalyzer = patch_momentum_analyzer()
        
        # 替换原始模块中的分析器
        import sys
        import momentum_analysis_enhanced_performance
        
        # 替换原始类
        momentum_analysis_enhanced_performance.EnhancedMomentumAnalyzer = FixedMomentumAnalyzer
        
        logger.info("动量分析模块修复已成功集成")
    except ImportError:
        logger.warning("未找到原始动量分析模块，跳过修复集成")
except ImportError:
    logger.error("导入修复模块失败，请确保momentum_fix_complete.py文件存在")
except Exception as e:
    logger.error(f"集成修复模块时出错: {str(e)}")
    import traceback
    traceback.print_exc()
