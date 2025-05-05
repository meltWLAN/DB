#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动量分析模块修复集成脚本
将修复后的动量分析模块集成到股票分析系统中
"""

import os
import sys
import shutil
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def backup_original_file(file_path):
    """
    备份原始文件
    
    Args:
        file_path: 需要备份的文件路径
    
    Returns:
        str: 备份文件的路径
    """
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return None
    
    # 创建备份文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}_backup_{timestamp}"
    
    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"已备份原始文件: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"备份文件失败: {str(e)}")
        return None

def install_momentum_fix():
    """安装动量分析模块修复"""
    logger.info("开始安装动量分析模块修复...")
    
    # 检查修复文件是否存在
    fix_file = "momentum_fix_complete.py"
    if not os.path.exists(fix_file):
        logger.error(f"修复文件不存在: {fix_file}")
        return False
    
    # 创建必要的目录
    os.makedirs("patches", exist_ok=True)
    
    # 将修复脚本复制到patches目录
    try:
        shutil.copy2(fix_file, os.path.join("patches", fix_file))
        logger.info(f"已复制修复文件到patches目录")
    except Exception as e:
        logger.error(f"复制修复文件失败: {str(e)}")
        return False
    
    # 修改启动脚本，加入修复逻辑
    scripts_to_update = ["run_optimized.sh", "run_main_gui.sh"]
    
    for script in scripts_to_update:
        if not os.path.exists(script):
            logger.warning(f"脚本文件不存在: {script}")
            continue
        
        # 备份原始脚本
        backup_path = backup_original_file(script)
        if not backup_path:
            continue
        
        try:
            with open(script, 'r') as f:
                content = f.read()
            
            # 检查是否已经包含修复代码
            if "# 应用动量分析模块修复" in content:
                logger.info(f"脚本 {script} 已包含修复代码，跳过修改")
                continue
            
            # 在Python启动命令前添加修复逻辑
            if "$PYTHON stock_analysis_gui.py" in content:
                modified_content = content.replace(
                    "$PYTHON stock_analysis_gui.py", 
                    "# 应用动量分析模块修复\n"
                    "echo \"正在应用动量分析模块修复...\"\n"
                    "if [ -f patches/momentum_fix_complete.py ]; then\n"
                    "    export PYTHONPATH=\"$PYTHONPATH:$SCRIPT_DIR/patches\"\n"
                    "    echo \"动量分析模块修复已应用\"\n"
                    "else\n"
                    "    echo \"警告: 未找到动量分析模块修复文件\"\n"
                    "fi\n\n"
                    "$PYTHON stock_analysis_gui.py"
                )
                
                with open(script, 'w') as f:
                    f.write(modified_content)
                
                logger.info(f"已更新启动脚本: {script}")
            else:
                logger.warning(f"未找到脚本 {script} 中的启动命令，跳过修改")
        
        except Exception as e:
            logger.error(f"修改脚本 {script} 失败: {str(e)}")
    
    # 创建集成文件，用于系统启动时自动加载修复
    integration_file = "patches/momentum_fix_integration.py"
    
    try:
        with open(integration_file, 'w') as f:
            f.write("""#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
动量分析模块修复集成文件
在系统启动时自动加载修复
\"\"\"

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
""")
        
        logger.info(f"已创建修复集成文件: {integration_file}")
    except Exception as e:
        logger.error(f"创建修复集成文件失败: {str(e)}")
        return False
    
    # 创建自动加载文件
    patches_init_file = "patches/__init__.py"
    
    try:
        with open(patches_init_file, 'w') as f:
            f.write("""# 自动加载动量分析模块修复
try:
    from . import momentum_fix_integration
except ImportError:
    pass
""")
        
        logger.info(f"已创建自动加载文件: {patches_init_file}")
    except Exception as e:
        logger.error(f"创建自动加载文件失败: {str(e)}")
        return False
    
    logger.info("动量分析模块修复安装完成!")
    logger.info("现在可以运行 ./run_optimized.sh 或 ./run_main_gui.sh 启动已修复的系统")
    
    return True

if __name__ == "__main__":
    install_momentum_fix() 