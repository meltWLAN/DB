#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
应用优化模块
将所有创建的优化应用到动量分析模块
"""

import os
import sys
import logging
import importlib
import time
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 确保当前目录在Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def print_separator(title):
    """打印带有标题的分隔线"""
    width = 80
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

def apply_optimizations():
    """应用所有优化模块"""
    start_time = time.time()
    
    print_separator("开始应用优化")
    
    # 步骤1: 应用参数优化
    print_separator("第1步：应用参数优化")
    try:
        from optimizations.hyper_optimize import get_optimized_analyzer
        analyzer = get_optimized_analyzer()
        print(f"成功创建优化版动量分析器: {analyzer}")
    except ImportError as e:
        logger.error(f"导入参数优化模块失败: {e}")
        print("参数优化失败，请确保optimizations目录存在且包含hyper_optimize.py")
    
    # 步骤2: 应用数据存储优化
    print_separator("第2步：应用数据存储优化")
    try:
        from optimizations.storage_optimizer import setup_optimized_storage, optimize_all_cache_files
        
        # 设置优化的存储环境
        cache_dir = setup_optimized_storage()
        print(f"优化存储环境设置完成: {cache_dir}")
        
        # 优化缓存文件
        print("开始优化缓存文件...")
        optimized_count = optimize_all_cache_files()
        print(f"缓存文件优化完成，共处理 {optimized_count} 个文件")
    except ImportError as e:
        logger.error(f"导入数据存储优化模块失败: {e}")
        print("数据存储优化失败，请确保optimizations目录存在且包含storage_optimizer.py")
    
    # 步骤3: 应用内存优化
    print_separator("第3步：应用内存优化")
    try:
        from optimizations.memory_optimizer import get_memory_usage, cleanup_memory
        
        # 显示内存使用情况
        mem_info = get_memory_usage()
        print("当前内存使用情况:")
        for key, value in mem_info.items():
            if 'percent' in key:
                print(f"  {key}: {value:.1f}%")
            else:
                print(f"  {key}: {value:.1f} MB")
        
        # 执行内存清理
        print("\n执行内存清理...")
        cleanup_result = cleanup_memory(force=True)
        if cleanup_result.get('cleanup_performed', False):
            saved = cleanup_result.get('memory_saved_mb', 0)
            print(f"内存清理完成，释放了 {saved:.2f} MB 内存")
        else:
            print("内存清理未执行")
    except ImportError as e:
        logger.error(f"导入内存优化模块失败: {e}")
        print("内存优化失败，请确保optimizations目录存在且包含memory_optimizer.py")
    
    # 步骤4: 应用异步数据预加载
    print_separator("第4步：应用异步数据预加载")
    try:
        from optimizations.async_prefetch import get_prefetcher, prefetch_data, get_prefetched_data
        
        # 初始化预取器
        prefetcher = get_prefetcher()
        print(f"异步数据预加载器初始化完成: {prefetcher}")
        
        # 启动预取示例任务
        def dummy_task(name):
            time.sleep(0.2)  # 模拟耗时操作
            return f"预取任务 {name} 完成"
        
        # 预取几个示例任务
        print("启动示例预取任务...")
        tasks = []
        for i in range(3):
            task_id = prefetch_data(dummy_task, f"示例任务{i}")
            tasks.append(task_id)
        
        # 等待任务完成
        print("等待预取任务完成...")
        time.sleep(0.5)  # 给予一些时间完成任务
        
        # 获取结果
        for task_id in tasks:
            result, error = get_prefetched_data(task_id)
            if error is None:
                print(f"获取预取结果: {result}")
            else:
                print(f"预取任务错误: {error}")
    except ImportError as e:
        logger.error(f"导入异步预加载模块失败: {e}")
        print("异步预加载优化失败，请确保optimizations目录存在且包含async_prefetch.py")
    
    # 集成所有优化并更新动量分析模块
    print_separator("集成所有优化到动量分析模块")
    try:
        # 检查原始动量模块
        import momentum_analysis
        print(f"原始动量分析模块版本: {getattr(momentum_analysis, '__version__', '未知')}")
        
        # 检查修复后的模块
        try:
            sys.path.append(os.path.join(current_dir, 'patches'))
            from patches.momentum_fix_integration import integrate_momentum_fix
            print("应用动量分析模块修复...")
            integrate_momentum_fix()
            print("动量分析模块修复已应用")
        except ImportError as e:
            print(f"无法加载动量修复模块: {e}")
            print("跳过动量分析模块修复集成")
        
        # 集成优化模块
        print("\n创建优化版动量分析对象...")
        from optimizations import get_optimized_analyzer
        optimized_analyzer = get_optimized_analyzer()
        if optimized_analyzer:
            print("优化版动量分析器创建成功！")
        else:
            print("警告: 创建优化版动量分析器失败")
    except ImportError as e:
        logger.error(f"集成优化到动量分析模块失败: {e}")
        print(f"无法集成优化到动量分析模块: {str(e)}")
    
    # 完成
    elapsed = time.time() - start_time
    print_separator(f"优化应用完成，耗时 {elapsed:.2f} 秒")
    print("\n现在您可以通过以下方式使用优化后的系统:")
    print("1. 运行 './run_super_optimized.sh' 启动超级优化版系统")
    print("2. 在代码中导入优化模块: 'from optimizations import get_optimized_analyzer'")
    print("3. 使用异步预加载: 'from optimizations import prefetch_data, get_prefetched_data'")
    print("\n祝您使用愉快！\n")

if __name__ == "__main__":
    apply_optimizations() 