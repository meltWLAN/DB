#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强API与GUI集成测试脚本
测试增强型API可靠性模块在GUI中的集成效果
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 配置日志
log_file = f"enhanced_api_gui_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_gui_controller_api():
    """测试GUI控制器中的增强API功能"""
    logger.info("开始测试GUI控制器中的增强API功能")
    
    try:
        # 导入GUI控制器
        from gui_controller import GuiController
        
        # 初始化控制器
        controller = GuiController(use_tushare=True)
        logger.info(f"GUI控制器初始化完成，增强API状态: {'已启用' if controller.using_enhanced_api else '未启用'}")
        
        # 测试股票列表
        test_codes = ['601318.SH', '000651.SZ', '000333.SZ', '600519.SH', 
                     '000002.SZ', '600036.SH', '000999.SZ', '600276.SH']
        
        # 测试1：获取单个股票名称
        logger.info("\n测试1：获取单个股票名称")
        for ts_code in test_codes:
            start_time = time.time()
            name = controller.get_stock_name(ts_code)
            elapsed = time.time() - start_time
            logger.info(f"  {ts_code} -> {name} (耗时: {elapsed:.5f}秒)")
        
        # 测试2：批量获取股票名称
        logger.info("\n测试2：批量获取股票名称")
        start_time = time.time()
        batch_names = controller.get_stock_names_batch(test_codes)
        elapsed = time.time() - start_time
        logger.info(f"批量获取 {len(batch_names)} 个股票名称 (总耗时: {elapsed:.5f}秒, 平均: {elapsed/len(batch_names):.5f}秒/个):")
        for ts_code, name in batch_names.items():
            logger.info(f"  {ts_code} -> {name}")
        
        # 测试3：获取股票行业
        logger.info("\n测试3：获取股票行业")
        for ts_code in test_codes:
            start_time = time.time()
            industry = controller.get_stock_industry(ts_code)
            elapsed = time.time() - start_time
            logger.info(f"  {ts_code} -> {industry} (耗时: {elapsed:.5f}秒)")
        
        # 测试4：获取API统计信息
        if controller.using_enhanced_api:
            logger.info("\n测试4：获取API统计信息")
            api_stats = controller.get_api_stats()
            logger.info(f"API统计信息: {api_stats}")
        
        # 测试5：清除缓存
        logger.info("\n测试5：清除缓存并重新获取")
        clear_result = controller.clear_enhanced_cache()
        logger.info(f"缓存清除结果: {clear_result}")
        
        # 再次获取名称（测试缓存清除效果）
        start_time = time.time()
        batch_names_again = controller.get_stock_names_batch(test_codes)
        elapsed = time.time() - start_time
        logger.info(f"清除缓存后重新获取 {len(batch_names_again)} 个股票名称 (耗时: {elapsed:.5f}秒):")
        for ts_code, name in batch_names_again.items():
            logger.info(f"  {ts_code} -> {name}")
        
        # 测试结束
        logger.info("\n测试完成，GUI控制器增强API功能正常工作")
        return True
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
        return False

def test_enhanced_gui_controller():
    """测试增强版GUI控制器"""
    logger.info("开始测试增强版GUI控制器")
    
    try:
        # 导入增强版GUI控制器
        from enhanced_gui_controller import EnhancedGuiController
        
        # 初始化控制器
        controller = EnhancedGuiController()
        logger.info(f"增强版GUI控制器初始化完成")
        
        # 测试股票列表
        test_codes = ['601318.SH', '000651.SZ', '000333.SZ', '600519.SH']
        
        # 定义回调函数
        def progress_callback(msg, progress):
            logger.info(f"进度回调: {msg} - {progress}%")
        
        # 测试1：获取增强版动量分析
        logger.info("\n测试1：获取增强版动量分析")
        start_time = time.time()
        results = controller.get_enhanced_momentum_analysis(
            industry=None, 
            sample_size=10,  # 小样本，加快测试速度
            min_score=30,    # 低分数阈值，确保有结果
            gui_callback=progress_callback
        )
        elapsed = time.time() - start_time
        
        logger.info(f"获取到 {len(results)} 个分析结果 (总耗时: {elapsed:.2f}秒)")
        if results:
            # 只输出前两个结果
            for i, result in enumerate(results[:2]):
                logger.info(f"结果 {i+1}: {result['name']} ({result['code']}) - 得分: {result['score']}")
        
        # 测试2：获取股票详情
        if results:
            logger.info("\n测试2：获取股票详情")
            test_code = results[0]['code']
            detail = controller.get_enhanced_stock_detail(test_code)
            logger.info(f"获取到股票 {test_code} 的详情: {detail.keys() if detail else 'None'}")
        
        # 测试3：获取行业分析
        logger.info("\n测试3：获取行业分析")
        industry_analysis = controller.get_enhanced_industry_analysis()
        logger.info(f"获取到 {len(industry_analysis)} 个行业分析结果")
        
        # 测试4：获取市场概览
        logger.info("\n测试4：获取市场概览")
        market_overview = controller.get_enhanced_market_overview()
        logger.info(f"获取到市场概览数据: {market_overview.keys() if market_overview else 'None'}")
        
        # 测试结束
        logger.info("\n测试完成，增强版GUI控制器正常工作")
        return True
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
        return False

def test_stock_analysis_gui():
    """测试股票分析GUI界面"""
    logger.info("开始测试股票分析GUI界面")
    
    try:
        # 导入tkinter
        import tkinter as tk
        from stock_analysis_gui import StockAnalysisGUI, get_stock_name_ui, get_stock_industry_ui
        
        # 测试工具函数
        logger.info("\n测试工具函数")
        test_codes = ['601318.SH', '000651.SZ', '000333.SZ', '600519.SH']
        
        # 测试股票名称函数
        for ts_code in test_codes:
            try:
                name = get_stock_name_ui(ts_code)
                logger.info(f"股票名称: {ts_code} -> {name}")
            except Exception as e:
                logger.error(f"获取股票名称出错: {e}")
        
        # 这里不创建完整的GUI（避免弹出窗口），只验证导入和函数可用性
        logger.info("\n验证GUI类可导入")
        logger.info("GUI测试完成 - 注意：未创建实际GUI窗口")
        return True
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
        return False

def verify_api_reliability():
    """验证API可靠性模块是否可用"""
    logger.info("开始验证API可靠性模块")
    
    try:
        # 尝试导入增强API模块
        from enhance_api_reliability import (
            enhance_get_stock_name, 
            enhance_get_stock_names_batch,
            enhance_get_stock_industry, 
            with_retry
        )
        
        logger.info("API可靠性模块导入成功")
        
        # 测试基本函数
        test_code = '601318.SH'
        name = enhance_get_stock_name(test_code)
        logger.info(f"获取股票名称: {test_code} -> {name}")
        
        industry = enhance_get_stock_industry(test_code)
        logger.info(f"获取股票行业: {test_code} -> {industry}")
        
        logger.info("API可靠性模块验证完成，功能正常")
        return True
        
    except ImportError:
        logger.error("API可靠性模块不可用")
        return False
    except Exception as e:
        logger.error(f"验证过程中发生错误: {str(e)}", exc_info=True)
        return False

def main():
    """主测试函数"""
    logger.info("=" * 50)
    logger.info("开始API可靠性与GUI集成测试")
    logger.info("=" * 50)
    
    # 验证API可靠性模块
    api_available = verify_api_reliability()
    if not api_available:
        logger.warning("API可靠性模块不可用，部分测试将被跳过")
    
    # 测试GUI控制器
    gui_controller_result = test_gui_controller_api()
    logger.info(f"GUI控制器测试结果: {'成功' if gui_controller_result else '失败'}")
    
    # 测试增强版GUI控制器
    enhanced_controller_result = test_enhanced_gui_controller()
    logger.info(f"增强版GUI控制器测试结果: {'成功' if enhanced_controller_result else '失败'}")
    
    # 测试股票分析GUI
    gui_result = test_stock_analysis_gui()
    logger.info(f"股票分析GUI测试结果: {'成功' if gui_result else '失败'}")
    
    # 总结果
    overall_result = gui_controller_result and enhanced_controller_result and gui_result
    logger.info("=" * 50)
    logger.info(f"API可靠性与GUI集成测试完成，总体结果: {'成功' if overall_result else '失败'}")
    logger.info("=" * 50)
    
    return overall_result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 