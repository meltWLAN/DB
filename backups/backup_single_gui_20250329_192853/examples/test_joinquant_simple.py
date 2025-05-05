#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单JoinQuant测试脚本
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加当前目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from src.config import DATA_SOURCE_CONFIG

def test_jq_account():
    """测试JoinQuant账号"""
    logger.info("=== 测试JoinQuant账号权限 ===")
    
    # 获取JoinQuant配置
    jq_config = DATA_SOURCE_CONFIG.get('joinquant', {})
    username = jq_config.get('username', '')
    password = jq_config.get('password', '')
    
    if not username or not password:
        logger.error("JoinQuant账号未配置，请在src/config/__init__.py中设置")
        return False
    
    logger.info(f"使用账号: {username}")
    
    # 导入JoinQuant SDK
    try:
        import jqdatasdk as jq
    except ImportError:
        logger.error("未安装jqdatasdk，请先安装: pip install jqdatasdk")
        return False
    
    # 登录验证
    try:
        logger.info("尝试登录JoinQuant...")
        jq.auth(username, password)
        logger.info("登录成功!")
        
        # 获取账号信息
        account_info = jq.get_account_info()
        logger.info(f"账号信息: {account_info}")
        
        # 获取剩余调用次数（接口可能已更新）
        try:
            quota = jq.get_query_count()
            logger.info(f"剩余可用额度: {quota}")
        except Exception as e:
            logger.warning(f"获取额度信息失败: {e}，继续测试...")
        
        # 测试获取股票列表
        logger.info("获取股票列表...")
        stocks = jq.get_all_securities(['stock'])
        logger.info(f"成功获取 {len(stocks)} 只股票")
        logger.info(f"前5只股票: {list(stocks.index[:5])}")
        
        # 获取单只股票数据
        stock_code = '000001.XSHE'  # 平安银行
        end_date = datetime.strptime('2024-12-20', '%Y-%m-%d')
        start_date = end_date - timedelta(days=10)
        
        logger.info(f"获取 {stock_code} 从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的数据...")
        df = jq.get_price(stock_code, start_date=start_date, end_date=end_date, 
                          frequency='daily', fields=['open', 'close', 'high', 'low', 'volume'])
        logger.info(f"成功获取 {len(df)} 条记录")
        if not df.empty:
            logger.info(f"最新价格: {df['close'].iloc[-1]}")
        
        # 获取基本面数据
        logger.info("获取公司基本信息...")
        try:
            q = jq.query(jq.finance.STK_COMPANY_INFO).filter(jq.finance.STK_COMPANY_INFO.code == stock_code)
            df_company = jq.finance.run_query(q)
            if not df_company.empty:
                logger.info(f"获取到公司基本信息，列名: {list(df_company.columns)}")
                # 打印前几行数据作为参考
                logger.info(f"公司信息概览: \n{df_company.iloc[0].to_dict()}")
        except Exception as e:
            logger.warning(f"获取公司基本信息失败: {str(e)}")
        
        # 登出
        jq.logout()
        logger.info("JoinQuant测试完成，已登出")
        return True
        
    except Exception as e:
        logger.error(f"JoinQuant测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print(" JoinQuant 账号权限测试 ".center(80, "="))
    print("=" * 80)
    
    # 运行测试
    success = test_jq_account()
    
    print("=" * 80)
    print(f" 测试结果: {'成功' if success else '失败'} ".center(80, "="))
    print("=" * 80) 