#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AKShare API 测试脚本
用于测试AKShare API连接是否正常，获取股票市场数据
"""

import logging
import sys
import pandas as pd
import time
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_akshare_api():
    """测试AKShare API连接和数据获取"""
    try:
        import akshare as ak
        
        logger.info("AKShare版本: " + ak.__version__)
        
        # 测试1: 获取股票行情
        logger.info("测试1: 获取上证指数行情...")
        stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol="sh000001")
        if stock_zh_index_daily_df is not None and not stock_zh_index_daily_df.empty:
            logger.info(f"成功获取上证指数行情，共{len(stock_zh_index_daily_df)}条记录")
            logger.info(f"最新收盘价: {stock_zh_index_daily_df.iloc[-1]['close']}")
            logger.info(f"最近数据:\n{stock_zh_index_daily_df.tail(3)}")
        else:
            logger.error("获取上证指数行情失败")
        
        # 测试2: 获取沪深300成分股
        logger.info("\n测试2: 获取沪深300成分股...")
        try:
            stock_index_300_weight_df = ak.stock_zh_index_weight(index_symbol="000300")
            if stock_index_300_weight_df is not None and not stock_index_300_weight_df.empty:
                logger.info(f"成功获取沪深300成分股，共{len(stock_index_300_weight_df)}条记录")
                logger.info(f"部分成份股:\n{stock_index_300_weight_df.head(5)}")
            else:
                logger.error("获取沪深300成分股失败")
        except Exception as e:
            logger.error(f"获取沪深300成分股出错: {str(e)}")
            # 尝试替代接口
            try:
                logger.info("尝试使用替代接口获取沪深300成分股...")
                stock_index_300_weight_df = ak.index_stock_cons(symbol="399300")
                if stock_index_300_weight_df is not None and not stock_index_300_weight_df.empty:
                    logger.info(f"成功获取沪深300成分股(替代接口)，共{len(stock_index_300_weight_df)}条记录")
                    logger.info(f"部分成份股:\n{stock_index_300_weight_df.head(5)}")
                else:
                    logger.error("获取沪深300成分股(替代接口)失败")
            except Exception as e2:
                logger.error(f"获取沪深300成分股(替代接口)出错: {str(e2)}")
        
        # 测试3: 获取A股当前行情信息
        logger.info("\n测试3: 获取A股当前行情信息...")
        try:
            stock_zh_a_spot_df = ak.stock_zh_a_spot()
            if stock_zh_a_spot_df is not None and not stock_zh_a_spot_df.empty:
                logger.info(f"成功获取A股当前行情，共{len(stock_zh_a_spot_df)}只股票")
                logger.info(f"部分股票行情:\n{stock_zh_a_spot_df.head(3)}")
            else:
                logger.error("获取A股当前行情失败")
        except Exception as e:
            logger.error(f"获取A股当前行情出错: {str(e)}")
        
        # 测试4: 获取个股日线数据
        logger.info("\n测试4: 获取个股历史行情数据...")
        try:
            stock_history_df = ak.stock_zh_a_hist(symbol="000001", period="daily", 
                                               start_date="20250401", end_date="20250430")
            if stock_history_df is not None and not stock_history_df.empty:
                logger.info(f"成功获取平安银行(000001)历史行情，共{len(stock_history_df)}条记录")
                logger.info(f"部分历史数据:\n{stock_history_df.head(3)}")
            else:
                logger.error("获取平安银行历史行情失败")
        except Exception as e:
            logger.error(f"获取个股历史行情出错: {str(e)}")
        
        # 测试5: 获取宏观经济指标
        logger.info("\n测试5: 获取GDP数据...")
        try:
            macro_china_gdp = ak.macro_china_gdp_yearly()
            if macro_china_gdp is not None and not macro_china_gdp.empty:
                logger.info(f"成功获取中国GDP年度数据，共{len(macro_china_gdp)}条记录")
                logger.info(f"最近GDP数据:\n{macro_china_gdp.tail(3)}")
            else:
                logger.error("获取GDP数据失败")
        except Exception as e:
            logger.error(f"获取GDP数据出错: {str(e)}")
        
        # 测试6: 获取北向资金数据
        logger.info("\n测试6: 获取北向资金数据...")
        try:
            stock_em_hsgt_df = ak.stock_em_hsgt_north_net_flow_in()
            if stock_em_hsgt_df is not None and not stock_em_hsgt_df.empty:
                logger.info(f"成功获取北向资金流入数据，共{len(stock_em_hsgt_df)}条记录")
                logger.info(f"最近北向资金数据:\n{stock_em_hsgt_df.head(3)}")
            else:
                logger.error("获取北向资金数据失败")
        except Exception as e:
            logger.error(f"获取北向资金数据出错: {str(e)}")
            # 尝试替代接口
            try:
                logger.info("尝试使用替代接口获取北向资金数据...")
                stock_em_hsgt_hist_df = ak.stock_em_hsgt_hist_em()
                if stock_em_hsgt_hist_df is not None and not stock_em_hsgt_hist_df.empty:
                    logger.info(f"成功获取北向资金历史数据(替代接口)，共{len(stock_em_hsgt_hist_df)}条记录")
                    logger.info(f"最近北向资金数据:\n{stock_em_hsgt_hist_df.head(3)}")
                else:
                    logger.error("获取北向资金数据(替代接口)失败")
            except Exception as e2:
                logger.error(f"获取北向资金数据(替代接口)出错: {str(e2)}")
        
        logger.info("\nAKShare API 测试完成!")
        return True
    
    except Exception as e:
        logger.error(f"AKShare API 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_akshare_api() 