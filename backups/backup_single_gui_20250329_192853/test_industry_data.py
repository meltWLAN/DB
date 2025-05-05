"""
测试东方财富行业数据获取
单独测试行业列表和行业成分股获取功能
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 尝试导入akshare库
try:
    import akshare as ak
    HAS_AKSHARE = True
    logger.info("成功导入akshare库")
except ImportError:
    HAS_AKSHARE = False
    logger.error("未安装akshare库，请使用pip install akshare安装")
    sys.exit(1)

def get_industry_list():
    """获取东方财富行业分类列表"""
    try:
        logger.info("开始获取东方财富行业分类列表...")
        industry_df = ak.stock_board_industry_name_em()
        
        if industry_df is not None and not industry_df.empty:
            logger.info(f"成功获取 {len(industry_df)} 个行业分类")
            logger.info(f"行业分类数据示例：\n{industry_df.head()}")
            
            # 保存行业分类数据
            os.makedirs("./data", exist_ok=True)
            industry_df.to_csv("./data/industry_list.csv", index=False, encoding="utf-8-sig")
            logger.info("行业分类数据已保存到 ./data/industry_list.csv")
            
            return industry_df
        else:
            logger.warning("获取的行业分类列表为空")
            return None
    except Exception as e:
        logger.error(f"获取行业分类列表出错: {str(e)}")
        return None

def get_industry_stocks(industry_name):
    """获取指定行业的成分股"""
    try:
        logger.info(f"开始获取 {industry_name} 行业的成分股...")
        industry_stocks = ak.stock_board_industry_cons_em(symbol=industry_name)
        
        if industry_stocks is not None and not industry_stocks.empty:
            logger.info(f"成功获取 {len(industry_stocks)} 支 {industry_name} 行业的股票")
            logger.info(f"行业成分股数据示例：\n{industry_stocks.head()}")
            
            # 转换股票代码格式
            industry_stocks["ts_code"] = industry_stocks["代码"].apply(
                lambda x: f"{x}.SH" if x.startswith(("6", "9")) else f"{x}.SZ"
            )
            
            # 保存行业成分股数据
            os.makedirs("./data/industry_stocks", exist_ok=True)
            industry_stocks.to_csv(f"./data/industry_stocks/{industry_name}.csv", index=False, encoding="utf-8-sig")
            logger.info(f"{industry_name} 行业成分股数据已保存到 ./data/industry_stocks/{industry_name}.csv")
            
            return industry_stocks
        else:
            logger.warning(f"获取的 {industry_name} 行业成分股列表为空")
            return None
    except Exception as e:
        logger.error(f"获取 {industry_name} 行业成分股出错: {str(e)}")
        return None

def test_get_stock_data(stock_code):
    """测试获取单个股票的数据"""
    try:
        logger.info(f"开始获取 {stock_code} 的历史行情数据...")
        # 获取股票日线数据
        stock_data = ak.stock_zh_a_hist(symbol=stock_code.split('.')[0], period="daily", 
                                      start_date="20230101", end_date="20240101", adjust="qfq")
        
        if stock_data is not None and not stock_data.empty:
            logger.info(f"成功获取 {stock_code} 的历史行情数据，共 {len(stock_data)} 条记录")
            logger.info(f"历史行情数据示例：\n{stock_data.head()}")
            
            # 保存股票数据
            os.makedirs("./data/stock_data", exist_ok=True)
            stock_data.to_csv(f"./data/stock_data/{stock_code}.csv", index=False, encoding="utf-8-sig")
            logger.info(f"{stock_code} 历史行情数据已保存到 ./data/stock_data/{stock_code}.csv")
            
            return stock_data
        else:
            logger.warning(f"获取的 {stock_code} 历史行情数据为空")
            return None
    except Exception as e:
        logger.error(f"获取 {stock_code} 历史行情数据出错: {str(e)}")
        return None

def main():
    """主函数"""
    logger.info("=== 开始测试东方财富行业数据获取 ===")
    
    # 1. 获取行业列表
    industry_df = get_industry_list()
    if industry_df is None:
        logger.error("获取行业列表失败，测试终止")
        return
    
    # 2. 选择一些行业进行测试
    test_industries = ["证券", "银行", "电子元件", "汽车整车"]
    for industry in test_industries:
        industry_stocks = get_industry_stocks(industry)
        if industry_stocks is not None:
            # 3. 测试获取行业中第一只股票的数据
            if len(industry_stocks) > 0:
                stock_code = industry_stocks["代码"].iloc[0]
                test_get_stock_data(stock_code)
    
    logger.info("=== 测试完成 ===")

if __name__ == "__main__":
    main() 