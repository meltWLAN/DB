"""
行业数据提供类
提供行业分类和行业成分股数据，用于在GUI中使用
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# 确保日志目录存在
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志
log_file = os.path.join(LOG_DIR, f"industry_data_provider_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 尝试导入akshare库
try:
    import akshare as ak
    HAS_AKSHARE = True
    logger.info("成功导入akshare库")
except ImportError:
    HAS_AKSHARE = False
    logger.warning("未安装akshare库，将使用本地行业数据")

class IndustryDataProvider:
    """行业数据提供类"""
    
    def __init__(self, use_local=True):
        """初始化行业数据提供类
        
        Args:
            use_local: 是否优先使用本地数据
        """
        self.use_local = use_local
        self.data_dir = "./data/industry_data"
        self.industry_list = None
        self.industry_stocks_dict = {}
        self.stock_industry_mapping = None
        
        # 加载数据
        self.load_industry_list()
        self.load_stock_industry_mapping()
    
    def load_industry_list(self):
        """加载行业列表"""
        try:
            # 检查本地数据
            industry_csv = os.path.join(self.data_dir, "industry_list.csv")
            
            if os.path.exists(industry_csv) and self.use_local:
                logger.info(f"从本地加载行业分类数据: {industry_csv}")
                # 读取本地数据
                self.industry_list = pd.read_csv(industry_csv, encoding='utf-8-sig')
                logger.info(f"成功加载本地行业分类数据，共 {len(self.industry_list)} 个行业")
                return True
            
            # 如果没有本地数据或不使用本地数据，则获取新数据
            if HAS_AKSHARE:
                logger.info("在线获取东方财富行业分类列表...")
                self.industry_list = ak.stock_board_industry_name_em()
                
                if self.industry_list is not None and not self.industry_list.empty:
                    logger.info(f"成功获取 {len(self.industry_list)} 个行业分类")
                    
                    # 保存行业分类数据
                    os.makedirs(self.data_dir, exist_ok=True)
                    self.industry_list.to_csv(industry_csv, index=False, encoding="utf-8-sig")
                    logger.info(f"行业分类数据已保存到 {industry_csv}")
                    
                    return True
                else:
                    logger.warning("获取的行业分类列表为空")
                    return False
            else:
                logger.error("未安装akshare库且没有本地行业数据")
                return False
        except Exception as e:
            logger.error(f"加载行业分类数据出错: {str(e)}")
            return False
    
    def load_stock_industry_mapping(self):
        """加载股票行业映射文件"""
        try:
            mapping_path = os.path.join(self.data_dir, "stock_industry_mapping.csv")
            
            if os.path.exists(mapping_path) and self.use_local:
                logger.info(f"从本地加载股票行业映射数据: {mapping_path}")
                self.stock_industry_mapping = pd.read_csv(mapping_path, encoding='utf-8-sig')
                logger.info(f"成功加载本地股票行业映射数据，共 {len(self.stock_industry_mapping)} 条记录")
                return True
            else:
                logger.warning(f"未找到股票行业映射文件: {mapping_path}")
                return False
        except Exception as e:
            logger.error(f"加载股票行业映射数据出错: {str(e)}")
            return False
    
    def get_industry_list(self):
        """获取行业列表
        
        Returns:
            list: 行业名称列表，如果加载失败则返回空列表
        """
        if self.industry_list is None:
            if not self.load_industry_list():
                return []
        
        # 返回行业名称列表，添加"全部"选项
        if "板块名称" in self.industry_list.columns:
            return ["全部"] + self.industry_list["板块名称"].tolist()
        else:
            return []
    
    def get_industry_stocks(self, industry_name):
        """获取指定行业的成分股
        
        Args:
            industry_name: 行业名称，如果为"全部"则返回所有股票
            
        Returns:
            DataFrame: 行业成分股数据，包含ts_code、name和industry列
        """
        if industry_name == "全部":
            if self.stock_industry_mapping is not None:
                return self.stock_industry_mapping
            else:
                return pd.DataFrame(columns=["ts_code", "name", "industry"])
        
        # 检查是否已经缓存
        if industry_name in self.industry_stocks_dict:
            return self.industry_stocks_dict[industry_name]
        
        try:
            # 先尝试从本地加载
            if self.use_local:
                industry_csv = os.path.join(self.data_dir, "industry_stocks", f"{industry_name}.csv")
                
                if os.path.exists(industry_csv):
                    logger.info(f"从本地加载 {industry_name} 行业成分股数据")
                    industry_stocks = pd.read_csv(industry_csv, encoding='utf-8-sig')
                    
                    # 确保数据格式正确
                    if 'ts_code' not in industry_stocks.columns:
                        if '代码' in industry_stocks.columns:
                            industry_stocks["ts_code"] = industry_stocks["代码"].apply(
                                lambda x: f"{x}.SH" if str(x).startswith(("6", "9")) else f"{x}.SZ"
                            )
                        else:
                            logger.warning(f"{industry_name} 行业数据中未找到股票代码列")
                            return pd.DataFrame(columns=["ts_code", "name", "industry"])
                    
                    # 确保有name列
                    if 'name' not in industry_stocks.columns and '名称' in industry_stocks.columns:
                        industry_stocks["name"] = industry_stocks["名称"]
                    
                    # 添加行业列
                    industry_stocks["industry"] = industry_name
                    
                    # 只保留需要的列
                    result = industry_stocks[["ts_code", "name", "industry"]]
                    
                    # 缓存结果
                    self.industry_stocks_dict[industry_name] = result
                    
                    return result
            
            # 如果本地没有或不使用本地，则尝试在线获取
            if HAS_AKSHARE:
                logger.info(f"在线获取 {industry_name} 行业的成分股...")
                industry_stocks = ak.stock_board_industry_cons_em(symbol=industry_name)
                
                if industry_stocks is not None and not industry_stocks.empty:
                    logger.info(f"成功获取 {len(industry_stocks)} 支 {industry_name} 行业的股票")
                    
                    # 转换股票代码格式
                    industry_stocks["ts_code"] = industry_stocks["代码"].apply(
                        lambda x: f"{x}.SH" if str(x).startswith(("6", "9")) else f"{x}.SZ"
                    )
                    
                    # 设置name列
                    industry_stocks["name"] = industry_stocks["名称"]
                    
                    # 添加行业列
                    industry_stocks["industry"] = industry_name
                    
                    # 只保留需要的列
                    result = industry_stocks[["ts_code", "name", "industry"]]
                    
                    # 保存到本地
                    os.makedirs(os.path.join(self.data_dir, "industry_stocks"), exist_ok=True)
                    csv_path = os.path.join(self.data_dir, "industry_stocks", f"{industry_name}.csv")
                    industry_stocks.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    logger.info(f"{industry_name} 行业成分股数据已保存到 {csv_path}")
                    
                    # 缓存结果
                    self.industry_stocks_dict[industry_name] = result
                    
                    return result
                else:
                    logger.warning(f"获取的 {industry_name} 行业成分股列表为空")
            
            # 如果仍然没有数据，尝试使用映射文件筛选
            if self.stock_industry_mapping is not None:
                industry_stocks = self.stock_industry_mapping[
                    self.stock_industry_mapping["industry"] == industry_name
                ]
                
                if not industry_stocks.empty:
                    logger.info(f"从映射文件中找到 {len(industry_stocks)} 支 {industry_name} 行业的股票")
                    
                    # 缓存结果
                    self.industry_stocks_dict[industry_name] = industry_stocks
                    
                    return industry_stocks
            
            # 如果所有方法都失败，返回空DataFrame
            logger.warning(f"未能获取 {industry_name} 行业的成分股数据")
            return pd.DataFrame(columns=["ts_code", "name", "industry"])
            
        except Exception as e:
            logger.error(f"获取 {industry_name} 行业成分股时出错: {str(e)}")
            return pd.DataFrame(columns=["ts_code", "name", "industry"])
    
    def get_stock_industry(self, ts_code):
        """获取指定股票的行业
        
        Args:
            ts_code: 股票代码
            
        Returns:
            str: 行业名称，如果未找到则返回空字符串
        """
        if self.stock_industry_mapping is None:
            if not self.load_stock_industry_mapping():
                return ""
        
        try:
            stock_data = self.stock_industry_mapping[
                self.stock_industry_mapping["ts_code"] == ts_code
            ]
            
            if not stock_data.empty:
                return stock_data.iloc[0]["industry"]
            else:
                return ""
        except Exception as e:
            logger.error(f"获取股票 {ts_code} 的行业信息时出错: {str(e)}")
            return ""

# 用于测试
if __name__ == "__main__":
    provider = IndustryDataProvider(use_local=True)
    
    # 测试获取行业列表
    industry_list = provider.get_industry_list()
    print(f"行业列表(前10个): {industry_list[:10]}")
    
    # 测试获取行业成分股
    if industry_list:
        test_industry = industry_list[1]  # 跳过"全部"
        stocks = provider.get_industry_stocks(test_industry)
        print(f"\n{test_industry}行业成分股(前5个):")
        print(stocks.head())
    
    # 测试获取股票行业
    test_stock = "601318.SH"  # 中国平安
    industry = provider.get_stock_industry(test_stock)
    print(f"\n股票{test_stock}的行业: {industry}") 