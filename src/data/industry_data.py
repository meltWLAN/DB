#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tushare as ts
import pandas as pd
import os
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("industry_data")

# 缓存目录设置
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cache", "industry")
os.makedirs(CACHE_DIR, exist_ok=True)

# Tushare Token配置 (使用环境变量或配置文件)
def init_tushare_api():
    """初始化Tushare API连接"""
    # 直接使用提供的Tushare token
    token = '0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10'
    logger.info("使用直接配置的Tushare token")
    
    if token:
        try:
            ts.set_token(token)
            pro = ts.pro_api()
            # 简单测试API连接
            pro.trade_cal(exchange='SSE', start_date='20210101', end_date='20210101')
            logger.info("Tushare API 初始化成功")
            return pro
        except Exception as e:
            logger.error(f"Tushare API 初始化失败: {e}")
            return None
    else:
        logger.error("未找到Tushare Token, 请设置环境变量TUSHARE_TOKEN或在config.json中配置")
        return None

# 缓存相关函数
def save_cache(data, cache_name):
    """保存数据到缓存文件"""
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    try:
        data.to_pickle(cache_path)
        # 同时保存更新时间
        with open(os.path.join(CACHE_DIR, f"{cache_name}_time.txt"), 'w') as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info(f"数据已缓存到 {cache_path}")
        return True
    except Exception as e:
        logger.error(f"缓存数据失败: {e}")
        return False

def load_cache(cache_name, max_age_hours=24):
    """从缓存加载数据，如果缓存过期则返回None"""
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    time_path = os.path.join(CACHE_DIR, f"{cache_name}_time.txt")
    
    if not os.path.exists(cache_path):
        logger.info(f"缓存文件不存在: {cache_path}")
        return None
    
    # 检查缓存是否过期
    cache_expired = True
    if os.path.exists(time_path):
        try:
            with open(time_path, 'r') as f:
                cache_time = datetime.strptime(f.read().strip(), '%Y-%m-%d %H:%M:%S')
                if datetime.now() - cache_time < timedelta(hours=max_age_hours):
                    cache_expired = False
        except Exception as e:
            logger.warning(f"读取缓存时间失败: {e}")
    
    if cache_expired:
        logger.info(f"缓存已过期: {cache_path}")
        return None
    
    try:
        data = pd.read_pickle(cache_path)
        logger.info(f"从缓存加载数据: {cache_path}")
        return data
    except Exception as e:
        logger.error(f"读取缓存失败: {e}")
        return None

# 数据获取函数
def get_industry_list():
    """获取行业列表"""
    # 先尝试从缓存加载
    cached_data = load_cache("industry_list")
    if cached_data is not None:
        return cached_data
    
    # 缓存未命中，从Tushare获取
    pro = init_tushare_api()
    if pro is None:
        return pd.DataFrame(columns=['index_code', 'industry_name'])
    
    try:
        # 方法1：尝试使用申万一级行业分类
        try:
            sw_df = pro.index_classify(level='L1', src='sw')
            if sw_df is not None and not sw_df.empty:
                # 数据清洗和格式化
                sw_df = sw_df[['index_code', 'industry_name']].copy()
                # 缓存数据
                save_cache(sw_df, "industry_list")
                logger.info(f"成功获取 {len(sw_df)} 个申万行业分类")
                return sw_df
        except Exception as e:
            logger.warning(f"使用申万行业分类API失败: {e}")
        
        # 方法2：使用股票基础信息中的行业字段
        try:
            stock_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
            if stock_info is not None and not stock_info.empty and 'industry' in stock_info.columns:
                # 提取唯一的行业名称
                unique_industries = stock_info['industry'].dropna().unique()
                # 创建一个行业代码-行业名称的映射
                industry_df = pd.DataFrame({
                    'index_code': [f'IND{i:03d}' for i in range(len(unique_industries))],
                    'industry_name': unique_industries
                })
                
                # 缓存数据
                save_cache(industry_df, "industry_list")
                logger.info(f"成功从股票基础信息获取 {len(industry_df)} 个行业分类")
                return industry_df
        except Exception as e:
            logger.warning(f"从股票基础信息获取行业分类失败: {e}")
        
        # 返回空DataFrame
        logger.warning("通过多种方法未能获取行业分类数据")
        return pd.DataFrame(columns=['index_code', 'industry_name'])
    except Exception as e:
        logger.error(f"获取行业分类失败: {e}")
        return pd.DataFrame(columns=['index_code', 'industry_name'])

def get_industry_stocks(industry_code):
    """获取指定行业的成分股"""
    # 参数检查
    if not industry_code:
        logger.error("行业代码不能为空")
        return pd.DataFrame(columns=['stock_code', 'stock_name'])
    
    # 先尝试从缓存加载
    cache_name = f"industry_stocks_{industry_code}"
    cached_data = load_cache(cache_name)
    if cached_data is not None:
        return cached_data
    
    # 缓存未命中，从Tushare获取
    pro = init_tushare_api()
    if pro is None:
        return pd.DataFrame(columns=['stock_code', 'stock_name'])
    
    try:
        # 判断是否是自定义行业代码
        if industry_code.startswith('IND'):
            # 获取行业列表
            industry_df = get_industry_list()
            if industry_df.empty:
                logger.warning("无法获取行业列表")
                return pd.DataFrame(columns=['stock_code', 'stock_name'])
            
            # 查找对应的行业名称
            industry_info = industry_df[industry_df['index_code'] == industry_code]
            if industry_info.empty:
                logger.warning(f"未找到行业代码: {industry_code}")
                return pd.DataFrame(columns=['stock_code', 'stock_name'])
            
            industry_name = industry_info.iloc[0]['industry_name']
            
            # 通过行业名称获取股票
            stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
            if stock_basic is None or stock_basic.empty:
                logger.warning("获取股票基础信息失败")
                return pd.DataFrame(columns=['stock_code', 'stock_name'])
            
            # 筛选特定行业的股票
            stocks_df = stock_basic[stock_basic['industry'] == industry_name][['ts_code', 'name']]
            if stocks_df.empty:
                logger.warning(f"未找到行业 {industry_name} 的股票")
                return pd.DataFrame(columns=['stock_code', 'stock_name'])
            
            # 重命名列
            stocks_df = stocks_df.rename(columns={'ts_code': 'stock_code', 'name': 'stock_name'})
            
            # 缓存数据
            save_cache(stocks_df, cache_name)
            logger.info(f"成功获取行业 {industry_name}({industry_code}) 的 {len(stocks_df)} 只成分股")
            return stocks_df
        else:
            # 尝试使用原始申万行业代码获取成分股
            try:
                # 获取行业成分股
                stocks_df = pro.index_member(index_code=industry_code)
                
                if stocks_df is not None and not stocks_df.empty:
                    # 数据清洗和格式化
                    if 'con_code' in stocks_df.columns:
                        # 进一步获取股票名称
                        stock_codes = stocks_df['con_code'].tolist()
                        stock_info = pro.stock_basic(ts_code=','.join(stock_codes), fields='ts_code,name')
                        
                        if stock_info is not None and not stock_info.empty:
                            # 合并股票代码和名称
                            stocks_df = pd.merge(
                                stocks_df, 
                                stock_info, 
                                left_on='con_code', 
                                right_on='ts_code', 
                                how='left'
                            )
                            # 只保留需要的列并重命名
                            stocks_df = stocks_df[['con_code', 'name']].rename(
                                columns={'con_code': 'stock_code', 'name': 'stock_name'}
                            )
                        else:
                            # 如果无法获取名称，只保留代码
                            stocks_df = stocks_df[['con_code']].rename(columns={'con_code': 'stock_code'})
                            stocks_df['stock_name'] = ''
                    else:
                        logger.warning(f"行业成分股数据缺少con_code列: {stocks_df.columns}")
                        return pd.DataFrame(columns=['stock_code', 'stock_name'])
                    
                    # 缓存数据
                    save_cache(stocks_df, cache_name)
                    logger.info(f"成功获取行业代码 {industry_code} 的 {len(stocks_df)} 只成分股")
                    return stocks_df
                else:
                    logger.warning(f"未能获取行业 {industry_code} 的成分股数据")
                    return pd.DataFrame(columns=['stock_code', 'stock_name'])
            except Exception as e:
                logger.warning(f"使用index_member获取行业 {industry_code} 的成分股失败: {e}")
                return pd.DataFrame(columns=['stock_code', 'stock_name'])
    except Exception as e:
        logger.error(f"获取行业 {industry_code} 的成分股失败: {e}")
        return pd.DataFrame(columns=['stock_code', 'stock_name'])

def get_simplified_industry_list():
    """获取简化版的行业列表（仅包含名称）"""
    # 先尝试从缓存加载
    cached_data = load_cache("industry_list")
    if cached_data is not None:
        # 如果缓存中有数据，直接返回行业名称列表
        return cached_data['industry_name'].tolist() if 'industry_name' in cached_data.columns else []
    
    try:
        # 使用绝对导入
        import sys
        import os
        
        # 添加项目根目录到路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        # 导入DataFetcher
        from src.data.data_fetcher import DataFetcher
        
        # 创建数据获取器并获取行业列表
        fetcher = DataFetcher()
        industry_df = fetcher.get_industry_list()
        
        if industry_df.empty:
            # 如果无法获取数据，返回一些常见行业作为后备
            fallback_industries = [
                "银行", "证券", "保险", "房地产", "医药", "电子", 
                "计算机", "通信", "有色金属", "钢铁", "煤炭", "石油石化",
                "电力", "建筑", "军工", "汽车", "家电", "食品饮料"
            ]
            return fallback_industries
        else:
            # 缓存数据
            save_cache(industry_df, "industry_list")
            return industry_df['industry_name'].tolist()
    except Exception as e:
        logger.error(f"获取行业列表失败: {e}")
        # 如果失败，返回后备行业列表
        fallback_industries = [
            "银行", "证券", "保险", "房地产", "医药", "电子", 
            "计算机", "通信", "有色金属", "钢铁", "煤炭", "石油石化",
            "电力", "建筑", "军工", "汽车", "家电", "食品饮料"
        ]
        return fallback_industries

def get_stocks_by_industry_name(industry_name):
    """根据行业名称获取成分股"""
    # 先获取行业列表
    industry_list = get_simplified_industry_list()
    
    if industry_name not in industry_list:
        logger.warning(f"未找到行业名称: {industry_name}")
        return pd.DataFrame(columns=['stock_code', 'stock_name'])
    
    # 查找缓存
    cache_name = f"industry_stocks_{industry_name}"
    cached_data = load_cache(cache_name)
    if cached_data is not None:
        return cached_data
    
    try:
        # 使用绝对导入
        import sys
        import os
        
        # 添加项目根目录到路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        # 导入DataFetcher
        from src.data.data_fetcher import DataFetcher
        
        # 获取该行业的股票列表
        fetcher = DataFetcher()
        
        # 获取行业对应的所有股票
        # 这里我们通过股票基本信息来筛选特定行业的股票
        stock_list = fetcher.pro.stock_basic(exchange='', list_status='L', 
                                            fields='ts_code,name,industry')
        
        if stock_list is None or stock_list.empty:
            logger.warning(f"获取股票列表失败")
            return pd.DataFrame(columns=['stock_code', 'stock_name'])
        
        # 筛选该行业的股票
        industry_stocks = stock_list[stock_list['industry'] == industry_name]
        
        if industry_stocks.empty:
            logger.warning(f"未找到行业 {industry_name} 的成分股")
            return pd.DataFrame(columns=['stock_code', 'stock_name'])
        
        # 格式化为需要的DataFrame
        result = pd.DataFrame({
            'stock_code': industry_stocks['ts_code'].apply(lambda x: x.split('.')[0]),
            'stock_name': industry_stocks['name']
        })
        
        # 缓存结果
        save_cache(result, cache_name)
        
        logger.info(f"成功获取 {industry_name} 行业的 {len(result)} 只成分股")
        return result
    except Exception as e:
        logger.error(f"获取行业 {industry_name} 的成分股失败: {e}")
        return pd.DataFrame(columns=['stock_code', 'stock_name'])

# 添加一个用于测试的主函数
if __name__ == "__main__":
    print("-" * 40)
    print("正在测试行业数据模块...")
    print("-" * 40)
    
    # 测试获取行业列表
    print("1. 获取简化版行业列表:")
    industries = get_simplified_industry_list()
    print(f"获取到 {len(industries)} 个行业")
    print(industries[:5])  # 显示前5个
    
    if industries:
        # 选择第一个行业测试
        test_industry = industries[0]
        
        print("-" * 40)
        # 测试根据行业名称获取成分股
        print(f"2. 获取行业 {test_industry} 的成分股:")
        stocks = get_stocks_by_industry_name(test_industry)
        print(f"获取到 {len(stocks)} 只股票")
        if not stocks.empty:
            print(stocks.head())
    
    print("-" * 40)
    print("行业数据模块测试完成!")