#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
整合行业数据到分析系统中
将从东方财富获取的行业分类数据整合到现有的分析系统中
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

# 确保日志目录存在
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志
log_file = os.path.join(LOG_DIR, f"integrate_industry_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
    logger.warning("未安装akshare库，将使用已下载的行业数据")

# 设置数据目录
DATA_DIR = "./data"
INDUSTRY_DATA_DIR = os.path.join(DATA_DIR, "industry_data")
os.makedirs(INDUSTRY_DATA_DIR, exist_ok=True)

def update_industry_list():
    """更新东方财富行业分类列表"""
    try:
        # 先检查是否有本地数据
        industry_csv = os.path.join(DATA_DIR, "industry_list.csv")
        
        if os.path.exists(industry_csv):
            logger.info(f"发现本地行业分类数据: {industry_csv}")
            # 读取本地数据
            industry_df = pd.read_csv(industry_csv, encoding='utf-8-sig')
            logger.info(f"成功加载本地行业分类数据，共 {len(industry_df)} 个行业")
            
            # 复制到行业数据目录
            shutil.copy(industry_csv, os.path.join(INDUSTRY_DATA_DIR, "industry_list.csv"))
            logger.info(f"已复制行业分类数据到 {INDUSTRY_DATA_DIR}")
            
            return industry_df
        
        # 如果没有本地数据且有akshare库，则获取新数据
        if HAS_AKSHARE:
            logger.info("开始获取东方财富行业分类列表...")
            industry_df = ak.stock_board_industry_name_em()
            
            if industry_df is not None and not industry_df.empty:
                logger.info(f"成功获取 {len(industry_df)} 个行业分类")
                
                # 保存行业分类数据
                industry_df.to_csv(os.path.join(INDUSTRY_DATA_DIR, "industry_list.csv"), index=False, encoding="utf-8-sig")
                logger.info(f"行业分类数据已保存到 {INDUSTRY_DATA_DIR}/industry_list.csv")
                
                return industry_df
            else:
                logger.warning("获取的行业分类列表为空")
                return None
        else:
            logger.error("未安装akshare库且没有本地行业数据")
            return None
    except Exception as e:
        logger.error(f"更新行业分类列表出错: {str(e)}")
        return None

def update_industry_stocks(industry_df, limit=10):
    """更新行业成分股数据
    
    Args:
        industry_df: 行业分类DataFrame
        limit: 限制更新的行业数量，设为None则更新所有行业
    """
    try:
        if industry_df is None or industry_df.empty:
            logger.error("行业分类数据为空，无法更新成分股")
            return False
            
        # 创建保存行业成分股的目录
        industry_stocks_dir = os.path.join(INDUSTRY_DATA_DIR, "industry_stocks")
        os.makedirs(industry_stocks_dir, exist_ok=True)
        
        # 获取行业列表
        industry_names = industry_df["板块名称"].tolist()
        if limit:
            industry_names = industry_names[:limit]
            
        success_count = 0
        
        # 首先检查本地已有数据
        local_industry_dir = os.path.join(DATA_DIR, "industry_stocks")
        local_files = []
        if os.path.exists(local_industry_dir):
            local_files = os.listdir(local_industry_dir)
            logger.info(f"发现 {len(local_files)} 个本地行业成分股数据文件")
            
            # 复制本地文件到新目录
            for file in local_files:
                if file.endswith('.csv'):
                    src_file = os.path.join(local_industry_dir, file)
                    dst_file = os.path.join(industry_stocks_dir, file)
                    shutil.copy(src_file, dst_file)
                    industry_name = file.replace('.csv', '')
                    logger.info(f"已复制 {industry_name} 行业成分股数据")
                    success_count += 1
                    
                    # 从待更新列表中移除已有的行业
                    if industry_name in industry_names:
                        industry_names.remove(industry_name)
        
        # 如果未安装akshare，则只能使用已有数据
        if not HAS_AKSHARE:
            logger.warning("未安装akshare库，跳过在线获取行业成分股")
            return success_count > 0
            
        # 更新剩余行业成分股
        for industry_name in industry_names:
            try:
                logger.info(f"开始获取 {industry_name} 行业的成分股...")
                industry_stocks = ak.stock_board_industry_cons_em(symbol=industry_name)
                
                if industry_stocks is not None and not industry_stocks.empty:
                    logger.info(f"成功获取 {len(industry_stocks)} 支 {industry_name} 行业的股票")
                    
                    # 转换股票代码格式
                    industry_stocks["ts_code"] = industry_stocks["代码"].apply(
                        lambda x: f"{x}.SH" if x.startswith(("6", "9")) else f"{x}.SZ"
                    )
                    
                    # 保存行业成分股数据
                    csv_path = os.path.join(industry_stocks_dir, f"{industry_name}.csv")
                    industry_stocks.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    logger.info(f"{industry_name} 行业成分股数据已保存到 {csv_path}")
                    success_count += 1
                else:
                    logger.warning(f"获取的 {industry_name} 行业成分股列表为空")
            except Exception as e:
                logger.error(f"获取 {industry_name} 行业成分股时出错: {str(e)}")
                continue
        
        logger.info(f"成功更新 {success_count} 个行业的成分股数据")
        return success_count > 0
    except Exception as e:
        logger.error(f"更新行业成分股数据出错: {str(e)}")
        return False

def update_mapping_file():
    """更新行业映射文件，用于将股票代码映射到行业"""
    try:
        industry_stocks_dir = os.path.join(INDUSTRY_DATA_DIR, "industry_stocks")
        if not os.path.exists(industry_stocks_dir):
            logger.error(f"行业成分股目录不存在: {industry_stocks_dir}")
            return False
            
        # 获取所有行业文件
        industry_files = [f for f in os.listdir(industry_stocks_dir) if f.endswith('.csv')]
        if not industry_files:
            logger.error("未找到行业成分股文件")
            return False
            
        # 创建映射DataFrame
        mapping_data = []
        
        for industry_file in industry_files:
            industry_name = industry_file.replace('.csv', '')
            file_path = os.path.join(industry_stocks_dir, industry_file)
            
            try:
                industry_df = pd.read_csv(file_path, encoding='utf-8-sig')
                if 'ts_code' not in industry_df.columns:
                    # 尝试转换股票代码
                    if '代码' in industry_df.columns:
                        industry_df["ts_code"] = industry_df["代码"].apply(
                            lambda x: f"{x}.SH" if str(x).startswith(("6", "9")) else f"{x}.SZ"
                        )
                    else:
                        logger.warning(f"{industry_file} 中未找到股票代码列")
                        continue
                        
                # 提取股票代码和名称
                for idx, row in industry_df.iterrows():
                    ts_code = row['ts_code']
                    name = row.get('名称', row.get('name', ''))
                    
                    mapping_data.append({
                        'ts_code': ts_code,
                        'name': name,
                        'industry': industry_name
                    })
            except Exception as e:
                logger.error(f"处理 {industry_file} 时出错: {str(e)}")
                continue
                
        # 创建映射DataFrame
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            
            # 去重（一只股票可能属于多个行业，保留第一个）
            mapping_df = mapping_df.drop_duplicates(subset=['ts_code'], keep='first')
            
            # 保存映射文件
            mapping_path = os.path.join(INDUSTRY_DATA_DIR, "stock_industry_mapping.csv")
            mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")
            logger.info(f"成功创建股票-行业映射文件，共 {len(mapping_df)} 条记录，保存至 {mapping_path}")
            
            return True
        else:
            logger.error("未能创建股票-行业映射数据")
            return False
    except Exception as e:
        logger.error(f"创建行业映射文件出错: {str(e)}")
        return False

def main():
    """主函数"""
    logger.info("=== 开始整合行业数据 ===")
    
    # 1. 更新行业列表
    industry_df = update_industry_list()
    if industry_df is None:
        logger.error("更新行业列表失败")
        return
        
    # 2. 更新行业成分股数据
    success = update_industry_stocks(industry_df, limit=None)  # 设置为None则更新所有行业
    if not success:
        logger.error("更新行业成分股数据失败")
        return
        
    # 3. 创建股票-行业映射文件
    success = update_mapping_file()
    if not success:
        logger.error("创建股票-行业映射文件失败")
        return
        
    logger.info("=== 行业数据整合完成 ===")

if __name__ == "__main__":
    main() 