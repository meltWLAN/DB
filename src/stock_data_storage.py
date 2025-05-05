#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票数据存储类
提供股票基础数据获取、缓存和管理功能
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path

class StockData:
    """提供股票基础数据的存储和检索功能"""
    
    def __init__(self):
        """初始化"""
        self.logger = logging.getLogger('StockData')
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.stock_info_file = os.path.join(self.data_dir, 'stock_info.json')
        self.stock_data_dir = os.path.join(self.data_dir, 'stock_data')
        
        # 确保目录存在
        self._ensure_directories()
        
        # 加载股票信息
        self.stock_info = self._load_stock_info()
        
        # 行业股票映射
        self.industry_stocks = {}
            
    def _ensure_directories(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        if not os.path.exists(self.stock_data_dir):
            os.makedirs(self.stock_data_dir)
            
    def _load_stock_info(self):
        """加载股票基本信息"""
        if os.path.exists(self.stock_info_file):
            try:
                with open(self.stock_info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading stock info: {e}")
        
        # 如果文件不存在或加载失败，返回空字典
        self.logger.warning("股票基础信息文件不存在或加载失败，返回空字典")
        return {}
    
    def get_stock_list(self):
        """获取所有股票代码列表"""
        return list(self.stock_info.keys())
    
    def get_stock_name(self, code):
        """获取股票名称"""
        if code in self.stock_info:
            return self.stock_info[code]["name"]
        return f"Unknown-{code}"
    
    def get_stock_industry(self, code):
        """获取股票所属行业"""
        if code in self.stock_info:
            return self.stock_info[code].get("industry", "Unknown")
        return "Unknown"
    
    def get_stock_market(self, code):
        """获取股票市场"""
        if code in self.stock_info:
            return self.stock_info[code].get("market", "Unknown")
        return "Unknown"
    
    def get_stocks_by_industry(self, industry):
        """获取指定行业的所有股票"""
        if industry in self.industry_stocks:
            return self.industry_stocks[industry]
        
        # 如果没有预先构建的行业分类，则遍历查找
        codes = []
        for code, info in self.stock_info.items():
            if info.get("industry") == industry:
                codes.append(code)
        return codes
    
    def get_industries(self):
        """获取所有行业列表"""
        # 从股票信息中提取所有行业
        industries = set()
        for info in self.stock_info.values():
            if "industry" in info and info["industry"]:
                industries.add(info["industry"])
        return list(industries)
    
    def get_stock_details(self, code):
        """获取股票详细信息"""
        if code not in self.stock_info:
            self.logger.warning(f"未找到股票 {code} 的基本信息")
            return {"code": code, "name": f"Unknown-{code}"}
            
        details = self.stock_info[code].copy()
        details["code"] = code
        
        # 获取真实K线数据
        daily_data = self.load_daily_data(code)
        if daily_data is not None and not daily_data.empty:
            details["daily_data"] = daily_data.to_dict('records')
        else:
            details["daily_data"] = []
            
        return details
    
    def update_stock_info(self, code, data):
        """更新股票基本信息"""
        if code not in self.stock_info:
            self.stock_info[code] = {}
        
        self.stock_info[code].update(data)
        
        # 保存到文件
        try:
            with open(self.stock_info_file, 'w', encoding='utf-8') as f:
                json.dump(self.stock_info, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error updating stock info: {e}")
            return False
    
    def save_daily_data(self, code, data_df):
        """保存股票的日线数据"""
        file_path = os.path.join(self.stock_data_dir, f"{code}_daily.csv")
        
        try:
            data_df.to_csv(file_path, index=False)
            return True
        except Exception as e:
            self.logger.error(f"Error saving daily data for {code}: {e}")
            return False
    
    def load_daily_data(self, code):
        """加载股票的日线数据"""
        file_path = os.path.join(self.stock_data_dir, f"{code}_daily.csv")
        
        if os.path.exists(file_path):
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                self.logger.error(f"Error loading daily data for {code}: {e}")
        
        return None

# tushare token设置 - 使用更新的token
DEFAULT_TUSHARE_TOKEN = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"

# 全局pro_api对象
_pro_api = None

def get_pro_api():
    """获取或初始化tushare的pro_api对象
    
    Returns:
        tushare.pro_api: 初始化后的pro_api对象
    """
    global _pro_api
    if _pro_api is None:
        try:
            import tushare as ts
            # 设置Tushare Token
            ts.set_token(DEFAULT_TUSHARE_TOKEN)
            _pro_api = ts.pro_api()
            logging.info(f"成功初始化Tushare API (Token前5位: {DEFAULT_TUSHARE_TOKEN[:5]}...)")
        except Exception as e:
            logging.error(f"初始化Tushare API失败: {str(e)}")
            _pro_api = None
    return _pro_api

def get_stock_name(ts_code):
    """根据股票代码获取股票名称（全局函数版本）
    
    Args:
        ts_code: 股票代码（格式如：000001.SZ, 600001.SH）
        
    Returns:
        str: 股票名称
    """
    # 先从全局对象中获取
    stock_data = StockData()
    name = stock_data.get_stock_name(ts_code)
    if name and not name.startswith("Unknown"):
        return name
    
    # 如果本地没有，尝试从tushare获取
    try:
        pro = get_pro_api()
        if pro:
            df = pro.stock_basic(ts_code=ts_code, fields='ts_code,name')
            if df is not None and not df.empty:
                # 更新本地缓存
                stock_data.update_stock_info(ts_code, {"name": df.iloc[0]['name']})
                return df.iloc[0]['name']
    except Exception as e:
        logging.error(f"从tushare获取股票名称失败: {str(e)}")
    
    return f"Unknown-{ts_code}"

def get_stock_names_batch(ts_codes):
    """批量获取股票名称
    
    Args:
        ts_codes: 股票代码列表
        
    Returns:
        dict: 股票代码到名称的映射
    """
    result = {}
    stock_data = StockData()
    
    # 先从本地获取
    for code in ts_codes:
        name = stock_data.get_stock_name(code)
        if name and not name.startswith("Unknown"):
            result[code] = name
    
    # 对于本地没有的，批量从tushare获取
    unknown_codes = [code for code in ts_codes if code not in result]
    if unknown_codes:
        try:
            pro = get_pro_api()
            if pro:
                # 由于tushare限制，可能需要分批处理
                batch_size = 100
                for i in range(0, len(unknown_codes), batch_size):
                    batch_codes = unknown_codes[i:i+batch_size]
                    codes_str = ",".join(batch_codes)
                    df = pro.stock_basic(ts_code=codes_str, fields='ts_code,name')
                    if df is not None and not df.empty:
                        for _, row in df.iterrows():
                            code = row['ts_code']
                            name = row['name']
                            result[code] = name
                            # 更新本地缓存
                            stock_data.update_stock_info(code, {"name": name})
        except Exception as e:
            logging.error(f"批量从tushare获取股票名称失败: {str(e)}")
    
    # 对于仍然没有获取到的，使用Unknown
    for code in ts_codes:
        if code not in result:
            result[code] = f"Unknown-{code}"
    
    return result

if __name__ == "__main__":
    # 简单测试
    stock_data = StockData()
    stock_list = stock_data.get_stock_list()
    print(f"Total stocks: {len(stock_list)}")
    
    if stock_list:
        code = stock_list[0]
        name = stock_data.get_stock_name(code)
        industry = stock_data.get_stock_industry(code)
        print(f"Sample stock: {code} - {name} ({industry})")
        
        details = stock_data.get_stock_details(code)
        print(f"PE: {details.get('pe')}, PB: {details.get('pb')}")
        print(f"Daily data samples: {len(details.get('daily_data', []))}") 