#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票数据存储类
提供股票基础数据获取、缓存和管理功能
"""

import os
import json
import logging
import datetime
import pandas as pd
import numpy as np
import random
from pathlib import Path

class StockData:
    """提供股票基础数据的存储和检索功能"""
    
    def __init__(self):
        self.logger = logging.getLogger('StockData')
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.stock_info_file = os.path.join(self.data_dir, 'stock_info.json')
        self.stock_data_dir = os.path.join(self.data_dir, 'stock_data')
        
        # 确保目录存在
        self._ensure_directories()
        
        # 加载股票信息
        self.stock_info = self._load_stock_info()
        
        # 模拟数据（仅测试使用）
        self.mock_industries = [
            "银行", "保险", "证券", "房地产", "医药生物", "计算机", "通信", 
            "电子", "传媒", "汽车", "食品饮料", "家用电器", "建筑材料", 
            "电力设备", "机械设备", "钢铁", "煤炭", "石油石化", "有色金属", 
            "化工", "纺织服装", "农林牧渔", "商业贸易", "休闲服务"
        ]
        
        # 行业股票映射
        self.industry_stocks = {}
        for industry in self.mock_industries:
            self.industry_stocks[industry] = self._get_random_stocks(5, 15)
        
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
        
        # 如果文件不存在或加载失败，返回空字典并创建模拟数据
        return self._create_mock_stock_info()
    
    def _create_mock_stock_info(self):
        """创建模拟股票数据（仅测试使用）"""
        stock_info = {}
        
        # 生成随机股票代码列表
        stock_codes = []
        
        # 沪市
        for i in range(100):
            code = f"60{random.randint(1000, 9999)}"
            stock_codes.append(code)
        
        # 深市
        for i in range(100):
            code = f"00{random.randint(1000, 9999)}"
            stock_codes.append(code)
        
        # 创业板
        for i in range(50):
            code = f"30{random.randint(1000, 9999)}"
            stock_codes.append(code)
        
        # 科创板
        for i in range(50):
            code = f"68{random.randint(1000, 9999)}"
            stock_codes.append(code)
        
        # 添加基本信息
        for code in stock_codes:
            industry = random.choice(self.mock_industries)
            stock_info[code] = {
                "name": f"测试股票{code}",
                "industry": industry,
                "list_date": f"201{random.randint(0, 9)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "area": random.choice(["上海", "北京", "深圳", "广州", "杭州", "南京"]),
                "market": "主板" if code.startswith("60") or code.startswith("00") else 
                         ("创业板" if code.startswith("30") else "科创板")
            }
        
        # 保存到文件
        try:
            with open(self.stock_info_file, 'w', encoding='utf-8') as f:
                json.dump(stock_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving mock stock info: {e}")
        
        return stock_info
    
    def _get_random_stocks(self, min_count, max_count):
        """从股票池中随机选择一定数量的股票（用于模拟行业板块）"""
        count = random.randint(min_count, max_count)
        codes = list(self.stock_info.keys())
        if len(codes) <= count:
            return codes
        return random.sample(codes, count)
            
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
            return self.stock_info[code]["industry"]
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
        return self.mock_industries
    
    def get_stock_details(self, code):
        """获取股票详细信息"""
        details = {}
        
        if code in self.stock_info:
            details.update(self.stock_info[code])
        
        # 添加股票基本面数据（模拟）
        details.update({
            "code": code,
            "pe": round(random.uniform(10, 100), 2),
            "pb": round(random.uniform(1, 10), 2),
            "market_cap": round(random.uniform(1, 1000), 2),
            "circulation_cap": round(random.uniform(0.5, 500), 2),
            "revenue": round(random.uniform(1, 100), 2),
            "net_profit": round(random.uniform(0.1, 10), 2),
            "roe": round(random.uniform(0, 30), 2)
        })
        
        # 添加股票K线数据（模拟）
        details["daily_data"] = self._get_mock_daily_data(code)
        
        return details
    
    def _get_mock_daily_data(self, code, days=60):
        """获取模拟的股票K线数据"""
        daily_data = []
        
        today = datetime.datetime.now()
        price = random.uniform(10, 100)
        
        for i in range(days):
            date = (today - datetime.timedelta(days=days-i)).strftime('%Y%m%d')
            
            # 随机生成涨跌
            price_change = price * random.uniform(-0.05, 0.05)
            open_price = price
            close_price = price + price_change
            
            # 确保价格为正
            if close_price <= 0:
                close_price = price
            
            # 涨跌停模拟
            if random.random() < 0.05:  # 5%概率涨停或跌停
                if random.random() < 0.5:  # 一半概率涨停
                    close_price = round(price * 1.1, 2)
                else:
                    close_price = round(price * 0.9, 2)
            
            # 确保最高最低价合理
            high = max(open_price, close_price) * (1 + random.uniform(0, 0.03))
            low = min(open_price, close_price) * (1 - random.uniform(0, 0.03))
            
            # 成交量
            volume = random.uniform(1000, 10000)
            
            daily_data.append({
                "date": date,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2),
                "change": round((close_price - price) / price * 100, 2)
            })
            
            # 更新价格为下一天的起始价
            price = close_price
        
        return daily_data
    
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
    logger = logging.getLogger(__name__)
    
    # 常见股票硬编码映射 - 作为回退方案
    common_stocks = {
        '601318.SH': '中国平安',
        '000651.SZ': '格力电器',
        '000333.SZ': '美的集团',
        '600519.SH': '贵州茅台',
        '000002.SZ': '万科A',
        '600036.SH': '招商银行',
        '000999.SZ': '华润三九',
        '600276.SH': '恒瑞医药',
        '000001.SZ': '平安银行',
        '600000.SH': '浦发银行',
        '601398.SH': '工商银行',
        '601939.SH': '建设银行',
    }
    
    # 先检查硬编码映射
    if ts_code in common_stocks:
        return common_stocks[ts_code]
    
    # 查询股票名称
    try:
        pro = get_pro_api()
        if pro:
            # 提取股票交易所代码
            code_parts = ts_code.split('.')
            if len(code_parts) != 2:
                logger.warning(f"股票代码格式不正确: {ts_code}")
                return ts_code
            
            # 获取股票基本信息
            stock_info = pro.stock_basic(ts_code=ts_code, fields='ts_code,name')
            if not stock_info.empty:
                return stock_info.iloc[0]['name']
            
            logger.warning(f"未找到股票信息: {ts_code}")
        else:
            logger.warning("Tushare API未初始化")
    except Exception as e:
        logger.error(f"获取股票名称时出错: {str(e)}")
    
    # 安全地尝试从StockData获取
    try:
        # 仅创建一个简化版的StockData，不需要完整的mock_industries
        simple_data = {"name": f"股票{ts_code.split('.')[0]}"}
        return simple_data["name"]
    except Exception as e:
        logger.error(f"从简化数据获取股票名称时出错: {str(e)}")
    
    # 使用模拟数据，根据股票代码生成固定的名称
    import hashlib
    hash_obj = hashlib.md5(ts_code.encode())
    hash_int = int(hash_obj.hexdigest(), 16) % 1000
    return f"股票{hash_int}"

def get_stock_names_batch(ts_codes):
    """批量获取股票名称
    
    Args:
        ts_codes: 股票代码列表
        
    Returns:
        dict: 股票代码到股票名称的映射
    """
    logger = logging.getLogger(__name__)
    result = {}
    
    if not ts_codes:
        return result
    
    # 常见股票硬编码映射
    common_stocks = {
        '601318.SH': '中国平安',
        '000651.SZ': '格力电器',
        '000333.SZ': '美的集团',
        '600519.SH': '贵州茅台',
        '000002.SZ': '万科A',
        '600036.SH': '招商银行',
        '000999.SZ': '华润三九',
        '600276.SH': '恒瑞医药',
        '000001.SZ': '平安银行',
        '600000.SH': '浦发银行',
        '601398.SH': '工商银行',
        '601939.SH': '建设银行',
    }
    
    # 先检查硬编码映射
    for ts_code in ts_codes:
        if ts_code in common_stocks:
            result[ts_code] = common_stocks[ts_code]
    
    # 如果所有的股票都已经有了，就直接返回
    if len(result) == len(ts_codes):
        logger.info(f"所有{len(ts_codes)}只股票都从本地映射获取到了名称")
        return result
    
    # 找出还未获取到名称的股票
    missing_codes = [code for code in ts_codes if code not in result]
    
    # 尝试使用Tushare批量获取
    try:
        pro = get_pro_api()
        if pro and missing_codes:
            # 使用单个获取，因为批量接口可能有问题
            for code in missing_codes:
                try:
                    name = get_stock_name(code)
                    result[code] = name
                    logger.debug(f"获取到股票名称: {code} -> {name}")
                except Exception as e:
                    logger.error(f"获取单个股票名称出错: {code}, {str(e)}")
                    result[code] = f"股票{code.split('.')[0]}"
            
            return result
    except Exception as e:
        logger.error(f"批量获取股票名称时出错: {str(e)}")
    
    # 如果还有未获取到的，使用简单映射
    for code in ts_codes:
        if code not in result:
            simple_name = f"股票{code.split('.')[0]}"
            result[code] = simple_name
            logger.debug(f"使用简单映射: {code} -> {simple_name}")
    
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