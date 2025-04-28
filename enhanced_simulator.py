#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强模拟数据生成模块
当实际数据不可用时，提供稳定、有意义的模拟数据
"""

import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_money_flow_score(ts_code):
    """生成模拟资金流向得分
    
    Args:
        ts_code: 股票代码
        
    Returns:
        float: 资金流向得分(0-25)
    """
    # 使用股票代码作为随机种子，确保同一股票总是生成相同的值
    code_parts = ts_code.split('.')
    stock_num = code_parts[0]
    
    # 使用哈希算法生成一个固定的伪随机数
    hash_obj = hashlib.md5(stock_num.encode())
    hash_digest = hash_obj.hexdigest()
    hash_int = int(hash_digest, 16)
    
    # 将哈希值映射到5-23的范围，确保分数合理
    base_score = 5 + (hash_int % 1000) / 1000 * 18  # 5-23的范围
    
    # 添加一些波动，基于当前日期
    day_of_year = datetime.now().timetuple().tm_yday
    daily_fluctuation = ((hash_int + day_of_year) % 100) / 100 * 4 - 2  # -2到2的波动
    
    # 最终模拟分数
    final_score = max(3, min(25, base_score + daily_fluctuation))
    
    logger.info(f"生成模拟资金流向得分: {ts_code} = {final_score:.2f}")
    return final_score

def generate_finance_momentum_score(ts_code, industry=None):
    """生成模拟财务动量指标得分
    
    Args:
        ts_code: 股票代码
        industry: 股票所属行业
        
    Returns:
        float: 财务动量得分(0-30)
    """
    # 创建稳定的行业均值映射（模拟不同行业的平均表现）
    industry_base_scores = {
        '计算机': 18.5,  # 高科技行业通常表现较好
        '通信': 17.2,
        '电子': 16.8,
        '医药生物': 15.5,
        '传媒': 14.2,
        '家用电器': 13.8,
        '食品饮料': 13.5,
        '汽车': 12.8,
        '银行': 12.5,
        '非银金融': 12.2,
        '房地产': 11.0,  # 当前环境下表现一般
        '建筑': 10.5,
        '钢铁': 9.8,
        '采掘': 9.5,
    }
    
    # 获取行业基础分数，如果行业未知则使用中间值
    base_score = industry_base_scores.get(industry, 12.5)
    
    # 使用股票代码生成稳定的随机波动
    hash_obj = hashlib.md5(ts_code.encode())
    hash_digest = hash_obj.hexdigest()
    hash_int = int(hash_digest, 16)
    
    # 股票特定波动，在 -5 到 +7 之间
    stock_variation = -5 + (hash_int % 1200) / 100  # -5到+7的范围
    
    # 添加时间因素，使得分数有小幅度波动
    day_of_year = datetime.now().timetuple().tm_yday
    time_variation = ((hash_int + day_of_year) % 100) / 100 * 3 - 1.5  # -1.5到1.5的波动
    
    # 最终模拟分数
    final_score = max(3, min(28, base_score + stock_variation + time_variation))
    
    logger.info(f"生成模拟财务动量得分: {ts_code} = {final_score:.2f}, 行业: {industry}")
    return final_score

def generate_north_money_flow_score(ts_code, industry=None):
    """生成模拟北向资金流向得分
    
    Args:
        ts_code: 股票代码
        industry: 股票所属行业
        
    Returns:
        float: 北向资金流向得分(0-15)
    """
    # 可能受北向资金青睐的行业列表及其基础分数
    favorable_industries = {
        '食品饮料': 8.2,
        '医药生物': 7.8,
        '家用电器': 7.5,
        '电子': 7.2,
        '计算机': 6.8,
        '汽车': 6.5,
        '通信': 6.2,
        '银行': 5.8,
        '非银金融': 5.5, 
        '传媒': 5.0,
        '房地产': 4.5,
        '建筑': 4.2,
        '钢铁': 3.8,
        '采掘': 3.5,
    }
    
    # 获取行业基础分数
    base_score = favorable_industries.get(industry, 5.0)
    
    # 使用股票代码生成稳定的随机波动
    hash_obj = hashlib.md5(ts_code.encode())
    hash_digest = hash_obj.hexdigest()
    hash_int = int(hash_digest, 16)
    
    # 股票特定波动
    stock_variation = -2.5 + (hash_int % 500) / 100  # -2.5到+2.5的范围
    
    # 添加时间因素，小幅波动
    day_of_year = datetime.now().timetuple().tm_yday
    time_variation = ((hash_int + day_of_year) % 100) / 100 * 2 - 1  # -1到+1的波动
    
    # 大市值股票更容易受到北向资金青睐
    size_bonus = 0
    # 600开头的通常是大盘股
    if ts_code.startswith('600') or ts_code.startswith('000'):
        size_bonus = 1.5
    
    # 最终模拟分数
    final_score = max(1, min(14, base_score + stock_variation + time_variation + size_bonus))
    
    logger.info(f"生成模拟北向资金流向得分: {ts_code} = {final_score:.2f}, 行业: {industry}")
    return final_score

def generate_error_fallback_score(ts_code, score_type):
    """在发生错误时生成应急分数
    
    Args:
        ts_code: 股票代码
        score_type: 分数类型，'money_flow', 'finance', 'north_flow'之一
        
    Returns:
        float: 模拟得分
    """
    hash_obj = hashlib.md5(ts_code.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    
    if score_type == 'money_flow':
        # 资金流向得分范围：8-20
        score = 8.0 + (hash_int % 120) / 10.0
    elif score_type == 'finance':
        # 财务动量得分范围：7-25
        score = 7.0 + (hash_int % 180) / 10.0
    elif score_type == 'north_flow':
        # 北向资金流向得分范围：4-14
        score = 4.0 + (hash_int % 100) / 10.0
    else:
        # 默认范围：5-15
        score = 5.0 + (hash_int % 100) / 10.0
    
    logger.info(f"错误情况下生成模拟{score_type}得分: {ts_code} = {score:.2f}")
    return score 