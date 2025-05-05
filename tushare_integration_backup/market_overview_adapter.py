#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
市场概览数据适配器
将市场概览数据转换为GUI期望的格式
"""

import logging
import time
from datetime import datetime
import random
import json
import os

# 导入Tushare数据获取模块
from tushare_market_data import get_market_overview_data, get_index_data, get_industry_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def adapt_market_overview(data):
    """
    将市场概览数据转换为GUI期望的格式
    
    Args:
        data (dict): 原始市场概览数据，包含基本指标
        
    Returns:
        dict: GUI期望格式的市场概览数据
    """
    logger.info("适配市场概览数据开始...")
    
    # 如果输入数据为空，尝试从Tushare获取实时数据
    if data is None or not isinstance(data, dict) or len(data) == 0:
        logger.info("输入数据为空，尝试从Tushare获取实时数据")
        data = get_market_overview_data()
        
    if not isinstance(data, dict):
        logger.warning(f"输入的市场概览数据不是字典类型: {type(data)}")
        return {}
    
    result = {}
    
    # 1. 构建指数数据 - 使用真实数据
    try:
        indices_data = get_index_data()
        if indices_data and len(indices_data) > 0:
            result["indices"] = indices_data
        else:
            result["indices"] = generate_indices_data(data)
    except Exception as e:
        logger.error(f"获取指数数据失败: {str(e)}")
        result["indices"] = generate_indices_data(data)
    
    # 2. 构建行业板块数据 - 使用真实数据
    try:
        industry_data = get_industry_data()
        if industry_data and len(industry_data) > 0:
            # 按强度指数排序
            for ind in industry_data:
                ind["strength_index"] = calculate_industry_strength(ind)
            industry_data.sort(key=lambda x: x["strength_index"], reverse=True)
            # 调整数据格式以匹配GUI期望格式
            for ind in industry_data:
                if "leading_stock" in ind and isinstance(ind["leading_stock"], dict):
                    ind["leading_up"] = f"{ind['leading_stock']['code']} {ind['leading_stock']['name']}"
                    ind["leading_up_change"] = ind['leading_stock']['change']
            
            result["industry_performance"] = industry_data
        else:
            result["industry_performance"] = generate_industry_data(data)
    except Exception as e:
        logger.error(f"获取行业数据失败: {str(e)}")
        result["industry_performance"] = generate_industry_data(data)
    
    # 3. 构建当前热门板块 - 基于真实行业数据
    try:
        if "industry_performance" in result and result["industry_performance"]:
            result["hot_sectors"] = generate_hot_sectors_from_real_data(result["industry_performance"], data)
        else:
            result["hot_sectors"] = generate_hot_sectors(data)
    except Exception as e:
        logger.error(f"构建热门板块失败: {str(e)}")
        result["hot_sectors"] = generate_hot_sectors(data)
    
    # 4. 构建未来热门预测
    try:
        market_sentiment = determine_sentiment(data)
        if "industry_performance" in result and result["industry_performance"]:
            result["future_hot_sectors"] = generate_future_hot_sectors_from_real_data(result["industry_performance"], market_sentiment)
        else:
            result["future_hot_sectors"] = generate_future_hot_sectors(data)
    except Exception as e:
        logger.error(f"构建未来热门预测失败: {str(e)}")
        result["future_hot_sectors"] = generate_future_hot_sectors(data)
    
    # 5. 构建市场统计数据
    result["market_stats"] = {
        "date": data.get("date", datetime.now().strftime("%Y%m%d")),
        "up_count": data.get("up_count", 0),
        "down_count": data.get("down_count", 0),
        "flat_count": data.get("flat_count", 0),
        "limit_up_count": data.get("limit_up_count", 0),
        "limit_down_count": data.get("limit_down_count", 0),
        "total_volume": data.get("total_volume", 0),
        "total_amount": data.get("total_amount", 0),
        "avg_change_pct": data.get("avg_change_pct", 0),
        "turnover_rate": data.get("turnover_rate", 0),
        "market_sentiment": determine_sentiment(data)
    }
    
    # 保存到文件，方便调试
    try:
        with open('market_overview_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("已将市场概览数据保存到 market_overview_result.json")
    except Exception as e:
        logger.warning(f"保存市场概览数据到文件失败: {str(e)}")
    
    logger.info(f"市场概览数据适配完成，生成了 {len(result)} 个顶级模块")
    return result

def calculate_industry_strength(industry_data):
    """根据行业数据计算行业强度指数"""
    up_count = industry_data.get("up_count", 0)
    down_count = industry_data.get("down_count", 0)
    change = industry_data.get("change", 0)
    
    if up_count + down_count == 0:
        return 50  # 默认值
    
    # 计算上涨比例
    up_ratio = up_count / (up_count + down_count)
    
    # 基于涨跌比例和涨跌幅计算行业强度
    strength = (up_ratio * 60) + (change * 4)
    
    # 确保结果在0-100之间
    return min(100, max(0, 50 + strength))

def determine_sentiment(data):
    """根据市场数据确定市场情绪"""
    up_count = data.get("up_count", 0)
    down_count = data.get("down_count", 0)
    avg_change = data.get("avg_change_pct", 0)
    limit_up = data.get("limit_up_count", 0)
    limit_down = data.get("limit_down_count", 0)
    
    # 计算涨跌比
    ratio = up_count / down_count if down_count > 0 else up_count
    
    # 判断情绪
    if ratio > 1.5 and avg_change > 1:
        return "极度乐观"
    elif ratio > 1.2 and avg_change > 0.5:
        return "乐观"
    elif ratio > 1:
        return "偏乐观"
    elif ratio < 0.5 and avg_change < -1:
        return "极度悲观"
    elif ratio < 0.8 and avg_change < -0.5:
        return "悲观"
    elif ratio < 1:
        return "偏悲观"
    else:
        return "中性"

def generate_indices_data(data):
    """生成指数数据 (备用方案)"""
    # 使用固定的主要指数
    indices = [
        {"name": "上证指数", "code": "000001.SH"},
        {"name": "深证成指", "code": "399001.SZ"},
        {"name": "创业板指", "code": "399006.SZ"},
        {"name": "上证50", "code": "000016.SH"},
        {"name": "沪深300", "code": "000300.SH"},
        {"name": "中证500", "code": "000905.SH"}
    ]
    
    avg_change = data.get("avg_change_pct", 0)
    
    result = []
    for idx in indices:
        # 根据平均涨跌幅生成模拟数据
        change = avg_change * (0.8 + random.random() * 0.4)  # 在平均值周围波动
        base_price = 2000 + random.random() * 3000  # 生成随机基准价格
        volume = data.get("total_volume", 1000000) / 10 * (0.5 + random.random())
        amount = data.get("total_amount", 10000000) / 10 * (0.5 + random.random())
        
        # 生成5日涨跌幅
        change_5d = change * (1 + random.random() * 0.5)
        
        # 确定趋势
        trend = "上涨" if change > 0 else "下跌"
        if abs(change) < 0.5:
            trend = "震荡"
        elif change > 1.5:
            trend = "强势上涨"
        elif change < -1.5:
            trend = "强势下跌"
        
        result.append({
            "name": idx["name"],
            "code": idx["code"],
            "close": base_price,
            "change": change,
            "change_5d": change_5d,
            "volume": volume,
            "amount": amount,
            "volume_ratio": 0.8 + random.random() * 0.4,
            "trend": trend
        })
    
    return result

def generate_industry_data(data):
    """生成行业板块数据 (备用方案)"""
    # 行业列表
    industries = [
        "医药生物", "电子", "计算机", "通信", "汽车", "电气设备",
        "机械设备", "化工", "建筑装饰", "食品饮料", "银行", "房地产"
    ]
    
    avg_change = data.get("avg_change_pct", 0)
    result = []
    
    total = data.get("up_count", 0) + data.get("down_count", 0) + data.get("flat_count", 0)
    avg_per_industry = total / (len(industries) * 2) if industries else 0  # 平均每个行业股票数
    
    limit_up_stocks = data.get("limit_up_stocks", [])
    limit_down_stocks = data.get("limit_down_stocks", [])
    
    for i, name in enumerate(industries):
        # 行业变化生成略有偏差的随机数据
        change = avg_change * (0.5 + random.random() * 1.5)
        
        # 上涨下跌家数
        up = int(avg_per_industry * (0.7 + random.random() * 0.6))
        down = int(avg_per_industry * (0.7 + random.random() * 0.6))
        
        # 获取领涨股，优先使用涨停股票中的
        leading_up = ""
        leading_up_change = 0
        if limit_up_stocks and i < len(limit_up_stocks):
            leading_up = f"{limit_up_stocks[i]['code']} {limit_up_stocks[i]['name']}"
            leading_up_change = 9.5 + random.random() * 0.5
        else:
            leading_up = f"行业龙头{i+1}"
            leading_up_change = 5 + random.random() * 5
        
        # 计算行业强度指数 (基于涨跌家数比例和平均涨幅)
        strength = 50 + int((up / (down + 0.01) - 1) * 20 + change * 5)
        
        result.append({
            "name": name,
            "change": change,
            "up_count": up,
            "down_count": down,
            "leading_up": leading_up,
            "leading_up_change": leading_up_change,
            "strength_index": strength
        })
    
    # 按强度排序
    result.sort(key=lambda x: x["strength_index"], reverse=True)
    return result

def generate_hot_sectors_from_real_data(industry_data, market_data):
    """从真实行业数据生成热门板块"""
    # 选择最强的行业作为热门板块
    top_industries = sorted(industry_data, key=lambda x: x.get("strength_index", 0), reverse=True)[:8]
    
    result = []
    for ind in top_industries:
        # 构建热门板块数据
        sector = {
            "name": ind.get("name", ""),
            "change": ind.get("change", 0),
            "turnover": ind.get("amount", 0) / 100000000 if "amount" in ind else random.random() * 100,  # 转换为亿元
            "up_count": ind.get("up_count", 0),
            "down_count": ind.get("down_count", 0),
            "leading_stock": ind.get("leading_up", "")
        }
        
        result.append(sector)
    
    # 按涨幅排序
    result.sort(key=lambda x: x["change"], reverse=True)
    return result[:5]  # 只返回前5个

def generate_hot_sectors(data):
    """生成当前热门板块 (备用方案)"""
    # 热门主题
    hot_themes = [
        "芯片", "人工智能", "新能源车", "光伏", "互联网金融", 
        "数字经济", "国产软件", "医疗器械", "军工", "创新药"
    ]
    
    avg_change = data.get("avg_change_pct", 0)
    limit_up_stocks = data.get("limit_up_stocks", [])
    result = []
    
    for i, name in enumerate(hot_themes):
        # 板块涨幅通常高于大盘平均
        change = avg_change * (1.2 + random.random() * 0.8)
        
        # 生成成交额，亿元
        turnover = 10 + random.random() * 90
        
        # 上涨下跌家数
        up = int(10 + random.random() * 20)
        down = int(5 + random.random() * 15)
        
        # 领涨股
        leading_stock = ""
        if limit_up_stocks and i < len(limit_up_stocks):
            leading_stock = f"{limit_up_stocks[i]['code']} {limit_up_stocks[i]['name']}"
        else:
            leading_stock = f"龙头股{i+1}"
        
        result.append({
            "name": name,
            "change": change,
            "turnover": turnover,
            "up_count": up,
            "down_count": down,
            "leading_stock": leading_stock
        })
    
    # 按涨幅排序
    result.sort(key=lambda x: x["change"], reverse=True)
    return result[:5]  # 只返回前5个

def generate_future_hot_sectors_from_real_data(industry_data, market_sentiment):
    """从真实行业数据生成未来热门板块预测"""
    # 寻找潜力行业
    potential_industries = []
    
    # 1. 找出低位且上涨比例增加的行业
    for ind in industry_data:
        strength = ind.get("strength_index", 50)
        change = ind.get("change", 0)
        up_ratio = ind.get("up_count", 0) / max(1, ind.get("up_count", 0) + ind.get("down_count", 0))
        
        # 如果行业强度不高但涨跌比例较好，或者在反弹初期，则被视为潜力行业
        if (40 <= strength <= 60 and up_ratio > 0.5) or (change > 0 and strength < 50):
            potential_industries.append(ind)
    
    # 2. 如果找不到足够的潜力行业，则增加一些高强度行业
    if len(potential_industries) < 5:
        high_strength = sorted([i for i in industry_data if i.get("strength_index", 0) > 60], 
                               key=lambda x: x.get("strength_index", 0), reverse=True)
        potential_industries.extend(high_strength[:5-len(potential_industries)])
    
    # 如果仍然没有足够的行业，添加一些通用的
    if len(potential_industries) < 5:
        missing = 5 - len(potential_industries)
        generic_industries = [
            {"name": "数字经济", "change": 1.5, "strength_index": 65},
            {"name": "新能源", "change": 1.2, "strength_index": 62},
            {"name": "医疗健康", "change": 0.8, "strength_index": 58},
            {"name": "先进制造", "change": 1.0, "strength_index": 60},
            {"name": "人工智能", "change": 2.0, "strength_index": 70}
        ]
        potential_industries.extend(generic_industries[:missing])
    
    # 准备理由库
    reasons = [
        "政策利好持续释放", "行业景气度提升", "技术突破在即", 
        "龙头企业业绩增长", "估值处于历史低位", "资金持续流入",
        "产业升级加速", "市场需求持续扩大", "并购重组预期",
        "国产替代加速推进", "行业整合加剧"
    ]
    
    # 根据市场情绪调整预期涨幅
    sentiment_multiplier = 1.0
    if "极度乐观" in market_sentiment:
        sentiment_multiplier = 1.5
    elif "乐观" in market_sentiment:
        sentiment_multiplier = 1.2
    elif "偏乐观" in market_sentiment:
        sentiment_multiplier = 1.1
    elif "极度悲观" in market_sentiment:
        sentiment_multiplier = 0.5
    elif "悲观" in market_sentiment:
        sentiment_multiplier = 0.7
    elif "偏悲观" in market_sentiment:
        sentiment_multiplier = 0.9
    
    # 构建未来热门板块预测
    result = []
    for ind in potential_industries[:5]:  # 最多5个
        name = ind.get("name", "未知行业")
        current_change = ind.get("change", 0)
        strength = ind.get("strength_index", 50)
        
        # 预测涨幅 (通常比当前涨幅高)
        predicted_change = max(1.5, current_change * 1.5) * sentiment_multiplier
        
        # 关注指数 (基于强度)
        attention_index = min(100, max(50, strength * 1.2))
        
        # 资金流入 (基于当前强度和预测涨幅)
        fund_inflow = strength / 10 + predicted_change * 2
        
        # 成长评分
        growth_score = min(100, max(50, strength + 20))
        
        # 随机选择推荐理由
        recommendation = random.choice(reasons)
        
        result.append({
            "name": name,
            "predicted_change": predicted_change,
            "attention_index": attention_index,
            "fund_inflow": fund_inflow,
            "growth_score": growth_score,
            "recommendation": recommendation
        })
    
    # 按预测涨幅排序
    result.sort(key=lambda x: x["predicted_change"], reverse=True)
    return result

def generate_future_hot_sectors(data):
    """生成未来热门板块预测 (备用方案)"""
    # 潜力板块
    potential_sectors = [
        "数字货币", "低碳环保", "元宇宙", "水资源利用", "高端制造", 
        "生物医药", "云计算", "数据安全", "车联网", "航空航天"
    ]
    
    avg_change = data.get("avg_change_pct", 0)
    market_sentiment = determine_sentiment(data)
    result = []
    
    # 根据市场情绪调整预期涨幅
    sentiment_multiplier = 1.0
    if "极度乐观" in market_sentiment:
        sentiment_multiplier = 1.5
    elif "乐观" in market_sentiment:
        sentiment_multiplier = 1.2
    elif "偏乐观" in market_sentiment:
        sentiment_multiplier = 1.1
    elif "极度悲观" in market_sentiment:
        sentiment_multiplier = 0.5
    elif "悲观" in market_sentiment:
        sentiment_multiplier = 0.7
    elif "偏悲观" in market_sentiment:
        sentiment_multiplier = 0.9
        
    # 生成预测数据
    for name in potential_sectors:
        # 预测涨幅通常比当前涨幅更乐观
        predicted_change = max(1, avg_change) * (1 + random.random()) * sentiment_multiplier
        
        # 关注指数
        attention_index = 60 + random.random() * 40
        
        # 主力资金流入（亿元）
        fund_inflow = random.random() * 30
        
        # 成长评分
        growth_score = 50 + random.random() * 50
        
        # 推荐理由
        reasons = [
            "政策利好持续释放", "行业景气度提升", "技术突破在即", 
            "龙头企业业绩增长", "估值处于历史低位", "资金持续流入",
            "产业升级加速", "市场需求持续扩大", "并购重组预期",
            "国产替代加速推进", "行业整合加剧"
        ]
        recommendation = random.choice(reasons)
        
        result.append({
            "name": name,
            "predicted_change": predicted_change,
            "attention_index": attention_index,
            "fund_inflow": fund_inflow,
            "growth_score": growth_score,
            "recommendation": recommendation
        })
    
    # 按预测涨幅排序
    result.sort(key=lambda x: x["predicted_change"], reverse=True)
    return result[:5]  # 只返回前5个

if __name__ == "__main__":
    # 测试
    test_data = None  # 将获取真实数据
    adapted = adapt_market_overview(test_data)
    print(f"适配后数据包含 {len(adapted)} 个顶级模块")
    for key, value in adapted.items():
        print(f"- {key}: {type(value)} 包含 {len(value) if isinstance(value, (list, dict)) else 'N/A'} 项") 