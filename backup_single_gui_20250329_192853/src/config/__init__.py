# -*- coding: utf-8 -*-

"""
配置模块
"""

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 数据源配置
DATA_SOURCE_CONFIG = {
    # Tushare配置
    "tushare": {
        "token": "",
        "enable": True,
        "is_primary": True,
    },
    # AKShare配置
    "akshare": {
        "enable": True,
        "is_primary": False,
    },
    # JoinQuant配置
    "joinquant": {
        "username": "",
        "password": "",
        "enable": False,
        "is_primary": False,
    },
}

# 回测参数
BACKTEST_PARAMS = {
    "initial_capital": 1000000.0,  # 初始资金
    "commission": 0.0003,  # 交易佣金
    "slippage": 0.0002,  # 滑点
}

# 风险控制参数
RISK_CONTROL_PARAMS = {
    "max_position_per_stock": 0.05,  # 单个持仓风险
    "max_position_per_industry": 0.20,  # 行业最大持仓比例
    "max_industry_allocation": 0.30,  # 单一行业最大配置比例
    "stop_loss": 0.05,  # 默认止损比例
    "take_profit": 0.15,  # 默认止盈比例
    "use_trailing_stop": True,  # 是否使用追踪止损
    "max_drawdown": 0.10,  # 最大回撤
    "risk_free_rate": 0.03,  # 无风险利率
}

# 股票选择参数
STOCK_SELECTION_PARAMS = {
    "momentum_lookback_period": 20,  # 动量回看期
    "momentum_threshold": 0.05,  # 动量阈值
    "max_positions": 5,  # 最大持仓数量
    "update_frequency": 60,  # 更新频率（秒）
    "continuous_limit_up_days": [1, 2, 3],  # 连续涨停天数
}

# 通知配置
NOTIFICATION_CONFIG = {
    "enable_email": False,
    "email_user": "",
    "email_password": "",
    "email_host": "smtp.163.com",
    "email_port": 465,
    "email_receiver": "",
    "enable_wechat": False,
    "wechat_token": "",
}

# 实时监控参数
REALTIME_MONITOR_PARAMS = {
    "update_interval": 60,  # 更新间隔（秒）
    "monitor_stocks": [],  # 监控的股票列表
    "price_alert_threshold": 0.05,  # 价格预警阈值
    "volume_alert_threshold": 2.0,  # 成交量预警阈值
    "enable_auto_trade": False,  # 是否启用自动交易
    "auto_trade_params": {
        "max_positions": 5,  # 最大持仓数量
        "position_size": 0.2,  # 单个持仓比例
        "stop_loss": 0.05,  # 止损比例
        "take_profit": 0.15,  # 止盈比例
    }
}

# 技术指标参数
INDICATOR_PARAMS = {
    "ma_periods": [5, 10, 20, 30],  # 调整移动平均周期
    "rsi_period": 14,  # RSI周期
    "macd_params": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    },
    "kdj_params": {
        "fastk_period": 9,
        "slowk_period": 3,
        "slowd_period": 3
    },
    "boll_params": {
        "timeperiod": 20,
        "nbdevup": 2.5,  # 放宽布林带范围
        "nbdevdn": 2.5
    }
}

# 资金流向参数
MONEY_FLOW_PARAMS = {
    "lookback_period": 20,  # 回看周期
    "volume_ratio_threshold": 1.5,  # 降低成交量比率阈值
    "amount_threshold": 5000000,  # 降低成交额阈值
    "net_inflow_threshold": 2000000,  # 降低净流入阈值
    "industry_flow_threshold": 50000000,  # 降低行业资金流入阈值
    "market_cap_threshold": 5000000000,  # 降低市值阈值
    "turnover_rate_threshold": 0.03  # 降低换手率阈值
}

# 情感分析参数
SENTIMENT_PARAMS = {
    "news_lookback_days": 7,  # 新闻回看天数
    "min_news_count": 5,  # 最小新闻数量
    "sentiment_threshold": 0.6,  # 情感阈值
    "news_weight": 0.6,  # 新闻权重
    "social_media_weight": 0.4,  # 社交媒体权重
    "min_confidence": 0.7,  # 最小置信度
    "update_interval": 3600  # 更新间隔（秒）
} 