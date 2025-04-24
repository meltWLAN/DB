#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件
系统的全局配置，包括路径、参数等
"""

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    # JoinQuant配置
    "joinquant": {
        "username": "",  # JoinQuant用户名
        "password": "",  # JoinQuant密码
        "enabled": True,
    },
    # Tushare配置
    "tushare": {
        "token": "b82eb228e75ef3b76f18633ef5b0d3b9ef70904d883be5bbee8a2321", # Tushare token
        "enabled": True,
    },
    # AKShare配置
    "akshare": {
        "enabled": True,
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
    "max_position_risk": 0.02,  # 单个持仓风险
    "max_portfolio_risk": 0.05,  # 组合最大风险
    "max_industry_allocation": 0.30,  # 单一行业最大配置比例
    "default_stop_loss_pct": 0.05,  # 默认止损比例
    "default_take_profit_pct": 0.15,  # 默认止盈比例
    "use_trailing_stop": True,  # 是否使用追踪止损
}

# 股票选择参数
STOCK_SELECTION_PARAMS = {
    "momentum_lookback_period": 20,  # 动量回看期
    "momentum_threshold": 0.05,  # 动量阈值
    "max_positions": 5,  # 最大持仓数量
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