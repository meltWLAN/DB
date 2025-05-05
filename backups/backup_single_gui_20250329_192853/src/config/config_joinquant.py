#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JoinQuant账号配置脚本
"""

import os
import sys
import json
import getpass

# 确保目录存在
os.makedirs("src/config", exist_ok=True)

def update_config_file():
    """更新配置文件"""
    config_file = "src/config/__init__.py"
    
    # 检查文件是否存在
    if not os.path.exists(config_file):
        # 创建基本结构
        with open(config_file, "w", encoding="utf-8") as f:
            f.write("""# -*- coding: utf-8 -*-

\"\"\"
配置模块初始化文件
\"\"\"

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
    # JoinQuant配置
    "joinquant": {
        "username": "",
        "password": "",
        "enabled": True,
    },
    # Tushare配置
    "tushare": {
        "token": "",
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
""")
    
    # 读取配置文件
    with open(config_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 获取用户输入
    print("=" * 80)
    print(" JoinQuant账号配置 ".center(80, "="))
    print("=" * 80)
    print("请输入您的JoinQuant账号信息：")
    username = input("用户名: ")
    password = getpass.getpass("密码: ")
    
    # 更新内容
    # 替换username字段
    content = content.replace(
        '"username": "",', 
        f'"username": "{username}",'
    )
    
    # 替换password字段
    content = content.replace(
        '"password": "",', 
        f'"password": "{password}",'
    )
    
    # 写回文件
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print("\n✅ JoinQuant账号配置已更新")
    print(f"配置文件路径: {os.path.abspath(config_file)}")
    print("=" * 80)
    
    # 询问是否测试连接
    test_connection = input("\n是否测试JoinQuant连接? (y/n) [y]: ").lower() or "y"
    if test_connection == "y":
        print("\n运行测试脚本...")
        os.system(f"{sys.executable} test_joinquant.py")

if __name__ == "__main__":
    update_config_file() 