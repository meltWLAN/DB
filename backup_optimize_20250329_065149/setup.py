#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
安装和配置脚本
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import configparser
import getpass
from setuptools import setup, find_packages

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_environment():
    """设置环境与安装依赖"""
    print("=" * 80)
    print(" 正在设置交易系统环境 ".center(80, "="))
    print("=" * 80)
    
    # 检查并安装依赖
    requirements_file = os.path.join(ROOT_DIR, "requirements.txt")
    if os.path.exists(requirements_file):
        print("正在安装依赖包，请稍候...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], check=True)
            print("✅ 依赖包安装完成")
        except subprocess.CalledProcessError:
            print("❌ 依赖包安装失败，请检查网络或手动安装")
            return False
    else:
        print("❌ 未找到requirements.txt文件")
        return False
    
    # 创建必要的目录
    for directory in ["data", "data/cache", "results", "logs"]:
        dir_path = os.path.join(ROOT_DIR, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ 创建目录: {directory}")
    
    return True

def setup_data_sources():
    """设置数据源配置"""
    print("\n" + "=" * 80)
    print(" 数据源配置 ".center(80, "="))
    print("=" * 80)
    
    config_file = os.path.join(ROOT_DIR, "data_source_config.ini")
    config = configparser.ConfigParser()
    
    # 如果配置文件存在，读取现有配置
    if os.path.exists(config_file):
        config.read(config_file)
    
    # JoinQuant配置
    if "joinquant" not in config:
        config["joinquant"] = {}
    
    use_joinquant = input("是否使用JoinQuant数据源? (y/n) [y]: ").lower() or "y"
    if use_joinquant == "y":
        config["joinquant"]["enabled"] = "true"
        
        # 获取账号信息
        current_username = config["joinquant"].get("username", "")
        if current_username:
            change_creds = input(f"当前JoinQuant用户名为: {current_username}，是否修改? (y/n) [n]: ").lower() or "n"
            if change_creds == "y":
                username = input("请输入JoinQuant用户名: ")
                password = getpass.getpass("请输入JoinQuant密码: ")
                config["joinquant"]["username"] = username
                config["joinquant"]["password"] = password
        else:
            username = input("请输入JoinQuant用户名: ")
            password = getpass.getpass("请输入JoinQuant密码: ")
            config["joinquant"]["username"] = username
            config["joinquant"]["password"] = password
    else:
        config["joinquant"]["enabled"] = "false"
    
    # Tushare配置
    if "tushare" not in config:
        config["tushare"] = {}
    
    use_tushare = input("是否使用Tushare数据源? (y/n) [y]: ").lower() or "y"
    if use_tushare == "y":
        config["tushare"]["enabled"] = "true"
        
        # 获取token信息
        current_token = config["tushare"].get("token", "")
        if current_token:
            change_token = input(f"当前Tushare token为: {current_token}，是否修改? (y/n) [n]: ").lower() or "n"
            if change_token == "y":
                token = input("请输入Tushare token: ")
                config["tushare"]["token"] = token
        else:
            token = input("请输入Tushare token: ")
            config["tushare"]["token"] = token
    else:
        config["tushare"]["enabled"] = "false"
    
    # AKShare配置
    if "akshare" not in config:
        config["akshare"] = {}
    
    use_akshare = input("是否使用AKShare数据源? (y/n) [y]: ").lower() or "y"
    if use_akshare == "y":
        config["akshare"]["enabled"] = "true"
    else:
        config["akshare"]["enabled"] = "false"
    
    # 保存配置
    with open(config_file, 'w') as f:
        config.write(f)
    
    print(f"✅ 数据源配置已保存到: {config_file}")
    
    # 生成Python配置文件
    generate_python_config(config)
    
    return True

def generate_python_config(config):
    """根据配置文件生成Python配置模块"""
    config_dir = os.path.join(ROOT_DIR, "src", "config")
    os.makedirs(config_dir, exist_ok=True)
    
    init_file = os.path.join(config_dir, "__init__.py")
    
    with open(init_file, 'w') as f:
        f.write("""# -*- coding: utf-8 -*-

\"\"\"
配置模块初始化文件
由setup.py自动生成，请勿手动修改
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
        "username": "%s",
        "password": "%s",
        "enabled": %s,
    },
    # Tushare配置
    "tushare": {
        "token": "%s",
        "enabled": %s,
    },
    # AKShare配置
    "akshare": {
        "enabled": %s,
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
""" % (
            config["joinquant"].get("username", ""),
            config["joinquant"].get("password", ""),
            config["joinquant"].get("enabled", "false").lower() == "true",
            config["tushare"].get("token", ""),
            config["tushare"].get("enabled", "false").lower() == "true",
            config["akshare"].get("enabled", "false").lower() == "true",
        ))
    
    print(f"✅ Python配置文件已生成: {init_file}")

def test_data_sources():
    """测试数据源连接"""
    print("\n" + "=" * 80)
    print(" 测试数据源连接 ".center(80, "="))
    print("=" * 80)
    
    try:
        # 测试JoinQuant连接
        use_joinquant = input("是否测试JoinQuant连接? (y/n) [y]: ").lower() or "y"
        if use_joinquant == "y":
            print("正在测试JoinQuant连接...")
            subprocess.run([sys.executable, "test_joinquant.py"], check=True)
        
        # 未来可以添加其他数据源的测试
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False
    
    return True

def main():
    """主函数"""
    print("\n欢迎使用量化交易系统安装配置工具")
    print("此工具将帮助您设置环境并配置数据源\n")
    
    # 设置环境
    if not setup_environment():
        print("❌ 环境设置失败，请检查错误并重试")
        return
    
    # 配置数据源
    if not setup_data_sources():
        print("❌ 数据源配置失败，请检查错误并重试")
        return
    
    # 测试数据源
    if not test_data_sources():
        print("❌ 数据源测试失败，但设置已完成")
    
    print("\n" + "=" * 80)
    print(" 安装配置完成 ".center(80, "="))
    print("=" * 80)
    print("您现在可以运行以下命令开始使用系统:")
    print("1. 运行示例回测: python example_joinquant.py")
    print("2. 测试数据源连接: python test_joinquant.py")
    print("=" * 80)

if __name__ == "__main__":
    main()

setup(
    name="stock_analysis",
    version="0.1.0",
    description="A comprehensive stock data analysis system",
    author="StockAnalysis Team",
    author_email="info@stockanalysis.example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "akshare>=1.10.0",
        "tushare>=1.2.89",
        "requests>=2.31.0",
        "tqdm>=4.62.0",
        "reportlab>=3.6.0",
        "ta-lib>=0.4.24",
        "python-dateutil>=2.8.2",
        "jieba>=0.42.1",
        "snownlp>=0.12.3"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)