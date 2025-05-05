"""
系统配置文件
包含所有可配置的参数和常量
"""

import os
from pathlib import Path
from typing import Dict, Any
import json
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

# 基础路径配置
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"

# 创建必要的目录
for dir_path in [DATA_DIR, LOG_DIR, CACHE_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOG_DIR / "stock_recommender.log"

# 数据源配置
DATA_SOURCES = {
    "tushare": {
        "token": os.getenv("TUSHARE_TOKEN", ""),
        "timeout": 30,
    },
    "akshare": {
        "timeout": 30,
    }
}

# 数据获取配置
DATA_FETCH_CONFIG = {
    "default_source": "yfinance",  # 默认数据源
    "api_keys": {
        "alpha_vantage": os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
        "quandl": os.environ.get("QUANDL_API_KEY", ""),
        "iex": os.environ.get("IEX_API_KEY", ""),
    },
    "timeout": 30,  # 请求超时时间（秒）
    "retry_count": 3,  # 请求失败重试次数
    "rate_limit": {  # 请求频率限制
        "yfinance": {"requests": 2000, "period": "day"},
        "alpha_vantage": {"requests": 500, "period": "day"},
        "quandl": {"requests": 300, "period": "day"},
        "iex": {"requests": 100, "period": "minute"},
    }
}

# 数据缓存配置
CACHE_CONFIG = {
    "enabled": True,  # 是否启用缓存
    "expiry": {  # 缓存过期时间（秒）
        "price_data": 86400,  # 价格数据缓存1天
        "fundamental_data": 604800,  # 基本面数据缓存7天
        "market_data": 3600,  # 市场数据缓存1小时
    },
    "max_size": 1000,  # 最大缓存条目数
}

# 技术指标配置
INDICATORS_CONFIG = {
    "default_periods": {
        "sma": [5, 10, 20, 30, 60],  # 简单移动平均线周期
        "ema": [5, 10, 20, 30, 60],  # 指数移动平均线周期
        "rsi": 14,  # 相对强弱指数周期
        "macd": {"fast": 12, "slow": 26, "signal": 9},  # MACD参数
        "bollinger": {"period": 20, "std_dev": 2},  # 布林带参数
        "atr": 14,  # 平均真实范围周期
        "kdj": {"k_period": 9, "d_period": 3, "j_period": 3},  # KDJ参数
    },
    "thresholds": {
        "rsi": {"oversold": 30, "overbought": 70},  # RSI超买超卖阈值
        "stochastic": {"oversold": 20, "overbought": 80},  # 随机指标超买超卖阈值
    }
}

# 策略参数配置
STRATEGY_CONFIG = {
    "default_parameters": {
        "ma_crossover": {
            "short_period": 5,
            "long_period": 20,
        },
        "rsi_strategy": {
            "period": 14,
            "oversold": 30,
            "overbought": 70,
        },
        "breakout_strategy": {
            "period": 20,
        },
    },
    "position_sizing": {
        "default_method": "percentage",  # 仓位大小计算方法
        "percentage": 0.1,  # 每笔交易使用资金百分比
        "fixed": 10000,  # 固定金额
        "risk_percentage": 0.02,  # 每笔交易风险百分比
    }
}

# 回测配置
BACKTEST_CONFIG = {
    "default_initial_capital": 100000,  # 默认初始资金
    "commission": {  # 手续费设置
        "percentage": 0.0003,  # 交易金额的0.03%
        "minimum": 5,  # 最低手续费5元
    },
    "slippage": {  # 滑点设置
        "type": "fixed",  # 固定滑点
        "value": 0.01,  # 固定滑点值（元）
    },
    "benchmark": "^HSI",  # 默认基准指数（恒生指数）
}

# 数据预处理配置
PREPROCESSING_CONFIG = {
    "default_fillna_method": "ffill",  # 默认填充缺失值方法
    "outlier_detection": {
        "method": "z_score",  # 异常值检测方法
        "threshold": 3.0,  # Z分数阈值
    }
}

# 日志配置
LOG_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": BASE_DIR / "logs" / "stock_recommendation.log",
}

# GUI配置
GUI_CONFIG = {
    "theme": "default",  # 默认主题
    "themes": {
        "default": {
            "background": "#F0F0F0",
            "text": "#333333",
            "accent": "#1E88E5",
            "button": "#FFFFFF",
            "button_text": "#333333",
            "highlight": "#90CAF9",
        },
        "dark": {
            "background": "#333333",
            "text": "#F0F0F0",
            "accent": "#90CAF9",
            "button": "#555555",
            "button_text": "#FFFFFF",
            "highlight": "#1E88E5",
        },
        "light": {
            "background": "#FFFFFF",
            "text": "#333333",
            "accent": "#2196F3",
            "button": "#E0E0E0",
            "button_text": "#333333",
            "highlight": "#64B5F6",
        },
    },
    "chart_colors": {
        "up": "#4CAF50",  # 上涨颜色
        "down": "#F44336",  # 下跌颜色
        "volume": "#2196F3",  # 成交量颜色
        "ma5": "#FF5722",  # 5日均线颜色
        "ma10": "#2196F3",  # 10日均线颜色
        "ma20": "#4CAF50",  # 20日均线颜色
        "ma30": "#9C27B0",  # 30日均线颜色
    },
    "window_size": {
        "width": 1200,
        "height": 800,
    },
    "refresh_interval": 60000,  # 数据刷新间隔（毫秒）
}

# 通知配置
NOTIFICATION_CONFIG = {
    "enabled": True,  # 是否启用通知
    "methods": ["email"],  # 通知方式
    "email": {
        "smtp_server": os.environ.get("SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": int(os.environ.get("SMTP_PORT", 587)),
        "username": os.environ.get("EMAIL_USERNAME", ""),
        "password": os.environ.get("EMAIL_PASSWORD", ""),
        "from_addr": os.environ.get("EMAIL_FROM", ""),
        "to_addr": os.environ.get("EMAIL_TO", ""),
    }
}

# 数据配置
DATA_CONFIG = {
    "default_data_source": "yahoo",
    "cache_expiry": 3600,  # 秒
    "max_cache_size": 100,  # MB
    "batch_size": 50,
    "preload_days": 90,
    "technical_indicators": ["MA", "MACD", "RSI", "BOLL"],
    "default_timeframe": "1d"
}

# 系统配置
SYSTEM_CONFIG = {
    "log_level": "INFO",
    "log_file": "stockrecsys.log",
    "enable_debug": False,
    "max_threads": 4,
    "timeout": 30,  # 秒
    "data_dir": "data",
    "cache_dir": "cache",
    "output_dir": "output"
}

# 推荐系统配置
RECOMMENDATION_CONFIG = {
    "algorithms": ["momentum", "mean_reversion", "trend_following"],
    "default_algorithm": "momentum",
    "risk_levels": ["low", "medium", "high"],
    "default_risk_level": "medium",
    "backtest_period": 90,  # 天
    "confidence_threshold": 0.7,
    "max_recommendations": 10,
    "portfolio_size": 5,
    "rebalance_interval": 7  # 天
}

# 风险控制配置
RISK_CONFIG = {
    "max_position_size": 0.2,  # 单个股票最大仓位
    "max_sector_exposure": 0.4,  # 单个行业最大仓位
    "stop_loss": 0.1,  # 止损比例
    "take_profit": 0.2,  # 止盈比例
}

def load_config(config_file: str) -> Dict[str, Any]:
    """加载自定义配置文件"""
    config_path = BASE_DIR / config_file
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_config(config: Dict[str, Any], config_file: str) -> None:
    """保存配置到文件"""
    config_path = BASE_DIR / config_file
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False) 