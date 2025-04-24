import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = ROOT_DIR / "data"

# 创建数据目录
os.makedirs(DATA_DIR, exist_ok=True)

# 数据源配置
DATA_SOURCE = {
    "tushare": {
        "token": "YOUR_TUSHARE_TOKEN",
        "use": True
    },
    "akshare": {
        "use": True
    },
    "baostock": {
        "use": True
    }
}

# 数据库配置
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": None
}

CLICKHOUSE_CONFIG = {
    "host": "localhost",
    "port": 9000,
    "user": "default",
    "password": "",
    "database": "stock_analysis"
}

# 技术指标参数
INDICATOR_PARAMS = {
    "bollinger_bands": {
        "window": 20,
        "window_dev": 2
    },
    "macd": {
        "fast": 12,
        "slow": 26,
        "signal": 9
    },
    "rsi": {
        "window": 14,
        "threshold_overbought": 70,
        "threshold_oversold": 30,
        "threshold_uptrend": 50
    },
    "volume": {
        "short_window": 5,
        "long_window": 20
    },
    "moving_averages": {
        "short_windows": [5, 10],
        "medium_windows": [20, 30],
        "long_windows": [60, 120, 250]
    }
}

# 资金流向参数
MONEY_FLOW_PARAMS = {
    "mfi": {
        "window": 14,
        "threshold_overbought": 80,
        "threshold_oversold": 20
    },
    "net_inflow": {
        "days": [1, 3, 5, 10, 20]
    }
}

# 市场情绪参数
SENTIMENT_PARAMS = {
    "news_sources": [
        "eastmoney", "sina", "netease", "jrj"
    ],
    "update_interval": 60  # 分钟
}

# 高频数据监控参数
REALTIME_MONITOR_PARAMS = {
    "interval": "1min",  # 1min, 5min, 15min, 30min, 60min
    "price_change_threshold": 0.03,  # 3%
    "volume_change_threshold": 2.0,  # 量能翻倍
    "update_frequency": 60  # 秒
}

# 风险控制参数
RISK_CONTROL_PARAMS = {
    "max_position_per_stock": 0.1,  # 单只股票最大仓位
    "max_position_per_industry": 0.3,  # 单个行业最大仓位
    "stop_loss": {
        "fixed": 0.05,  # 固定止损
        "atr_multiplier": 2.0  # ATR倍数
    },
    "take_profit": {
        "fixed": 0.1,  # 固定止盈
        "trailing_percentage": 0.5  # 跟踪止盈比例
    }
}

# 股票筛选参数
STOCK_SELECTION_PARAMS = {
    "continuous_limit_up_days": [2, 3, 5],  # 连续涨停天数
    "max_stocks_to_monitor": 100,  # 最大监控股票数量
    "score_threshold": 80,  # 综合评分阈值 (0-100)
}

# 通知推送配置
NOTIFICATION_CONFIG = {
    "enable_sms": False,
    "enable_email": True,
    "enable_app_push": True,
    "email": {
        "sender": "your_email@example.com",
        "password": "your_email_password",
        "smtp_server": "smtp.example.com",
        "smtp_port": 587
    }
}

# 回测参数
BACKTEST_PARAMS = {
    "start_date": "2020-01-01",
    "end_date": "2023-01-01",
    "initial_capital": 1000000,
    "commission": 0.0003,  # 手续费
    "slippage": 0.0001  # 滑点
} 