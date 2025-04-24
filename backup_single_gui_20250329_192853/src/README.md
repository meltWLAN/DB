# 股票推荐系统 - 模块结构

本项目是一个股票推荐系统，主要功能包括数据获取、分析、推荐和可视化等。

## 目录结构

```
src/
├── config/              # 配置相关模块
│   └── settings.py      # 全局配置文件
│
├── data/                # 数据处理相关模块
│   ├── fetcher.py       # 数据获取
│   ├── processor.py     # 数据处理
│   └── storage.py       # 数据存储
│
├── analysis/            # 分析相关模块
│   ├── indicators.py    # 技术指标
│   ├── sentiment.py     # 情感分析
│   └── patterns.py      # 图表模式识别
│
├── recommendation/      # 推荐相关模块
│   ├── engine.py        # 推荐引擎
│   ├── strategies.py    # 推荐策略
│   └── backtesting.py   # 回测系统
│
├── visualization/       # 可视化相关模块
│   ├── ui_layout.py     # UI布局管理模块
│   ├── app_ui.py        # 应用程序界面
│   ├── charts.py        # 图表绘制
│   └── templates.py     # 报告模板
│
├── utils/               # 工具函数
│   ├── logger.py        # 日志工具
│   ├── cache.py         # 缓存管理
│   └── helpers.py       # 辅助函数
│
└── main.py              # 主程序入口
```

## 模块说明

### 配置模块 (config)

- `settings.py`: 全局配置文件，包含各种参数设置，如数据源、API密钥、UI配置等。

### 数据模块 (data)

- `fetcher.py`: 负责从各种数据源（如Yahoo Finance、Alpha Vantage等）获取股票数据。
- `processor.py`: 负责数据清洗、归一化、特征工程等处理。
- `storage.py`: 管理数据的存储和检索，支持本地文件和数据库存储。

### 分析模块 (analysis)

- `indicators.py`: 实现各种技术指标的计算，如移动平均线、RSI、MACD等。
- `sentiment.py`: 实现对新闻、社交媒体等文本的情感分析。
- `patterns.py`: 实现对股票图表模式的识别，如头肩顶、双底等。

### 推荐模块 (recommendation)

- `engine.py`: 推荐引擎核心，整合各种分析结果，生成推荐。
- `strategies.py`: 实现各种推荐策略，如动量策略、价值策略等。
- `backtesting.py`: 提供回测功能，评估推荐策略的历史表现。

### 可视化模块 (visualization)

- `ui_layout.py`: UI布局管理模块，定义主题、风格和组件。
- `app_ui.py`: 应用程序主界面，使用Tkinter构建GUI。
- `charts.py`: 负责各种图表的绘制，如K线图、技术指标图等。
- `templates.py`: 定义报告模板，用于生成分析报告。

### 工具模块 (utils)

- `logger.py`: 日志工具，用于记录应用程序运行状态和错误。
- `cache.py`: 缓存管理，优化数据访问性能。
- `helpers.py`: 通用辅助函数，如日期处理、数值格式化等。

## 主程序

- `main.py`: 应用程序入口，初始化各模块并启动UI。 