# 增强版市场概览模块

## 概述
增强版市场概览模块提供多维度的市场分析、预测和可视化功能，为交易决策提供全面的市场情报支持。本模块以更深入的数据分析、直观的可视化展示和智能的预测功能，全面提升用户对市场的理解和洞察能力。

## 主要特性

### 1. 多维度市场数据整合
- **指数分析**：对主要指数进行深度技术分析，提供趋势、强度、评分等指标
- **行业板块分析**：分析各行业板块表现，识别强势和弱势板块
- **宏观经济数据**：整合GDP、CPI、PPI等宏观经济指标
- **跨市场数据**：中美股市、债市、商品市场的联动分析

### 2. 资金流向分析
- **北向资金**：详细分析北向资金流入流出情况及其影响
- **机构资金**：跟踪公募、私募、保险、外资等各类机构资金动向
- **融资融券**：融资融券余额变化及其市场影响
- **行业资金轮动**：识别资金在不同行业间的轮动效应

### 3. 市场情绪分析
- **恐慌/贪婪指数**：量化市场情绪，预警过热或过冷
- **技术面综合分析**：市场宽度、强度、动量等多指标综合评估
- **市场状态分类**：基于历史数据对当前市场状态进行分类

### 4. 交互式可视化
- **热力图**：直观展示行业和个股表现
- **资金流向图**：使用桑基图展示资金流向
- **情绪仪表盘**：可视化市场情绪变化
- **多周期对比**：提供多个时间周期的数据对比

### 5. 预测能力
- **机器学习模型**：集成多种机器学习算法预测市场走势
- **时间序列分析**：识别市场规律和周期
- **极端行情预警**：提前识别可能出现的市场异常波动

## 使用方法

### 获取市场概览
```python
from src.enhanced.market.enhanced_market_overview import EnhancedMarketOverview

# 初始化市场概览
market_overview = EnhancedMarketOverview()

# 获取今日市场概览
overview_data = market_overview.get_market_overview()

# 获取指定日期的市场概览
overview_data = market_overview.get_market_overview('2025-05-01')

# 生成市场报告
report = market_overview.generate_market_report()
```

### 使用市场仪表盘UI
```python
from src.enhanced.market.ui.market_dashboard import MarketDashboard
import tkinter as tk

# 创建主窗口
root = tk.Tk()
root.title("市场仪表盘")
root.geometry("1200x800")

# 创建仪表盘
dashboard = MarketDashboard(root)

# 运行
root.mainloop()
```

## 数据字段说明

增强版市场概览返回的数据结构如下：

```
{
    "date": "2025-05-01",
    "market_base": {
        "up_count": 2130,
        "down_count": 1540,
        "flat_count": 95,
        "total_count": 3765,
        "limit_up_count": 78,
        "limit_down_count": 12,
        "total_volume": 598762145263,
        "total_amount": 823561483972.35,
        "avg_change_pct": 0.85,
        "turnover_rate": 5.68
    },
    "indices": [
        {
            "code": "000001.SH",
            "name": "上证指数",
            "type": "综合",
            "close": 3556.23,
            "change": 0.85,
            "change_5d": 2.15,
            "change_10d": 3.78,
            "change_20d": 5.63,
            "volume": 358762145823,
            "amount": 423561927635.25,
            "volume_ratio": 1.08,
            "trend": "上涨",
            "trend_strength": 72.5,
            "rsi": 58.6,
            "macd": 12.53,
            "macd_signal": 8.75,
            "macd_hist": 3.78,
            "score": 78.5
        },
        // 更多指数...
    ],
    "industry": [
        {
            "code": "ELE",
            "name": "电子",
            "change": 1.85,
            "turnover": 7.25,
            "pe": 28.56,
            "total_market_cap": 5623789456213,
            "volume": 58762145823,
            "amount": 83561927635.25,
            "up_count": 135,
            "down_count": 48,
            "flat_count": 5,
            "total_count": 188,
            "limit_up_count": 12,
            "limit_down_count": 1,
            "leading_up": {"code": "000001.SZ", "name": "平安银行", "change": 9.86},
            "leading_down": {"code": "600001.SH", "name": "浦发银行", "change": -3.25},
            "strength_index": 78.6,
            "trend": "强势上涨",
            "change_5d": 5.28,
            "change_10d": 8.35,
            "change_20d": 12.67,
            "momentum_score": 82.5
        },
        // 更多行业...
    ],
    "macro_economic": {
        "gdp_growth": 6.5,
        "cpi": 2.1,
        "ppi": 3.5,
        "m2_growth": 8.5,
        "benchmark_rate": 3.85,
        "exchange_rate": 6.38
    },
    "money_flow": {
        "north_inflow": 586721358.52,
        "north_inflow_5d": 2853621478.65,
        "north_inflow_20d": 10562378936.75,
        "main_force_inflow": 12567835612.35,
        "retail_inflow": -3562897563.24,
        "sector_inflows": [
            {"name": "电子", "inflow": 1562897563.24},
            {"name": "医药", "inflow": 989562375.12},
            // 更多行业资金流向...
        ]
    },
    "market_sentiment": {
        "status": "乐观",
        "score": 72.5,
        "fear_greed_index": 65.8,
        "factors": {
            "market_momentum": 75.2,
            "market_volatility": 35.6,
            "market_breadth": 68.3,
            "money_flow": 62.7,
            "north_flow": 78.9
        }
    },
    "future_hot_sectors": [
        {"name": "半导体", "probability": 0.85, "factors": ["政策支持", "产业升级", "资金流入"]},
        {"name": "新能源", "probability": 0.78, "factors": ["政策支持", "技术突破"]},
        {"name": "生物医药", "probability": 0.65, "factors": ["医改推进", "老龄化"]},
        // 更多预测的热门板块...
    ]
}
```

## 依赖关系
本模块依赖以下组件：
- src.enhanced.data.fetchers.data_source_manager - 数据源管理
- matplotlib, pandas, numpy - 数据处理和可视化
- tkinter - UI界面 