# 股票分析系统

## 简介
这是一个集成了动量分析和均线交叉策略的股票分析系统，提供了可视化的分析工具和回测功能。

## 功能
- **动量分析**: 基于多种技术指标(RSI、MACD等)的动量评分系统
- **均线交叉策略**: 短期/长期均线交叉信号生成和回测
- **组合策略**: 动量分析与均线交叉策略的加权组合
- **市场概览**: 市场指数和行业表现统计

## 安装
```bash
pip install -r requirements.txt
```

## 运行
```bash
# 在macOS/Linux上
./run_main_gui.sh

# 在Windows上
python stock_analysis_gui.py
```

## 系统要求
- Python 3.7+
- 依赖库: pandas, numpy, matplotlib, tkinter, pillow等

## 数据源
系统支持从Tushare获取数据，也可以使用本地数据文件。

## 目录结构
- `stock_analysis_gui.py`: 主界面
- `momentum_analysis.py`: 动量分析模块
- `ma_cross_strategy.py`: 均线交叉策略模块
- `gui_controller.py`: GUI控制器
- `data/`: 数据目录
- `results/`: 结果目录
- `logs/`: 日志目录
