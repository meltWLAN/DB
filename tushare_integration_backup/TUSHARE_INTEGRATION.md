# Tushare 市场数据集成

## 概述

本项目集成了 Tushare API 作为真实市场数据源，用于市场概览功能。通过使用 Tushare 提供的数据，系统可以展示真实的市场情况，包括指数数据、行业板块数据、涨跌统计等。

## 文件结构

- `tushare_market_data.py`: Tushare 数据获取模块，负责从 API 获取市场数据
- `market_overview_adapter.py`: 数据适配器，将 Tushare 数据转换为系统需要的格式
- `gui_controller.py`: GUI 控制器，负责调用适配器并展示数据

## 功能

1. **实时市场数据获取**：
   - 获取最近交易日的市场概览数据
   - 获取主要指数数据（上证指数、深证成指等）
   - 获取行业板块数据
   - 统计涨停跌停股票

2. **数据适配转换**：
   - 将原始 Tushare 数据转换为系统所需格式
   - 提供备用生成机制，确保数据获取失败时系统仍能运行
   - 保存获取的数据到 JSON 文件，方便调试

3. **数据展示**：
   - 展示市场整体情况（涨跌家数、平均涨跌幅等）
   - 展示主要指数表现
   - 展示热门行业板块
   - 预测未来可能热门的板块

## 配置

Tushare API 需要使用 token 进行认证。当前系统使用以下 token:

```python
TUSHARE_TOKEN = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
```

如需更新 token，请修改 `tushare_market_data.py` 文件中的 `TUSHARE_TOKEN` 常量。

## 使用方法

1. 确保已安装 Tushare 包：
   ```
   pip install tushare
   ```

2. 启动系统：
   ```
   python stock_analysis_gui.py
   ```

3. 系统将自动使用 Tushare 数据源获取市场概览数据

## 数据结构

市场概览数据包括以下几个部分：

1. **indices**: 主要指数数据
2. **industry_performance**: 行业板块表现
3. **hot_sectors**: 当前热门板块
4. **future_hot_sectors**: 未来可能热门的板块
5. **market_stats**: 市场统计数据（涨跌家数、涨停跌停等）

## 测试

可以单独运行数据获取模块进行测试：

```
python tushare_market_data.py
```

也可以单独运行适配器模块进行测试：

```
python market_overview_adapter.py
```

## 错误处理

系统实现了完善的错误处理机制：

1. 如果 Tushare API 数据获取失败，系统会使用备用的模拟数据
2. 所有异常都会被记录在日志中
3. GUI 会显示适当的错误信息，不会因数据问题而崩溃 