# 股票分析系统 - 单一界面版本

## 简介
这是股票分析系统的单一界面版本，整合了所有核心功能，提供了更简洁的用户体验。通过一个主界面就能完成所有分析任务，不再需要切换多个工具。

## 优势
- **统一入口**: 所有分析功能集成在一个界面中
- **简化操作**: 不再需要学习多个工具的用法
- **一致体验**: 所有功能保持相同的UI风格
- **维护简便**: 只需维护一个核心界面文件

## 包含的功能
系统保留了所有核心分析功能:
1. **动量分析**: 基于多种技术指标的动量分析系统
2. **均线交叉策略**: 短期/长期均线交叉信号生成和回测
3. **组合策略**: 动量和均线交叉的加权组合分析
4. **市场概览**: 市场与行业表现数据

## 系统结构
系统由以下几个关键组件构成:
- `stock_analysis_gui.py`: 主GUI界面
- `momentum_analysis.py`: 动量分析模块
- `ma_cross_strategy.py`: 均线交叉策略模块
- `gui_controller.py`: GUI控制器
- `main_launcher.py`: 图形化启动器
- `launch.command`: macOS兼容的启动脚本

## 启动方式
以下是几种启动股票分析系统的方式:

### 1. 通过启动脚本(推荐)
```bash
# 在macOS上双击或在终端执行
./launch.command
```

### 2. 通过图形化启动器
```bash
python main_launcher.py
```

### 3. 直接启动主界面
```bash
python stock_analysis_gui.py
```

## 系统要求
- Python 3.7+
- 依赖库: pandas, numpy, matplotlib, tkinter, pillow等
- 运行环境: macOS, Windows或Linux

## 故障排除
如果遇到启动问题，可以尝试以下方法:
1. 确保已安装所有依赖: `pip install -r requirements.txt`
2. 检查日志目录中的错误信息: `logs/`
3. 在终端中直接运行`python stock_analysis_gui.py`查看详细错误信息

## 注意事项
- 新版本只保留必要组件，如果需要完整功能集，请使用备份目录中的原始文件
- 系统首次启动时可能需要较长时间加载数据
- 实时行情数据需要有效的Tushare API Token 