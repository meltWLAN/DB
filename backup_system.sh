#!/bin/bash

# 股票分析系统备份脚本
# 作者: Claude
# 版本: 1.0
# 创建日期: 2025-03-30

# 设置变量
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
BACKUP_ROOT="$HOME/Desktop/backup"
BACKUP_DIR="$BACKUP_ROOT/DB_$TIMESTAMP"

# 创建备份目录
echo "创建备份目录: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# 执行备份
echo "开始备份系统文件..."
rsync -av --exclude='backup*' --exclude='.DS_Store' --exclude='__pycache__' . "$BACKUP_DIR/"

# 创建备份完成标记
echo "创建备份完成确认文件..."
cat > "$BACKUP_DIR/BACKUP_COMPLETE.md" << EOF
# 股票分析系统备份确认

## 备份详情
- **备份时间**: $(date "+%Y-%m-%d %H:%M:%S")
- **备份位置**: $BACKUP_DIR
- **备份内容**: 股票分析系统全部文件
- **备份方式**: rsync 增量备份
- **排除内容**: 
  - .DS_Store 系统文件
  - __pycache__ 缓存文件
  - backup* 备份文件

## 系统文件结构
- src/ - 核心源代码目录
- data/ - 数据文件目录
- results/ - 分析结果输出目录
- logs/ - 系统日志文件
- tests/ - 测试脚本

## 重要功能模块
1. 动量分析策略
2. 均线交叉策略
3. 行业分析工具
4. 回测引擎
5. 数据获取与处理
6. 可视化报表生成

## 恢复注意事项
请确保您的Python环境已正确配置，系统需要以下依赖库：
- numpy
- pandas
- matplotlib
- tkinter
- tushare/akshare

您可以通过运行 \`run_system.py\` 或 \`main_gui.py\` 启动系统。
EOF

# 记录备份信息到日志
echo "$TIMESTAMP: 系统备份已完成，备份位置: $BACKUP_DIR" >> "$HOME/Desktop/backup/backup_history.log"

echo "备份完成！系统已备份至: $BACKUP_DIR"
echo "备份历史记录已添加到: $HOME/Desktop/backup/backup_history.log" 