#!/bin/bash

# 代码清理脚本 - 不使用Python来避开pyenv拦截
echo "=============================================="
echo "股票分析系统 - 代码清理工具"
echo "=============================================="
echo "正在执行..."

# 创建备份目录
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "已创建备份目录: $BACKUP_DIR"

# 需要保留的文件
KEEP_FILES=(
  "main_gui.py"
  "stock_analysis_gui.py"
  "simple_gui.py"
  "headless_gui.py"
  "momentum_analysis.py"
  "ma_cross_strategy.py"
  "gui_controller.py"
  "run_final.sh"
  "README.md"
  "requirements.txt"
)

# 需要保留的目录
KEEP_DIRS=(
  "data"
  "logs"
  "results"
  "charts"
  "assets"
  "src"
)

# 清理临时文件
echo "清理临时文件..."
find . -name "__pycache__" -type d | while read dir; do
  echo "备份并删除: $dir"
  cp -r "$dir" "$BACKUP_DIR/"
  rm -rf "$dir"
done

find . -name "*.pyc" -type f | while read file; do
  echo "备份并删除: $file"
  cp "$file" "$BACKUP_DIR/"
  rm -f "$file"
done

find . -name ".DS_Store" -type f | while read file; do
  echo "删除: $file"
  rm -f "$file"
done

# 处理多余的启动文件
echo "清理多余的启动文件..."
for file in $(find . -maxdepth 1 -name "*.py" -o -name "*.sh" | grep -v -E 'run_final.sh|clean_code.sh'); do
  base=$(basename "$file")
  keep=0
  
  # 检查是否在保留列表中
  for keep_file in "${KEEP_FILES[@]}"; do
    if [ "$base" = "$keep_file" ]; then
      keep=1
      break
    fi
  done
  
  # 如果不在保留列表中且符合启动文件模式，则备份并删除
  if [ $keep -eq 0 ]; then
    if [[ $base =~ ^(start|run|launch|direct|bypass).*$ || $base =~ ^.*_(start|run|launch)\..*$ ]]; then
      echo "备份并删除启动文件: $file"
      cp "$file" "$BACKUP_DIR/"
      rm -f "$file"
    fi
  fi
done

# 格式化保留的Python文件
echo "格式化保留的Python文件..."
for file in "${KEEP_FILES[@]}"; do
  if [ -f "$file" ] && [[ $file == *.py ]]; then
    echo "格式化: $file"
    sed -i '' -e 's/[ \t]*$//' "$file"  # 移除行尾空白
    sed -i '' -e '/^$/d' "$file"        # 移除空行
  fi
done

# 统计结果
REMOVED_COUNT=$(find "$BACKUP_DIR" -type f | wc -l)
echo "=============================================="
echo "清理完成!"
echo "- 已备份文件数量: $REMOVED_COUNT"
echo "- 备份目录: $BACKUP_DIR"
echo "==============================================" 