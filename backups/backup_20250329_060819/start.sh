#!/bin/bash

# 创建必要的目录
mkdir -p logs results/charts results/ma_charts results/momentum results/ma_cross data

# 显示说明
echo "=========================================="
echo "   股票分析系统 - 结果查看器启动脚本"
echo "=========================================="
echo ""
echo "正在启动结果查看器..."
echo "这是一个独立的查看器，用于显示分析结果"
echo "它不会执行任何分析操作，仅用于查看已有结果"
echo ""

# 将最新的结果文件复制到指定目录
if ls results/momentum_*.csv &> /dev/null; then
    echo "正在整理动量分析结果文件..."
    cp results/momentum_*.csv results/momentum/ 2>/dev/null
fi

if ls results/ma_strategy_*.csv &> /dev/null; then
    echo "正在整理均线交叉分析结果文件..."
    cp results/ma_strategy_*.csv results/ma_cross/ 2>/dev/null
fi

# 启动结果查看器
echo "启动结果查看器..."
python view_results.py 