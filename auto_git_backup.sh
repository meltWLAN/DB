#!/bin/bash

# 股票分析系统GitHub自动备份脚本

# 设置日志文件
LOG_FILE="git_backup_$(date +%Y%m%d_%H%M%S).log"
REPO_DIR="$(pwd)"

echo "===== 开始Git备份 $(date) =====" | tee -a "$LOG_FILE"

# 进入项目目录
cd "$REPO_DIR"

# 检查是否在正确的目录
if [ ! -d .git ]; then
    echo "错误：当前目录不是Git仓库根目录" | tee -a "$LOG_FILE"
    exit 1
fi

# 添加所有修改（排除缓存和临时文件）
echo "添加修改的文件..." | tee -a "$LOG_FILE"
git add *.py README*.md requirements.txt .gitignore src/ 2>&1 | tee -a "$LOG_FILE"

# 获取工作目录状态
STATUS=$(git status --porcelain)

# 如果有修改，则提交并推送
if [ -n "$STATUS" ]; then
    echo "发现修改，准备提交..." | tee -a "$LOG_FILE"
    COMMIT_MSG="自动备份：$(date +%Y-%m-%d_%H:%M:%S)"
    git commit -m "$COMMIT_MSG" 2>&1 | tee -a "$LOG_FILE"
    
    # 推送到GitHub
    echo "推送到GitHub..." | tee -a "$LOG_FILE"
    git push origin main 2>&1 | tee -a "$LOG_FILE"
    
    echo "备份完成！" | tee -a "$LOG_FILE"
else
    echo "没有发现修改，无需备份" | tee -a "$LOG_FILE"
fi

echo "===== 备份过程结束 $(date) =====" | tee -a "$LOG_FILE"

# 可选：清理30天前的日志文件
find . -name "git_backup_*.log" -type f -mtime +30 -delete

exit 0 