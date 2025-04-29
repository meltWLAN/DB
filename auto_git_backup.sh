#!/bin/bash

# 自动Git备份脚本
# 用法: ./auto_git_backup.sh [commit message]

# 设置日志文件
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/git_backup_$(date +%Y%m%d_%H%M%S).log"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 错误处理函数
handle_error() {
    log "错误: $1"
    exit 1
}

# 检查git是否安装
if ! command -v git &> /dev/null; then
    handle_error "Git未安装，请先安装Git"
fi

# 检查是否在git仓库中
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    log "初始化Git仓库..."
    git init || handle_error "Git初始化失败"
    log "Git仓库初始化成功"
fi

# 获取commit消息
COMMIT_MSG="$1"
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="自动备份 - $(date '+%Y-%m-%d %H:%M:%S')"
fi

# 执行备份
log "开始备份..."

# 添加所有更改
log "添加文件到暂存区..."
git add . || handle_error "添加文件失败"

# 提交更改
log "提交更改..."
git commit -m "$COMMIT_MSG" || {
    # 如果没有更改，则提示并退出
    if [ $? -eq 1 ]; then
        log "没有需要提交的更改"
        exit 0
    else
        handle_error "提交更改失败"
    fi
}

# 检查是否配置了远程仓库
REMOTE_URL=$(git remote get-url origin 2>/dev/null)
if [ -z "$REMOTE_URL" ]; then
    log "警告: 未配置远程仓库，仅完成本地备份"
    log "请使用以下命令添加远程仓库："
    log "git remote add origin <your-github-repo-url>"
    log "git branch -M main"
    log "git push -u origin main"
else
    # 推送到远程仓库
    log "推送到远程仓库..."
    git push || handle_error "推送到远程仓库失败"
    log "成功推送到远程仓库"
fi

log "备份完成"

# 可选：清理30天前的日志文件
find . -name "git_backup_*.log" -type f -mtime +30 -delete

exit 0 