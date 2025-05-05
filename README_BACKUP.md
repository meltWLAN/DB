# 股票分析系统 - GitHub备份指南

## 配置步骤

1. 确保已安装Git
```bash
git --version
```

2. 配置Git用户信息
```bash
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"
```

3. 在GitHub上创建新仓库
- 访问 https://github.com/new
- 输入仓库名称（例如：stock-analysis-system）
- 选择私有（Private）或公开（Public）
- 不要初始化仓库（不要添加README、.gitignore或许可证）

4. 配置本地仓库
```bash
# 初始化本地仓库（如果还没有）
git init

# 添加远程仓库
git remote add origin https://github.com/你的用户名/你的仓库名.git

# 设置主分支名称
git branch -M main
```

## 使用自动备份脚本

1. 赋予脚本执行权限
```bash
chmod +x auto_git_backup.sh
```

2. 运行备份脚本
```bash
# 使用默认提交信息
./auto_git_backup.sh

# 或指定自定义提交信息
./auto_git_backup.sh "更新了分析算法"
```

## 自动备份说明

- 脚本会自动处理以下任务：
  - 检查Git安装状态
  - 初始化Git仓库（如果需要）
  - 添加所有更改的文件
  - 创建提交
  - 推送到远程仓库
  - 记录详细日志

- 日志文件位置：
  - 所有备份操作的日志都保存在 `logs/git_backup_*.log`

## 注意事项

1. 敏感信息保护
- 确保敏感配置信息（如API密钥）已添加到.gitignore
- 检查提交历史中是否意外包含敏感信息

2. 大文件处理
- 避免提交大型数据文件
- 考虑使用Git LFS存储大文件

3. 定期备份
- 建议每天进行至少一次备份
- 可以设置cron任务自动执行备份脚本

## 设置自动备份（可选）

1. 编辑crontab
```bash
crontab -e
```

2. 添加定时任务（例如每天晚上11点备份）
```bash
0 23 * * * cd /path/to/your/project && ./auto_git_backup.sh
```

## 故障排除

1. 如果推送失败：
- 检查网络连接
- 验证GitHub凭据
- 确认远程仓库URL正确

2. 如果提交失败：
- 检查是否有文件权限问题
- 确认Git用户配置正确

3. 如果脚本执行失败：
- 检查脚本权限
- 查看日志文件了解详细错误信息

## 支持

如果遇到问题，请：
1. 查看日志文件了解详细信息
2. 检查GitHub状态页面
3. 联系系统管理员寻求帮助 