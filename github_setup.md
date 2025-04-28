# GitHub仓库设置指南

## 步骤1：创建GitHub仓库
1. 登录您的GitHub账户
2. 点击右上角的"+"图标，然后选择"New repository"
3. 填写仓库名称，例如"stock-analysis-system"
4. 添加描述："基于Python的股票分析系统，支持动量分析和技术指标"
5. 选择"Private"（如果您希望保持代码私密）或"Public"
6. 不要勾选"Initialize this repository with a README"
7. 点击"Create repository"

## 步骤2：将本地仓库连接到GitHub
创建仓库后，GitHub会显示设置说明。请执行以下命令：

```bash
# 添加远程仓库（替换USERNAME为您的GitHub用户名）
git remote add origin https://github.com/USERNAME/stock-analysis-system.git

# 将代码推送到GitHub
git push -u origin main
```

## 步骤3：推送所有分支（如果有）
如果您有其他分支，可以使用以下命令推送：

```bash
git push --all
```

## 步骤4：验证
1. 刷新GitHub仓库页面
2. 您应该能看到所有的代码文件已经上传

## 注意事项
- 确保您的GitHub账户已设置了SSH密钥或通过HTTPS认证
- 对于大文件，可能需要使用Git LFS（大文件存储）
- 定期推送更新：`git push origin main` 