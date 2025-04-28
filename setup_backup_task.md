# 设置定时备份任务

## macOS系统设置定时备份

### 方法1：使用crontab

1. 打开终端
2. 输入以下命令编辑crontab：
   ```bash
   crontab -e
   ```
3. 添加以下内容（每天凌晨2点执行备份）：
   ```
   0 2 * * * cd /Users/mac/Desktop/DB && ./auto_git_backup.sh
   ```
4. 保存并退出编辑器

### 方法2：使用launchd

1. 创建一个plist文件：
   ```bash
   nano ~/Library/LaunchAgents/com.stockanalysis.backup.plist
   ```

2. 添加以下内容：
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
   <dict>
       <key>Label</key>
       <string>com.stockanalysis.backup</string>
       <key>ProgramArguments</key>
       <array>
           <string>/Users/mac/Desktop/DB/auto_git_backup.sh</string>
       </array>
       <key>StartCalendarInterval</key>
       <dict>
           <key>Hour</key>
           <integer>2</integer>
           <key>Minute</key>
           <integer>0</integer>
       </dict>
       <key>WorkingDirectory</key>
       <string>/Users/mac/Desktop/DB</string>
       <key>StandardErrorPath</key>
       <string>/Users/mac/Desktop/DB/logs/backup_error.log</string>
       <key>StandardOutPath</key>
       <string>/Users/mac/Desktop/DB/logs/backup_output.log</string>
   </dict>
   </plist>
   ```

3. 加载任务：
   ```bash
   launchctl load ~/Library/LaunchAgents/com.stockanalysis.backup.plist
   ```

## Windows系统设置定时备份

1. 打开任务计划程序
2. 点击"创建任务"
3. 填写名称："股票分析系统备份"
4. 选择"配置为Windows 10"
5. 切换到"触发器"选项卡，点击"新建"
6. 选择"每天"，设置开始时间为凌晨2:00
7. 切换到"操作"选项卡，点击"新建"
8. 选择"启动程序"
9. 在"程序或脚本"中输入：`bash`
10. 在"添加参数"中输入：`C:\path\to\auto_git_backup.sh`
11. 在"起始于"中输入：`C:\path\to\DB`
12. 点击"确定"保存任务

## Linux系统设置定时备份

1. 打开终端
2. 输入以下命令编辑crontab：
   ```bash
   crontab -e
   ```
3. 添加以下内容（每天凌晨2点执行备份）：
   ```
   0 2 * * * cd /path/to/DB && ./auto_git_backup.sh
   ```
4. 保存并退出编辑器

## 注意事项

- 确保备份脚本有执行权限：`chmod +x auto_git_backup.sh`
- 定时任务可能需要配置SSH密钥，以便无需输入密码进行推送
- 对于macOS的launchd方式，需要创建logs目录：`mkdir -p logs`
- 定期检查备份日志，确保备份正常进行 