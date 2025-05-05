#!/bin/bash

# 修复pyenv配置问题
echo "正在修复pyenv配置问题..."

# 移除已经存在但可能损坏的Python版本
rm -rf ~/.pyenv/versions/3.8.18

# 禁用pyenv的自动安装功能
if [ -f ~/.pyenv/version ]; then
  mv ~/.pyenv/version ~/.pyenv/version.bak
  echo "已备份~/.pyenv/version到~/.pyenv/version.bak"
fi

# 暂时禁用pyenv的shell集成
echo "禁用pyenv shell集成..."
cat > ~/.pyenv_fix << 'EOF'
# 暂时禁用pyenv初始化
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
# 仅初始化pyenv但不启用自动安装等功能
pyenv() {
  local command
  command="${1:-}"
  if [ "$#" -gt 0 ]; then
    shift
  fi
  case "$command" in
    shell|activate|deactivate)
      echo "pyenv $command已暂时禁用，系统修复中"
      ;;
    *)
      command pyenv "$command" "$@"
      ;;
  esac
}
EOF

echo "修复完成！请在新的终端窗口中运行以下命令："
echo "source ~/.pyenv_fix && ./pure_start.sh"

echo "或者，您可以完全关闭终端，然后在Finder中双击direct_bypass.command文件" 