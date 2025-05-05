#!/bin/bash

# 完全绕过pyenv的终极方案
(
# 进入子shell，不传递任何环境变量
exec -c /usr/bin/env -i \
  HOME="$HOME" \
  PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin" \
  /usr/bin/python3 "$(dirname "$0")/stock_analysis_gui.py"
) 