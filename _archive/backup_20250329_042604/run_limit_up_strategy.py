#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行连续涨停和大幅上涨股票捕捉策略
"""

import os
import sys
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入策略
from limit_up_capture_strategy import LimitUpCaptureStrategy

def main():
    """主函数，运行涨停和大幅上涨股票捕捉策略"""
    print("=" * 80)
    print(" 连续涨停和大幅上涨股票捕捉策略 ".center(80, "="))
    print("=" * 80)
    
    try:
        # 配置JoinQuant账号 (如果有)
        config_joinquant = input("是否配置JoinQuant账号? (y/n) [n]: ").lower() or "n"
        
        if config_joinquant == "y":
            username = input("JoinQuant用户名: ")
            password = input("JoinQuant密码: ")
            
            # 更新配置文件
            config_file = "src/config/__init__.py"
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # 替换配置项
                content = content.replace(
                    '"username": "",',
                    f'"username": "{username}",'
                )
                content = content.replace(
                    '"password": "",',
                    f'"password": "{password}",'
                )
                
                # 写回配置文件
                with open(config_file, "w", encoding="utf-8") as f:
                    f.write(content)
                
                print("✅ JoinQuant账号配置已更新")
        
        # 执行策略
        strategy = LimitUpCaptureStrategy()
        high_potential_stocks = strategy.run()
        
        if high_potential_stocks is not None and len(high_potential_stocks) > 0:
            print("\n🔍 分析结果已保存到results/limit_up_capture/目录下")
            print("\n✅ 策略运行成功")
        else:
            print("\n❌ 策略运行失败或未找到符合条件的股票")
            
    except Exception as e:
        logger.error(f"运行策略时发生错误: {e}")
        print(f"\n❌ 运行策略时发生错误: {e}")
    
    print("=" * 80)

if __name__ == "__main__":
    main() 