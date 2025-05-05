#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from jqdatasdk import auth, get_price, get_all_securities, get_query_count

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_config():
    try:
        if os.path.exists('config.json'):
            with open('config.json', 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"加载配置文件失败: {str(e)}")
    return {}

def check_account_info():
    try:
        # 从配置文件获取用户名和密码
        config = load_config()
        username = config.get('username')
        password = config.get('password')
        
        if not username or not password:
            logging.error("未找到JoinQuant账号信息，请确保config.json文件中包含正确的username和password")
            return False
            
        logging.info("正在尝试登录JoinQuant...")
        auth(username, password)
        logging.info("登录成功！")
        
        # 获取查询剩余次数
        count = get_query_count()
        logging.info("\n=== JoinQuant账户信息 ===")
        logging.info(f"用户名: {username}")
        logging.info(f"今日剩余查询次数: {count['total']}")
        logging.info(f"当前可用积分: {count['spare']}")
        logging.info("======================\n")
        
        return True
        
    except Exception as e:
        logging.error(f"获取账户信息失败: {str(e)}")
        return False

if __name__ == "__main__":
    check_account_info() 