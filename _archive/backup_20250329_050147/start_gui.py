#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票分析系统启动脚本
设置必要的环境并启动GUI界面
"""

import os
import sys
from pathlib import Path
import logging
import traceback
import tushare as ts

# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 创建必要的目录
LOG_DIR = os.path.join(current_dir, "logs")
DATA_DIR = os.path.join(current_dir, "data")
RESULTS_DIR = os.path.join(current_dir, "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
MA_CHARTS_DIR = os.path.join(RESULTS_DIR, "ma_charts")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(MA_CHARTS_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "start_gui.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """检查依赖库是否安装"""
    try:
        import matplotlib
        import pandas
        import numpy
        import tushare
        logger.info("依赖库检查完成，所有必要的库都已安装")
        return True
    except ImportError as e:
        logger.error(f"缺少必要的依赖库: {str(e)}")
        print(f"缺少必要的依赖库: {str(e)}")
        print("请安装缺失的库后再运行程序")
        return False

def main():
    """主函数"""
    print("正在启动股票分析系统...")
    
    # 检查依赖库
    if not check_dependencies():
        sys.exit(1)
    
    # 设置Tushare Token
    tushare_token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    
    # 检查配置文件中是否已设置token
    try:
        config_path = Path(current_dir) / "src" / "enhanced" / "config" / "settings.py"
        if config_path.exists():
            # 读取配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # 检查是否包含token
            if 'TUSHARE_TOKEN' in config_content and tushare_token not in config_content:
                logger.info("正在更新配置文件中的Tushare Token...")
                # 更新配置文件
                updated_content = config_content.replace(
                    'TUSHARE_TOKEN = ""', 
                    f'TUSHARE_TOKEN = "{tushare_token}"'
                )
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                logger.info("配置文件更新完成")
    except Exception as e:
        logger.warning(f"更新配置文件失败: {str(e)}")
    
    try:
        # 设置Tushare Token
        ts.set_token(tushare_token)
        pro = ts.pro_api()
        
        # 测试连接
        stock_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        logger.info(f"Tushare连接成功，获取到 {len(stock_info)} 支股票信息")
        
        # 保存股票列表到本地
        stock_list_path = os.path.join(DATA_DIR, "stock_list.csv")
        stock_info.to_csv(stock_list_path, index=False, encoding='utf-8-sig')
        logger.info(f"股票列表已保存至 {stock_list_path}")
        
        # 导入GUI模块
        try:
            from stock_analysis_gui import main as start_gui
            logger.info("正在启动GUI界面...")
            
            # 启动GUI
            start_gui()
            
        except Exception as e:
            logger.error(f"启动GUI界面失败: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"启动GUI界面失败: {str(e)}")
            print("请检查日志文件了解详细信息")
        
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"启动失败: {str(e)}，详细错误请查看日志文件")
        sys.exit(1)

if __name__ == "__main__":
    main() 