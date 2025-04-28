#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tushare API检查工具
用于验证Tushare API的连接、版本和权限设置
"""

import os
import sys
import json
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
import tushare as ts

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"tushare_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8')
    ]
)

logger = logging.getLogger("tushare_checker")

# 从环境变量或配置中获取Tushare Token
def get_tushare_token():
    # 首先检查环境变量
    token = os.environ.get("TUSHARE_TOKEN", "")
    
    if token:
        logger.info(f"从环境变量获取到Tushare Token: {token[:5]}...{token[-5:]}")
        return token
    
    # 然后尝试从项目配置文件获取
    try:
        # 首先尝试项目特定配置
        try:
            from src.enhanced.config.settings import TUSHARE_TOKEN
            if TUSHARE_TOKEN:
                logger.info(f"从enhanced配置获取到Tushare Token: {TUSHARE_TOKEN[:5]}...{TUSHARE_TOKEN[-5:]}")
                return TUSHARE_TOKEN
        except ImportError:
            pass
        
        # 然后尝试基础配置
        try:
            from src.config.settings import TUSHARE_TOKEN
            if TUSHARE_TOKEN:
                logger.info(f"从基础配置获取到Tushare Token: {TUSHARE_TOKEN[:5]}...{TUSHARE_TOKEN[-5:]}")
                return TUSHARE_TOKEN
        except ImportError:
            pass
        
        # 最后使用默认备用Token
        default_token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
        logger.info(f"使用默认备用Tushare Token: {default_token[:5]}...{default_token[-5:]}")
        return default_token
    
    except Exception as e:
        logger.error(f"获取Tushare Token失败: {str(e)}")
        return ""

def check_basic_connectivity(token):
    """检查基础连接"""
    logger.info("=== 开始检查Tushare API基础连接 ===")
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        logger.info(f"Tushare版本: {ts.__version__}")
        
        # 测试简单接口调用
        start_time = time.time()
        df = pro.trade_cal(exchange='SSE', start_date='20230101', end_date='20230110')
        end_time = time.time()
        
        if df is not None and len(df) > 0:
            logger.info(f"基础连接测试成功，获取到{len(df)}条交易日历数据，耗时: {end_time - start_time:.2f}秒")
            return pro
        else:
            logger.error("基础连接测试失败: 返回空数据")
            return None
    except Exception as e:
        logger.error(f"基础连接测试失败: {str(e)}")
        return None

def check_api_permissions(pro):
    """检查API权限和限制"""
    logger.info("=== 开始检查Tushare API权限 ===")
    if pro is None:
        logger.error("无法检查API权限: 连接失败")
        return
    
    api_tests = [
        {
            "name": "股票列表",
            "func": lambda: pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        },
        {
            "name": "日线行情",
            "func": lambda: pro.daily(ts_code='000001.SZ', start_date='20230101', end_date='20230110')
        },
        {
            "name": "资金流向",
            "func": lambda: pro.moneyflow(ts_code='000001.SZ', start_date='20230101', end_date='20230110')
        },
        {
            "name": "财务指标",
            "func": lambda: pro.fina_indicator(ts_code='000001.SZ', period='20221231')
        },
        {
            "name": "北向资金持股",
            "func": lambda: pro.hk_hold(ts_code='000001.SZ', start_date='20230101', end_date='20230110')
        }
    ]
    
    results = {}
    for test in api_tests:
        name = test["name"]
        logger.info(f"测试接口: {name}")
        try:
            start_time = time.time()
            df = test["func"]()
            end_time = time.time()
            
            if df is not None and len(df) > 0:
                results[name] = {
                    "status": "成功",
                    "rows": len(df),
                    "time": end_time - start_time
                }
                logger.info(f"  - 状态: 成功, 获取到{len(df)}条数据, 耗时: {end_time - start_time:.2f}秒")
            else:
                results[name] = {
                    "status": "失败: 空数据",
                    "rows": 0,
                    "time": end_time - start_time
                }
                logger.warning(f"  - 状态: 失败，获取到空数据，耗时: {end_time - start_time:.2f}秒")
        except Exception as e:
            results[name] = {
                "status": f"错误: {str(e)}",
                "rows": 0,
                "time": 0
            }
            logger.error(f"  - 状态: 错误, 信息: {str(e)}")
    
    # 检查结果汇总
    success_count = sum(1 for r in results.values() if r["status"] == "成功")
    logger.info(f"=== API检查结果汇总: {success_count}/{len(api_tests)} 成功 ===")
    return results

def check_rate_limits(pro):
    """检查API调用频率限制"""
    logger.info("=== 开始检查Tushare API调用频率限制 ===")
    if pro is None:
        logger.error("无法检查频率限制: 连接失败")
        return
    
    try:
        # 连续调用API检查频率限制
        times = []
        errors = 0
        
        # 测试10次连续调用
        for i in range(1, 11):
            start_time = time.time()
            try:
                df = pro.daily(ts_code='000001.SZ', start_date='20230101', end_date='20230110')
                if df is not None and len(df) > 0:
                    end_time = time.time()
                    call_time = end_time - start_time
                    times.append(call_time)
                    logger.info(f"调用 #{i}: 成功, 耗时: {call_time:.2f}秒")
                else:
                    logger.warning(f"调用 #{i}: 失败，返回空数据")
                    errors += 1
            except Exception as e:
                logger.error(f"调用 #{i}: 错误, 信息: {str(e)}")
                errors += 1
            
            # 短暂等待避免触发限流
            time.sleep(0.2)
        
        if times:
            avg_time = sum(times) / len(times)
            logger.info(f"=== 频率测试结果: 平均调用时间: {avg_time:.2f}秒, 错误次数: {errors} ===")
        else:
            logger.error("所有API调用都失败，无法评估频率限制")
    except Exception as e:
        logger.error(f"检查频率限制时出错: {str(e)}")

def generate_report(token, permissions_results):
    """生成检查报告"""
    logger.info("=== 生成Tushare API检查报告 ===")
    
    report = {
        "检查时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Tushare版本": ts.__version__,
        "Token信息": {
            "Token前5位": token[:5] if token else "无",
            "Token后5位": token[-5:] if token else "无",
            "Token长度": len(token) if token else 0
        },
        "接口测试结果": permissions_results
    }
    
    report_file = f"tushare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    logger.info(f"检查报告已保存至: {report_file}")
    
    # 打印报告摘要
    logger.info("=== Tushare API检查报告摘要 ===")
    logger.info(f"检查时间: {report['检查时间']}")
    logger.info(f"Tushare版本: {report['Tushare版本']}")
    logger.info(f"Token: {token[:5]}...{token[-5:]}, 长度: {len(token)}")
    
    success_count = sum(1 for r in permissions_results.values() if r["status"] == "成功")
    logger.info(f"接口测试: {success_count}/{len(permissions_results)} 成功")
    
    # 给出建议
    if success_count == len(permissions_results):
        logger.info("结论: Tushare API工作正常，所有接口测试通过")
    elif success_count > 0:
        logger.info("结论: Tushare API部分工作，但有些接口测试失败")
    else:
        logger.info("结论: Tushare API不可用，所有接口测试均失败")

def main():
    """主函数"""
    try:
        logger.info("开始Tushare API检查")
        
        # 获取Token
        token = get_tushare_token()
        if not token:
            logger.error("未找到有效的Tushare Token，无法继续检查")
            return
        
        logger.info(f"成功获取Token: {token[:5]}...{token[-5:]}")
        
        # 检查基础连接
        try:
            pro = check_basic_connectivity(token)
            if pro is None:
                logger.error("Tushare API基础连接失败，无法继续检查")
                return
        except Exception as e:
            logger.error(f"基础连接检查过程中发生异常: {str(e)}", exc_info=True)
            return
        
        # 检查API权限
        try:
            permissions_results = check_api_permissions(pro)
        except Exception as e:
            logger.error(f"API权限检查过程中发生异常: {str(e)}", exc_info=True)
            permissions_results = {}
        
        # 检查频率限制
        try:
            check_rate_limits(pro)
        except Exception as e:
            logger.error(f"频率限制检查过程中发生异常: {str(e)}", exc_info=True)
        
        # 生成报告
        try:
            generate_report(token, permissions_results)
        except Exception as e:
            logger.error(f"生成报告过程中发生异常: {str(e)}", exc_info=True)
        
        logger.info("Tushare API检查完成")
    except Exception as e:
        logger.error(f"执行检查过程中发生严重错误: {str(e)}", exc_info=True)
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main() 