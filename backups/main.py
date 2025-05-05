#!/usr/bin/env python3
"""
股票分析系统主入口
支持命令行参数启动不同的分析模式
"""

import os
import sys
import argparse
import pandas as pd
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加当前目录到Python路径
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    # 导入GUI控制器
    from gui_controller import GuiController
    from enhanced_gui_controller import EnhancedGuiController
except ImportError as e:
    logger.error(f"导入模块失败: {str(e)}")
    logger.error("请确保已安装所有依赖，并且文件在正确的路径")
    sys.exit(1)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='股票分析系统')
    
    # 添加子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 标准分析子命令
    standard_parser = subparsers.add_parser('standard', help='使用标准动量分析')
    standard_parser.add_argument('--sample', type=int, default=100, help='样本大小 (默认: 100)')
    standard_parser.add_argument('--score', type=int, default=70, help='最低分数 (默认: 70)')
    standard_parser.add_argument('--industry', type=str, help='指定行业 (可选)')
    
    # 增强分析子命令
    enhanced_parser = subparsers.add_parser('enhanced', help='使用增强版动量分析')
    enhanced_parser.add_argument('--sample', type=int, default=100, help='样本大小 (默认: 100)')
    enhanced_parser.add_argument('--score', type=int, default=70, help='最低分数 (默认: 70)')
    enhanced_parser.add_argument('--industry', type=str, help='指定行业 (可选)')
    
    # 市场概览子命令
    subparsers.add_parser('market', help='显示市场概览')
    
    # 行业分析子命令
    subparsers.add_parser('industry', help='显示行业分析')
    
    # 详细分析子命令
    detail_parser = subparsers.add_parser('detail', help='详细分析单只股票')
    detail_parser.add_argument('--code', type=str, required=True, help='股票代码 (必需)')
    detail_parser.add_argument('--enhanced', action='store_true', help='使用增强分析 (可选)')
    
    # 导出结果子命令
    export_parser = subparsers.add_parser('export', help='导出分析结果')
    export_parser.add_argument('--sample', type=int, default=100, help='样本大小 (默认: 100)')
    export_parser.add_argument('--score', type=int, default=70, help='最低分数 (默认: 70)')
    export_parser.add_argument('--industry', type=str, help='指定行业 (可选)')
    export_parser.add_argument('--enhanced', action='store_true', help='使用增强分析 (可选)')
    export_parser.add_argument('--output', type=str, default='analysis_results.csv', help='输出文件名 (默认: analysis_results.csv)')
    
    return parser.parse_args()

def print_results(results, detailed=False):
    """打印分析结果"""
    if not results:
        print("没有符合条件的结果")
        return
        
    # 打印表头
    if detailed:
        header = f"{'代码':<12}{'名称':<12}{'行业':<15}{'价格':<8}{'RSI':<8}{'MACD':<8}{'动量':<8}{'得分':<8}"
        if 'money_flow_score' in results[0]:
            header += f"{'资金':<6}{'财务':<6}{'北向':<6}{'行业':<6}"
        print(header)
        print("-" * 100)
    else:
        print(f"{'代码':<12}{'名称':<12}{'行业':<15}{'价格':<8}{'得分':<8}")
        print("-" * 55)
    
    # 打印结果
    for r in results:
        if detailed:
            line = f"{r['code']:<12}{r['name'][:10]:<12}{r.get('industry', '')[:13]:<15}"
            line += f"{r.get('close', 0):<8.2f}{r.get('rsi', 0):<8.2f}{r.get('macd_hist', 0):<8.2f}"
            line += f"{r.get('momentum_20d', 0):<8.2f}{r.get('score', 0):<8.2f}"
            if 'money_flow_score' in r:
                line += f"{r.get('money_flow_score', 0):<6.1f}{r.get('finance_score', 0):<6.1f}"
                line += f"{r.get('north_flow_score', 0):<6.1f}{r.get('industry_factor', 1.0):<6.2f}"
            print(line)
        else:
            print(f"{r['code']:<12}{r['name'][:10]:<12}{r.get('industry', '')[:13]:<15}"
                 f"{r.get('close', 0):<8.2f}{r.get('score', 0):<8.2f}")

def print_industry_analysis(results):
    """打印行业分析结果"""
    if not results:
        print("没有行业分析结果")
        return
        
    # 打印表头
    print(f"{'行业':<15}{'股票数量':<10}{'平均得分':<10}{'行业系数':<10}")
    print("-" * 50)
    
    # 打印结果
    for r in results:
        print(f"{r['industry'][:13]:<15}{r['stock_count']:<10}{r['avg_score']:<10.2f}{r['industry_factor']:<10.2f}")

def print_market_overview(market_data):
    """打印市场概览"""
    if not market_data:
        print("没有市场概览数据")
        return
        
    # 打印指数数据
    print("市场指数:")
    print(f"{'指数名称':<12}{'价格':<8}{'涨跌幅':<8}{'RSI':<8}{'MACD':<8}{'得分':<8}{'趋势':<8}")
    print("-" * 65)
    
    for name, data in market_data.items():
        if isinstance(data, dict) and 'code' in data:
            trend = data.get('trend', '')
            trend_display = '↑↑' if trend == 'up' else ('↓↓' if trend == 'down' else '→')
            print(f"{name:<12}{data.get('close', 0):<8.2f}{data.get('change_pct', 0):<8.2f}"
                 f"{data.get('rsi', 0):<8.2f}{data.get('macd_hist', 0):<8.2f}"
                 f"{data.get('score', 0):<8.2f}{trend_display:<8}")
    
    # 打印市场热度
    if 'market_heat' in market_data:
        heat = market_data['market_heat']
        print("\n市场热度指标:")
        print(f"样本大小: {heat.get('sample_size', 0)}")
        print(f"强势股比例: {heat.get('strong_ratio', 0)*100:.2f}%")
        print(f"弱势股比例: {heat.get('weak_ratio', 0)*100:.2f}%")
        print(f"中性股比例: {heat.get('neutral_ratio', 0)*100:.2f}%")
        print(f"市场状态: {'热' if heat.get('market_status', '') == 'hot' else ('冷' if heat.get('market_status', '') == 'cold' else '中性')}")
    
    # 打印北向资金
    if 'north_flow' in market_data:
        north_flow = market_data['north_flow']
        print("\n北向资金:")
        print(f"最近5日净流入: {north_flow.get('recent_days_flow', 0)/100000000:.2f}亿")
        print(f"月度净流入: {north_flow.get('monthly_flow', 0)/100000000:.2f}亿")
        print(f"趋势: {'流入' if north_flow.get('trend', '') == 'inflow' else '流出'}")

def print_stock_detail(detail):
    """打印股票详情"""
    if not detail:
        print("没有股票详情数据")
        return
        
    print(f"\n{detail['name']}({detail['ts_code']}) - {detail.get('industry', '')}")
    print(f"最新价: {detail.get('close', 0):.2f}")
    print("-" * 50)
    
    print("基础指标:")
    print(f"RSI: {detail.get('base_indicators', {}).get('rsi', 0):.2f}")
    print(f"MACD: {detail.get('base_indicators', {}).get('macd', 0):.2f}")
    print(f"MACD柱: {detail.get('base_indicators', {}).get('macd_hist', 0):.2f}")
    print(f"20日动量: {detail.get('base_indicators', {}).get('momentum_20', 0):.2f}%")
    print(f"成交量比: {detail.get('base_indicators', {}).get('volume_ratio', 0):.2f}")
    
    if 'enhanced_indicators' in detail:
        print("\n增强指标:")
        print(f"资金流向得分: {detail.get('enhanced_indicators', {}).get('money_flow_score', 0):.2f}")
        print(f"财务动量得分: {detail.get('enhanced_indicators', {}).get('finance_score', 0):.2f}")
        print(f"北向资金得分: {detail.get('enhanced_indicators', {}).get('north_flow_score', 0):.2f}")
        print(f"行业因子: {detail.get('enhanced_indicators', {}).get('industry_factor', 1.0):.2f}")
    
    print(f"\n总得分: {detail.get('total_score', 0):.2f}")
    
    if 'chart_path' in detail:
        print(f"\n图表已保存至: {detail.get('chart_path', '')}")

def export_to_csv(results, output_file):
    """导出分析结果到CSV文件"""
    if not results:
        print("没有结果可导出")
        return False
        
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"结果已导出到: {output_file}")
        return True
    except Exception as e:
        print(f"导出失败: {str(e)}")
        return False

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 根据命令选择操作
    if not args.command:
        print("请指定命令。使用 --help 查看帮助。")
        return
    
    try:
        # 确定使用哪个控制器
        is_enhanced = args.command == 'enhanced'
        if hasattr(args, 'enhanced'):
            is_enhanced = is_enhanced or args.enhanced
            
        controller = EnhancedGuiController() if is_enhanced else GuiController()
        
        # 执行命令
        if args.command in ['standard', 'enhanced']:
            # 获取分析结果
            if args.command == 'standard':
                results = controller.get_momentum_analysis(
                    industry=args.industry, sample_size=args.sample, min_score=args.score
                )
            else:
                results = controller.get_enhanced_momentum_analysis(
                    industry=args.industry, sample_size=args.sample, min_score=args.score
                )
            
            # 打印结果
            print(f"{'增强版 ' if args.command == 'enhanced' else ''}动量分析结果:")
            print_results(results, detailed=True)
            
        elif args.command == 'market':
            # 获取市场概览
            if is_enhanced:
                market_data = controller.get_enhanced_market_overview()
            else:
                market_data = controller.get_market_overview()
                
            # 打印市场概览
            print("市场概览:")
            print_market_overview(market_data)
            
        elif args.command == 'industry':
            # 获取行业分析
            if is_enhanced:
                industry_data = controller.get_enhanced_industry_analysis()
            else:
                industry_data = controller.get_industry_analysis()
                
            # 打印行业分析
            print("行业分析:")
            print_industry_analysis(industry_data)
            
        elif args.command == 'detail':
            # 获取股票详情
            if is_enhanced:
                detail = controller.get_enhanced_stock_detail(args.code)
            else:
                detail = controller.get_stock_detail(args.code)
                
            # 打印股票详情
            print_stock_detail(detail)
            
        elif args.command == 'export':
            # 获取分析结果
            if args.enhanced:
                results = controller.get_enhanced_momentum_analysis(
                    industry=args.industry, sample_size=args.sample, min_score=args.score
                )
            else:
                results = controller.get_momentum_analysis(
                    industry=args.industry, sample_size=args.sample, min_score=args.score
                )
                
            # 导出结果
            export_to_csv(results, args.output)
    
    except Exception as e:
        logger.error(f"执行命令时出错: {str(e)}")
        print(f"错误: {str(e)}")
        
if __name__ == "__main__":
    main() 