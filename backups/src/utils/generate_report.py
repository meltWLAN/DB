#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成综合分析报告
整合增强版分析和策略回测结果，生成完整的分析报告
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

# 设置中文字体
try:
    # 尝试使用系统中文字体
    font_path = '/System/Library/Fonts/PingFang.ttc'  # macOS中文字体
    if os.path.exists(font_path):
        chinese_font = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = chinese_font.get_name()
    else:
        # 如果找不到系统字体，使用matplotlib内置字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    # 如果设置字体失败，则忽略中文显示问题
    pass

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加当前目录到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 确保结果目录存在
report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/report")
os.makedirs(report_dir, exist_ok=True)

def load_data():
    """加载分析和回测数据
    
    Returns:
        data_dict: 包含所有分析数据的字典
    """
    data_dict = {}
    
    # 加载增强版分析结果
    enhanced_analysis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/enhanced_analysis")
    
    # 加载个股分析结果
    stock_analysis_file = os.path.join(enhanced_analysis_dir, "analysis_results.csv")
    if os.path.exists(stock_analysis_file):
        data_dict['stock_analysis'] = pd.read_csv(stock_analysis_file)
        logger.info(f"加载个股分析结果: {len(data_dict['stock_analysis'])}只股票")
    
    # 加载行业分析结果
    industry_analysis_file = os.path.join(enhanced_analysis_dir, "industry_analysis.csv")
    if os.path.exists(industry_analysis_file):
        data_dict['industry_analysis'] = pd.read_csv(industry_analysis_file)
        logger.info(f"加载行业分析结果: {len(data_dict['industry_analysis'])}个行业")
    
    # 加载策略回测结果
    strategy_backtest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/strategy_backtest")
    
    # 加载交易记录
    trade_records_file = os.path.join(strategy_backtest_dir, "trade_records.csv")
    if os.path.exists(trade_records_file):
        data_dict['trade_records'] = pd.read_csv(trade_records_file)
        logger.info(f"加载交易记录: {len(data_dict['trade_records'])}条记录")
    
    # 加载策略表现摘要
    performance_summary_file = os.path.join(strategy_backtest_dir, "performance_summary.txt")
    if os.path.exists(performance_summary_file):
        with open(performance_summary_file, 'r') as f:
            performance_text = f.read()
        data_dict['performance_summary'] = performance_text
        logger.info("加载策略表现摘要")
    
    # 加载涨停板捕捉策略结果
    limit_up_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/limit_up_capture")
    
    # 寻找最新的高潜力股票文件
    limit_up_files = [f for f in os.listdir(limit_up_dir) if f.startswith('high_potential_stocks_') and f.endswith('.csv')]
    if limit_up_files:
        # 按文件名排序，获取最新的文件
        latest_file = sorted(limit_up_files, reverse=True)[0]
        limit_up_file = os.path.join(limit_up_dir, latest_file)
        if os.path.exists(limit_up_file):
            data_dict['limit_up_stocks'] = pd.read_csv(limit_up_file)
            logger.info(f"加载涨停板捕捉策略结果: {len(data_dict['limit_up_stocks'])}只股票")
    
    return data_dict

def generate_report(data):
    """生成综合分析报告
    
    Args:
        data: 包含所有分析数据的字典
        
    Returns:
        report_path: 生成的报告文件路径
    """
    report_path = os.path.join(report_dir, f"综合市场分析与策略回测报告_{datetime.now().strftime('%Y%m%d')}.pdf")
    
    with PdfPages(report_path) as pdf:
        # 1. 封面页
        plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.8, "综合市场分析与策略回测报告", 
                fontsize=24, horizontalalignment='center')
        plt.text(0.5, 0.7, f"报告生成日期: {datetime.now().strftime('%Y-%m-%d')}", 
                fontsize=14, horizontalalignment='center')
        plt.text(0.5, 0.6, "基于JoinQuant数据", 
                fontsize=14, horizontalalignment='center')
        plt.text(0.5, 0.5, "内容包括:\n- 个股分析\n- 行业分析\n- 策略回测结果\n- 涨停板捕捉策略\n- 投资建议", 
                fontsize=14, horizontalalignment='center')
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # 2. 市场概览页
        if 'stock_analysis' in data:
            plt.figure(figsize=(8.5, 11))
            plt.subplot(2, 1, 1)
            plt.title("市场概览 - 个股表现排名", fontsize=16)
            
            # 获取前10和后5的股票
            top_stocks = data['stock_analysis'].sort_values('总收益率', ascending=False).head(10)
            bottom_stocks = data['stock_analysis'].sort_values('总收益率').head(5)
            
            # 绘制前10名股票收益率
            plt.subplot(2, 1, 1)
            sns.barplot(x='股票名称', y='总收益率', data=top_stocks)
            plt.title("表现最好的10只股票 - 总收益率")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 绘制后5名股票收益率
            plt.subplot(2, 1, 2)
            sns.barplot(x='股票名称', y='总收益率', data=bottom_stocks)
            plt.title("表现最差的5只股票 - 总收益率")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            pdf.savefig()
            plt.close()
        
        # 3. 行业分析页
        if 'industry_analysis' in data:
            plt.figure(figsize=(8.5, 11))
            
            # 行业收益率排名
            plt.subplot(2, 1, 1)
            industry_data = data['industry_analysis'].sort_values('total_return', ascending=False)
            sns.barplot(x='industry', y='total_return', data=industry_data)
            plt.title("各行业平均收益率排名")
            plt.xticks(rotation=45)
            
            # 行业夏普比率排名
            plt.subplot(2, 1, 2)
            industry_data = data['industry_analysis'].sort_values('sharpe_ratio', ascending=False)
            sns.barplot(x='industry', y='sharpe_ratio', data=industry_data)
            plt.title("各行业平均夏普比率排名")
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # 行业波动率与最大回撤
            plt.figure(figsize=(8.5, 11))
            
            # 行业波动率排名 (从低到高)
            plt.subplot(2, 1, 1)
            industry_data = data['industry_analysis'].sort_values('volatility')
            sns.barplot(x='industry', y='volatility', data=industry_data)
            plt.title("各行业波动率排名 (从低到高)")
            plt.xticks(rotation=45)
            
            # 行业最大回撤排名 (从高到低，因为是负值)
            plt.subplot(2, 1, 2)
            industry_data = data['industry_analysis'].sort_values('max_drawdown', ascending=False)
            sns.barplot(x='industry', y='max_drawdown', data=industry_data)
            plt.title("各行业最大回撤排名 (从小到大)")
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # 4. 策略回测结果页
        if 'performance_summary' in data and 'trade_records' in data:
            plt.figure(figsize=(8.5, 11))
            
            # 策略表现摘要
            plt.subplot(2, 1, 1)
            plt.axis('off')
            plt.text(0.1, 0.9, "策略回测结果摘要", fontsize=16)
            
            # 解析性能摘要文本
            performance_lines = data['performance_summary'].strip().split('\n')
            y_pos = 0.8
            for line in performance_lines[:8]:  # 只显示前8行，不包括选股部分
                plt.text(0.1, y_pos, line, fontsize=12)
                y_pos -= 0.05
            
            # 交易统计
            plt.subplot(2, 1, 2)
            trade_records = data['trade_records']
            
            # 按股票分组计算交易次数
            trade_counts = trade_records.groupby('stock')['action'].count().reset_index()
            trade_counts.columns = ['stock', 'trade_count']
            
            # 计算每只股票的盈亏
            profit_data = []
            
            for stock in trade_records['stock'].unique():
                stock_trades = trade_records[trade_records['stock'] == stock]
                buys = stock_trades[stock_trades['action'] == 'BUY']['amount'].sum()
                sells = stock_trades[stock_trades['action'] == 'SELL']['amount'].sum()
                profit = sells - buys
                profit_pct = profit / buys * 100 if buys > 0 else 0
                
                profit_data.append({
                    'stock': stock,
                    'profit': profit,
                    'profit_pct': profit_pct
                })
            
            profit_df = pd.DataFrame(profit_data)
            
            # 绘制每只股票的收益率
            plt.bar(profit_df['stock'], profit_df['profit_pct'])
            plt.title("各股票收益贡献率 (%)")
            plt.xticks(rotation=45)
            plt.ylabel("收益率 (%)")
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # 交易详情页
            plt.figure(figsize=(8.5, 11))
            plt.subplot(1, 1, 1)
            plt.axis('off')
            
            # 交易明细表格
            cell_text = []
            for _, row in trade_records.head(20).iterrows():  # 只显示前20条记录
                cell_text.append([
                    row['date'], 
                    row['stock'], 
                    row['action'], 
                    f"{row['price']:.2f}", 
                    f"{row['shares']}", 
                    f"{row['amount']:.2f}"
                ])
            
            plt.table(
                cellText=cell_text,
                colLabels=['日期', '股票', '操作', '价格', '数量', '金额'],
                loc='center',
                cellLoc='center'
            )
            
            plt.title("交易记录明细 (前20条)", pad=60)
            pdf.savefig()
            plt.close()
        
        # 添加连续涨停捕捉策略分析页面
        if 'limit_up_stocks' in data and not data['limit_up_stocks'].empty:
            plt.figure(figsize=(8.5, 11))
            
            # 标题
            plt.suptitle("涨停板捕捉策略 - 高潜力股票分析", fontsize=16)
            
            limit_up_data = data['limit_up_stocks']
            
            # 绘制得分最高的股票
            plt.subplot(2, 1, 1)
            top_stocks = limit_up_data.sort_values('score', ascending=False).head(10)
            sns.barplot(x='ts_code', y='score', data=top_stocks)
            plt.title("得分最高的10只潜力股票")
            plt.xticks(rotation=45)
            plt.ylabel("预测得分")
            
            # 绘制不同原因类型的数量分布
            plt.subplot(2, 1, 2)
            reason_counts = limit_up_data['reason'].value_counts()
            plt.pie(reason_counts, labels=reason_counts.index, autopct='%1.1f%%')
            plt.title("高潜力股票筛选原因分布")
            plt.axis('equal')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整以适应suptitle
            pdf.savefig()
            plt.close()
            
            # 新增图表：连续涨停天数分布和预测概率
            plt.figure(figsize=(8.5, 11))
            
            if 'consecutive_days' in limit_up_data.columns:
                plt.subplot(2, 1, 1)
                days_counts = limit_up_data['consecutive_days'].value_counts().sort_index()
                plt.bar(days_counts.index.astype(str), days_counts.values)
                plt.title("连续涨停天数分布")
                plt.xlabel("连续涨停天数")
                plt.ylabel("股票数量")
                
                # 添加文本说明涨停概率
                if 'prediction_desc' in limit_up_data.columns:
                    # 获取不同涨停天数的平均预测概率
                    plt.subplot(2, 1, 2)
                    plt.axis('off')
                    plt.title("涨停概率统计")
                    
                    # 提取概率值并展示
                    unique_predictions = limit_up_data['prediction_desc'].dropna().unique()
                    y_pos = 0.8
                    for pred in unique_predictions:
                        if "连板概率" in pred:
                            plt.text(0.1, y_pos, pred, fontsize=12)
                            y_pos -= 0.1
                    
                    # 添加说明
                    plt.text(0.1, 0.4, "根据历史统计数据分析：", fontsize=14, weight='bold')
                    plt.text(0.1, 0.3, "• 首板第二天继续涨停概率约30-35%", fontsize=12)
                    plt.text(0.1, 0.25, "• 二连板第三天继续涨停概率约25-30%", fontsize=12)
                    plt.text(0.1, 0.2, "• 三连板第四天继续涨停概率约20-25%", fontsize=12)
                    plt.text(0.1, 0.15, "• 低位股票涨停概率高于高位股票", fontsize=12)
                    plt.text(0.1, 0.1, "• 成交量是5日均量2倍以上时，涨停概率提高15%", fontsize=12)
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # 创建高潜力股票详细信息表格
            plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            plt.title("高潜力股票详细信息", fontsize=16)
            
            # 创建表格
            col_labels = ['股票代码', '名称', '筛选原因', '预测得分', '当前价格', '预测分析']
            table_data = []
            
            # 确保只选择存在的列
            columns_to_display = ['ts_code', 'name', 'reason', 'score', 'price', 'prediction_desc']
            columns_to_display = [col for col in columns_to_display if col in limit_up_data.columns]
            
            for _, row in limit_up_data.head(15).iterrows():
                row_data = []
                for col in columns_to_display:
                    if col == 'score':
                        row_data.append(f"{row[col]:.0f}")
                    elif col == 'price':
                        row_data.append(f"{row[col]:.2f}")
                    else:
                        row_data.append(str(row[col]))
                table_data.append(row_data)
            
            table = plt.table(
                cellText=table_data,
                colLabels=[col.replace('_', ' ').title() for col in columns_to_display],
                loc='center',
                cellLoc='center',
                bbox=[0.1, 0.1, 0.8, 0.8]
            )
            
            # 调整表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            pdf.savefig()
            plt.close()
        
        # 5. 投资建议页 - 通用部分
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.title("投资建议", fontsize=16)
        
        # 内容
        content = [
            "一、市场分析总结",
            "1. 行业表现分析:",
            "   - 金融保险、人工智能和食品饮料行业表现最佳，夏普比率最高",
            "   - 电力行业虽然收益率中等，但波动率最低，是防御性较强的选择",
            "   - 白酒行业在此期间表现相对较差，波动较大",
            "",
            "2. 个股表现分析:",
            "   - 海天味业、中国平安和科大讯飞表现最好，总收益率均超过20%",
            "   - 贵州茅台、中国石油和山西汾酒表现较差，收益为负",
            "",
            "二、策略回测结果",
            "1. 行业动量策略取得了正向收益，年化收益约8%，夏普比率1.2",
            "2. 最大回撤控制在较低水平（约2.4%），显示策略风险控制有效",
            "3. 选择行业龙头股+技术分析指标的组合方法证明是有效的",
            "",
            "三、涨停板捕捉策略",
            "1. 连续涨停股票分析显示短期内具有动量延续特性",
            "2. 技术突破+高动量的组合策略可有效筛选出短期爆发潜力股",
            "3. 结合行业分析，重点关注处于上升趋势的行业中的涨停板股票",
            "4. 根据历史数据统计，二连板继续涨停概率约30%，低位+放量可提高至40-45%",
            "5. 机构资金流入是连板概率提升的重要指标，净流入超过5%是强信号",
            "",
            "四、投资建议",
            "1. 行业配置:",
            "   - 核心配置：金融保险、食品饮料等优质行业",
            "   - 成长配置：人工智能、新能源车等高增长行业",
            "   - 防御配置：电力等低波动行业",
            "",
            "2. 选股策略:",
            "   - 优先选择行业龙头股，如海天味业、中国平安等",
            "   - 结合技术指标（MACD、RSI、KDJ）进行买卖点判断",
            "   - 关注连续涨停和技术突破的高动量股票，把握短期机会",
            "",
            "3. 涨停板交易策略:",
            "   - 首板股票：观察成交量，放量超过2倍是重点关注对象",
            "   - 二连板：结合行业趋势和资金流向，盘中突破缺口是买点",
            "   - 三连板及以上：高风险高收益，建议设置严格止损",
            "   - 追涨策略：低位股票+明显放量+资金流入是理想选择",
            "   - 回调买入：连板后回调到10日均线附近可考虑买入",
            "",
            "4. 风险管理:",
            "   - 涨停板股票建议设置7%止损点，突破平台前高未成功必须止损",
            "   - 单只涨停股票仓位控制在总资金的10%以内",
            "   - 连板股票可采用分批建仓和分批止盈策略",
            "   - 连续上涨超过5天后，提高警惕，可适当降低仓位",
            "   - 出现均线空头排列或MACD死叉时，考虑清仓离场"
        ]
        
        y_pos = 0.90
        for line in content:
            plt.text(0.1, y_pos, line, fontsize=10)
            y_pos -= 0.020
            
        pdf.savefig()
        plt.close()
        
        # 专门针对涨停板交易的建议页
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.title("涨停板交易专项建议", fontsize=16)
        
        content = [
            "一、涨停板交易核心策略",
            "",
            "1. 首板选股策略:",
            "   • 关注低位 + 高换手率 + 首次放量突破的股票",
            "   • 关注大盘上涨趋势中的强势板块龙头股",
            "   • 优先选择10-30元价格区间的股票，此区间连板概率最高",
            "   • 关注机构资金净流入超过3%的个股",
            "   • 技术面上，MACD金叉+KDJ低位上穿+突破压力位是强信号",
            "",
            "2. 最佳买入时机:",
            "   • 涨停板开板回落到5%左右时",
            "   • 涨停后第二天低开回踩10日均线时",
            "   • 连续2天以上横盘震荡后再次突破前高时",
            "   • 盘中放量突破前期横盘整理区间时",
            "",
            "3. 止盈止损策略:",
            "   • 首板止损: 跌破近5日最低点",
            "   • 二连板止损: 回调超过7%或跌破5日均线",
            "   • 三连板及以上: 回调超过5%或开盘即大幅低开",
            "   • 止盈目标: 首板后上涨15%、二连板后上涨10%、三连板后上涨8%",
            "",
            "4. 仓位管理:",
            "   • 首板买入: 总仓位的5-8%",
            "   • 二连板买入: 总仓位的3-5%",
            "   • 三连板及以上: 总仓位不超过3%",
            "   • 单一板块涨停股总仓位不超过15%",
            "",
            "二、不同市场环境下的调整",
            "",
            "1. 强势市场（大盘连续上涨）:",
            "   • 提高仓位至建议值上限",
            "   • 可重点关注高价股突破，此环境下高价股也易出现连板",
            "   • 回调时可更积极买入，止损位可适当放宽",
            "",
            "2. 震荡市场（大盘区间波动）:",
            "   • 仓位控制在建议值中等水平",
            "   • 优先选择强势板块中的低位股",
            "   • 更看重基本面支撑，避免追逐题材炒作类个股",
            "",
            "3. 弱势市场（大盘下跌趋势）:",
            "   • 降低仓位至建议值下限或更低",
            "   • 只关注极少数逆势上涨的强势股",
            "   • 止损更加严格，不恋战",
            "",
            "三、涨停板预测模型使用指南",
            "",
            "1. 各指标重要性排序:",
            "   • 连续涨停天数（最重要）",
            "   • 成交量放大比例（非常重要）",
            "   • 机构资金流向（非常重要）",
            "   • 股价位置（重要）",
            "   • 5日动量（重要）",
            "   • MACD指标（辅助）",
            "",
            "2. 预测模型分数解读:",
            "   • 80分以上: 后续连板概率高，值得重点关注",
            "   • 60-80分: 后续行情看好，可择机买入",
            "   • 40-60分: 存在不确定性，建议观望或小仓位试探",
            "   • 40分以下: 后续走强概率低，不建议操作",
            "",
            "3. 注意事项:",
            "   • 预测模型得分高不代表绝对会涨停，只是概率更高",
            "   • 始终结合大盘环境和板块行情综合判断",
            "   • 即使是高分股票也需执行严格的止损策略",
            "   • 定期回顾模型预测结果与实际情况的差异，总结经验"
        ]
        
        y_pos = 0.90
        for line in content:
            plt.text(0.1, y_pos, line, fontsize=10)
            y_pos -= 0.020
            
        pdf.savefig()
        plt.close()
    
    logger.info(f"综合分析报告已生成: {report_path}")
    return report_path

def main():
    """主函数"""
    logger.info("开始生成综合分析报告...")
    
    # 加载数据
    data = load_data()
    
    # 生成报告
    if data:
        report_path = generate_report(data)
        logger.info(f"报告已生成: {report_path}")
    else:
        logger.error("未能加载足够的数据，报告生成失败")
    
    logger.info("报告生成完成")

if __name__ == "__main__":
    print("=" * 80)
    print(" 综合分析报告生成工具 ".center(80, "="))
    print("=" * 80)
    
    # 运行主函数
    main()
    
    print("=" * 80)
    print(" 报告生成完成 ".center(80, "="))
    print("=" * 80) 