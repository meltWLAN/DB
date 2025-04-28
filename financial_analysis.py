#!/usr/bin/env python3
"""
财务分析模块
提供股票财务指标的获取和分析功能
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from pathlib import Path
import tushare as ts
import warnings
warnings.filterwarnings('ignore')

# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 导入项目配置
try:
    from src.enhanced.config.settings import TUSHARE_TOKEN, LOG_DIR, DATA_DIR, RESULTS_DIR
except ImportError:
    # 设置默认配置
    TUSHARE_TOKEN = ""
    LOG_DIR = "./logs"
    DATA_DIR = "./data"
    RESULTS_DIR = "./results"

# 确保目录存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "finance_charts"), exist_ok=True)

# 配置日志
logger = logging.getLogger(__name__)

# 设置Tushare
if not TUSHARE_TOKEN:
    # 直接在代码中设置Token（如果配置文件中没有设置）
    TUSHARE_TOKEN = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
    # 直接创建API实例，不依赖token文件
    try:
        pro = ts.pro_api(TUSHARE_TOKEN)
        logger.info(f"成功初始化Tushare API (Token前5位: {TUSHARE_TOKEN[:5]}...)")
    except Exception as e:
        logger.error(f"初始化Tushare API失败: {str(e)}")
        pro = None
else:
    pro = None

class FinancialAnalyzer:
    """财务分析器类，提供财务指标分析相关功能"""
    
    def __init__(self, use_tushare=True):
        """初始化财务分析器"""
        self.use_tushare = use_tushare
        if use_tushare and not TUSHARE_TOKEN:
            logger.warning("未设置Tushare Token，将使用本地数据")
            self.use_tushare = False
        self.data_cache = {}  # 数据缓存
        
    def get_financial_indicator(self, ts_code, start_date=None, end_date=None, period=None):
        """
        获取股票财务指标数据
        
        参数:
            ts_code: TS股票代码
            start_date: 开始日期，默认为近3年
            end_date: 结束日期，默认为当前日期
            period: 报告期，例如'20211231'表示2021年年报
            
        返回:
            pandas DataFrame包含财务指标数据
        """
        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d')
            
        # 缓存键
        cache_key = f"fina_indicator_{ts_code}_{start_date}_{end_date}_{period}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
            
        if self.use_tushare:
            try:
                # 从Tushare获取财务指标数据
                params = {'ts_code': ts_code}
                if start_date:
                    params['start_date'] = start_date
                if end_date:
                    params['end_date'] = end_date
                if period:
                    params['period'] = period
                    
                df = pro.fina_indicator(**params)
                
                if not df.empty:
                    # 缓存数据
                    self.data_cache[cache_key] = df
                    return df
                else:
                    logger.warning(f"获取{ts_code}的财务指标数据为空")
                    return self._get_mock_financial_data(ts_code, start_date, end_date)
            except Exception as e:
                logger.error(f"从Tushare获取{ts_code}的财务指标数据失败: {str(e)}")
                return self._get_mock_financial_data(ts_code, start_date, end_date)
        else:
            return self._get_mock_financial_data(ts_code, start_date, end_date)
            
    def _get_mock_financial_data(self, ts_code, start_date=None, end_date=None):
        """生成模拟财务数据（当无法从API获取时使用）"""
        try:
            # 尝试从本地文件读取
            finance_file = os.path.join(DATA_DIR, f"{ts_code.replace('.', '_')}_finance.csv")
            if os.path.exists(finance_file):
                df = pd.read_csv(finance_file)
                # 处理日期
                if 'end_date' in df.columns:
                    df['end_date'] = pd.to_datetime(df['end_date'])
                    # 日期筛选
                    if start_date:
                        start_date = pd.to_datetime(start_date)
                        df = df[df['end_date'] >= start_date]
                    if end_date:
                        end_date = pd.to_datetime(end_date)
                        df = df[df['end_date'] <= end_date]
                return df
            else:
                # 创建模拟数据
                logger.warning(f"无法获取{ts_code}的财务数据，创建模拟数据")
                # 生成报告期序列
                periods = [
                    '20210331', '20210630', '20210930', '20211231',
                    '20220331', '20220630', '20220930', '20221231',
                    '20230331', '20230630', '20230930', '20231231'
                ]
                
                # 生成基本财务数据
                np.random.seed(sum(ord(c) for c in ts_code))  # 根据股票代码生成确定性随机数
                n = len(periods)
                base_eps = np.random.uniform(0.2, 2.0)
                growth_factor = np.random.uniform(1.05, 1.15)
                
                data = {
                    'ts_code': [ts_code] * n,
                    'ann_date': [(datetime.strptime(p, '%Y%m%d') + timedelta(days=np.random.randint(20, 30))).strftime('%Y%m%d') for p in periods],
                    'end_date': periods,
                    'eps': [base_eps * (growth_factor ** (i % 4)) for i in range(n)],
                    'dt_eps': [base_eps * (growth_factor ** (i % 4)) for i in range(n)],
                    'total_revenue_ps': [base_eps * 10 * (growth_factor ** (i % 4)) for i in range(n)],
                    'revenue_ps': [base_eps * 9.8 * (growth_factor ** (i % 4)) for i in range(n)],
                    'bps': [np.random.uniform(5, 20) * (1.02 ** i) for i in range(n)],
                    'roe': [np.random.uniform(5, 15) for _ in range(n)],
                    'profit_dedt': [np.random.uniform(5000, 50000) * (growth_factor ** (i % 4)) for i in range(n)],
                    'nprofitdedt_yoy': [np.random.uniform(-5, 30) for _ in range(n)],
                    'grossprofit_margin': [np.random.uniform(20, 50) for _ in range(n)],
                    'debt_to_assets': [np.random.uniform(20, 60) for _ in range(n)]
                }
                
                df = pd.DataFrame(data)
                return df
        except Exception as e:
            logger.error(f"生成模拟财务数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_disclosure_date(self, ts_code=None, end_date=None):
        """获取财报披露计划日期"""
        if self.use_tushare:
            try:
                params = {}
                if ts_code:
                    params['ts_code'] = ts_code
                if end_date:
                    params['end_date'] = end_date
                    
                df = pro.disclosure_date(**params)
                return df
            except Exception as e:
                logger.error(f"从Tushare获取财报披露日期失败: {str(e)}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def get_stk_holders(self, ts_code, start_date=None, end_date=None):
        """获取股东人数数据"""
        if self.use_tushare:
            try:
                params = {'ts_code': ts_code}
                if start_date:
                    params['start_date'] = start_date
                if end_date:
                    params['end_date'] = end_date
                    
                df = pro.stk_holdernumber(**params)
                return df
            except Exception as e:
                logger.error(f"从Tushare获取股东人数数据失败: {str(e)}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def get_stk_holdertrade(self, ts_code=None, start_date=None, end_date=None, trade_type=None):
        """获取股东增减持数据"""
        if self.use_tushare:
            try:
                params = {}
                if ts_code:
                    params['ts_code'] = ts_code
                if start_date:
                    params['start_date'] = start_date
                if end_date:
                    params['end_date'] = end_date
                if trade_type:
                    params['trade_type'] = trade_type
                    
                df = pro.stk_holdertrade(**params)
                return df
            except Exception as e:
                logger.error(f"从Tushare获取股东增减持数据失败: {str(e)}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def get_institutional_survey(self, ts_code, start_date=None, end_date=None):
        """获取机构调研数据"""
        if self.use_tushare:
            try:
                params = {'ts_code': ts_code}
                if start_date:
                    params['start_date'] = start_date
                if end_date:
                    params['end_date'] = end_date
                    
                df = pro.stk_surv(**params)
                return df
            except Exception as e:
                logger.error(f"从Tushare获取机构调研数据失败: {str(e)}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def get_broker_recommend(self, month):
        """获取券商月度金股推荐"""
        if self.use_tushare:
            try:
                df = pro.broker_recommend(month=month)
                return df
            except Exception as e:
                logger.error(f"从Tushare获取券商推荐数据失败: {str(e)}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def calculate_financial_score(self, financial_data):
        """
        根据财务指标计算财务健康得分
        
        参数:
            financial_data: 财务指标数据DataFrame
            
        返回:
            得分和详细评分项目
        """
        if financial_data.empty:
            return 0, {}
        
        # 获取最新一期财务数据
        latest = financial_data.iloc[0]
        
        score = 0
        score_details = {}
        
        # 1. EPS评分（20分）
        try:
            eps = latest.get('eps', 0)
            if eps > 1:
                eps_score = 20
            elif eps > 0.5:
                eps_score = 15
            elif eps > 0.2:
                eps_score = 10
            elif eps > 0:
                eps_score = 5
            else:
                eps_score = 0
            score += eps_score
            score_details['eps_score'] = eps_score
        except:
            score_details['eps_score'] = 0
        
        # 2. ROE评分（20分）
        try:
            roe = latest.get('roe', 0)
            if roe > 15:
                roe_score = 20
            elif roe > 10:
                roe_score = 15
            elif roe > 5:
                roe_score = 10
            elif roe > 0:
                roe_score = 5
            else:
                roe_score = 0
            score += roe_score
            score_details['roe_score'] = roe_score
        except:
            score_details['roe_score'] = 0
        
        # 3. 毛利率评分（20分）
        try:
            gp_margin = latest.get('grossprofit_margin', 0)
            if gp_margin > 40:
                gp_score = 20
            elif gp_margin > 30:
                gp_score = 15
            elif gp_margin > 20:
                gp_score = 10
            elif gp_margin > 10:
                gp_score = 5
            else:
                gp_score = 0
            score += gp_score
            score_details['gp_margin_score'] = gp_score
        except:
            score_details['gp_margin_score'] = 0
        
        # 4. 资产负债率评分（20分）
        try:
            debt_ratio = latest.get('debt_to_assets', 0)
            if debt_ratio < 30:
                debt_score = 20
            elif debt_ratio < 40:
                debt_score = 15
            elif debt_ratio < 60:
                debt_score = 10
            elif debt_ratio < 80:
                debt_score = 5
            else:
                debt_score = 0
            score += debt_score
            score_details['debt_ratio_score'] = debt_score
        except:
            score_details['debt_ratio_score'] = 0
        
        # 5. 业绩增长评分（20分）
        try:
            growth = latest.get('nprofitdedt_yoy', 0)
            if growth > 30:
                growth_score = 20
            elif growth > 20:
                growth_score = 15
            elif growth > 10:
                growth_score = 10
            elif growth > 0:
                growth_score = 5
            else:
                growth_score = 0
            score += growth_score
            score_details['growth_score'] = growth_score
        except:
            score_details['growth_score'] = 0
        
        return score, score_details
    
    def plot_financial_trends(self, financial_data, ts_code, save_path=None):
        """
        绘制财务指标趋势图
        
        参数:
            financial_data: 财务指标数据DataFrame
            ts_code: 股票代码
            save_path: 保存路径
            
        返回:
            图表保存路径
        """
        if financial_data.empty:
            return None
        
        # 确保数据按时间排序
        if 'end_date' in financial_data.columns:
            financial_data['end_date'] = pd.to_datetime(financial_data['end_date'])
            financial_data = financial_data.sort_values('end_date')
        
        # 创建图表
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{ts_code} 财务指标趋势分析", fontsize=16)
        
        # 1. EPS趋势
        if 'eps' in financial_data.columns:
            axs[0, 0].plot(financial_data['end_date'], financial_data['eps'], 'o-', color='blue')
            axs[0, 0].set_title('每股收益(EPS)趋势')
            axs[0, 0].set_ylabel('每股收益(元)')
            axs[0, 0].grid(True)
        
        # 2. ROE趋势
        if 'roe' in financial_data.columns:
            axs[0, 1].plot(financial_data['end_date'], financial_data['roe'], 'o-', color='green')
            axs[0, 1].set_title('净资产收益率(ROE)趋势')
            axs[0, 1].set_ylabel('净资产收益率(%)')
            axs[0, 1].grid(True)
        
        # 3. 毛利率趋势
        if 'grossprofit_margin' in financial_data.columns:
            axs[1, 0].plot(financial_data['end_date'], financial_data['grossprofit_margin'], 'o-', color='red')
            axs[1, 0].set_title('毛利率趋势')
            axs[1, 0].set_ylabel('毛利率(%)')
            axs[1, 0].grid(True)
        
        # 4. 资产负债率趋势
        if 'debt_to_assets' in financial_data.columns:
            axs[1, 1].plot(financial_data['end_date'], financial_data['debt_to_assets'], 'o-', color='purple')
            axs[1, 1].set_title('资产负债率趋势')
            axs[1, 1].set_ylabel('资产负债率(%)')
            axs[1, 1].grid(True)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(RESULTS_DIR, "finance_charts", f"{ts_code.replace('.', '_')}_finance.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def analyze_financial_stocks(self, stock_list, sample_size=100, min_score=60):
        """
        分析股票财务状况并返回评分结果
        
        参数:
            stock_list: 股票列表DataFrame
            sample_size: 分析的股票数量
            min_score: 最低得分阈值
            
        返回:
            包含财务评分的结果列表
        """
        results = []
        
        # 限制样本大小
        if len(stock_list) > sample_size:
            stock_list = stock_list.sample(sample_size)
        
        for _, stock in stock_list.iterrows():
            ts_code = stock['ts_code']
            stock_name = stock.get('name', '')
            industry = stock.get('industry', '')
            
            try:
                # 获取财务数据
                financial_data = self.get_financial_indicator(ts_code)
                
                if not financial_data.empty:
                    # 计算财务评分
                    score, score_details = self.calculate_financial_score(financial_data)
                    
                    # 如果得分大于阈值，添加到结果中
                    if score >= min_score:
                        # 获取最新一期财务数据
                        latest = financial_data.iloc[0]
                        
                        # 绘制财务趋势图
                        chart_path = self.plot_financial_trends(financial_data, ts_code)
                        
                        # 构建结果
                        result = {
                            'ts_code': ts_code,
                            'name': stock_name,
                            'industry': industry,
                            'eps': latest.get('eps', 0),
                            'roe': latest.get('roe', 0),
                            'bps': latest.get('bps', 0),
                            'grossprofit_margin': latest.get('grossprofit_margin', 0),
                            'debt_to_assets': latest.get('debt_to_assets', 0),
                            'latest_report': latest.get('end_date', ''),
                            'score': score,
                            'score_details': score_details,
                            'chart_path': chart_path
                        }
                        
                        results.append(result)
                        
                        logger.info(f"分析 {ts_code} {stock_name} 完成，得分: {score}")
                    else:
                        logger.info(f"分析 {ts_code} {stock_name} 完成，得分: {score}，低于阈值 {min_score}")
                else:
                    logger.warning(f"无法获取 {ts_code} {stock_name} 的财务数据")
            except Exception as e:
                logger.error(f"分析 {ts_code} {stock_name} 时出错: {str(e)}")
        
        # 按得分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def combine_financial_technical(self, financial_results, technical_results, financial_weight=0.5):
        """
        综合财务分析和技术分析结果
        
        参数:
            financial_results: 财务分析结果列表
            technical_results: 技术分析结果列表
            financial_weight: 财务分析权重(0-1)
            
        返回:
            综合分析结果列表
        """
        combined_results = []
        
        # 创建查找表以便快速访问结果
        financial_dict = {item['ts_code']: item for item in financial_results}
        technical_dict = {item['ts_code']: item for item in technical_results}
        
        # 找出两个结果集中共有的股票代码
        common_stocks = set(financial_dict.keys()) & set(technical_dict.keys())
        
        for ts_code in common_stocks:
            financial = financial_dict[ts_code]
            technical = technical_dict[ts_code]
            
            # 计算综合得分
            combined_score = financial['score'] * financial_weight + technical['score'] * (1 - financial_weight)
            
            # 创建结合结果
            result = {
                'ts_code': ts_code,
                'name': financial.get('name', technical.get('name', '')),
                'industry': financial.get('industry', technical.get('industry', '')),
                'financial_score': financial['score'],
                'technical_score': technical['score'],
                'combined_score': combined_score,
                'eps': financial.get('eps', 0),
                'roe': financial.get('roe', 0),
                'momentum': technical.get('20日动量', 0),
                'rsi': technical.get('RSI', 0),
                'financial_chart': financial.get('chart_path', ''),
                'technical_chart': technical.get('chart_path', '')
            }
            
            combined_results.append(result)
        
        # 按综合得分排序
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return combined_results
    
    def get_chips_data(self, ts_code, trade_date=None):
        """获取筹码分布数据"""
        if self.use_tushare:
            try:
                params = {'ts_code': ts_code}
                if trade_date:
                    params['trade_date'] = trade_date
                else:
                    params['start_date'] = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                    params['end_date'] = datetime.now().strftime('%Y%m%d')
                    
                df = pro.cyq_chips(**params)
                return df
            except Exception as e:
                logger.error(f"从Tushare获取筹码分布数据失败: {str(e)}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def get_chips_perf(self, ts_code, trade_date=None):
        """获取筹码与胜率数据"""
        if self.use_tushare:
            try:
                params = {'ts_code': ts_code}
                if trade_date:
                    params['trade_date'] = trade_date
                else:
                    params['start_date'] = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                    params['end_date'] = datetime.now().strftime('%Y%m%d')
                    
                df = pro.cyq_perf(**params)
                return df
            except Exception as e:
                logger.error(f"从Tushare获取筹码胜率数据失败: {str(e)}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def analyze_chip_distribution(self, ts_code, stock_name, chart_path=None):
        """分析筹码分布并生成可视化图表"""
        chips_data = self.get_chips_data(ts_code)
        chips_perf = self.get_chips_perf(ts_code)
        
        if chips_data.empty or chips_perf.empty:
            logger.warning(f"无法获取 {ts_code} 的筹码分布数据")
            return None
        
        # 获取最新日期的数据
        latest_date = chips_perf['trade_date'].max()
        latest_perf = chips_perf[chips_perf['trade_date'] == latest_date].iloc[0]
        latest_chips = chips_data[chips_data['trade_date'] == latest_date]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f"{ts_code} {stock_name} 筹码分布分析", fontsize=16)
        
        # 1. 筹码分布图
        ax1.bar(latest_chips['price'], latest_chips['percent'], width=0.2, alpha=0.7)
        ax1.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='重要筹码密集线(2%)')
        ax1.axhline(y=5.0, color='orange', linestyle='--', alpha=0.7, label='极重要筹码密集线(5%)')
        ax1.set_title('筹码分布')
        ax1.set_xlabel('价格')
        ax1.set_ylabel('持仓比例(%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加关键筹码区域标注
        key_chips = latest_chips[latest_chips['percent'] > 2.0]
        if not key_chips.empty:
            max_chip = key_chips.loc[key_chips['percent'].idxmax()]
            ax1.annotate(f"主力筹码: {max_chip['price']:.2f}元 ({max_chip['percent']:.2f}%)",
                         xy=(max_chip['price'], max_chip['percent']),
                         xytext=(max_chip['price'], max_chip['percent'] + 1),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                         fontsize=10, ha='center')
        
        # 2. 筹码胜率和成本分布
        ax2.plot(chips_perf['trade_date'], chips_perf['weight_avg'], 'b-', label='平均成本')
        ax2.plot(chips_perf['trade_date'], chips_perf['cost_5pct'], 'g--', alpha=0.7, label='5%成本')
        ax2.plot(chips_perf['trade_date'], chips_perf['cost_95pct'], 'r--', alpha=0.7, label='95%成本')
        ax2.set_title('平均成本与胜率分布')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('价格', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.grid(True, alpha=0.3)
        
        # 添加胜率的次坐标轴
        ax3 = ax2.twinx()
        ax3.plot(chips_perf['trade_date'], chips_perf['winner_rate'], 'purple', label='胜率')
        ax3.set_ylabel('胜率(%)', color='purple')
        ax3.tick_params(axis='y', labelcolor='purple')
        
        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 添加关键信息标注
        latest_date_str = pd.to_datetime(latest_date).strftime('%Y-%m-%d')
        info_text = (
            f"日期: {latest_date_str}\n"
            f"平均成本: {latest_perf['weight_avg']:.2f}\n"
            f"胜率: {latest_perf['winner_rate']:.2f}%\n"
            f"低档成本(5%): {latest_perf['cost_5pct']:.2f}\n"
            f"高档成本(95%): {latest_perf['cost_95pct']:.2f}"
        )
        ax2.annotate(info_text, xy=(0.02, 0.05), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8))
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图表
        if chart_path is None:
            chart_path = os.path.join(RESULTS_DIR, "finance_charts", f"{ts_code.replace('.', '_')}_chips.png")
        
        plt.savefig(chart_path)
        plt.close()
        
        return {
            'chart_path': chart_path,
            'avg_cost': latest_perf['weight_avg'],
            'winner_rate': latest_perf['winner_rate'],
            'low_cost': latest_perf['cost_5pct'],
            'high_cost': latest_perf['cost_95pct'],
            'his_low': latest_perf['his_low'],
            'his_high': latest_perf['his_high']
        }

# 测试函数
def test():
    analyzer = FinancialAnalyzer()
    
    # 测试获取财务指标
    financial_data = analyzer.get_financial_indicator('000001.SZ')
    print(financial_data.head())
    
    # 测试财务评分
    score, details = analyzer.calculate_financial_score(financial_data)
    print(f"财务评分: {score}, 详情: {details}")
    
    # 测试绘制趋势图
    chart_path = analyzer.plot_financial_trends(financial_data, '000001.SZ')
    print(f"图表保存在: {chart_path}")

if __name__ == "__main__":
    test() 