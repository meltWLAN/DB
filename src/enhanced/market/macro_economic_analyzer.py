"""
宏观经济数据分析模块
提供宏观经济指标的获取、分析和可视化功能
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import requests
import json
from typing import Dict, List, Optional, Tuple, Union
import tushare as ts

# 配置日志
logger = logging.getLogger(__name__)

class MacroEconomicAnalyzer:
    """宏观经济分析器类，提供宏观经济指标的获取和分析功能"""
    
    def __init__(self, token: str = "", cache_dir: str = "./cache/macro"):
        """初始化宏观经济分析器
        
        Args:
            token: Tushare API Token
            cache_dir: 缓存目录
        """
        self.token = token
        if token:
            ts.set_token(token)
            self.pro = ts.pro_api()
        else:
            self.pro = None
            
        # 创建缓存目录
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 缓存数据
        self.cache = {}
        
    def get_gdp_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取GDP数据
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD，默认为近5年
            end_date: 结束日期，格式为YYYYMMDD，默认为今天
            
        Returns:
            pd.DataFrame: GDP数据
        """
        # 基本框架，实际实现略
        pass
    
    def get_cpi_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取CPI数据
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD，默认为近5年
            end_date: 结束日期，格式为YYYYMMDD，默认为今天
            
        Returns:
            pd.DataFrame: CPI数据
        """
        # 基本框架，实际实现略
        pass
    
    def get_ppi_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取PPI数据
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD，默认为近5年
            end_date: 结束日期，格式为YYYYMMDD，默认为今天
            
        Returns:
            pd.DataFrame: PPI数据
        """
        # 基本框架，实际实现略
        pass
    
    def get_money_supply(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取货币供应量数据（M0, M1, M2）
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD，默认为近5年
            end_date: 结束日期，格式为YYYYMMDD，默认为今天
            
        Returns:
            pd.DataFrame: 货币供应量数据
        """
        try:
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y%m%d')
                
            # 检查缓存
            cache_key = f"money_supply_{start_date}_{end_date}"
            if cache_key in self.cache:
                logger.info(f"使用缓存的货币供应量数据")
                return self.cache[cache_key]
                
            # 缓存文件路径
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.csv")
            if os.path.exists(cache_file):
                logger.info(f"从缓存文件加载货币供应量数据: {cache_file}")
                df = pd.read_csv(cache_file)
                if 'month' in df.columns:
                    df['month'] = pd.to_datetime(df['month'])
                self.cache[cache_key] = df
                return df
                
            # 使用Tushare API获取数据
            if self.pro:
                logger.info(f"通过API获取货币供应量数据")
                df = self.pro.cn_m(start_m=start_date[:6], end_m=end_date[:6])
                
                if not df.empty:
                    # 处理日期格式
                    if 'month' in df.columns:
                        df['month'] = pd.to_datetime(df['month'])
                    
                    # 保存缓存
                    df.to_csv(cache_file, index=False)
                    self.cache[cache_key] = df
                    return df
                else:
                    logger.warning(f"获取的货币供应量数据为空")
                    return pd.DataFrame()
            else:
                logger.warning("未设置Tushare API Token，无法获取货币供应量数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取货币供应量数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_interest_rate(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取利率数据
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD，默认为近5年
            end_date: 结束日期，格式为YYYYMMDD，默认为今天
            
        Returns:
            pd.DataFrame: 利率数据
        """
        # 基本框架，实际实现略
        pass
    
    def get_forex_reserves(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取外汇储备数据
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD，默认为近5年
            end_date: 结束日期，格式为YYYYMMDD，默认为今天
            
        Returns:
            pd.DataFrame: 外汇储备数据
        """
        # 基本框架，实际实现略
        pass
    
    def get_exchange_rate(self, currency: str = "USD/CNY", 
                         start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取汇率数据
        
        Args:
            currency: 货币对，如USD/CNY（美元/人民币）
            start_date: 开始日期，格式为YYYYMMDD，默认为近1年
            end_date: 结束日期，格式为YYYYMMDD，默认为今天
            
        Returns:
            pd.DataFrame: 汇率数据
        """
        try:
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                
            # 检查缓存
            cache_key = f"exchange_rate_{currency}_{start_date}_{end_date}"
            if cache_key in self.cache:
                logger.info(f"使用缓存的汇率数据")
                return self.cache[cache_key]
                
            # 缓存文件路径
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.csv")
            if os.path.exists(cache_file):
                logger.info(f"从缓存文件加载汇率数据: {cache_file}")
                df = pd.read_csv(cache_file)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                self.cache[cache_key] = df
                return df
                
            # 使用Tushare API获取数据
            if self.pro:
                logger.info(f"通过API获取汇率数据")
                
                # 解析货币对
                currencies = currency.split('/')
                if len(currencies) == 2:
                    from_currency = currencies[0]
                    to_currency = currencies[1]
                    
                    # Tushare API格式
                    ts_currency = f"{from_currency}{to_currency}"
                    df = self.pro.fx_daily(ts_code=ts_currency, 
                                         start_date=start_date, 
                                         end_date=end_date)
                    
                    if not df.empty:
                        # 处理日期格式
                        if 'trade_date' in df.columns:
                            df['date'] = pd.to_datetime(df['trade_date'])
                        
                        # 保存缓存
                        df.to_csv(cache_file, index=False)
                        self.cache[cache_key] = df
                        return df
                    else:
                        logger.warning(f"获取的汇率数据为空")
                        return pd.DataFrame()
                else:
                    logger.warning(f"货币对格式不正确: {currency}")
                    return pd.DataFrame()
            else:
                logger.warning("未设置Tushare API Token，无法获取汇率数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取汇率数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_pmi_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取采购经理人指数(PMI)数据
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD，默认为近5年
            end_date: 结束日期，格式为YYYYMMDD，默认为今天
            
        Returns:
            pd.DataFrame: PMI数据
        """
        # 基本框架，实际实现略
        pass
    
    def get_macro_overview(self) -> Dict:
        """获取宏观经济概览数据
        
        Returns:
            Dict: 宏观经济概览数据
        """
        try:
            logger.info("获取宏观经济概览数据")
            
            # 当前日期
            today = datetime.now()
            one_year_ago = (today - timedelta(days=365)).strftime('%Y%m%d')
            today_str = today.strftime('%Y%m%d')
            
            # 获取各项指标数据
            gdp_data = self.get_gdp_data()
            cpi_data = self.get_cpi_data(one_year_ago, today_str)
            ppi_data = self.get_ppi_data(one_year_ago, today_str)
            money_supply = self.get_money_supply(one_year_ago, today_str)
            interest_rate = self.get_interest_rate(one_year_ago, today_str)
            forex_reserves = self.get_forex_reserves(one_year_ago, today_str)
            exchange_rate = self.get_exchange_rate(start_date=one_year_ago, end_date=today_str)
            pmi_data = self.get_pmi_data(one_year_ago, today_str)
            
            # 构建概览数据
            overview = {}
            
            # GDP数据
            if not gdp_data.empty:
                latest_gdp = gdp_data.iloc[-1]
                previous_gdp = gdp_data.iloc[-2] if len(gdp_data) > 1 else None
                
                overview['gdp'] = {
                    'latest_value': float(latest_gdp.get('gdp', 0)),
                    'latest_period': latest_gdp.get('quarter', ''),
                    'yoy_change': float(latest_gdp.get('gdp_yoy', 0)),
                    'trend': self._analyze_trend(gdp_data, 'gdp_yoy', periods=4)
                }
            
            # CPI数据
            if not cpi_data.empty:
                latest_cpi = cpi_data.iloc[-1]
                overview['cpi'] = {
                    'latest_value': float(latest_cpi.get('cpi', 0)),
                    'latest_period': latest_cpi.get('month', ''),
                    'yoy_change': float(latest_cpi.get('cpi_yoy', 0)),
                    'trend': self._analyze_trend(cpi_data, 'cpi_yoy', periods=6)
                }
            
            # PPI数据
            if not ppi_data.empty:
                latest_ppi = ppi_data.iloc[-1]
                overview['ppi'] = {
                    'latest_value': float(latest_ppi.get('ppi', 0)),
                    'latest_period': latest_ppi.get('month', ''),
                    'yoy_change': float(latest_ppi.get('ppi_yoy', 0)),
                    'trend': self._analyze_trend(ppi_data, 'ppi_yoy', periods=6)
                }
            
            # 货币供应量数据
            if not money_supply.empty:
                latest_ms = money_supply.iloc[-1]
                overview['money_supply'] = {
                    'latest_period': latest_ms.get('month', ''),
                    'm2': float(latest_ms.get('m2', 0)),
                    'm2_yoy': float(latest_ms.get('m2_yoy', 0)),
                    'm1': float(latest_ms.get('m1', 0)),
                    'm1_yoy': float(latest_ms.get('m1_yoy', 0)),
                    'm0': float(latest_ms.get('m0', 0)),
                    'm0_yoy': float(latest_ms.get('m0_yoy', 0)),
                    'trend': self._analyze_trend(money_supply, 'm2_yoy', periods=6)
                }
            
            # 利率数据
            if not interest_rate.empty:
                latest_ir = interest_rate.iloc[-1]
                overview['interest_rate'] = {
                    'latest_period': latest_ir.get('date', ''),
                    'loan_rate': float(latest_ir.get('loan_rate', 0)),
                    'deposit_rate': float(latest_ir.get('deposit_rate', 0)),
                    'reverse_repo_rate': float(latest_ir.get('reverse_repo_rate', 0)),
                    'trend': self._analyze_trend(interest_rate, 'loan_rate', periods=6, reverse=True)
                }
            
            # PMI数据
            if not pmi_data.empty:
                latest_pmi = pmi_data.iloc[-1]
                overview['pmi'] = {
                    'latest_period': latest_pmi.get('month', ''),
                    'pmi': float(latest_pmi.get('pmi', 0)),
                    'non_mfg_pmi': float(latest_pmi.get('non_mfg_pmi', 0)),
                    'trend': 'expansion' if float(latest_pmi.get('pmi', 0)) > 50 else 'contraction'
                }
                
            # 添加宏观环境评估
            overview['assessment'] = self._assess_macro_environment(overview)
            
            return overview
            
        except Exception as e:
            logger.error(f"获取宏观经济概览数据失败: {str(e)}")
            return {}
    
    def plot_macro_indicators(self, indicators: List[str] = None, 
                             periods: int = 20, save_path: str = None) -> str:
        """绘制宏观经济指标图表
        
        Args:
            indicators: 要绘制的指标列表，如['gdp', 'cpi', 'pmi']，默认为所有指标
            periods: 显示的时间段数量，默认为20个时间点
            save_path: 保存路径，默认为None（不保存）
            
        Returns:
            str: 图表保存路径或空字符串
        """
        # 基本框架，实际实现略
        pass
    
    def _analyze_trend(self, data: pd.DataFrame, column: str, 
                      periods: int = 6, reverse: bool = False) -> str:
        """分析数据的趋势
        
        Args:
            data: 数据框
            column: 分析的列名
            periods: 分析的时间段数量
            reverse: 是否反转趋势（如利率下降为正面）
            
        Returns:
            str: 趋势描述
        """
        try:
            if data.empty or len(data) < 3 or column not in data.columns:
                return "数据不足"
                
            # 获取最近几期数据
            recent_data = data.iloc[-periods:]
            values = recent_data[column].values
            
            # 计算变化率
            changes = np.diff(values)
            
            # 计算趋势分数（正变化比例）
            positive_changes = sum(change > 0 for change in changes)
            trend_score = positive_changes / len(changes)
            
            # 拟合线性回归计算斜率
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            slope = coeffs[0]
            
            # 计算最新值相对范围位置
            latest = values[-1]
            data_range = max(values) - min(values)
            if data_range == 0:
                position = 0.5
            else:
                position = (latest - min(values)) / data_range
            
            # 根据斜率和趋势分数判断趋势
            if reverse:
                slope = -slope
                trend_score = 1 - trend_score
                
            if abs(slope) < 0.01 * np.mean(abs(values)):
                trend = "平稳"
            elif slope > 0 and trend_score > 0.7:
                trend = "强势上升"
            elif slope > 0 and trend_score > 0.5:
                trend = "上升"
            elif slope > 0:
                trend = "波动上升"
            elif slope < 0 and trend_score < 0.3:
                trend = "强势下降"
            elif slope < 0 and trend_score < 0.5:
                trend = "下降"
            elif slope < 0:
                trend = "波动下降"
            else:
                trend = "震荡"
                
            # 添加位置信息
            if position > 0.8:
                trend += "（高位）"
            elif position < 0.2:
                trend += "（低位）"
            elif position > 0.6:
                trend += "（偏高）"
            elif position < 0.4:
                trend += "（偏低）"
            else:
                trend += "（中位）"
                
            return trend
            
        except Exception as e:
            logger.error(f"分析趋势失败: {str(e)}")
            return "分析失败"
    
    def _assess_macro_environment(self, overview: Dict) -> Dict:
        """评估宏观经济环境
        
        Args:
            overview: 宏观经济概览数据
            
        Returns:
            Dict: 宏观环境评估
        """
        try:
            assessment = {}
            
            # 评估经济增长
            growth_score = 0
            growth_count = 0
            
            if 'gdp' in overview:
                gdp_yoy = overview['gdp'].get('yoy_change', 0)
                if gdp_yoy > 7:
                    growth_score += 3
                elif gdp_yoy > 6:
                    growth_score += 2
                elif gdp_yoy > 5:
                    growth_score += 1
                elif gdp_yoy < 4:
                    growth_score -= 1
                growth_count += 1
                
            if 'pmi' in overview:
                pmi = overview['pmi'].get('pmi', 0)
                if pmi > 52:
                    growth_score += 2
                elif pmi > 50:
                    growth_score += 1
                elif pmi < 48:
                    growth_score -= 2
                elif pmi < 50:
                    growth_score -= 1
                growth_count += 1
                
            # 评估通胀水平
            inflation_score = 0
            inflation_count = 0
            
            if 'cpi' in overview:
                cpi_yoy = overview['cpi'].get('yoy_change', 0)
                if cpi_yoy > 5:
                    inflation_score -= 2  # 高通胀为负面
                elif cpi_yoy > 3:
                    inflation_score -= 1
                elif cpi_yoy < 0:
                    inflation_score -= 1  # 通缩也为负面
                else:
                    inflation_score += 1  # 温和通胀为正面
                inflation_count += 1
                
            if 'ppi' in overview:
                ppi_yoy = overview['ppi'].get('yoy_change', 0)
                if ppi_yoy > 8:
                    inflation_score -= 2
                elif ppi_yoy > 5:
                    inflation_score -= 1
                elif ppi_yoy < -2:
                    inflation_score -= 1
                else:
                    inflation_score += 1
                inflation_count += 1
                
            # 评估货币环境
            monetary_score = 0
            monetary_count = 0
            
            if 'money_supply' in overview:
                m2_yoy = overview['money_supply'].get('m2_yoy', 0)
                if m2_yoy > 13:
                    monetary_score -= 1  # 货币增长过快为负面
                elif m2_yoy > 8:
                    monetary_score += 1  # 适度货币增长为正面
                elif m2_yoy < 7:
                    monetary_score -= 1  # 货币增长过慢为负面
                monetary_count += 1
                
            if 'interest_rate' in overview:
                loan_rate = overview['interest_rate'].get('loan_rate', 0)
                if loan_rate > 6:
                    monetary_score -= 2  # 高利率为负面
                elif loan_rate > 5:
                    monetary_score -= 1
                monetary_count += 1
                
            # 计算总体评分
            overall_score = 0
            total_count = 0
            
            if growth_count > 0:
                overall_score += growth_score / growth_count
                total_count += 1
                
            if inflation_count > 0:
                overall_score += inflation_score / inflation_count
                total_count += 1
                
            if monetary_count > 0:
                overall_score += monetary_score / monetary_count
                total_count += 1
                
            if total_count > 0:
                overall_score = overall_score / total_count
            
            # 生成评估结果
            assessment['growth'] = {
                'score': growth_score / growth_count if growth_count > 0 else 0,
                'status': self._get_status_text(growth_score / growth_count if growth_count > 0 else 0)
            }
            
            assessment['inflation'] = {
                'score': inflation_score / inflation_count if inflation_count > 0 else 0,
                'status': self._get_status_text(inflation_score / inflation_count if inflation_count > 0 else 0)
            }
            
            assessment['monetary'] = {
                'score': monetary_score / monetary_count if monetary_count > 0 else 0,
                'status': self._get_status_text(monetary_score / monetary_count if monetary_count > 0 else 0)
            }
            
            assessment['overall'] = {
                'score': overall_score,
                'status': self._get_status_text(overall_score)
            }
            
            # 添加投资建议
            assessment['investment_advice'] = self._get_investment_advice(assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"评估宏观环境失败: {str(e)}")
            return {}
    
    def _get_status_text(self, score: float) -> str:
        """根据评分获取状态文本
        
        Args:
            score: 评分
            
        Returns:
            str: 状态文本
        """
        if score > 1.5:
            return "非常良好"
        elif score > 0.5:
            return "良好"
        elif score > -0.5:
            return "中性"
        elif score > -1.5:
            return "疲软"
        else:
            return "非常疲软"
    
    def _get_investment_advice(self, assessment: Dict) -> str:
        """根据宏观环境评估生成投资建议
        
        Args:
            assessment: 宏观环境评估数据
            
        Returns:
            str: 投资建议
        """
        try:
            # 提取评分
            growth_score = assessment.get('growth', {}).get('score', 0)
            inflation_score = assessment.get('inflation', {}).get('score', 0)
            monetary_score = assessment.get('monetary', {}).get('score', 0)
            overall_score = assessment.get('overall', {}).get('score', 0)
            
            # 根据不同经济环境组合生成建议
            if overall_score > 1:
                return "经济环境良好，可增加权益类资产配置，关注高景气度行业的成长机会。"
            elif overall_score > 0:
                if inflation_score < -0.5:
                    return "经济平稳但通胀压力较大，建议配置部分抗通胀资产，适度持有股票，关注消费和必需品行业。"
                else:
                    return "宏观环境稳健，可均衡配置股票和债券，关注业绩稳定且有成长性的蓝筹股。"
            elif overall_score > -1:
                if monetary_score > 0.5:
                    return "经济略显疲软但货币环境宽松，可增加配置债券及高股息类资产，关注政策受益行业。"
                else:
                    return "宏观环境中性偏弱，建议降低风险暴露，增加防御性资产配置，关注消费、医疗等必需品行业。"
            else:
                return "经济环境较弱，建议保持较高现金仓位，降低风险资产配置，关注黄金等避险资产和稳定性较强的股票。"
                
        except Exception as e:
            logger.error(f"生成投资建议失败: {str(e)}")
            return "无法生成投资建议"

# 如果直接运行该模块，进行简单测试
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 初始化宏观经济分析器
    analyzer = MacroEconomicAnalyzer()
    
    # 获取宏观经济概览
    overview = analyzer.get_macro_overview()
    print("\n宏观经济概览:")
    print(json.dumps(overview, indent=2, ensure_ascii=False)) 