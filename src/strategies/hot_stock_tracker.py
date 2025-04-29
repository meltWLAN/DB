#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
暴涨股跟踪器
实现连续涨停股票扫描和潜在暴涨股识别
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# 设置日志
logger = logging.getLogger(__name__)

class HotStockTracker:
    """暴涨股跟踪器，识别连续涨停股和潜在暴涨股"""
    
    def __init__(self, tushare_fetcher=None, data_processor=None):
        """初始化跟踪器
        
        Args:
            tushare_fetcher: TuShare数据获取器
            data_processor: 数据处理器
        """
        self.fetcher = tushare_fetcher
        self.processor = data_processor
        self.cache = {}
        
        # 创建结果目录
        self.results_dir = os.path.join('results', 'hot_stocks')
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("热门股票跟踪器初始化完成")
    
    def get_consecutive_limit_up_stocks(self, consecutive_days=1, end_date=None):
        """获取连续涨停股票
        
        Args:
            consecutive_days: 连续涨停天数
            end_date: 结束日期，格式 YYYY-MM-DD
            
        Returns:
            pd.DataFrame: 连续涨停股票数据
        """
        try:
            # 使用模拟数据进行演示
            if not self.fetcher:
                return self._generate_mock_limit_up_data(consecutive_days)
            
            # 格式化日期
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # 计算开始日期 (往前推30天获取足够数据)
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # 获取股票列表
            stock_list = self.fetcher.get_stock_list()
            if stock_list is None or stock_list.empty:
                logger.error("无法获取股票列表")
                return pd.DataFrame()
            
            # 筛选股票列表 (排除ST股和新股)
            if self.processor:
                filters = {
                    'exclude_st': True,
                    'min_list_days': 60  # 上市至少60天
                }
                stock_list = self.processor.filter_stock_list(stock_list, filters)
            
            # 查找涨停股
            limit_up_stocks = []
            
            for _, stock in stock_list.iterrows():
                # 获取日线数据
                stock_code = stock['code']
                daily_data = self.fetcher.get_daily_data(stock_code, start_date, end_date)
                
                if daily_data is None or daily_data.empty:
                    continue
                
                # 处理日线数据
                if self.processor:
                    daily_data = self.processor.process_daily_data(daily_data)
                
                # 计算涨停价
                # 涨停标准: 非ST股票涨幅达到9.5%以上，ST股票涨幅达到4.5%以上
                is_st = 'ST' in stock['name'] if 'name' in stock else False
                limit_pct = 4.5 if is_st else 9.5
                
                # 添加涨停标记
                daily_data['is_limit_up'] = daily_data['pct_change'] >= limit_pct
                
                # 查找连续涨停
                consecutive_count = 0
                last_date = None
                
                # 按日期倒序遍历最近的交易日
                for date, row in daily_data.sort_index(ascending=False).iterrows():
                    if row['is_limit_up']:
                        consecutive_count += 1
                        if last_date is None:
                            last_date = date
                    else:
                        break
                
                # 如果连续涨停天数达到要求
                if consecutive_count >= consecutive_days:
                    # 获取最后一个交易日的数据
                    last_data = daily_data.loc[last_date]
                    
                    # 获取涨停前一日的成交量
                    prev_volume = daily_data.shift(consecutive_count)['volume'].iloc[-1] if consecutive_count > 0 else 0
                    
                    # 计算量比
                    volume_ratio = last_data['volume'] / prev_volume if prev_volume > 0 else 0
                    
                    limit_up_stocks.append({
                        'code': stock_code,
                        'name': stock['name'] if 'name' in stock else '',
                        'industry': stock['industry'] if 'industry' in stock else '',
                        'consecutive_days': consecutive_count,
                        'last_close': last_data['close'],
                        'change_percent': last_data['pct_change'],
                        'volume_ratio': volume_ratio,
                        'last_date': last_date.strftime('%Y-%m-%d') if isinstance(last_date, datetime) else last_date
                    })
            
            # 创建DataFrame并按连续涨停天数降序排序
            result = pd.DataFrame(limit_up_stocks)
            if not result.empty:
                result = result.sort_values(['consecutive_days', 'volume_ratio'], ascending=False)
                logger.info(f"找到 {len(result)} 只连续{consecutive_days}天涨停的股票")
            
            return result
            
        except Exception as e:
            logger.error(f"获取连续涨停股票时出错: {str(e)}")
            return pd.DataFrame()
    
    def identify_potential_breakout_stocks(self, end_date=None, threshold_score=70):
        """识别潜在暴涨股
        
        Args:
            end_date: 结束日期，格式 YYYY-MM-DD
            threshold_score: 最小爆发潜力得分 (0-100)
            
        Returns:
            pd.DataFrame: 潜在暴涨股数据
        """
        try:
            # 使用模拟数据进行演示
            if not self.fetcher:
                return self._generate_mock_breakout_data(threshold_score)
            
            # 格式化日期
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # 计算开始日期 (往前推60天获取足够数据)
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=60)).strftime('%Y-%m-%d')
            
            # 获取股票列表
            stock_list = self.fetcher.get_stock_list()
            if stock_list is None or stock_list.empty:
                logger.error("无法获取股票列表")
                return pd.DataFrame()
            
            # 筛选股票列表 (排除ST股和新股)
            if self.processor:
                filters = {
                    'exclude_st': True,
                    'min_list_days': 90  # 上市至少90天
                }
                stock_list = self.processor.filter_stock_list(stock_list, filters)
            
            # 分析潜在暴涨股
            potential_stocks = []
            
            for _, stock in stock_list.iterrows():
                # 获取日线数据
                stock_code = stock['code']
                daily_data = self.fetcher.get_daily_data(stock_code, start_date, end_date)
                
                if daily_data is None or daily_data.empty or len(daily_data) < 30:
                    continue
                
                # 处理日线数据
                if self.processor:
                    daily_data = self.processor.process_daily_data(daily_data)
                
                # 计算技术指标得分
                breakout_score = self._calculate_breakout_score(daily_data)
                
                # 如果得分超过阈值，添加到结果
                if breakout_score >= threshold_score:
                    # 获取最后一个交易日的数据
                    last_data = daily_data.iloc[-1]
                    
                    # 计算资金流入
                    money_flow_score = self._calculate_money_flow_score(daily_data)
                    
                    potential_stocks.append({
                        'code': stock_code,
                        'name': stock['name'] if 'name' in stock else '',
                        'industry': stock['industry'] if 'industry' in stock else '',
                        'breakout_score': breakout_score,
                        'close': last_data['close'] if 'close' in last_data else 0,
                        'change_percent': last_data['pct_change'] if 'pct_change' in last_data else 0,
                        'volume_ratio': last_data['volume_ratio'] if 'volume_ratio' in last_data else 0,
                        'money_flow_score': money_flow_score,
                        'rsi': last_data['rsi'] if 'rsi' in last_data else 0,
                        'macd_signal': 'BUY' if 'macd' in last_data and last_data['macd'] > 0 else 'SELL'
                    })
            
            # 创建DataFrame并按爆发潜力得分降序排序
            result = pd.DataFrame(potential_stocks)
            if not result.empty:
                result = result.sort_values('breakout_score', ascending=False)
                logger.info(f"找到 {len(result)} 只潜在暴涨股")
            
            return result
            
        except Exception as e:
            logger.error(f"识别潜在暴涨股时出错: {str(e)}")
            return pd.DataFrame()
    
    def analyze_hot_sectors(self, top_n=10, consecutive_days=3, end_date=None):
        """分析热门板块
        
        Args:
            top_n: 返回前N个热门板块
            consecutive_days: 分析的天数
            end_date: 结束日期，格式 YYYY-MM-DD
            
        Returns:
            Dict: 热门板块数据和统计信息
        """
        try:
            # 使用模拟数据进行演示
            if not self.fetcher:
                return self._generate_mock_hot_sectors(top_n)
            
            # 待实现实际数据分析...
            # 暂时使用模拟数据
            return self._generate_mock_hot_sectors(top_n)
            
        except Exception as e:
            logger.error(f"分析热门板块时出错: {str(e)}")
            return {'hot_sectors': []}
    
    def predict_limit_up_continuation(self, stock_code, end_date=None):
        """预测涨停是否会延续
        
        Args:
            stock_code: 股票代码
            end_date: 结束日期，格式 YYYY-MM-DD
            
        Returns:
            Dict: 预测结果
        """
        try:
            # 使用模拟数据进行演示
            if not self.fetcher:
                return self._generate_mock_prediction(stock_code)
            
            # 待实现实际数据分析...
            # 暂时使用模拟数据
            return self._generate_mock_prediction(stock_code)
            
        except Exception as e:
            logger.error(f"预测涨停延续性时出错: {str(e)}")
            return {'continuation_probability': 0, 'error': str(e)}
    
    def _calculate_breakout_score(self, data):
        """计算股票的爆发潜力得分
        
        Args:
            data: 股票日线数据
            
        Returns:
            float: 爆发潜力得分 (0-100)
        """
        # 简化的得分计算
        score = 50  # 基础分数
        
        # 如果数据不足，返回0分
        if data is None or len(data) < 20:
            return 0
        
        try:
            # 获取最近的数据
            recent_data = data.iloc[-20:]
            
            # 1. 价格动量 (20%)
            price_momentum = 0
            if 'close' in recent_data.columns:
                recent_close = recent_data['close'].iloc[-1]
                prev_close = recent_data['close'].iloc[0]
                price_change = (recent_close - prev_close) / prev_close * 100
                price_momentum = min(20, max(0, price_change))
            
            # 2. 成交量变化 (20%)
            volume_score = 0
            if 'volume' in recent_data.columns:
                recent_vol = recent_data['volume'].iloc[-5:].mean()
                prev_vol = recent_data['volume'].iloc[:-5].mean()
                volume_change = recent_vol / prev_vol if prev_vol > 0 else 1
                volume_score = min(20, max(0, (volume_change - 1) * 10))
            
            # 3. 技术指标 (30%)
            tech_score = 0
            # 使用MA5上穿MA20、MACD金叉等信号
            if all(col in recent_data.columns for col in ['ma5', 'ma20']):
                ma_cross = (recent_data['ma5'].iloc[-1] > recent_data['ma20'].iloc[-1]) and (recent_data['ma5'].iloc[-2] <= recent_data['ma20'].iloc[-2])
                if ma_cross:
                    tech_score += 15
            
            if all(col in recent_data.columns for col in ['macd', 'macd_signal']):
                macd_cross = (recent_data['macd'].iloc[-1] > recent_data['macd_signal'].iloc[-1]) and (recent_data['macd'].iloc[-2] <= recent_data['macd_signal'].iloc[-2])
                if macd_cross:
                    tech_score += 15
            
            # 4. 形态识别 (30%)
            pattern_score = 0
            # 检测底部放量、突破颈线等形态
            # 简化实现...
            
            # 计算总分
            total_score = score + price_momentum + volume_score + tech_score + pattern_score
            
            return min(100, max(0, total_score))
            
        except Exception as e:
            logger.error(f"计算爆发潜力得分时出错: {str(e)}")
            return 0
    
    def _calculate_money_flow_score(self, data):
        """计算资金流入得分
        
        Args:
            data: 股票日线数据
            
        Returns:
            float: 资金流入得分 (0-100)
        """
        # 简化的资金流入分数
        if data is None or len(data) < 5:
            return 50
        
        try:
            # 如果有成交额，计算资金流入
            if 'amount' in data.columns:
                recent_amount = data['amount'].iloc[-5:].sum()
                prev_amount = data['amount'].iloc[-10:-5].sum()
                ratio = recent_amount / prev_amount if prev_amount > 0 else 1
                
                # 计算得分 (50为中性，大于50表示资金流入，小于50表示资金流出)
                score = 50 + 25 * (ratio - 1)
                return min(100, max(0, score))
            
            return 50
            
        except Exception as e:
            logger.error(f"计算资金流入得分时出错: {str(e)}")
            return 50
    
    def _generate_mock_limit_up_data(self, consecutive_days):
        """生成模拟的连续涨停数据
        
        Args:
            consecutive_days: 连续涨停天数
            
        Returns:
            pd.DataFrame: 模拟的连续涨停股票数据
        """
        # 模拟数据
        mock_data = []
        
        # 生成股票数量根据连续涨停天数减少 (3天以上涨停的股票比较少)
        num_stocks = max(3, 20 - 5 * consecutive_days)
        
        for i in range(num_stocks):
            code = f"00{1000+i}.SZ" if i % 2 == 0 else f"60{1000+i}.SH"
            mock_data.append({
                'code': code,
                'name': f"模拟股票{i+1}",
                'industry': np.random.choice(['电子', '医药', '计算机', '银行', '房地产']),
                'consecutive_days': consecutive_days + np.random.randint(0, 2),
                'last_close': np.random.uniform(10, 50).round(2),
                'change_percent': np.random.uniform(9.5, 10.5).round(2),
                'volume_ratio': np.random.uniform(0.5, 3).round(2),
                'last_date': datetime.now().strftime('%Y-%m-%d')
            })
        
        return pd.DataFrame(mock_data)
    
    def _generate_mock_breakout_data(self, threshold_score):
        """生成模拟的潜在暴涨股数据
        
        Args:
            threshold_score: 最小爆发潜力得分
            
        Returns:
            pd.DataFrame: 模拟的潜在暴涨股数据
        """
        # 模拟数据
        mock_data = []
        
        # 生成10-20只股票
        num_stocks = np.random.randint(10, 21)
        
        for i in range(num_stocks):
            code = f"00{2000+i}.SZ" if i % 2 == 0 else f"60{2000+i}.SH"
            score = max(threshold_score, np.random.uniform(threshold_score, 95).round(1))
            mock_data.append({
                'code': code,
                'name': f"模拟股票{i+1}",
                'industry': np.random.choice(['电子', '医药', '计算机', '银行', '房地产']),
                'breakout_score': score,
                'close': np.random.uniform(10, 50).round(2),
                'change_percent': np.random.uniform(0, 8).round(2),
                'volume_ratio': np.random.uniform(1, 4).round(2),
                'money_flow_score': np.random.uniform(60, 95).round(1),
                'rsi': np.random.uniform(50, 80).round(1),
                'macd_signal': np.random.choice(['BUY', 'SELL'], p=[0.7, 0.3])
            })
        
        return pd.DataFrame(mock_data)
    
    def _generate_mock_hot_sectors(self, top_n):
        """生成模拟的热门板块数据
        
        Args:
            top_n: 返回前N个热门板块
            
        Returns:
            Dict: 模拟的热门板块数据
        """
        # 模拟数据
        sectors = []
        
        # 定义行业列表
        industries = ['电子', '医药', '计算机', '银行', '房地产', '有色金属', '通信', '汽车', '食品饮料', '建筑', '家电', '纺织', '石油']
        
        # 生成top_n个热门板块
        for i in range(min(top_n, len(industries))):
            # 生成模拟的龙头股
            top_performers = []
            for j in range(np.random.randint(2, 6)):
                code = f"00{3000+i*10+j}.SZ" if j % 2 == 0 else f"60{3000+i*10+j}.SH"
                top_performers.append({
                    'code': code,
                    'name': f"{industries[i]}股{j+1}",
                    'change': np.random.uniform(5, 9.9).round(2)
                })
            
            sectors.append({
                'sector': industries[i],
                'limit_up_count': np.random.randint(1, 10),
                'avg_change': np.random.uniform(2, 8).round(2),
                'top_performer': {
                    'code': top_performers[0]['code'],
                    'name': top_performers[0]['name'],
                    'change': top_performers[0]['change']
                },
                'stocks': top_performers,
                'score': (100 - i * 5).round(1)  # 得分递减
            })
        
        return {
            'hot_sectors': sectors,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'market_heat': np.random.uniform(0, 100).round(1)
        }
    
    def _generate_mock_prediction(self, stock_code):
        """生成模拟的涨停延续性预测结果
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict: 模拟的预测结果
        """
        # 模拟数据
        probability = np.random.uniform(30, 85).round(1)
        
        # 根据概率生成推荐
        if probability >= 70:
            recommendation = "强烈推荐关注"
        elif probability >= 50:
            recommendation = "可以关注"
        else:
            recommendation = "建议观望"
        
        # 生成影响因素
        factors = [
            {
                'name': '市场情绪',
                'value': np.random.uniform(0, 100).round(1),
                'weight': np.random.uniform(0.1, 0.3).round(2),
                'score': np.random.uniform(0, 1).round(2),
                'interpretation': '市场情绪较好' if np.random.random() > 0.5 else '市场情绪一般'
            },
            {
                'name': '成交量变化',
                'value': np.random.uniform(0.5, 3).round(2),
                'weight': np.random.uniform(0.1, 0.3).round(2),
                'score': np.random.uniform(0, 1).round(2),
                'interpretation': '放量明显' if np.random.random() > 0.5 else '量能一般'
            },
            {
                'name': '技术形态',
                'value': np.random.uniform(0, 100).round(1),
                'weight': np.random.uniform(0.1, 0.3).round(2),
                'score': np.random.uniform(0, 1).round(2),
                'interpretation': '形态良好' if np.random.random() > 0.5 else '形态一般'
            },
            {
                'name': '行业趋势',
                'value': np.random.uniform(0, 100).round(1),
                'weight': np.random.uniform(0.1, 0.3).round(2),
                'score': np.random.uniform(0, 1).round(2),
                'interpretation': '行业强势' if np.random.random() > 0.5 else '行业一般'
            }
        ]
        
        return {
            'stock_code': stock_code,
            'name': f"模拟股票{stock_code[-4:]}",
            'continuation_probability': probability,
            'recommendation': recommendation,
            'factors': factors,
            'date': datetime.now().strftime('%Y-%m-%d')
        } 