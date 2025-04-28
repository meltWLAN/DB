#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
热门股票扫描器
提供连续涨停股扫描、潜在暴涨股分析和热门板块识别功能
"""

import pandas as pd
import numpy as np
import datetime
import logging
import os
import random
from stock_data_storage import StockData
import talib as ta

class HotStockScanner:
    """
    扫描市场中的连续涨停股和潜在暴涨股
    """
    def __init__(self, use_real_data=False):
        self.logger = logging.getLogger('HotStockScanner')
        self.use_real_data = use_real_data
        
        # Initialize StockData for basic operations
        self.stock_data = StockData()
        
        # Cache for faster repeated scans
        self.cache = {}
        self.cache_timestamp = None
        
        # Set up logging
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        log_filename = os.path.join('logs', 'hot_stock_scanner.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
    
    def scan_consecutive_limit_up(self, days_look_back=10, min_consecutive_days=2, refresh_cache=False):
        """
        扫描连续涨停股票
        
        参数:
            days_look_back: 向前查询天数
            min_consecutive_days: 最小连续涨停天数
            refresh_cache: 是否刷新缓存
        
        返回:
            list: 包含连续涨停股信息的字典列表
        """
        now = datetime.datetime.now()
        
        # 使用缓存避免频繁查询
        cache_key = f"consecutive_limit_up_{days_look_back}_{min_consecutive_days}"
        if not refresh_cache and self.cache_timestamp and (now - self.cache_timestamp).seconds < 300 and cache_key in self.cache:
            self.logger.info("Using cached consecutive limit up stocks data")
            return self.cache[cache_key]
        
        self.logger.info(f"Scanning for consecutive limit up stocks, min_days={min_consecutive_days}")
        
        # 获取市场上所有股票代码
        stock_codes = self.stock_data.get_stock_list()
        
        result_data = []
        
        for code in stock_codes[:100]:  # 限制处理数量，提高性能
            try:
                # 获取股票详情（包含日K数据）
                stock_detail = self.stock_data.get_stock_details(code)
                daily_data = stock_detail.get('daily_data', [])
                
                if not daily_data or len(daily_data) < days_look_back:
                    continue
                
                # 确保日期升序
                daily_data = sorted(daily_data, key=lambda x: x.get('date', ''))
                
                # 计算涨停价格 (10% 涨幅，ST股票5%)
                is_st = 'ST' in self.stock_data.get_stock_name(code)
                limit_pct = 0.05 if is_st else 0.1
                
                # 识别连续涨停
                consecutive_days = 0
                for i in range(len(daily_data)-1, 0, -1):  # 从最近日期往前
                    curr_close = daily_data[i].get('close', 0)
                    prev_close = daily_data[i-1].get('close', 0)
                    
                    if prev_close > 0:
                        change_pct = (curr_close - prev_close) / prev_close
                        is_limit_up = change_pct >= limit_pct - 0.005  # 允许0.5%的误差
                        
                        if is_limit_up:
                            consecutive_days += 1
                        else:
                            break
                
                if consecutive_days >= min_consecutive_days:
                    # 添加到结果中
                    latest_data = daily_data[-1]
                    prev_data = daily_data[-2] if len(daily_data) > 1 else {"volume": 1}
                    name = self.stock_data.get_stock_name(code)
                    industry = self.stock_data.get_stock_industry(code)
                    
                    # 计算量比
                    volume_ratio = latest_data.get('volume', 0) / prev_data.get('volume', 1)
                    
                    result_data.append({
                        'code': code,
                        'name': name,
                        'consecutive_days': consecutive_days,
                        'last_close': latest_data.get('close', 0),
                        'change_percent': latest_data.get('change', 0),
                        'volume_ratio': volume_ratio,
                        'industry': industry
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing stock {code}: {str(e)}")
        
        # 按连续涨停天数排序
        result_data = sorted(result_data, key=lambda x: x.get('consecutive_days', 0), reverse=True)
        
        # 更新缓存
        self.cache[cache_key] = result_data
        self.cache_timestamp = now
        
        self.logger.info(f"Found {len(result_data)} consecutive limit up stocks")
        return result_data
    
    def scan_potential_breakout(self, days_look_back=20, min_score=70, sector=None, top_n=50):
        """
        扫描潜在的暴涨股，寻找突破形态的股票
        
        参数:
            days_look_back: 向前查询天数
            min_score: 最小分数
            sector: 行业板块, None表示全部
            top_n: 返回前N只股票
        
        返回:
            list: 包含潜在暴涨股信息的字典列表
        """
        self.logger.info(f"Scanning for potential breakout stocks in sector: {sector}")
        
        # 获取市场上所有股票代码或按板块筛选
        if sector and sector != "全部":
            stock_codes = self.stock_data.get_stocks_by_industry(sector)
        else:
            stock_codes = self.stock_data.get_stock_list()
        
        result_data = []
        
        for code in stock_codes[:min(100, len(stock_codes))]:  # 限制处理数量，提高性能
            try:
                # 获取股票详情
                stock_detail = self.stock_data.get_stock_details(code)
                daily_data = stock_detail.get('daily_data', [])
                
                if not daily_data or len(daily_data) < days_look_back:
                    continue
                
                # 为了计算技术指标，转换为pandas DataFrame
                df = pd.DataFrame(daily_data)
                
                # 计算技术指标
                df = self._calculate_indicators(df)
                
                # 计算突破分数
                breakout_score = self._calculate_breakout_score(df)
                
                if breakout_score > min_score:  # 仅保留高分股票
                    latest_data = df.iloc[-1]
                    name = self.stock_data.get_stock_name(code)
                    industry = self.stock_data.get_stock_industry(code)
                    
                    # 计算资金流向
                    money_flow = self._calculate_money_flow()
                    
                    result_data.append({
                        'code': code,
                        'name': name,
                        'breakout_score': breakout_score,
                        'momentum_score': self._calculate_momentum(df),
                        'volume_score': round(latest_data.get('volume_ratio', 1) * 20),  # 量比转为分数
                        'rsi': round(latest_data.get('rsi', 50), 1),
                        'macd_signal': "BUY" if latest_data.get('macd_hist', 0) > 0 else "SELL",
                        'industry': industry,
                        'money_flow_score': money_flow
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing stock {code}: {str(e)}")
        
        # 按分数排序
        result_data = sorted(result_data, key=lambda x: x.get('breakout_score', 0), reverse=True)
        
        # 截取前N个结果
        result_data = result_data[:top_n]
        
        self.logger.info(f"Found {len(result_data)} potential breakout stocks")
        return result_data
    
    def get_hot_sectors(self, top_n=5):
        """
        获取当前热门行业板块
        
        参数:
            top_n: 返回前N个热门板块
        
        返回:
            list: 包含热门板块信息的字典列表
        """
        self.logger.info(f"Analyzing hot sectors, top_n={top_n}")
        
        all_industries = self.stock_data.get_industries()
        
        result_data = []
        
        for industry in all_industries:
            # 获取行业内股票
            stocks = self.stock_data.get_stocks_by_industry(industry)
            if not stocks:
                continue
                
            # 分析行业涨跌幅
            limit_up_count = 0
            total_change = 0
            sector_stocks = []
            
            for code in stocks:
                try:
                    stock_detail = self.stock_data.get_stock_details(code)
                    daily_data = stock_detail.get('daily_data', [])
                    
                    if daily_data and len(daily_data) > 1:
                        latest_data = daily_data[-1]
                        change = latest_data.get('change', 0)
                        name = self.stock_data.get_stock_name(code)
                        
                        # 涨停判断（简化为涨幅>9.5%）
                        if change > 9.5:
                            limit_up_count += 1
                            
                        total_change += change
                        
                        sector_stocks.append({
                            'code': code,
                            'name': name,
                            'change': change
                        })
                except Exception as e:
                    self.logger.error(f"Error processing stock {code} in sector {industry}: {str(e)}")
            
            if sector_stocks:
                # 计算行业平均涨幅
                avg_change = total_change / len(sector_stocks)
                
                # 根据涨停数量、平均涨幅和最大涨幅计算热度分
                sector_stocks = sorted(sector_stocks, key=lambda x: x.get('change', 0), reverse=True)
                top_performer = sector_stocks[0] if sector_stocks else None
                
                # 热度分数 = 涨停比例*50 + 平均涨幅*5
                score = (limit_up_count / len(sector_stocks) * 50) + (avg_change * 5)
                
                result_data.append({
                    'sector': industry,
                    'limit_up_count': limit_up_count,
                    'avg_change': avg_change,
                    'top_performer': top_performer,
                    'score': round(score, 1)
                })
        
        # 按热度分数排序
        result_data = sorted(result_data, key=lambda x: x.get('score', 0), reverse=True)
        
        # 截取前N个结果
        result_data = result_data[:top_n]
        
        self.logger.info(f"Found {len(result_data)} hot sectors")
        return result_data
    
    def predict_limit_up_continuation(self, stock_code, consecutive_days=2):
        """
        预测股票的涨停延续性
        
        参数:
            stock_code: 股票代码
            consecutive_days: 已经连续涨停的天数
        
        返回:
            dict: 预测结果和各因素详情
        """
        self.logger.info(f"Predicting limit-up continuation for {stock_code}, consecutive_days={consecutive_days}")
        
        try:
            # 获取股票详情
            stock_detail = self.stock_data.get_stock_details(stock_code)
            daily_data = stock_detail.get('daily_data', [])
            name = self.stock_data.get_stock_name(stock_code)
            
            if not daily_data or len(daily_data) < consecutive_days + 10:
                raise ValueError("Insufficient data for prediction")
                
            # 计算预测因素
            factors = []
            
            # 1. 成交量因素
            volume_data = [d.get('volume', 0) for d in daily_data[-consecutive_days-5:]]
            volume_trend = 0
            for i in range(consecutive_days):
                if i < len(volume_data)-1:
                    volume_trend += volume_data[-i-1] / max(1, volume_data[-i-2])
            volume_trend = volume_trend / consecutive_days if consecutive_days > 0 else 1
            
            volume_score = min(1.0, max(0.1, (volume_trend - 0.8) * 2))
            factors.append({
                'name': '成交量趋势',
                'value': f"{volume_trend:.2f}",
                'weight': 0.25,
                'score': volume_score,
                'interpretation': "量能充足" if volume_score > 0.6 else "量能不足"
            })
            
            # 2. 市场强度
            market_strength = random.uniform(0.3, 0.9)  # 模拟市场强度
            market_score = market_strength
            factors.append({
                'name': '市场强度',
                'value': f"{market_strength:.2f}",
                'weight': 0.20,
                'score': market_score,
                'interpretation': "市场强势" if market_score > 0.6 else "市场弱势"
            })
            
            # 3. 行业热度
            industry = self.stock_data.get_stock_industry(stock_code)
            industry_heat = random.uniform(0.2, 1.0)  # 模拟行业热度
            industry_score = industry_heat
            factors.append({
                'name': '行业热度',
                'value': f"{industry_heat:.2f}",
                'weight': 0.20,
                'score': industry_score,
                'interpretation': f"{industry}板块热度" + ("较高" if industry_score > 0.6 else "较低")
            })
            
            # 4. 连续涨停天数影响
            days_factor = min(1.0, consecutive_days / 5)
            days_score = 1.0 - days_factor
            factors.append({
                'name': '连续涨停天数',
                'value': f"{consecutive_days}",
                'weight': 0.15,
                'score': days_score,
                'interpretation': "连续涨停天数越多，继续涨停概率降低"
            })
            
            # 5. 技术指标
            tech_score = random.uniform(0.3, 0.9)  # 模拟技术指标评分
            factors.append({
                'name': '技术指标',
                'value': f"{tech_score:.2f}",
                'weight': 0.20,
                'score': tech_score,
                'interpretation': "技术指标" + ("看多" if tech_score > 0.6 else "看空")
            })
            
            # 计算总得分和概率
            total_score = sum(f['score'] * f['weight'] for f in factors)
            probability = total_score
            
            # 确定推荐
            if probability > 0.7:
                recommendation = "很可能继续涨停"
            elif probability > 0.5:
                recommendation = "可能继续上涨"
            elif probability > 0.3:
                recommendation = "震荡整理"
            else:
                recommendation = "可能回调"
                
            return {
                'name': name,
                'code': stock_code,
                'probability': probability,
                'recommendation': recommendation,
                'factors': factors
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting continuation for {stock_code}: {str(e)}")
            raise
    
    def _generate_mock_data(self, code, days):
        """
        生成模拟数据用于测试
        
        参数:
            code: 股票代码
            days: 天数
        
        返回:
            dataframe: 模拟的股票数据
        """
        today = datetime.datetime.now()
        data = []
        price = random.uniform(10, 50)
        
        for i in range(days):
            date = (today - datetime.timedelta(days=days-i-1)).strftime('%Y%m%d')
            
            # 随机涨跌
            change_pct = random.uniform(-0.05, 0.05)
            price = price * (1 + change_pct)
            price = max(1, price)  # 确保价格大于1
            
            # 确保价格为正数
            open_price = price * random.uniform(0.98, 1.02)
            high_price = price * random.uniform(1.01, 1.05)
            low_price = price * random.uniform(0.95, 0.99)
            close_price = price
            
            # 成交量
            volume = random.uniform(100000, 10000000)
            
            data.append({
                'trade_date': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'vol': round(volume, 0),
                'change_pct': round(change_pct * 100, 2)
            })
        
        return pd.DataFrame(data)
    
    def _calculate_indicators(self, df):
        """
        计算常用技术指标
        
        参数:
            df: 包含K线数据的DataFrame
        
        返回:
            dataframe: 添加了技术指标的DataFrame
        """
        try:
            # 确保df包含必要的列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns and col.upper() not in df.columns:
                    df[col] = df.get(col.upper(), 0) if col.upper() in df.columns else 0
            
            # 转换列名为小写
            df.columns = [col.lower() for col in df.columns]
            
            # 计算RSI
            df['rsi'] = np.random.uniform(30, 70, len(df))  # 模拟RSI
            
            # 计算MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = np.random.normal(0, 1, (3, len(df)))
            
            # 计算均线
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            
            # 计算量比
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=5).mean()
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return df
    
    def _calculate_breakout_score(self, df):
        """
        计算股票的突破形态分数
        
        参数:
            df: 包含K线数据和技术指标的DataFrame
        
        返回:
            float: 突破形态分数(0-100)
        """
        try:
            if len(df) < 10:
                return 0
                
            latest = df.iloc[-1]
            
            # 计算各项评分因素
            
            # 1. 价格突破均线 (30分)
            price_ma_score = 0
            close = latest.get('close', 0)
            ma5 = latest.get('ma5', 0)
            ma10 = latest.get('ma10', 0)
            ma20 = latest.get('ma20', 0)
            
            if ma5 > 0 and ma10 > 0 and ma20 > 0:
                if close > ma5:
                    price_ma_score += 10
                if close > ma10:
                    price_ma_score += 10
                if close > ma20:
                    price_ma_score += 10
            
            # 2. MACD指标 (20分)
            macd_score = 0
            macd_hist = latest.get('macd_hist', 0)
            macd_hist_prev = df.iloc[-2].get('macd_hist', 0) if len(df) > 1 else 0
            
            if macd_hist > 0:
                macd_score += 10
            if macd_hist > macd_hist_prev:
                macd_score += 10
            
            # 3. RSI指标 (20分)
            rsi_score = 0
            rsi = latest.get('rsi', 0)
            
            if rsi > 50 and rsi < 70:
                rsi_score += 20
            elif rsi >= 70:
                rsi_score += 10
            elif rsi > 30:
                rsi_score += 5
                
            # 4. 成交量突破 (30分)
            volume_score = 0
            volume_ratio = latest.get('volume_ratio', 0)
            
            if volume_ratio > 2:
                volume_score += 30
            elif volume_ratio > 1.5:
                volume_score += 20
            elif volume_ratio > 1:
                volume_score += 10
            
            # 总分
            total_score = price_ma_score + macd_score + rsi_score + volume_score
            
            return total_score
        except Exception as e:
            self.logger.error(f"Error calculating breakout score: {str(e)}")
            return 0
    
    def _calculate_momentum(self, df):
        """
        计算股票动量评分
        
        参数:
            df: 包含K线数据的DataFrame
        
        返回:
            float: 动量分数(0-100)
        """
        try:
            if len(df) < 10:
                return 0
                
            # 计算近期涨跌幅
            recent_changes = []
            for i in range(min(10, len(df)-1)):
                prev_close = df.iloc[-i-2].get('close', 0)
                curr_close = df.iloc[-i-1].get('close', 0)
                if prev_close > 0:
                    change = (curr_close - prev_close) / prev_close
                    recent_changes.append(change)
            
            if not recent_changes:
                return 0
                
            # 计算加权平均涨幅
            weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            weights = weights[:len(recent_changes)]
            weighted_changes = [c * w for c, w in zip(recent_changes, weights)]
            
            avg_change = sum(weighted_changes) / sum(weights)
            
            # 转换为分数
            score = min(100, max(0, (avg_change * 1000) + 50))
            
            return round(score, 1)
        except Exception as e:
            self.logger.error(f"Error calculating momentum: {str(e)}")
            return 0
    
    def _calculate_money_flow(self):
        """
        计算资金流向评分（模拟）
        
        返回:
            float: 资金流向分数(0-100)
        """
        # 随机生成资金流向分数
        return round(random.uniform(0, 100), 1)

if __name__ == "__main__":
    # 简单测试
    scanner = HotStockScanner()
    
    # 扫描连续涨停股
    limit_up_stocks = scanner.scan_consecutive_limit_up()
    print(f"Found {len(limit_up_stocks)} consecutive limit-up stocks")
    
    # 扫描潜在暴涨股
    breakout_stocks = scanner.scan_potential_breakout()
    print(f"Found {len(breakout_stocks)} potential breakout stocks")
    
    # 获取热门板块
    hot_sectors = scanner.get_hot_sectors()
    print(f"Found {len(hot_sectors)} hot sectors")
    
    # 如果有连续涨停股，预测一下延续性
    if limit_up_stocks:
        sample_code = limit_up_stocks[0].get('code', '')
        prediction = scanner.predict_limit_up_continuation(sample_code)
        print(f"Prediction for {sample_code}: {prediction.get('recommendation')}") 