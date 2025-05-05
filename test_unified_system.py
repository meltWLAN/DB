#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一系统测试
"""

import os
import sys
import unittest
from datetime import datetime
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_system import UnifiedSystem

class TestUnifiedSystem(unittest.TestCase):
    """统一系统测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        # 创建统一系统实例
        cls.system = UnifiedSystem()
        
    def setUp(self):
        """每个测试用例初始化"""
        # 创建测试数据目录
        for directory in ['data', 'cache', 'results', 'models', 'config']:
            os.makedirs(directory, exist_ok=True)
            
    def test_system_initialization(self):
        """测试系统初始化"""
        self.assertIsNotNone(self.system)
        self.assertIsNotNone(self.system.stock_data)
        self.assertIsNotNone(self.system.data_provider)
        self.assertIsNotNone(self.system.data_checker)
        self.assertIsNotNone(self.system.hot_stock_scanner)
        self.assertIsNotNone(self.system.momentum_analyzer)
        self.assertIsNotNone(self.system.ma_strategy)
        self.assertIsNotNone(self.system.financial_analyzer)
        self.assertIsNotNone(self.system.ml_momentum)
        self.assertIsNotNone(self.system.risk_manager)
        self.assertIsNotNone(self.system.backtest_engine)
        self.assertIsNotNone(self.system.capital_flow)
        self.assertIsNotNone(self.system.sentiment)
        self.assertIsNotNone(self.system.notification)
        
    def test_stock_data_mock(self):
        """测试股票数据模拟"""
        # 获取股票列表
        stock_list = self.system.stock_data.get_stock_list()
        self.assertIsInstance(stock_list, list)
        self.assertGreater(len(stock_list), 0)
        
        # 获取行业列表
        industries = self.system.stock_data.get_industries()
        self.assertIsInstance(industries, list)
        self.assertGreater(len(industries), 0)
        
        # 获取股票详情
        stock_code = stock_list[0]
        details = self.system.stock_data.get_stock_details(stock_code)
        self.assertIsInstance(details, dict)
        self.assertIn('daily_data', details)
        
    def test_hot_stock_scanner(self):
        """测试热门股票扫描"""
        # 扫描连续涨停股
        results = self.system.hot_stock_scanner.scan_consecutive_limit_up(days=2)
        self.assertIsInstance(results, list)
        
    def test_momentum_analyzer(self):
        """测试动量分析"""
        # 获取一个测试股票
        stock_list = self.system.stock_data.get_stock_list()
        test_stock = stock_list[0]
        
        # 获取股票数据
        stock_data = self.system.stock_data.get_stock_details(test_stock)
        
        # 创建测试数据
        df = pd.DataFrame(stock_data['daily_data'])
        
        # 分析动量
        results = self.system.momentum_analyzer.analyze_stocks(
            sample_size=1,
            min_score=50
        )
        self.assertIsInstance(results, list)
        
    def test_risk_manager(self):
        """测试风险管理"""
        # 测试仓位计算
        position_size = self.system.risk_manager.calculate_position_size(
            capital=100000,
            price=10.0,
            stop_loss=9.0
        )
        self.assertIsInstance(position_size, (int, float))
        self.assertGreater(position_size, 0)
        
    def test_sentiment_analyzer(self):
        """测试情感分析"""
        # 创建测试数据
        test_data = pd.DataFrame({
            'close': np.random.random(100) * 100,
            'volume': np.random.random(100) * 1000000,
            'high': np.random.random(100) * 100,
            'low': np.random.random(100) * 100
        })
        
        # 分析市场情绪
        results = self.system.sentiment.analyze_market_sentiment(test_data)
        self.assertIsInstance(results, dict)
        self.assertIn('sentiment_score', results)
        
    def test_notification(self):
        """测试通知系统"""
        # 发送测试通知
        results = self.system.notification.send_notification(
            title="测试通知",
            message="这是一条测试消息",
            notification_type='all'
        )
        self.assertIsInstance(results, dict)
        
    def tearDown(self):
        """每个测试用例清理"""
        pass
        
    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        pass

if __name__ == '__main__':
    unittest.main() 