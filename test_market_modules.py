#!/usr/bin/env python3
"""
Test script to verify the market overview upgrade modules
"""
import logging
import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_north_fund_analyzer():
    """Test the North Fund Analyzer module"""
    logger.info("Testing North Fund Analyzer...")
    
    from src.enhanced.market.north_fund_analyzer import NorthFundAnalyzer
    
    # Create analyzer instance
    analyzer = NorthFundAnalyzer()
    
    # Create simulated data for testing
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    
    # Simulate north fund flow data
    north_flow_data = []
    for i, date in enumerate(dates):
        # Create oscillating pattern with trend
        flow = 5000 + 3000 * np.sin(i/10) + i * 10
        north_flow_data.append({
            'trade_date': date,
            'north_money': flow,
            'south_money': flow * 0.8,
            'north_acc': flow * (i+1),
            'south_acc': flow * 0.8 * (i+1)
        })
    
    df = pd.DataFrame(north_flow_data)
    
    # Mock the get_north_fund_flow method
    analyzer.get_north_fund_flow = lambda start_date=None, end_date=None: df
    
    # Test fund flow trend analysis
    trend = analyzer.analyze_fund_flow_trend(days=30)
    logger.info(f"Fund flow trend analysis result: {trend}")
    
    # Test plotting
    plot_path = analyzer.plot_fund_flow_trend(days=30, save_path="./results/north_fund_trend_test.png")
    logger.info(f"Fund flow trend plot saved to: {plot_path}")
    
    # Create simulated sector allocation data
    sectors = ['电子', '医药', '食品饮料', '银行', '房地产', '汽车', '计算机', '通信', '电气设备']
    sector_data = []
    for i, sector in enumerate(sectors):
        sector_data.append({
            'industry': sector,
            'weight': 20 - i * 2,
            'stock_count': 30 - i * 3,
            'total_ratio': (20 - i * 2) / 100,
            'total_vol': (20 - i * 2) * 1000000
        })
    
    sector_df = pd.DataFrame(sector_data)
    
    # Mock the get_sector_allocation method
    analyzer.get_sector_allocation = lambda date=None: sector_df
    
    # Test sector allocation plotting
    plot_path = analyzer.plot_sector_allocation(top_n=5, save_path="./results/north_fund_sector_test.png")
    logger.info(f"Sector allocation plot saved to: {plot_path}")
    
    return True

def test_market_heatmap():
    """Test the Market Heatmap module"""
    logger.info("Testing Market Heatmap...")
    
    from src.enhanced.market.market_heatmap import MarketHeatmap
    
    # Create heatmap instance
    heatmap = MarketHeatmap(save_dir="./results/heatmaps")
    
    # Create simulated industry data
    industry_data = pd.DataFrame({
        'name': ['电子', '医药', '食品饮料', '银行', '房地产', '汽车', '计算机', '通信', '电气设备', '建筑'],
        'change': [3.5, 2.1, 1.8, -0.5, -1.2, 2.8, 3.2, 1.5, -0.8, -1.5]
    })
    
    # Test industry performance heatmap
    plot_path = heatmap.create_industry_performance_heatmap(
        industry_data, 
        save_path="./results/heatmaps/industry_heatmap_test.png"
    )
    logger.info(f"Industry performance heatmap saved to: {plot_path}")
    
    # Create simulated stock data
    stocks = []
    industries = ['电子', '医药', '食品饮料', '银行', '房地产']
    for industry in industries:
        for i in range(5):
            change = np.random.uniform(-5, 5)
            stocks.append({
                'name': f"{industry}股票{i+1}",
                'industry': industry,
                'change': change
            })
    
    stock_data = pd.DataFrame(stocks)
    
    # Test stock change heatmap
    plot_path = heatmap.create_stock_change_heatmap(
        stock_data,
        save_path="./results/heatmaps/stock_heatmap_test.png"
    )
    logger.info(f"Stock change heatmap saved to: {plot_path}")
    
    # Create simulated market emotion indicators
    emotion_indicators = {
        '量比': 1.2,
        '涨跌家数比': 1.5,
        '北向资金': 25.6,
        'MACD': 0.02,
        'RSI': 65,
        '恐慌指数': 25,
        '市场宽度': 0.68,
        '换手率': 1.8,
        '情绪指数': 0.75
    }
    
    # Test market emotion heatmap
    plot_path = heatmap.create_market_emotion_heatmap(
        emotion_indicators,
        save_path="./results/heatmaps/emotion_heatmap_test.png"
    )
    logger.info(f"Market emotion heatmap saved to: {plot_path}")
    
    # Create correlation data
    corr_data = pd.DataFrame(np.random.uniform(-1, 1, size=(5, 5)), 
                           columns=['沪指', '深成指', '创业板', '上证50', '北向资金'])
    
    # Make it symmetrical for correlation matrix
    corr_data = (corr_data + corr_data.T) / 2
    np.fill_diagonal(corr_data.values, 1)
    
    # Test correlation heatmap
    plot_path = heatmap.create_correlation_heatmap(
        corr_data,
        save_path="./results/heatmaps/correlation_heatmap_test.png"
    )
    logger.info(f"Correlation heatmap saved to: {plot_path}")
    
    return True

def test_macro_economic_analyzer():
    """Test the Macro Economic Analyzer module"""
    logger.info("Testing Macro Economic Analyzer...")
    
    from src.enhanced.market.macro_economic_analyzer import MacroEconomicAnalyzer
    
    # Create analyzer instance
    analyzer = MacroEconomicAnalyzer()
    
    # Create simulated money supply data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
    
    money_supply_data = []
    for i, date in enumerate(dates):
        # Simulate increasing trend with seasonal variations
        m2 = 2500000 + i * 50000 + np.random.normal(0, 20000)
        m1 = 600000 + i * 10000 + np.random.normal(0, 5000)
        m0 = 100000 + i * 2000 + np.random.normal(0, 1000)
        
        money_supply_data.append({
            'month': date,
            'm2': m2,
            'm2_yoy': 8.5 + np.random.normal(0, 0.5),
            'm1': m1,
            'm1_yoy': 7.0 + np.random.normal(0, 0.8),
            'm0': m0,
            'm0_yoy': 5.5 + np.random.normal(0, 1.0),
        })
    
    money_supply_df = pd.DataFrame(money_supply_data)
    
    # Mock the get_money_supply method
    analyzer.get_money_supply = lambda start_date=None, end_date=None: money_supply_df
    
    # Create exchange rate data
    exchange_rate_data = []
    for i, date in enumerate(pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')):
        # Simulate slightly decreasing trend (CNY strengthening against USD)
        rate = 7.2 - i * 0.0002 + np.random.normal(0, 0.01)
        
        exchange_rate_data.append({
            'date': date,
            'trade_date': date.strftime('%Y%m%d'),
            'ts_code': 'USDCNY',
            'open': rate - 0.01,
            'high': rate + 0.02,
            'low': rate - 0.02,
            'close': rate,
        })
    
    exchange_rate_df = pd.DataFrame(exchange_rate_data)
    
    # Mock the get_exchange_rate method
    analyzer.get_exchange_rate = lambda currency=None, start_date=None, end_date=None: exchange_rate_df
    
    # Mock other methods to return empty DataFrames
    analyzer.get_gdp_data = lambda start_date=None, end_date=None: pd.DataFrame()
    analyzer.get_cpi_data = lambda start_date=None, end_date=None: pd.DataFrame()
    analyzer.get_ppi_data = lambda start_date=None, end_date=None: pd.DataFrame()
    analyzer.get_interest_rate = lambda start_date=None, end_date=None: pd.DataFrame()
    analyzer.get_forex_reserves = lambda start_date=None, end_date=None: pd.DataFrame()
    analyzer.get_pmi_data = lambda start_date=None, end_date=None: pd.DataFrame()
    
    # Test getting macro overview
    overview = analyzer.get_macro_overview()
    logger.info(f"Macro overview available keys: {list(overview.keys())}")
    
    # Test investment advice functionality
    if 'assessment' in overview and 'investment_advice' in overview['assessment']:
        logger.info(f"Investment advice: {overview['assessment']['investment_advice']}")
    
    return True

def main():
    """Main test function"""
    logger.info("Starting tests for market overview upgrade modules...")
    
    # Create required directories
    os.makedirs("./cache/north_fund", exist_ok=True)
    os.makedirs("./cache/macro", exist_ok=True)
    os.makedirs("./results/heatmaps", exist_ok=True)
    
    tests = [
        test_north_fund_analyzer,
        test_market_heatmap,
        test_macro_economic_analyzer
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            logger.info(f"{test.__name__}: {'SUCCESS' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"{test.__name__} failed with error: {str(e)}")
            results.append(False)
    
    # Print summary
    success_count = sum(results)
    logger.info(f"Test summary: {success_count}/{len(tests)} tests passed")
    
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main()) 