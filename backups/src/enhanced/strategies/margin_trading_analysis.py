import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import json
import tushare as ts
from src.enhanced.config.settings import TUSHARE_TOKEN, DATA_DIR, CACHE_ENABLED
from src.enhanced.data.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)

class MarginTradingAnalysis:
    """
    Class for analyzing margin trading data and identifying potential trading opportunities
    based on margin trading patterns and volumes.
    """
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.cache_dir = os.path.join(DATA_DIR, 'margin_trading_cache')
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Configure Tushare
        try:
            self.ts_api = ts.pro_api(TUSHARE_TOKEN)
            logger.info("Tushare API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tushare API: {e}")
            self.ts_api = None
    
    def get_margin_data(self, stock_code, start_date, end_date):
        """
        Fetch margin trading data for a specific stock within a date range
        
        Args:
            stock_code (str): Stock code in the format '000001.SZ'
            start_date (str): Start date in 'YYYYMMDD' format
            end_date (str): End date in 'YYYYMMDD' format
            
        Returns:
            pd.DataFrame: DataFrame containing margin trading data
        """
        cache_file = os.path.join(self.cache_dir, f"{stock_code}_margin_{start_date}_{end_date}.csv")
        
        # Check if data is cached and CACHE_ENABLED is True
        if CACHE_ENABLED and os.path.exists(cache_file):
            logger.info(f"Loading cached margin data for {stock_code}")
            return pd.read_csv(cache_file)
        
        try:
            if self.ts_api:
                df = self.ts_api.margin_detail(
                    ts_code=stock_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is not None and not df.empty:
                    # Save to cache if CACHE_ENABLED
                    if CACHE_ENABLED:
                        df.to_csv(cache_file, index=False)
                    return df
                else:
                    logger.warning(f"No margin data found for {stock_code}")
                    return self._generate_mock_margin_data(stock_code, start_date, end_date)
            else:
                return self._generate_mock_margin_data(stock_code, start_date, end_date)
        
        except Exception as e:
            logger.error(f"Error fetching margin data for {stock_code}: {e}")
            return self._generate_mock_margin_data(stock_code, start_date, end_date)
    
    def _generate_mock_margin_data(self, stock_code, start_date, end_date):
        """
        Generate mock margin trading data for testing
        
        Args:
            stock_code (str): Stock code
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            pd.DataFrame: Mock margin trading data
        """
        logger.info(f"Generating mock margin data for {stock_code}")
        
        # Convert string dates to datetime
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        
        # Generate date range
        date_range = [start + timedelta(days=x) for x in range((end-start).days + 1)]
        dates = [d.strftime('%Y%m%d') for d in date_range]
        
        # Generate mock data
        np.random.seed(42)  # For reproducibility
        n = len(dates)
        
        mock_data = {
            'trade_date': dates,
            'ts_code': [stock_code] * n,
            'rzye': np.random.normal(1000000, 200000, n),  # Financing balance
            'rqye': np.random.normal(500000, 100000, n),   # Securities lending balance
            'rzmre': np.random.normal(100000, 30000, n),   # Financing buy
            'rqyl': np.random.normal(50000, 15000, n),     # Securities lending volume
            'rzche': np.random.normal(90000, 30000, n),    # Financing repayment
            'rqchl': np.random.normal(45000, 15000, n),    # Securities lending repayment
            'rzjmre': np.random.normal(10000, 5000, n),    # Net financing
            'rqjmcl': np.random.normal(5000, 2500, n)      # Net securities lending
        }
        
        return pd.DataFrame(mock_data)
    
    def analyze_margin_pressure(self, stock_code, start_date, end_date):
        """
        Analyze margin trading pressure for a stock
        
        Args:
            stock_code (str): Stock code
            start_date (str): Start date in 'YYYYMMDD' format
            end_date (str): End date in 'YYYYMMDD' format
            
        Returns:
            dict: Analysis results
        """
        margin_data = self.get_margin_data(stock_code, start_date, end_date)
        
        if margin_data is None or margin_data.empty:
            return {
                'status': 'error',
                'message': 'No margin data available for analysis'
            }
        
        # Sort by date
        margin_data['trade_date'] = pd.to_datetime(margin_data['trade_date'])
        margin_data = margin_data.sort_values('trade_date')
        
        # Calculate margin trading pressure indicators
        margin_data['rzrqye'] = margin_data['rzye'] + margin_data['rqye']  # Total margin trading balance
        margin_data['rzrqye_change'] = margin_data['rzrqye'].pct_change()  # Daily change rate
        
        # Calculate 5-day, 10-day moving averages
        margin_data['rzrqye_ma5'] = margin_data['rzrqye'].rolling(window=5).mean()
        margin_data['rzrqye_ma10'] = margin_data['rzrqye'].rolling(window=10).mean()
        
        # Calculate financing/securities lending ratio
        margin_data['rz_rq_ratio'] = margin_data['rzye'] / margin_data['rqye'].replace(0, np.nan)
        
        # Check for margin trading pressure signals
        results = {
            'stock_code': stock_code,
            'analysis_period': f"{start_date} to {end_date}",
            'latest_data': margin_data.iloc[-1]['trade_date'].strftime('%Y-%m-%d'),
            'metrics': {
                'latest_rzye': float(margin_data.iloc[-1]['rzye']),
                'latest_rqye': float(margin_data.iloc[-1]['rqye']),
                'latest_rzrqye': float(margin_data.iloc[-1]['rzrqye']),
                'latest_rz_rq_ratio': float(margin_data.iloc[-1]['rz_rq_ratio']),
                'rzrqye_change_5d': float(margin_data.iloc[-1]['rzrqye'] / margin_data.iloc[-5]['rzrqye'] - 1 if len(margin_data) > 5 else 0),
                'rzrqye_change_10d': float(margin_data.iloc[-1]['rzrqye'] / margin_data.iloc[-10]['rzrqye'] - 1 if len(margin_data) > 10 else 0)
            },
            'signals': {}
        }
        
        # Identify specific margin trading signals
        results['signals']['high_financing_pressure'] = bool(
            margin_data.iloc[-1]['rzye'] > margin_data['rzye'].mean() * 1.2
        )
        
        results['signals']['increasing_short_pressure'] = bool(
            len(margin_data) > 5 and
            margin_data.iloc[-1]['rqye'] > margin_data.iloc[-5]['rqye'] * 1.1
        )
        
        results['signals']['margin_balance_divergence'] = bool(
            len(margin_data) > 10 and
            margin_data.iloc[-1]['rzrqye'] > margin_data.iloc[-1]['rzrqye_ma10'] * 1.15
        )
        
        # Overall margin pressure recommendation
        if results['signals']['high_financing_pressure'] and not results['signals']['increasing_short_pressure']:
            results['recommendation'] = 'bullish'
            results['recommendation_reason'] = 'High financing with low short selling pressure indicates bullish sentiment'
        elif results['signals']['increasing_short_pressure'] and not results['signals']['high_financing_pressure']:
            results['recommendation'] = 'bearish'
            results['recommendation_reason'] = 'Increasing securities lending pressure indicates bearish sentiment'
        elif results['signals']['margin_balance_divergence']:
            results['recommendation'] = 'caution'
            results['recommendation_reason'] = 'Margin balance significantly above moving average suggests potential reversal'
        else:
            results['recommendation'] = 'neutral'
            results['recommendation_reason'] = 'No significant margin trading pressure detected'
            
        return results
    
    def plot_margin_trends(self, stock_code, start_date, end_date, save_path=None):
        """
        Generate a plot showing margin trading trends
        
        Args:
            stock_code (str): Stock code
            start_date (str): Start date in 'YYYYMMDD' format
            end_date (str): End date in 'YYYYMMDD' format
            save_path (str): Path to save the plot, if None, the plot will be displayed
            
        Returns:
            str: Path to the saved plot or None if display only
        """
        margin_data = self.get_margin_data(stock_code, start_date, end_date)
        
        if margin_data is None or margin_data.empty:
            logger.error(f"No margin data available for plotting {stock_code}")
            return None
        
        # Sort by date
        margin_data['trade_date'] = pd.to_datetime(margin_data['trade_date'])
        margin_data = margin_data.sort_values('trade_date')
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Financing and Securities lending balance
        plt.subplot(2, 1, 1)
        plt.plot(margin_data['trade_date'], margin_data['rzye'], 'b-', label='Financing Balance')
        plt.plot(margin_data['trade_date'], margin_data['rqye'], 'r-', label='Securities Lending Balance')
        plt.title(f'Margin Trading Analysis for {stock_code}')
        plt.ylabel('Balance')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Financing buy and repayment
        plt.subplot(2, 1, 2)
        plt.plot(margin_data['trade_date'], margin_data['rzmre'], 'g-', label='Financing Buy')
        plt.plot(margin_data['trade_date'], margin_data['rzche'], 'm-', label='Financing Repayment')
        plt.ylabel('Amount')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save or display plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            plt.show()
            return None
            
    def get_margin_trading_summary(self, stock_list, days=30):
        """
        Generate a summary of margin trading for a list of stocks over the last N days
        
        Args:
            stock_list (list): List of stock codes
            days (int): Number of days to look back
            
        Returns:
            pd.DataFrame: Summary DataFrame
        """
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        summary_list = []
        
        for stock in stock_list:
            try:
                analysis = self.analyze_margin_pressure(stock, start_date, end_date)
                
                if 'metrics' in analysis:
                    summary_list.append({
                        'stock_code': stock,
                        'rzye': analysis['metrics']['latest_rzye'],
                        'rqye': analysis['metrics']['latest_rqye'],
                        'rz_rq_ratio': analysis['metrics']['latest_rz_rq_ratio'],
                        'rzrqye_change_5d': analysis['metrics']['rzrqye_change_5d'],
                        'recommendation': analysis['recommendation']
                    })
            except Exception as e:
                logger.error(f"Error analyzing margin data for {stock}: {e}")
                
        if summary_list:
            df = pd.DataFrame(summary_list)
            # Sort by recommendation priority (bullish first, then neutral, then bearish)
            recommendation_order = {'bullish': 0, 'neutral': 1, 'caution': 2, 'bearish': 3}
            df['recommendation_order'] = df['recommendation'].map(recommendation_order)
            df = df.sort_values('recommendation_order')
            df = df.drop('recommendation_order', axis=1)
            return df
        else:
            return pd.DataFrame()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the margin trading analysis
    analyzer = MarginTradingAnalysis()
    
    # Test with a single stock
    stock_code = '000001.SZ'  # Ping An Bank
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
    
    print(f"Analyzing margin trading for {stock_code}...")
    results = analyzer.analyze_margin_pressure(stock_code, start_date, end_date)
    print(json.dumps(results, indent=2))
    
    # Generate and save plot
    plot_path = os.path.join(DATA_DIR, f"{stock_code}_margin_analysis.png")
    analyzer.plot_margin_trends(stock_code, start_date, end_date, save_path=plot_path)
    print(f"Plot saved to {plot_path}") 