#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for debugging issues with the DataSourceManager
"""

import logging
import sys
from datetime import datetime
from src.enhanced.data.fetchers.data_source_manager import DataSourceManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def test_data_source_manager():
    """Test DataSourceManager functionality"""
    print("Starting DataSourceManager test...")
    
    # Initialize manager
    print("Initializing DataSourceManager...")
    manager = DataSourceManager()
    
    # Print the health status of data sources
    print("\nData sources status:")
    for source_name, status in manager.health_status.items():
        print(f"  - {source_name}: {'Healthy' if status else 'Unhealthy'}")
    
    print("\nAvailable sources:", manager.available_sources)
    print("Primary source:", manager.primary_source)
    
    # Test getting index data
    print("\nTesting index data retrieval...")
    start_date = "2023-05-01"
    end_date = "2023-05-10"
    
    indices = [
        ("上证指数", "000001.SH"),
        ("上证50", "000016.SH"),
        ("沪深300", "000300.SH"),
        ("中证500", "000905.SH"),
        ("创业板指", "399006.SZ"),
        ("深证成指", "399001.SZ")
    ]
    
    for name, code in indices:
        try:
            print(f"\nTesting {name} ({code})...")
            # Get the data
            index_data = manager.get_stock_index_data(code, start_date, end_date)
            
            if index_data is not None and not index_data.empty:
                print(f"Successfully retrieved {len(index_data)} records")
                print("Sample data:")
                print(index_data.head(2))
            else:
                print(f"Failed to retrieve data")
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Test the DataSourceManager's is_trading_day and trading date methods
    print("\nTesting trading calendar methods...")
    
    # Testing is_trading_day
    test_date = "2023-05-05"  # Friday, should be a trading day
    try:
        is_trading = manager.is_trading_day(test_date)
        print(f"Is {test_date} a trading day? {is_trading}")
    except Exception as e:
        print(f"Error checking if date is trading day: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Testing get_latest_trading_date
    try:
        latest_date = manager.get_latest_trading_date()
        print(f"Latest trading date: {latest_date}")
    except Exception as e:
        print(f"Error getting latest trading date: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Testing get_previous_trading_date
    try:
        prev_date = manager.get_previous_trading_date("2023-05-10", 1)
        print(f"Previous trading date before 2023-05-10: {prev_date}")
    except Exception as e:
        print(f"Error getting previous trading date: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed.")

if __name__ == "__main__":
    test_data_source_manager() 