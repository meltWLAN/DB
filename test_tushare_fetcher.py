#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for debugging issues with the TuShare fetcher
"""

import logging
import sys
from datetime import datetime
from src.enhanced.data.fetchers.tushare_fetcher import EnhancedTushareFetcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def test_tushare_fetcher():
    """Test TuShare fetcher functionality"""
    print("Starting TuShare fetcher test...")
    
    # Create configuration
    config = {
        "token": "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10",
        "connection_retries": 3,
        "retry_delay": 2,
        "rate_limit": 3
    }
    
    # Initialize fetcher
    print("Initializing TuShare fetcher...")
    fetcher = EnhancedTushareFetcher(config)
    
    # Test API connection
    print("Testing API health...")
    is_healthy = fetcher.check_health()
    print(f"API health check result: {is_healthy}")
    
    # Test getting index data
    print("\nTesting index data retrieval for 上证指数 (000001.SH)...")
    start_date = "2023-05-01"
    end_date = "2023-05-10"
    index_code = "000001.SH"
    
    try:
        index_data = fetcher.get_stock_index_data(index_code, start_date, end_date)
        if index_data is not None and not index_data.empty:
            print(f"Successfully retrieved {len(index_data)} records for index {index_code}")
            print("Sample data:")
            print(index_data.head())
        else:
            print(f"Failed to retrieve data for index {index_code}")
    except Exception as e:
        print(f"Error retrieving index data: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test with different index codes
    indices = [
        ("上证50", "000016.SH"),
        ("沪深300", "000300.SH"),
        ("中证500", "000905.SH"),
        ("创业板指", "399006.SZ"),
        ("深证成指", "399001.SZ")
    ]
    
    print("\nTesting multiple indices...")
    for name, code in indices:
        try:
            print(f"\nTesting {name} ({code})...")
            index_data = fetcher.get_stock_index_data(code, start_date, end_date)
            if index_data is not None and not index_data.empty:
                print(f"Successfully retrieved {len(index_data)} records")
            else:
                print(f"Failed to retrieve data")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    test_tushare_fetcher() 