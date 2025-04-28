#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import random
from datetime import datetime

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import enhanced API module
try:
    from enhance_api_reliability import (
        enhance_get_stock_name,
        enhance_get_stock_names_batch,
        enhance_get_stock_industry,
        CacheManager,
        APIHandler,
        update_cache_now,
        API_TIMEOUT
    )
except ImportError as e:
    print(f"Failed to import enhanced API module: {str(e)}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_api")


def test_api_enhancements():
    """Test API enhancement features"""
    logger.info("Starting API enhancement tests...")
    
    # Test set - stock codes
    test_stock_codes = [
        '601318.SH',  # 中国平安
        '000651.SZ',  # 格力电器
        '000333.SZ',  # 美的集团
        '600519.SH',  # 贵州茅台
        '000002.SZ',  # 万科A
        '600036.SH',  # 招商银行
        '000999.SZ',  # 华润三九
        '600276.SH',  # 恒瑞医药
        '123456.SZ',  # Non-existent stock
    ]
    
    # Test 1: Individual stock name retrieval
    logger.info("\n===== Test 1: Individual Stock Name Retrieval =====")
    for code in test_stock_codes:
        try:
            start_time = time.time()
            name = enhance_get_stock_name(code)
            elapsed = time.time() - start_time
            logger.info(f"Stock: {code} -> Name: {name} (Time: {elapsed:.3f}s)")
        except Exception as e:
            logger.error(f"Failed to get name for stock {code}: {str(e)}")
    
    # Test 2: Batch stock name retrieval
    logger.info("\n===== Test 2: Batch Stock Name Retrieval =====")
    try:
        start_time = time.time()
        batch_result = enhance_get_stock_names_batch(test_stock_codes)
        elapsed = time.time() - start_time
        
        logger.info(f"Batch retrieval results (Time: {elapsed:.3f}s):")
        for code, name in batch_result.items():
            logger.info(f"  {code} -> {name}")
        
        logger.info(f"Successfully retrieved {len(batch_result)}/{len(test_stock_codes)} stock names")
    except Exception as e:
        logger.error(f"Batch stock name retrieval failed: {str(e)}")
    
    # Test 3: Stock industry retrieval
    logger.info("\n===== Test 3: Stock Industry Retrieval =====")
    for code in test_stock_codes[:5]:  # Test only first 5
        try:
            start_time = time.time()
            industry = enhance_get_stock_industry(code)
            elapsed = time.time() - start_time
            logger.info(f"Stock: {code} -> Industry: {industry} (Time: {elapsed:.3f}s)")
        except Exception as e:
            logger.error(f"Failed to get industry for stock {code}: {str(e)}")
    
    # Test 4: Cache performance
    logger.info("\n===== Test 4: Cache Performance =====")
    try:
        # After first retrieval, subsequent calls should use cache
        logger.info("Retrieving same stock names again to test cache performance:")
        start_time = time.time()
        for code in test_stock_codes[:5]:  # Test only first 5
            name = enhance_get_stock_name(code)
        elapsed = time.time() - start_time
        logger.info(f"Total time to retrieve 5 stock names from cache: {elapsed:.6f}s")
        logger.info(f"Average time per stock: {elapsed/5:.6f}s")
    except Exception as e:
        logger.error(f"Cache performance test failed: {str(e)}")
    
    # Test 5: Manual cache update
    logger.info("\n===== Test 5: Manual Cache Update =====")
    try:
        # Get current cache counts
        cache_mgr = CacheManager()
        before_names_count = len(cache_mgr.stock_names_cache.get("data", {}))
        before_industry_count = len(cache_mgr.stock_industry_cache.get("data", {}))
        
        logger.info(f"Pre-update cache counts - Names: {before_names_count}, Industries: {before_industry_count}")
        
        # Execute update
        start_time = time.time()
        update_result = update_cache_now()
        elapsed = time.time() - start_time
        
        # Get cache counts after update
        cache_mgr = CacheManager()
        after_names_count = len(cache_mgr.stock_names_cache.get("data", {}))
        after_industry_count = len(cache_mgr.stock_industry_cache.get("data", {}))
        
        logger.info(f"Cache update result: {update_result} (Time: {elapsed:.3f}s)")
        logger.info(f"Post-update cache counts - Names: {after_names_count}, Industries: {after_industry_count}")
        logger.info(f"New entries - Names: {after_names_count - before_names_count}, Industries: {after_industry_count - before_industry_count}")
    except Exception as e:
        logger.error(f"Manual cache update failed: {str(e)}")
    
    # Test 6: Timeout and retry
    logger.info("\n===== Test 6: API Timeout and Retry =====")
    try:
        # Create API handler with very short timeout
        short_timeout = 0.01  # 10ms, almost certain to timeout
        api_handler = APIHandler(timeout=short_timeout, max_retries=2)
        
        logger.info(f"Testing retry mechanism with extremely short timeout {short_timeout}s...")
        
        try:
            start_time = time.time()
            # Try to get a name, expect timeout and retry
            name = api_handler.get_stock_name_with_retry('601318.SH')
            elapsed = time.time() - start_time
            logger.info(f"Result: {name} (Total time: {elapsed:.3f}s)")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.info(f"Expected timeout error: {str(e)} (Time: {elapsed:.3f}s)")
            logger.info("Retry mechanism working as expected")
    except Exception as e:
        logger.error(f"Timeout retry test failed: {str(e)}")
    
    logger.info("\n===== All tests completed =====")


if __name__ == "__main__":
    test_api_enhancements() 