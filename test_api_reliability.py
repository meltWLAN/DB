#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_reliability_test")

# Print execution path for debugging
print("Script starting...")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Import enhancement module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    print("Attempting to import enhance_api_reliability...")
    from enhance_api_reliability import (
        enhance_get_stock_name,
        enhance_get_stock_names_batch,
        enhance_get_stock_industry,
        update_cache_now,
        get_cache_manager,
        api_handler,
        FALLBACK_STOCK_DB
    )
    print("Successfully imported API reliability enhancements")
    logger.info("Successfully imported API reliability enhancements")
except ImportError as e:
    print(f"Failed to import enhancement module: {str(e)}")
    logger.error(f"Failed to import enhancement module: {str(e)}")
    import traceback
    print(traceback.format_exc())
    logger.error(traceback.format_exc())
    sys.exit(1)

print("Starting tests...")

def run_test(test_name, test_func, *args, **kwargs):
    """Run a test function and log results"""
    print(f"Starting test: {test_name}")
    logger.info(f"Starting test: {test_name}")
    start_time = time.time()
    success = False
    result = None
    error = None
    
    try:
        result = test_func(*args, **kwargs)
        success = True
    except Exception as e:
        error = str(e)
        print(f"Test failed: {error}")
        logger.error(f"Test failed: {error}")
        import traceback
        print(traceback.format_exc())
        logger.error(traceback.format_exc())
    
    elapsed = time.time() - start_time
    
    return {
        "test_name": test_name,
        "success": success,
        "elapsed_time": elapsed,
        "result": result,
        "error": error,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def log_test_result(result):
    """Log test results in a standardized format"""
    status = "SUCCESS" if result["success"] else "FAILED"
    logger.info(f"Test '{result['test_name']}' {status} in {result['elapsed_time']:.4f}s")
    
    if result["success"]:
        if isinstance(result["result"], dict) and len(result["result"]) > 0:
            logger.info(f"Results: {len(result['result'])} items")
            for key, value in list(result["result"].items())[:5]:  # Log first 5 items
                logger.info(f"  {key} -> {value}")
            if len(result["result"]) > 5:
                logger.info(f"  ... and {len(result['result']) - 5} more items")
        else:
            logger.info(f"Result: {result['result']}")
    else:
        logger.error(f"Error: {result['error']}")
    
    logger.info("-" * 50)

def test_stock_name_reliability():
    """Test individual stock name retrieval reliability"""
    # Test codes including valid and potentially problematic ones
    test_codes = [
        '601318.SH',  # 中国平安 (common stock)
        '000651.SZ',  # 格力电器 (common stock)
        '000333.SZ',  # 美的集团 (common stock)
        '600519.SH',  # 贵州茅台 (common stock)
        '000001.SZ',  # 平安银行 (common stock)
        '999999.XX',  # Invalid code
        '123456.SZ',  # Non-existent code
    ]
    
    results = {}
    for ts_code in test_codes:
        try:
            name = enhance_get_stock_name(ts_code)
            results[ts_code] = name
            logger.info(f"Retrieved name for {ts_code}: {name}")
        except Exception as e:
            results[ts_code] = f"ERROR: {str(e)}"
            logger.error(f"Failed to get name for {ts_code}: {str(e)}")
    
    return results

def test_batch_retrieval():
    """Test batch stock name retrieval"""
    # Test with a larger set of codes
    test_codes = list(FALLBACK_STOCK_DB.keys())[:10]  # First 10 codes
    test_codes.append('123456.SZ')  # Add non-existent code
    
    try:
        start = time.time()
        results = enhance_get_stock_names_batch(test_codes)
        elapsed = time.time() - start
        
        logger.info(f"Batch retrieved {len(results)} stock names in {elapsed:.4f}s")
        logger.info(f"Average time per stock: {elapsed/len(results):.4f}s")
        
        return results
    except Exception as e:
        logger.error(f"Batch retrieval failed: {str(e)}")
        raise

def test_industry_reliability():
    """Test stock industry retrieval reliability"""
    # Test codes including valid and potentially problematic ones
    test_codes = [
        '601318.SH',  # 中国平安 (common stock)
        '000651.SZ',  # 格力电器 (common stock)
        '000333.SZ',  # 美的集团 (common stock)
        '600519.SH',  # 贵州茅台 (common stock)
        '999999.XX',  # Invalid code
        '123456.SZ',  # Non-existent code
    ]
    
    results = {}
    for ts_code in test_codes:
        try:
            industry = enhance_get_stock_industry(ts_code)
            results[ts_code] = industry
            logger.info(f"Retrieved industry for {ts_code}: {industry}")
        except Exception as e:
            results[ts_code] = f"ERROR: {str(e)}"
            logger.error(f"Failed to get industry for {ts_code}: {str(e)}")
    
    return results

def test_cache_performance():
    """Test cache performance by making repeated calls"""
    test_code = '601318.SH'  # Use a common stock
    
    # First call (may hit API)
    start = time.time()
    first_result = enhance_get_stock_name(test_code)
    first_time = time.time() - start
    
    # Second call (should hit cache)
    start = time.time()
    second_result = enhance_get_stock_name(test_code)
    second_time = time.time() - start
    
    # Compare times
    speed_improvement = (first_time / second_time) if second_time > 0 else float('inf')
    
    return {
        "first_call": {
            "result": first_result,
            "time": first_time
        },
        "second_call": {
            "result": second_result,
            "time": second_time
        },
        "speed_improvement": f"{speed_improvement:.2f}x faster",
        "cache_hit": first_result == second_result
    }

def test_manual_cache_update():
    """Test manual cache update functionality"""
    # Get initial cache state
    initial_cache = get_cache_manager()
    
    # Trigger manual update
    update_result = update_cache_now()
    
    # Get new cache state
    after_update = get_cache_manager()
    
    return {
        "initial_cache": initial_cache,
        "after_update": after_update,
        "update_result": update_result
    }

def test_fallback_mechanism():
    """Test fallback mechanism with non-existent stocks"""
    non_existent_codes = [
        '123456.SZ',
        '654321.SH',
        '999999.XX'
    ]
    
    results = {}
    for ts_code in non_existent_codes:
        try:
            name = enhance_get_stock_name(ts_code)
            industry = enhance_get_stock_industry(ts_code)
            results[ts_code] = {
                "name": name,
                "industry": industry
            }
        except Exception as e:
            results[ts_code] = {
                "error": str(e)
            }
    
    return results

def run_all_tests():
    """Run all reliability tests"""
    logger.info("=== Starting API Reliability Tests ===")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define all tests
    tests = [
        ("Individual Stock Name Retrieval", test_stock_name_reliability),
        ("Batch Stock Name Retrieval", test_batch_retrieval),
        ("Stock Industry Retrieval", test_industry_reliability),
        ("Cache Performance", test_cache_performance),
        ("Manual Cache Update", test_manual_cache_update),
        ("Fallback Mechanism", test_fallback_mechanism)
    ]
    
    results = []
    
    # Run each test and collect results
    for test_name, test_func in tests:
        result = run_test(test_name, test_func)
        log_test_result(result)
        results.append(result)
    
    # Log final summary
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"=== Test Summary: {success_count}/{len(results)} tests passed ===")
    
    for i, result in enumerate(results, 1):
        status = "✓" if result["success"] else "✗"
        logger.info(f"{i}. {status} {result['test_name']} ({result['elapsed_time']:.2f}s)")
    
    # Log API stats (fixed to work with dict structure)
    try:
        # Check if api_handler is a dict with api_call_stats key
        if isinstance(api_handler, dict) and 'api_call_stats' in api_handler:
            logger.info(f"API call statistics: {api_handler['api_call_stats']}")
        else:
            # Try attribute access
            logger.info(f"API call statistics: {api_handler.api_call_stats}")
    except AttributeError:
        logger.info(f"API call statistics: {api_handler}")
    
    return results

if __name__ == "__main__":
    run_all_tests() 