#!/usr/bin/env python3
"""
Test script for the Market Dashboard UI
"""
import os
import sys
import logging
import tkinter as tk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_market_dashboard():
    """Test the Market Dashboard UI component"""
    logger.info("Testing Market Dashboard UI component...")
    
    try:
        # Import the MarketDashboard class
        from src.enhanced.market.ui.market_dashboard import MarketDashboard
        
        # Create a root window
        root = tk.Tk()
        root.title("市场仪表盘 - 测试")
        root.geometry("1200x800")
        
        # Create the dashboard
        dashboard = MarketDashboard(root)
        logger.info("Successfully created MarketDashboard instance")
        
        # Refresh data
        dashboard.refresh_data()
        logger.info("Refreshed market data")
        
        # Configure closing behavior
        def on_closing():
            logger.info("Closing dashboard")
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Run the UI
        logger.info("Starting dashboard UI - Close the window to finish the test")
        root.mainloop()
        
        return True
    
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error testing Market Dashboard: {str(e)}")
        return False

def main():
    """Main test function"""
    logger.info("Starting test for Market Dashboard UI component...")
    
    # Create required directories
    os.makedirs("./cache/market_overview", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    # Run the test
    result = test_market_dashboard()
    
    if result:
        logger.info("Market Dashboard test completed!")
    else:
        logger.error("Market Dashboard test failed!")
        
    return result

if __name__ == "__main__":
    main() 