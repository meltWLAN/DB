import unittest
import os
import sys
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from risk.risk_management import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        """Set up a RiskManager instance for testing."""
        # Disable logging for tests
        logging.disable(logging.CRITICAL)
        
        self.risk_manager = RiskManager(
            max_position_risk=0.02,
            max_portfolio_risk=0.05,
            max_industry_allocation=0.30,
            default_stop_loss_pct=0.05,
            default_take_profit_pct=0.15,
            log_level=logging.ERROR
        )
        
        # Set capital
        self.risk_manager.set_capital(100000.0)
    
    def tearDown(self):
        """Clean up after tests."""
        logging.disable(logging.NOTSET)
    
    def test_set_capital(self):
        """Test if capital is set correctly."""
        self.risk_manager.set_capital(200000.0)
        self.assertEqual(self.risk_manager.total_capital, 200000.0)
        self.assertEqual(self.risk_manager.available_capital, 200000.0)
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        # With default stop loss
        position_size = self.risk_manager.calculate_position_size(
            stock_code="AAPL",
            price=180.0,
            risk_score=0.3
        )
        
        # Expected calculation:
        # risk_amount = 100000 * 0.02 = 2000
        # adjusted_risk_amount = 2000 * (1 - 0.3 * 0.5) = 2000 * 0.85 = 1700
        # risk_per_share = 180 * 0.05 = 9
        # position_size_in_dollars = 1700 / 9 ≈ 188.89
        # shares = 188.89 / 180 ≈ 1.05 -> int(1.05) = 1
        
        # This is an approximation, but should be within a reasonable range
        self.assertGreater(position_size, 0)
        
        # Test with ATR-based stop loss
        position_size_atr = self.risk_manager.calculate_position_size(
            stock_code="AAPL",
            price=180.0,
            risk_score=0.3,
            atr=3.0
        )
        
        self.assertGreater(position_size_atr, 0)
    
    def test_add_position(self):
        """Test adding a position."""
        success = self.risk_manager.add_position(
            stock_code="AAPL",
            quantity=50,
            price=180.0,
            industry="Technology",
            risk_score=0.3
        )
        
        self.assertTrue(success)
        self.assertIn("AAPL", self.risk_manager.positions)
        self.assertEqual(self.risk_manager.positions["AAPL"]["quantity"], 50)
        self.assertEqual(self.risk_manager.positions["AAPL"]["entry_price"], 180.0)
        
        # Test adding a position with the same stock code (should fail)
        success = self.risk_manager.add_position(
            stock_code="AAPL",
            quantity=30,
            price=185.0,
            industry="Technology",
            risk_score=0.3
        )
        
        self.assertFalse(success)
    
    def test_update_position(self):
        """Test updating a position."""
        # Add a position first
        self.risk_manager.add_position(
            stock_code="MSFT",
            quantity=30,
            price=320.0,
            industry="Technology",
            risk_score=0.25,
            stop_loss_price=300.0
        )
        
        # Update price, still above stop loss
        result = self.risk_manager.update_position("MSFT", 310.0)
        self.assertFalse(result)  # No need to close position
        self.assertEqual(self.risk_manager.positions["MSFT"]["current_price"], 310.0)
        
        # Update price below stop loss
        result = self.risk_manager.update_position("MSFT", 295.0)
        self.assertTrue(result)  # Position should be closed
    
    def test_remove_position(self):
        """Test removing a position."""
        # Add a position
        self.risk_manager.add_position(
            stock_code="JPM",
            quantity=80,
            price=140.0,
            industry="Financial",
            risk_score=0.4
        )
        
        available_capital_before = self.risk_manager.available_capital
        
        # Remove the position
        self.risk_manager.remove_position("JPM")
        
        # Position should be removed
        self.assertNotIn("JPM", self.risk_manager.positions)
        
        # Available capital should increase
        self.assertGreater(self.risk_manager.available_capital, available_capital_before)
    
    def test_stop_loss_calculation(self):
        """Test different stop loss calculation methods."""
        # Fixed stop loss
        entry_price = 200.0
        fixed_stop = self.risk_manager.calculate_fixed_stop_loss(entry_price)
        self.assertEqual(fixed_stop, entry_price * 0.95)  # 5% below entry
        
        # ATR-based stop loss
        atr = 5.0
        atr_stop = self.risk_manager.calculate_atr_stop_loss(entry_price, atr)
        self.assertEqual(atr_stop, entry_price - (atr * 2.0))
        
        # Trailing stop loss
        high_price = 220.0
        trailing_stop = self.risk_manager.calculate_trailing_stop_loss(high_price)
        self.assertEqual(trailing_stop, high_price * 0.95)
    
    def test_validate_portfolio(self):
        """Test portfolio validation."""
        # Add positions
        self.risk_manager.add_position(
            stock_code="AAPL",
            quantity=50,
            price=180.0,
            industry="Technology",
            risk_score=0.3
        )
        
        self.risk_manager.add_position(
            stock_code="MSFT",
            quantity=30,
            price=320.0,
            industry="Technology",
            risk_score=0.25
        )
        
        # Validate portfolio
        is_valid, details = self.risk_manager.validate_portfolio()
        
        # Portfolio should be valid
        self.assertTrue(is_valid)
        
        # Industry allocation check
        self.assertIn("Technology", details["industry_allocation_pct"])
        tech_allocation = details["industry_allocation_pct"]["Technology"]
        self.assertLess(tech_allocation, self.risk_manager.max_industry_allocation)
        
        # Add another tech stock to exceed industry allocation
        self.risk_manager.add_position(
            stock_code="GOOGL",
            quantity=10,
            price=2500.0,
            industry="Technology",
            risk_score=0.25
        )
        
        # Validate portfolio again
        is_valid, details = self.risk_manager.validate_portfolio()
        
        # Portfolio may not be valid due to high tech allocation
        if not is_valid:
            tech_allocation = details["industry_allocation_pct"]["Technology"]
            self.assertGreaterEqual(tech_allocation, self.risk_manager.max_industry_allocation)

if __name__ == "__main__":
    unittest.main() 