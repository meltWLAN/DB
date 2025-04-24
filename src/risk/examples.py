import logging
from .risk_management import RiskManager

def basic_risk_management_example():
    """
    A basic example of using the RiskManager class.
    """
    # Initialize the risk manager with custom parameters
    risk_manager = RiskManager(
        max_position_risk=0.02,
        max_portfolio_risk=0.06,
        max_industry_allocation=0.30,
        default_stop_loss_pct=0.05,
        default_take_profit_pct=0.15,
        use_trailing_stop=True,
        atr_multiplier=2.0,
        trailing_stop_activation_pct=0.03,
        log_level=logging.INFO
    )
    
    # Set the capital
    risk_manager.set_capital(100000.0)
    
    # Add positions
    risk_manager.add_position(
        stock_code="AAPL",
        quantity=50,
        price=180.0,
        industry="Technology",
        risk_score=0.3,
        atr=4.5  # Using ATR-based stop loss
    )
    
    risk_manager.add_position(
        stock_code="MSFT",
        quantity=30,
        price=320.0,
        industry="Technology",
        risk_score=0.25,
        stop_loss_price=295.0  # Manual stop loss
    )
    
    risk_manager.add_position(
        stock_code="JPM",
        quantity=80,
        price=140.0,
        industry="Financial",
        risk_score=0.4
    )
    
    # Get position status
    positions = risk_manager.get_position_status()
    print("\nPosition Status:")
    for stock, details in positions.items():
        print(f"{stock}: Entry: ${details['entry_price']:.2f}, Current: ${details['current_price']:.2f}, "
              f"Stop Loss: ${details['stop_loss']:.2f}, Take Profit: ${details['take_profit']:.2f}, "
              f"P/L: {details['profit_loss_pct'] * 100:.2f}%")
    
    # Update positions with new prices
    print("\nUpdating positions...")
    risk_manager.update_position("AAPL", 190.0)  # Price increased
    risk_manager.update_position("MSFT", 310.0)  # Price decreased but above stop loss
    risk_manager.update_position("JPM", 130.0)   # Price decreased below stop loss
    
    # Validate portfolio risk
    is_valid, details = risk_manager.validate_portfolio()
    print(f"\nPortfolio Risk Valid: {is_valid}")
    print(f"Portfolio Value: ${details['portfolio_value']:.2f}")
    print(f"Available Capital: ${details['available_capital']:.2f}")
    print(f"Portfolio Risk: {details['portfolio_risk_pct'] * 100:.2f}% (Max: {details['max_portfolio_risk_pct'] * 100:.2f}%)")
    
    print("\nIndustry Allocations:")
    for industry, pct in details['industry_allocation_pct'].items():
        print(f"{industry}: {pct * 100:.2f}% (Max: {details['max_industry_allocation_pct'] * 100:.2f}%)")
    
    # Check which positions have triggered stop loss
    print("\nPositions to close:")
    for stock_code in list(risk_manager.positions.keys()):
        if risk_manager.check_stop_loss(stock_code):
            print(f"{stock_code} - Stop Loss triggered")
            risk_manager.remove_position(stock_code)
    
    # Get updated position status
    positions = risk_manager.get_position_status()
    print("\nRemaining Positions:")
    for stock, details in positions.items():
        print(f"{stock}: Entry: ${details['entry_price']:.2f}, Current: ${details['current_price']:.2f}, "
              f"Stop Loss: ${details['stop_loss']:.2f}, P/L: {details['profit_loss_pct'] * 100:.2f}%")

if __name__ == "__main__":
    basic_risk_management_example() 