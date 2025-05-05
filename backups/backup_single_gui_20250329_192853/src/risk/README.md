# Risk Management Module

A comprehensive risk management module for trading applications that handles position sizing, stop loss management, and portfolio risk validation.

## Overview

The Risk Management module provides a powerful, customizable framework for managing trading risk through:

1. **Position Sizing**: Calculate optimal position sizes based on total capital, risk tolerance, and individual security characteristics.
2. **Stop Loss Management**: Implement multiple stop loss strategies including fixed percentage, ATR-based, and trailing stop losses.
3. **Portfolio Risk Validation**: Ensure overall portfolio risk stays within defined parameters.
4. **Industry Allocation Limits**: Prevent overexposure to any single industry.

## Classes

### RiskManager

The `RiskManager` class is the main component, responsible for all risk management functionality:

```python
from risk.risk_management import RiskManager

risk_manager = RiskManager(
    max_position_risk=0.02,
    max_portfolio_risk=0.05,
    max_industry_allocation=0.25,
    default_stop_loss_pct=0.05
)
```

## Key Methods

- `set_capital(total_capital)`: Set the total capital available for trading.
- `calculate_position_size(stock_code, price, risk_score, [stop_loss_price], [atr])`: Calculate the optimal position size.
- `add_position(stock_code, quantity, price, industry, risk_score, [stop_loss_price], [take_profit_price], [atr])`: Add a new position.
- `update_position(stock_code, current_price)`: Update a position with the current price, and check stop loss/take profit conditions.
- `remove_position(stock_code)`: Remove a position and update available capital.
- `validate_portfolio()`: Validate the portfolio against risk parameters.

## Stop Loss Types

1. **Fixed Percentage Stop Loss**:
   ```python
   # Default behavior, uses default_stop_loss_pct
   stop_loss_price = risk_manager.calculate_fixed_stop_loss(entry_price)
   ```

2. **ATR-Based Stop Loss**:
   ```python
   stop_loss_price = risk_manager.calculate_atr_stop_loss(entry_price, atr)
   ```

3. **Trailing Stop Loss**:
   ```python
   # Automatically applied when use_trailing_stop=True
   # and price moves up by trailing_stop_activation_pct
   ```

## Usage Example

```python
# Initialize risk manager
risk_manager = RiskManager(
    max_position_risk=0.02,
    max_portfolio_risk=0.05,
    max_industry_allocation=0.25
)

# Set capital
risk_manager.set_capital(100000.0)

# Add positions
risk_manager.add_position(
    stock_code="AAPL",
    quantity=50,
    price=180.0,
    industry="Technology",
    risk_score=0.3,
    atr=4.5
)

# Update positions
for stock_code in risk_manager.positions:
    # Get current price from your data source
    current_price = get_current_price(stock_code)
    should_close = risk_manager.update_position(stock_code, current_price)
    
    if should_close:
        risk_manager.remove_position(stock_code)

# Validate portfolio
is_valid, details = risk_manager.validate_portfolio()
if not is_valid:
    print("Portfolio risk exceeded:", details)
``` 