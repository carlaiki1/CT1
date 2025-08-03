# 🚀 Quick Start Guide - Cryptocurrency Trading Agent

## 📋 Prerequisites

1. **Python 3.8+** installed on your system
2. **Coinbase Advanced Trade account** (for live trading)
3. **API credentials** from Coinbase (for live trading)

## ⚡ Quick Installation

```bash
# 1. Navigate to the project directory
cd crypto_trader

# 2. Run the installation script
chmod +x install.sh
./install.sh

# 3. Activate the virtual environment
source venv/bin/activate
```

## 🎮 Demo Mode (No API Required)

Start with demo mode to test the system without real money:

```bash
# Run interactive demo
python demo.py

# Or start the web interface in demo mode
python app.py
# Then open: http://localhost:12000
```

## 📊 Run Comprehensive Backtest

Test strategies on top 25 cryptocurrencies with historical data:

```bash
# Run the backtest (takes 10-30 minutes)
chmod +x run_backtest.sh
./run_backtest.sh

# Or run directly
python backtest.py
```

## 🔧 Live Trading Setup

### Step 1: Get Coinbase API Credentials

1. Go to [Coinbase Advanced Trade](https://pro.coinbase.com/)
2. Navigate to API settings
3. Create new API key with trading permissions
4. Save your API Key, Secret, and Passphrase

### Step 2: Configure the Agent

Edit the `.env` file:

```bash
# Coinbase Advanced Trade API
COINBASE_API_KEY=your_actual_api_key_here
COINBASE_API_SECRET=your_actual_api_secret_here
COINBASE_PASSPHRASE=your_actual_passphrase_here

# Trading Configuration
DEMO_MODE=false  # Set to false for live trading
DEFAULT_TRADE_AMOUNT=100
RISK_PERCENTAGE=2
MAX_POSITIONS=5
```

### Step 3: Start Trading

```bash
# Start the web interface
python app.py

# Or run command-line trading
python -c "
from exchange_adapter import ExchangeAdapter
from trading_agent import TradingAgent
import os

# Load your API credentials
exchange = ExchangeAdapter('coinbase', 
    os.getenv('COINBASE_API_KEY'),
    os.getenv('COINBASE_API_SECRET'), 
    os.getenv('COINBASE_PASSPHRASE'),
    demo_mode=False)

agent = TradingAgent(exchange)
print('🚀 Trading agent ready!')
"
```

## 🌐 Web Interface

The web interface provides:

- **Real-time monitoring** of positions and performance
- **Strategy analysis** for any cryptocurrency
- **Automated trading** controls
- **Performance metrics** and charts
- **Configuration management**

Access at: `http://localhost:12000`

## 📈 Understanding the Strategies

### 1. SMA (Simple Moving Average)
- **Signal**: Buy when short MA crosses above long MA
- **Best for**: Trending markets
- **Parameters**: 20-day and 50-day moving averages

### 2. RSI (Relative Strength Index)
- **Signal**: Buy when oversold (<30), sell when overbought (>70)
- **Best for**: Range-bound markets
- **Parameters**: 14-day period

### 3. MACD (Moving Average Convergence Divergence)
- **Signal**: Buy/sell on MACD line crossovers
- **Best for**: Momentum trading
- **Parameters**: 12, 26, 9 periods

### 4. Bollinger Bands
- **Signal**: Buy near lower band, sell near upper band
- **Best for**: Mean reversion
- **Parameters**: 20-day period, 2 standard deviations

## ⚠️ Risk Management

The agent includes built-in risk management:

- **Position sizing**: Based on account percentage
- **Stop losses**: Automatic exit at 5% loss
- **Take profits**: Automatic exit at 10% gain
- **Maximum positions**: Limit concurrent trades
- **Demo mode**: Test without real money

## 📊 Backtest Results Interpretation

After running the backtest, you'll get:

### Performance Metrics
- **Total Return**: Overall profit/loss percentage
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline

### Strategy Comparison
- **Average Returns**: Mean performance across all symbols
- **Success Rate**: Percentage of profitable symbols
- **Best/Worst Performance**: Range of outcomes

### Symbol Analysis
- **Top Performers**: Best cryptocurrencies for each strategy
- **Consistency**: Which symbols work well across strategies

## 🔍 Troubleshooting

### Common Issues

**"No data available"**
- Check internet connection
- Verify symbol format (e.g., 'BTC/USD')
- Try different time periods

**"API Error"**
- Verify API credentials in .env file
- Check API permissions on Coinbase
- Ensure sufficient account balance

**"Installation failed"**
- Check Python version (3.8+ required)
- Install missing system dependencies
- Try manual package installation

**"Backtest takes too long"**
- Reduce number of symbols
- Decrease historical period
- Check system resources

### Getting Help

1. Check the error logs in the console
2. Verify all dependencies are installed
3. Test with demo mode first
4. Review the README.md for detailed documentation

## 🎯 Next Steps

1. **Start with demo mode** to understand the system
2. **Run backtests** to see historical performance
3. **Analyze results** to choose best strategies
4. **Configure risk parameters** based on your tolerance
5. **Start with small amounts** in live trading
6. **Monitor performance** and adjust as needed

## 📞 Support

- Check `README.md` for detailed documentation
- Review `CHECKLIST.md` before going live
- Test thoroughly in demo mode
- Start with small amounts

---

**⚠️ Important Disclaimer**: Cryptocurrency trading involves significant risk. Never invest more than you can afford to lose. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor.