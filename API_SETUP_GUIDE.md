# 🔐 Coinbase API Setup Guide

## 📋 Prerequisites

Before setting up your API keys, ensure you have:
1. ✅ A Coinbase Advanced Trade account
2. ✅ Completed identity verification
3. ✅ Some funds in your account (for live trading)
4. ✅ Understanding of cryptocurrency trading risks

## 🔑 Step 1: Create Coinbase API Keys

### 1.1 Access API Settings
1. Go to [Coinbase Advanced Trade](https://advanced.coinbase.com/)
2. Log in to your account
3. Click on your profile icon (top right)
4. Select **"Settings"**
5. Navigate to **"API"** section

### 1.2 Create New API Key
1. Click **"Create API Key"**
2. Choose **"Advanced Trade"** (not Legacy)
3. Set permissions:
   - ✅ **View** (required for market data)
   - ✅ **Trade** (required for placing orders)
   - ❌ **Transfer** (not recommended for security)
4. Add IP whitelist (optional but recommended):
   - Add your server's IP address
   - Leave blank for any IP (less secure)
5. Click **"Create API Key"**

### 1.3 Save Your Credentials
You'll receive three pieces of information:
- **API Key**: `cb_access_key_xxxxxxxx`
- **API Secret**: Long string of characters
- **Passphrase**: Your chosen passphrase

⚠️ **IMPORTANT**: Save these immediately! The secret is only shown once.

## 🛠️ Step 2: Configure Your Trading Agent

### 2.1 Interactive Setup (Recommended)
```bash
cd /workspace/crypto_trader
source venv/bin/activate
python setup_api.py
```

Follow the prompts to:
1. Enter your API credentials securely
2. Test the connection
3. Choose demo or live mode
4. Generate 24/7 trading script

### 2.2 Manual Setup
Edit the `.env` file directly:
```bash
nano .env
```

Update these lines:
```env
COINBASE_API_KEY=your_actual_api_key_here
COINBASE_API_SECRET=your_actual_api_secret_here
COINBASE_PASSPHRASE=your_actual_passphrase_here
DEMO_MODE=true  # Set to false for live trading
```

## 🧪 Step 3: Test Your Setup

### 3.1 Quick API Test
```bash
cd /workspace/crypto_trader
source venv/bin/activate
python test_api.py
```

This will test:
- ✅ API connection
- ✅ Account balance retrieval
- ✅ Market data access
- ✅ Trading signal generation

### 3.2 Expected Output
```
🧪 Testing Coinbase API Connection
==================================================
🔑 API Key: cb_access...xyz
🎮 Demo Mode: true

🔌 Initializing exchange connection...
✅ Using coinbaseadvanced exchange class
💰 Testing account balance...
✅ Balance retrieved successfully!
   USD: 1000.00000000

📈 Testing market data...
✅ BTC/USD Price: $65,432.10
   24h Change: +2.45%

📊 Testing historical data...
✅ Retrieved 10 days of historical data
   Latest close: $65,432.10

🤖 Testing trading agent...
✅ Generated 4 trading signals
   Best signal: BUY (SMA_Crossover)
   Confidence: 0.75
   Reason: SMA bullish crossover

🎉 API test completed successfully!
🚀 Your trading agent is ready to simulate trades!
```

## 🚀 Step 4: Start Trading

### 4.1 Demo Mode (Recommended First)
```bash
# Start web interface
python app.py

# Or run 24/7 automated trading
python trader_24_7.py
```

### 4.2 Live Trading Mode
1. **Test thoroughly in demo mode first**
2. Set `DEMO_MODE=false` in `.env`
3. Start with small amounts
4. Monitor closely

```bash
# Update configuration
echo "DEMO_MODE=false" >> .env

# Start live trading
python trader_24_7.py
```

## ⚙️ Configuration Options

### Trading Parameters
```env
# Risk Management
RISK_PERCENTAGE=2              # Risk 2% per trade
MAX_POSITIONS=5                # Maximum 5 open positions
STOP_LOSS_PERCENTAGE=5         # 5% stop loss
TAKE_PROFIT_PERCENTAGE=10      # 10% take profit
CONFIDENCE_THRESHOLD=0.3       # Minimum signal confidence

# Trade Sizing
DEFAULT_TRADE_AMOUNT=100       # $100 per trade
```

### Strategy Selection
Edit `trader_24_7.py` to customize:
```python
# Symbols to trade
'symbols': ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD'],

# Check interval (seconds)
'check_interval': 300,  # 5 minutes
```

## 🛡️ Security Best Practices

### 🔒 API Key Security
1. **Never share your API keys**
2. **Use IP whitelisting** when possible
3. **Limit permissions** to only what's needed
4. **Rotate keys regularly** (monthly)
5. **Monitor API usage** in Coinbase dashboard

### 💰 Trading Security
1. **Start with demo mode**
2. **Use small amounts initially**
3. **Set appropriate stop losses**
4. **Monitor positions regularly**
5. **Keep emergency stop procedures ready**

### 🖥️ System Security
1. **Keep server updated**
2. **Use strong passwords**
3. **Enable 2FA on all accounts**
4. **Regular backups of trading data**
5. **Monitor system logs**

## 📊 Monitoring Your Trading

### 📈 Web Dashboard
Access at: http://localhost:12000
- Real-time portfolio value
- Open positions
- Trade history
- Strategy performance
- Market analysis

### 📋 Log Files
```bash
# Application logs
tail -f server.log

# 24/7 trader logs
tail -f 24_7_trader.log

# Trade history
cat trades/trade_history.json
```

### 📱 Alerts (Optional)
Set up monitoring alerts:
```bash
# Check if trader is running
ps aux | grep trader_24_7.py

# Monitor log for errors
grep "ERROR" 24_7_trader.log
```

## 🆘 Troubleshooting

### Common Issues

#### ❌ "Invalid API Key"
- Verify API key is correct
- Check if key has expired
- Ensure proper permissions are set

#### ❌ "IP Not Whitelisted"
- Add your IP to Coinbase whitelist
- Or remove IP restrictions temporarily

#### ❌ "Insufficient Funds"
- Check account balance
- Verify currency availability
- Reduce trade amounts

#### ❌ "Rate Limit Exceeded"
- Increase check intervals
- Reduce API calls frequency
- Wait and retry

### Getting Help
1. Check log files for detailed errors
2. Review Coinbase API documentation
3. Test with smaller amounts
4. Use demo mode for debugging

## 🎯 Next Steps

1. ✅ **Complete API setup**
2. ✅ **Test in demo mode**
3. ✅ **Review strategy performance**
4. ✅ **Adjust risk parameters**
5. ✅ **Monitor for 24-48 hours**
6. ✅ **Gradually increase position sizes**
7. ✅ **Set up monitoring alerts**

## ⚠️ Important Disclaimers

- **Cryptocurrency trading is highly risky**
- **Past performance doesn't guarantee future results**
- **Only trade with money you can afford to lose**
- **This software is for educational purposes**
- **Always do your own research**
- **Consider consulting a financial advisor**

---

**🚀 Ready to start? Run `python setup_api.py` to begin!**