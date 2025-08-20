#!/usr/bin/env python3
"""
Quick API Test Script
Test your Coinbase API credentials quickly
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from exchange_adapter import ExchangeAdapter
from trading_agent import TradingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_connection():
    """Test API connection with current credentials"""
    print("🧪 Testing Coinbase API Connection")
    print("=" * 50)
    
    # Check if credentials are set
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    passphrase = os.getenv('COINBASE_PASSPHRASE')
    demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
    
    if not api_key or api_key == 'your_coinbase_api_key_here':
        print("❌ COINBASE_API_KEY not set in .env file")
        print("💡 Run: python setup_api.py to configure your API keys")
        return False
    
    if not api_secret or api_secret == 'your_coinbase_api_secret_here':
        print("❌ COINBASE_API_SECRET not set in .env file")
        print("💡 Run: python setup_api.py to configure your API keys")
        return False
    
    print(f"🔑 API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"🎮 Demo Mode: {demo_mode}")
    print()
    
    try:
        # Initialize exchange adapter
        print("🔌 Initializing exchange connection...")
        exchange = ExchangeAdapter(exchange_name='coinbase', demo_mode=demo_mode)
        
        # Test 1: Get account balance
        print("💰 Testing account balance...")
        balance = exchange.get_account_balance()
        if balance:
            print("✅ Balance retrieved successfully!")
            for currency, amounts in balance.items():
                if isinstance(amounts, dict) and amounts.get('total', 0) > 0:
                    print(f"   {currency}: {amounts['total']:.8f}")
        else:
            print("⚠️ No balance data returned")
        
        # Test 2: Get market data
        print("\n📈 Testing market data...")
        ticker = exchange.get_ticker('BTC/USD')
        if ticker and ticker.get('price'):
            print(f"✅ BTC/USD Price: ${ticker['price']:,.2f}")
            if 'percentage' in ticker:
                print(f"   24h Change: {ticker['percentage']:.2f}%")
        else:
            print("⚠️ Could not fetch BTC/USD ticker")
        
        # Test 3: Get historical data
        print("\n📊 Testing historical data...")
        df = exchange.get_historical_data('BTC/USD', limit=10)
        if not df.empty:
            print(f"✅ Retrieved {len(df)} days of historical data")
            print(f"   Latest close: ${df['close'].iloc[-1]:,.2f}")
        else:
            print("⚠️ No historical data retrieved")
        
        # Test 4: Initialize trading agent
        print("\n🤖 Testing trading agent...")
        agent = TradingAgent(exchange)
        
        # Generate a signal
        if not df.empty:
            signals = agent.analyze_symbol('BTC/USD', df)
            if signals:
                best_signal = max(signals, key=lambda x: x.confidence)
                print(f"✅ Generated {len(signals)} trading signals")
                print(f"   Best signal: {best_signal.signal.value} ({best_signal.strategy_name})")
                print(f"   Confidence: {best_signal.confidence:.2f}")
                print(f"   Reason: {best_signal.reason}")
            else:
                print("⚠️ No trading signals generated")
        
        print(f"\n🎉 API test completed successfully!")
        print(f"🚀 Your trading agent is ready to {'simulate' if demo_mode else 'execute'} trades!")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Verify your API credentials are correct")
        print("2. Check that your API key has 'Trade' permissions")
        print("3. Ensure your IP is whitelisted (if required)")
        print("4. Try running in demo mode first")
        return False

def show_trading_status():
    """Show current trading configuration"""
    print("\n📋 Current Trading Configuration")
    print("=" * 40)
    
    config_items = [
        ('DEMO_MODE', 'Demo Mode'),
        ('DEFAULT_TRADE_AMOUNT', 'Default Trade Amount'),
        ('RISK_PERCENTAGE', 'Risk Per Trade (%)'),
        ('MAX_POSITIONS', 'Max Positions'),
        ('STOP_LOSS_PERCENTAGE', 'Stop Loss (%)'),
        ('TAKE_PROFIT_PERCENTAGE', 'Take Profit (%)'),
        ('CONFIDENCE_THRESHOLD', 'Confidence Threshold'),
    ]
    
    for env_var, description in config_items:
        value = os.getenv(env_var, 'Not set')
        print(f"   {description}: {value}")

if __name__ == "__main__":
    print("🎯 Cryptocurrency Trading Agent - API Test")
    print("=" * 60)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test API connection
    success = test_api_connection()
    
    # Show configuration
    show_trading_status()
    
    if success:
        print("\n🚀 Next Steps:")
        print("1. Review your configuration in .env file")
        print("2. Start with demo mode: DEMO_MODE=true")
        print("3. Run the web interface: python app.py")
        print("4. Or start 24/7 trading: python trader_24_7.py")
        print("\n⚠️  Remember: Always test thoroughly before live trading!")
    else:
        print("\n🔧 Setup Required:")
        print("Run: python setup_api.py")
        print("This will help you configure your API credentials securely.")