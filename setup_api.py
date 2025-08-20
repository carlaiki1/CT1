#!/usr/bin/env python3
"""
Secure API Key Setup and Testing Script
Helps you configure and test your Coinbase API connection safely
"""

import os
import sys
import getpass
from pathlib import Path
import ccxt
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_coinbase_api():
    """Interactive setup for Coinbase API credentials"""
    print("🔐 Coinbase API Setup")
    print("=" * 50)
    print("Please enter your Coinbase Advanced Trade API credentials.")
    print("You can find these in your Coinbase Advanced Trade account:")
    print("1. Go to https://advanced.coinbase.com/")
    print("2. Navigate to Settings > API")
    print("3. Create a new API key with 'Trade' permissions")
    print()
    
    # Get API credentials securely
    api_key = input("Enter your API Key: ").strip()
    if not api_key:
        print("❌ API Key is required!")
        return None
    
    api_secret = getpass.getpass("Enter your API Secret (hidden): ").strip()
    if not api_secret:
        print("❌ API Secret is required!")
        return None
    
    passphrase = getpass.getpass("Enter your Passphrase (optional, press Enter to skip): ").strip()
    
    return {
        'api_key': api_key,
        'api_secret': api_secret,
        'passphrase': passphrase
    }

def test_coinbase_connection(credentials, sandbox=True):
    """Test Coinbase API connection"""
    print(f"\n🧪 Testing Coinbase API Connection ({'Sandbox' if sandbox else 'Live'})")
    print("=" * 60)
    
    try:
        # Try different CCXT exchange names for Coinbase
        exchange_names = ['coinbase', 'coinbaseadvanced', 'coinbasepro']
        exchange = None
        
        for name in exchange_names:
            try:
                if hasattr(ccxt, name):
                    exchange_class = getattr(ccxt, name)
                    exchange = exchange_class({
                        'apiKey': credentials['api_key'],
                        'secret': credentials['api_secret'],
                        'password': credentials['passphrase'],
                        'sandbox': sandbox,
                        'enableRateLimit': True,
                    })
                    print(f"✅ Successfully initialized {name} exchange")
                    break
            except Exception as e:
                print(f"⚠️ Failed to initialize {name}: {e}")
                continue
        
        if not exchange:
            print("❌ Could not initialize any Coinbase exchange")
            return False
        
        # Test 1: Check exchange status
        print("\n📊 Testing exchange status...")
        try:
            status = exchange.fetch_status()
            print(f"✅ Exchange Status: {status.get('status', 'Unknown')}")
        except Exception as e:
            print(f"⚠️ Status check failed: {e}")
        
        # Test 2: Fetch account balance
        print("\n💰 Testing account balance...")
        try:
            balance = exchange.fetch_balance()
            print("✅ Balance fetched successfully!")
            
            # Show non-zero balances
            for currency, amounts in balance.items():
                if isinstance(amounts, dict) and amounts.get('total', 0) > 0:
                    print(f"   {currency}: {amounts['total']:.8f}")
            
            if not any(isinstance(v, dict) and v.get('total', 0) > 0 for v in balance.values()):
                print("   📝 No balances found (normal for new accounts)")
                
        except Exception as e:
            print(f"❌ Balance check failed: {e}")
            return False
        
        # Test 3: Fetch market data
        print("\n📈 Testing market data...")
        try:
            # Try to fetch BTC-USD ticker
            ticker = exchange.fetch_ticker('BTC-USD')
            print(f"✅ BTC-USD Price: ${ticker['last']:,.2f}")
            print(f"   24h Change: {ticker['percentage']:.2f}%")
        except Exception as e:
            print(f"⚠️ Market data test failed: {e}")
        
        # Test 4: Fetch trading pairs
        print("\n🔄 Testing available trading pairs...")
        try:
            markets = exchange.load_markets()
            btc_pairs = [symbol for symbol in markets.keys() if 'BTC' in symbol][:5]
            print(f"✅ Found {len(markets)} trading pairs")
            print(f"   Sample BTC pairs: {', '.join(btc_pairs)}")
        except Exception as e:
            print(f"⚠️ Markets test failed: {e}")
        
        print(f"\n🎉 Coinbase API connection test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def update_env_file(credentials, demo_mode=True):
    """Update .env file with new credentials"""
    env_path = Path('.env')
    
    # Read current .env file
    env_content = []
    if env_path.exists():
        with open(env_path, 'r') as f:
            env_content = f.readlines()
    
    # Update or add credentials
    updated_lines = []
    keys_updated = set()
    
    for line in env_content:
        if line.startswith('COINBASE_API_KEY='):
            updated_lines.append(f'COINBASE_API_KEY={credentials["api_key"]}\n')
            keys_updated.add('api_key')
        elif line.startswith('COINBASE_API_SECRET='):
            updated_lines.append(f'COINBASE_API_SECRET={credentials["api_secret"]}\n')
            keys_updated.add('api_secret')
        elif line.startswith('COINBASE_PASSPHRASE='):
            updated_lines.append(f'COINBASE_PASSPHRASE={credentials["passphrase"]}\n')
            keys_updated.add('passphrase')
        elif line.startswith('DEMO_MODE='):
            updated_lines.append(f'DEMO_MODE={str(demo_mode).lower()}\n')
        else:
            updated_lines.append(line)
    
    # Add missing keys
    if 'api_key' not in keys_updated:
        updated_lines.append(f'COINBASE_API_KEY={credentials["api_key"]}\n')
    if 'api_secret' not in keys_updated:
        updated_lines.append(f'COINBASE_API_SECRET={credentials["api_secret"]}\n')
    if 'passphrase' not in keys_updated:
        updated_lines.append(f'COINBASE_PASSPHRASE={credentials["passphrase"]}\n')
    
    # Write updated .env file
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"✅ Updated .env file with new credentials")

def create_24_7_trader():
    """Create a 24/7 automated trading script"""
    trader_script = '''#!/usr/bin/env python3
"""
24/7 Cryptocurrency Trading Agent
Automated trading with multiple strategies and risk management
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from trading_agent import TradingAgent
from exchange_adapter import ExchangeAdapter
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('24_7_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoTrader:
    def __init__(self):
        self.running = False
        self.exchange = None
        self.agent = None
        self.positions = {}
        self.trade_history = []
        self.last_analysis = {}
        
        # Load configuration
        self.config = {
            'symbols': ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD'],
            'check_interval': 300,  # 5 minutes
            'max_positions': int(os.getenv('MAX_POSITIONS', 5)),
            'risk_per_trade': float(os.getenv('RISK_PERCENTAGE', 2)) / 100,
            'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', 0.3)),
            'stop_loss': float(os.getenv('STOP_LOSS_PERCENTAGE', 5)) / 100,
            'take_profit': float(os.getenv('TAKE_PROFIT_PERCENTAGE', 10)) / 100,
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def initialize(self):
        """Initialize exchange and trading agent"""
        try:
            # Initialize exchange adapter
            self.exchange = ExchangeAdapter(
                exchange_name='coinbase',
                demo_mode=os.getenv('DEMO_MODE', 'true').lower() == 'true'
            )
            
            # Initialize trading agent
            self.agent = TradingAgent(self.exchange)
            
            logger.info("✅ AutoTrader initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AutoTrader: {e}")
            return False
    
    def analyze_market(self):
        """Analyze market for all configured symbols"""
        analysis_results = {}
        
        for symbol in self.config['symbols']:
            try:
                # Get current market data
                ticker = self.exchange.get_ticker(symbol)
                if not ticker:
                    continue
                
                # Get historical data for analysis
                df = self.exchange.get_historical_data(symbol, limit=100)
                if df.empty:
                    continue
                
                # Generate signals from all strategies
                signals = self.agent.analyze_symbol(symbol, df)
                
                analysis_results[symbol] = {
                    'price': ticker['price'],
                    'signals': signals,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"📊 {symbol}: ${ticker['price']:.2f} | Signals: {len(signals)}")
                
            except Exception as e:
                logger.error(f"❌ Error analyzing {symbol}: {e}")
        
        self.last_analysis = analysis_results
        return analysis_results
    
    def execute_trades(self, analysis):
        """Execute trades based on analysis"""
        for symbol, data in analysis.items():
            try:
                # Skip if no signals
                if not data['signals']:
                    continue
                
                # Get best signal
                best_signal = max(data['signals'], key=lambda x: x.confidence)
                
                # Check if signal meets confidence threshold
                if best_signal.confidence < self.config['confidence_threshold']:
                    continue
                
                # Check position limits
                if len(self.positions) >= self.config['max_positions']:
                    continue
                
                # Execute trade based on signal
                if best_signal.signal.value == 'BUY' and symbol not in self.positions:
                    self.open_position(symbol, 'buy', data['price'], best_signal)
                elif best_signal.signal.value == 'SELL' and symbol in self.positions:
                    self.close_position(symbol, data['price'], best_signal)
                
            except Exception as e:
                logger.error(f"❌ Error executing trade for {symbol}: {e}")
    
    def open_position(self, symbol, side, price, signal):
        """Open a new trading position"""
        try:
            # Calculate position size based on risk
            balance = self.exchange.get_account_balance()
            usd_balance = balance.get('USD', {}).get('free', 0)
            
            if usd_balance < 10:  # Minimum $10 balance
                logger.warning(f"⚠️ Insufficient balance: ${usd_balance:.2f}")
                return
            
            # Risk-based position sizing
            risk_amount = usd_balance * self.config['risk_per_trade']
            position_size = min(risk_amount / price, usd_balance * 0.1 / price)  # Max 10% per trade
            
            # Place order
            order = self.exchange.place_order(
                symbol=symbol,
                side=side,
                amount=position_size,
                order_type='market'
            )
            
            if order and order.get('status') == 'filled':
                # Record position
                self.positions[symbol] = {
                    'side': side,
                    'amount': position_size,
                    'entry_price': price,
                    'entry_time': datetime.now().isoformat(),
                    'signal': signal.strategy_name,
                    'confidence': signal.confidence,
                    'stop_loss': price * (1 - self.config['stop_loss']),
                    'take_profit': price * (1 + self.config['take_profit'])
                }
                
                logger.info(f"🟢 OPENED {side.upper()} position: {symbol} @ ${price:.2f} | Size: {position_size:.6f}")
                
        except Exception as e:
            logger.error(f"❌ Error opening position for {symbol}: {e}")
    
    def close_position(self, symbol, price, signal):
        """Close an existing position"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return
            
            # Place closing order
            order = self.exchange.place_order(
                symbol=symbol,
                side='sell' if position['side'] == 'buy' else 'buy',
                amount=position['amount'],
                order_type='market'
            )
            
            if order and order.get('status') == 'filled':
                # Calculate P&L
                entry_price = position['entry_price']
                pnl = (price - entry_price) * position['amount'] if position['side'] == 'buy' else (entry_price - price) * position['amount']
                pnl_pct = (pnl / (entry_price * position['amount'])) * 100
                
                # Record trade
                trade_record = {
                    'symbol': symbol,
                    'side': position['side'],
                    'amount': position['amount'],
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now().isoformat(),
                    'entry_signal': position['signal'],
                    'exit_signal': signal.strategy_name,
                    'duration': (datetime.now() - datetime.fromisoformat(position['entry_time'])).total_seconds() / 3600
                }
                
                self.trade_history.append(trade_record)
                del self.positions[symbol]
                
                logger.info(f"🔴 CLOSED {position['side'].upper()} position: {symbol} @ ${price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                
        except Exception as e:
            logger.error(f"❌ Error closing position for {symbol}: {e}")
    
    def check_risk_management(self):
        """Check and enforce risk management rules"""
        for symbol, position in list(self.positions.items()):
            try:
                # Get current price
                ticker = self.exchange.get_ticker(symbol)
                if not ticker:
                    continue
                
                current_price = ticker['price']
                
                # Check stop loss
                if ((position['side'] == 'buy' and current_price <= position['stop_loss']) or
                    (position['side'] == 'sell' and current_price >= position['stop_loss'])):
                    
                    logger.warning(f"🛑 Stop loss triggered for {symbol} @ ${current_price:.2f}")
                    # Create dummy signal for stop loss
                    from trading_agent import TradingSignal, SignalType
                    stop_signal = TradingSignal(SignalType.SELL, 1.0, "Stop loss triggered", "Risk Management")
                    self.close_position(symbol, current_price, stop_signal)
                
                # Check take profit
                elif ((position['side'] == 'buy' and current_price >= position['take_profit']) or
                      (position['side'] == 'sell' and current_price <= position['take_profit'])):
                    
                    logger.info(f"🎯 Take profit triggered for {symbol} @ ${current_price:.2f}")
                    # Create dummy signal for take profit
                    from trading_agent import TradingSignal, SignalType
                    profit_signal = TradingSignal(SignalType.SELL, 1.0, "Take profit triggered", "Risk Management")
                    self.close_position(symbol, current_price, profit_signal)
                
            except Exception as e:
                logger.error(f"❌ Error in risk management for {symbol}: {e}")
    
    def save_state(self):
        """Save current state to file"""
        try:
            state = {
                'positions': self.positions,
                'trade_history': self.trade_history[-100:],  # Keep last 100 trades
                'last_analysis': self.last_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            with open('trader_state.json', 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Error saving state: {e}")
    
    def load_state(self):
        """Load previous state from file"""
        try:
            if os.path.exists('trader_state.json'):
                with open('trader_state.json', 'r') as f:
                    state = json.load(f)
                
                self.positions = state.get('positions', {})
                self.trade_history = state.get('trade_history', [])
                self.last_analysis = state.get('last_analysis', {})
                
                logger.info(f"✅ Loaded state: {len(self.positions)} positions, {len(self.trade_history)} trades")
                
        except Exception as e:
            logger.error(f"❌ Error loading state: {e}")
    
    def print_status(self):
        """Print current trading status"""
        total_pnl = sum(trade['pnl'] for trade in self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        total_trades = len(self.trade_history)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        logger.info("=" * 60)
        logger.info("📊 TRADING STATUS")
        logger.info("=" * 60)
        logger.info(f"💰 Total P&L: ${total_pnl:.2f}")
        logger.info(f"📈 Win Rate: {win_rate:.1f}% ({winning_trades}/{total_trades})")
        logger.info(f"🎯 Open Positions: {len(self.positions)}")
        logger.info(f"📋 Symbols Monitored: {len(self.config['symbols'])}")
        
        if self.positions:
            logger.info("🔄 Current Positions:")
            for symbol, pos in self.positions.items():
                logger.info(f"   {symbol}: {pos['side'].upper()} @ ${pos['entry_price']:.2f}")
        
        logger.info("=" * 60)
    
    def run(self):
        """Main trading loop"""
        if not self.initialize():
            logger.error("❌ Failed to initialize, exiting...")
            return
        
        # Load previous state
        self.load_state()
        
        logger.info("🚀 Starting 24/7 Automated Trading...")
        logger.info(f"📊 Monitoring {len(self.config['symbols'])} symbols")
        logger.info(f"⏰ Check interval: {self.config['check_interval']} seconds")
        logger.info(f"🎯 Confidence threshold: {self.config['confidence_threshold']}")
        
        self.running = True
        iteration = 0
        
        while self.running:
            try:
                iteration += 1
                logger.info(f"🔄 Trading iteration #{iteration}")
                
                # Analyze market
                analysis = self.analyze_market()
                
                # Execute trades based on analysis
                self.execute_trades(analysis)
                
                # Check risk management
                self.check_risk_management()
                
                # Save current state
                self.save_state()
                
                # Print status every 10 iterations (50 minutes)
                if iteration % 10 == 0:
                    self.print_status()
                
                # Wait for next iteration
                if self.running:
                    time.sleep(self.config['check_interval'])
                
            except KeyboardInterrupt:
                logger.info("🛑 Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
        
        logger.info("🏁 24/7 Trading stopped")
        self.save_state()

if __name__ == "__main__":
    trader = AutoTrader()
    trader.run()
'''
    
    with open('trader_24_7.py', 'w') as f:
        f.write(trader_script)
    
    # Make it executable
    os.chmod('trader_24_7.py', 0o755)
    
    print("✅ Created 24/7 trading script: trader_24_7.py")

def main():
    """Main setup function"""
    print("🚀 Cryptocurrency Trading Agent - API Setup")
    print("=" * 60)
    
    while True:
        print("\nChoose an option:")
        print("1. Setup Coinbase API credentials")
        print("2. Test existing API connection")
        print("3. Create 24/7 trading script")
        print("4. View current configuration")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            credentials = setup_coinbase_api()
            if credentials:
                print("\n🧪 Testing API connection...")
                
                # Test in sandbox first
                sandbox_works = test_coinbase_connection(credentials, sandbox=True)
                
                if sandbox_works:
                    # Ask about live testing
                    test_live = input("\n🔴 Test live API connection? (y/N): ").lower().startswith('y')
                    if test_live:
                        live_works = test_coinbase_connection(credentials, sandbox=False)
                    else:
                        live_works = True
                    
                    if live_works:
                        # Ask about demo mode
                        demo_mode = input("\n🎮 Start in demo mode? (Y/n): ").lower()
                        demo_mode = not demo_mode.startswith('n')
                        
                        update_env_file(credentials, demo_mode)
                        print(f"\n✅ API setup complete! Demo mode: {demo_mode}")
                        
                        if not demo_mode:
                            print("⚠️  WARNING: Live trading mode enabled!")
                            print("   Make sure you understand the risks before starting automated trading.")
        
        elif choice == '2':
            # Test existing credentials
            api_key = os.getenv('COINBASE_API_KEY')
            api_secret = os.getenv('COINBASE_API_SECRET')
            passphrase = os.getenv('COINBASE_PASSPHRASE')
            
            if not all([api_key, api_secret, passphrase]):
                print("❌ No API credentials found in .env file")
                continue
            
            credentials = {
                'api_key': api_key,
                'api_secret': api_secret,
                'passphrase': passphrase
            }
            
            demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
            test_coinbase_connection(credentials, sandbox=demo_mode)
        
        elif choice == '3':
            create_24_7_trader()
            print("\n📋 To start 24/7 trading:")
            print("   python trader_24_7.py")
            print("\n📋 To run in background:")
            print("   nohup python trader_24_7.py > trader.log 2>&1 &")
        
        elif choice == '4':
            print("\n📋 Current Configuration:")
            print("-" * 30)
            env_file = Path('.env')
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        if '=' in line and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            if 'SECRET' in key or 'PASSPHRASE' in key:
                                value = '*' * len(value) if value else 'Not set'
                            print(f"   {key}: {value}")
            else:
                print("   No .env file found")
        
        elif choice == '5':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice, please try again")

if __name__ == "__main__":
    main()