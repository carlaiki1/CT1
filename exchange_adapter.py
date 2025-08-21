"""
Exchange Adapter for Cryptocurrency Trading
Supports Coinbase Advanced Trade and Kraken
"""

import ccxt
import requests
import hmac
import hashlib
import base64
import time
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _read_keys_from_file(filename="API.txt"):
    """Reads API key and secret from the JSON file downloaded from Coinbase."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)

            api_key = data.get("name")
            api_secret = data.get("privateKey")

            if api_key and api_secret:
                logger.info(f"✅ Loaded API credentials from {filename}")
                # The 'name' from the file is the apiKey for ccxt
                # The 'privateKey' from the file is the secret for ccxt
                return api_key, api_secret
            else:
                logger.error(f"❌ {filename} is missing 'name' or 'privateKey' fields.")
    except FileNotFoundError:
        pass # Silently fail if file not found, fallback to other methods
    except json.JSONDecodeError:
        logger.error(f"❌ Could not decode JSON from {filename}. Please ensure it is a valid JSON file.")
    except Exception as e:
        logger.error(f"❌ Error reading {filename}: {e}")
    return None, None

class ExchangeAdapter:
    def __init__(self, exchange_name: str, api_key: str = None, api_secret: str = None, passphrase: str = None, demo_mode: bool = True):
        # Try to load from file first, then from environment variables, then from direct parameters
        file_api_key, file_api_secret = _read_keys_from_file()

        self.exchange_name = exchange_name.lower()
        self.api_key = file_api_key or os.getenv('COINBASE_API_KEY') or api_key
        self.api_secret = file_api_secret or os.getenv('COINBASE_API_SECRET') or api_secret
        self.passphrase = os.getenv('COINBASE_PASSPHRASE') or passphrase

        # If keys are found, we are not in demo mode. Otherwise, we are.
        self.demo_mode = not (self.api_key and self.api_secret)
        self.exchange = None

        logger.info(f"Starting in {'Demo Mode' if self.demo_mode else 'Live Mode'}")
        
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize the exchange connection"""
        try:
            if self.demo_mode:
                logger.info("⚠️ Using simulated exchange for demo mode")
                self.exchange = None
                return

            if self.exchange_name == 'coinbase':
                logger.info("Attempting to connect to Coinbase Advanced Trade API...")
                exchange_class = getattr(ccxt, 'coinbaseadvanced')

                config = {
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'enableRateLimit': True,
                }

                self.exchange = exchange_class(config)
                logger.info("✅ Successfully connected to Coinbase Advanced Trade API.")

            elif self.exchange_name == 'kraken':
                self.exchange = ccxt.kraken({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'enableRateLimit': True,
                })
            else:
                raise ValueError(f"Unsupported exchange: {self.exchange_name}")
            
            logger.info(f"✅ {self.exchange_name.title()} exchange initialized (demo_mode: {self.demo_mode})")
            
        except Exception as e:
            if self.demo_mode:
                logger.info(f"⚠️ Demo mode: Using simulated exchange for {self.exchange_name}")
                self.exchange = None
            else:
                logger.error(f"❌ Failed to initialize {self.exchange_name}: {e}")
                raise

    def get_account_balance(self) -> Dict:
        """Get account balance"""
        try:
            if self.demo_mode:
                return {
                    'USD': {'free': 10000.0, 'used': 0.0, 'total': 10000.0},
                    'BTC': {'free': 0.0, 'used': 0.0, 'total': 0.0},
                    'ETH': {'free': 0.0, 'used': 0.0, 'total': 0.0}
                }
            
            balance = self.exchange.fetch_balance()
            return balance
            
        except Exception as e:
            logger.error(f"❌ Error fetching balance: {e}")
            return {}

    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data for a symbol"""
        try:
            if self.demo_mode or self.exchange is None:
                # Generate demo ticker data
                return self._generate_demo_ticker(symbol)
            
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'change': ticker['change'],
                'percentage': ticker['percentage'],
                'timestamp': ticker['timestamp']
            }
        except Exception as e:
            logger.error(f"❌ Error fetching ticker for {symbol}: {e}")
            if self.demo_mode:
                return self._generate_demo_ticker(symbol)
            return {}
    
    def _generate_demo_ticker(self, symbol: str) -> Dict:
        """Generate demo ticker data"""
        import numpy as np
        import time
        
        # Use a hash for consistent-ish random prices for any symbol
        np.random.seed(hash(symbol) & 0xFFFFFFFF)

        # Determine a plausible base price range based on common crypto values
        if 'BTC' in symbol:
            base_price = 65000
        elif 'ETH' in symbol:
            base_price = 3200
        elif 'USD' in symbol and len(symbol.split('/')[0]) > 3: # Likely an altcoin
            base_price = np.random.uniform(0.00001, 10.0)
        else: # Other major pairs
            base_price = np.random.uniform(10, 500)

        price = max(0.000001, base_price * (1 + np.random.normal(0, 0.05)))
        spread = price * 0.001
        change = np.random.normal(0, price * 0.02)
        
        return {
            'symbol': symbol,
            'price': price,
            'bid': price - spread/2,
            'ask': price + spread/2,
            'volume': np.random.randint(1000000, 10000000),
            'change': change,
            'percentage': (change / price) * 100 if price > 0 else 0,
            'timestamp': int(time.time() * 1000)
        }

    def get_historical_data(self, symbol: str, timeframe: str = '1d', limit: int = 1000) -> pd.DataFrame:
        """Get historical OHLCV data"""
        try:
            if self.demo_mode or self.exchange is None:
                # Generate sample data for demo mode
                return self._generate_sample_data(symbol, limit)
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except ccxt.base.errors.NetworkError as e:
            logger.error(f"❌ Network error fetching historical data for {symbol}: {e}")
        except ccxt.base.errors.ExchangeError as e:
            logger.error(f"❌ Exchange error fetching historical data for {symbol}: {e}")
        except ccxt.base.errors.BaseError as e:
            logger.error(f"❌ CCXT error fetching historical data for {symbol}: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching historical data for {symbol}: {e}")

        if self.demo_mode:
            return self._generate_sample_data(symbol, limit)
        return pd.DataFrame()
    
    def _generate_sample_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Generate sample data for demo mode"""
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=limit)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')[:limit]
        
        # Generate realistic price movements
        np.random.seed(hash(symbol) & 0xFFFFFFFF)
        
        # Base price depends on symbol
        if 'BTC' in symbol:
            base_price = 65000
            volatility = 0.03
        elif 'ETH' in symbol:
            base_price = 3200
            volatility = 0.04
        elif 'USD' in symbol and len(symbol.split('/')[0]) > 3:
            base_price = np.random.uniform(0.00001, 10.0)
            volatility = 0.08
        else:
            base_price = np.random.uniform(10, 500)
            volatility = 0.05
        
        # Generate price series
        returns = np.random.normal(0.001, volatility, len(dates))
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            open_price = prices[i-1] if i > 0 else price
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'open': open_price,
                'high': max(open_price, high, price),
                'low': min(open_price, low, price),
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df

    def get_top_cryptocurrencies(self, limit: int = 25) -> List[str]:
        """Get top cryptocurrencies by volume"""
        try:
            tickers = self.exchange.fetch_tickers()
            
            # Filter USD pairs and sort by volume
            usd_pairs = []
            for symbol, ticker in tickers.items():
                if '/USD' in symbol and ticker['baseVolume']:
                    usd_pairs.append({
                        'symbol': symbol,
                        'volume': ticker['baseVolume']
                    })
            
            # Sort by volume and return top symbols
            usd_pairs.sort(key=lambda x: x['volume'], reverse=True)
            return [pair['symbol'] for pair in usd_pairs[:limit]]
            
        except Exception as e:
            logger.error(f"❌ Error fetching top cryptocurrencies: {e}")
            # Return default top cryptos if API fails
            return [
                'BTC/USD', 'ETH/USD', 'ADA/USD', 'SOL/USD', 'XRP/USD',
                'DOT/USD', 'DOGE/USD', 'AVAX/USD', 'SHIB/USD', 'MATIC/USD',
                'LTC/USD', 'UNI/USD', 'LINK/USD', 'ALGO/USD', 'BCH/USD',
                'XLM/USD', 'VET/USD', 'ICP/USD', 'FIL/USD', 'TRX/USD',
                'ETC/USD', 'THETA/USD', 'AAVE/USD', 'ATOM/USD', 'XTZ/USD'
            ]

    def place_order(self, symbol: str, side: str, amount: float, order_type: str = 'market', price: float = None) -> Dict:
        """Place a trading order"""
        try:
            if self.demo_mode:
                # Simulate order execution
                current_price = self.get_ticker(symbol).get('price', 0)
                return {
                    'id': f"demo_{int(time.time())}",
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': current_price,
                    'type': order_type,
                    'status': 'filled',
                    'timestamp': datetime.now().isoformat(),
                    'demo': True
                }
            
            if order_type == 'market':
                order = self.exchange.create_market_order(symbol, side, amount)
            else:
                order = self.exchange.create_limit_order(symbol, side, amount, price)
            
            return order
            
        except Exception as e:
            logger.error(f"❌ Error placing order: {e}")
            return {}

    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Get order status"""
        try:
            if self.demo_mode:
                return {
                    'id': order_id,
                    'status': 'filled',
                    'filled': 100,
                    'remaining': 0
                }
            
            order = self.exchange.fetch_order(order_id, symbol)
            return order
            
        except Exception as e:
            logger.error(f"❌ Error fetching order status: {e}")
            return {}

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            if self.demo_mode:
                return True
            
            self.exchange.cancel_order(order_id, symbol)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error canceling order: {e}")
            return False

    def get_trading_fees(self, symbol: str) -> Dict:
        """Get trading fees for a symbol"""
        try:
            fees = self.exchange.fetch_trading_fees()
            return fees.get(symbol, {'maker': 0.005, 'taker': 0.005})
        except Exception as e:
            logger.error(f"❌ Error fetching trading fees: {e}")
            return {'maker': 0.005, 'taker': 0.005}

if __name__ == "__main__":
    # Test the exchange adapter
    print("🧪 Testing Exchange Adapter...")
    
    # You can add test logic for ExchangeAdapter here if needed
    # For example, initializing it in demo mode
    try:
        adapter = ExchangeAdapter('coinbase', demo_mode=True)
        balance = adapter.get_account_balance()
        print(f"Demo mode balance: {balance.get('USD')}")
        
        ticker = adapter.get_ticker('BTC/USD')
        print(f"Demo mode ticker for BTC/USD: {ticker}")
        
        hist_data = adapter.get_historical_data('BTC/USD', limit=5)
        print("Demo mode historical data for BTC/USD:")
        print(hist_data.head())

    except Exception as e:
        print(f"Error testing adapter: {e}")
