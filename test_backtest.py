"""
Test backtest with sample data and well-known cryptocurrencies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from backtest import BacktestEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample cryptocurrency data for testing"""
    # Create 3 years of sample data
    dates = pd.date_range(start='2021-08-01', end='2024-08-01', freq='D')
    
    # Generate realistic crypto price movements
    np.random.seed(42)
    
    # BTC-like data starting at $30,000
    btc_returns = np.random.normal(0.001, 0.04, len(dates))  # Higher volatility
    btc_prices = [30000]
    for ret in btc_returns[1:]:
        btc_prices.append(btc_prices[-1] * (1 + ret))
    
    # ETH-like data starting at $2,000
    eth_returns = np.random.normal(0.0015, 0.05, len(dates))  # Even higher volatility
    eth_prices = [2000]
    for ret in eth_returns[1:]:
        eth_prices.append(eth_prices[-1] * (1 + ret))
    
    # Create OHLCV data
    def create_ohlcv(prices):
        data = []
        for i, price in enumerate(prices):
            # Add some intraday volatility
            high = price * (1 + abs(np.random.normal(0, 0.02)))
            low = price * (1 - abs(np.random.normal(0, 0.02)))
            open_price = prices[i-1] if i > 0 else price
            close = price
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'open': open_price,
                'high': max(open_price, high, close),
                'low': min(open_price, low, close),
                'close': close,
                'volume': volume
            })
        return pd.DataFrame(data, index=dates)
    
    return {
        'BTC/USD': create_ohlcv(btc_prices),
        'ETH/USD': create_ohlcv(eth_prices)
    }

def test_strategies():
    """Test individual strategies with sample data"""
    print("🧪 Testing Trading Strategies with Sample Data")
    print("=" * 60)
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Import strategies
    from trading_agent import SMAStrategy, RSIStrategy, MACDStrategy, BollingerBandsStrategy
    
    strategies = [
        SMAStrategy(short_window=20, long_window=50),
        RSIStrategy(period=14, overbought=70, oversold=30),
        MACDStrategy(fast_period=12, slow_period=26, signal_period=9),
        BollingerBandsStrategy(period=20, std_dev=2.0)
    ]
    
    for symbol, df in sample_data.items():
        print(f"\n📊 Testing {symbol}")
        print("-" * 40)
        
        for strategy in strategies:
            try:
                signal = strategy.generate_signal(df, symbol)
                print(f"✅ {strategy.name:15} | Signal: {signal.signal.value:4} | Confidence: {signal.confidence:.2f}")
                print(f"   Reason: {signal.reason}")
            except Exception as e:
                print(f"❌ {strategy.name:15} | Error: {e}")
        
        print()

def run_sample_backtest():
    """Run backtest with sample data"""
    print("\n🚀 Running Sample Backtest")
    print("=" * 60)
    
    # Create custom backtest engine
    engine = BacktestEngine(initial_capital=10000, commission=0.005)
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Override the data fetching
    engine.results = {}
    
    # Run backtests on sample data
    for symbol, df in sample_data.items():
        engine.results[symbol] = {}
        
        for strategy_name, strategy in engine.strategies.items():
            print(f"📈 Testing {strategy_name} on {symbol}...")
            result = engine.backtest_strategy(strategy, df, symbol)
            engine.results[symbol][strategy_name] = result
    
    # Generate performance report
    report = engine.generate_performance_report()
    
    # Print results
    print("\n📊 BACKTEST RESULTS")
    print("=" * 60)
    
    for strategy_name, metrics in report['summary'].items():
        print(f"\n🎯 {strategy_name} Strategy:")
        print(f"   Average Return: {metrics['avg_return']:.2f}%")
        print(f"   Median Return: {metrics['median_return']:.2f}%")
        print(f"   Best Return: {metrics['best_return']:.2f}%")
        print(f"   Worst Return: {metrics['worst_return']:.2f}%")
        print(f"   Win Rate: {metrics['avg_win_rate']:.1f}%")
        print(f"   Sharpe Ratio: {metrics['avg_sharpe']:.2f}")
        print(f"   Max Drawdown: {metrics['avg_max_drawdown']:.1f}%")
        print(f"   Success Rate: {metrics['success_rate']:.1f}% ({metrics['profitable_symbols']}/{metrics['symbols_tested']} symbols)")
    
    # Save results
    engine.save_results('backtest_results/sample_backtest_results.json')
    
    print(f"\n✅ Sample backtest completed!")
    print(f"📁 Results saved to backtest_results/sample_backtest_results.json")

def test_yfinance_data():
    """Test with real data from Yahoo Finance"""
    print("\n📈 Testing with Real Yahoo Finance Data")
    print("=" * 60)
    
    symbols = ['BTC-USD', 'ETH-USD']
    
    for symbol in symbols:
        try:
            print(f"\n📊 Fetching {symbol} data...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='1y')  # 1 year of data
            
            if not df.empty:
                df.columns = df.columns.str.lower()
                print(f"✅ {symbol}: {len(df)} days of data")
                print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
                print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
                
                # Test one strategy
                from trading_agent import SMAStrategy
                strategy = SMAStrategy()
                signal = strategy.generate_signal(df, symbol)
                print(f"   SMA Signal: {signal.signal.value} (confidence: {signal.confidence:.2f})")
            else:
                print(f"❌ {symbol}: No data available")
                
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")

if __name__ == "__main__":
    print("🎯 Cryptocurrency Trading Strategy Testing")
    print("=" * 60)
    
    # Test 1: Individual strategies
    test_strategies()
    
    # Test 2: Sample backtest
    run_sample_backtest()
    
    # Test 3: Real data test
    test_yfinance_data()
    
    print("\n🎉 All tests completed!")