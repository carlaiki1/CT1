import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtest import BacktestEngine
from trading_agent import TradingAgent

@pytest.fixture
def sample_data():
    """Create sample cryptocurrency data for testing"""
    dates = pd.date_range(start='2021-08-01', end='2024-08-01', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)))
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(100, 1000, len(dates))
    }, index=dates)
    return {'TEST/USD': df}

class TestBacktestEngine:

    def test_run_comprehensive_backtest(self, sample_data):
        """Test the full backtesting process with sample data"""
        engine = BacktestEngine(initial_capital=10000, commission=0.005)
        
        # Mock the data fetching
        engine.fetch_historical_data = lambda symbols, years: sample_data
        
        results = engine.run_comprehensive_backtest(symbols=['TEST/USD'], years=1)
        
        assert results is not None
        assert 'TEST/USD' in results
        assert 'Combined' in results['TEST/USD']

        report = engine.generate_performance_report()

        assert report is not None
        assert 'summary' in report
        assert 'Combined' in report['summary']

        summary = report['summary']['Combined']
        assert summary['avg_return'] is not None

        # Check final capital in the detailed results
        final_capital = report['detailed_results']['TEST/USD']['Combined']['final_capital']
        assert final_capital > 0