import pytest
from unittest.mock import MagicMock
import pandas as pd
from exchange_adapter import ExchangeAdapter
from trading_agent import TradingAgent

@pytest.fixture
def mock_exchange_adapter():
    """Fixture for a mocked ExchangeAdapter"""
    mock = MagicMock(spec=ExchangeAdapter)
    mock.get_account_balance.return_value = {
        'USD': {'free': 10000.0, 'used': 0.0, 'total': 10000.0}
    }
    mock.get_ticker.return_value = {
        'symbol': 'BTC/USD', 'price': 50000.0
    }
    mock.get_historical_data.return_value = pd.DataFrame({
        'open': [45000, 46000], 'high': [47000, 48000],
        'low': [44000, 45000], 'close': [46000, 47000],
        'volume': [1000, 1200]
    })
    return mock

class TestApi:

    def test_get_balance(self, mock_exchange_adapter):
        balance = mock_exchange_adapter.get_account_balance()
        assert balance is not None
        assert 'USD' in balance
        assert balance['USD']['total'] == 10000.0

    def test_get_ticker(self, mock_exchange_adapter):
        ticker = mock_exchange_adapter.get_ticker('BTC/USD')
        assert ticker is not None
        assert ticker['symbol'] == 'BTC/USD'
        assert ticker['price'] > 0

    def test_get_historical_data(self, mock_exchange_adapter):
        df = mock_exchange_adapter.get_historical_data('BTC/USD', limit=2)
        assert df is not None
        assert not df.empty
        assert len(df) == 2

    def test_analyze_symbol(self, mock_exchange_adapter):
        agent = TradingAgent(mock_exchange_adapter)
        analysis = agent.analyze_symbol('BTC/USD')
        assert analysis is not None
        assert 'signals' in analysis
        assert 'consensus' in analysis
        assert len(analysis['signals']) == 4 # One for each strategy