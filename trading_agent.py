#
# trading_agent.py
#
"""
AI-Powered Trading Agent with Multiple Strategies
Supports SMA, RSI, MACD, and Bollinger Bands strategies with a weighted consensus model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import time
from dataclasses import dataclass, field
from enum import Enum
import talib

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Configuration ---
# All agent settings are here. This is the main place to make adjustments.
AGENT_CONFIG = {
    # --- Risk Management ---
    "max_open_positions": 5,          # The maximum number of different coins to hold at once.
    "risk_per_trade_percentage": 2.0, # The percentage of our total USD balance to risk on a single trade.
    "stop_loss_percentage": 5.0,      # Sell a position if it drops by this percentage.
    "take_profit_percentage": 10.0,   # Sell a position if it rises by this percentage.

    # --- Strategy Weights for Consensus Model ---
    # The agent combines signals from all strategies. These weights decide how much
    # influence each strategy has on the final trading decision.
    "strategy_weights": {
        "SMA_Crossover": 1.0,         # A strong trend-following indicator.
        "MACD": 1.0,                  # Another strong trend and momentum indicator.
        "RSI": 0.75,                  # Good for confirmation but can be noisy.
        "Bollinger_Bands": 0.5        # More of a volatility/reversal indicator, given less weight.
    },
    
    # --- Strategy-Specific Parameters ---
    "sma_short_window": 20,
    "sma_long_window": 50,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "macd_fast_period": 12,
    "macd_slow_period": 26,
    "macd_signal_period": 9,
    "bollinger_period": 20,
    "bollinger_std_dev": 2.0
}


# --- Data Models ---
class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    symbol: str
    signal: SignalType
    confidence: float
    price: float
    strategy: str
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""

@dataclass
class Position:
    symbol: str
    side: str
    amount: float
    entry_price: float
    current_price: float = 0.0
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


# --- Strategy Definitions ---
class TradingStrategy:
    """Base class for all trading strategies."""
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """
        Generates a trading signal. Assumes indicators are already in the DataFrame.
        This method must be implemented by each specific strategy.
        """
        raise NotImplementedError

class SMAStrategy(TradingStrategy):
    def __init__(self, config: Dict):
        super().__init__("SMA_Crossover", config)
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        price = latest['close']
        
        # Check for bullish crossover
        if previous['sma_short'] < previous['sma_long'] and latest['sma_short'] > latest['sma_long']:
            sma_diff = abs(latest['sma_short'] - latest['sma_long']) / latest['sma_long']
            confidence = min(sma_diff * 10, 1.0) # Scale confidence
            return TradingSignal(symbol, SignalType.BUY, confidence, price, self.name, reason=f"SMA Bullish Crossover")
            
        # Check for bearish crossover
        elif previous['sma_short'] > previous['sma_long'] and latest['sma_short'] < latest['sma_long']:
            sma_diff = abs(latest['sma_short'] - latest['sma_long']) / latest['sma_long']
            confidence = min(sma_diff * 10, 1.0)
            return TradingSignal(symbol, SignalType.SELL, confidence, price, self.name, reason=f"SMA Bearish Crossover")
            
        return TradingSignal(symbol, SignalType.HOLD, 0.0, price, self.name, reason="No Crossover")

class RSIStrategy(TradingStrategy):
    def __init__(self, config: Dict):
        super().__init__("RSI", config)
        self.overbought = self.config['rsi_overbought']
        self.oversold = self.config['rsi_oversold']
        
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        latest = df.iloc[-1]
        rsi = latest['rsi']
        price = latest['close']

        if rsi <= self.oversold:
            confidence = (self.oversold - rsi) / self.oversold
            return TradingSignal(symbol, SignalType.BUY, min(confidence, 1.0), price, self.name, reason=f"RSI Oversold ({rsi:.2f})")
            
        elif rsi >= self.overbought:
            confidence = (rsi - self.overbought) / (100 - self.overbought)
            return TradingSignal(symbol, SignalType.SELL, min(confidence, 1.0), price, self.name, reason=f"RSI Overbought ({rsi:.2f})")
            
        return TradingSignal(symbol, SignalType.HOLD, 0.0, price, self.name, reason=f"RSI Neutral ({rsi:.2f})")

class MACDStrategy(TradingStrategy):
    def __init__(self, config: Dict):
        super().__init__("MACD", config)
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        price = latest['close']

        # Check for bullish crossover
        if previous['macd'] < previous['macd_signal'] and latest['macd'] > latest['macd_signal']:
            confidence = min(abs(latest['macd_hist']) / abs(latest['macd']) if latest['macd'] != 0 else 0, 1.0)
            return TradingSignal(symbol, SignalType.BUY, confidence, price, self.name, reason="MACD Bullish Crossover")

        # Check for bearish crossover
        elif previous['macd'] > previous['macd_signal'] and latest['macd'] < latest['macd_signal']:
            confidence = min(abs(latest['macd_hist']) / abs(latest['macd']) if latest['macd'] != 0 else 0, 1.0)
            return TradingSignal(symbol, SignalType.SELL, confidence, price, self.name, reason="MACD Bearish Crossover")

        return TradingSignal(symbol, SignalType.HOLD, 0.0, price, self.name, reason="No MACD Crossover")

class BollingerBandsStrategy(TradingStrategy):
    def __init__(self, config: Dict):
        super().__init__("Bollinger_Bands", config)

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        latest = df.iloc[-1]
        price = latest['close']
        
        if price <= latest['bb_lower']:
            confidence = (latest['bb_lower'] - price) / (latest['bb_middle'] - latest['bb_lower'])
            return TradingSignal(symbol, SignalType.BUY, min(confidence, 1.0), price, self.name, reason="Price below lower band")

        elif price >= latest['bb_upper']:
            confidence = (price - latest['bb_upper']) / (latest['bb_upper'] - latest['bb_middle'])
            return TradingSignal(symbol, SignalType.SELL, min(confidence, 1.0), price, self.name, reason="Price above upper band")

        return TradingSignal(symbol, SignalType.HOLD, 0.0, price, self.name, reason="Price within bands")


# --- The Main Trading Agent ---
class TradingAgent:
    """The main trading agent that orchestrates strategies, risk, and execution."""
    
    def __init__(self, exchange_adapter, config: Dict = AGENT_CONFIG):
        self.exchange = exchange_adapter
        self.config = config
        self.strategies = self._initialize_strategies()
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        
        logger.info(f"🤖 Trading Agent initialized with {len(self.strategies)} strategies.")

    def _initialize_strategies(self) -> List[TradingStrategy]:
        """Loads all strategies defined in the config."""
        strategy_map = {
            "SMA_Crossover": SMAStrategy,
            "RSI": RSIStrategy,
            "MACD": MACDStrategy,
            "Bollinger_Bands": BollingerBandsStrategy,
        }
        
        initialized_strategies = []
        for name, weight in self.config['strategy_weights'].items():
            if weight > 0 and name in strategy_map:
                initialized_strategies.append(strategy_map[name](self.config))
        return initialized_strategies

    def _add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Efficiently calculates and adds all required indicators to a DataFrame."""
        df = df.copy()
        # SMA
        df['sma_short'] = talib.SMA(df['close'], timeperiod=self.config['sma_short_window'])
        df['sma_long'] = talib.SMA(df['close'], timeperiod=self.config['sma_long_window'])
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.config['rsi_period'])
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], 
            fastperiod=self.config['macd_fast_period'], 
            slowperiod=self.config['macd_slow_period'], 
            signalperiod=self.config['macd_signal_period']
        )
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], 
            timeperiod=self.config['bollinger_period'], 
            nbdevup=self.config['bollinger_std_dev'], 
            nbdevdn=self.config['bollinger_std_dev']
        )
        return df

    def _calculate_consensus(self, signals: List[TradingSignal]) -> TradingSignal:
        """Calculates a final signal based on a weighted average of all strategy signals."""
        buy_score = 0
        sell_score = 0
        total_weight = sum(self.config['strategy_weights'].get(s.strategy, 0) for s in signals)

        for s in signals:
            weight = self.config['strategy_weights'].get(s.strategy, 0)
            if s.signal == SignalType.BUY:
                buy_score += s.confidence * weight
            elif s.signal == SignalType.SELL:
                sell_score += s.confidence * weight
        
        # Normalize scores
        buy_confidence = buy_score / total_weight if total_weight > 0 else 0
        sell_confidence = sell_score / total_weight if total_weight > 0 else 0
        
        # Determine final signal
        if buy_confidence > sell_confidence and buy_confidence > 0.3: # Confidence threshold
            return TradingSignal(signals[0].symbol, SignalType.BUY, buy_confidence, signals[0].price, "Consensus")
        elif sell_confidence > buy_confidence and sell_confidence > 0.3:
            return TradingSignal(signals[0].symbol, SignalType.SELL, sell_confidence, signals[0].price, "Consensus")
        else:
            return TradingSignal(signals[0].symbol, SignalType.HOLD, 0.0, signals[0].price, "Consensus")

    def analyze_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """Performs a full analysis of a single symbol and returns a consensus signal."""
        try:
            df = self.exchange.get_historical_data(symbol, timeframe='1h', limit=200)
            if df.empty or len(df) < self.config['sma_long_window']:
                logger.warning(f"⚠️ Insufficient data for {symbol}.")
                return None
            
            df_with_indicators = self._add_all_indicators(df)
            df_with_indicators.dropna(inplace=True)
            
            if len(df_with_indicators) < 2:
                logger.warning(f"⚠️ Not enough data for {symbol} after indicator calculation.")
                return None
                
            all_signals = [strategy.generate_signal(df_with_indicators, symbol) for strategy in self.strategies]
            
            return self._calculate_consensus(all_signals)
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None

    def execute_trade(self, signal: TradingSignal):
        """Executes a trade based on risk management rules."""
        if len(self.positions) >= self.config['max_open_positions']:
            logger.warning("⚠️ Max open positions reached. Skipping new trade.")
            return

        balance = self.exchange.get_account_balance()
        usd_balance = balance.get('USD', {}).get('free', 0)
        
        if usd_balance < 100: # Minimum trade size
            logger.warning(f"⚠️ Insufficient USD balance ({usd_balance:.2f}) to trade.")
            return
            
        # Calculate amount to buy based on risk percentage
        investment_usd = usd_balance * (self.config['risk_per_trade_percentage'] / 100)
        amount_to_trade = investment_usd / signal.price

        logger.info(f"Attempting to place {signal.signal.value} order for {amount_to_trade:.6f} {signal.symbol}...")
        
        order = self.exchange.place_order(signal.symbol, signal.signal.value.lower(), amount_to_trade)

        if order and order.get('id'):
            self.positions[signal.symbol] = Position(
                symbol=signal.symbol,
                side=signal.signal.value.lower(),
                amount=amount_to_trade,
                entry_price=order.get('price', signal.price)
            )
            self.trade_history.append(order)
            logger.info(f"✅ Successfully opened {signal.signal.value} position for {signal.symbol}.")
        else:
            logger.error(f"❌ Failed to place order for {signal.symbol}.")

    def update_and_check_positions(self):
        """Updates P&L for open positions and checks for stop-loss or take-profit triggers."""
        positions_to_close = []
        for symbol, pos in self.positions.items():
            ticker = self.exchange.get_ticker(symbol)
            current_price = ticker.get('price', pos.current_price)
            pos.current_price = current_price
            
            # Update P&L
            pnl_mult = 1 if pos.side == 'buy' else -1
            pos.pnl = (current_price - pos.entry_price) * pos.amount * pnl_mult
            pos.pnl_percentage = (pos.pnl / (pos.entry_price * pos.amount)) * 100
            
            # Check exit conditions
            if pos.pnl_percentage <= -self.config['stop_loss_percentage']:
                positions_to_close.append((symbol, "Stop-Loss"))
            elif pos.pnl_percentage >= self.config['take_profit_percentage']:
                positions_to_close.append((symbol, "Take-Profit"))
        
        # Close positions that met exit criteria
        for symbol, reason in positions_to_close:
            self.close_position(symbol, reason)
    
    def close_position(self, symbol: str, reason: str):
        """Closes an open position."""
        if symbol not in self.positions:
            return
            
        pos = self.positions[symbol]
        opposite_side = 'sell' if pos.side == 'buy' else 'buy'
        
        logger.info(f"Attempting to close {symbol} position due to: {reason}. P&L: ${pos.pnl:.2f} ({pos.pnl_percentage:.2f}%)")
        order = self.exchange.place_order(symbol, opposite_side, pos.amount)

        if order and order.get('id'):
            del self.positions[symbol]
            self.trade_history.append(order)
            logger.info(f"✅ Successfully closed position for {symbol}.")
        else:
            logger.error(f"❌ Failed to close position for {symbol}.")
            
    def run(self, watchlist: List[str], interval_seconds: int = 900):
        """The main operational loop for the trading agent."""
        logger.info(f"🚀 Agent starting continuous run. Watchlist: {watchlist}, Interval: {interval_seconds}s")
        while True:
            try:
                logger.info("---  Cycle Start ---")
                
                # 1. Update and manage existing positions
                self.update_and_check_positions()
                
                # 2. Look for new trading opportunities
                for symbol in watchlist:
                    if symbol not in self.positions and len(self.positions) < self.config['max_open_positions']:
                        consensus_signal = self.analyze_symbol(symbol)
                        if consensus_signal and consensus_signal.signal != SignalType.HOLD:
                            logger.info(f"💡 New Signal: {consensus_signal.signal.value} for {consensus_signal.symbol} with confidence {consensus_signal.confidence:.2f}")
                            self.execute_trade(consensus_signal)
                
                logger.info(f"Open Positions: {len(self.positions)}")
                logger.info("--- Cycle End. Sleeping... ---")
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("🛑 Manual shutdown detected. Exiting.")
                break
            except Exception as e:
                logger.error(f"🚨 An unexpected error occurred in the run loop: {e}")
                time.sleep(60) # Wait a minute before retrying on major error
                
                
if __name__ == "__main__":
    # This block is for testing purposes. It simulates the ExchangeAdapter
    # so you can run this file directly to see if the agent's logic works.
    
    class MockExchangeAdapter:
        """A fake exchange adapter for testing the agent without real money."""
        def __init__(self):
            self.balance = {'USD': {'free': 5000.0, 'total': 5000.0}}
            self.order_id = 0

        def get_account_balance(self):
            return self.balance

        def get_historical_data(self, symbol, timeframe, limit):
            # Generate some fake, random price data
            np.random.seed(sum(map(ord, symbol))) # Consistent randomness for a symbol
            dates = pd.date_range(end=datetime.now(), periods=limit, freq='H')
            price = 100 + np.random.randint(-10, 10)
            close = price + np.random.randn(limit).cumsum()
            return pd.DataFrame({'close': close}, index=dates)
            
        def get_ticker(self, symbol):
            price = 100 + np.random.uniform(-5, 5)
            return {'symbol': symbol, 'price': price}
            
        def place_order(self, symbol, side, amount):
            self.order_id += 1
            price = self.get_ticker(symbol)['price']
            logger.info(f"[MOCK] Placing {side} order for {symbol} @ ${price:.2f}")
            # Simulate updating balance
            cost = price * amount
            if side == 'buy':
                self.balance['USD']['free'] -= cost
            else: # sell
                self.balance['USD']['free'] += cost
            return {'id': f'mock_{self.order_id}', 'price': price}

    logger.info("--- Starting Agent in Test Mode ---")
    
    # 1. Create the mock exchange and the agent
    mock_exchange = MockExchangeAdapter()
    agent = TradingAgent(exchange_adapter=mock_exchange, config=AGENT_CONFIG)
    
    # 2. Define a watchlist for the test run
    test_watchlist = ['BTC/USD', 'ETH/USD', 'SOL/USD']
    
    # 3. Start the agent's main loop
    # We use a short interval for testing. In a real scenario, this would be
    # 900 (15 mins), 3600 (1 hour), etc.
    agent.run(watchlist=test_watchlist, interval_seconds=15)