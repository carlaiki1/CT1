"""
AI-Powered Trading Agent with Multiple Strategies
Supports SMA, RSI, MACD, and Bollinger Bands strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass
from enum import Enum

# Manual implementation of technical indicators
def sma(data, window):
    """Simple Moving Average"""
    return data.rolling(window=window).mean()

def ema(data, window):
    """Exponential Moving Average"""
    return data.ewm(span=window).mean()

def rsi(data, window=14):
    """Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(data, fast=12, slow=26, signal=9):
    """MACD indicator"""
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(data, window=20, std_dev=2):
    """Bollinger Bands"""
    middle = sma(data, window)
    std = data.rolling(window=window).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    timestamp: datetime
    reason: str

@dataclass
class Position:
    symbol: str
    side: str
    amount: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percentage: float
    timestamp: datetime

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.signals_history = []
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """Generate trading signal - to be implemented by subclasses"""
        raise NotImplementedError
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators - to be implemented by subclasses"""
        raise NotImplementedError

class SMAStrategy(TradingStrategy):
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__("SMA_Crossover")
        self.short_window = short_window
        self.long_window = long_window
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA indicators"""
        df = df.copy()
        df['sma_short'] = sma(df['close'], self.short_window)
        df['sma_long'] = sma(df['close'], self.long_window)
        df['sma_signal'] = np.where(df['sma_short'] > df['sma_long'], 1, -1)
        df['sma_position'] = df['sma_signal'].diff()
        return df
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """Generate SMA crossover signal"""
        try:
            df = self.calculate_indicators(df)
            
            if len(df) < self.long_window:
                return TradingSignal(
                    symbol=symbol,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    price=df['close'].iloc[-1],
                    strategy=self.name,
                    timestamp=datetime.now(),
                    reason="Insufficient data for SMA calculation"
                )
            
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            current_price = latest['close']
            sma_short = latest['sma_short']
            sma_long = latest['sma_long']
            
            # Calculate confidence based on distance between SMAs
            sma_diff = abs(sma_short - sma_long) / sma_long
            confidence = min(sma_diff * 10, 1.0)  # Scale to 0-1
            
            # Generate signal
            if latest['sma_position'] > 0:  # Bullish crossover
                signal = SignalType.BUY
                reason = f"SMA bullish crossover: {sma_short:.2f} > {sma_long:.2f}"
            elif latest['sma_position'] < 0:  # Bearish crossover
                signal = SignalType.SELL
                reason = f"SMA bearish crossover: {sma_short:.2f} < {sma_long:.2f}"
            else:
                signal = SignalType.HOLD
                reason = f"No crossover: SMA_short={sma_short:.2f}, SMA_long={sma_long:.2f}"
            
            return TradingSignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                price=current_price,
                strategy=self.name,
                timestamp=datetime.now(),
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating SMA signal for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.HOLD,
                confidence=0.0,
                price=df['close'].iloc[-1] if not df.empty else 0,
                strategy=self.name,
                timestamp=datetime.now(),
                reason=f"Error: {str(e)}"
            )

class RSIStrategy(TradingStrategy):
    """RSI Overbought/Oversold Strategy"""
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        super().__init__("RSI")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator"""
        df = df.copy()
        df['rsi'] = rsi(df['close'], self.period)
        return df
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """Generate RSI signal"""
        try:
            df = self.calculate_indicators(df)
            
            if len(df) < self.period:
                return TradingSignal(
                    symbol=symbol,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    price=df['close'].iloc[-1],
                    strategy=self.name,
                    timestamp=datetime.now(),
                    reason="Insufficient data for RSI calculation"
                )
            
            latest = df.iloc[-1]
            current_price = latest['close']
            rsi = latest['rsi']
            
            # Calculate confidence based on RSI extremes
            if rsi <= self.oversold:
                confidence = (self.oversold - rsi) / self.oversold
                signal = SignalType.BUY
                reason = f"RSI oversold: {rsi:.2f} <= {self.oversold}"
            elif rsi >= self.overbought:
                confidence = (rsi - self.overbought) / (100 - self.overbought)
                signal = SignalType.SELL
                reason = f"RSI overbought: {rsi:.2f} >= {self.overbought}"
            else:
                confidence = 0.0
                signal = SignalType.HOLD
                reason = f"RSI neutral: {rsi:.2f}"
            
            return TradingSignal(
                symbol=symbol,
                signal=signal,
                confidence=min(confidence, 1.0),
                price=current_price,
                strategy=self.name,
                timestamp=datetime.now(),
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating RSI signal for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.HOLD,
                confidence=0.0,
                price=df['close'].iloc[-1] if not df.empty else 0,
                strategy=self.name,
                timestamp=datetime.now(),
                reason=f"Error: {str(e)}"
            )

class MACDStrategy(TradingStrategy):
    """MACD Strategy"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators"""
        df = df.copy()
        macd_line, macd_signal, macd_hist = macd(
            df['close'], 
            fast=self.fast_period,
            slow=self.slow_period,
            signal=self.signal_period
        )
        df['macd'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        df['macd_position'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        df['macd_crossover'] = df['macd_position'].diff()
        return df
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """Generate MACD signal"""
        try:
            df = self.calculate_indicators(df)
            
            min_periods = self.slow_period + self.signal_period
            if len(df) < min_periods:
                return TradingSignal(
                    symbol=symbol,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    price=df['close'].iloc[-1],
                    strategy=self.name,
                    timestamp=datetime.now(),
                    reason="Insufficient data for MACD calculation"
                )
            
            latest = df.iloc[-1]
            current_price = latest['close']
            macd = latest['macd']
            macd_signal = latest['macd_signal']
            macd_hist = latest['macd_hist']
            
            # Calculate confidence based on MACD histogram strength
            confidence = min(abs(macd_hist) / abs(macd) if macd != 0 else 0, 1.0)
            
            # Generate signal
            if latest['macd_crossover'] > 0:  # Bullish crossover
                signal = SignalType.BUY
                reason = f"MACD bullish crossover: {macd:.4f} > {macd_signal:.4f}"
            elif latest['macd_crossover'] < 0:  # Bearish crossover
                signal = SignalType.SELL
                reason = f"MACD bearish crossover: {macd:.4f} < {macd_signal:.4f}"
            else:
                signal = SignalType.HOLD
                reason = f"No MACD crossover: MACD={macd:.4f}, Signal={macd_signal:.4f}"
            
            return TradingSignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                price=current_price,
                strategy=self.name,
                timestamp=datetime.now(),
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating MACD signal for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.HOLD,
                confidence=0.0,
                price=df['close'].iloc[-1] if not df.empty else 0,
                strategy=self.name,
                timestamp=datetime.now(),
                reason=f"Error: {str(e)}"
            )

class BollingerBandsStrategy(TradingStrategy):
    """Bollinger Bands Strategy"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("Bollinger_Bands")
        self.period = period
        self.std_dev = std_dev
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands indicators"""
        df = df.copy()
        upper, middle, lower = bollinger_bands(
            df['close'], 
            window=self.period,
            std_dev=self.std_dev
        )
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        df['bb_position'] = (df['close'] - lower) / (upper - lower)
        return df
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """Generate Bollinger Bands signal"""
        try:
            df = self.calculate_indicators(df)
            
            if len(df) < self.period:
                return TradingSignal(
                    symbol=symbol,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    price=df['close'].iloc[-1],
                    strategy=self.name,
                    timestamp=datetime.now(),
                    reason="Insufficient data for Bollinger Bands calculation"
                )
            
            latest = df.iloc[-1]
            current_price = latest['close']
            bb_upper = latest['bb_upper']
            bb_lower = latest['bb_lower']
            bb_position = latest['bb_position']
            
            # Generate signal based on band position
            if bb_position <= 0.1:  # Near lower band
                confidence = (0.1 - bb_position) / 0.1
                signal = SignalType.BUY
                reason = f"Price near lower BB: {current_price:.2f} <= {bb_lower:.2f}"
            elif bb_position >= 0.9:  # Near upper band
                confidence = (bb_position - 0.9) / 0.1
                signal = SignalType.SELL
                reason = f"Price near upper BB: {current_price:.2f} >= {bb_upper:.2f}"
            else:
                confidence = 0.0
                signal = SignalType.HOLD
                reason = f"Price within BB: position={bb_position:.2f}"
            
            return TradingSignal(
                symbol=symbol,
                signal=signal,
                confidence=min(confidence, 1.0),
                price=current_price,
                strategy=self.name,
                timestamp=datetime.now(),
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating Bollinger Bands signal for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol,
                signal=SignalType.HOLD,
                confidence=0.0,
                price=df['close'].iloc[-1] if not df.empty else 0,
                strategy=self.name,
                timestamp=datetime.now(),
                reason=f"Error: {str(e)}"
            )

class TradingAgent:
    """Main Trading Agent that combines multiple strategies"""
    
    def __init__(self, exchange_adapter, config: Dict = None):
        self.exchange = exchange_adapter
        self.config = config or {}
        self.strategies = self._initialize_strategies()
        self.positions = {}
        self.trade_history = []
        self.performance_metrics = {}
        
        # Risk management settings
        self.max_positions = self.config.get('max_positions', 5)
        self.risk_percentage = self.config.get('risk_percentage', 2.0)
        self.stop_loss_percentage = self.config.get('stop_loss_percentage', 5.0)
        self.take_profit_percentage = self.config.get('take_profit_percentage', 10.0)
        
        logger.info(f"🤖 Trading Agent initialized with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self) -> List[TradingStrategy]:
        """Initialize all trading strategies"""
        strategies = [
            SMAStrategy(short_window=20, long_window=50),
            RSIStrategy(period=14, overbought=70, oversold=30),
            MACDStrategy(fast_period=12, slow_period=26, signal_period=9),
            BollingerBandsStrategy(period=20, std_dev=2.0)
        ]
        return strategies
    
    def analyze_symbol(self, symbol: str, timeframe: str = '1d', limit: int = 100) -> Dict:
        """Analyze a symbol using all strategies"""
        try:
            # Get historical data
            df = self.exchange.get_historical_data(symbol, timeframe, limit)
            
            if df.empty:
                logger.warning(f"⚠️ No data available for {symbol}")
                return {'symbol': symbol, 'signals': [], 'error': 'No data available'}
            
            # Generate signals from all strategies
            signals = []
            for strategy in self.strategies:
                signal = strategy.generate_signal(df, symbol)
                signals.append({
                    'strategy': signal.strategy,
                    'signal': signal.signal.value,
                    'confidence': signal.confidence,
                    'reason': signal.reason,
                    'timestamp': signal.timestamp.isoformat()
                })
            
            # Calculate consensus
            consensus = self._calculate_consensus(signals)
            
            # Get current price
            ticker = self.exchange.get_ticker(symbol)
            current_price = ticker.get('price', 0)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'signals': signals,
                'consensus': consensus,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return {'symbol': symbol, 'signals': [], 'error': str(e)}
    
    def _calculate_consensus(self, signals: List[Dict]) -> Dict:
        """Calculate consensus from multiple strategy signals"""
        buy_signals = [s for s in signals if s['signal'] == 'BUY']
        sell_signals = [s for s in signals if s['signal'] == 'SELL']
        hold_signals = [s for s in signals if s['signal'] == 'HOLD']
        
        buy_confidence = sum(s['confidence'] for s in buy_signals) / len(signals)
        sell_confidence = sum(s['confidence'] for s in sell_signals) / len(signals)
        
        if buy_confidence > sell_confidence and buy_confidence > 0.3:
            consensus_signal = 'BUY'
            consensus_confidence = buy_confidence
        elif sell_confidence > buy_confidence and sell_confidence > 0.3:
            consensus_signal = 'SELL'
            consensus_confidence = sell_confidence
        else:
            consensus_signal = 'HOLD'
            consensus_confidence = 0.0
        
        return {
            'signal': consensus_signal,
            'confidence': consensus_confidence,
            'buy_count': len(buy_signals),
            'sell_count': len(sell_signals),
            'hold_count': len(hold_signals)
        }
    
    def execute_trade(self, symbol: str, signal: str, confidence: float, amount: float = None) -> Dict:
        """Execute a trade based on signal"""
        try:
            if not amount:
                # Calculate position size based on risk management
                balance = self.exchange.get_account_balance()
                usd_balance = balance.get('USD', {}).get('free', 0)
                amount = (usd_balance * self.risk_percentage / 100) / self.exchange.get_ticker(symbol)['price']
            
            # Check if we already have a position
            if symbol in self.positions:
                logger.info(f"⚠️ Already have position in {symbol}")
                return {'status': 'skipped', 'reason': 'Position already exists'}
            
            # Check maximum positions limit
            if len(self.positions) >= self.max_positions:
                logger.info(f"⚠️ Maximum positions limit reached ({self.max_positions})")
                return {'status': 'skipped', 'reason': 'Maximum positions limit reached'}
            
            # Execute the trade
            if signal == 'BUY':
                order = self.exchange.place_order(symbol, 'buy', amount, 'market')
            elif signal == 'SELL':
                order = self.exchange.place_order(symbol, 'sell', amount, 'market')
            else:
                return {'status': 'skipped', 'reason': 'HOLD signal'}
            
            if order:
                # Record the position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=signal.lower(),
                    amount=amount,
                    entry_price=order.get('price', 0),
                    current_price=order.get('price', 0),
                    pnl=0.0,
                    pnl_percentage=0.0,
                    timestamp=datetime.now()
                )
                
                # Record trade history
                self.trade_history.append({
                    'symbol': symbol,
                    'side': signal.lower(),
                    'amount': amount,
                    'price': order.get('price', 0),
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat(),
                    'order_id': order.get('id')
                })
                
                logger.info(f"✅ Executed {signal} order for {symbol}: {amount} @ {order.get('price', 0)}")
                return {'status': 'executed', 'order': order}
            
        except Exception as e:
            logger.error(f"❌ Error executing trade for {symbol}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def update_positions(self):
        """Update all open positions with current prices and P&L"""
        for symbol, position in self.positions.items():
            try:
                ticker = self.exchange.get_ticker(symbol)
                current_price = ticker.get('price', position.current_price)
                
                position.current_price = current_price
                
                if position.side == 'buy':
                    position.pnl = (current_price - position.entry_price) * position.amount
                    position.pnl_percentage = ((current_price - position.entry_price) / position.entry_price) * 100
                else:  # sell
                    position.pnl = (position.entry_price - current_price) * position.amount
                    position.pnl_percentage = ((position.entry_price - current_price) / position.entry_price) * 100
                
            except Exception as e:
                logger.error(f"❌ Error updating position for {symbol}: {e}")
    
    def check_exit_conditions(self):
        """Check if any positions should be closed based on stop-loss or take-profit"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            # Check stop-loss
            if position.pnl_percentage <= -self.stop_loss_percentage:
                positions_to_close.append((symbol, 'stop_loss'))
            
            # Check take-profit
            elif position.pnl_percentage >= self.take_profit_percentage:
                positions_to_close.append((symbol, 'take_profit'))
        
        # Close positions
        for symbol, reason in positions_to_close:
            self.close_position(symbol, reason)
    
    def close_position(self, symbol: str, reason: str = 'manual'):
        """Close a position"""
        try:
            if symbol not in self.positions:
                return {'status': 'error', 'error': 'Position not found'}
            
            position = self.positions[symbol]
            
            # Execute opposite order
            opposite_side = 'sell' if position.side == 'buy' else 'buy'
            order = self.exchange.place_order(symbol, opposite_side, position.amount, 'market')
            
            if order:
                # Record the closing trade
                duration = datetime.now() - position.timestamp
                self.trade_history.append({
                    'symbol': symbol,
                    'side': opposite_side,
                    'amount': position.amount,
                    'price': order.get('price', 0),
                    'reason': reason,
                    'pnl': position.pnl,
                    'pnl_percentage': position.pnl_percentage,
                    'duration_seconds': duration.total_seconds(),
                    'timestamp': datetime.now().isoformat(),
                    'order_id': order.get('id')
                })
                
                # Remove position
                del self.positions[symbol]
                
                logger.info(f"✅ Closed position for {symbol} ({reason}): P&L = {position.pnl:.2f} ({position.pnl_percentage:.2f}%)")
                return {'status': 'closed', 'pnl': position.pnl, 'pnl_percentage': position.pnl_percentage}
            
        except Exception as e:
            logger.error(f"❌ Error closing position for {symbol}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if not self.trade_history:
            return {'total_trades': 0, 'total_pnl': 0, 'win_rate': 0}
        
        closed_trades = [t for t in self.trade_history if 'pnl' in t]
        
        if not closed_trades:
            return {'total_trades': len(self.trade_history), 'total_pnl': 0, 'win_rate': 0}
        
        total_pnl = sum(t['pnl'] for t in closed_trades)
        winning_trades = len([t for t in closed_trades if t['pnl'] > 0])
        win_rate = (winning_trades / len(closed_trades)) * 100 if closed_trades else 0
        
        return {
            'total_trades': len(self.trade_history),
            'closed_trades': len(closed_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'average_pnl': total_pnl / len(closed_trades) if closed_trades else 0,
            'open_positions': len(self.positions)
        }

if __name__ == "__main__":
    # Test the trading strategies
    print("🧪 Testing Trading Strategies...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    df = pd.DataFrame({
        'close': prices,
        'open': prices * (1 + np.random.randn(len(dates)) * 0.001),
        'high': prices * (1 + abs(np.random.randn(len(dates))) * 0.002),
        'low': prices * (1 - abs(np.random.randn(len(dates))) * 0.002),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Test each strategy
    strategies = [
        SMAStrategy(),
        RSIStrategy(),
        MACDStrategy(),
        BollingerBandsStrategy()
    ]
    
    for strategy in strategies:
        signal = strategy.generate_signal(df, 'TEST/USD')
        print(f"📊 {strategy.name}: {signal.signal.value} (confidence: {signal.confidence:.2f})")
        print(f"   Reason: {signal.reason}")
        print()