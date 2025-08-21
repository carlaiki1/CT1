"""
Comprehensive Backtesting Engine for Cryptocurrency Trading Strategies
Tests strategies on top 25 cryptocurrencies with up to 3 years of historical data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from exchange_adapter import ExchangeAdapter
from trading_agent import SMAStrategy, RSIStrategy, MACDStrategy, BollingerBandsStrategy, SignalType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.exchange_adapter = ExchangeAdapter(exchange_name='coinbase', demo_mode=True)
        
        # Initialize strategies
        self.strategies = {
            'SMA': SMAStrategy(short_window=20, long_window=50),
            'RSI': RSIStrategy(period=14, overbought=70, oversold=30),
            'MACD': MACDStrategy(fast_period=12, slow_period=26, signal_period=9),
            'Bollinger': BollingerBandsStrategy(period=20, std_dev=2.0)
        }
        
        self.results = {}
        
    def fetch_historical_data(self, symbols: List[str], years: int = 3) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for multiple symbols"""
        logger.info(f"📊 Fetching historical data for {len(symbols)} symbols...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        data = {}
        
        for symbol in tqdm(symbols, desc="Fetching data"):
            try:
                # Use ExchangeAdapter to get historical data
                df = self.exchange_adapter.get_historical_data(symbol, '1d', years * 365)
                
                if not df.empty and len(df) > 100:  # Minimum data requirement
                    data[symbol] = df
                    logger.info(f"✅ {symbol}: {len(df)} days of data")
                else:
                    logger.warning(f"⚠️ {symbol}: Insufficient data ({len(df)} days)")
                    
            except Exception as e:
                logger.error(f"❌ Error fetching data for {symbol}: {e}")
                
        logger.info(f"📈 Successfully fetched data for {len(data)} symbols")
        return data
    
    def backtest_strategy(self, strategy, df: pd.DataFrame, symbol: str) -> Dict:
        """Backtest a single strategy on historical data"""
        try:
            # Calculate indicators
            df_with_indicators = strategy.calculate_indicators(df.copy())
            
            # Initialize tracking variables
            capital = self.initial_capital
            position = 0  # 0 = no position, 1 = long, -1 = short
            entry_price = 0
            trades = []
            equity_curve = []
            
            for i in range(len(df_with_indicators)):
                current_data = df_with_indicators.iloc[:i+1]
                
                if len(current_data) < 50:  # Skip early periods
                    equity_curve.append(capital)
                    continue
                
                # Generate signal
                signal = strategy.generate_signal(current_data, symbol)
                current_price = current_data['close'].iloc[-1]
                
                # Execute trades based on signals
                if signal.signal == SignalType.BUY and position <= 0 and signal.confidence > 0.3:
                    # Close short position if any
                    if position == -1:
                        pnl = (entry_price - current_price) * abs(position) * capital / entry_price
                        capital += pnl - (capital * self.commission)
                        trades.append({
                            'type': 'close_short',
                            'price': current_price,
                            'pnl': pnl,
                            'date': current_data.index[-1],
                            'confidence': signal.confidence
                        })
                    
                    # Open long position
                    position = 1
                    entry_price = current_price
                    capital -= capital * self.commission  # Commission
                    trades.append({
                        'type': 'buy',
                        'price': current_price,
                        'date': current_data.index[-1],
                        'confidence': signal.confidence
                    })
                
                elif signal.signal == SignalType.SELL and position >= 0 and signal.confidence > 0.3:
                    # Close long position if any
                    if position == 1:
                        pnl = (current_price - entry_price) * position * capital / entry_price
                        capital += pnl - (capital * self.commission)
                        trades.append({
                            'type': 'close_long',
                            'price': current_price,
                            'pnl': pnl,
                            'date': current_data.index[-1],
                            'confidence': signal.confidence
                        })
                    
                    # Open short position
                    position = -1
                    entry_price = current_price
                    capital -= capital * self.commission  # Commission
                    trades.append({
                        'type': 'sell',
                        'price': current_price,
                        'date': current_data.index[-1],
                        'confidence': signal.confidence
                    })
                
                # Calculate current equity
                if position != 0:
                    unrealized_pnl = (current_price - entry_price) * position * capital / entry_price
                    current_equity = capital + unrealized_pnl
                else:
                    current_equity = capital
                
                equity_curve.append(current_equity)
            
            # Close final position
            if position != 0:
                final_price = df_with_indicators['close'].iloc[-1]
                if position == 1:
                    pnl = (final_price - entry_price) * position * capital / entry_price
                else:
                    pnl = (entry_price - final_price) * abs(position) * capital / entry_price
                
                capital += pnl - (capital * self.commission)
                trades.append({
                    'type': 'close_final',
                    'price': final_price,
                    'pnl': pnl,
                    'date': df_with_indicators.index[-1],
                    'confidence': 0.5
                })
            
            # Calculate performance metrics
            total_return = (capital - self.initial_capital) / self.initial_capital * 100
            
            # Calculate additional metrics
            equity_series = pd.Series(equity_curve, index=df_with_indicators.index)
            returns = equity_series.pct_change().dropna()
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Maximum drawdown
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Win rate
            profitable_trades = [t for t in trades if 'pnl' in t and t['pnl'] > 0]
            total_closed_trades = [t for t in trades if 'pnl' in t]
            win_rate = len(profitable_trades) / len(total_closed_trades) * 100 if total_closed_trades else 0
            
            # Average trade
            avg_trade = np.mean([t['pnl'] for t in total_closed_trades]) if total_closed_trades else 0
            
            return {
                'symbol': symbol,
                'strategy': strategy.name,
                'total_return': total_return,
                'final_capital': capital,
                'total_trades': len(trades),
                'closed_trades': len(total_closed_trades),
                'win_rate': win_rate,
                'avg_trade': avg_trade,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trades': trades,
                'equity_curve': equity_curve,
                'dates': df_with_indicators.index.tolist()
            }
            
        except Exception as e:
            logger.error(f"❌ Error backtesting {strategy.name} on {symbol}: {e}")
            return {
                'symbol': symbol,
                'strategy': strategy.name,
                'error': str(e),
                'total_return': 0,
                'final_capital': self.initial_capital
            }
    
    def run_comprehensive_backtest(self, symbols: List[str] = None, years: int = 3) -> Dict:
        """Run comprehensive backtest on multiple symbols and strategies"""
        logger.info("🚀 Starting comprehensive backtest...")
        
        # Get top cryptocurrencies if not provided
        if symbols is None:
            symbols = [
                'BTC/USD', 'ETH/USD', 'ADA/USD', 'SOL/USD', 'XRP/USD',
                'DOGE/USD', 'AVAX/USD', 'SHIB/USD', 'MATIC/USD', 'LTC/USD',
                'UNI/USD', 'LINK/USD', 'ALGO/USD', 'BCH/USD', 'XLM/USD'
            ]
        
        # Fetch historical data
        historical_data = self.fetch_historical_data(symbols, years)
        
        if not historical_data:
            logger.error("❌ No historical data available")
            return {}
        
        # Run backtests
        results = {}
        total_combinations = len(historical_data) * len(self.strategies)
        
        with tqdm(total=total_combinations, desc="Running backtests") as pbar:
            for symbol, df in historical_data.items():
                results[symbol] = {}
                
                for strategy_name, strategy in self.strategies.items():
                    result = self.backtest_strategy(strategy, df, symbol)
                    results[symbol][strategy_name] = result
                    pbar.update(1)
        
        self.results = results
        return results
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.results:
            logger.error("❌ No backtest results available")
            return {}
        
        logger.info("📊 Generating performance report...")
        
        # Aggregate results
        strategy_performance = {}
        symbol_performance = {}
        
        for symbol, strategies in self.results.items():
            symbol_performance[symbol] = {}
            
            for strategy_name, result in strategies.items():
                if 'error' in result:
                    continue
                
                # Strategy performance
                if strategy_name not in strategy_performance:
                    strategy_performance[strategy_name] = {
                        'total_returns': [],
                        'win_rates': [],
                        'sharpe_ratios': [],
                        'max_drawdowns': [],
                        'avg_trades': [],
                        'symbols_tested': 0,
                        'profitable_symbols': 0
                    }
                
                strategy_performance[strategy_name]['total_returns'].append(result['total_return'])
                strategy_performance[strategy_name]['win_rates'].append(result['win_rate'])
                strategy_performance[strategy_name]['sharpe_ratios'].append(result['sharpe_ratio'])
                strategy_performance[strategy_name]['max_drawdowns'].append(result['max_drawdown'])
                strategy_performance[strategy_name]['avg_trades'].append(result['avg_trade'])
                strategy_performance[strategy_name]['symbols_tested'] += 1
                
                if result['total_return'] > 0:
                    strategy_performance[strategy_name]['profitable_symbols'] += 1
                
                # Symbol performance
                symbol_performance[symbol][strategy_name] = result['total_return']
        
        # Calculate summary statistics
        summary = {}
        for strategy_name, perf in strategy_performance.items():
            summary[strategy_name] = {
                'avg_return': np.mean(perf['total_returns']),
                'median_return': np.median(perf['total_returns']),
                'std_return': np.std(perf['total_returns']),
                'best_return': np.max(perf['total_returns']),
                'worst_return': np.min(perf['total_returns']),
                'avg_win_rate': np.mean(perf['win_rates']),
                'avg_sharpe': np.mean(perf['sharpe_ratios']),
                'avg_max_drawdown': np.mean(perf['max_drawdowns']),
                'avg_trade_pnl': np.mean(perf['avg_trades']),
                'symbols_tested': perf['symbols_tested'],
                'profitable_symbols': perf['profitable_symbols'],
                'success_rate': perf['profitable_symbols'] / perf['symbols_tested'] * 100
            }
        
        return {
            'summary': summary,
            'detailed_results': self.results,
            'symbol_performance': symbol_performance,
            'strategy_performance': strategy_performance
        }
    
    def save_results(self, filename: str = None):
        """Save backtest results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results/backtest_results_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Prepare data for JSON serialization
        json_results = {}
        for symbol, strategies in self.results.items():
            json_results[symbol] = {}
            for strategy_name, result in strategies.items():
                # Convert datetime objects to strings
                json_result = result.copy()
                if 'dates' in json_result:
                    json_result['dates'] = [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in json_result['dates']]
                if 'trades' in json_result:
                    for trade in json_result['trades']:
                        if 'date' in trade:
                            trade['date'] = trade['date'].isoformat() if hasattr(trade['date'], 'isoformat') else str(trade['date'])
                
                json_results[symbol][strategy_name] = json_result
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {filename}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if not self.results:
            logger.error("❌ No results to visualize")
            return
        
        logger.info("📈 Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Strategy Performance Comparison
        ax1 = plt.subplot(2, 3, 1)
        strategy_returns = {}
        for symbol, strategies in self.results.items():
            for strategy_name, result in strategies.items():
                if 'error' not in result:
                    if strategy_name not in strategy_returns:
                        strategy_returns[strategy_name] = []
                    strategy_returns[strategy_name].append(result['total_return'])
        
        strategy_names = list(strategy_returns.keys())
        avg_returns = [np.mean(strategy_returns[name]) for name in strategy_names]
        
        bars = ax1.bar(strategy_names, avg_returns, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('Average Returns by Strategy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_returns):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Win Rate Comparison
        ax2 = plt.subplot(2, 3, 2)
        strategy_win_rates = {}
        for symbol, strategies in self.results.items():
            for strategy_name, result in strategies.items():
                if 'error' not in result:
                    if strategy_name not in strategy_win_rates:
                        strategy_win_rates[strategy_name] = []
                    strategy_win_rates[strategy_name].append(result['win_rate'])
        
        avg_win_rates = [np.mean(strategy_win_rates[name]) for name in strategy_names]
        
        bars = ax2.bar(strategy_names, avg_win_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_title('Average Win Rate by Strategy', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Win Rate (%)')
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, avg_win_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Return Distribution
        ax3 = plt.subplot(2, 3, 3)
        all_returns = []
        strategy_labels = []
        
        for strategy_name in strategy_names:
            returns = strategy_returns[strategy_name]
            all_returns.extend(returns)
            strategy_labels.extend([strategy_name] * len(returns))
        
        df_returns = pd.DataFrame({'Strategy': strategy_labels, 'Returns': all_returns})
        sns.boxplot(data=df_returns, x='Strategy', y='Returns', ax=ax3)
        ax3.set_title('Return Distribution by Strategy', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Returns (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Top Performing Symbols
        ax4 = plt.subplot(2, 3, 4)
        symbol_avg_returns = {}
        for symbol, strategies in self.results.items():
            returns = []
            for strategy_name, result in strategies.items():
                if 'error' not in result:
                    returns.append(result['total_return'])
            if returns:
                symbol_avg_returns[symbol] = np.mean(returns)
        
        # Get top 10 symbols
        top_symbols = sorted(symbol_avg_returns.items(), key=lambda x: x[1], reverse=True)[:10]
        symbols, returns = zip(*top_symbols)
        
        bars = ax4.barh(range(len(symbols)), returns, color='green', alpha=0.7)
        ax4.set_yticks(range(len(symbols)))
        ax4.set_yticklabels([s.replace('/USD', '') for s in symbols])
        ax4.set_title('Top 10 Performing Cryptocurrencies', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Average Return (%)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Sharpe Ratio Comparison
        ax5 = plt.subplot(2, 3, 5)
        strategy_sharpe = {}
        for symbol, strategies in self.results.items():
            for strategy_name, result in strategies.items():
                if 'error' not in result:
                    if strategy_name not in strategy_sharpe:
                        strategy_sharpe[strategy_name] = []
                    strategy_sharpe[strategy_name].append(result['sharpe_ratio'])
        
        avg_sharpe = [np.mean(strategy_sharpe[name]) for name in strategy_names]
        
        bars = ax5.bar(strategy_names, avg_sharpe, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax5.set_title('Average Sharpe Ratio by Strategy', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, avg_sharpe):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Maximum Drawdown
        ax6 = plt.subplot(2, 3, 6)
        strategy_drawdown = {}
        for symbol, strategies in self.results.items():
            for strategy_name, result in strategies.items():
                if 'error' not in result:
                    if strategy_name not in strategy_drawdown:
                        strategy_drawdown[strategy_name] = []
                    strategy_drawdown[strategy_name].append(abs(result['max_drawdown']))
        
        avg_drawdown = [np.mean(strategy_drawdown[name]) for name in strategy_names]
        
        bars = ax6.bar(strategy_names, avg_drawdown, color='red', alpha=0.7)
        ax6.set_title('Average Maximum Drawdown by Strategy', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Max Drawdown (%)')
        ax6.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, avg_drawdown):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results/backtest_visualization_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Visualization saved to {filename}")
        
        plt.show()

def main():
    """Main function to run comprehensive backtest"""
    print("🚀 Cryptocurrency Trading Strategy Backtest")
    print("=" * 50)
    
    # Initialize backtest engine
    engine = BacktestEngine(initial_capital=10000, commission=0.005)
    
    # Run comprehensive backtest
    print("📊 Fetching top 25 cryptocurrencies and running backtest...")
    results = engine.run_comprehensive_backtest(years=3)
    
    if not results:
        print("❌ No results generated")
        return
    
    # Generate performance report
    report = engine.generate_performance_report()
    
    # Print summary
    print("\n📈 BACKTEST RESULTS SUMMARY")
    print("=" * 50)
    
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
    engine.save_results()
    
    # Create visualizations
    engine.create_visualizations()
    
    print(f"\n✅ Backtest completed successfully!")
    print(f"📁 Results saved in backtest_results/ directory")
    print(f"📊 Check the generated visualization and JSON files")

if __name__ == "__main__":
    main()