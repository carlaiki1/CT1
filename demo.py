"""
Demo Script for Cryptocurrency Trading Agent
Test the trading strategies without real money
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List

from exchange_adapter import ExchangeAdapter, CoinbaseDataFetcher
from trading_agent import TradingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoTrader:
    """Demo trading simulation"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.data_fetcher = CoinbaseDataFetcher()
        
        # Initialize demo exchange adapter
        self.exchange = ExchangeAdapter(
            exchange_name='coinbase',
            api_key='demo_key',
            api_secret='demo_secret',
            passphrase='demo_passphrase',
            demo_mode=True
        )
        
        # Initialize trading agent
        self.agent = TradingAgent(
            exchange_adapter=self.exchange,
            config={
                'max_positions': 3,
                'risk_percentage': 5.0,
                'stop_loss_percentage': 10.0,
                'take_profit_percentage': 20.0
            }
        )
        
        logger.info(f"🎮 Demo Trader initialized with ${initial_capital:,.2f}")
    
    def run_demo_session(self, duration_minutes: int = 30, symbols: List[str] = None):
        """Run a demo trading session"""
        logger.info(f"🚀 Starting {duration_minutes}-minute demo session...")
        
        if symbols is None:
            symbols = self.data_fetcher.get_top_volume_products(10)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        iteration = 0
        
        print("\n" + "="*80)
        print("🎮 CRYPTOCURRENCY TRADING AGENT - DEMO MODE")
        print("="*80)
        print(f"📅 Session Duration: {duration_minutes} minutes")
        print(f"💰 Starting Capital: ${self.initial_capital:,.2f}")
        print(f"📊 Analyzing Symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
        print("="*80)
        
        try:
            while datetime.now() < end_time:
                iteration += 1
                current_time = datetime.now()
                remaining_time = (end_time - current_time).total_seconds() / 60
                
                print(f"\n🔄 Iteration {iteration} - {remaining_time:.1f} minutes remaining")
                print("-" * 60)
                
                # Analyze symbols and generate signals
                for symbol in symbols[:5]:  # Limit to top 5 for demo
                    try:
                        analysis = self.agent.analyze_symbol(symbol, limit=50)
                        
                        if 'error' in analysis:
                            continue
                        
                        consensus = analysis.get('consensus', {})
                        signal = consensus.get('signal', 'HOLD')
                        confidence = consensus.get('confidence', 0)
                        current_price = analysis.get('current_price', 0)
                        
                        print(f"📈 {symbol:12} | Price: ${current_price:8.2f} | Signal: {signal:4} | Confidence: {confidence:5.2f}")
                        
                        # Display individual strategy signals
                        signals = analysis.get('signals', [])
                        strategy_summary = []
                        for s in signals:
                            if s['confidence'] > 0.3:
                                strategy_summary.append(f"{s['strategy'][:4]}:{s['signal'][:1]}")
                        
                        if strategy_summary:
                            print(f"   └─ Strategies: {' | '.join(strategy_summary)}")
                        
                        # Execute trade if signal is strong
                        if signal in ['BUY', 'SELL'] and confidence > 0.4:
                            result = self.agent.execute_trade(symbol, signal, confidence, amount=None)
                            if result.get('status') == 'executed':
                                print(f"   ✅ EXECUTED {signal} order for {symbol}")
                            elif result.get('status') == 'skipped':
                                print(f"   ⏭️  SKIPPED: {result.get('reason', 'Unknown')}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error analyzing {symbol}: {e}")
                        continue
                
                # Update positions and check exit conditions
                self.agent.update_positions()
                self.agent.check_exit_conditions()
                
                # Display current status
                self.display_status()
                
                # Wait before next iteration (demo runs faster than real trading)
                time.sleep(10)  # 10 seconds between iterations
                
        except KeyboardInterrupt:
            print("\n⏹️  Demo session interrupted by user")
        
        # Final summary
        self.display_final_summary()
    
    def display_status(self):
        """Display current trading status"""
        performance = self.agent.get_performance_summary()
        
        print(f"\n📊 Current Status:")
        print(f"   💰 Total P&L: ${performance.get('total_pnl', 0):+.2f}")
        print(f"   📈 Win Rate: {performance.get('win_rate', 0):.1f}%")
        print(f"   🎯 Open Positions: {performance.get('open_positions', 0)}")
        print(f"   📋 Total Trades: {performance.get('total_trades', 0)}")
        
        # Show open positions
        if self.agent.positions:
            print(f"\n🎯 Open Positions:")
            for symbol, position in self.agent.positions.items():
                pnl_symbol = "📈" if position.pnl >= 0 else "📉"
                print(f"   {pnl_symbol} {symbol:12} | {position.side.upper():4} | "
                      f"Entry: ${position.entry_price:7.2f} | "
                      f"Current: ${position.current_price:7.2f} | "
                      f"P&L: ${position.pnl:+7.2f} ({position.pnl_percentage:+5.1f}%)")
    
    def display_final_summary(self):
        """Display final demo session summary"""
        performance = self.agent.get_performance_summary()
        
        print("\n" + "="*80)
        print("🏁 DEMO SESSION COMPLETED")
        print("="*80)
        
        print(f"💰 Final Performance:")
        print(f"   Starting Capital: ${self.initial_capital:,.2f}")
        print(f"   Total P&L: ${performance.get('total_pnl', 0):+,.2f}")
        print(f"   Return: {(performance.get('total_pnl', 0) / self.initial_capital * 100):+.2f}%")
        print(f"   Win Rate: {performance.get('win_rate', 0):.1f}%")
        print(f"   Total Trades: {performance.get('total_trades', 0)}")
        print(f"   Closed Trades: {performance.get('closed_trades', 0)}")
        print(f"   Average Trade: ${performance.get('average_pnl', 0):+.2f}")
        
        # Show trade history
        if self.agent.trade_history:
            print(f"\n📋 Trade History:")
            closed_trades = [t for t in self.agent.trade_history if 'pnl' in t]
            
            for i, trade in enumerate(closed_trades[-10:], 1):  # Show last 10 closed trades
                pnl_symbol = "✅" if trade['pnl'] >= 0 else "❌"
                print(f"   {i:2}. {pnl_symbol} {trade['symbol']:12} | {trade['side'].upper():4} | "
                      f"${trade['price']:7.2f} | P&L: ${trade['pnl']:+7.2f} ({trade['pnl_percentage']:+5.1f}%)")
        
        print("\n🎯 Strategy Performance:")
        strategy_stats = self.calculate_strategy_performance()
        for strategy, stats in strategy_stats.items():
            print(f"   {strategy:15} | Signals: {stats['signals']:3} | "
                  f"Executed: {stats['executed']:3} | Success: {stats['success_rate']:5.1f}%")
        
        print("\n" + "="*80)
        print("🎮 Demo completed! Ready for live trading when you are.")
        print("💡 Tip: Review the results and adjust your strategy parameters if needed.")
        print("="*80)
    
    def calculate_strategy_performance(self) -> Dict:
        """Calculate performance by strategy"""
        strategy_stats = {}
        
        for trade in self.agent.trade_history:
            # This is a simplified version - in a real implementation,
            # you'd track which strategy triggered each trade
            strategy = "Mixed"  # Placeholder
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'signals': 0,
                    'executed': 0,
                    'profitable': 0,
                    'success_rate': 0
                }
            
            strategy_stats[strategy]['executed'] += 1
            if 'pnl' in trade and trade['pnl'] > 0:
                strategy_stats[strategy]['profitable'] += 1
        
        # Calculate success rates
        for strategy, stats in strategy_stats.items():
            if stats['executed'] > 0:
                stats['success_rate'] = (stats['profitable'] / stats['executed']) * 100
        
        return strategy_stats

def run_quick_demo():
    """Run a quick 5-minute demo"""
    demo = DemoTrader(initial_capital=10000)
    demo.run_demo_session(duration_minutes=5)

def run_extended_demo():
    """Run an extended 30-minute demo"""
    demo = DemoTrader(initial_capital=10000)
    demo.run_demo_session(duration_minutes=30)

def run_custom_demo():
    """Run a custom demo with user input"""
    print("🎮 Custom Demo Configuration")
    print("-" * 30)
    
    try:
        capital = float(input("💰 Starting capital (default $10,000): ") or 10000)
        duration = int(input("⏱️  Duration in minutes (default 10): ") or 10)
        
        print(f"\n🚀 Starting custom demo with ${capital:,.2f} for {duration} minutes...")
        
        demo = DemoTrader(initial_capital=capital)
        demo.run_demo_session(duration_minutes=duration)
        
    except ValueError:
        print("❌ Invalid input. Using default values.")
        run_quick_demo()
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled by user.")

if __name__ == "__main__":
    print("🎮 Cryptocurrency Trading Agent - Demo Mode")
    print("=" * 50)
    print("Choose a demo option:")
    print("1. Quick Demo (5 minutes)")
    print("2. Extended Demo (30 minutes)")
    print("3. Custom Demo")
    print("4. Exit")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            run_quick_demo()
        elif choice == "2":
            run_extended_demo()
        elif choice == "3":
            run_custom_demo()
        elif choice == "4":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice. Running quick demo...")
            run_quick_demo()
            
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled by user.")
    except Exception as e:
        logger.error(f"❌ Error running demo: {e}")
        print(f"❌ An error occurred: {e}")