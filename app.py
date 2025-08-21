"""
Cryptocurrency Trading Agent Web Application
Flask-based web interface for monitoring and controlling the trading agent
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import logging
from typing import Dict, List
import plotly.graph_objs as go
import plotly.utils

from exchange_adapter import ExchangeAdapter, CoinbaseDataFetcher
from trading_agent import TradingAgent
from backtest import BacktestEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Global variables
trading_agent = None
exchange_adapter = None
trading_active = False
trading_thread = None
app_config = {
    'demo_mode': True,
    'auto_trading': False,
    'risk_percentage': 2.0,
    'max_positions': 5
}

def load_config():
    """Load configuration from .env file"""
    config = {}
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    config[key] = value
    return config

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify(app_config)

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    global app_config, trading_agent, exchange_adapter
    
    try:
        data = request.get_json()
        app_config.update(data)
        
        # Reinitialize trading agent if exchange settings changed
        if 'api_key' in data or 'api_secret' in data:
            exchange_adapter = ExchangeAdapter(
                exchange_name='coinbase',
                api_key=data.get('api_key', ''),
                api_secret=data.get('api_secret', ''),
                passphrase=data.get('passphrase', ''),
                demo_mode=app_config['demo_mode']
            )
            
            trading_agent = TradingAgent(
                exchange_adapter=exchange_adapter,
                config=app_config
            )
        
        return jsonify({'status': 'success', 'config': app_config})
        
    except Exception as e:
        logger.error(f"❌ Error updating config: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/balance', methods=['GET'])
def get_balance():
    """Get account balance"""
    try:
        if not exchange_adapter:
            return jsonify({'error': 'Exchange not configured'}), 400
        
        balance = exchange_adapter.get_account_balance()
        return jsonify(balance)
        
    except Exception as e:
        logger.error(f"❌ Error fetching balance: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    """Get available trading symbols"""
    try:
        fetcher = CoinbaseDataFetcher()
        # Fetch all products efficiently using a single API call
        all_products = fetcher.get_products()
        # Filter for online USD pairs and format them correctly
        usd_symbols = [
            p['id'].replace('-', '/')
            for p in all_products
            if p.get('quote_currency') == 'USD' and p.get('status') == 'online'
        ]
        # Sort alphabetically for user convenience
        usd_symbols.sort()
        return jsonify(usd_symbols)
        
    except Exception as e:
        logger.error(f"❌ Error fetching symbols: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/<path:symbol>', methods=['GET'])
def analyze_symbol(symbol):
    """Analyze a specific symbol"""
    try:
        if not trading_agent:
            return jsonify({'error': 'Trading agent not initialized'}), 400
        
        analysis = trading_agent.analyze_symbol(symbol)
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"❌ Error analyzing {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get current positions"""
    try:
        if not trading_agent:
            return jsonify({'error': 'Trading agent not initialized'}), 400
        
        # Update positions with current prices
        trading_agent.update_positions()
        
        positions = []
        for symbol, position in trading_agent.positions.items():
            positions.append({
                'symbol': position.symbol,
                'side': position.side,
                'amount': position.amount,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'pnl': position.pnl,
                'pnl_percentage': position.pnl_percentage,
                'timestamp': position.timestamp.isoformat()
            })
        
        return jsonify(positions)
        
    except Exception as e:
        logger.error(f"❌ Error fetching positions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades', methods=['GET'])
def get_trades():
    """Get trade history"""
    try:
        if not trading_agent:
            return jsonify({'error': 'Trading agent not initialized'}), 400
        
        return jsonify(trading_agent.trade_history)
        
    except Exception as e:
        logger.error(f"❌ Error fetching trades: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get performance summary"""
    try:
        if not trading_agent:
            return jsonify({'error': 'Trading agent not initialized'}), 400
        
        performance = trading_agent.get_performance_summary()
        return jsonify(performance)
        
    except Exception as e:
        logger.error(f"❌ Error fetching performance: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading/start', methods=['POST'])
def start_trading():
    """Start automated trading"""
    global trading_active, trading_thread
    
    try:
        if not trading_agent:
            return jsonify({'error': 'Trading agent not initialized'}), 400
        
        if trading_active:
            return jsonify({'error': 'Trading already active'}), 400
        
        trading_active = True
        trading_thread = threading.Thread(target=trading_loop)
        trading_thread.daemon = True
        trading_thread.start()
        
        logger.info("🚀 Automated trading started")
        return jsonify({'status': 'Trading started'})
        
    except Exception as e:
        logger.error(f"❌ Error starting trading: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    """Stop automated trading"""
    global trading_active
    
    try:
        trading_active = False
        logger.info("⏹️ Automated trading stopped")
        return jsonify({'status': 'Trading stopped'})
        
    except Exception as e:
        logger.error(f"❌ Error stopping trading: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading/status', methods=['GET'])
def get_trading_status():
    """Get trading status"""
    return jsonify({
        'active': trading_active,
        'agent_initialized': trading_agent is not None,
        'exchange_connected': exchange_adapter is not None
    })

@app.route('/api/connect', methods=['POST'])
def connect_to_exchange():
    """Connect to the live exchange"""
    global exchange_adapter, trading_agent
    try:
        exchange_adapter = ExchangeAdapter(
            exchange_name='coinbase',
            demo_mode=False
        )
        if exchange_adapter.demo_mode: # This will be true if API keys are not found
            return jsonify({'status': 'error', 'message': 'API keys not found or invalid.'}), 400

        trading_agent = TradingAgent(
            exchange_adapter=exchange_adapter,
            config=app_config
        )
        balance = exchange_adapter.get_account_balance()
        return jsonify({
            'status': 'success',
            'message': 'Successfully connected to Coinbase.',
            'balance': balance,
            'demo_mode': False
        })
    except Exception as e:
        logger.error(f"❌ Error connecting to exchange: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/democonnect', methods=['POST'])
def connect_to_demo():
    """Connect to the demo exchange"""
    global exchange_adapter, trading_agent
    try:
        initialize_demo_mode()
        balance = exchange_adapter.get_account_balance()
        return jsonify({
            'status': 'success',
            'message': 'Successfully connected to Demo Mode.',
            'balance': balance,
            'demo_mode': True
        })
    except Exception as e:
        logger.error(f"❌ Error connecting to demo: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run backtest"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', None)
        years = data.get('years', 3)
        
        # Run backtest in a separate thread
        def backtest_worker():
            engine = BacktestEngine(initial_capital=10000, commission=0.005)
            results = engine.run_comprehensive_backtest(symbols=symbols, years=years)
            report = engine.generate_performance_report()
            engine.save_results()
            return report
        
        # For now, return a simple response
        # In production, you might want to run this asynchronously
        return jsonify({'status': 'Backtest started', 'message': 'Check backtest_results/ directory for results'})
        
    except Exception as e:
        logger.error(f"❌ Error running backtest: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart/<path:symbol>')
def get_chart_data(symbol):
    """Get chart data for a symbol"""
    try:
        if not exchange_adapter:
            fetcher = CoinbaseDataFetcher()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            df = fetcher.get_historical_data(symbol, start_date.isoformat(), end_date.isoformat())
        else:
            df = exchange_adapter.get_historical_data(symbol, '1d', 30)
        
        if df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        # Create candlestick chart
        trace = go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        )
        
        layout = go.Layout(
            title=f'{symbol} Price Chart',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price (USD)'),
            template='plotly_white'
        )
        
        fig = go.Figure(data=[trace], layout=layout)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({'chart': graphJSON})
        
    except Exception as e:
        logger.error(f"❌ Error generating chart for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

def trading_loop():
    """Main trading loop"""
    global trading_active
    
    logger.info("🔄 Trading loop started")
    
    while trading_active:
        try:
            if not trading_agent:
                time.sleep(60)
                continue
            
            # Get top symbols to analyze
            fetcher = CoinbaseDataFetcher()
            symbols = fetcher.get_top_volume_products(10)  # Analyze top 10
            
            for symbol in symbols:
                if not trading_active:
                    break
                
                try:
                    # Analyze symbol
                    analysis = trading_agent.analyze_symbol(symbol)
                    
                    if 'error' in analysis:
                        continue
                    
                    consensus = analysis.get('consensus', {})
                    signal = consensus.get('signal', 'HOLD')
                    confidence = consensus.get('confidence', 0)
                    
                    # Execute trade if signal is strong enough
                    if signal in ['BUY', 'SELL'] and confidence > 0.5:
                        result = trading_agent.execute_trade(symbol, signal, confidence)
                        if result.get('status') == 'executed':
                            logger.info(f"✅ Executed {signal} for {symbol} (confidence: {confidence:.2f})")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {symbol}: {e}")
                    continue
            
            # Update positions and check exit conditions
            trading_agent.update_positions()
            trading_agent.check_exit_conditions()
            
            # Wait before next iteration
            time.sleep(300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"❌ Error in trading loop: {e}")
            time.sleep(60)
    
    logger.info("⏹️ Trading loop stopped")

# Initialize demo mode by default
def initialize_demo_mode():
    """Initialize demo mode"""
    global exchange_adapter, trading_agent
    
    try:
        exchange_adapter = ExchangeAdapter(
            exchange_name='coinbase',
            api_key='demo_key',
            api_secret='demo_secret',
            passphrase='demo_passphrase',
            demo_mode=True
        )
        
        trading_agent = TradingAgent(
            exchange_adapter=exchange_adapter,
            config=app_config
        )
        
        logger.info("✅ Demo mode initialized")
        
    except Exception as e:
        logger.error(f"❌ Error initializing demo mode: {e}")

if __name__ == '__main__':
    # Create templates directory and basic template
    os.makedirs('templates', exist_ok=True)
    
    # Start the Flask app
    port = int(os.environ.get('FLASK_PORT', 12000))
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    
    logger.info(f"🚀 Starting Cryptocurrency Trading Agent on {host}:{port}")
    logger.info(f"🌐 Access the web interface at: http://localhost:{port}")
    
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )