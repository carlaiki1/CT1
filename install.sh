#!/bin/bash

# Cryptocurrency Trading Agent Installation Script
# Created by OpenHands AI Assistant

echo "🚀 Installing Cryptocurrency Trading Agent..."
echo "=============================================="

# Check if Python 3.8+ is installed
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected"
else
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "📚 Installing required packages..."
pip install flask flask-cors requests pandas numpy ta-lib yfinance ccxt plotly dash dash-bootstrap-components python-dotenv schedule websocket-client

# Install additional packages for backtesting
echo "📊 Installing backtesting packages..."
pip install matplotlib seaborn scipy scikit-learn joblib tqdm

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env configuration file..."
    cat > .env << EOF
# Coinbase Advanced Trade API
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_API_SECRET=your_coinbase_api_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here

# Kraken API (Optional)
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_API_SECRET=your_kraken_api_secret_here

# Trading Configuration
DEMO_MODE=true
DEFAULT_TRADE_AMOUNT=100
RISK_PERCENTAGE=2
MAX_POSITIONS=5

# Web Interface
FLASK_PORT=12000
FLASK_HOST=0.0.0.0
FLASK_DEBUG=false
EOF
    echo "⚠️  Please edit .env file with your API credentials"
fi

# Make scripts executable
chmod +x *.py
chmod +x run_backtest.sh

echo ""
echo "✅ Installation completed successfully!"
echo ""
echo "🔧 Next Steps:"
echo "1. Edit .env file with your API credentials"
echo "2. Run: python app.py (for web interface)"
echo "3. Run: python demo.py (for demo trading)"
echo "4. Run: python backtest.py (for strategy backtesting)"
echo "5. Open: http://localhost:12000 (web interface)"
echo ""
echo "📖 Read QUICK_START_GUIDE.md for detailed instructions"
echo "🎯 Happy Trading! 🚀"