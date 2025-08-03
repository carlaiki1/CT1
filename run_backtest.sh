#!/bin/bash

# Comprehensive Backtest Runner
# Tests all strategies on top 25 cryptocurrencies

echo "🚀 Starting Comprehensive Cryptocurrency Backtest"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if required packages are installed
python -c "import pandas, numpy, matplotlib, seaborn, ccxt, talib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Required packages not installed. Please run ./install.sh first."
    exit 1
fi

echo "📊 Running backtest on top 25 cryptocurrencies..."
echo "⏱️  This may take 10-30 minutes depending on data availability..."
echo ""

# Run the backtest
python backtest.py

echo ""
echo "✅ Backtest completed!"
echo "📁 Results saved in backtest_results/ directory"
echo "📊 Check the generated visualization and JSON files"
echo ""
echo "📈 Key files generated:"
echo "   - backtest_results_YYYYMMDD_HHMMSS.json (detailed results)"
echo "   - backtest_visualization_YYYYMMDD_HHMMSS.png (performance charts)"
echo ""
echo "🎯 Next steps:"
echo "   1. Review the visualization to see strategy performance"
echo "   2. Check JSON file for detailed trade-by-trade results"
echo "   3. Use insights to optimize your trading parameters"