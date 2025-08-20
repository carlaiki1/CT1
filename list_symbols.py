# list_symbols.py
import sys
from pathlib import Path
import logging

# Add project root to path to allow importing exchange_adapter
sys.path.append(str(Path(__file__).parent))

from exchange_adapter import CoinbaseDataFetcher

# Configure logging to suppress informational messages from the adapter
logging.basicConfig(level=logging.WARNING)

def main():
    """
    Fetches and prints the list of all tradable USD-based symbols
    from the Coinbase Advanced Trade API.
    """
    print("Fetching list of all tradable symbols from Coinbase...")

    fetcher = CoinbaseDataFetcher()

    # Fetch a large number of products to get a comprehensive list
    try:
        # The function name is a bit misleading, it can get all products
        all_symbols = fetcher.get_top_volume_products(limit=500)

        if all_symbols:
            print("\n--- Available Symbols (format: BASE/QUOTE) ---")
            # Print in multiple columns for better readability
            columns = 4
            for i in range(0, len(all_symbols), columns):
                print("  ".join(f"{s:<15}" for s in all_symbols[i:i+columns]))
            print("\n--- End of List ---")

            # Specifically check for BONK/USD
            if "BONK/USD" in all_symbols:
                print("\n✅ BONK/USD is available for trading.")
            else:
                print("\n❌ BONK/USD was not found in the list of tradable symbols.")
        else:
            print("Could not retrieve the list of symbols. The API might be down or there was an issue.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
