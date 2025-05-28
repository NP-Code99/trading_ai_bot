# In this file, I will be fetching all of the crypto currency data from exchanges like Coinbase or Binance

import ccxt
import pandas as pd
import os

def fetch_ohlcv(exchange_name, symbol, timeframe='1h', limit=300):
    """
    Fetch historical OHLCV (Open, High, Low, Close, Volume) data from Coinbase.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTC/USD')
        timeframe (str): Interval ('1m', '5m', '1h', '1d', etc.)
        limit (int): Number of candles to fetch (Coinbase caps at ~300 per request)
    
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    
    try:
        print(f"Fetching {symbol} data from {exchange_name.title()}...")
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching {symbol} from {exchange_name}: {e}")
        return pd.DataFrame()


def save_to_csv(df, exchange_name, symbol):
    """
    Save a DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): The DataFrame to save
        filename (str): Name of the CSV file
    """
    os.makedirs('data', exist_ok=True)
    filename = f"{exchange_name}_{symbol.replace('/', '_').lower()}_hourly.csv"
    filepath = os.path.join('data', filename)
    df.to_csv(filepath, index=False)
    print(f"Saved data to {filepath}")


def main():
    
    exchanges = {
        'coinbase': ['BTC/USD', 'ETH/USD'],
        'kucoin' : ['PEPE/USDT', 'FLOKI/USDT', 'SHIB/USDT']
    }

    for exchange_name, symbols in exchanges.items():
        for symbol in symbols:
            df = fetch_ohlcv(exchange_name, symbol, timeframe='1h')
            if not df.empty:
                save_to_csv(df, exchange_name, symbol)

if __name__ == "__main__":
    main()
