# In this file, I will be creating all of the market indicators that may help influence the model to buy or sell coins


import pandas as pd
import ta 

def add_indicators(df):
    
    df = df.sort_values('timestamp')

    # Overbought/Oversold Pressure (RSI)
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    # Trend Direction and Strength (MACD)
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Volatility and Deviation from Trend (Bollinger Bands)
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']

    # Trend Smoothing (detects trend continuation) (EMAs)
    df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()

    # Volume Change (%)
    df['volume_change'] = df['volume'].pct_change()

    # Drop empty rows
    df.dropna(inplace=True)

    return df

df = pd.read_csv("data/kucoin_pepe_usdt_hourly.csv")
df = add_indicators(df)

print(df)