import pandas as pd
import numpy as np

def calculate_obv(df):
    """
    Calculates On-Balance Volume (OBV).
    """
    close = df['Close']
    volume = df['Volume']
    close_diff = close.diff().fillna(0)
    direction = np.sign(close_diff)
    obv = (direction * volume).cumsum()
    return obv

def calculate_lagged_returns(df, lags=[1, 3, 5]):
    """
    Calculates percentage returns over different lag periods.
    """
    lagged_features = {}
    for lag in lags:
        lagged_features[f'Return_Lag_{lag}'] = df['Close'].pct_change(periods=lag)
    return pd.DataFrame(lagged_features, index=df.index)

def calculate_rolling_volatility(df, window=10):
    """
    Calculates rolling volatility (standard deviation of daily returns) over a window.
    """
    daily_returns = df['Close'].pct_change().fillna(0)
    rolling_vol = daily_returns.rolling(window=window).std()
    return rolling_vol
