import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════
#  EXISTING INDICATORS
# ═══════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════
#  NEW TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════

def calculate_rsi(df, period=14):
    """
    Relative Strength Index (RSI).
    Measures overbought/oversold conditions on a 0-100 scale.
    """
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral default


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    MACD (Moving Average Convergence Divergence).
    Returns a DataFrame with MACD line, Signal line, and Histogram.
    """
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'MACD': macd_line,
        'MACD_Signal': signal_line,
        'MACD_Hist': histogram,
    }, index=df.index)


def calculate_bollinger_bands(df, window=20, num_std=2):
    """
    Bollinger Bands.
    Returns upper band, lower band, and bandwidth (normalized width).
    """
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()

    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower) / sma.replace(0, np.nan)

    return pd.DataFrame({
        'BB_Upper': upper,
        'BB_Lower': lower,
        'BB_Width': width,
    }, index=df.index)


def calculate_atr(df, period=14):
    """
    Average True Range (ATR).
    Measures volatility based on true range (accounts for gaps).
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, min_periods=period).mean()
    return atr


def calculate_stochastic(df, k_period=14, d_period=3):
    """
    Stochastic Oscillator (%K and %D).
    Compares close to the high-low range over k_period days.
    """
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()

    denom = (high_max - low_min).replace(0, np.nan)
    stoch_k = 100 * (df['Close'] - low_min) / denom
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return pd.DataFrame({
        'Stoch_K': stoch_k,
        'Stoch_D': stoch_d,
    }, index=df.index)


def calculate_williams_r(df, period=14):
    """
    Williams %R.
    Similar to stochastic but inverted scale (-100 to 0).
    """
    high_max = df['High'].rolling(window=period).max()
    low_min = df['Low'].rolling(window=period).min()

    denom = (high_max - low_min).replace(0, np.nan)
    wr = -100 * (high_max - df['Close']) / denom
    return wr


def calculate_cci(df, period=20):
    """
    Commodity Channel Index (CCI).
    Measures deviation from statistical mean — identifies cyclical turns.
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    cci = (typical_price - sma_tp) / (0.015 * mad.replace(0, np.nan))
    return cci


def calculate_mfi(df, period=14):
    """
    Money Flow Index (MFI).
    Volume-weighted RSI — combines price AND volume pressure.
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']

    delta = typical_price.diff()
    positive_flow = money_flow.where(delta > 0, 0.0)
    negative_flow = money_flow.where(delta < 0, 0.0)

    pos_sum = positive_flow.rolling(window=period).sum()
    neg_sum = negative_flow.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))
    return mfi.fillna(50)


def calculate_ema_ratio(df, fast=12, slow=26):
    """
    EMA Ratio (EMA_fast / EMA_slow).
    Values > 1 indicate bullish trend, < 1 bearish.
    """
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    return ema_fast / ema_slow.replace(0, np.nan)


def calculate_return_lag(df, lag=10):
    """
    Single lagged return for a specific period.
    """
    return df['Close'].pct_change(periods=lag)


# ═══════════════════════════════════════════════════════
#  CONVENIENCE: Compute all indicators at once
# ═══════════════════════════════════════════════════════

def compute_all_technical_features(df):
    """
    Computes ALL available technical indicators for a single-ticker DataFrame.
    Returns a DataFrame with one column per indicator, aligned to df.index.
    """
    result = pd.DataFrame(index=df.index)

    # Existing
    result['OBV'] = calculate_obv(df)
    lagged = calculate_lagged_returns(df, lags=[1, 3, 5])
    result = result.join(lagged)
    result['Rolling_Vol'] = calculate_rolling_volatility(df, window=10)

    # New
    result['RSI_14'] = calculate_rsi(df, period=14)

    macd_df = calculate_macd(df)
    result = result.join(macd_df)

    bb_df = calculate_bollinger_bands(df)
    result = result.join(bb_df)

    result['ATR_14'] = calculate_atr(df, period=14)

    stoch_df = calculate_stochastic(df)
    result = result.join(stoch_df)

    result['Williams_R'] = calculate_williams_r(df)
    result['CCI_20'] = calculate_cci(df, period=20)
    result['MFI_14'] = calculate_mfi(df, period=14)
    result['EMA_Ratio'] = calculate_ema_ratio(df)
    result['Return_Lag_10'] = calculate_return_lag(df, lag=10)

    return result
