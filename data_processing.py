import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from features import calculate_obv, calculate_lagged_returns, calculate_rolling_volatility

# Mapping target stocks to their corresponding sector ETFs
SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK",
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY",
    "JNJ": "XLV", "PFE": "XLV", "LLY": "XLV",
    "XOM": "XLE", "CVX": "XLE",
}

def prepare_data(tickers, start_date, end_date, n_steps=30, batch_size=32):
    """
    Descarga datos en lote para múltiples tickers, procesa y escala las características
    individualmente por ticker, y concatena las secuencias temporales resultantes
    para entrenar un modelo global.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
        
    print(f"Descargando datos históricos para {len(tickers)} tickers...")
    # Descarga masiva en un solo llamado
    multi_df = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    
    # Garantizar estructura de columnas MultiIndex
    if not isinstance(multi_df.columns, pd.MultiIndex):
        if len(tickers) == 1:
            multi_df.columns = pd.MultiIndex.from_product([[tickers[0]], multi_df.columns])

    # Determinar qué ETFs de sector necesitamos descargar
    sectors_needed = list(set(SECTOR_MAP.get(t.upper(), "SPY") for t in tickers))
    external_tickers = sectors_needed + ["^VIX"]
    
    print(f"Descargando datos de sectores y volatilidad: {external_tickers}...")
    ext_df = yf.download(external_tickers, start=start_date, end=end_date, group_by='ticker')
    if not isinstance(ext_df.columns, pd.MultiIndex):
        if len(external_tickers) == 1:
            ext_df.columns = pd.MultiIndex.from_product([[external_tickers[0]], ext_df.columns])

    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []
    y_test_eval_all = []

    # Procesar cada ticker por separado
    for ticker in tickers:
        ticker = ticker.upper()
        if ticker not in multi_df.columns.get_level_values(0):
            print(f"Advertencia: {ticker} no se encontró en los datos descargados. Omitiendo...")
            continue
            
        df = multi_df[ticker].dropna(how='all')
        if df.empty or len(df) < n_steps + 15:
            print(f"Advertencia: {ticker} tiene datos insuficientes. Omitiendo...")
            continue
            
        # Obtener sector ETF
        sector_ticker = SECTOR_MAP.get(ticker, "SPY")
        
        # Extraer Close de Sector ETF
        sector_close = pd.Series(0.0, index=df.index)
        if sector_ticker in ext_df.columns.get_level_values(0):
            s_df = ext_df[sector_ticker].dropna(how='all')
            if 'Close' in s_df.columns:
                sector_close = s_df['Close']
                
        # Extraer Close de VIX
        vix_close = pd.Series(0.0, index=df.index)
        if "^VIX" in ext_df.columns.get_level_values(0):
            v_df = ext_df["^VIX"].dropna(how='all')
            if 'Close' in v_df.columns:
                vix_close = v_df['Close']
                
        # Calcular características técnicas y estadísticas
        try:
            obv = calculate_obv(df)
            lagged_returns = calculate_lagged_returns(df, lags=[1, 3, 5])
            rolling_vol = calculate_rolling_volatility(df, window=10)
        except Exception as e:
            print(f"Error al calcular características para {ticker}: {e}. Omitiendo...")
            continue
            
        # Construir matriz de características
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data_t = df[features].copy()
        
        data_t['OBV'] = obv
        data_t = data_t.join(lagged_returns)
        data_t['Rolling_Vol'] = rolling_vol
        
        sector_close_df = pd.DataFrame({'Sector_Close': sector_close}, index=df.index)
        data_t = data_t.join(sector_close_df, how='left')
        
        vix_close_df = pd.DataFrame({'VIX_Close': vix_close}, index=df.index)
        data_t = data_t.join(vix_close_df, how='left')
        
        # Alinear y rellenar valores faltantes
        data_t = data_t.ffill().bfill()
        
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'OBV', 'Return_Lag_1', 'Return_Lag_3', 'Return_Lag_5',
            'Rolling_Vol', 'Sector_Close', 'VIX_Close'
        ]
        
        # Crear Target: 1 si sube mañana por >0.5%, 0 si baja por >0.5%, omitir días planos
        daily_return = data_t['Close'].pct_change().shift(-1)
        target = pd.Series(-1, index=data_t.index)
        target[daily_return > 0.005] = 1
        target[daily_return < -0.005] = 0
        data_t['Target'] = target
        data_t = data_t[data_t['Target'] != -1]

        
        if len(data_t) < n_steps + 5:
            print(f"Advertencia: {ticker} tiene datos insuficientes tras alineación. Omitiendo...")
            continue
            
        # Normalización individual por stock
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(data_t[feature_cols])
        
        # Creación de secuencias temporales (ventanas)
        X_t, y_t = [], []
        for i in range(len(scaled_features) - n_steps):
            X_t.append(scaled_features[i : i + n_steps])
            y_t.append(data_t['Target'].iloc[i + n_steps])
            
        X_t = np.array(X_t, dtype=np.float32)
        y_t = np.array(y_t, dtype=np.float32).reshape(-1, 1)
        
        # División cronológica por stock
        train_size = int(len(X_t) * 0.8)
        X_train_t, X_test_t = X_t[:train_size], X_t[train_size:]
        y_train_t, y_test_t = y_t[:train_size], y_t[train_size:]
        
        X_train_all.append(X_train_t)
        y_train_all.append(y_train_t)
        X_test_all.append(X_test_t)
        y_test_all.append(y_test_t)
        y_test_eval_all.append(y_test_t)

    if not X_train_all:
        raise ValueError("No se pudieron extraer datos válidos para ningún ticker de la lista.")

    # Concatenar todos los conjuntos individuales
    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_test = np.concatenate(X_test_all, axis=0)
    y_test = np.concatenate(y_test_all, axis=0)
    y_test_eval = np.concatenate(y_test_eval_all, axis=0)
    
    print(f"Dataset total compilado:")
    print(f"  - Entrenamiento: {X_train.shape[0]} muestras")
    print(f"  - Validación/Prueba: {X_test.shape[0]} muestras")

    # Conversión a Tensores de PyTorch
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)
    
    # Creación de DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    # shuffle=True en entrenamiento para favorecer generalización (romper bloques continuos de un solo stock)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, n_steps, len(feature_cols), y_test_eval
