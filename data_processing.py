import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from features import calculate_lagged_returns, calculate_bollinger_bands

def download_and_compute_features(tickers, start_date, end_date):
    """
    Descarga datos en lote para múltiples tickers y calcula las características.
    Devuelve un diccionario {ticker: dataframe_con_features}.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
        
    print(f"Descargando datos históricos para {len(tickers)} tickers...")
    multi_df = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    
    if not isinstance(multi_df.columns, pd.MultiIndex):
        if len(tickers) == 1:
            multi_df.columns = pd.MultiIndex.from_product([[tickers[0]], multi_df.columns])

    all_ticker_data = {}
    
    for ticker in tickers:
        ticker = ticker.upper()
        if ticker not in multi_df.columns.get_level_values(0):
            print(f"Advertencia: {ticker} no se encontró en los datos descargados. Omitiendo...")
            continue
            
        df = multi_df[ticker].dropna(how='all')
        if df.empty or len(df) < 50:
            print(f"Advertencia: {ticker} tiene datos insuficientes. Omitiendo...")
            continue
            
        # Calcular características técnicas
        try:
            lagged_returns = calculate_lagged_returns(df, lags=[5])
            bb_df = calculate_bollinger_bands(df)
        except Exception as e:
            print(f"Error al calcular características para {ticker}: {e}. Omitiendo...")
            continue
            
        # Construir matriz de características
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data_t = df[features].copy()
        
        data_t = data_t.join(lagged_returns)
        data_t['BB_Upper'] = bb_df['BB_Upper']
        
        # Alinear y rellenar valores faltantes
        data_t = data_t.ffill().bfill()
        
        # Crear Target: 1 si sube mañana por >0.5%, 0 si baja por >0.5%, omitir días planos
        daily_return = data_t['Close'].pct_change().shift(-1)
        target = pd.Series(-1, index=data_t.index)
        target[daily_return > 0.005] = 1
        target[daily_return < -0.005] = 0
        data_t['Target'] = target
        data_t = data_t[data_t['Target'] != -1]
        
        if len(data_t) < 50:
            print(f"Advertencia: {ticker} tiene datos insuficientes tras alineación. Omitiendo...")
            continue
            
        all_ticker_data[ticker] = data_t

    if not all_ticker_data:
        raise ValueError("No se pudieron extraer datos válidos para ningún ticker de la lista.")

    return all_ticker_data


def create_dataloaders(all_ticker_data, train_start, train_end, test_start=None, test_end=None, n_steps=30, batch_size=128):
    """
    Filtra los datos por fechas, ajusta el scaler SOLO en los datos de entrenamiento
    (previniendo data leakage), y crea los DataLoaders de secuencias temporales.
    """
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Return_Lag_5', 'BB_Upper'
    ]
    
    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []
    y_test_eval_all = []
    
    scaler_dict = {}

    for ticker, data_t in all_ticker_data.items():
        # Filtrar por fechas
        train_df = data_t.loc[train_start:train_end]
        
        if len(train_df) < n_steps + 5:
            continue
            
        # 1. Ajustar scaler SOLO en Train
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_df[feature_cols])
        scaler_dict[ticker] = scaler # Guardamos el scaler para predict.py
        
        # 2. Transformar Train y crear secuencias
        train_scaled = scaler.transform(train_df[feature_cols])
        
        X_t_train, y_t_train = [], []
        for i in range(len(train_scaled) - n_steps):
            X_t_train.append(train_scaled[i : i + n_steps])
            y_t_train.append(train_df['Target'].iloc[i + n_steps])
            
        if len(X_t_train) > 0:
            X_train_all.append(np.array(X_t_train, dtype=np.float32))
            y_train_all.append(np.array(y_t_train, dtype=np.float32).reshape(-1, 1))
            
        # 3. Transformar Test (si existe) y crear secuencias
        if test_start and test_end:
            # Importante: para el primer día de test, necesitamos los `n_steps` días previos.
            test_mask = (data_t.index >= test_start) & (data_t.index <= test_end)
            if not test_mask.any():
                continue
                
            first_test_idx = np.where(test_mask)[0][0]
            start_idx = max(0, first_test_idx - n_steps)
            
            test_df_extended = data_t.iloc[start_idx : np.where(test_mask)[0][-1] + 1]
            
            if len(test_df_extended) < n_steps + 1:
                continue
                
            test_scaled = scaler.transform(test_df_extended[feature_cols])
            
            X_t_test, y_t_test = [], []
            for i in range(len(test_scaled) - n_steps):
                X_t_test.append(test_scaled[i : i + n_steps])
                # El target corresponde al último día de la ventana
                y_t_test.append(test_df_extended['Target'].iloc[i + n_steps])
                
            if len(X_t_test) > 0:
                X_test_all.append(np.array(X_t_test, dtype=np.float32))
                y_test_all.append(np.array(y_t_test, dtype=np.float32).reshape(-1, 1))
                y_test_eval_all.append(np.array(y_t_test, dtype=np.float32).reshape(-1, 1))

    if not X_train_all:
        return None, None, n_steps, len(feature_cols), None, scaler_dict

    # Concatenar todos los conjuntos individuales
    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_loader = None
    y_test_eval = None
    
    if X_test_all:
        X_test = np.concatenate(X_test_all, axis=0)
        y_test = np.concatenate(y_test_all, axis=0)
        y_test_eval = np.concatenate(y_test_eval_all, axis=0)
        
        X_test_t = torch.tensor(X_test)
        y_test_t = torch.tensor(y_test)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, n_steps, len(feature_cols), y_test_eval, scaler_dict
