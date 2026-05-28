"""
predict.py — Predice la dirección del precio de una acción para el día siguiente.

Uso:
    python predict.py <TICKER>          # Desde la línea de comandos
    python predict.py                   # Solicita el ticker de forma interactiva

Ejemplo:
    python predict.py AAPL
    python predict.py NVDA

IMPORTANTE: Debes haber ejecutado `python main.py` al menos una vez para
            que el modelo entrenado esté guardado en 'transformer_model.pt'.
"""

import sys
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import torch
from sklearn.preprocessing import MinMaxScaler

from features import calculate_lagged_returns, calculate_bollinger_bands
from transformer_model import TransformerModel
from lstm_model import LSTMModel


TIMESTEPS            = 30
D_MODEL              = 128
NUM_HEADS            = 4
NUM_LAYERS           = 3
DROPOUT              = 0.2
TRANSFORMER_PATH     = "transformer_model.pt"
LSTM_PATH            = "lstm_model.pt"
LSTM_HIDDEN_DIM      = 128
LSTM_NUM_LAYERS      = 2

SECTOR_ETF_MAP = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",
    "Consumer Cyclical":      "XLY",
    "Healthcare":             "XLV",
    "Energy":                 "XLE",
    "Communication Services": "XLC",
    "Industrials":            "XLI",
    "Consumer Defensive":     "XLP",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Basic Materials":        "XLB",
}

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Return_Lag_5', 'BB_Upper'
]
NUM_FEATURES = len(FEATURE_COLS)  # 12


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _get_sector_etf(ticker: str) -> str:
    """Consulta yfinance para obtener el ETF de sector correspondiente al ticker."""
    try:
        info = yf.Ticker(ticker).info
        sector = info.get('sector', '')
        return SECTOR_ETF_MAP.get(sector, 'SPY')
    except Exception:
        return 'SPY'


def _safe_close(ext_df, symbol: str, ref_index: pd.Index) -> pd.Series:
    """Extrae la serie Close de un símbolo dentro de un DataFrame MultiIndex."""
    try:
        if isinstance(ext_df.columns, pd.MultiIndex):
            if symbol in ext_df.columns.get_level_values(0):
                sub = ext_df[symbol].dropna(how='all')
                if 'Close' in sub.columns:
                    return sub['Close'].reindex(ref_index).ffill().bfill()
        else:
            if 'Close' in ext_df.columns:
                return ext_df['Close'].reindex(ref_index).ffill().bfill()
    except Exception:
        pass
    return pd.Series(0.0, index=ref_index)


def build_feature_window(ticker: str, n_steps: int = TIMESTEPS) -> np.ndarray:
    """
    Descarga datos recientes para `ticker` y construye la ventana de características
    más reciente lista para ser consumida por el modelo.

    Retorna:
        np.ndarray de shape (1, n_steps, NUM_FEATURES) y dtype float32.
    """
    # Necesitamos suficientes días para calcular lags + rolling + n_steps
    # 120 días calendario ≈ ~85 días de trading, margen amplio
    download_period = "120d"

    print(f"  › Obteniendo sector ETF para {ticker}...")
    sector_etf = _get_sector_etf(ticker)
    print(f"    Sector ETF asignado: {sector_etf}")

    print(f"  › Descargando precios históricos ({download_period})...")
    df = yf.download(ticker, period=download_period, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"yfinance no devolvió datos para '{ticker}'. Verifica que el ticker sea válido.")

    # yfinance a veces devuelve columnas MultiIndex incluso para 1 ticker
    if isinstance(df.columns, pd.MultiIndex):
        df = df[ticker] if ticker in df.columns.get_level_values(0) else df.droplevel(1, axis=1)

    if len(df) < n_steps + 10:
        raise ValueError(
            f"Datos insuficientes para '{ticker}': se obtuvieron {len(df)} filas, "
            f"se necesitan al menos {n_steps + 10}."
        )

    # ── Indicadores técnicos ──────────────────
    lagged       = calculate_lagged_returns(df, lags=[5])
    bb_df        = calculate_bollinger_bands(df)

    # ── Construir DataFrame de características ─
    data_t = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    data_t = data_t.join(lagged)
    data_t['BB_Upper']  = bb_df['BB_Upper']

    data_t = data_t[FEATURE_COLS].ffill().bfill().dropna()

    if len(data_t) < n_steps:
        raise ValueError(
            f"Filas limpias insuficientes para '{ticker}' tras preprocesado: "
            f"{len(data_t)} < {n_steps}."
        )

    # ── Normalización (Evitar Data Leakage) ───
    try:
        with open("scalers.pkl", "rb") as f:
            scaler_dict = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("No se encontró 'scalers.pkl'. Debes ejecutar 'python main.py' primero.")
        
    scaler = scaler_dict.get(ticker)
    
    if scaler is None:
        print(f"  [!] Advertencia: '{ticker}' no estaba en el conjunto de entrenamiento.")
        print(f"      Ajustando un nuevo scaler con los últimos 120 días (menos preciso).")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(data_t)
    else:
        # Solo transformamos, usando los mismos mínimos y máximos que en entrenamiento
        scaled = scaler.transform(data_t)

    # Ventana de los últimos n_steps días
    window = scaled[-n_steps:]                          # (n_steps, NUM_FEATURES)
    return window[np.newaxis, :, :].astype(np.float32)  # (1, n_steps, NUM_FEATURES)


def _format_result(name: str, prob: float) -> str:
    """Formatea el resultado de un modelo individual."""
    is_bullish = prob >= 0.5
    direction  = "SUBIDA  (BULLISH)" if is_bullish else "BAJADA  (BEARISH)"
    confidence = prob if is_bullish else 1.0 - prob
    filled = int(confidence * 20)
    bar = "█" * filled + "░" * (20 - filled)

    lines = [
        f"  Prob. subida  : {prob*100:6.2f}%",
        f"  Prob. bajada  : {(1-prob)*100:6.2f}%",
        f"  Predicción    : {direction}",
        f"  Confianza     : [{bar}] {confidence*100:.1f}%",
    ]
    return "\n".join(lines)


def predict(ticker: str) -> None:
    ticker = ticker.upper().strip()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'═'*56}")
    print(f"   PREDICCION NEXT-DAY: {ticker}")
    print(f"{'═'*56}")

    # ── 1. Construir ventana de características (compartida) ─
    print(f"\n[1/3] Construyendo secuencia de características para {ticker}...")
    try:
        X = build_feature_window(ticker, n_steps=TIMESTEPS)
    except ValueError as e:
        print(f"\n  {e}")
        sys.exit(1)

    X_tensor = torch.tensor(X).to(device)  # (1, 30, 12)

    results = {}  # {model_name: probability}

    # ── 2. Transformer ────────────────────────
    print(f"\n[2/3] Cargando modelos...")
    try:
        transformer = TransformerModel(
            input_dim=NUM_FEATURES, timesteps=TIMESTEPS,
            d_model=D_MODEL, num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS, dropout=DROPOUT,
        )
        state = torch.load(TRANSFORMER_PATH, map_location=device, weights_only=True)
        transformer.load_state_dict(state)
        transformer.to(device).eval()
        with torch.no_grad():
            prob_t = torch.sigmoid(transformer(X_tensor)).item()
        results["Transformer"] = prob_t
        print(f"    Transformer cargado desde '{TRANSFORMER_PATH}'")
    except FileNotFoundError:
        print(f"    Transformer no encontrado ('{TRANSFORMER_PATH}') — omitido")
    except Exception as e:
        print(f"    Error cargando Transformer: {e} — omitido")

    # ── 3. LSTM ───────────────────────────────
    try:
        lstm = LSTMModel(
            input_dim=NUM_FEATURES,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
        )
        state = torch.load(LSTM_PATH, map_location=device, weights_only=True)
        lstm.load_state_dict(state)
        lstm.to(device).eval()
        with torch.no_grad():
            prob_l = torch.sigmoid(lstm(X_tensor)).item()
        results["LSTM"] = prob_l
        print(f"    LSTM cargado desde '{LSTM_PATH}'")
    except FileNotFoundError:
        print(f"    LSTM no encontrado ('{LSTM_PATH}') — omitido")
    except Exception as e:
        print(f"    Error cargando LSTM: {e} — omitido")

    if not results:
        print("\n  No se pudo cargar ningun modelo. Ejecuta primero: python main.py")
        sys.exit(1)

    # ── 4. Mostrar resultados ─────────────────
    print(f"\n[3/3] Ejecutando inferencia...")

    for name, prob in results.items():
        print(f"\n{'─'*56}")
        print(f"  Modelo        : {name}")
        print(_format_result(name, prob))

    # Consenso (promedio de ambos modelos)
    if len(results) > 1:
        avg_prob   = sum(results.values()) / len(results)
        is_bullish = avg_prob >= 0.5
        direction  = "SUBIDA  (BULLISH)" if is_bullish else "BAJADA  (BEARISH)"
        confidence = avg_prob if is_bullish else 1.0 - avg_prob
        filled = int(confidence * 20)
        bar = "█" * filled + "░" * (20 - filled)

        print(f"\n{'═'*56}")
        print(f"  CONSENSO ({' + '.join(results.keys())})")
        print(f"  Ticker        : {ticker}")
        print(f"  Prob. subida  : {avg_prob*100:6.2f}%")
        print(f"  Prob. bajada  : {(1-avg_prob)*100:6.2f}%")
        print(f"  Predicción    : {direction}")
        print(f"  Confianza     : [{bar}] {confidence*100:.1f}%")
        print(f"{'═'*56}")



if __name__ == "__main__":
    if len(sys.argv) > 1:
        ticker_arg = sys.argv[1]
    else:
        ticker_arg = input("Ingresa el ticker a predecir (e.g. AAPL): ").strip()

    if not ticker_arg:
        print("  No se proporcionó ningún ticker.")
        sys.exit(1)

    predict(ticker_arg)
