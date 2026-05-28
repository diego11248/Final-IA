"""
feature_selection.py — Automated forward feature selection for stock prediction.

Downloads data for a subset of tickers, computes the full feature pool (~34 features),
then uses forward selection to find the best feature combination by training
a lightweight Transformer and measuring F1 score on a held-out test set.

Usage:
    python feature_selection.py

Output:
    - Ranked leaderboard printed to console
    - Results saved to feature_selection_results.csv
"""

import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import concurrent.futures

from features import compute_all_technical_features
from transformer_model import TransformerModel


# Representative subset of tickers (diverse sectors, keeps runtime manageable)
SELECTION_TICKERS = [
    "AAPL", "MSFT",   # Tech
    "JPM", "BAC",     # Finance
    "AMZN", "HD",     # Consumer
    "JNJ", "LLY",    # Healthcare
    "XOM", "CVX",     # Energy
]

START_DATE     = "2020-01-01"
END_DATE       = "2025-12-31"
N_STEPS        = 30
BATCH_SIZE     = 128
SELECTION_EPOCHS = 10    # Fewer epochs for speed during selection
SELECTION_LR   = 0.005

# Macroeconomic external tickers to download
MACRO_TICKERS = ["^TNX", "^IRX", "GLD", "USO", "UUP", "^GSPC"]

# Sector ETF mapping (same as data_processing.py)
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


# Base features are always included (OHLCV)
BASE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

# Candidate features to evaluate (each is toggled on/off during selection)
CANDIDATE_FEATURES = [
    # Existing indicators
    'OBV',
    'Return_Lag_1',
    'Return_Lag_3',
    'Return_Lag_5',
    'Rolling_Vol',
    # New technical indicators
    'RSI_14',
    'MACD',
    'MACD_Signal',
    'MACD_Hist',
    'BB_Upper',
    'BB_Lower',
    'BB_Width',
    'ATR_14',
    'Stoch_K',
    'Stoch_D',
    'Williams_R',
    'CCI_20',
    'MFI_14',
    'EMA_Ratio',
    'Return_Lag_10',
    # External signals
    'Sector_Close',
    'VIX_Close',
    'Treasury_10Y',
    'Treasury_2Y',
    'Yield_Spread',
    'Gold_Close',
    'Oil_Close',
    'Dollar_Index',
    'SP500_Return',
]



def _get_sector_etf(ticker):
    try:
        info = yf.Ticker(ticker).info
        sector = info.get('sector', '')
        return ticker.upper(), SECTOR_ETF_MAP.get(sector, 'SPY')
    except Exception:
        return ticker.upper(), 'SPY'


def _safe_series(ext_df, symbol, col, ref_index):
    """Extract a column from a MultiIndex DataFrame safely."""
    try:
        if isinstance(ext_df.columns, pd.MultiIndex):
            if symbol in ext_df.columns.get_level_values(0):
                sub = ext_df[symbol].dropna(how='all')
                if col in sub.columns:
                    return sub[col].reindex(ref_index).ffill().bfill()
        elif col in ext_df.columns:
            return ext_df[col].reindex(ref_index).ffill().bfill()
    except Exception:
        pass
    return pd.Series(0.0, index=ref_index)


def download_all_data():
    """
    Downloads all ticker + external data in bulk.
    Returns multi_df, ext_df, macro_df, sector_map.
    """
    tickers = SELECTION_TICKERS

    print(f"Obteniendo sectores para {len(tickers)} tickers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        sector_tuples = list(ex.map(_get_sector_etf, tickers))
    sector_map = dict(sector_tuples)

    print(f"Descargando datos historicos para {len(tickers)} tickers...")
    multi_df = yf.download(tickers, start=START_DATE, end=END_DATE, group_by='ticker')
    if not isinstance(multi_df.columns, pd.MultiIndex) and len(tickers) == 1:
        multi_df.columns = pd.MultiIndex.from_product([[tickers[0]], multi_df.columns])

    # Sector ETFs + VIX
    sectors_needed = list(set(sector_map.values()))
    ext_tickers = sectors_needed + ["^VIX"]
    print(f"Descargando ETFs de sector + VIX: {ext_tickers}...")
    ext_df = yf.download(ext_tickers, start=START_DATE, end=END_DATE, group_by='ticker')
    if not isinstance(ext_df.columns, pd.MultiIndex) and len(ext_tickers) == 1:
        ext_df.columns = pd.MultiIndex.from_product([[ext_tickers[0]], ext_df.columns])

    # Macro tickers
    print(f"Descargando datos macroeconomicos: {MACRO_TICKERS}...")
    macro_df = yf.download(MACRO_TICKERS, start=START_DATE, end=END_DATE, group_by='ticker')
    if not isinstance(macro_df.columns, pd.MultiIndex) and len(MACRO_TICKERS) == 1:
        macro_df.columns = pd.MultiIndex.from_product([[MACRO_TICKERS[0]], macro_df.columns])

    return multi_df, ext_df, macro_df, sector_map


def build_full_feature_matrix(multi_df, ext_df, macro_df, sector_map):
    """
    For each ticker, builds a DataFrame with ALL possible features.
    Returns dict: {ticker: DataFrame with all features + Target column}.
    """
    all_ticker_data = {}

    for ticker in SELECTION_TICKERS:
        ticker = ticker.upper()
        if ticker not in multi_df.columns.get_level_values(0):
            print(f"  {ticker} no encontrado en datos descargados, omitiendo...")
            continue

        df = multi_df[ticker].dropna(how='all')
        if df.empty or len(df) < N_STEPS + 30:
            print(f"  {ticker} datos insuficientes, omitiendo...")
            continue

        # ── Technical indicators ────────────────
        tech = compute_all_technical_features(df)
        data_t = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data_t = data_t.join(tech)

        # ── External: Sector ETF + VIX ──────────
        sector_ticker = sector_map.get(ticker, 'SPY')
        data_t['Sector_Close'] = _safe_series(ext_df, sector_ticker, 'Close', df.index).values
        data_t['VIX_Close'] = _safe_series(ext_df, '^VIX', 'Close', df.index).values

        # ── Macroeconomic ───────────────────────
        data_t['Treasury_10Y'] = _safe_series(macro_df, '^TNX', 'Close', df.index).values
        data_t['Treasury_2Y']  = _safe_series(macro_df, '^IRX', 'Close', df.index).values
        data_t['Yield_Spread'] = data_t['Treasury_10Y'] - data_t['Treasury_2Y']
        data_t['Gold_Close']   = _safe_series(macro_df, 'GLD', 'Close', df.index).values
        data_t['Oil_Close']    = _safe_series(macro_df, 'USO', 'Close', df.index).values
        data_t['Dollar_Index'] = _safe_series(macro_df, 'UUP', 'Close', df.index).values

        sp500_close = _safe_series(macro_df, '^GSPC', 'Close', df.index)
        data_t['SP500_Return'] = sp500_close.pct_change().fillna(0).values

        # ── Target ──────────────────────────────
        daily_return = data_t['Close'].pct_change().shift(-1)
        target = pd.Series(-1, index=data_t.index)
        target[daily_return > 0.005] = 1
        target[daily_return < -0.005] = 0
        data_t['Target'] = target

        # Fill and clean
        data_t = data_t.ffill().bfill()
        data_t = data_t[data_t['Target'] != -1]

        if len(data_t) < N_STEPS + 5:
            print(f"  {ticker} insuficiente tras filtrado, omitiendo...")
            continue

        all_ticker_data[ticker] = data_t
        print(f"  {ticker}: {len(data_t)} filas listas ({data_t.shape[1]-1} features)")

    return all_ticker_data


def build_dataloaders(all_ticker_data, feature_cols):
    """
    Given the per-ticker DataFrames and a list of feature columns,
    creates train/test DataLoaders with sliding windows.
    """
    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []

    for ticker, data_t in all_ticker_data.items():
        # Check all feature columns exist
        missing = [f for f in feature_cols if f not in data_t.columns]
        if missing:
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(data_t[feature_cols])

        X_t, y_t = [], []
        for i in range(len(scaled) - N_STEPS):
            X_t.append(scaled[i:i + N_STEPS])
            y_t.append(data_t['Target'].iloc[i + N_STEPS])

        if len(X_t) < 10:
            continue

        X_t = np.array(X_t, dtype=np.float32)
        y_t = np.array(y_t, dtype=np.float32).reshape(-1, 1)

        split = int(len(X_t) * 0.8)
        X_train_all.append(X_t[:split])
        y_train_all.append(y_t[:split])
        X_test_all.append(X_t[split:])
        y_test_all.append(y_t[split:])

    if not X_train_all:
        return None, None, None

    X_train = np.concatenate(X_train_all)
    y_train = np.concatenate(y_train_all)
    X_test = np.concatenate(X_test_all)
    y_test = np.concatenate(y_test_all)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    return train_loader, test_loader, y_test



def quick_train_evaluate(feature_cols, all_ticker_data, device):
    """
    Trains a small Transformer with the given feature set and returns metrics.
    Returns dict with f1, accuracy, precision, recall, or None on failure.
    """
    n_features = len(feature_cols)

    train_loader, test_loader, y_test = build_dataloaders(all_ticker_data, feature_cols)
    if train_loader is None:
        return None

    model = TransformerModel(
        input_dim=n_features,
        timesteps=N_STEPS,
        d_model=64,        # Smaller model for speed
        num_heads=4,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=SELECTION_LR)

    # Train
    model.train()
    for epoch in range(SELECTION_EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            probs = torch.sigmoid(model(X_batch)).cpu().numpy()
            all_probs.append(probs)

    y_probs = np.vstack(all_probs)
    threshold = np.median(y_probs)
    y_pred = (y_probs > threshold).astype(int)

    f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    return {'f1': f1, 'accuracy': acc, 'precision': prec, 'recall': rec}


def forward_selection(all_ticker_data, device):
    """
    Forward feature selection:
    1. Start with BASE_FEATURES (OHLCV)
    2. At each step, try adding each remaining candidate
    3. Keep the one that improves F1 the most
    4. Stop when no candidate improves the score
    """
    selected = list(BASE_FEATURES)
    remaining = list(CANDIDATE_FEATURES)
    history = []

    # Baseline score with just OHLCV
    print("\n" + "=" * 60)
    print("  BASELINE: Solo OHLCV")
    print("=" * 60)
    baseline = quick_train_evaluate(selected, all_ticker_data, device)
    if baseline is None:
        print("Error: No se pudo evaluar el baseline.")
        sys.exit(1)

    best_f1 = baseline['f1']
    history.append({
        'step': 0,
        'added_feature': '(baseline OHLCV)',
        'features': ', '.join(selected),
        'n_features': len(selected),
        **baseline,
    })
    print(f"  F1={best_f1:.4f}  Acc={baseline['accuracy']:.4f}  "
          f"Prec={baseline['precision']:.4f}  Rec={baseline['recall']:.4f}")

    step = 0
    while remaining:
        step += 1
        print(f"\n{'─' * 60}")
        print(f"  PASO {step}: Evaluando {len(remaining)} candidatos...")
        print(f"  Features actuales ({len(selected)}): {selected}")
        print(f"{'─' * 60}")

        best_candidate = None
        best_candidate_f1 = best_f1
        best_candidate_metrics = None

        for i, candidate in enumerate(remaining):
            trial_features = selected + [candidate]
            sys.stdout.write(f"\r    [{i+1}/{len(remaining)}] Probando '{candidate}'...")
            sys.stdout.flush()

            metrics = quick_train_evaluate(trial_features, all_ticker_data, device)
            if metrics is None:
                print(f" skip (datos insuficientes)")
                continue

            f1 = metrics['f1']
            sys.stdout.write(f" F1={f1:.4f}")

            if f1 > best_candidate_f1:
                best_candidate = candidate
                best_candidate_f1 = f1
                best_candidate_metrics = metrics
                sys.stdout.write(" *MEJOR*")

            sys.stdout.write("\n")
            sys.stdout.flush()

        if best_candidate is None:
            print(f"\n  Ningún candidato mejoró F1. Deteniendo seleccion.")
            break

        # Add the best candidate
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_f1 = best_candidate_f1

        history.append({
            'step': step,
            'added_feature': best_candidate,
            'features': ', '.join(selected),
            'n_features': len(selected),
            **best_candidate_metrics,
        })

        print(f"\n  >>> Agregado: '{best_candidate}'")
        print(f"  >>> F1={best_f1:.4f}  Acc={best_candidate_metrics['accuracy']:.4f}")

    return selected, history




def ablation_study(best_features, all_ticker_data, device):
    """
    After finding the best set, remove each feature one at a time
    to measure individual importance.
    """
    print("\n" + "=" * 60)
    print("  ESTUDIO DE ABLACION: Importancia individual")
    print("=" * 60)

    # Full set score
    full_metrics = quick_train_evaluate(best_features, all_ticker_data, device)
    if full_metrics is None:
        return []

    full_f1 = full_metrics['f1']
    print(f"  Set completo ({len(best_features)} features): F1={full_f1:.4f}\n")

    ablation_results = []
    for feat in best_features:
        if feat in BASE_FEATURES:
            continue  # Don't remove OHLCV

        reduced = [f for f in best_features if f != feat]
        metrics = quick_train_evaluate(reduced, all_ticker_data, device)
        if metrics is None:
            continue

        drop = full_f1 - metrics['f1']
        ablation_results.append({
            'removed_feature': feat,
            'f1_without': metrics['f1'],
            'f1_drop': drop,
        })
        sign = "+" if drop < 0 else "-"
        print(f"  Sin '{feat}': F1={metrics['f1']:.4f} ({sign}{abs(drop):.4f})")

    # Sort by importance (biggest F1 drop = most important)
    ablation_results.sort(key=lambda x: x['f1_drop'], reverse=True)
    return ablation_results

def main():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    print(f"Tickers de seleccion: {SELECTION_TICKERS}")
    print(f"Epochs por iteracion: {SELECTION_EPOCHS}")

    # ── 1. Download all data ───────────────────
    print("\n[1/4] Descargando datos...")
    multi_df, ext_df, macro_df, sector_map = download_all_data()

    # ── 2. Build full feature matrix ───────────
    print("\n[2/4] Construyendo pool completo de features...")
    all_ticker_data = build_full_feature_matrix(multi_df, ext_df, macro_df, sector_map)

    if not all_ticker_data:
        print("Error: No se obtuvieron datos validos.")
        sys.exit(1)

    print(f"\n  Tickers procesados: {list(all_ticker_data.keys())}")
    sample_df = next(iter(all_ticker_data.values()))
    all_available = [c for c in sample_df.columns if c != 'Target']
    print(f"  Features disponibles ({len(all_available)}): {all_available}")

    # ── 3. Forward selection ───────────────────
    print("\n[3/4] Ejecutando Forward Feature Selection...")
    best_features, history = forward_selection(all_ticker_data, device)

    # ── 4. Ablation study ──────────────────────
    print("\n[4/4] Ejecutando estudio de ablacion...")
    ablation = ablation_study(best_features, all_ticker_data, device)

    # ── Results ────────────────────────────────
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("  RESULTADOS FINALES")
    print("=" * 70)

    # Selection history table
    print("\n  FORWARD SELECTION - HISTORIAL:")
    print(f"  {'Paso':<5} {'Feature Agregada':<20} {'N':<4} {'F1':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8}")
    print("  " + "-" * 65)
    for row in history:
        print(f"  {row['step']:<5} {row['added_feature']:<20} {row['n_features']:<4} "
              f"{row['f1']:<8.4f} {row['accuracy']:<8.4f} "
              f"{row['precision']:<8.4f} {row['recall']:<8.4f}")

    # Best combination
    print(f"\n  MEJOR COMBINACION ({len(best_features)} features):")
    for i, f in enumerate(best_features, 1):
        tag = " (base)" if f in BASE_FEATURES else ""
        print(f"    {i:2d}. {f}{tag}")

    # Ablation ranking
    if ablation:
        print("\n  RANKING DE IMPORTANCIA (ablacion):")
        print(f"  {'Feature':<20} {'F1 sin ella':<12} {'Impacto':<10}")
        print("  " + "-" * 42)
        for row in ablation:
            sign = "+" if row['f1_drop'] < 0 else "-"
            print(f"  {row['removed_feature']:<20} {row['f1_without']:<12.4f} "
                  f"{sign}{abs(row['f1_drop']):<10.4f}")

    # Save to CSV
    results_df = pd.DataFrame(history)
    results_df.to_csv("feature_selection_results.csv", index=False)
    print(f"\n  Resultados guardados en 'feature_selection_results.csv'")

    if ablation:
        ablation_df = pd.DataFrame(ablation)
        ablation_df.to_csv("ablation_results.csv", index=False)
        print(f"  Ablacion guardada en 'ablation_results.csv'")

    print(f"\n  Tiempo total: {elapsed/60:.1f} minutos")
    print("=" * 70)


if __name__ == "__main__":
    main()
