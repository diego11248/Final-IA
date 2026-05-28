import pickle
import torch
from data_processing import download_and_compute_features, create_dataloaders
from lstm_model import LSTMModel
from transformer_model import TransformerModel
from evaluation import train_and_evaluate

def main():
    # Configuración de hiperparámetros
    TICKERS = [
        "AAPL", "MSFT", "NVDA", "AMD",  
        "JPM", "BAC", "WFC",            
        "AMZN", "TSLA", "HD",           
        "JNJ", "PFE", "LLY",            
        "XOM", "CVX",
        "INTC", "CSCO", "ORCL", "ADBE",
        "GOOGL", "FB", "AMZN", "NFLX",
        "VZ", "T", "TMUS", "CMCSA",
        "KO", "PEP", "MCD", "NKE",
        "PG", "WMT", "COST", "SBUX",
        "CAT", "BA", "GE", "MMM",
        "AXP", "MA", "V", "PYPL",
        "ZM", "PINS", "SNAP"
    ]
    
    START_DATE = "2020-01-01"
    END_DATE = "2025-12-31"
    TIMESTEPS = 30
    BATCH_SIZE = 128
    EPOCHS = 15  # Ajustado para que el Walk-Forward sea eficiente
    
    # 1. Descarga y cálculo de características inicial (Solo se hace una vez)
    print("\n" + "="*50)
    print("   [FASE 1] DESCARGA Y PROCESAMIENTO INICIAL")
    print("="*50)
    all_ticker_data = download_and_compute_features(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # 2. Validación Walk-Forward (Evaluación de robustez temporal)
    print("\n" + "="*50)
    print("   [FASE 2] VALIDACIÓN WALK-FORWARD")
    print("="*50)
    
    folds = [
        {"train_start": "2020-01-01", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
        {"train_start": "2020-01-01", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
        {"train_start": "2020-01-01", "train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31"}
    ]
    
    for i, fold in enumerate(folds, 1):
        print(f"\n{'─'*50}")
        print(f"  FOLD {i}")
        print(f"  Train: {fold['train_start']} -> {fold['train_end']}")
        print(f"  Test:  {fold['test_start']} -> {fold['test_end']}")
        print(f"{'─'*50}")
        
        train_loader, test_loader, timesteps, features, y_test, _ = create_dataloaders(
            all_ticker_data,
            train_start=fold["train_start"],
            train_end=fold["train_end"],
            test_start=fold["test_start"],
            test_end=fold["test_end"],
            n_steps=TIMESTEPS,
            batch_size=BATCH_SIZE
        )
        
        if train_loader is None or test_loader is None:
            print("Datos insuficientes para este fold. Saltando...")
            continue
            
        # Evaluar LSTM en este fold
        lstm_net = LSTMModel(input_dim=features)
        train_and_evaluate(lstm_net, f"LSTM (Fold {i})", train_loader, test_loader, y_test, epochs=EPOCHS, lr=0.01)
        
        # Evaluar Transformer en este fold
        transformer_net = TransformerModel(input_dim=features, timesteps=timesteps)
        train_and_evaluate(transformer_net, f"Transformer (Fold {i})", train_loader, test_loader, y_test, epochs=EPOCHS, lr=0.01)

    # 3. Entrenamiento Final de Producción
    print("\n" + "="*50)
    print("   [FASE 3] ENTRENAMIENTO FINAL (PRODUCCIÓN)")
    print("="*50)
    
    # Para la fase de producción, usamos el máximo posible de datos de entrenamiento (hasta Sept 2025)
    # y los últimos 3 meses como validación final (para determinar el umbral de confianza).
    prod_train_start = "2020-01-01"
    prod_train_end = "2025-09-30"
    prod_test_start = "2025-10-01"
    prod_test_end = "2025-12-31"
    
    print(f"Train Producción: {prod_train_start} -> {prod_train_end}")
    print(f"Test  Producción: {prod_test_start} -> {prod_test_end}\n")
    
    train_loader, test_loader, timesteps, features, y_test, scaler_dict = create_dataloaders(
        all_ticker_data,
        train_start=prod_train_start,
        train_end=prod_train_end,
        test_start=prod_test_start,
        test_end=prod_test_end,
        n_steps=TIMESTEPS,
        batch_size=BATCH_SIZE
    )
    
    # GUARDAR LOS SCALERS: Esto evita el Data Leakage en predict.py
    with open("scalers.pkl", "wb") as f:
        pickle.dump(scaler_dict, f)
    print(">>> Scalers (MinMaxScaler) guardados exitosamente en 'scalers.pkl' <<<")
    
    if train_loader and test_loader:
        print("\nEntrenando modelo LSTM final...")
        final_lstm = LSTMModel(input_dim=features)
        train_and_evaluate(final_lstm, "LSTM (FINAL)", train_loader, test_loader, y_test, epochs=EPOCHS+5, lr=0.01)
        torch.save(final_lstm.state_dict(), "lstm_model.pt")
        print(">>> Modelo LSTM guardado en 'lstm_model.pt' <<<")
        
        print("\nEntrenando modelo Transformer final...")
        final_transformer = TransformerModel(input_dim=features, timesteps=timesteps)
        train_and_evaluate(final_transformer, "Transformer (FINAL)", train_loader, test_loader, y_test, epochs=EPOCHS+5, lr=0.01)
        torch.save(final_transformer.state_dict(), "transformer_model.pt")
        print(">>> Modelo Transformer guardado en 'transformer_model.pt' <<<")
    else:
        print("Error: No se pudieron generar los dataloaders para la fase final.")

if __name__ == "__main__":
    main()
